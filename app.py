import os
import re
import json
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from openai import OpenAI
import glob

# =========================
# 기본 설정/폰트
# =========================
import matplotlib.font_manager as fm
import os

FONT_PATH = os.path.join("fonts", "NotoSansKR-Regular.ttf")

if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = font_prop.get_name()

plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="리뷰 인사이트 대시보드", layout="wide")

# 폴더 안 모든 엑셀 자동 로딩
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXCEL_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))

COL_DATE = "구매 일자"     # YYYYMMDD
COL_SKU = "SKU명"
COL_RATING = "상품별점"
COL_REVIEW = "리뷰내용"

# =========================
# 파이/도넛 색상 팔레트(1번 이미지 느낌으로 통일)
# - 파이 범례/조각 색을 모두 이 팔레트 기반으로 지정
# - 조각 수가 많아지면 base 팔레트를 "확장"해서 중복을 최소화
# =========================
BASE_PIE_COLORS = [
    "#4E79A7",  # blue
    "#59A14F",  # green
    "#9C6ADE",  # purple
    "#E15799",  # pink
    "#F1CE63",  # yellow
    "#76B7B2",  # light teal
]

def extend_palette(base_colors, n):
    """
    base_colors를 기준으로 n개가 필요하면 톤을 조금씩 변형해 확장.
    (완전 다른 색이 아니라, '1번 이미지 느낌' 유지하면서 중복 최소화)
    """
    if n <= len(base_colors):
        return base_colors[:n]

    out = list(base_colors)
    # HSV에서 V(밝기)를 살짝씩 올리고/내려 변형
    hsv_base = [mcolors.rgb_to_hsv(mcolors.to_rgb(c)) for c in base_colors]

    k = 1
    while len(out) < n:
        for hsv in hsv_base:
            if len(out) >= n:
                break
            h, s, v = hsv
            # 밝기/채도 미세 조정 (k가 커질수록 변형 폭 증가)
            v2 = np.clip(v * (0.92 + 0.06 * (k % 3)), 0, 1)
            s2 = np.clip(s * (0.95 - 0.05 * (k % 2)), 0, 1)
            rgb2 = mcolors.hsv_to_rgb([h, s2, v2])
            out.append(mcolors.to_hex(rgb2))
        k += 1

    return out[:n]


# =========================
# API Key
# =========================
def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None

    if not key:
        return None

    # 공백/줄바꿈/BOM 제거 + 양끝 따옴표 제거
    key = str(key).strip().strip('"').strip("'")
    key = key.replace("\ufeff", "")  # BOM 제거
    return key


# =========================
# 날짜 파싱
# =========================
def parse_yyyymmdd(x):
    if pd.isna(x):
        return pd.NaT
    s = re.sub(r"[^0-9]", "", str(x).strip())
    if len(s) != 8:
        return pd.NaT
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


# =========================
# 텍스트 토큰화 / 키워드
# - 유니그램(1단어)만
# - 제품명(키워드) 제거
# - "탐정칸쵸" 같은 합성 토큰은 "칸쵸"만 제거하고 "탐정" 살리기
# =========================
STOPWORDS = set("""
그리고 그래서 그러나 하지만 또한 너무 정말 진짜 그냥 매우 조금 좀
제가 저는 우리는 너희
있어요 합니다 했어요 했습니다 됩니다 되는 같다 같아요 같음
구매 구매한 배송 제품 상품 사용 후기 리뷰 평점 별점
맛있 맛있어요 맛있음
""".split())

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    t = text.lower()
    tokens = re.findall(r"[가-힣]{2,}|[a-z]{2,}", t)
    return [w for w in tokens if w not in STOPWORDS]

def keyword_core_token(keyword: str):
    kw = (keyword or "").strip().lower()
    # 핵심 토큰(가장 긴 토큰)
    toks = re.findall(r"[가-힣a-z0-9]+", kw)
    return max(toks, key=len) if toks else ""

def build_keyword_exact_pattern(keyword: str):
    """
    '칸쵸', '칸쵸가', '칸쵸는', '칸쵸인줄' 같은 형태를 1:1로 제거
    """
    kw_core = keyword_core_token(keyword)
    if not kw_core:
        return None

    suffixes = r"(가|이|는|은|을|를|에|의|와|과|도|만|로|으로|에서|에게|한테|랑|처럼|마저|까지|부터|조차|이나|나|이며|이고|인줄|같|같아|같네요|임|입니다|네요|요|ㅠ|ㅜ|ㅎㅎ|ㅋㅋ)?"
    return re.compile(rf"^{re.escape(kw_core)}{suffixes}$")

def strip_keyword_from_token(token: str, kw_core: str):
    """
    합성 토큰에서 kw_core만 제거하고 남는 의미 토큰이 2글자 이상이면 살린다.
    예) 탐정칸쵸 -> 탐정
        칸쵸인줄 -> (남는 토큰이 2글자 미만이면 버림)
    """
    if not kw_core:
        return None
    if kw_core not in token:
        return None
    rest = token.replace(kw_core, "")
    rest = rest.strip()
    if len(rest) >= 2 and rest not in STOPWORDS:
        return rest
    return None

def make_exclude_set_from_keyword(keyword: str):
    """
    제품명(키워드)에서 토큰을 뽑아 제외 세트 생성
    예: "탐정칸쵸" -> {"탐정칸쵸"} (tokenize 기준)
    ※ 실제 제거는 kw_core 기반으로 더 강하게 처리함(아래 top_words)
    """
    if not keyword:
        return set()
    return set(tokenize(str(keyword)))

def top_words(series: pd.Series, topn=10, keyword=None):
    """
    - 유니그램(1단어) 빈도
    - 제품명(키워드) 제거(정확형/합성형/조사형)
    - 합성형은 키워드만 제거 후 남는 토큰 살리기(탐정칸쵸 -> 탐정)
    """
    kw_core = keyword_core_token(keyword)
    exact_pat = build_keyword_exact_pattern(keyword)

    c = Counter()
    for txt in series.fillna("").astype(str).tolist():
        for w in tokenize(txt):
            wl = str(w).lower()

            # 1) 정확형(칸쵸, 칸쵸가, 칸쵸는, 칸쵸인줄 등) 제거
            if exact_pat and exact_pat.match(wl):
                continue

            # 2) 합성형(탐정칸쵸, 칸쵸맛 등) 처리
            if kw_core and (kw_core in wl):
                # 남는 의미 토큰이 있으면 그것으로 카운트(2글자 이상)
                rest = strip_keyword_from_token(wl, kw_core)
                if rest:
                    c[rest] += 1
                # 원래 토큰은 제거
                continue

            c[wl] += 1

    return c.most_common(topn)


# =========================
# 도넛 차트(섹션1 스타일)
# - 작은 조각: 오른쪽 분리 표기
# - 범례 색/조각 색: palette로 통일
# =========================
def donut_right_split_labels(
    values,
    labels,
    palette,
    figsize=(4.4, 3.8),
    small_pct=5.0,
    legend_ncol=3,
    legend_y=-0.18,
    text_color="#EAEAEA",
    center_text=None,
    center_text_size=18,
    right_x=1.35,
    right_y_start=0.55,
    right_y_step=0.20
):
    total = float(sum(values)) if sum(values) else 1.0
    pct_list = [(v / total) * 100 for v in values]
    colors = extend_palette(palette, len(values))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    def _autopct(pct):
        return f"{pct:.1f}%" if pct >= small_pct else ""

    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        startangle=90,
        colors=colors,
        autopct=_autopct,
        pctdistance=0.75,
        textprops={"fontsize": 11, "color": text_color},
        wedgeprops=dict(width=0.45)
    )
    ax.axis("equal")

    if center_text:
        ax.text(
            0, 0, center_text,
            ha="center", va="center",
            fontsize=center_text_size,
            color=text_color,
            fontweight="bold"
        )

    # 작은 조각 오른쪽 분리 표기
    small_indices = [i for i, pct in enumerate(pct_list) if (pct > 0) and (pct < small_pct)]
    y_positions = [right_y_start - k * right_y_step for k in range(len(small_indices))]

    for rank, i in enumerate(small_indices):
        w = wedges[i]
        pct = pct_list[i]
        angle = (w.theta2 + w.theta1) / 2
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        xy = (x * 0.95, y * 0.95)
        xytext = (right_x, y_positions[rank])

        ax.annotate(
            f"{labels[i]}, {pct:.1f}%",
            xy=xy,
            xytext=xytext,
            ha="left",
            va="center",
            fontsize=12,
            color=text_color,
            arrowprops=dict(arrowstyle="-", color=text_color, lw=1.2),
        )

    leg = ax.legend(
        wedges,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_ncol,
        frameon=False
    )
    for t in leg.get_texts():
        t.set_color(text_color)

    plt.tight_layout()
    return fig


# =========================
# 연령 추정(룰 기반) - '취식 대상' 추정
# =========================
AGE_RULES = {
    "유아/어린이": ["유아", "아기", "아이", "애기", "어린이", "키즈", "유치원", "유치", "어린"],
    "초등": ["초등", "초1", "초2", "초3", "초4", "초5", "초6", "초딩", "학교", "학원"],
    "청소년": ["중딩", "중학생", "고딩", "고등", "청소년", "시험", "수능"],
    "20대": ["대학생", "캠퍼스", "자취", "20대", "군대", "훈련소"],
    "30~40대": ["직장", "회사", "출근", "육아", "부모", "엄마", "아빠", "30대", "40대", "가정"],
    "50대+": ["50대", "60대", "어르신", "부모님", "할머니", "할아버지"],
}

def estimate_age_segments(texts):
    """
    리뷰 1건당 세그먼트 1개만 배정(우선순위 기반)
    - 단서가 여러 개여도 가장 우선순위가 높은 1개만 카운트
    - 단서가 없으면 '미상'
    => 표/차트가 100%로 정확히 일치함
    """
    if not texts:
        return pd.DataFrame(columns=["세그먼트", "건수", "비율(%)", "단서(요약)"])

    # 우선순위(원하는 순서로 조정 가능)
    PRIORITY = ["유아/어린이", "초등", "청소년", "20대", "30~40대", "50대+"]

    total = len(texts)
    counts = Counter()
    hints = {k: set() for k in AGE_RULES.keys()}
    hints["미상"] = set()

    for t in texts:
        s = str(t)

        matched = None
        for seg in PRIORITY:
            kws = AGE_RULES.get(seg, [])
            hit = [k for k in kws if k in s]
            if hit:
                matched = seg
                for h in hit[:3]:
                    hints[seg].add(h)
                break

        if matched is None:
            matched = "미상"

        counts[matched] += 1

    rows = []
    for seg, cnt in counts.items():
        rows.append([
            seg,
            cnt,
            round(cnt / total * 100, 1),
            ", ".join(list(hints.get(seg, set()))[:3]) if seg != "미상" else "-"
        ])

    df = pd.DataFrame(rows, columns=["세그먼트", "건수", "비율(%)", "단서(요약)"]).sort_values("건수", ascending=False)
    return df


# =========================
# 데이터 로드
# =========================
def _file_sig(paths):
    return [(p, os.path.getmtime(p), os.path.getsize(p)) for p in paths]

@st.cache_data(show_spinner=False)
def load_data(file_sig):
    paths = [p for (p, _, _) in file_sig]

    if not paths:
        st.error(f"엑셀 데이터 파일이 없습니다. DATA_DIR={DATA_DIR}")
        st.stop()

    df_list = [pd.read_excel(p) for p in paths]
    df = pd.concat(df_list, ignore_index=True)

    df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")
    df["_dt"] = df[COL_DATE].apply(parse_yyyymmdd) if COL_DATE in df.columns else pd.NaT
    df[COL_SKU] = df[COL_SKU].astype(str)
    df[COL_REVIEW] = df[COL_REVIEW].astype(str)

    return df

df = load_data(_file_sig(EXCEL_FILES))


# =========================
# Sidebar UI
# =========================
st.sidebar.title("필터")
keyword = st.sidebar.text_input("제품명(키워드) 입력", value="칸쵸")
period_label = st.sidebar.selectbox("기간", ["최근 30일", "최근 90일", "최근 180일", "전체"], index=1)
run_btn = st.sidebar.button("분석 실행")
st.sidebar.caption("SKU명에 키워드가 포함된 제품만 자동 필터링됩니다.")

def apply_filters(df_in: pd.DataFrame):
    f = df_in.copy()
    f = f[f[COL_SKU].str.contains(keyword, case=False, na=False)]

    if period_label != "전체":
        days = int(period_label.split()[1].replace("일", ""))
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        f = f[f["_dt"] >= cutoff]

    return f


# =========================
# Session state init
# =========================
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "ai_payload" not in st.session_state:
    st.session_state.ai_payload = None
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "ai_raw" not in st.session_state:
    st.session_state.ai_raw = None


# =========================
# 상단 스타일(CSS)
# =========================
st.markdown(
    """
    <style>
      .pill-row {display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 6px 0;}
      .pill {
        padding:10px 14px; border-radius:999px;
        border:1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.05);
        font-weight:700; font-size:16px;
      }
      .card {
        border:1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.04);
        padding:16px 16px;
        border-radius:18px;
      }
      .card-title {font-weight:900; font-size:18px; margin-bottom:10px;}
      .muted {opacity:0.8; font-size:13px;}
      .hr {border-top:1px solid rgba(255,255,255,0.12); margin: 14px 0;}
      .kpi {font-size:28px; font-weight:900;}
      .kpi-sub {opacity:0.85;}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Title
# =========================
st.title("리뷰 인사이트 대시보드")
st.caption("키워드+기간으로 필터링한 리뷰를 기반으로, 차트/표 + AI 요약 + 마케팅 인사이트를 한 화면에 표시합니다.")

# 최초 진입 안내
if (not run_btn) and (st.session_state.filtered_df is None):
    st.info("왼쪽에서 제품 키워드와 기간을 선택하고 **분석 실행**을 눌러주세요.")
    st.stop()

# run_btn을 누르지 않았더라도, 이미 분석된 상태면 세션에서 사용
if run_btn or (st.session_state.filtered_df is None):
    f = apply_filters(df)
    st.session_state.filtered_df = f
else:
    f = st.session_state.filtered_df

if len(f) == 0:
    st.warning("조건에 맞는 데이터가 없습니다. 키워드/기간을 바꿔주세요.")
    st.stop()


# =========================
# 집계
# =========================
total = len(f)
avg = float(f[COL_RATING].mean())

# 감성 비율 (여기서는 표기용)
count_5 = int((f[COL_RATING] == 5).sum())
count_1_3 = int((f[COL_RATING] <= 3).sum())
pct_5 = (count_5 / total * 100) if total else 0.0
pct_1_3 = (count_1_3 / total * 100) if total else 0.0

# 1) 긍정/중립/부정 도넛 (4~5 / 3 / 1~2)
def sentiment_bucket(r):
    if pd.isna(r):
        return "미상"
    if r >= 4:
        return "긍정(4~5)"
    if r == 3:
        return "중립(3)"
    return "부정(1~2)"

sent = f[COL_RATING].apply(sentiment_bucket).value_counts()
sent_main = sent.drop(labels=["미상"], errors="ignore")
order_sent = ["긍정(4~5)", "중립(3)", "부정(1~2)"]
sent_main2 = sent_main.reindex(order_sent).fillna(0)
sent_labels = sent_main2.index.tolist()
sent_values = sent_main2.values.tolist()
sent_ratio = (sent_main2 / sent_main2.sum() * 100).round(1) if sent_main2.sum() else sent_main2

# 2) SKU 분포(상위 + 기타) -> 색 중복 최소화를 위해 5개 + 기타로 제한(<=6색)
sku_counts = f[COL_SKU].value_counts()
topN = 5
top_sku = sku_counts.head(topN)
others = int(sku_counts.iloc[topN:].sum())
sku_pie = top_sku.copy()
if others > 0:
    sku_pie.loc["기타"] = others
sku_labels = sku_pie.index.tolist()
sku_values = sku_pie.values.tolist()

# 3) 긍정 리뷰(5점) 키워드 TOP10 (유니그램)
pos_df = f[f[COL_RATING] == 5]
pos_top10 = top_words(pos_df[COL_REVIEW], topn=10, keyword=keyword)

# 4) 부정 리뷰(1~3점) 키워드 TOP10 (유니그램)
low_df = f[f[COL_RATING] <= 3]
low_top10 = top_words(low_df[COL_REVIEW], topn=10, keyword=keyword)

# 5) 실제 취식 연령대(룰 기반)
age_df = estimate_age_segments(f[COL_REVIEW].fillna("").astype(str).tolist())
# 도넛은 상위5 + 기타(<=6색)
topK = 5
age_top = age_df.head(topK)
age_other = int(age_df.iloc[topK:]["건수"].sum())
age_labels = age_top["세그먼트"].tolist()
age_values = age_top["건수"].tolist()
if age_other > 0:
    age_labels.append("기타")
    age_values.append(age_other)

# 6) 취식 TPO(키워드 빈도) - 룰 기반 (LLM 요약은 버튼 누르면)
TPO_RULES = {
    "학교/학원": ["학교", "학원", "등교", "하교", "수업", "급식", "도시락"],
    "집/가정": ["집", "가정", "집에서", "가족", "엄마", "아빠", "아이", "육아"],
    "이동/외출": ["차", "드라이브", "여행", "외출", "캠핑", "피크닉", "산책", "버스", "지하철"],
    "직장/사무": ["회사", "직장", "출근", "사무실", "회의", "야근"],
    "간식/야식": ["간식", "야식", "출출", "배고", "출출할", "심심", "티타임"],
    "선물/행사": ["선물", "답례", "행사", "파티", "모임", "어린이날", "생일"],
}

def tpo_counts(texts):
    cnt = Counter()
    for t in texts:
        s = str(t)
        for cat, kws in TPO_RULES.items():
            if any(k in s for k in kws):
                cnt[cat] += 1
    total = len(texts) if texts else 1
    rows = []
    for cat, v in cnt.most_common():
        rows.append([cat, v, round(v / total * 100, 1)])
    return pd.DataFrame(rows, columns=["상황(TPO)", "언급수", "비율(%)"])

tpo_df = tpo_counts(f[COL_REVIEW].fillna("").astype(str).tolist())


# =========================
# LLM 호출(상단 버튼 1회로 전체 생성)
# - 섹션1~4 요약(📌요약 코멘트)
# - 섹션6 TPO 요약
# - 마케팅 인사이트 리포트(이모지 구조)
# =========================
def call_llm_all(payload: dict):
    api_key = get_api_key()
    if not api_key:
        return None, "OPENAI_API_KEY가 설정되지 않았습니다(secrets.toml 또는 환경변수)."

    client = OpenAI(api_key=api_key)

    prompt = f"""
너는 FMCG/스낵 브랜드 마케터의 시니어 전략가다.
아래 '근거 데이터'만으로 대시보드에 바로 붙일 수 있는 한국어 문장을 만든다.
과장 금지. 근거에서 벗어나지 말 것.
불필요한 설명(예: ~할 수 있습니다, ~유도할 수 있습니다)은 최소화하고, 꼭 필요한 전략 포인트만 짧게 쓴다.

[근거 데이터]
- 제품 키워드: {payload["keyword"]}
- 기간: {payload["period"]}
- 총 리뷰 수: {payload["total"]}
- 평균 별점: {payload["avg"]:.2f}
- 5점 비율: {payload["pct_5"]:.1f}% ({payload["count_5"]}건)
- 1~3점 비율: {payload["pct_1_3"]:.1f}% ({payload["count_1_3"]}건)

- SKU TOP(상위): {payload["sku_top_str"]}
- 5점 키워드 TOP10: {payload["pos_words_str"]}
- 1~3점 키워드 TOP10: {payload["low_words_str"]}

- 취식 연령대(룰) 상위 분포: {payload["age_dist_str"]}
- 취식 연령대 단서(요약): {payload["age_clue_str"]}
- 취식 연령대(룰) 미상 비율: {payload["age_unknown_pct"]:.1f}%

- TPO(룰) 상위: {payload["tpo_str"]}

[출력 형식: JSON만 출력]
{{
  "sec1_summary": "섹션1 전체 현황 요약(3~5줄, 핵심만).",
  "sec2_summary": "섹션2 SKU 요약(2~4줄).",
  "sec3_summary": "섹션3(5점) 요약 코멘트(3~5줄).",
  "sec4_summary": "섹션4(1~3점) 요약 코멘트(3~5줄).",
  "sec5_age_summary": "섹션5(취식 연령대) 요약 코멘트(3~5줄). 미상 비중이 높을 경우 해석 주의 1줄 포함.",
  "sec6_tpo_summary": "섹션6 TPO 요약(3~5줄).",

  "marketing_report": {{
    "consumer_insight": {{
      "bullets": ["...", "...", "..."],
      "meaning": "👉 의미: ..."
    }},
    "ad_banner": {{
      "copies": ["...", "...", "...", "...", "..."],
      "strategy_point": "👉 전략 포인트: ..."
    }},
    "planning": {{
      "bullets": ["...", "...", "..."],
      "meaning": "👉 ... "
    }},
    "improvement": {{
      "bullets": ["...", "..."],
      "meaning": "👉 ... "
    }},
    "strategic": {{
      "bullets": ["...", "...", "..."]
    }}
  }}
}}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=900,
    )
    content = resp.choices[0].message.content
    return content, None

def safe_json_load(s: str):
    """
    LLM 응답이 아래 형태여도 안전하게 JSON만 추출해서 파싱
    - ```json { ... } ```
    - 앞/뒤에 안내 문구가 섞인 경우
    """
    if not s:
        return None

    text = str(s).strip()

    # 1) 코드펜스 제거: ```json ... ``` / ``` ... ```
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # 2) 혹시 앞뒤 잡텍스트가 남아있으면, 첫 { 부터 마지막 } 까지 잘라내기
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1].strip()

    # 3) JSON 파싱
    try:
        return json.loads(candidate)
    except Exception:
        return None


# =========================
# 상단 LLM 버튼(가장 위)
# =========================
def fmt_kv_list(lst, limit=5):
    return ", ".join([f"{k}({v})" for k, v in lst[:limit]])

def fmt_age_dist(df_age, limit=5):
    top = df_age.head(limit)
    return ", ".join([f"{r['세그먼트']} {r['비율(%)']}%" for _, r in top.iterrows()])

def fmt_age_clue(df_age, limit=5):
    top = df_age.head(limit)
    return ", ".join([f"{r['세그먼트']}({r['단서(요약)']})" for _, r in top.iterrows()])

def fmt_tpo(df_tpo, limit=5):
    if df_tpo is None or len(df_tpo) == 0:
        return "-"
    top = df_tpo.head(limit)
    return ", ".join([f"{r['상황(TPO)']} {r['비율(%)']}%" for _, r in top.iterrows()])

sku_top_str = ", ".join([f"{k}({int(v)}건)" for k, v in sku_counts.head(5).items()])

unknown_row = age_df[age_df["세그먼트"] == "미상"]
age_unknown_pct = float(unknown_row["비율(%)"].iloc[0]) if len(unknown_row) else 0.0

ai_payload = {
    "keyword": keyword,
    "period": period_label,
    "total": total,
    "avg": avg,
    "count_5": count_5,
    "pct_5": pct_5,
    "count_1_3": count_1_3,
    "pct_1_3": pct_1_3,
    "sku_top_str": sku_top_str,
    "pos_words_str": fmt_kv_list(pos_top10, limit=10),
    "low_words_str": fmt_kv_list(low_top10, limit=10),
    "age_dist_str": fmt_age_dist(age_df, limit=5),
    "age_clue_str": fmt_age_clue(age_df, limit=5),
    "age_unknown_pct": age_unknown_pct,
    "tpo_str": fmt_tpo(tpo_df, limit=6),
}
st.session_state.ai_payload = ai_payload

top_bar = st.container()
with top_bar:
    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown(
            f"""
            <div class="pill-row">
              <div class="pill">🔎 키워드: {keyword}</div>
              <div class="pill">🗓️ 기간: {period_label}</div>
              <div class="pill">🧾 리뷰: {total:,}건</div>
              <div class="pill">⭐ 평균: {avg:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with cB:
        ai_btn = st.button("🤖 AI 요약/인사이트 생성(1회 호출)", use_container_width=True)

        if ai_btn:
            with st.spinner("AI 생성 중..."):
                raw, err = call_llm_all(ai_payload)
            if err:
                st.error(err)
            else:
                data = safe_json_load(raw)
                st.session_state.ai_raw = raw
                st.session_state.ai_result = data

# 디버그(원하면 펼쳐보기)
with st.expander("AI에 전달되는 근거 데이터(확인용)", expanded=False):
    st.json(ai_payload)


# =========================
# 공통: 요약 코멘트(평범하게 보이게)
# =========================
def render_summary_comment(text: str):
    if not text:
        st.write("-")
        return
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📌 요약 코멘트</div>', unsafe_allow_html=True)
    for line in str(text).split("\n"):
        line = line.strip()
        if line:
            st.markdown(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Section 1
# =========================
st.subheader("1️⃣ 전체 현황 요약")

k1, k2 = st.columns(2)
with k1:
    st.markdown(f'<div class="kpi">{total:,}건</div><div class="kpi-sub">총 리뷰 수</div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi">{avg:.2f}점</div><div class="kpi-sub">평균 별점</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])
with c1:
    st.write("긍정/중립/부정 비율")
    pos_pct = float(sent_ratio.get("긍정(4~5)", 0.0))
    center = f"긍정\n{pos_pct:.1f}%"
    fig1 = donut_right_split_labels(
        values=sent_values,
        labels=sent_labels,
        palette=BASE_PIE_COLORS,
        figsize=(4.4, 3.8),
        small_pct=5.0,
        legend_ncol=3,
        legend_y=-0.18,
        text_color="#EAEAEA",
        center_text=center,
        center_text_size=18,
        right_x=1.35,
        right_y_start=0.55,
        right_y_step=0.20
    )
    st.pyplot(fig1)

with c2:
    data = st.session_state.ai_result
    render_summary_comment(data.get("sec1_summary") if data else "")

st.divider()


# =========================
# Section 2
# =========================
st.subheader("2️⃣ 판매 집중 SKU")

l, r = st.columns([1, 1])
with l:
    st.write("SKU별 리뷰 건수 비중(상위 + 기타)")
    sku_total = sum(sku_values) if sku_values else 1
    top_share = (sku_values[0] / sku_total) * 100 if sku_values else 0
    center = f"TOP1\n{top_share:.1f}%"

    fig2 = donut_right_split_labels(
        values=sku_values,
        labels=sku_labels,
        palette=BASE_PIE_COLORS,
        figsize=(4.4, 3.8),
        small_pct=5.0,
        legend_ncol=3,
        legend_y=-0.18,
        text_color="#EAEAEA",
        center_text=center,
        center_text_size=18,
        right_x=1.35,
        right_y_start=0.55,
        right_y_step=0.20
    )
    st.pyplot(fig2)

with r:
    st.write("SKU TOP 리스트")
    sku_table = sku_counts.head(10).reset_index()
    sku_table.columns = ["SKU명", "리뷰수"]
    st.dataframe(sku_table, use_container_width=True, hide_index=True)

    data = st.session_state.ai_result
    render_summary_comment(data.get("sec2_summary") if data else "")

st.divider()


# =========================
# Section 3 (5점)
# =========================
st.subheader("3️⃣ 긍정 리뷰 분석 (5점)")

a, b = st.columns([1, 1])
with a:
    st.write("상위 반복 키워드 TOP10 (유니그램) — 제품명(키워드) 제거")
    st.dataframe(
        pd.DataFrame(pos_top10, columns=["키워드", "빈도"]),
        use_container_width=True,
        hide_index=True
    )

with b:
    data = st.session_state.ai_result
    render_summary_comment(data.get("sec3_summary") if data else "")

st.divider()


# =========================
# Section 4 (1~3점)
# =========================
st.subheader("4️⃣ 부정 리뷰 분석 (1~3점)")

a, b = st.columns([1, 1])
with a:
    st.write("반복 키워드 TOP10 (유니그램) — 제품명(키워드) 제거")
    st.dataframe(
        pd.DataFrame(low_top10, columns=["키워드", "빈도"]),
        use_container_width=True,
        hide_index=True
    )

with b:
    data = st.session_state.ai_result
    render_summary_comment(data.get("sec4_summary") if data else "")

st.divider()


# =========================
# Section 5 (연령)
# =========================
st.subheader("5️⃣ 실제 취식 연령대 분석(룰 기반)")

p, q = st.columns([1, 1])

with p:
    st.write("연령 단서 기반 분포(상위 + 기타)")

    if age_labels and age_values:

        total_v = sum(age_values) if sum(age_values) else 1.0
        pct_list = [(v / total_v) * 100 for v in age_values]

        max_i = int(np.argmax(pct_list))
        center = f"{age_labels[max_i]}\n{pct_list[max_i]:.1f}%"

        fig5 = donut_right_split_labels(
            values=age_values,
            labels=age_labels,
            palette=BASE_PIE_COLORS,
            figsize=(4.4, 3.8),
            small_pct=5.0,
            legend_ncol=3,
            legend_y=-0.18,
            text_color="#EAEAEA",
            center_text=center,
            center_text_size=18,
            right_x=1.35,
            right_y_start=0.55,
            right_y_step=0.20
        )
        st.pyplot(fig5)

    else:
        st.write("표시할 데이터가 없습니다.")

with q:
    st.write("세그먼트 상세(추정 단서 요약)")
    st.dataframe(
        age_df[["세그먼트", "건수", "비율(%)", "단서(요약)"]],
        use_container_width=True,
        hide_index=True
    )
    st.caption(
        "※ ‘미상’은 리뷰에서 연령 단서 키워드가 발견되지 않은 경우입니다. "
        "(구매자 연령이 아니라, 리뷰 텍스트에 언급된 취식 대상 추정)"
    )
    data = st.session_state.ai_result
    render_summary_comment(data.get("sec5_age_summary") if data else "")

st.divider()


# =========================
# Section 6 (TPO)
# =========================
st.subheader("6️⃣ 취식 TPO 분석")

a, b = st.columns([1, 1])
with a:
    st.write("TPO 키워드 언급 빈도(룰 기반)")
    if len(tpo_df) > 0:
        st.dataframe(tpo_df, use_container_width=True, hide_index=True)
    else:
        st.write("TPO 단서가 거의 없습니다(또는 리뷰에 TPO 언급이 적습니다).")

with b:
    data = st.session_state.ai_result
    render_summary_comment(data.get("sec6_tpo_summary") if data else "")

st.divider()


# =========================
# 마케팅 인사이트(LLM 기반) - 이모지 포인트 + 구조화
# =========================
st.subheader("📊 마케팅 인사이트 리포트")

# 상단 칩(2번 이미지 느낌)
st.markdown(
    f"""
    <div class="pill-row">
      <div class="pill">🔎 키워드: {keyword}</div>
      <div class="pill">🗓️ 기간: {period_label}</div>
      <div class="pill">🧾 리뷰: {total:,}건</div>
      <div class="pill">⭐ 평균: {avg:.2f}</div>
    </div>
    """,
    unsafe_allow_html=True
)

data = st.session_state.ai_result
mr = data.get("marketing_report") if data else None

if not mr:
    st.write("AI 버튼을 누르면 마케팅 인사이트가 생성됩니다.")
else:
    # 레이아웃: 2열 카드 구성
    left, right = st.columns([1, 1])

    # 1) 소비자 인사이트
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧠 소비자 인사이트</div>', unsafe_allow_html=True)
        for b in mr["consumer_insight"]["bullets"]:
            st.markdown(f"- {b}")
        st.markdown(mr["consumer_insight"]["meaning"])
        st.markdown("</div>", unsafe_allow_html=True)

    # 2) 광고/배너 문구 추천 (문구 먼저)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 광고 · 배너 메시지 방향</div>', unsafe_allow_html=True)

        # 문구 먼저 보여주기
        for c in mr["ad_banner"]["copies"]:
            st.markdown(f"- **{c}**")

        # 아래쪽에 서브 설명(전략 포인트)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted">{mr["ad_banner"]["strategy_point"]}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 3) 기획 상품 방향 / 4) 제품 개선 포인트
    left2, right2 = st.columns([1, 1])

    with left2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📦 기획 상품 방향</div>', unsafe_allow_html=True)
        for b in mr["planning"]["bullets"]:
            st.markdown(f"- {b}")
        st.markdown(mr["planning"]["meaning"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔧 제품 개선 포인트</div>', unsafe_allow_html=True)
        for b in mr["improvement"]["bullets"]:
            st.markdown(f"- {b}")
        st.markdown(mr["improvement"]["meaning"])
        st.markdown("</div>", unsafe_allow_html=True)

    # 5) 전략적 시사점
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📌 전략적 시사점</div>', unsafe_allow_html=True)
    for b in mr["strategic"]["bullets"]:
        st.markdown(f"- {b}")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# (선택) AI 파싱 실패 시 원문 확인
# =========================
if st.session_state.ai_result is None and st.session_state.ai_raw is not None:
    st.error("AI 응답을 JSON으로 파싱하지 못했습니다. 원문을 표시합니다.")
    st.text(st.session_state.ai_raw)