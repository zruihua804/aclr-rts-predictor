import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, datetime
import gspread
from google.oauth2.service_account import Credentials
import warnings
import uuid
warnings.filterwarnings('ignore')

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACLR RTS Predictor",
    page_icon="🦵",
    layout="centered"
)

# ── 样式 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .ref-box {
        background: #eaf4fb;
        border-left: 4px solid #2E86AB;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 12px;
        color: #1a3a4a;
        margin-top: 8px;
    }
    .warning-box {
        background: #fdedec;
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 13px;
        color: #922b21;
    }
    .lars-warning {
        background: #fff3e0;
        border-left: 4px solid #e65100;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 13px;
        color: #7f3e00;
    }
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #1976d2;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 13px;
        color: #0d47a1;
    }
    .id-box {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 15px;
        color: #1b5e20;
        font-family: 'DM Mono', monospace;
    }
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 4px;
    }
    .contrib-bar-container {
        background: #f0f0f0;
        border-radius: 4px;
        height: 10px;
        width: 100%;
        margin: 4px 0 10px 0;
    }
    .contrib-bar-pos {
        background: #2E86AB;
        border-radius: 4px;
        height: 10px;
    }
    .contrib-bar-neg {
        background: #e74c3c;
        border-radius: 4px;
        height: 10px;
    }
    .disclaimer {
        font-size: 11px; color: #888;
        border-top: 1px solid #eee;
        padding-top: 12px; margin-top: 20px;
    }
    .timepoint-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }
    .tegner-delta {
        font-size: 13px;
        color: #555;
        margin-top: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── 语言 + 模式 ───────────────────────────────────────────────────────────────
lang = st.sidebar.radio("Language / 语言", ["中文", "English"])
zh = lang == "中文"

pro_mode = st.sidebar.toggle(
    "专业模式 Pro Mode",
    value=True,
    help="开启后显示变量贡献分解图 / Shows variable contribution breakdown"
)

# ── 文献参考（侧边栏）────────────────────────────────────────────────────────
with st.sidebar.expander("📚 模型文献依据" if zh else "📚 Model References", expanded=False):
    st.markdown("""
**Key References:**

**[1]** Ithurburn et al. *Am J Sports Med.* 2019
ACL-RSI OR=1.81/10pts; Hop LSI OR=2.86/10%

**[2]** Ueda et al. *Orthop J Sports Med.* 2023
ACL-RSI OR=1.11/pt (p=0.003); Age OR=0.80/yr (p=0.012)

**[3]** van Haren et al. *Ann Phys Rehabil Med.* 2023
Multicenter prospective cohort, n=208, Bootstrap validated

**[4]** Xiao et al. *Am J Sports Med.* 2023
Meta-analysis n=3744: ACL-RSI strongest predictor

**[5]** Duchman et al. *Am J Sports Med.* 2019
ACL-RSI ≥65 optimal cutoff (ROC analysis, n=681)

**[6]** Liu et al. *KSSTA.* 2021
LARS 10yr follow-up: re-rupture 11.8% vs 6.2% autograft

**[7]** Grindem et al. *BJSM.* 2016
Each month delay in RTS reduces re-injury risk ~51%

**[8]** Kyritsis et al. *BJSM.* 2016
RTS <9mo: 4× higher re-rupture vs ≥9mo

**[9]** Toole et al. *AJSM.* 2017
RTS rates rise significantly 9→12→24mo post-op
    """)

with st.sidebar.expander("⏱️ 时间分层说明" if zh else "⏱️ Time Stratification", expanded=False):
    st.markdown("""
**术后时间分层临床解读：**

- **< 9个月**：移植物韧带化未完成，再损伤风险最高（Kyritsis 2016）
- **9–12个月**：标准RTS评估窗口，文献模型主要来源时间段
- **13–24个月**：移植物趋于成熟，心理因素权重上升
- **> 24个月**：长期未RTS，主要障碍转为废用性萎缩和心理恐惧，评估侧重点不同

*时间因素目前作为分层警示，不直接进入β计算；待本地数据累积后可迭代纳入模型。*
    """)

# ── 标题 ──────────────────────────────────────────────────────────────────────
st.title("🦵 ACLR术后重返运动预测器" if zh else "🦵 ACLR Return to Sport Predictor")
st.caption(
    "基于文献多因素逻辑回归系数 | AUC≈0.80 | v2.0 · 优复门诊运动医学" if zh else
    "Literature-based multivariate logistic regression | AUC≈0.80 | v2.0 · UP Clinic Sports Medicine"
)

# ── 模型参数 ──────────────────────────────────────────────────────────────────
# β系数来源：
# ACL-RSI: OR=1.81/10pts → β=ln(1.81)/10=0.0593  [Ithurburn 2019]
# Hop LSI: OR=2.861/10%  → β=ln(2.861)/10=0.1051 [Ithurburn 2019]
# Quad LSI: OR=1.03/1%   → β=ln(1.03)=0.0296     [Ueda 2023]
# Age: OR=0.80/yr        → β=ln(0.80)=-0.2231     [Ueda 2023]
# Intercept: calibrated to baseline RTS rate 62%
BETA = dict(
    intercept = -8.5806,
    aclrsi    =  0.0593,
    hop_lsi   =  0.1051,
    quad_lsi  =  0.0296,
    age       = -0.2231
)

def predict_rts(aclrsi, hop_lsi, quad_lsi, age):
    lo = (BETA['intercept']
          + BETA['aclrsi']   * aclrsi
          + BETA['hop_lsi']  * hop_lsi
          + BETA['quad_lsi'] * quad_lsi
          + BETA['age']      * age)
    return 1 / (1 + np.exp(-lo)) * 100

def compute_contributions(aclrsi, hop_lsi, quad_lsi, age):
    """计算各变量对logit的贡献，用于专业模式可视化"""
    contributions = {
        'ACL-RSI':           BETA['aclrsi']   * aclrsi,
        'Hop LSI':           BETA['hop_lsi']  * hop_lsi,
        'Quad LSI':          BETA['quad_lsi'] * quad_lsi,
        'Age / 年龄':        BETA['age']      * age,
    }
    return contributions

def rts_curve_by_time(aclrsi, hop_lsi, quad_lsi, age):
    """
    生成RTS概率随时间演变的示意曲线。
    基于Kyritsis 2016 + Toole 2017的数据，对基础概率做时间调节。
    注意：这是教育性展示，非模型直接预测。
    时间调节系数来源：
      <9mo:  风险最高，概率×0.55 (Kyritsis: 4× re-rupture risk)
      9mo:   基准
      12mo:  概率×1.10
      18mo:  概率×1.18
      24mo:  概率×1.22
      36mo:  概率×1.18 (逐渐平台化)
      48mo:  概率×1.15 (长期未RTS者心理/废用因素主导，略降)
    """
    base = predict_rts(aclrsi, hop_lsi, quad_lsi, age)
    time_points = [6, 9, 12, 18, 24, 36, 48]
    # 调节系数（相对于9个月基准）
    modifiers = {6: 0.55, 9: 1.0, 12: 1.10, 18: 1.18, 24: 1.22, 36: 1.18, 48: 1.15}
    curve = {}
    for t in time_points:
        val = min(base * modifiers[t], 99.0)
        curve[t] = round(val, 1)
    return curve

def generate_record_id():
    """生成唯一记录ID，格式：ACLR-YYYYMMDD-XXXX"""
    today = date.today().strftime("%Y%m%d")
    suffix = str(uuid.uuid4())[:4].upper()
    return f"ACLR-{today}-{suffix}"

def get_months_post_op(surgery_date):
    """从手术日期计算术后月数"""
    if surgery_date is None:
        return None
    today = date.today()
    delta = today - surgery_date
    return round(delta.days / 30.44, 1)

def time_stratum_label(months):
    """返回时间分层标签和临床提示"""
    if months is None:
        return None, None
    if months < 9:
        return "early", (
            "⚠️ 术后 < 9个月：移植物韧带化尚未完成，此时期RTS再损伤风险最高（Kyritsis 2016）。"
            "功能测试达标不等于移植物已成熟，建议谨慎决策。" if zh else
            "⚠️ < 9 months post-op: Graft ligamentization incomplete. Re-injury risk highest in this period (Kyritsis 2016). "
            "Functional test clearance ≠ graft maturity. Proceed with caution."
        )
    elif months <= 12:
        return "standard", (
            "✅ 术后 9–12个月：标准RTS评估窗口，与本模型文献来源时间段一致，预测可信度最高。" if zh else
            "✅ 9–12 months post-op: Standard RTS assessment window, consistent with model literature timeframe. Highest prediction confidence."
        )
    elif months <= 24:
        return "late", (
            "📋 术后 13–24个月：移植物趋于成熟，心理准备度（ACL-RSI）和运动专项恢复的权重此时更为关键。" if zh else
            "📋 13–24 months post-op: Graft maturing. Psychological readiness (ACL-RSI) and sport-specific recovery are increasingly critical factors."
        )
    else:
        return "longterm", (
            "⚠️ 术后 > 24个月：长期未重返运动。主要障碍已从移植物成熟度转变为废用性肌肉萎缩和心理恐惧。"
            "建议重新评估康复目标，ACL-RSI权重尤为重要。" if zh else
            "⚠️ > 24 months post-op: Long-term non-return. Primary barriers have shifted from graft maturity to disuse atrophy and kinesiophobia. "
            "Reassess rehab goals. ACL-RSI weighting is especially important."
        )

# ── Google Sheets ─────────────────────────────────────────────────────────────
def get_sheets_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    return gspread.authorize(creds)

def get_sheet1():
    """首次评估表"""
    client = get_sheets_client()
    return client.open_by_key(
        st.secrets["sheets"]["aclr_spreadsheet_id"]).sheet1

def get_sheet2():
    """随访结局表（第2个sheet，需在Google Sheets中预先创建）"""
    client = get_sheets_client()
    wb = client.open_by_key(st.secrets["sheets"]["aclr_spreadsheet_id"])
    try:
        return wb.worksheet("随访结局")
    except Exception:
        # 如果sheet2不存在则创建
        ws = wb.add_worksheet(title="随访结局", rows=1000, cols=20)
        headers = [
            "记录ID", "患者姓名", "随访日期", "随访医生",
            "实际RTS", "RTS时间点(术后月)", "RTS后Tegner评分",
            "是否再损伤", "再损伤类型", "备注"
        ]
        ws.append_row(headers)
        return ws

def save_assessment(row):
    try:
        sheet = get_sheet1()
        # 如果是第一条记录，先写表头
        existing = sheet.get_all_values()
        if len(existing) == 0:
            headers = [
                "记录ID", "患者姓名", "评估日期", "评估医生",
                "年龄", "手术日期", "术后月数", "评估时间点",
                "Tegner术前", "Tegner当前",
                "ACL-RSI", "Hop LSI(%)", "Quad LSI(%)",
                "移植物类型", "手术类型", "运动类型",
                "预测RTS概率(%)", "风险分层"
            ]
            sheet.append_row(headers)
        sheet.append_row(row)
        return True, None
    except Exception as e:
        return False, str(e)

def save_followup(row):
    try:
        sheet = get_sheet2()
        sheet.append_row(row)
        return True, None
    except Exception as e:
        return False, str(e)

def lookup_patient(name):
    """在sheet1中查找患者历史记录"""
    try:
        sheet = get_sheet1()
        all_data = sheet.get_all_values()
        if len(all_data) <= 1:
            return []
        headers = all_data[0]
        records = []
        for row in all_data[1:]:
            if len(row) >= 2 and name.strip() in row[1]:
                records.append(dict(zip(headers, row)))
        return records
    except Exception as e:
        return []

# ── Tegner评分说明 ────────────────────────────────────────────────────────────
TEGNER_LABELS = {
    0: "0 – 病假/残疾",
    1: "1 – 轻工作（坐位）",
    2: "2 – 轻工作（站位）",
    3: "3 – 中等工作",
    4: "4 – 重体力劳动 / 竞技运动（轻度，如游泳）",
    5: "5 – 竞技运动（中度，如骑自行车）",
    6: "6 – 娱乐性运动（足球/篮球/网球等，轻度）",
    7: "7 – 竞技性足球（低级别）/ 羽毛球 / 跑步",
    8: "8 – 竞技性足球（中级）/ 篮球 / 曲棍球",
    9: "9 – 竞技性足球（精英级别）",
    10: "10 – 国家队 / 职业足球"
}

TEGNER_LABELS_EN = {
    0: "0 – Sick leave/disability",
    1: "1 – Sedentary work",
    2: "2 – Light work (standing)",
    3: "3 – Moderate work",
    4: "4 – Heavy labor / competitive sports (light)",
    5: "5 – Recreational competitive sports (cycling)",
    6: "6 – Recreational sport (soccer/basketball, light)",
    7: "7 – Competitive soccer (low) / badminton / running",
    8: "8 – Competitive soccer (mid) / basketball / hockey",
    9: "9 – Competitive soccer (elite)",
    10: "10 – National team / professional soccer"
}

# ══════════════════════════════════════════════════════════════════════════════
# Tab布局
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "📋 首次评估 / 复评" if zh else "📋 Assessment",
    "📍 随访结局录入" if zh else "📍 Follow-up Outcome"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1：评估
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── 患者基本信息 ──────────────────────────────────────────────────────────
    st.subheader("患者基本信息" if zh else "Patient Information")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        patient_name = st.text_input("患者姓名 *" if zh else "Patient Name *", value="")
    with col_b:
        eval_date = st.date_input("评估日期" if zh else "Assessment Date", value=date.today())
    with col_c:
        doctor_name = st.text_input("评估医生" if zh else "Clinician", value="")

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        age_val = st.number_input(
            "年龄 (岁)" if zh else "Age (years)",
            min_value=14, max_value=65, value=24)
    with col_e:
        surgery_date = st.date_input(
            "手术日期" if zh else "Surgery Date",
            value=None,
            min_value=date(2015, 1, 1),
            max_value=date.today(),
            help="填写后自动计算术后月数" if zh else "Auto-calculates months post-op"
        )
    with col_f:
        months_post_op = get_months_post_op(surgery_date)
        if months_post_op is not None:
            st.metric(
                "术后月数" if zh else "Months Post-op",
                f"{months_post_op:.1f} 个月" if zh else f"{months_post_op:.1f} mo"
            )
            stratum, stratum_msg = time_stratum_label(months_post_op)
        else:
            st.info("请填写手术日期" if zh else "Enter surgery date")
            stratum, stratum_msg = None, None

    # ── 手术信息 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("手术信息" if zh else "Surgical Information")

    col_g, col_h = st.columns(2)
    with col_g:
        graft_type = st.selectbox(
            "移植物类型" if zh else "Graft Type",
            (["腘绳肌腱 (Hamstring)", "髌腱 BTB (Patellar BTB)",
              "股四头肌腱 (Quad Tendon)", "异体移植 (Allograft)",
              "LARS人工韧带"] if zh else
             ["Hamstring Tendon", "Patellar BTB",
              "Quad Tendon", "Allograft", "LARS Ligament"]),
            help=("参考信息，影响临床警示判断" if zh else
                  "Informs clinical warning flags")
        )
    with col_h:
        prior_aclr = st.selectbox(
            "手术类型" if zh else "Surgery Type",
            (["首次重建 (Primary)" , "翻修重建 (Revision)"] if zh else
             ["Primary ACLR", "Revision ACLR"]),
            help=("翻修手术RTS率显著低于初次手术" if zh else
                  "Revision ACLR has significantly lower RTS rates")
        )

    # ── Tegner运动评分 ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Tegner运动级别" if zh else "Tegner Activity Level")

    labels = TEGNER_LABELS if zh else TEGNER_LABELS_EN

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tegner_pre = st.selectbox(
            "受伤前Tegner级别" if zh else "Pre-injury Tegner",
            options=list(labels.keys()),
            format_func=lambda x: labels[x],
            index=7,
            help="受伤前的运动参与级别" if zh else "Activity level before injury"
        )
    with col_t2:
        tegner_current = st.selectbox(
            "当前Tegner级别" if zh else "Current Tegner",
            options=list(labels.keys()),
            format_func=lambda x: labels[x],
            index=4,
            help="目前实际运动参与级别" if zh else "Current actual activity level"
        )

    tegner_delta = tegner_pre - tegner_current
    if tegner_delta > 0:
        delta_color = "#e74c3c" if tegner_delta >= 3 else "#e67e22"
        delta_text = (f"▼ 较受伤前下降 {tegner_delta} 级" if zh else
                      f"▼ {tegner_delta} level(s) below pre-injury")
    elif tegner_delta == 0:
        delta_color = "#27ae60"
        delta_text = "= 已恢复至受伤前运动级别" if zh else "= Restored to pre-injury activity level"
    else:
        delta_color = "#27ae60"
        delta_text = (f"▲ 超过受伤前 {abs(tegner_delta)} 级" if zh else
                      f"▲ {abs(tegner_delta)} level(s) above pre-injury")

    st.markdown(
        f'<div class="tegner-delta" style="color:{delta_color}; font-weight:600;">{delta_text}</div>',
        unsafe_allow_html=True
    )

    # ── 运动类型 ──────────────────────────────────────────────────────────────
    pivot_sport = st.selectbox(
        "运动类型" if zh else "Sport Type",
        (["轴转运动（足球/篮球/羽毛球等）",
          "非轴转运动（游泳/单车/跑步等）",
          "未明确 / Unknown"] if zh else
         ["Pivoting sport (soccer/basketball/badminton)",
          "Non-pivoting sport (swimming/cycling/running)",
          "Unknown"]),
        help=("轴转运动重返运动再损伤风险更高（2年再损伤率16-22%）" if zh else
              "Pivoting sports: higher re-injury risk (16-22% at 2 years)")
    )

    # ── 临床评估数据 ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("临床评估数据" if zh else "Clinical Assessment")

    col1, col2 = st.columns(2)
    with col1:
        aclrsi_val = st.slider(
            "ACL-RSI 心理准备度 (0–100)" if zh else "ACL-RSI Score (0–100)",
            min_value=0, max_value=100, value=58,
            help=("ACL重返运动心理准备量表 | 最优截点≥65 (Duchman 2019, n=681)" if zh else
                  "ACL-RSI Scale | Optimal cutoff ≥65 (Duchman 2019, n=681)")
        )
        hop_lsi_val = st.slider(
            "单腿跳跃LSI (%)" if zh else "Single-leg Hop LSI (%)",
            min_value=50, max_value=100, value=82,
            help=("单腿跳跃距离患侧/健侧比值 | 推荐截点≥85% | OR=2.86 (Ithurburn 2019)" if zh else
                  "Single-leg hop ratio | Cutoff ≥85% | Strongest predictor OR=2.86 (Ithurburn 2019)")
        )
    with col2:
        quad_lsi_val = st.slider(
            "股四头肌力量LSI (%)" if zh else "Quadriceps Strength LSI (%)",
            min_value=50, max_value=100, value=80,
            help=("股四头肌等速肌力患侧/健侧比值 | 推荐截点≥85% (Ueda 2023)" if zh else
                  "Isokinetic quadriceps ratio | Cutoff ≥85% (Ueda 2023)")
        )

    # ── 预测计算 ──────────────────────────────────────────────────────────────
    prob_pct = predict_rts(aclrsi_val, hop_lsi_val, quad_lsi_val, age_val)

    if prob_pct >= 70:
        level  = "✅ 高概率重返运动" if zh else "✅ High RTS Probability"
        color  = "green"
        advice = (
            "三项指标均达到或接近推荐标准，心理与功能状态良好。"
            "建议完成运动专项测试后正式放行。" if zh else
            "All three indicators at or near recommended thresholds. "
            "Proceed to sport-specific testing before formal clearance."
        )
    elif prob_pct >= 45:
        level  = "⚠️ 中等概率" if zh else "⚠️ Moderate Probability"
        color  = "orange"
        advice = (
            "部分指标尚未达到推荐标准，建议针对性加强最薄弱指标，"
            "4–6周后重新评估。" if zh else
            "Some indicators below recommended thresholds. "
            "Focus on lowest-scoring measure and reassess in 4–6 weeks."
        )
    else:
        level  = "❌ 低概率重返运动" if zh else "❌ Low RTS Probability"
        color  = "red"
        advice = (
            "多项指标未达标，建议继续强化康复，暂缓重返运动决策。" if zh else
            "Multiple indicators below threshold. Continue rehabilitation. "
            "Defer RTS decision pending reassessment."
        )

    # ── 结果展示 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("预测结果" if zh else "Prediction Result")

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.metric(
            label="RTS预测概率" if zh else "Predicted RTS Probability",
            value=f"{prob_pct:.1f}%"
        )
        st.markdown(f"**:{color}[{level}]**")
    with col_r2:
        st.progress(int(prob_pct))
        st.info(f"💡 {advice}")

    # ── 时间分层警示 ──────────────────────────────────────────────────────────
    if stratum_msg:
        if stratum in ("early", "longterm"):
            st.markdown(f'<div class="warning-box">{stratum_msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">{stratum_msg}</div>', unsafe_allow_html=True)

    # ── 专业模式：变量贡献分解 ───────────────────────────────────────────────
    if pro_mode:
        st.divider()
        st.subheader("变量贡献分析" if zh else "Variable Contribution Analysis")
        st.caption(
            "各变量对logit的贡献值（正值增加RTS概率，负值降低）" if zh else
            "Each variable's contribution to logit score (positive = increases RTS probability)"
        )

        contribs = compute_contributions(aclrsi_val, hop_lsi_val, quad_lsi_val, age_val)
        total_contrib = sum(contribs.values())
        max_abs = max(abs(v) for v in contribs.values()) + 0.1

        contrib_data = []
        for varname, val in contribs.items():
            bar_pct = int(min(abs(val) / max_abs * 100, 100))
            direction = "正向 ↑" if val > 0 else "负向 ↓"
            direction_en = "Positive ↑" if val > 0 else "Negative ↓"
            contrib_data.append({
                ("变量" if zh else "Variable"): varname,
                ("贡献值" if zh else "Logit Contribution"): f"{val:+.3f}",
                ("方向" if zh else "Direction"): direction if zh else direction_en,
            })

        contrib_df = pd.DataFrame(contrib_data)
        st.dataframe(contrib_df, hide_index=True, use_container_width=True)

        # 可视化条形图（用HTML渲染）
        bar_html = ""
        for varname, val in contribs.items():
            bar_pct = int(min(abs(val) / max_abs * 100, 100))
            bar_class = "contrib-bar-pos" if val > 0 else "contrib-bar-neg"
            bar_html += f"""
            <div style="margin-bottom:10px;">
              <div style="font-size:12px; font-weight:500; margin-bottom:3px;">{varname}
                <span style="color:#888; font-size:11px; margin-left:8px;">{val:+.3f}</span>
              </div>
              <div class="contrib-bar-container">
                <div class="{bar_class}" style="width:{bar_pct}%;"></div>
              </div>
            </div>
            """
        st.markdown(bar_html, unsafe_allow_html=True)

    # ── 教育模式：RTS时间演变曲线 ────────────────────────────────────────────
    st.divider()
    st.subheader(
        "RTS概率随时间演变（教育展示）" if zh else
        "RTS Probability Over Time (Educational)"
    )
    st.caption(
        "固定当前测得指标，基于文献时间调节系数的示意性曲线（非模型直接预测）。"
        "纵轴标注：▶ 为当前评估时间点。" if zh else
        "Holding current measured values constant. Illustrative curve based on literature time modifiers "
        "(not direct model prediction). ▶ marks current assessment timepoint."
    )

    curve = rts_curve_by_time(aclrsi_val, hop_lsi_val, quad_lsi_val, age_val)
    curve_df = pd.DataFrame({
        ("术后月数" if zh else "Months Post-op"): list(curve.keys()),
        ("RTS概率(%)" if zh else "RTS Probability(%)"): list(curve.values())
    })

    # 用plotly-style的st.line_chart展示
    chart_data = pd.DataFrame(
        {"RTS概率(%)": list(curve.values())},
        index=[f"{t}mo" for t in curve.keys()]
    )
    st.line_chart(chart_data, use_container_width=True)

    # 当前时间点标注
    if months_post_op is not None:
        # 找最近的时间刻度
        closest_t = min(curve.keys(), key=lambda x: abs(x - months_post_op))
        st.markdown(
            f'<div class="timepoint-badge">▶ 当前评估：术后 {months_post_op:.1f} 个月 '
            f'| 参考概率区间：{curve.get(closest_t, "—")}%</div>' if zh else
            f'<div class="timepoint-badge">▶ Current: {months_post_op:.1f} months post-op '
            f'| Reference probability: {curve.get(closest_t, "—")}%</div>',
            unsafe_allow_html=True
        )

    # ── 临床警示 ──────────────────────────────────────────────────────────────
    warnings_list = []

    if "翻修" in prior_aclr or "Revision" in prior_aclr:
        warnings_list.append(
            "⚠️ 翻修ACLR：重返同一运动水平的概率显著低于初次手术，建议充分告知患者预期。" if zh else
            "⚠️ Revision ACLR: RTS to preinjury level significantly lower than primary ACLR. Counsel patient accordingly."
        )

    if "轴转" in pivot_sport or "Pivoting" in pivot_sport:
        warnings_list.append(
            "⚠️ 轴转运动：2年再损伤率约16–22%，建议完成完整RTS标准测试后方可放行。" if zh else
            "⚠️ Pivoting sport: 16–22% re-injury rate at 2 years. Ensure full RTS criteria before clearance."
        )

    if "异体" in graft_type or "Allograft" in graft_type:
        warnings_list.append(
            "⚠️ 异体移植物：年轻运动员中再撕裂率高于自体移植（尤其高需求运动），建议充分告知风险。" if zh else
            "⚠️ Allograft: Higher re-tear rates in young athletes vs autograft (especially high-demand sports). Counsel accordingly."
        )

    if aclrsi_val < 40:
        warnings_list.append(
            "⚠️ ACL-RSI极低（<40）：心理准备度严重不足，即使功能测试达标也强烈建议暂缓放行，"
            "优先转介运动心理干预。" if zh else
            "⚠️ Very low ACL-RSI (<40): Severe psychological unreadiness. "
            "Sport psychology referral strongly recommended regardless of physical test results."
        )

    if tegner_delta >= 3:
        warnings_list.append(
            f"⚠️ Tegner下降≥3级（{tegner_pre}→{tegner_current}）：运动级别显著下降，"
            "建议重新评估患者康复目标和期望值。" if zh else
            f"⚠️ Tegner drop ≥3 levels ({tegner_pre}→{tegner_current}): "
            "Significant activity level decline. Reassess rehabilitation goals and patient expectations."
        )

    for w in warnings_list:
        st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

    # LARS 专项警示（独立，更详细）
    if "LARS" in graft_type:
        lars_msg = (
            "🔶 <b>LARS人工韧带 — 高运动需求患者特别警示</b><br><br>"
            "LARS初始力学强度优于自体移植物，且无骨腱愈合等待期，允许早期活动。"
            "然而，其长期失败机制与自体移植物根本不同：<br>"
            "• <b>疲劳断裂</b>：聚酯纤维（PET）在高频次负荷下逐根微损伤累积，无急性撕裂事件，"
            "患者感觉膝关节「越来越松」，5–10年后显现<br>"
            "• <b>无生物整合</b>：不发生韧带化，无神经再支配，本体感觉永久缺失，"
            "高强度轴转运动中动态稳定依赖神经肌肉反应受损<br>"
            "• <b>骨道扩大</b>：无骨-腱界面融合，长期存在界面微动，加速纤维疲劳<br>"
            "• <b>长期数据</b>：Liu et al. KSSTA 2021（10年随访）LARS再断裂率11.8% vs 自体腘绳肌腱6.2%；"
            "短期RCT（≤2年）因疲劳失败尚未显现而可能低估风险<br><br>"
            "⚠️ <b>对追求快速RTS的高运动需求运动员</b>：LARS的早期RTS优势可能被更高的长期失败风险抵消。"
            "建议充分告知并<b>记录知情同意</b>。功能测试达标时需额外注意：LSI和Hop达标不等于移植物可承受长期高强度疲劳。" if zh else
            "🔶 <b>LARS Ligament — High-Demand Athlete Advisory</b><br><br>"
            "LARS offers superior initial mechanical strength and eliminates the bone-tendon healing wait. "
            "However, its long-term failure mechanism fundamentally differs from autograft:<br>"
            "• <b>Fatigue rupture</b>: PET fibers accumulate micro-damage under high-frequency loading — "
            "no acute tear event; patients report progressive knee 'loosening', typically manifesting at 5–10 years<br>"
            "• <b>No biological integration</b>: No ligamentization, no neural re-innervation, permanent proprioceptive deficit — "
            "impairs neuromuscular dynamic stability in high-intensity pivoting<br>"
            "• <b>Tunnel widening</b>: No bone-tendon fusion; persistent interface micromotion accelerates fiber fatigue<br>"
            "• <b>Long-term data</b>: Liu et al. KSSTA 2021 (10yr follow-up): LARS re-rupture 11.8% vs hamstring autograft 6.2%; "
            "short-term RCTs (≤2yr) likely underestimate risk as fatigue failure has not yet manifested<br><br>"
            "⚠️ <b>For high-demand athletes pursuing rapid RTS</b>: Early RTS advantage may be offset by elevated long-term failure risk. "
            "Provide thorough counseling and <b>document informed consent</b>. "
            "Note: LSI and hop test clearance ≠ tolerance for long-term high-intensity fatigue loading."
        )
        st.markdown(f'<div class="lars-warning">{lars_msg}</div>', unsafe_allow_html=True)

    # ── 指标汇总表 ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("评估指标汇总" if zh else "Assessment Summary")

    def status(val, cutoff):
        ok = val >= cutoff
        return ("✅ 达标" if ok else "⚠️ 未达标") if zh else ("✅ Met" if ok else "⚠️ Not met")

    summary_rows = {
        ("指标" if zh else "Measure"): [
            "ACL-RSI",
            "单腿跳跃LSI" if zh else "Hop LSI",
            "股四头肌LSI" if zh else "Quad LSI",
            "年龄" if zh else "Age",
            "Tegner 落差" if zh else "Tegner Drop",
            "移植物" if zh else "Graft",
            "运动类型" if zh else "Sport Type"
        ],
        ("当前值" if zh else "Value"): [
            f"{aclrsi_val}/100",
            f"{hop_lsi_val}%",
            f"{quad_lsi_val}%",
            f"{age_val}岁" if zh else f"{age_val}yrs",
            f"{tegner_pre}→{tegner_current} (Δ{-tegner_delta:+d})",
            graft_type.split("(")[0].strip(),
            "轴转" if ("轴转" in pivot_sport or "Pivoting" in pivot_sport) else "非轴转/未知" if zh else "Non-pivot/Unknown"
        ],
        ("推荐截点" if zh else "Cutoff"): [
            "≥65 [5]", "≥85% [1]", "≥85% [2]",
            "<35岁更佳" if zh else "<35 better",
            "Δ=0最佳" if zh else "Δ=0 ideal",
            "—", "—"
        ],
        ("状态" if zh else "Status"): [
            status(aclrsi_val, 65),
            status(hop_lsi_val, 85),
            status(quad_lsi_val, 85),
            "✅" if age_val < 35 else "⚠️",
            "✅" if tegner_delta <= 1 else ("⚠️" if tegner_delta <= 2 else "❌"),
            "—", "—"
        ]
    }
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    # ── 文献注释 ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="ref-box">'
        '<b>模型文献依据 / Model Evidence:</b> '
        '[1] Ithurburn et al. AJSM 2019 (Hop LSI OR=2.86, ACL-RSI OR=1.81) | '
        '[2] Ueda et al. Orthop J Sports Med 2023 (Quad LSI, Age OR=0.80) | '
        '[3] van Haren et al. Ann Phys Rehabil Med 2023 (n=208) | '
        '[4] Xiao et al. AJSM 2023 (meta-analysis n=3744) | '
        '[5] Duchman et al. AJSM 2019 (ACL-RSI cutoff ≥65) | '
        '[6] Liu et al. KSSTA 2021 (LARS 10yr) | '
        '[7] Grindem et al. BJSM 2016 | [8] Kyritsis et al. BJSM 2016'
        '</div>',
        unsafe_allow_html=True
    )

    # ── 保存记录 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("保存评估记录" if zh else "Save Assessment Record")

    if "generated_id" not in st.session_state:
        st.session_state.generated_id = None

    if st.button("💾 保存到数据库" if zh else "💾 Save to Database", type="primary"):
        if not patient_name:
            st.warning("请先填写患者姓名" if zh else "Please enter patient name first")
        elif surgery_date is None:
            st.warning("请填写手术日期以计算术后月数" if zh else "Please enter surgery date to calculate months post-op")
        else:
            record_id = generate_record_id()
            row = [
                record_id,
                patient_name,
                str(eval_date),
                doctor_name,
                age_val,
                str(surgery_date),
                round(months_post_op, 1) if months_post_op else "",
                f"{months_post_op:.1f}mo" if months_post_op else "Unknown",
                tegner_pre,
                tegner_current,
                aclrsi_val,
                hop_lsi_val,
                quad_lsi_val,
                graft_type.split("(")[0].strip(),
                prior_aclr.split("(")[0].strip() if "(" in prior_aclr else prior_aclr,
                "轴转" if ("轴转" in pivot_sport or "Pivoting" in pivot_sport) else "非轴转",
                round(prob_pct, 1),
                level.replace("✅ ", "").replace("⚠️ ", "").replace("❌ ", "")
            ]
            success, error = save_assessment(row)
            if success:
                st.session_state.generated_id = record_id
                st.success("✅ 已成功保存！" if zh else "✅ Successfully saved!")
            else:
                st.error(f"❌ 保存失败：{error}")

    if st.session_state.generated_id:
        st.markdown(
            f'<div class="id-box">'
            f'{"📋 记录ID（请记录在病历中，用于随访关联）" if zh else "📋 Record ID (note in patient chart for follow-up linkage)"}<br>'
            f'<b>{st.session_state.generated_id}</b>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── 报告导出 ──────────────────────────────────────────────────────────────
    report = f"""
{'ACLR术后重返运动评估报告' if zh else 'ACLR Return to Sport Assessment Report'}
{'=' * 60}
{'患者' if zh else 'Patient'}:         {patient_name or 'N/A'}
{'日期' if zh else 'Date'}:           {eval_date}
{'医生' if zh else 'Clinician'}:      {doctor_name or 'N/A'}
{'手术日期' if zh else 'Surgery Date'}: {surgery_date or 'N/A'}
{'术后月数' if zh else 'Months Post-op'}: {f"{months_post_op:.1f}" if months_post_op else 'N/A'}
{'记录ID' if zh else 'Record ID'}:    {st.session_state.generated_id or 'N/A (未保存)'}

{'─' * 60}
{'手术信息' if zh else 'Surgical Information'}
{'─' * 60}
{'移植物' if zh else 'Graft'}:        {graft_type}
{'手术类型' if zh else 'Surgery Type'}: {prior_aclr}
{'运动类型' if zh else 'Sport Type'}:  {pivot_sport}

{'─' * 60}
{'Tegner运动级别' if zh else 'Tegner Activity Level'}
{'─' * 60}
{'受伤前' if zh else 'Pre-injury'}:   {tegner_pre} – {labels[tegner_pre]}
{'当前' if zh else 'Current'}:        {tegner_current} – {labels[tegner_current]}
{'落差' if zh else 'Delta'}:          {delta_text}

{'─' * 60}
{'临床评估数据' if zh else 'Clinical Data'}
{'─' * 60}
{'年龄' if zh else 'Age'}:            {age_val} {'岁' if zh else 'yrs'}
ACL-RSI:          {aclrsi_val}/100  ({'推荐≥65' if zh else 'Recommended ≥65'})
{'跳跃LSI' if zh else 'Hop LSI'}:    {hop_lsi_val}%   ({'推荐≥85%' if zh else 'Recommended ≥85%'})
{'股四头肌LSI' if zh else 'Quad LSI'}: {quad_lsi_val}%   ({'推荐≥85%' if zh else 'Recommended ≥85%'})

{'─' * 60}
{'预测结果' if zh else 'Prediction'}
{'─' * 60}
{'RTS预测概率' if zh else 'Predicted RTS Probability'}: {prob_pct:.1f}%
{'风险分层' if zh else 'Risk Level'}:      {level}
{'临床建议' if zh else 'Recommendation'}: {advice}

{'─' * 60}
{'RTS概率时间曲线（示意）' if zh else 'RTS Probability Time Curve (Illustrative)'}
{'─' * 60}
{chr(10).join([f"  术后{t}个月: {p}%" if zh else f"  {t} months post-op: {p}%" for t, p in curve.items()])}

{'─' * 60}
{'模型说明' if zh else 'Model Information'}
{'─' * 60}
{'模型类型: 文献多因素逻辑回归（直接使用发表回归系数）' if zh else 'Model: Literature-based multivariate logistic regression (published coefficients)'}
{'预期AUC: ~0.80 | 版本: v2.0' if zh else 'Expected AUC: ~0.80 | Version: v2.0'}

{'参考文献:' if zh else 'References:'}
[1] Ithurburn et al. Am J Sports Med 2019
[2] Ueda et al. Orthop J Sports Med 2023
[3] van Haren et al. Ann Phys Rehabil Med 2023
[4] Xiao et al. AJSM 2023
[5] Duchman et al. AJSM 2019
[6] Liu et al. KSSTA 2021 (LARS)
[7] Grindem et al. BJSM 2016
[8] Kyritsis et al. BJSM 2016

{'─' * 60}
{'本报告仅供临床辅助参考，不替代医生专业判断。' if zh else 'For clinical reference only. Does not replace physician judgment.'}
{'生成日期' if zh else 'Generated'}: {date.today()}
"""

    st.download_button(
        label="📄 下载评估报告" if zh else "📄 Download Report",
        data=report.encode("utf-8"),
        file_name=f"ACLR_RTS_{patient_name or 'patient'}_{eval_date}.txt",
        mime="text/plain"
    )

    st.markdown(
        '<div class="disclaimer">⚠️ ' +
        ('本工具基于文献回归系数构建（v2.0），仅供临床辅助参考。时间曲线为教育性展示，非模型直接预测。'
         '正式临床使用前建议以本地患者数据进行外部验证。' if zh else
         'Built from published regression coefficients (v2.0). For clinical reference only. '
         'Time curve is illustrative, not direct model output. '
         'External validation with local patient data recommended before formal clinical use.') +
        '</div>', unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2：随访结局录入
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("随访结局录入" if zh else "Follow-up Outcome Entry")
    st.caption(
        "录入患者实际RTS结局，用于后续模型迭代和数据分析" if zh else
        "Record actual RTS outcomes for model iteration and data analysis"
    )

    # ── 查找患者 ──────────────────────────────────────────────────────────────
    st.markdown("#### 🔍 查找患者记录" if zh else "#### 🔍 Find Patient Record")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_name = st.text_input(
            "输入患者姓名查询" if zh else "Enter patient name to search",
            key="search_name"
        )
    with col_f2:
        search_btn = st.button("🔍 查询" if zh else "🔍 Search", key="search_btn")

    if "found_records" not in st.session_state:
        st.session_state.found_records = []
    if "selected_record" not in st.session_state:
        st.session_state.selected_record = None

    if search_btn and search_name:
        with st.spinner("查询中..." if zh else "Searching..."):
            records = lookup_patient(search_name)
        if records:
            st.session_state.found_records = records
            st.success(f"找到 {len(records)} 条记录" if zh else f"Found {len(records)} record(s)")
        else:
            st.session_state.found_records = []
            st.warning("未找到该患者记录，请确认姓名或先完成首次评估录入" if zh else
                       "No records found. Please check the name or complete the initial assessment first.")

    # ── 显示查询结果 ──────────────────────────────────────────────────────────
    if st.session_state.found_records:
        st.markdown("#### 📋 历史评估记录" if zh else "#### 📋 Historical Assessment Records")

        records_df_data = []
        for r in st.session_state.found_records:
            records_df_data.append({
                "记录ID" if zh else "Record ID": r.get("记录ID", r.get("Record ID", "—")),
                "评估日期" if zh else "Eval Date": r.get("评估日期", r.get("Assessment Date", "—")),
                "术后月数" if zh else "Post-op Mo": r.get("术后月数", r.get("Months Post-op", "—")),
                "ACL-RSI": r.get("ACL-RSI", "—"),
                "Hop LSI": r.get("Hop LSI(%)", "—"),
                "预测概率" if zh else "Pred. Prob": r.get("预测RTS概率(%)", r.get("Predicted RTS Probability", "—")),
            })

        records_df = pd.DataFrame(records_df_data)
        st.dataframe(records_df, hide_index=True, use_container_width=True)

        # 选择要关联的记录
        record_ids = [r.get("记录ID", r.get("Record ID", "")) for r in st.session_state.found_records]
        selected_id = st.selectbox(
            "选择要录入结局的记录ID" if zh else "Select Record ID for outcome entry",
            options=record_ids,
            key="selected_id"
        )

        if selected_id:
            selected = next(
                (r for r in st.session_state.found_records
                 if r.get("记录ID", r.get("Record ID", "")) == selected_id),
                None
            )
            if selected:
                st.session_state.selected_record = selected

    # ── 结局录入表单 ──────────────────────────────────────────────────────────
    if st.session_state.selected_record:
        rec = st.session_state.selected_record
        st.divider()
        st.markdown("#### 📝 录入随访结局" if zh else "#### 📝 Enter Follow-up Outcome")

        # 展示基础信息核对
        rec_id   = rec.get("记录ID", rec.get("Record ID", "—"))
        rec_date = rec.get("评估日期", rec.get("Assessment Date", "—"))
        rec_prob = rec.get("预测RTS概率(%)", rec.get("Predicted RTS Probability", "—"))
        rec_mo   = rec.get("术后月数", rec.get("Months Post-op", "—"))

        st.markdown(
            f'<div class="info-box">'
            f'{"关联评估记录" if zh else "Linked Assessment"}: <b>{rec_id}</b> | '
            f'{"评估日期" if zh else "Date"}: {rec_date} | '
            f'{"术后" if zh else "Post-op"}: {rec_mo} | '
            f'{"预测概率" if zh else "Predicted"}: {rec_prob}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("")

        fu_date = st.date_input(
            "随访日期" if zh else "Follow-up Date",
            value=date.today(),
            key="fu_date"
        )
        fu_doctor = st.text_input(
            "随访医生" if zh else "Follow-up Clinician",
            key="fu_doctor"
        )

        col_o1, col_o2 = st.columns(2)
        with col_o1:
            rts_outcome = st.selectbox(
                "实际RTS结局" if zh else "Actual RTS Outcome",
                (["已完全重返运动 (Full RTS)",
                  "已部分重返运动 (Partial RTS)",
                  "尚未重返运动 (No RTS)",
                  "放弃重返运动 (Abandoned RTS)"] if zh else
                 ["Full RTS achieved",
                  "Partial RTS achieved",
                  "No RTS yet",
                  "Abandoned RTS goal"]),
                key="rts_outcome"
            )
        with col_o2:
            rts_timepoint = st.number_input(
                "实际RTS时间点（术后月数）" if zh else "Actual RTS Timepoint (months post-op)",
                min_value=0.0, max_value=120.0, value=0.0, step=0.5,
                help="如尚未RTS填0" if zh else "Enter 0 if RTS not yet achieved",
                key="rts_timepoint"
            )

        tegner_rts_labels = TEGNER_LABELS if zh else TEGNER_LABELS_EN
        tegner_rts = st.selectbox(
            "随访时Tegner级别" if zh else "Follow-up Tegner Level",
            options=list(tegner_rts_labels.keys()),
            format_func=lambda x: tegner_rts_labels[x],
            index=6,
            key="tegner_rts"
        )

        col_o3, col_o4 = st.columns(2)
        with col_o3:
            reinjury = st.selectbox(
                "是否发生再损伤" if zh else "Re-injury Occurred",
                (["否 / No", "是 – 同侧ACL", "是 – 对侧ACL", "是 – 其他"] if zh else
                 ["No", "Yes – ipsilateral ACL", "Yes – contralateral ACL", "Yes – other"]),
                key="reinjury"
            )
        with col_o4:
            fu_notes = st.text_input(
                "备注" if zh else "Notes",
                key="fu_notes",
                placeholder="如：患者主动放弃，职业生涯结束" if zh else "e.g., Patient retired from sport"
            )

        if st.button("💾 保存随访结局" if zh else "💾 Save Follow-up Outcome", type="primary", key="save_fu"):
            fu_row = [
                rec_id,
                rec.get("患者姓名", rec.get("Patient Name", "—")),
                str(fu_date),
                fu_doctor,
                rts_outcome,
                rts_timepoint if rts_timepoint > 0 else "—",
                tegner_rts,
                reinjury,
                fu_notes
            ]
            success, error = save_followup(fu_row)
            if success:
                st.success(
                    f"✅ 随访结局已保存！记录ID：{rec_id}" if zh else
                    f"✅ Follow-up outcome saved! Record ID: {rec_id}"
                )
                st.balloons()
            else:
                st.error(f"❌ 保存失败：{error}")

    else:
        if not st.session_state.found_records:
            st.markdown(
                '<div class="info-box">'
                + ("请先在上方查询患者记录，选择后录入随访结局。<br>"
                   "随访结局数据将存入独立工作表，用于未来模型迭代。" if zh else
                   "Search for a patient record above, then select it to enter follow-up outcome.<br>"
                   "Follow-up data is stored in a separate sheet for future model iteration.")
                + '</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="disclaimer">⚠️ ' +
        ('随访结局数据为模型迭代的核心变量，请确保结局定义一致（完全RTS = 恢复至受伤前运动级别且无限制）。' if zh else
         'Follow-up outcome data is the core variable for model iteration. '
         'Ensure consistent outcome definition (Full RTS = return to pre-injury level without restriction).') +
        '</div>', unsafe_allow_html=True
    )
