import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, datetime
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
import warnings
import uuid
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ACLR RTS Predictor", page_icon="🦵", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.ref-box { background:#eaf4fb; border-left:4px solid #2E86AB; padding:10px 14px; border-radius:6px; font-size:12px; color:#1a3a4a; margin-top:8px; }
.warning-box { background:#fdedec; border-left:4px solid #e74c3c; padding:12px 16px; border-radius:6px; margin-top:10px; font-size:13px; color:#922b21; }
.revision-box { background:#fdf3e3; border-left:4px solid #d68910; padding:12px 16px; border-radius:6px; margin-top:10px; font-size:13px; color:#7d4e07; }
.lars-box { background:#fff3e0; border-left:4px solid #e65100; padding:12px 16px; border-radius:6px; margin-top:10px; font-size:13px; color:#7f3e00; }
.info-box { background:#f0f7ff; border-left:4px solid #1976d2; padding:12px 16px; border-radius:6px; margin-top:10px; font-size:13px; color:#0d47a1; }
.success-box { background:#e8f5e9; border-left:4px solid #2e7d32; padding:12px 16px; border-radius:6px; margin-top:10px; font-size:13px; color:#1b5e20; }
.id-box { background:#e8f5e9; border-left:4px solid #2e7d32; padding:14px 18px; border-radius:6px; margin-top:12px; font-size:14px; color:#1b5e20; font-family:'DM Mono',monospace; }
.result-card { background:#f8f9fa; border:1px solid #e0e0e0; border-radius:10px; padding:16px 20px; margin-bottom:12px; }
.result-card-title { font-size:11px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#888; margin-bottom:6px; }
.tegner-delta { font-size:13px; font-weight:600; margin-top:6px; }
.timepoint-badge { display:inline-block; background:#e3f2fd; color:#1565c0; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; margin-top:4px; }
.disclaimer { font-size:11px; color:#888; border-top:1px solid #eee; padding-top:12px; margin-top:20px; }
.contrib-bar-container { background:#f0f0f0; border-radius:4px; height:10px; width:100%; margin:3px 0 10px 0; }
.contrib-bar-pos { background:#2E86AB; border-radius:4px; height:10px; }
.contrib-bar-neg { background:#e74c3c; border-radius:4px; height:10px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 语言 + 模式
# ══════════════════════════════════════════════════════════════════════════════
lang     = st.sidebar.radio("Language / 语言", ["中文", "English"])
zh       = lang == "中文"
pro_mode = st.sidebar.toggle("专业模式 Pro Mode", value=True,
    help="开启后显示变量贡献分解 / Shows variable contribution breakdown")

# ── 侧边栏文献 ────────────────────────────────────────────────────────────────
with st.sidebar.expander("📚 模型文献依据" if zh else "📚 Model References", expanded=False):
    st.markdown("""
**[1]** Ithurburn et al. *AJSM* 2019 — ACL-RSI OR=1.81/10pts; Hop LSI OR=2.86/10%
**[2]** Ueda et al. *Orthop J Sports Med* 2023 — Quad LSI; Age OR=0.80/yr (样本15–38岁)
**[3]** van Haren et al. *Ann Phys Rehabil Med* 2023 — n=208, bootstrap validated
**[4]** Xiao et al. *AJSM* 2023 — meta-analysis n=3744
**[5]** Duchman et al. *AJSM* 2019 — ACL-RSI cutoff ≥65
**[6]** Liu et al. *KSSTA* 2021 — LARS 10yr re-rupture 11.8% vs 6.2%
**[7]** Grindem et al. *BJSM* 2016 — each month delay RTS → re-injury ↓51%
**[8]** Kyritsis et al. *BJSM* 2016 — RTS <9mo: 4× re-rupture risk
**[9]** Wright et al. *AJSM* 2011 — Revision ACLR RTS 51% vs Primary 72%
**[10]** Mohtadi et al. *AJSM* 2016 — Revision RTS OR≈0.25 vs Primary
    """)

with st.sidebar.expander("⏱️ 时间分层说明" if zh else "⏱️ Time Stratification", expanded=False):
    st.markdown("""
- **< 9个月**：移植物韧带化未完成，再损伤风险最高 [8]
- **9–12个月**：标准RTS评估窗口，模型可信度最高
- **13–24个月**：移植物趋于成熟，心理因素权重上升
- **> 24个月**：主要障碍转为废用萎缩和心理恐惧

*时间因素目前作为分层警示；待本地数据累积后迭代纳入β系数。*
    """)

with st.sidebar.expander("📊 运动分级说明 Level Classification", expanded=False):
    st.markdown("""
**Level I** — 急停急转 + 身体对抗（足球、篮球、手球、橄榄球）
**Level II** — 急停急转、无身体对抗（网球、羽毛球、滑雪、体操）
**Level III** — 直线跑动为主（跑步、田径、自行车竞技）
**Level IV** — 无跑动（游泳、划船、高尔夫）

*Hefti/Noyes分级体系；运动类型作为背景信息，不进入β计算，影响警示内容和随访结局定义。*
    """)

# ══════════════════════════════════════════════════════════════════════════════
# 模型参数
# ══════════════════════════════════════════════════════════════════════════════
# β系数来源（全部来自发表文献OR值）：
# ACL-RSI: OR=1.81/10pts → β=ln(1.81)/10=0.0593  [Ithurburn 2019]
# Hop LSI: OR=2.861/10%  → β=ln(2.861)/10=0.1051 [Ithurburn 2019]
# Quad LSI: OR=1.03/1%   → β=ln(1.03)=0.0296     [Ueda 2023]
# Age: OR=0.80/yr        → β=ln(0.80)=-0.2231     [Ueda 2023, 样本范围15-38岁]
# Intercept: 校准至文献基线RTS率62%
BETA = dict(intercept=-8.5806, aclrsi=0.0593, hop_lsi=0.1051, quad_lsi=0.0296, age=-0.2231)

def predict_rts(aclrsi, hop_lsi, quad_lsi, age):
    lo = (BETA['intercept'] + BETA['aclrsi']*aclrsi +
          BETA['hop_lsi']*hop_lsi + BETA['quad_lsi']*quad_lsi + BETA['age']*age)
    return 1 / (1 + np.exp(-lo)) * 100

def compute_contributions(aclrsi, hop_lsi, quad_lsi, age):
    return {
        'ACL-RSI':    BETA['aclrsi']   * aclrsi,
        'Hop LSI':    BETA['hop_lsi']  * hop_lsi,
        'Quad LSI':   BETA['quad_lsi'] * quad_lsi,
        'Age / 年龄': BETA['age']      * age,
    }

def rts_curve_by_time(aclrsi, hop_lsi, quad_lsi, age):
    """
    教育性时间演变曲线。时间调节系数来源：
    Kyritsis 2016（<9mo风险）、Toole 2017（9-24mo梯度）、
    临床共识（>24mo废用萎缩效应）
    """
    base = predict_rts(aclrsi, hop_lsi, quad_lsi, age)
    modifiers = {6:0.55, 9:1.0, 12:1.10, 18:1.18, 24:1.22, 36:1.18, 48:1.15}
    return {t: round(min(base * m, 99.0), 1) for t, m in modifiers.items()}

# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def generate_pid():
    return f"PT-{date.today().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"

def generate_rid():
    return f"ACLR-{date.today().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"

def generate_oid():
    return f"OUT-{date.today().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"

def get_months_post_op(surgery_date, eval_date):
    if surgery_date is None or eval_date is None:
        return None
    delta = eval_date - surgery_date
    return round((delta.days / 365.25) * 12, 1)

def time_stratum(months):
    if months is None:
        return None, None
    if months < 9:
        return "early", (
            "⚠️ 术后 < 9个月：移植物韧带化尚未完成，此阶段RTS再损伤风险最高（Kyritsis 2016）。"
            "功能测试达标不等于移植物已成熟，建议谨慎决策。" if zh else
            "⚠️ < 9 months: Graft ligamentization incomplete. Re-injury risk highest (Kyritsis 2016). "
            "Functional clearance ≠ graft maturity.")
    elif months <= 12:
        return "standard", (
            "✅ 术后 9–12个月：标准RTS评估窗口，与本模型文献来源时间段一致，预测可信度最高。" if zh else
            "✅ 9–12 months: Standard RTS assessment window. Highest model prediction confidence.")
    elif months <= 24:
        return "late", (
            "📋 术后 13–24个月：移植物趋于成熟，ACL-RSI心理准备度的权重在此阶段更为关键。" if zh else
            "📋 13–24 months: Graft maturing. ACL-RSI psychological readiness is increasingly critical.")
    else:
        return "longterm", (
            "⚠️ 术后 > 24个月：长期未重返运动。主要障碍已转为废用性萎缩和运动恐惧，"
            "建议重新评估康复目标，ACL-RSI权重尤为重要。" if zh else
            "⚠️ > 24 months: Long-term non-return. Barriers have shifted to disuse atrophy and kinesiophobia. "
            "Reassess rehab goals; ACL-RSI weighting is especially important.")

def tegner_goal_assessment(pre, current):
    """评估Tegner达标情况，考虑落差和目标级别绝对难度"""
    delta = pre - current
    # 高级别运动的每一级落差，临床意义更大
    difficulty_weight = 1.0 if pre <= 6 else (1.5 if pre <= 8 else 2.0)
    weighted_delta = delta * difficulty_weight

    if delta <= 0:
        return "achieved", (
            f"✅ 已达到或超越术前运动级别（Tegner {pre}）" if zh else
            f"✅ Achieved or exceeded pre-injury activity level (Tegner {pre})")
    elif weighted_delta < 1.5:
        return "near", (
            f"🟡 接近目标（落差{delta}级，目标Tegner {pre}）——针对性专项训练可达" if zh else
            f"🟡 Near target (gap {delta} level, target Tegner {pre}) — sport-specific training likely sufficient")
    elif weighted_delta < 3.0:
        return "gap", (
            f"🟠 存在差距（落差{delta}级，目标Tegner {pre}）——需3–6个月强化专项训练，建议重新讨论康复目标" if zh else
            f"🟠 Significant gap (drop {delta}, target Tegner {pre}) — 3–6 months targeted training needed; reassess goals")
    else:
        return "major", (
            f"🔴 显著差距（落差{delta}级，目标Tegner {pre}，加权难度系数×{difficulty_weight}）——"
            f"重返原运动级别难度极高，强烈建议与患者重新讨论现实可达的运动目标" if zh else
            f"🔴 Major gap (drop {delta}, target Tegner {pre}, difficulty ×{difficulty_weight}) — "
            f"Return to pre-injury level highly unlikely; strongly recommend discussion of realistic goals")

# ══════════════════════════════════════════════════════════════════════════════
# Tegner 标签
# ══════════════════════════════════════════════════════════════════════════════
TEGNER_ZH = {
    0:"0 – 病假/残疾", 1:"1 – 轻工作（坐位）", 2:"2 – 轻工作（站位）",
    3:"3 – 中等工作", 4:"4 – 重体力劳动 / 轻度竞技运动（游泳）",
    5:"5 – 竞技运动中度（骑自行车）", 6:"6 – 娱乐性运动（足球/篮球/网球，轻度）",
    7:"7 – 竞技性足球低级别 / 羽毛球 / 跑步", 8:"8 – 竞技性足球中级 / 篮球 / 曲棍球",
    9:"9 – 竞技性足球精英级", 10:"10 – 国家队 / 职业足球"
}
TEGNER_EN = {
    0:"0 – Sick leave/disability", 1:"1 – Sedentary work", 2:"2 – Light work (standing)",
    3:"3 – Moderate work", 4:"4 – Heavy labor / competitive sport (light, swimming)",
    5:"5 – Recreational competitive sport (cycling)", 6:"6 – Recreational sport (soccer/basketball, light)",
    7:"7 – Competitive soccer (low) / badminton / running", 8:"8 – Competitive soccer (mid) / basketball",
    9:"9 – Competitive soccer (elite)", 10:"10 – National team / professional soccer"
}

SPORT_LEVELS_ZH = [
    "Level I — 急停急转+身体对抗（足球、篮球、手球、橄榄球）",
    "Level II — 急停急转、无身体对抗（网球、羽毛球、滑雪、体操）",
    "Level III — 直线跑动为主（跑步、田径、竞技自行车）",
    "Level IV — 无跑动（游泳、划船、高尔夫）",
    "未明确 / Unknown"
]
SPORT_LEVELS_EN = [
    "Level I — Cutting/pivoting + contact (soccer, basketball, handball, rugby)",
    "Level II — Cutting/pivoting, non-contact (tennis, badminton, skiing, gymnastics)",
    "Level III — Primarily linear running (running, athletics, competitive cycling)",
    "Level IV — Non-running (swimming, rowing, golf)",
    "Unknown"
]

# ══════════════════════════════════════════════════════════════════════════════
# Google Sheets — 三表结构
# ══════════════════════════════════════════════════════════════════════════════
SHEET_PATIENTS    = "患者主表"
SHEET_ASSESSMENTS = "评估记录"
SHEET_OUTCOMES    = "随访结局"

HEADERS_PATIENTS = [
    "患者ID(PID)", "患者姓名", "手术日期", "移植物类型", "手术类型",
    "运动级别", "Tegner术前", "首次录入日期"
]
HEADERS_ASSESSMENTS = [
    "记录ID(RID)", "患者ID(PID)", "评估日期", "评估医生", "年龄",
    "术后月数", "ACL-RSI", "Hop_LSI(%)", "Quad_LSI(%)",
    "Tegner当前", "Tegner落差", "预测RTS概率(%)", "风险分层"
]
HEADERS_OUTCOMES = [
    "结局ID(OID)", "记录ID(RID)", "患者ID(PID)", "患者姓名",
    "随访日期", "随访医生",
    "RTS状态", "RTS(0/1)", "实际RTS月数",
    "随访Tegner", "TegnerGoalMet(0/1)",
    "再损伤(0/1)", "再损伤类型", "备注"
]

def get_workbook():
    """每次调用都建立新连接，确保读取最新数据"""
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["sheets"]["aclr_spreadsheet_id"])

def ensure_sheet(wb, name, headers):
    """确保工作表存在，不存在则创建并写入表头"""
    titles = [ws.title for ws in wb.worksheets()]
    if name not in titles:
        ws = wb.add_worksheet(title=name, rows=2000, cols=len(headers)+2)
        ws.append_row(headers)
    return wb.worksheet(name)

def init_sheets():
    """初始化三张表，返回(ws_patients, ws_assessments, ws_outcomes)"""
    try:
        wb = get_workbook()
        wp = ensure_sheet(wb, SHEET_PATIENTS,    HEADERS_PATIENTS)
        wa = ensure_sheet(wb, SHEET_ASSESSMENTS, HEADERS_ASSESSMENTS)
        wo = ensure_sheet(wb, SHEET_OUTCOMES,    HEADERS_OUTCOMES)
        return wp, wa, wo, None
    except Exception as e:
        return None, None, None, str(e)

def lookup_patient_by_name(name):
    """在患者主表中查找患者，返回记录列表"""
    try:
        wb = get_workbook()
        wp = wb.worksheet(SHEET_PATIENTS)
        all_rows = wp.get_all_records()
        return [r for r in all_rows if name.strip() in str(r.get("患者姓名",""))]
    except Exception as e:
        return []

def lookup_assessments_by_pid(pid, patient_name=""):
    """查找某患者所有评估记录，PID精确匹配，兜底用姓名模糊匹配"""
    try:
        wb = get_workbook()
        wa = wb.worksheet(SHEET_ASSESSMENTS)
        all_rows = wa.get_all_records()
        if not all_rows:
            return []
        # 主查询：PID精确匹配
        results = [r for r in all_rows if str(r.get("患者ID(PID)","")).strip() == str(pid).strip()]
        # 兜底：如果PID查不到，用姓名模糊匹配（兼容旧版无PID的数据）
        if not results and patient_name:
            results = [r for r in all_rows if patient_name.strip() in str(r.get("患者姓名", r.get("患者", "")))]
        return results
    except Exception as e:
        st.warning(f"评估记录查询出错：{e}")
        return []

def save_patient(wp, row):
    try:
        wp.append_row(row); return True, None
    except Exception as e:
        return False, str(e)

def save_assessment(wa, row):
    try:
        wa.append_row(row); return True, None
    except Exception as e:
        return False, str(e)

def save_outcome(wo, row):
    try:
        wo.append_row(row); return True, None
    except Exception as e:
        return False, str(e)

# ══════════════════════════════════════════════════════════════════════════════
# 标题
# ══════════════════════════════════════════════════════════════════════════════
st.title("🦵 ACLR术后重返运动预测器" if zh else "🦵 ACLR Return to Sport Predictor")
st.caption(
    "基于文献多因素逻辑回归系数 | AUC≈0.80 | v3.0 · 张瑞华医生" if zh else
    "Literature-based multivariate logistic regression | AUC≈0.80 | v3.0 · Dr. Jason Zhang")

# ══════════════════════════════════════════════════════════════════════════════
# Tab布局
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "📋 首次评估 / 复评" if zh else "📋 Assessment",
    "📍 随访结局录入"   if zh else "📍 Follow-up Outcome"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 评估
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── 患者基本信息 ──────────────────────────────────────────────────────────
    st.subheader("患者基本信息" if zh else "Patient Information")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        patient_name = st.text_input("患者姓名 *" if zh else "Patient Name *")
    with col_b:
        eval_date = st.date_input("评估日期" if zh else "Assessment Date", value=date.today())
    with col_c:
        doctor_name = st.text_input("评估医生" if zh else "Clinician")

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        age_val = st.number_input(
            "年龄 (岁)" if zh else "Age (years)",
            min_value=14, max_value=40, value=24,
            help=("模型文献来源人群年龄范围约15–38岁（Ueda 2023）。"
                  ">40岁患者模型外推可信度显著下降，结果仅供参考。" if zh else
                  "Model literature population age range ~15–38 yrs (Ueda 2023). "
                  "Extrapolation beyond 40 yrs has substantially reduced reliability."))
    with col_e:
        surgery_date = st.date_input(
            "手术日期" if zh else "Surgery Date",
            value=None, min_value=date(2015,1,1), max_value=date.today())
    with col_f:
        months_post_op = get_months_post_op(surgery_date, eval_date)
        if months_post_op is not None:
            st.metric("术后月数" if zh else "Months Post-op",
                      f"{months_post_op:.1f} 个月" if zh else f"{months_post_op:.1f} mo")
            stratum, stratum_msg = time_stratum(months_post_op)
        else:
            st.info("请填写手术日期" if zh else "Enter surgery date")
            stratum, stratum_msg = None, None

    # ── 手术背景信息 ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("手术背景信息" if zh else "Surgical Background")
    st.caption(
        "以下信息作为临床背景记录，不进入RTS概率计算，影响警示内容" if zh else
        "Background information — recorded for clinical context, does not alter RTS probability calculation")

    col_g, col_h, col_i = st.columns(3)
    with col_g:
        graft_type = st.selectbox(
            "移植物类型" if zh else "Graft Type",
            ["腘绳肌腱 (Hamstring)", "髌腱 BTB (Patellar BTB)",
             "股四头肌腱 (Quad Tendon)", "异体移植 (Allograft)", "LARS人工韧带"] if zh else
            ["Hamstring Tendon", "Patellar BTB", "Quad Tendon", "Allograft", "LARS Ligament"])
    with col_h:
        surgery_type = st.selectbox(
            "手术类型" if zh else "Surgery Type",
            ["首次重建 (Primary)", "翻修重建 (Revision)"] if zh else
            ["Primary ACLR", "Revision ACLR"])
    with col_i:
        sport_level = st.selectbox(
            "运动级别" if zh else "Sport Level",
            SPORT_LEVELS_ZH if zh else SPORT_LEVELS_EN,
            help=("Hefti/Noyes四级分类体系 | 详见侧边栏" if zh else
                  "Hefti/Noyes 4-level classification | See sidebar"))

    # ── Tegner运动级别 ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Tegner运动级别" if zh else "Tegner Activity Level")
    t_labels = TEGNER_ZH if zh else TEGNER_EN

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tegner_pre = st.selectbox(
            "受伤前Tegner级别" if zh else "Pre-injury Tegner",
            list(t_labels.keys()), format_func=lambda x: t_labels[x], index=7)
    with col_t2:
        tegner_now = st.selectbox(
            "当前Tegner级别" if zh else "Current Tegner",
            list(t_labels.keys()), format_func=lambda x: t_labels[x], index=4)

    tegner_delta = tegner_pre - tegner_now
    if tegner_delta > 0:
        delta_color = "#e74c3c" if tegner_delta >= 3 else "#e67e22"
        delta_txt   = f"▼ 较受伤前下降 {tegner_delta} 级" if zh else f"▼ {tegner_delta} level(s) below pre-injury"
    elif tegner_delta == 0:
        delta_color = "#27ae60"
        delta_txt   = "= 已恢复至受伤前运动级别" if zh else "= Restored to pre-injury level"
    else:
        delta_color = "#27ae60"
        delta_txt   = f"▲ 超过受伤前 {abs(tegner_delta)} 级" if zh else f"▲ {abs(tegner_delta)} level(s) above pre-injury"
    st.markdown(f'<div class="tegner-delta" style="color:{delta_color};">{delta_txt}</div>', unsafe_allow_html=True)

    # ── 临床评估数据 ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("临床评估数据" if zh else "Clinical Assessment")

    col1, col2 = st.columns(2)
    with col1:
        aclrsi_val = st.slider(
            "ACL-RSI 心理准备度 (0–100)" if zh else "ACL-RSI Score (0–100)",
            0, 100, 58,
            help="最优截点≥65 (Duchman 2019, n=681)")
        hop_lsi_val = st.slider(
            "单腿跳跃LSI (%)" if zh else "Single-leg Hop LSI (%)",
            50, 100, 82,
            help="推荐截点≥85% | 最强功能预测因子 OR=2.86 (Ithurburn 2019)")
    with col2:
        quad_lsi_val = st.slider(
            "股四头肌力量LSI (%)" if zh else "Quadriceps Strength LSI (%)",
            50, 100, 80,
            help="推荐截点≥85% (Ueda 2023)")

    # ══════════════════════════════════════════════════════════════════════════
    # 预测计算
    # ══════════════════════════════════════════════════════════════════════════
    prob_pct = predict_rts(aclrsi_val, hop_lsi_val, quad_lsi_val, age_val)

    if prob_pct >= 70:
        level="✅ 高概率重返运动" if zh else "✅ High RTS Probability"; color="green"
        advice=("三项指标均达到或接近推荐标准，心理与功能状态良好。建议完成运动专项测试后正式放行。" if zh else
                "All indicators at or near recommended thresholds. Proceed to sport-specific testing before clearance.")
    elif prob_pct >= 45:
        level="⚠️ 中等概率" if zh else "⚠️ Moderate Probability"; color="orange"
        advice=("部分指标尚未达到推荐标准，建议针对性加强最薄弱指标，4–6周后重新评估。" if zh else
                "Some indicators below recommended thresholds. Focus on lowest-scoring measure; reassess in 4–6 weeks.")
    else:
        level="❌ 低概率重返运动" if zh else "❌ Low RTS Probability"; color="red"
        advice=("多项指标未达标，建议继续强化康复，暂缓重返运动决策。" if zh else
                "Multiple indicators below threshold. Continue rehabilitation; defer RTS decision.")

    # Tegner达标评估
    tg_status, tg_msg = tegner_goal_assessment(tegner_pre, tegner_now)

    # ══════════════════════════════════════════════════════════════════════════
    # 双维度结果展示
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("预测结果" if zh else "Prediction Results")

    # 维度1：RTS概率
    st.markdown(
        f'<div class="result-card">'
        f'<div class="result-card-title">{"维度 1 · 重返运动可能性" if zh else "Dimension 1 · Return to Sport Likelihood"}</div>',
        unsafe_allow_html=True)
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.metric(label="RTS预测概率" if zh else "Predicted RTS Probability", value=f"{prob_pct:.1f}%")
        st.markdown(f"**:{color}[{level}]**")
    with col_r2:
        st.progress(int(prob_pct))
        st.info(f"💡 {advice}")
    st.markdown('</div>', unsafe_allow_html=True)

    # 翻修患者定量说明（紧接在维度1下方）
    is_revision = "翻修" in surgery_type or "Revision" in surgery_type
    if is_revision:
        st.markdown(
            f'<div class="revision-box">'
            f'{"⚠️ <b>翻修ACLR修正提示</b>：上方概率基于首次重建文献数据。"
               "翻修患者实际RTS率较首次重建低约20–30%（Wright et al. AJSM 2011：翻修51% vs 首次72%；"
               "Mohtadi et al. AJSM 2016：翻修OR≈0.25）。"
               "模型尚无可靠的多因素校正系数，建议专业用户在解读时自行下调预期，"
               "并在随访数据积累后迭代纳入翻修变量。" if zh else
               "⚠️ <b>Revision ACLR Adjustment Note</b>: The probability above is derived from primary ACLR literature. "
               "Revision patients achieve RTS approximately 20–30% less frequently than primary ACLR "
               "(Wright et al. AJSM 2011: revision 51% vs primary 72%; Mohtadi et al. AJSM 2016: revision OR≈0.25). "
               "No validated multivariate correction coefficient is currently available. "
               "Clinicians should adjust expectations accordingly; revision variable will be incorporated after local data accumulation."}'
            f'</div>',
            unsafe_allow_html=True)

    # 维度2：Tegner达标
    st.markdown(
        f'<div class="result-card" style="margin-top:12px;">'
        f'<div class="result-card-title">{"维度 2 · 重返术前运动级别（Tegner达标评估）" if zh else "Dimension 2 · Return to Pre-injury Activity Level (Tegner Goal Assessment)"}</div>',
        unsafe_allow_html=True)
    col_t3, col_t4 = st.columns([1, 2])
    with col_t3:
        st.metric(
            label="Tegner目标" if zh else "Tegner Target",
            value=str(tegner_pre),
            delta=f"当前 {tegner_now}" if zh else f"Current {tegner_now}")
    with col_t4:
        st.markdown(f"<br>{tg_msg}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 时间分层警示 ──────────────────────────────────────────────────────────
    if stratum_msg:
        box_class = "warning-box" if stratum in ("early","longterm") else "info-box"
        st.markdown(f'<div class="{box_class}">{stratum_msg}</div>', unsafe_allow_html=True)

    # ── 专业模式：变量贡献分解 ───────────────────────────────────────────────
    if pro_mode:
        st.divider()
        st.subheader("变量贡献分析" if zh else "Variable Contribution Analysis")
        st.caption("各变量对logit的贡献（正值增加RTS概率，负值降低）" if zh else
                   "Each variable's logit contribution (positive = increases probability)")
        contribs = compute_contributions(aclrsi_val, hop_lsi_val, quad_lsi_val, age_val)
        max_abs  = max(abs(v) for v in contribs.values()) + 0.1
        contrib_df = pd.DataFrame({
            ("变量" if zh else "Variable"):          list(contribs.keys()),
            ("贡献值" if zh else "Logit Contribution"): [f"{v:+.3f}" for v in contribs.values()],
            ("方向" if zh else "Direction"):         ["正向↑" if v>0 else "负向↓" for v in contribs.values()],
        })
        st.dataframe(contrib_df, hide_index=True, use_container_width=True)
        bar_html = ""
        for vname, val in contribs.items():
            pct   = int(min(abs(val)/max_abs*100, 100))
            cls   = "contrib-bar-pos" if val>0 else "contrib-bar-neg"
            bar_html += (f'<div style="margin-bottom:10px;">'
                         f'<div style="font-size:12px;font-weight:500;">{vname}'
                         f'<span style="color:#888;font-size:11px;margin-left:8px;">{val:+.3f}</span></div>'
                         f'<div class="contrib-bar-container"><div class="{cls}" style="width:{pct}%;"></div></div>'
                         f'</div>')
        st.markdown(bar_html, unsafe_allow_html=True)

    # ── RTS时间演变曲线 ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("RTS概率随时间演变（教育展示）" if zh else "RTS Probability Over Time (Educational)")
    st.caption(
        "固定当前测得指标，基于文献时间调节系数的示意性曲线（非模型直接预测）" if zh else
        "Illustrative curve based on literature time modifiers; not a direct model prediction")

    curve       = rts_curve_by_time(aclrsi_val, hop_lsi_val, quad_lsi_val, age_val)
    time_points = list(curve.keys())
    prob_values = list(curve.values())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points, y=prob_values,
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2.5),
        marker=dict(size=7, color="#2E86AB"),
        hovertemplate=(
            "术后 %{x} 个月<br>参考RTS概率：%{y:.1f}%<extra></extra>" if zh else
            "%{x} months post-op<br>Reference RTS probability: %{y:.1f}%<extra></extra>")))

    if months_post_op is not None and 6 <= months_post_op <= 48:
        fig.add_vline(
            x=months_post_op, line_dash="dash", line_color="#e74c3c", line_width=1.8,
            annotation_text=(f"▶ 当前评估\n{months_post_op:.1f}个月" if zh else f"▶ Now\n{months_post_op:.1f}mo"),
            annotation_position="top right",
            annotation_font_color="#e74c3c", annotation_font_size=11)

    fig.update_layout(
        xaxis=dict(title="术后月数" if zh else "Months Post-op",
                   tickmode="array", tickvals=time_points,
                   ticktext=[f"{t}{'个月' if zh else 'mo'}" for t in time_points],
                   showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="RTS概率 (%)" if zh else "RTS Probability (%)",
                   range=[0,100], showgrid=True, gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10,r=10,t=20,b=10), height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if months_post_op is not None:
        closest_t = min(time_points, key=lambda x: abs(x-months_post_op))
        st.markdown(
            f'<div class="timepoint-badge">▶ 当前评估：术后 {months_post_op:.1f} 个月 | 参考概率：{curve[closest_t]}%</div>' if zh else
            f'<div class="timepoint-badge">▶ Current: {months_post_op:.1f} months post-op | Reference: {curve[closest_t]}%</div>',
            unsafe_allow_html=True)

    # ── 临床警示 ──────────────────────────────────────────────────────────────
    st.divider()
    warnings_list = []

    if aclrsi_val < 40:
        warnings_list.append(
            "⚠️ ACL-RSI极低（<40）：心理准备度严重不足，即使功能测试达标也强烈建议暂缓放行，优先转介运动心理干预。" if zh else
            "⚠️ Very low ACL-RSI (<40): Severe psychological unreadiness. Sport psychology referral strongly recommended regardless of physical test results.")

    if "Level I" in sport_level:
        warnings_list.append(
            "⚠️ Level I运动（急停急转+对抗）：2年再损伤率约16–22%，为最高风险运动类型。建议完成完整RTS测试标准后方可放行。" if zh else
            "⚠️ Level I sport (cutting + contact): 2-year re-injury rate 16–22%, highest risk category. Ensure full RTS criteria before clearance.")
    elif "Level II" in sport_level:
        warnings_list.append(
            "📋 Level II运动（急停急转、无对抗）：再损伤风险中等，建议完成跳跃-落地动作质量评估。" if zh else
            "📋 Level II sport (cutting, non-contact): Moderate re-injury risk. Recommend landing mechanics assessment.")

    if "异体" in graft_type or "Allograft" in graft_type:
        warnings_list.append(
            "⚠️ 异体移植物：年轻高需求运动员中再撕裂率高于自体移植，建议充分告知风险。" if zh else
            "⚠️ Allograft: Higher re-tear rates in young high-demand athletes vs autograft. Counsel accordingly.")

    if tegner_delta >= 3:
        warnings_list.append(
            f"⚠️ Tegner落差≥3级（{tegner_pre}→{tegner_now}）：运动级别显著下降，建议与患者重新讨论康复目标和运动期望值。" if zh else
            f"⚠️ Tegner drop ≥3 levels ({tegner_pre}→{tegner_now}): Major activity decline. Reassess rehab goals and patient expectations.")

    for w in warnings_list:
        st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

    if "LARS" in graft_type:
        lars_msg = (
            "🔶 <b>LARS人工韧带 — 高运动需求患者特别警示</b><br>"
            "LARS初始强度高、无骨腱愈合等待期，但长期失败机制与自体移植物根本不同：<br>"
            "• <b>疲劳断裂</b>：PET纤维高频负荷下逐根微损伤累积，无急性撕裂事件，5–10年后显现<br>"
            "• <b>无生物整合</b>：永久缺失本体感觉神经支配，轴转运动中动态稳定受损<br>"
            "• <b>骨道扩大</b>：界面微动加速纤维疲劳<br>"
            "• <b>长期数据</b>：Liu et al. KSSTA 2021（10年）：LARS再断裂率11.8% vs 腘绳肌腱6.2%<br>"
            "⚠️ <b>追求快速RTS的高运动需求运动员</b>：早期RTS优势可能被长期失败风险抵消。建议充分告知并<b>记录知情同意</b>。" if zh else
            "🔶 <b>LARS Ligament — High-Demand Athlete Advisory</b><br>"
            "LARS offers high initial strength and no bone-tendon healing wait, but long-term failure mechanism fundamentally differs:<br>"
            "• <b>Fatigue rupture</b>: PET fiber micro-damage accumulates under high-frequency loading; no acute event; manifests 5–10 years post-op<br>"
            "• <b>No biological integration</b>: Permanent proprioceptive deficit; impaired neuromuscular dynamic stability in pivoting<br>"
            "• <b>Tunnel widening</b>: Interface micromotion accelerates fiber fatigue<br>"
            "• <b>Long-term data</b>: Liu et al. KSSTA 2021 (10yr): LARS re-rupture 11.8% vs hamstring 6.2%<br>"
            "⚠️ <b>High-demand athletes pursuing rapid RTS</b>: Early RTS advantage may be offset by long-term failure risk. Document informed consent.")
        st.markdown(f'<div class="lars-box">{lars_msg}</div>', unsafe_allow_html=True)

    # ── 评估指标汇总表 ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("评估指标汇总" if zh else "Assessment Summary")

    def met(val, cutoff): return ("✅ 达标" if val>=cutoff else "⚠️ 未达标") if zh else ("✅ Met" if val>=cutoff else "⚠️ Not met")

    summary = pd.DataFrame({
        ("指标" if zh else "Measure"):     ["ACL-RSI","Hop LSI","Quad LSI","年龄" if zh else "Age","Tegner落差" if zh else "Tegner Drop"],
        ("当前值" if zh else "Value"):     [f"{aclrsi_val}/100",f"{hop_lsi_val}%",f"{quad_lsi_val}%",
                                            f"{age_val}岁" if zh else f"{age_val}yrs",f"{tegner_pre}→{tegner_now} (Δ{-tegner_delta:+d})"],
        ("推荐截点" if zh else "Cutoff"):  ["≥65 [5]","≥85% [1]","≥85% [2]","15–40岁(模型范围)" if zh else "15–40yrs (model range)","Δ=0最佳" if zh else "Δ=0 ideal"],
        ("状态" if zh else "Status"):      [met(aclrsi_val,65),met(hop_lsi_val,85),met(quad_lsi_val,85),
                                            "✅" if age_val<=38 else "⚠️",
                                            "✅" if tegner_delta<=1 else ("⚠️" if tegner_delta<=2 else "❌")]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="ref-box"><b>模型文献依据:</b> '
        '[1] Ithurburn AJSM 2019 | [2] Ueda Orthop J Sports Med 2023 | '
        '[3] van Haren Ann Phys Rehabil Med 2023 | [4] Xiao AJSM 2023 | '
        '[5] Duchman AJSM 2019 | [6] Liu KSSTA 2021 | '
        '[7] Grindem BJSM 2016 | [8] Kyritsis BJSM 2016 | '
        '[9] Wright AJSM 2011 | [10] Mohtadi AJSM 2016</div>',
        unsafe_allow_html=True)

    # ── 保存记录 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("保存评估记录" if zh else "Save Assessment Record")

    if "saved_pid" not in st.session_state: st.session_state.saved_pid = None
    if "saved_rid" not in st.session_state: st.session_state.saved_rid = None

    if st.button("💾 保存到数据库" if zh else "💾 Save to Database", type="primary"):
        if not patient_name:
            st.warning("请先填写患者姓名" if zh else "Please enter patient name")
        elif surgery_date is None:
            st.warning("请填写手术日期" if zh else "Please enter surgery date")
        else:
            wp, wa, wo, err = init_sheets()
            if err:
                st.error(f"数据库连接失败：{err}")
            else:
                # 检查患者主表是否已有该患者（PID必须非空才算有效）
                existing = lookup_patient_by_name(patient_name)
                existing_valid = [r for r in existing if str(r.get("患者ID(PID)","")).strip()]
                if existing_valid:
                    pid = existing_valid[0].get("患者ID(PID)")
                    st.info(f"{'找到已有患者记录，复用患者ID：' if zh else 'Existing patient found, reusing PID: '}{pid}")
                else:
                    pid = generate_pid()
                    p_row = [pid, patient_name, str(surgery_date),
                             graft_type.split("(")[0].strip(),
                             surgery_type.split("(")[0].strip(),
                             sport_level.split("—")[0].strip(),
                             tegner_pre, str(date.today())]
                    ok, e = save_patient(wp, p_row)
                    if not ok:
                        st.error(f"患者主表保存失败：{e}")
                    else:
                        # 如果该患者已有行但PID为空，回填PID
                        if existing:
                            try:
                                wb2 = get_workbook()
                                wp2 = wb2.worksheet(SHEET_PATIENTS)
                                all_vals = wp2.get_all_values()
                                for i, row in enumerate(all_vals[1:], start=2):
                                    if len(row) > 1 and row[1] == patient_name and not str(row[0]).strip():
                                        wp2.update_cell(i, 1, pid)
                            except Exception:
                                pass  # 回填失败不影响主流程

                rid = generate_rid()
                a_row = [rid, pid, str(eval_date), doctor_name, age_val,
                         round(months_post_op,1) if months_post_op else "",
                         aclrsi_val, hop_lsi_val, quad_lsi_val,
                         tegner_now, tegner_delta,
                         round(prob_pct,1),
                         level.replace("✅ ","").replace("⚠️ ","").replace("❌ ","")]
                ok, e = save_assessment(wa, a_row)
                if ok:
                    st.session_state.saved_pid = pid
                    st.session_state.saved_rid = rid
                    st.success("✅ 已成功保存！" if zh else "✅ Successfully saved!")
                else:
                    st.error(f"评估记录保存失败：{e}")

    if st.session_state.saved_rid:
        st.markdown(
            f'<div class="id-box">'
            f'{"📋 请将以下ID记录在病历中，用于随访关联<br>" if zh else "📋 Record these IDs in patient chart for follow-up linkage<br>"}'
            f'患者ID (PID): <b>{st.session_state.saved_pid}</b><br>'
            f'记录ID (RID): <b>{st.session_state.saved_rid}</b>'
            f'</div>', unsafe_allow_html=True)

    # ── 报告导出 ──────────────────────────────────────────────────────────────
    t_labels_use = TEGNER_ZH if zh else TEGNER_EN
    report = f"""
{'ACLR术后重返运动评估报告 v3.0' if zh else 'ACLR Return to Sport Assessment Report v3.0'}
{'='*60}
{'患者' if zh else 'Patient'}:          {patient_name or 'N/A'}
{'评估日期' if zh else 'Date'}:         {eval_date}
{'评估医生' if zh else 'Clinician'}:    {doctor_name or 'N/A'}
{'手术日期' if zh else 'Surgery Date'}: {surgery_date or 'N/A'}
{'术后月数' if zh else 'Post-op Mo'}:   {f"{months_post_op:.1f}" if months_post_op else 'N/A'}
PID: {st.session_state.saved_pid or 'N/A'}
RID: {st.session_state.saved_rid or 'N/A (未保存)'}

{'─'*60}
{'手术背景' if zh else 'Surgical Background'}
{'─'*60}
{'移植物' if zh else 'Graft'}:          {graft_type}
{'手术类型' if zh else 'Surgery Type'}:  {surgery_type}
{'运动级别' if zh else 'Sport Level'}:   {sport_level}

{'─'*60}
Tegner
{'─'*60}
{'受伤前' if zh else 'Pre-injury'}:     {tegner_pre} — {t_labels_use[tegner_pre]}
{'当前' if zh else 'Current'}:          {tegner_now} — {t_labels_use[tegner_now]}
{'落差' if zh else 'Delta'}:            {delta_txt}

{'─'*60}
{'临床评估数据' if zh else 'Clinical Data'}
{'─'*60}
{'年龄' if zh else 'Age'}:              {age_val} {'岁' if zh else 'yrs'}
ACL-RSI:           {aclrsi_val}/100
{'跳跃LSI' if zh else 'Hop LSI'}:       {hop_lsi_val}%
{'股四头肌LSI' if zh else 'Quad LSI'}:  {quad_lsi_val}%

{'─'*60}
{'预测结果' if zh else 'Prediction Results'}
{'─'*60}
{'维度1 RTS预测概率' if zh else 'Dim1 RTS Probability'}:  {prob_pct:.1f}%  {level}
{'临床建议' if zh else 'Recommendation'}: {advice}
{"翻修修正提示：实际RTS率可能较上述概率低20-30%" if is_revision else ""}

{'维度2 Tegner达标评估' if zh else 'Dim2 Tegner Goal Assessment'}:
{tg_msg}

{'─'*60}
{'RTS时间曲线（示意）' if zh else 'RTS Time Curve (Illustrative)'}
{'─'*60}
{chr(10).join([f"  术后{t}个月: {p}%" if zh else f"  {t}mo: {p}%" for t,p in curve.items()])}

{'─'*60}
{'模型说明' if zh else 'Model Info'}
{'─'*60}
{'文献多因素逻辑回归 | AUC≈0.80 | v3.0' if zh else 'Literature-based logistic regression | AUC≈0.80 | v3.0'}
{'年龄适用范围：15–40岁（超出范围可信度下降）' if zh else 'Age range: 15–40 yrs (reduced reliability outside range)'}
{'移植物/手术类型/运动级别不进入β计算，作为背景信息记录' if zh else 'Graft/surgery type/sport level recorded as background; not in β calculation'}

{'本报告仅供临床辅助参考，不替代医生专业判断。' if zh else 'For clinical reference only. Does not replace physician judgment.'}
{'生成日期' if zh else 'Generated'}: {date.today()}
"""
    st.download_button(
        label="📄 下载评估报告" if zh else "📄 Download Report",
        data=report.encode("utf-8"),
        file_name=f"ACLR_RTS_{patient_name or 'patient'}_{eval_date}.txt",
        mime="text/plain")

    st.markdown(
        '<div class="disclaimer">⚠️ ' +
        ('本工具基于文献回归系数构建（v3.0）。年龄上限40岁；移植物/手术类型/运动级别不进入概率计算；'
         '时间曲线为教育性展示。正式临床使用前建议以本地患者数据进行外部验证。' if zh else
         'Built from published regression coefficients (v3.0). Age capped at 40 yrs; '
         'graft/surgery type/sport level are background variables only; time curve is illustrative. '
         'External validation with local data recommended before formal clinical use.') +
        '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 随访结局录入
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("随访结局录入" if zh else "Follow-up Outcome Entry")
    st.caption(
        "录入患者实际RTS结局（RTS状态 + Tegner达标），用于后续模型迭代" if zh else
        "Record actual outcomes (RTS status + Tegner goal met) for future model iteration")

    st.markdown(
        '<div class="info-box">'
        + ("📊 <b>数据结构说明</b>：结局数据存入<b>随访结局表</b>，通过记录ID(RID)关联评估记录表，"
           "通过患者ID(PID)关联患者主表。两个结局变量分别记录：<br>"
           "• <b>RTS(0/1)</b>：是否重返任何运动活动（宽泛定义）<br>"
           "• <b>TegnerGoalMet(0/1)</b>：随访Tegner是否≥术前Tegner（严格定义）" if zh else
           "📊 <b>Data Structure</b>: Outcome data stored in <b>Outcomes sheet</b>, linked to Assessment sheet via RID "
           "and to Patient sheet via PID. Two outcome variables:<br>"
           "• <b>RTS(0/1)</b>: Return to any sporting activity (broad definition)<br>"
           "• <b>TegnerGoalMet(0/1)</b>: Follow-up Tegner ≥ pre-injury Tegner (strict definition)")
        + '</div>', unsafe_allow_html=True)

    # ── 查找患者 ──────────────────────────────────────────────────────────────
    st.markdown("#### 🔍 查找患者" if zh else "#### 🔍 Find Patient")
    col_s1, col_s2 = st.columns([2,1])
    with col_s1:
        search_name = st.text_input("输入患者姓名" if zh else "Enter patient name", key="fu_search_name")
    with col_s2:
        search_btn = st.button("🔍 查询" if zh else "🔍 Search", key="fu_search_btn")

    if "fu_patients"   not in st.session_state: st.session_state.fu_patients   = []
    if "fu_assessments" not in st.session_state: st.session_state.fu_assessments = []
    if "fu_selected_rid" not in st.session_state: st.session_state.fu_selected_rid = None
    if "fu_selected_pid" not in st.session_state: st.session_state.fu_selected_pid = None
    if "fu_tegner_pre"   not in st.session_state: st.session_state.fu_tegner_pre   = None

    if search_btn and search_name:
        with st.spinner("查询中..." if zh else "Searching..."):
            patients = lookup_patient_by_name(search_name)
        if patients:
            st.session_state.fu_patients = patients
            st.success(f"找到 {len(patients)} 位患者" if zh else f"Found {len(patients)} patient(s)")
        else:
            st.session_state.fu_patients = []
            st.warning("未找到患者记录" if zh else "No patient records found")

    # 选择患者
    if st.session_state.fu_patients:
        pid_options = {f"{p['患者姓名']} | {p['患者ID(PID)']} | 术前Tegner:{p.get('Tegner术前','-')}": p
                       for p in st.session_state.fu_patients}
        selected_p_key = st.selectbox(
            "选择患者" if zh else "Select Patient", list(pid_options.keys()), key="fu_select_patient")
        selected_p = pid_options[selected_p_key]
        pid_selected = selected_p.get("患者ID(PID)","")
        tegner_pre_fu = selected_p.get("Tegner术前", None)

        # 查找该患者的评估记录（选填，没有评估记录也可录入随访结局）
        patient_name_fu = selected_p.get("患者姓名", "")
        assessments = lookup_assessments_by_pid(pid_selected, patient_name_fu)
        rid_selected = ""
        selected_r = {}
        if assessments:
            rid_options = {"（不关联评估记录）": {}} if True else {}
            rid_options.update({
                f"RID:{a['记录ID(RID)']} | 评估:{a.get('评估日期','-')} | 术后:{a.get('术后月数','-')}月 | 预测:{a.get('预测RTS概率(%)','-')}%": a
                for a in assessments})
            selected_r_key = st.selectbox(
                "关联评估记录（选填）" if zh else "Link to assessment record (optional)",
                list(rid_options.keys()), key="fu_select_rid")
            selected_r = rid_options[selected_r_key]
            rid_selected = selected_r.get("记录ID(RID)","")
            if rid_selected:
                st.markdown(
                    f'<div class="info-box">'
                    f'{"关联评估" if zh else "Linked assessment"}: <b>{rid_selected}</b> | '
                    f'ACL-RSI: {selected_r.get("ACL-RSI","-")} | '
                    f'Hop LSI: {selected_r.get("Hop_LSI(%)","-")}% | '
                    f'{"预测概率" if zh else "Predicted"}: {selected_r.get("预测RTS概率(%)","-")}%'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.caption("该患者暂无评估记录，将直接录入随访结局（RID留空）" if zh else
                       "No assessment records found. Outcome will be recorded without RID link.")

    # ── 结局录入表单（查到患者即显示，不强制要求有评估记录）──────────────────
    if st.session_state.fu_patients and st.session_state.fu_patients:
        st.divider()
        st.markdown("#### 📝 录入随访结局" if zh else "#### 📝 Enter Outcome")

        col_fu1, col_fu2 = st.columns(2)
        with col_fu1:
            fu_date   = st.date_input("随访日期" if zh else "Follow-up Date", value=date.today(), key="fu_date")
            fu_doctor = st.text_input("随访医生" if zh else "Follow-up Clinician", key="fu_doctor")
        with col_fu2:
            fu_rts_status = st.selectbox(
                "RTS状态" if zh else "RTS Status",
                ["已完全重返运动", "已部分重返运动", "尚未重返运动", "放弃重返运动"] if zh else
                ["Full RTS achieved", "Partial RTS achieved", "No RTS yet", "Abandoned RTS goal"],
                key="fu_rts_status")
            fu_rts_months = st.number_input(
                "实际RTS时间点（术后月数）" if zh else "Actual RTS timepoint (months post-op)",
                min_value=0.0, max_value=120.0, value=0.0, step=0.5,
                help="未RTS填0" if zh else "Enter 0 if not yet returned",
                key="fu_rts_months")

        # RTS(0/1)自动编码
        rts_binary = 1 if "完全" in fu_rts_status or "Full" in fu_rts_status else (
                     0 if ("尚未" in fu_rts_status or "No RTS" in fu_rts_status or
                           "放弃" in fu_rts_status or "Abandoned" in fu_rts_status) else -1)
        rts_binary_display = {1:"1（完全RTS）", 0:"0（未RTS）", -1:"0.5（部分RTS，需人工确认）"}
        st.caption(f"RTS(0/1) 自动编码：{rts_binary_display[rts_binary]}" if zh else
                   f"RTS(0/1) auto-coded: {rts_binary_display[rts_binary]}")

        # 随访Tegner + TegnerGoalMet
        fu_tegner = st.selectbox(
            "随访Tegner级别" if zh else "Follow-up Tegner Level",
            list(t_labels_use.keys()), format_func=lambda x: t_labels_use[x],
            index=6, key="fu_tegner")

        try:
            tegner_pre_int = int(tegner_pre_fu)
            tegner_goal_met = 1 if fu_tegner >= tegner_pre_int else 0
            st.markdown(
                f'{"✅ TegnerGoalMet = 1（随访Tegner" if zh else "✅ TegnerGoalMet = 1 (follow-up Tegner"} '
                f'{fu_tegner} {"≥ 术前" if zh else "≥ pre-injury"} {tegner_pre_int}）' if tegner_goal_met else
                f'{"⚠️ TegnerGoalMet = 0（随访Tegner" if zh else "⚠️ TegnerGoalMet = 0 (follow-up Tegner"} '
                f'{fu_tegner} {"< 术前" if zh else "< pre-injury"} {tegner_pre_int}）')
        except (TypeError, ValueError):
            tegner_goal_met = -1
            st.caption("Tegner术前数据缺失，TegnerGoalMet需手动确认" if zh else
                        "Pre-injury Tegner missing; TegnerGoalMet requires manual confirmation")

        col_fu3, col_fu4 = st.columns(2)
        with col_fu3:
            fu_reinjury = st.selectbox(
                "是否发生再损伤" if zh else "Re-injury",
                ["否 / No","是 – 同侧ACL","是 – 对侧ACL","是 – 其他"] if zh else
                ["No","Yes – ipsilateral ACL","Yes – contralateral ACL","Yes – other"],
                key="fu_reinjury")
        with col_fu4:
            fu_notes = st.text_input("备注" if zh else "Notes", key="fu_notes",
                                     placeholder="如：职业生涯结束，主动放弃" if zh else "e.g., Retired from sport")

        reinjury_binary = 0 if ("否" in fu_reinjury or fu_reinjury=="No") else 1

        if st.button("💾 保存随访结局" if zh else "💾 Save Follow-up Outcome", type="primary", key="fu_save"):
            _, _, wo, err = init_sheets()
            if err:
                st.error(f"数据库连接失败：{err}")
            else:
                oid = generate_oid()
                o_row = [
                    oid, rid_selected, pid_selected,
                    selected_p.get("患者姓名",""),
                    str(fu_date), fu_doctor,
                    fu_rts_status,
                    rts_binary if rts_binary != -1 else "0.5",
                    fu_rts_months if fu_rts_months > 0 else "",
                    fu_tegner,
                    tegner_goal_met if tegner_goal_met != -1 else "",
                    reinjury_binary, fu_reinjury,
                    fu_notes
                ]
                ok, e = save_outcome(wo, o_row)
                if ok:
                    st.success(f"✅ 随访结局已保存！结局ID：{oid}" if zh else f"✅ Outcome saved! OID: {oid}")
                    st.balloons()
                else:
                    st.error(f"❌ 保存失败：{e}")

    st.markdown(
        '<div class="disclaimer">⚠️ ' +
        ('结局定义：RTS(0/1)=重返任何运动活动；TegnerGoalMet(0/1)=随访Tegner≥术前Tegner。'
         '部分RTS在数据库中编码为0.5，分析前需根据研究问题统一处理。' if zh else
         'Outcome definitions: RTS(0/1)=return to any sport; TegnerGoalMet(0/1)=follow-up Tegner≥pre-injury Tegner. '
         'Partial RTS coded as 0.5; standardize before analysis according to research question.') +
        '</div>', unsafe_allow_html=True)
