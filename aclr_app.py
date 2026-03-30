import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
import gspread
from google.oauth2.service_account import Credentials
import warnings
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
    .disclaimer {
        font-size: 11px; color: #888;
        border-top: 1px solid #eee;
        padding-top: 12px; margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── 语言 ──────────────────────────────────────────────────────────────────────
lang = st.sidebar.radio("Language / 语言", ["中文", "English"])
zh = lang == "中文"

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
    """)

# ── 标题 ──────────────────────────────────────────────────────────────────────
st.title("🦵 ACLR术后重返运动预测器" if zh else "🦵 ACLR Return to Sport Predictor")
st.caption(
    "基于文献多因素逻辑回归系数 | AUC≈0.80 | 参考文献: Ithurburn 2019, Ueda 2023, van Haren 2023" if zh else
    "Literature-based multivariate logistic regression | AUC≈0.80 | Ref: Ithurburn 2019, Ueda 2023, van Haren 2023"
)

# ── 文献来源模型参数 ──────────────────────────────────────────────────────────
# β系数直接来自文献OR值
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

# ── Google Sheets ─────────────────────────────────────────────────────────────
def get_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_key(
        st.secrets["sheets"]["aclr_spreadsheet_id"]).sheet1

def save_to_sheets(row):
    try:
        get_sheet().append_row(row)
        return True, None
    except Exception as e:
        return False, str(e)

# ── 患者信息 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("患者基本信息" if zh else "Patient Information")

col_a, col_b, col_c = st.columns(3)
with col_a:
    patient_name = st.text_input("患者姓名" if zh else "Patient Name", value="")
with col_b:
    eval_date = st.date_input("评估日期" if zh else "Date", value=date.today())
with col_c:
    doctor_name = st.text_input("评估医生" if zh else "Clinician", value="")

col_d, col_e = st.columns(2)
with col_d:
    age_val = st.number_input(
        "年龄 (岁)" if zh else "Age (years)",
        min_value=14, max_value=55, value=24)
with col_e:
    months_post_op = st.selectbox(
        "术后评估时间点" if zh else "Assessment time point",
        ["6个月 (6 months)" if zh else "6 months post-op",
         "9个月 (9 months)" if zh else "9 months post-op",
         "12个月 (12 months)" if zh else "12 months post-op",
         "其他 / Other"],
        help="文献推荐6-12个月进行RTS评估" if zh else
             "Literature recommends RTS assessment at 6-12 months"
    )

# ── 临床评估数据 ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("临床评估数据" if zh else "Clinical Assessment")

col1, col2 = st.columns(2)

with col1:
    aclrsi_val = st.slider(
        "ACL-RSI 心理准备度 (0–100)" if zh else "ACL-RSI Score (0–100)",
        min_value=0, max_value=100, value=58,
        help="ACL重返运动心理准备量表 | 最优截点≥65 (Duchman 2019, n=681)" if zh else
             "ACL-Return to Sport after Injury Scale | Optimal cutoff ≥65 (Duchman 2019)"
    )
    hop_lsi_val = st.slider(
        "单腿跳跃LSI (%)" if zh else "Single-leg Hop LSI (%)",
        min_value=50, max_value=100, value=82,
        help="单腿跳跃距离患侧/健侧比值 | 推荐截点≥85% | 最强功能预测因子 OR=2.86 (Ithurburn 2019)" if zh else
             "Single-leg hop distance ratio injured/uninjured | Cutoff ≥85% | Strongest predictor OR=2.86 (Ithurburn 2019)"
    )

with col2:
    quad_lsi_val = st.slider(
        "股四头肌力量LSI (%)" if zh else "Quadriceps Strength LSI (%)",
        min_value=50, max_value=100, value=80,
        help="股四头肌等速肌力患侧/健侧比值 | 推荐截点≥85% (Ueda 2023)" if zh else
             "Isokinetic quadriceps strength ratio | Cutoff ≥85% (Ueda 2023)"
    )
    graft_type = st.selectbox(
        "移植物类型" if zh else "Graft Type",
        ["腘绳肌腱 (Hamstring)" if zh else "Hamstring Tendon",
         "髌腱 BTB (Patellar BTB)" if zh else "Patellar BTB",
         "股四头肌腱 (Quad Tendon)" if zh else "Quad Tendon",
         "异体移植 (Allograft)" if zh else "Allograft"],
        help="参考信息，不参与概率计算，但影响临床建议" if zh else
             "Reference only — not in probability calculation, but informs clinical advice"
    )

# ── 附加评估项（不参与计算，提供临床警示）───────────────────────────────────
st.divider()
st.subheader("附加临床评估" if zh else "Additional Clinical Flags")

col3, col4 = st.columns(2)
with col3:
    pivot_sport = st.selectbox(
        "运动类型" if zh else "Sport Type",
        ["轴转运动（足球/篮球/羽毛球）" if zh else "Pivoting sport (soccer/basketball)",
         "非轴转运动（游泳/单车）" if zh else "Non-pivoting sport (swimming/cycling)",
         "未明确 / Unknown"],
        help="轴转运动重返运动再损伤风险更高" if zh else
             "Pivoting sports carry higher re-injury risk"
    )
with col4:
    prior_aclr = st.selectbox(
        "同侧ACL重建史" if zh else "Prior ipsilateral ACLR",
        ["首次 / Primary" if zh else "Primary ACLR",
         "翻修 / Revision" if zh else "Revision ACLR"],
        help="翻修手术RTS率显著低于初次手术" if zh else
             "Revision ACLR has significantly lower RTS rates"
    )

# ── 预测计算 ──────────────────────────────────────────────────────────────────
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
        "4-6周后重新评估。" if zh else
        "Some indicators below recommended thresholds. "
        "Focus on lowest-scoring measure and reassess in 4-6 weeks."
    )
else:
    level  = "❌ 低概率重返运动" if zh else "❌ Low RTS Probability"
    color  = "red"
    advice = (
        "多项指标未达标，建议继续强化康复，暂缓重返运动决策，"
        "充分评估后再行讨论。" if zh else
        "Multiple indicators below threshold. Continue rehabilitation. "
        "Defer RTS decision pending reassessment."
    )

# ── 结果展示 ──────────────────────────────────────────────────────────────────
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

# ── 临床警示 ──────────────────────────────────────────────────────────────────
warnings_list = []
if "翻修" in prior_aclr or "Revision" in prior_aclr:
    warnings_list.append(
        "⚠️ 翻修ACLR: 重返同一运动水平的概率显著低于初次手术，建议充分告知患者预期。" if zh else
        "⚠️ Revision ACLR: RTS to preinjury level significantly lower than primary ACLR. Counsel patient accordingly."
    )
if "轴转" in pivot_sport or "Pivoting" in pivot_sport:
    warnings_list.append(
        "⚠️ 轴转运动: 再损伤风险较高（2年再损伤率约16-22%），建议完成完整RTS标准测试。" if zh else
        "⚠️ Pivoting sport: Higher re-injury risk (16-22% at 2 years). Ensure full RTS criteria are met."
    )
if "异体" in graft_type or "Allograft" in graft_type:
    warnings_list.append(
        "⚠️ 异体移植物: 年轻运动员中再撕裂率高于自体移植，建议充分告知风险。" if zh else
        "⚠️ Allograft: Higher re-tear rates in young athletes vs autograft. Counsel on risk."
    )
if aclrsi_val < 40:
    warnings_list.append(
        "⚠️ ACL-RSI极低 (<40): 心理准备度严重不足，即使功能测试达标也不建议放行，"
        "建议转介运动心理干预。" if zh else
        "⚠️ Very low ACL-RSI (<40): Severely inadequate psychological readiness. "
        "Consider sport psychology referral regardless of physical test results."
    )

for w in warnings_list:
    st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

# ── 指标汇总表 ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("评估指标汇总" if zh else "Assessment Summary")

def status(val, cutoff, reverse=False):
    ok = val >= cutoff if not reverse else val <= cutoff
    return ("✅ 达标" if ok else "⚠️ 未达标") if zh else ("✅ Met" if ok else "⚠️ Not met")

factor_df = pd.DataFrame({
    ("指标" if zh else "Measure"): [
        "ACL-RSI", "单腿跳跃LSI" if zh else "Hop LSI",
        "股四头肌LSI" if zh else "Quad LSI",
        "年龄" if zh else "Age", "移植物" if zh else "Graft",
        "运动类型" if zh else "Sport Type"
    ],
    ("当前值" if zh else "Value"): [
        f"{aclrsi_val}/100", f"{hop_lsi_val}%",
        f"{quad_lsi_val}%", f"{age_val}岁" if zh else f"{age_val}yrs",
        graft_type.split("(")[0].strip(),
        "轴转" if ("轴转" in pivot_sport) else ("Pivoting" if "Pivoting" in pivot_sport else "Non-pivoting")
    ],
    ("推荐截点" if zh else "Cutoff"): [
        "≥65 [5]", "≥85% [1]", "≥85% [2]", "<35岁更佳" if zh else "<35 better", "—", "—"
    ],
    ("状态" if zh else "Status"): [
        status(aclrsi_val, 65),
        status(hop_lsi_val, 85),
        status(quad_lsi_val, 85),
        "✅" if age_val < 35 else "⚠️", "—", "—"
    ]
})
st.dataframe(factor_df, hide_index=True, use_container_width=True)

# ── 文献注释 ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ref-box">'
    '<b>模型文献依据 / Model Evidence:</b> '
    '[1] Ithurburn et al. Am J Sports Med 2019 (Hop LSI OR=2.86, ACL-RSI OR=1.81) &nbsp;|&nbsp; '
    '[2] Ueda et al. Orthop J Sports Med 2023 (Quad LSI p=0.037, Age OR=0.80) &nbsp;|&nbsp; '
    '[3] van Haren et al. Ann Phys Rehabil Med 2023 (prospective, n=208) &nbsp;|&nbsp; '
    '[4] Xiao et al. AJSM 2023 (meta-analysis, n=3744) &nbsp;|&nbsp; '
    '[5] Duchman et al. AJSM 2019 (ACL-RSI cutoff ≥65)'
    '</div>',
    unsafe_allow_html=True
)

# ── 保存记录 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("保存评估记录" if zh else "Save Record")

if st.button("💾 保存到数据库" if zh else "💾 Save to Database", type="primary"):
    if not patient_name:
        st.warning("请先填写患者姓名" if zh else "Please enter patient name first")
    else:
        row = [
            patient_name, str(eval_date), doctor_name,
            age_val, str(months_post_op).split("(")[0].strip(),
            aclrsi_val, hop_lsi_val, quad_lsi_val,
            graft_type.split("(")[0].strip(),
            round(prob_pct, 1),
            level.replace("✅ ","").replace("⚠️ ","").replace("❌ ","")
        ]
        success, error = save_to_sheets(row)
        if success:
            st.success("✅ 已成功保存！" if zh else "✅ Successfully saved!")
        else:
            st.error(f"❌ 保存失败：{error}")

# ── 报告导出 ──────────────────────────────────────────────────────────────────
report = f"""
{'ACLR术后重返运动评估报告' if zh else 'ACLR Post-operative Return to Sport Assessment Report'}
{'=' * 60}
{'患者' if zh else 'Patient'}:       {patient_name or 'N/A'}
{'日期' if zh else 'Date'}:         {eval_date}
{'医生' if zh else 'Clinician'}:    {doctor_name or 'N/A'}
{'评估时间点' if zh else 'Timepoint'}: {months_post_op}

{'─' * 60}
{'临床评估数据' if zh else 'Clinical Data'}
{'─' * 60}
{'年龄' if zh else 'Age'}:             {age_val} {'岁' if zh else 'yrs'}
ACL-RSI:         {aclrsi_val}/100  ({'推荐≥65' if zh else 'Recommended ≥65'})
{'跳跃LSI' if zh else 'Hop LSI'}:         {hop_lsi_val}%   ({'推荐≥85%' if zh else 'Recommended ≥85%'})
{'股四头肌LSI' if zh else 'Quad LSI'}:    {quad_lsi_val}%   ({'推荐≥85%' if zh else 'Recommended ≥85%'})
{'移植物' if zh else 'Graft'}:          {graft_type}
{'运动类型' if zh else 'Sport'}:         {pivot_sport}

{'─' * 60}
{'预测结果' if zh else 'Prediction'}
{'─' * 60}
{'RTS预测概率' if zh else 'Predicted RTS Probability'}: {prob_pct:.1f}%
{'风险分层' if zh else 'Risk Level'}:        {level}
{'临床建议' if zh else 'Recommendation'}: {advice}

{'─' * 60}
{'模型说明' if zh else 'Model Information'}
{'─' * 60}
{'模型类型: 文献多因素逻辑回归（直接使用发表回归系数）' if zh else 'Model type: Literature-based multivariate logistic regression'}
{'预期AUC: ~0.80 (基于文献模型性能报告)' if zh else 'Expected AUC: ~0.80 (based on published model performance)'}

{'参考文献:' if zh else 'References:'}
[1] Ithurburn et al. Am J Sports Med 2019 (Hop LSI OR=2.86, ACL-RSI OR=1.81)
[2] Ueda et al. Orthop J Sports Med 2023 (Quad LSI, Age)
[3] van Haren et al. Ann Phys Rehabil Med 2023 (n=208, prospective)
[4] Xiao et al. AJSM 2023 (meta-analysis, n=3744)
[5] Duchman et al. AJSM 2019 (ACL-RSI cutoff ≥65)

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
    ('本工具基于文献回归系数构建，仅供临床辅助参考。正式临床使用前建议以本地患者数据进行外部验证。' if zh else
     'Built from published regression coefficients. For clinical reference only. '
     'External validation with local patient data recommended before formal clinical use.') +
    '</div>', unsafe_allow_html=True
)
