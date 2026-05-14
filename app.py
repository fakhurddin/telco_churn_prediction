"""
app.py
======
Phase 7: Streamlit Web UI for Customer Churn Predictor

Run:
    streamlit run app.py

What this app does:
  - Sidebar: 18 customer input fields
  - Main panel:
      1. Churn probability gauge (animated)
      2. Risk badge  (LOW / MEDIUM / HIGH)
      3. SHAP waterfall chart (top 5 drivers)
      4. Retention recommendations
      5. Model info footer
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

# Add src/ to path so we can import predict.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_churn, load_artifacts, RISK_COLORS, RISK_EMOJI

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Customer Churn Predictor",
    page_icon  = "🔄",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F4F6F9; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E2A3A;
    }
    [data-testid="stSidebar"] * {
        color: #E8EDF2 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #A8B8C8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2E3F52 !important;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 24px 28px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 16px;
    }

    /* Risk badges */
    .badge-low {
        background: #E1F5EE; color: #0F6E56;
        padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 1rem;
        display: inline-block; letter-spacing: 0.06em;
    }
    .badge-medium {
        background: #FAEEDA; color: #854F0B;
        padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 1rem;
        display: inline-block; letter-spacing: 0.06em;
    }
    .badge-high {
        background: #FCEBEB; color: #A32D2D;
        padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 1rem;
        display: inline-block; letter-spacing: 0.06em;
    }

    /* Section headers */
    .section-header {
        font-size: 0.78rem;
        font-weight: 700;
        color: #8A9BB0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
        margin-top: 4px;
    }

    /* Recommendation cards */
    .rec-card {
        background: #F0F7FF;
        border-left: 4px solid #378ADD;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: #1A2B3C;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #378ADD, #1D9E75);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 14px 0;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        letter-spacing: 0.04em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Probability number */
    .prob-number {
        font-size: 3.8rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: #E8EDF2;
        margin: 16px 0;
    }

    /* SHAP bar labels */
    .shap-positive { color: #E24B4A; font-weight: 600; }
    .shap-negative { color: #1D9E75; font-weight: 600; }

    /* Footer */
    .footer-text {
        font-size: 0.78rem;
        color: #9AABB8;
        text-align: center;
        margin-top: 32px;
    }
</style>
""", unsafe_allow_html=True)


# ── gauge chart ───────────────────────────────────────────────────────────────
def draw_gauge(probability: float, risk_label: str) -> plt.Figure:
    """Draw a semicircular gauge showing churn probability."""
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Gauge spans from 180° (left) to 0° (right) = π to 0
    start, end = np.pi, 0.0
    n_bg_sections = 300

    # Background arc — grey
    theta_bg = np.linspace(start, end, n_bg_sections)
    for i in range(n_bg_sections - 1):
        ax.plot([theta_bg[i], theta_bg[i+1]], [0.75, 0.75],
                color="#E8EDF2", linewidth=14, solid_capstyle="butt")

    # Color zones: green (0–30%), amber (30–60%), red (60–100%)
    zones = [
        (0.0,  0.30, "#1D9E75"),
        (0.30, 0.60, "#E8A838"),
        (0.60, 1.00, "#E24B4A"),
    ]
    for lo, hi, color in zones:
        t = np.linspace(start - (start - end) * lo,
                        start - (start - end) * hi, 60)
        for j in range(len(t) - 1):
            ax.plot([t[j], t[j+1]], [0.75, 0.75],
                    color=color, linewidth=14, solid_capstyle="butt", alpha=0.25)

    # Filled arc up to probability
    fill_end   = start - (start - end) * probability
    theta_fill = np.linspace(start, fill_end, 200)
    fill_color = RISK_COLORS[risk_label]
    for i in range(len(theta_fill) - 1):
        ax.plot([theta_fill[i], theta_fill[i+1]], [0.75, 0.75],
                color=fill_color, linewidth=14, solid_capstyle="butt")

    # Needle
    needle_angle = start - (start - end) * probability
    ax.annotate("",
        xy=(needle_angle, 0.68),
        xytext=(needle_angle, 0.05),
        arrowprops=dict(
            arrowstyle="->,head_width=0.04,head_length=0.04",
            color="#1E2A3A", lw=2.5,
        ),
    )
    # Needle hub
    ax.plot(needle_angle, 0.05, "o", color="#1E2A3A", markersize=7, zorder=5)

    # Zone labels
    for angle_pct, label in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
        angle = start - (start - end) * angle_pct
        ax.text(angle, 0.96, label,
                ha="center", va="center", fontsize=8,
                color="#8A9BB0", fontweight="500")

    ax.set_ylim(0, 1.1)
    ax.set_theta_zero_location("W")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ── SHAP waterfall chart ──────────────────────────────────────────────────────
def draw_shap_chart(shap_reasons: list) -> plt.Figure:
    """Horizontal bar chart showing SHAP contributions."""
    if not shap_reasons:
        return None

    features = [r[0] for r in shap_reasons]
    values   = [r[1] for r in shap_reasons]

    # Clean feature names for display
    def clean_name(name):
        replacements = {
            "_No internet service": " (no internet)",
            "_No phone service"   : " (no phone)",
            "InternetService_"    : "Internet: ",
            "Contract_"           : "Contract: ",
            "PaymentMethod_"      : "Payment: ",
            "_Yes"                : " ✓",
            "_No"                 : " ✗",
            "_Male"               : " (Male)",
            "_Female"             : " (Female)",
        }
        n = name
        for old, new in replacements.items():
            n = n.replace(old, new)
        return n

    labels = [clean_name(f) for f in features]
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.65)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = range(len(features) - 1, -1, -1)
    bars  = ax.barh(list(y_pos), values, color=colors,
                    edgecolor="white", height=0.55)

    for bar, val in zip(bars, values[::-1]):
        x_pos = val + (0.002 if val >= 0 else -0.002)
        ha    = "left" if val >= 0 else "right"
        label = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        color = "#E24B4A" if val > 0 else "#1D9E75"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha,
                fontsize=9, fontweight="bold", color=color)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="#8A9BB0", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value  (positive = toward churn)", fontsize=9, color="#8A9BB0")
    ax.tick_params(axis="x", colors="#8A9BB0", labelsize=8)
    ax.tick_params(axis="y", colors="#1E2A3A")

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#E8EDF2")
    ax.grid(axis="x", color="#F0F3F7", linewidth=0.6)

    # Legend
    pos_patch = mpatches.Patch(color="#E24B4A", label="Increases churn risk")
    neg_patch = mpatches.Patch(color="#1D9E75", label="Decreases churn risk")
    ax.legend(handles=[pos_patch, neg_patch],
              fontsize=8, loc="lower right",
              framealpha=0.8, edgecolor="#E8EDF2")

    fig.tight_layout(pad=1.2)
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🔄 Churn Predictor")
        st.markdown("*Fill in customer details and click Predict.*")
        st.markdown("---")

        # ── Account Info ──────────────────────────────────────────────────────
        st.markdown("### 📋 Account Info")

        tenure = st.slider(
            "Tenure (months)", min_value=0, max_value=72, value=12,
            help="How long has the customer been with the company?"
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=18.0, max_value=120.0,
            value=65.0, step=0.50,
            help="Current monthly bill amount"
        )
        total_charges = round(tenure * monthly_charges, 2)
        st.caption(f"💡 Estimated Total Charges: **${total_charges:,.2f}**")

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            help="Longer contracts = lower churn risk"
        )

        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )

        paperless = st.checkbox("Paperless Billing", value=True)

        st.markdown("---")

        # ── Internet & Services ───────────────────────────────────────────────
        st.markdown("### 🌐 Internet & Services")

        internet_service = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"],
        )

        if internet_service != "No":
            col1, col2 = st.columns(2)
            with col1:
                online_security  = st.radio("Online Security",  ["No", "Yes"], horizontal=True)
                online_backup    = st.radio("Online Backup",    ["No", "Yes"], horizontal=True)
                device_protect   = st.radio("Device Protection",["No", "Yes"], horizontal=True)
            with col2:
                tech_support     = st.radio("Tech Support",     ["No", "Yes"], horizontal=True)
                streaming_tv     = st.radio("Streaming TV",     ["No", "Yes"], horizontal=True)
                streaming_movies = st.radio("Streaming Movies", ["No", "Yes"], horizontal=True)
        else:
            online_security = online_backup = device_protect = "No internet service"
            tech_support    = streaming_tv  = streaming_movies = "No internet service"

        st.markdown("---")

        # ── Phone ─────────────────────────────────────────────────────────────
        st.markdown("### 📱 Phone")

        phone_service = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
        if phone_service == "Yes":
            multiple_lines = st.radio("Multiple Lines", ["No", "Yes"], horizontal=True)
        else:
            multiple_lines = "No phone service"

        st.markdown("---")

        # ── Demographics ──────────────────────────────────────────────────────
        st.markdown("### 👤 Demographics")

        col1, col2 = st.columns(2)
        with col1:
            gender        = st.radio("Gender",     ["Male", "Female"], horizontal=True)
            senior        = st.checkbox("Senior Citizen", value=False)
        with col2:
            partner       = st.radio("Partner",    ["No", "Yes"], horizontal=True)
            dependents    = st.radio("Dependents", ["No", "Yes"], horizontal=True)

        st.markdown("---")

        # ── Predict button ────────────────────────────────────────────────────
        predict_clicked = st.button("🔮  Predict Churn Risk", use_container_width=True)

    customer = {
        "tenure"          : tenure,
        "MonthlyCharges"  : monthly_charges,
        "TotalCharges"    : total_charges,
        "SeniorCitizen"   : 1 if senior else 0,
        "gender"          : gender,
        "Partner"         : partner,
        "Dependents"      : dependents,
        "PhoneService"    : phone_service,
        "MultipleLines"   : multiple_lines,
        "InternetService" : internet_service,
        "OnlineSecurity"  : online_security,
        "OnlineBackup"    : online_backup,
        "DeviceProtection": device_protect,
        "TechSupport"     : tech_support,
        "StreamingTV"     : streaming_tv,
        "StreamingMovies" : streaming_movies,
        "Contract"        : contract,
        "PaperlessBilling": "Yes" if paperless else "No",
        "PaymentMethod"   : payment_method,
    }

    return customer, predict_clicked


# ── main panel ────────────────────────────────────────────────────────────────
def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🔄 Customer Churn Predictor")
        st.markdown(
            "Predict the likelihood a customer will cancel their subscription "
            "— with explainable AI showing exactly *why*."
        )
    with col2:
        try:
            artifacts = load_artifacts()
            model_name = type(artifacts["model"]).__name__
            feat_count = len(artifacts["feature_cols"])
            st.metric("Model", model_name)
            st.metric("Features", feat_count)
        except Exception:
            st.warning("Model not loaded yet")


def render_welcome():
    st.markdown("---")
    st.markdown("""
    <div class="metric-card">
        <h3 style="margin:0 0 8px 0; color:#1E2A3A;">👋 How to use this app</h3>
        <ol style="color:#4A5568; margin:0; padding-left:20px; line-height:2;">
            <li>Fill in the customer's details in the <strong>left sidebar</strong></li>
            <li>Click <strong>🔮 Predict Churn Risk</strong></li>
            <li>View the churn probability, risk badge, and top reasons</li>
            <li>Use the retention recommendations to take action</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 What the model looks at")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📋 Account**\nTenure, charges, contract type, payment method, paperless billing")
    with col2:
        st.info("**🌐 Services**\nInternet type, security, backup, tech support, streaming")
    with col3:
        st.info("**👤 Demographics**\nGender, senior citizen, partner, dependents")


def render_results(result: dict, customer: dict):
    st.markdown("---")

    # ── Row 1: Gauge + Risk Badge + Quick Stats ───────────────────────────────
    col_gauge, col_risk, col_stats = st.columns([1.8, 1.5, 1.7])

    with col_gauge:
        st.markdown('<p class="section-header">Churn Probability</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card" style="text-align:center; padding: 20px;">'
            f'<p class="prob-number" style="color:{result["risk_color"]}">'
            f'{result["probability_pct"]}</p>'
            f'<p style="color:#8A9BB0; font-size:0.85rem; margin:4px 0 12px 0;">'
            f'probability of churning</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        gauge_fig = draw_gauge(result["probability"], result["risk_label"])
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close()

    with col_risk:
        st.markdown('<p class="section-header">Risk Level</p>',
                    unsafe_allow_html=True)
        badge_class = f"badge-{result['risk_label'].lower()}"
        risk_messages = {
            "LOW"   : ("Customer is stable.", "No immediate action needed — continue standard engagement."),
            "MEDIUM": ("Monitor closely.", "Consider proactive outreach within 30 days."),
            "HIGH"  : ("Immediate action needed!", "High risk of cancellation — escalate to retention team now."),
        }
        title, body = risk_messages[result["risk_label"]]
        st.markdown(
            f'<div class="metric-card">'
            f'<span class="{badge_class}">'
            f'{result["risk_emoji"]} {result["risk_label"]} RISK'
            f'</span>'
            f'<p style="color:#1E2A3A; font-weight:600; margin:14px 0 4px 0">{title}</p>'
            f'<p style="color:#6B7A8D; font-size:0.88rem; margin:0">{body}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Probability breakdown
        retain_pct = 100 - result["probability"] * 100
        churn_pct  = result["probability"] * 100
        st.markdown(
            f'<div class="metric-card" style="padding:16px 20px">'
            f'<p class="section-header" style="margin-bottom:10px">Probability breakdown</p>'
            f'<div style="display:flex; justify-content:space-between; margin-bottom:6px">'
            f'<span style="color:#1D9E75; font-weight:600">Will Stay</span>'
            f'<span style="color:#1D9E75; font-weight:700">{retain_pct:.1f}%</span></div>'
            f'<div style="background:#E8EDF2; border-radius:6px; height:8px; margin-bottom:12px">'
            f'<div style="background:#1D9E75; width:{retain_pct:.1f}%; height:8px; border-radius:6px"></div></div>'
            f'<div style="display:flex; justify-content:space-between; margin-bottom:6px">'
            f'<span style="color:{result["risk_color"]}; font-weight:600">Will Churn</span>'
            f'<span style="color:{result["risk_color"]}; font-weight:700">{churn_pct:.1f}%</span></div>'
            f'<div style="background:#E8EDF2; border-radius:6px; height:8px">'
            f'<div style="background:{result["risk_color"]}; width:{churn_pct:.1f}%; height:8px; border-radius:6px"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_stats:
        st.markdown('<p class="section-header">Customer Profile</p>',
                    unsafe_allow_html=True)
        profile_items = [
            ("📅 Tenure",          f"{customer['tenure']} months"),
            ("💵 Monthly Bill",    f"${customer['MonthlyCharges']:.2f}"),
            ("💰 Total Spent",     f"${customer['TotalCharges']:,.2f}"),
            ("📄 Contract",        customer["Contract"]),
            ("🌐 Internet",        customer["InternetService"]),
            ("💳 Payment",         customer["PaymentMethod"]),
            ("🛡️ Tech Support",    customer["TechSupport"]),
            ("🔒 Online Security", customer["OnlineSecurity"]),
        ]
        rows_html = "".join(
            f'<div style="display:flex; justify-content:space-between; '
            f'padding:5px 0; border-bottom:1px solid #F0F3F7;">'
            f'<span style="color:#8A9BB0; font-size:0.85rem">{k}</span>'
            f'<span style="color:#1E2A3A; font-weight:600; font-size:0.85rem">{v}</span>'
            f'</div>'
            for k, v in profile_items
        )
        st.markdown(
            f'<div class="metric-card" style="padding:16px 20px">{rows_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Row 2: SHAP + Recommendations ─────────────────────────────────────────
    col_shap, col_recs = st.columns([1.6, 1.4])

    with col_shap:
        st.markdown('<p class="section-header">Top Churn Drivers (SHAP)</p>',
                    unsafe_allow_html=True)
        if result["shap_reasons"]:
            shap_fig = draw_shap_chart(result["shap_reasons"])
            if shap_fig:
                st.markdown('<div class="metric-card" style="padding:16px 20px">',
                            unsafe_allow_html=True)
                st.pyplot(shap_fig, use_container_width=True)
                plt.close()
                st.markdown(
                    '<p style="color:#8A9BB0; font-size:0.78rem; margin:8px 0 0 0">'
                    '🔴 Red bars push toward churn &nbsp;|&nbsp; '
                    '🟢 Green bars reduce churn risk</p>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("SHAP explanations not available. Ensure `models/shap_explainer.pkl` exists.")

    with col_recs:
        st.markdown('<p class="section-header">Retention Recommendations</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#6B7A8D; font-size:0.85rem; margin-bottom:12px">'
            'Actions to reduce this customer\'s churn risk:</p>',
            unsafe_allow_html=True,
        )
        for rec in result["recommendations"]:
            st.markdown(
                f'<div class="rec-card">{rec}</div>',
                unsafe_allow_html=True,
            )

        # Risk-based urgency banner
        if result["risk_label"] == "HIGH":
            st.error("🚨 **Escalate immediately** — assign to senior retention specialist.")
        elif result["risk_label"] == "MEDIUM":
            st.warning("⚠️ **Schedule outreach** — contact within the next 30 days.")
        else:
            st.success("✅ **Low priority** — include in standard quarterly check-in.")


def render_footer(result: dict):
    st.markdown("---")
    st.markdown(
        f'<p class="footer-text">'
        f'Model: <strong>{result["model_name"]}</strong> &nbsp;|&nbsp; '
        f'Features: <strong>30 encoded</strong> &nbsp;|&nbsp; '
        f'Dataset: <strong>Telco Customer Churn (IBM)</strong> &nbsp;|&nbsp; '
        f'Built with Python · scikit-learn · XGBoost · SHAP · Streamlit'
        f'</p>',
        unsafe_allow_html=True,
    )


# ── error state ───────────────────────────────────────────────────────────────
def render_model_missing():
    st.error("### ⚠️ Model files not found")
    st.markdown("""
    The trained model artifacts are missing. Run the pipeline first:

    ```bash
    python src/data_cleaning.py
    python src/eda.py
    python src/feature_engineering.py
    python src/train_model.py
    ```

    Then relaunch:
    ```bash
    streamlit run app.py
    ```
    """)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    # Check model exists before rendering anything
    try:
        load_artifacts()
        model_ok = True
    except FileNotFoundError:
        model_ok = False

    if not model_ok:
        render_model_missing()
        return

    render_header()
    customer, predict_clicked = render_sidebar()

    if not predict_clicked:
        render_welcome()
    else:
        with st.spinner("Running prediction..."):
            try:
                result = predict_churn(customer)
                render_results(result, customer)
                render_footer(result)

                # Store in session so result survives re-runs
                st.session_state["last_result"]   = result
                st.session_state["last_customer"] = customer

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
