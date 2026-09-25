import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy_financial as npf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zero Trust Business Value | MSc Research Artefact",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MDX_LOGO = os.path.join(
    BASE_DIR,
    "assets",
    "mdx_logo.png"
)

RF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_npv_model.pkl"
)

GB_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gradient_boosting_npv_model.pkl"
)


# ============================================================
# RESEARCH CONSTANTS
# ============================================================

RANDOM_SEED = 42
N_SIMULATIONS = 5000
ANALYSIS_HORIZON = 5

MATURITY_LEVELS = {
    1: "Traditional",
    2: "Initial",
    3: "Advanced",
    4: "Optimal"
}


PILLARS = {

    "Identity": [
        "Multi-Factor Authentication",
        "Identity Lifecycle Management",
        "Conditional Access",
        "Privileged Access Management"
    ],

    "Devices": [
        "Device Inventory",
        "Device Compliance",
        "Endpoint Protection",
        "Device Health Monitoring"
    ],

    "Networks": [
        "Network Segmentation",
        "Encrypted Communications",
        "Network Monitoring",
        "Software-Defined Access"
    ],

    "Applications & Workloads": [
        "Application Access Control",
        "Workload Identity",
        "Application Security Monitoring",
        "Secure Application Integration"
    ],

    "Data": [
        "Data Classification",
        "Data Encryption",
        "Data Loss Prevention",
        "Data Access Governance"
    ]
}


ML_FEATURES = [
    "Baseline_SLE",
    "Baseline_ARO",
    "Frequency_Reduction",
    "Severity_Reduction",
    "Initial_Investment",
    "Annual_OM",
    "Discount_Rate"
]


# ============================================================
# SESSION STATE
# ============================================================

if "assessment_results" not in st.session_state:

    st.session_state.assessment_results = {
        "organisation": {},
        "maturity": {},
        "risk": {},
        "financial": {},
        "monte_carlo": {},
        "machine_learning": {},
        "decision_intelligence": {}
    }


DEFAULTS = {

    "org_name":
        "Sample Organisation",

    "industry":
        "Financial Services",

    "employees":
        500,

    "baseline_sle":
        4_440_000.0,

    "baseline_aro":
        0.10,

    "frequency_reduction":
        0.20,

    "severity_reduction":
        0.20,

    "initial_investment":
        185_000.0,

    "annual_operating_cost":
        30_000.0,

    "annual_maintenance_cost":
        15_000.0,

    "discount_rate":
        0.08
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


for pillar, capabilities in PILLARS.items():

    for capability in capabilities:

        current_key = (
            f"current_{pillar}_{capability}"
        )

        target_key = (
            f"target_{pillar}_{capability}"
        )

        if current_key not in st.session_state:
            st.session_state[current_key] = 2

        if target_key not in st.session_state:
            st.session_state[target_key] = 3


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

def currency(value):

    if value is None:
        return "N/A"

    try:

        if not np.isfinite(value):
            return "N/A"

    except Exception:
        return "N/A"

    if abs(value) >= 1_000_000:

        return (
            f"${value / 1_000_000:,.2f}M"
        )

    if abs(value) >= 1_000:

        return (
            f"${value / 1_000:,.1f}K"
        )

    return f"${value:,.0f}"


def percent(value):

    if value is None:
        return "N/A"

    try:

        if not np.isfinite(value):
            return "N/A"

    except Exception:
        return "N/A"

    return f"{value * 100:.1f}%"


# ============================================================
# MATURITY CALCULATION
# ============================================================

def calculate_maturity():

    rows = []

    for pillar, capabilities in PILLARS.items():

        for capability in capabilities:

            current = st.session_state[
                f"current_{pillar}_{capability}"
            ]

            target = st.session_state[
                f"target_{pillar}_{capability}"
            ]

            rows.append(
                {
                    "Pillar": pillar,
                    "Capability": capability,
                    "Current": current,
                    "Target": target,
                    "Gap": target - current
                }
            )


    df = pd.DataFrame(rows)


    pillar_summary = (
        df
        .groupby("Pillar")[
            ["Current", "Target"]
        ]
        .mean()
        .reset_index()
    )


    current_maturity = (
        df["Current"].mean()
    )

    target_maturity = (
        df["Target"].mean()
    )

    maturity_gap = (
        target_maturity -
        current_maturity
    )


    results = {

        "Current_Maturity":
            current_maturity,

        "Target_Maturity":
            target_maturity,

        "Maturity_Gap":
            maturity_gap,

        "Capability_Data":
            df,

        "Pillar_Summary":
            pillar_summary
    }


    st.session_state.assessment_results[
        "maturity"
    ] = results


    return results


# ============================================================
# CYBER RISK CALCULATION
# ============================================================

def calculate_risk():

    sle = float(
        st.session_state.baseline_sle
    )

    aro = float(
        st.session_state.baseline_aro
    )

    frequency_reduction = float(
        st.session_state.frequency_reduction
    )

    severity_reduction = float(
        st.session_state.severity_reduction
    )


    baseline_ale = (
        sle * aro
    )


    post_sle = (
        sle *
        (1 - severity_reduction)
    )


    post_aro = (
        aro *
        (1 - frequency_reduction)
    )


    post_ale = (
        post_sle *
        post_aro
    )


    avoided_loss = (
        baseline_ale -
        post_ale
    )


    ale_reduction = (

        avoided_loss /
        baseline_ale

        if baseline_ale > 0

        else 0
    )


    results = {

        "Baseline_SLE":
            sle,

        "Baseline_ARO":
            aro,

        "Frequency_Reduction":
            frequency_reduction,

        "Severity_Reduction":
            severity_reduction,

        "Baseline_ALE":
            baseline_ale,

        "Post_ZT_ALE":
            post_ale,

        "Avoided_Annual_Loss":
            avoided_loss,

        "ALE_Reduction":
            ale_reduction
    }


    st.session_state.assessment_results[
        "risk"
    ] = results


    return results


# ============================================================
# FINANCIAL CALCULATION
# ============================================================

def calculate_financial():

    risk = calculate_risk()


    investment = float(
        st.session_state.initial_investment
    )


    operating = float(
        st.session_state.annual_operating_cost
    )


    maintenance = float(
        st.session_state.annual_maintenance_cost
    )


    annual_om = (
        operating +
        maintenance
    )


    discount = float(
        st.session_state.discount_rate
    )


    avoided_loss = (
        risk["Avoided_Annual_Loss"]
    )


    net_annual_benefit = (
        avoided_loss -
        annual_om
    )


    simple_roi = (

        net_annual_benefit /
        investment

        if investment > 0

        else np.nan
    )


    payback = (

        investment /
        net_annual_benefit

        if net_annual_benefit > 0

        else np.inf
    )


    cash_flows = [
        -investment
    ]


    for _ in range(
        ANALYSIS_HORIZON
    ):

        cash_flows.append(
            net_annual_benefit
        )


    npv = sum(

        cash_flows[t] /
        ((1 + discount) ** t)

        for t in range(
            len(cash_flows)
        )
    )


    try:

        irr = npf.irr(
            cash_flows
        )

    except Exception:

        irr = np.nan


    results = {

        "Initial_Investment":
            investment,

        "Annual_Operating_Cost":
            operating,

        "Annual_Maintenance_Cost":
            maintenance,

        "Annual_OM":
            annual_om,

        "Discount_Rate":
            discount,

        "Avoided_Annual_Loss":
            avoided_loss,

        "Net_Annual_Benefit":
            net_annual_benefit,

        "Simple_ROI":
            simple_roi,

        "Payback_Years":
            payback,

        "NPV":
            npv,

        "IRR":
            irr,

        "Cash_Flows":
            cash_flows
    }


    st.session_state.assessment_results[
        "financial"
    ] = results


    return results


# ============================================================
# MONTE CARLO ENGINE
# ============================================================

@st.cache_data
def run_monte_carlo():

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    sle = rng.triangular(
        2_000_000,
        4_440_000,
        7_000_000,
        N_SIMULATIONS
    )


    aro = rng.triangular(
        0.03,
        0.10,
        0.30,
        N_SIMULATIONS
    )


    frequency_reduction = (
        rng.triangular(
            0.05,
            0.20,
            0.40,
            N_SIMULATIONS
        )
    )


    severity_reduction = (
        rng.triangular(
            0.05,
            0.20,
            0.40,
            N_SIMULATIONS
        )
    )


    investment = rng.triangular(
        140_000,
        185_000,
        280_000,
        N_SIMULATIONS
    )


    annual_om = rng.triangular(
        30_000,
        45_000,
        70_000,
        N_SIMULATIONS
    )


    discount_rate = rng.triangular(
        0.04,
        0.08,
        0.15,
        N_SIMULATIONS
    )


    baseline_ale = (
        sle * aro
    )


    post_ale = (

        sle *
        (1 - severity_reduction) *
        aro *
        (1 - frequency_reduction)
    )


    avoided_loss = (
        baseline_ale -
        post_ale
    )


    net_annual_benefit = (
        avoided_loss -
        annual_om
    )


    roi = (
        net_annual_benefit /
        investment
    )


    payback = np.where(

        net_annual_benefit > 0,

        investment /
        net_annual_benefit,

        np.inf
    )


    npv_values = (
        -investment.copy()
    )


    for year in range(
        1,
        ANALYSIS_HORIZON + 1
    ):

        npv_values += (

            net_annual_benefit /

            (
                (1 + discount_rate)
                ** year
            )
        )


    df = pd.DataFrame(
        {

            "Baseline_SLE":
                sle,

            "Baseline_ARO":
                aro,

            "Frequency_Reduction":
                frequency_reduction,

            "Severity_Reduction":
                severity_reduction,

            "Initial_Investment":
                investment,

            "Annual_OM":
                annual_om,

            "Discount_Rate":
                discount_rate,

            "Baseline_ALE":
                baseline_ale,

            "Post_ZT_ALE":
                post_ale,

            "Avoided_Annual_Loss":
                avoided_loss,

            "Net_Annual_Benefit":
                net_annual_benefit,

            "ROI":
                roi,

            "Payback_Years":
                payback,

            "NPV":
                npv_values
        }
    )


    df["Positive_NPV"] = (
        df["NPV"] > 0
    )


    return df


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_ml_models():

    rf_model = None
    gb_model = None


    if os.path.exists(
        RF_MODEL_PATH
    ):

        try:

            rf_model = joblib.load(
                RF_MODEL_PATH
            )

        except Exception:

            rf_model = None


    if os.path.exists(
        GB_MODEL_PATH
    ):

        try:

            gb_model = joblib.load(
                GB_MODEL_PATH
            )

        except Exception:

            gb_model = None


    return (
        rf_model,
        gb_model
    )


# ============================================================
# CSS DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 92% 5%,
            rgba(37, 99, 235, 0.06),
            transparent 23%
        ),
        #F6F8FC;

    color: #172033;
}


.block-container {

    max-width: 1500px;

    padding-top: 2rem;

    padding-left: 2.6rem;

    padding-right: 2.6rem;

    padding-bottom: 4rem;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {

    background: #FFFFFF !important;

    border-right:
        1px solid #E3E8F0;

    box-shadow:
        6px 0 25px
        rgba(15, 23, 42, 0.025);
}


.project-name {

    color: #172033 !important;

    font-size: 18px;

    font-weight: 850;

    margin-top: 8px;
}


.project-type {

    color: #7C879A !important;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 1.2px;

    margin-top: 4px;

    margin-bottom: 18px;
}


/* HERO */

.hero {

    position: relative;

    overflow: hidden;

    padding: 48px;

    border-radius: 26px;

    background:
        linear-gradient(
            120deg,
            #FFFFFF 0%,
            #FAFBFF 60%,
            #EDF3FF 100%
        );

    border:
        1px solid #DFE6F0;

    box-shadow:
        0 16px 50px
        rgba(15, 23, 42, 0.055);

    margin-bottom: 30px;
}


.hero::after {

    content: "";

    position: absolute;

    width: 350px;

    height: 350px;

    border-radius: 50%;

    right: -130px;

    top: -180px;

    background:
        linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.13),
            rgba(79, 70, 229, 0.04)
        );
}


.hero-label {

    position: relative;

    z-index: 2;

    color: #2563EB !important;

    font-size: 10px;

    font-weight: 850;

    letter-spacing: 1.6px;
}


.hero-title {

    position: relative;

    z-index: 2;

    color: #111827 !important;

    font-size: 44px;

    font-weight: 900;

    line-height: 1.08;

    letter-spacing: -1.6px;

    margin-top: 15px;

    max-width: 930px;
}


.hero-text {

    position: relative;

    z-index: 2;

    color: #64748B !important;

    font-size: 15px;

    line-height: 1.75;

    margin-top: 17px;

    max-width: 850px;
}


/* SECTION HEADERS */

.section-label {

    color: #2563EB !important;

    font-size: 9px;

    font-weight: 850;

    letter-spacing: 1.5px;

    margin-top: 15px;
}


.section-title {

    color: #172033 !important;

    font-size: 27px;

    font-weight: 850;

    margin-top: 5px;

    margin-bottom: 5px;
}


.section-text {

    color: #718096 !important;

    font-size: 13px;

    line-height: 1.6;

    margin-bottom: 20px;
}


/* RESEARCH CARDS */

.research-card {

    background: #FFFFFF !important;

    border:
        1px solid #E2E7EF;

    border-radius: 18px;

    padding: 23px;

    min-height: 175px;

    box-shadow:
        0 7px 24px
        rgba(15, 23, 42, 0.04);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


.research-card:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 14px 35px
        rgba(15, 23, 42, 0.08);
}


.card-kicker {

    color: #2563EB !important;

    font-size: 9px;

    font-weight: 850;

    letter-spacing: 1.2px;
}


.card-title {

    color: #172033 !important;

    font-size: 18px;

    font-weight: 850;

    margin-top: 13px;
}


.card-text {

    color: #697586 !important;

    font-size: 12px;

    line-height: 1.65;

    margin-top: 9px;
}


/* DARK DATA SCIENCE PANEL */

.dark-panel {

    background:
        linear-gradient(
            125deg,
            #101B35,
            #172554
        );

    border-radius: 20px;

    padding: 30px;

    margin-top: 12px;

    margin-bottom: 20px;

    box-shadow:
        0 15px 40px
        rgba(15, 23, 42, 0.13);
}


.dark-kicker {

    color: #93C5FD !important;

    font-size: 9px;

    font-weight: 850;

    letter-spacing: 1.4px;
}


.dark-title {

    color: #FFFFFF !important;

    font-size: 24px;

    font-weight: 850;

    margin-top: 8px;
}


.dark-text {

    color: #CBD5E1 !important;

    font-size: 13px;

    line-height: 1.7;

    margin-top: 8px;

    max-width: 1000px;
}


/* METHOD CARDS */

.method-card {

    background: #FFFFFF !important;

    border:
        1px solid #E2E7EF;

    border-radius: 16px;

    padding: 18px;

    min-height: 120px;

    box-shadow:
        0 5px 20px
        rgba(15, 23, 42, 0.03);
}


.method-label {

    color: #64748B !important;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 1px;
}


.method-value {

    color: #111827 !important;

    font-size: 24px;

    font-weight: 900;

    margin-top: 8px;
}


.method-note {

    color: #94A3B8 !important;

    font-size: 10px;

    margin-top: 4px;
}


/* STREAMLIT METRICS */

div[data-testid="stMetric"] {

    background: #FFFFFF;

    border:
        1px solid #E2E7EF;

    border-radius: 16px;

    padding: 18px;

    box-shadow:
        0 6px 20px
        rgba(15, 23, 42, 0.035);
}


/* EXPANDERS */

div[data-testid="stExpander"] {

    background: #FFFFFF;

    border:
        1px solid #E2E7EF;

    border-radius: 14px;

    overflow: hidden;
}


/* DATAFRAMES */

div[data-testid="stDataFrame"] {

    border:
        1px solid #E2E7EF;

    border-radius: 14px;

    overflow: hidden;
}


/* BUTTONS */

.stButton > button,
.stDownloadButton > button {

    min-height: 44px;

    border-radius: 11px;

    font-weight: 750;
}


h1,
h2,
h3 {

    color: #172033 !important;
}


hr {

    border: none;

    height: 1px;

    background: #E5EAF1;
}


#MainMenu {
    visibility: hidden;
}


footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if os.path.exists(
        MDX_LOGO
    ):

        st.image(
            MDX_LOGO,
            width=210
        )

    else:

        st.warning(
            "MDX logo not found in assets/mdx_logo.png"
        )


    sidebar_html = """
<div class="project-name">Zero Trust Business Value</div>
<div class="project-type">MSc DATA SCIENCE & AI RESEARCH ARTEFACT</div>
"""

    st.markdown(
        sidebar_html,
        unsafe_allow_html=True
    )


    page = st.radio(

        "Research Navigation",

        [
            "01  Research Overview",
            "02  Data & Assumptions",
            "03  Zero Trust Assessment",
            "04  Cyber Risk Analytics",
            "05  Financial Analytics",
            "06  Monte Carlo Simulation",
            "07  Machine Learning Lab",
            "08  Decision Intelligence",
            "09  Executive Report",
            "10  Methodology & Limitations"
        ],

        label_visibility="collapsed"
    )


    st.divider()


    st.caption(
        "MSc Data Science & Artificial Intelligence"
    )


    st.caption(
        "Middlesex University Dubai"
    )


    st.caption(
        "Python • Streamlit • Scikit-learn • Plotly"
    )


# ============================================================
# PAGE HEADER
# ============================================================

def page_header(
    label,
    title,
    description
):

    html = f"""
<div class="section-label">{label}</div>
<div class="section-title">{title}</div>
<div class="section-text">{description}</div>
"""

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 01 — RESEARCH OVERVIEW
# ============================================================

if page == "01  Research Overview":

    hero_html = """
<div class="hero">
<div class="hero-label">MSc DATA SCIENCE & ARTIFICIAL INTELLIGENCE • RESEARCH ARTEFACT</div>
<div class="hero-title">AI-Enabled Zero Trust Business Value<br>Decision Support System</div>
<div class="hero-text">
An interactive research artefact integrating cybersecurity maturity assessment,
quantitative cyber-risk modelling, financial analytics, Monte Carlo simulation
and supervised machine learning to evaluate Zero Trust investment under uncertainty.
</div>
</div>
"""

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )


    page_header(
        "RESEARCH ARCHITECTURE",
        "A Multi-Layer Analytical Framework",
        """
        The artefact translates cybersecurity inputs into
        quantitative, financial, probabilistic and predictive
        evidence through six connected analytical components.
        """
    )


    cards = [

        (
            "01 • CYBERSECURITY",
            "🛡️ Zero Trust Maturity",
            """
            Assess current and target maturity across five
            Zero Trust pillars and twenty security capabilities.
            """
        ),

        (
            "02 • QUANTITATIVE RISK",
            "📊 Cyber Risk Analytics",
            """
            Quantify expected financial cyber exposure using
            SLE, ARO and Annual Loss Expectancy.
            """
        ),

        (
            "03 • FINANCIAL ANALYTICS",
            "💰 Business Value",
            """
            Evaluate avoided cyber losses, ROI, NPV,
            IRR and investment payback.
            """
        )
    ]


    cols = st.columns(3)


    for col, card in zip(
        cols,
        cards
    ):

        with col:

            card_html = f"""
<div class="research-card">
<div class="card-kicker">{card[0]}</div>
<div class="card-title">{card[1]}</div>
<div class="card-text">{card[2]}</div>
</div>
"""

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )


    st.write("")


    cards = [

        (
            "04 • DATA SCIENCE",
            "🎲 Monte Carlo Simulation",
            """
            Explore parameter uncertainty through 5,000
            synthetic probabilistic simulation records.
            """
        ),

        (
            "05 • ARTIFICIAL INTELLIGENCE",
            "🤖 Machine Learning",
            """
            Apply Random Forest and Gradient Boosting
            regression to five-year NPV prediction.
            """
        ),

        (
            "06 • DECISION SCIENCE",
            "🧠 Decision Intelligence",
            """
            Integrate deterministic, probabilistic and
            predictive evidence into decision support.
            """
        )
    ]


    cols = st.columns(3)


    for col, card in zip(
        cols,
        cards
    ):

        with col:

            card_html = f"""
<div class="research-card">
<div class="card-kicker">{card[0]}</div>
<div class="card-title">{card[1]}</div>
<div class="card-text">{card[2]}</div>
</div>
"""

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )


    st.write("")


    page_header(
        "DATA SCIENCE & AI",
        "Analytical Implementation",
        """
        The research design combines deterministic modelling,
        stochastic simulation and supervised machine learning.
        """
    )


    ds_html = """
<div class="dark-panel">
<div class="dark-kicker">DATA SCIENCE & ARTIFICIAL INTELLIGENCE ENGINE</div>
<div class="dark-title">Three Complementary Analytical Layers</div>
<div class="dark-text">
Rather than relying on a single ROI calculation, the artefact evaluates
Zero Trust using deterministic financial modelling, Monte Carlo uncertainty
analysis and supervised machine learning. Their outputs are integrated into
a decision-intelligence layer for research and executive interpretation.
</div>
</div>
"""

    st.markdown(
        ds_html,
        unsafe_allow_html=True
    )


    metrics = [

        (
            "5,000",
            "SIMULATIONS",
            "Synthetic observations"
        ),

        (
            "7",
            "ML FEATURES",
            "Raw predictors"
        ),

        (
            "2",
            "ML MODELS",
            "Ensemble regressors"
        ),

        (
            "70 / 30",
            "TRAIN / TEST",
            "Hold-out evaluation"
        ),

        (
            "5 Years",
            "TARGET HORIZON",
            "NPV prediction"
        )
    ]


    metric_cols = st.columns(5)


    for col, item in zip(
        metric_cols,
        metrics
    ):

        with col:

            metric_html = f"""
<div class="method-card">
<div class="method-label">{item[1]}</div>
<div class="method-value">{item[0]}</div>
<div class="method-note">{item[2]}</div>
</div>
"""

            st.markdown(
                metric_html,
                unsafe_allow_html=True
            )


    st.write("")


    st.info(
        """
        **Research interpretation:** Zero Trust maturity is
        represented using an ordinal 1–4 assessment scale.
        The maturity score is not interpreted as breach
        probability or percentage security. Cyber-risk
        reduction is modelled separately through changes in
        incident frequency and loss severity.
        """
    )


# ============================================================
# 02 — DATA & ASSUMPTIONS
# ============================================================

elif page == "02  Data & Assumptions":

    page_header(
        "RESEARCH DATA",
        "Data, Variables & Model Assumptions",
        """
        This module documents the variables, distributions,
        data-generation process and assumptions supporting
        the analytical framework.
        """
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Simulation Records",
        "5,000"
    )


    c2.metric(
        "ML Predictors",
        "7"
    )


    c3.metric(
        "Random Seed",
        "42"
    )


    c4.metric(
        "Analysis Horizon",
        "5 Years"
    )


    st.subheader(
        "Machine Learning Feature Schema"
    )


    feature_table = pd.DataFrame(
        {

            "Feature": [
                "Baseline_SLE",
                "Baseline_ARO",
                "Frequency_Reduction",
                "Severity_Reduction",
                "Initial_Investment",
                "Annual_OM",
                "Discount_Rate"
            ],

            "Description": [
                "Baseline Single Loss Expectancy",
                "Baseline Annual Rate of Occurrence",
                "Expected reduction in incident frequency",
                "Expected reduction in loss severity",
                "Initial Zero Trust investment",
                "Annual operating and maintenance cost",
                "Financial discount rate"
            ],

            "ML Role": [
                "Predictor",
                "Predictor",
                "Predictor",
                "Predictor",
                "Predictor",
                "Predictor",
                "Predictor"
            ]
        }
    )


    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Monte Carlo Probability Distributions"
    )


    distribution_table = pd.DataFrame(
        {

            "Variable": [
                "Baseline SLE",
                "Baseline ARO",
                "Frequency Reduction",
                "Severity Reduction",
                "Initial Investment",
                "Annual O&M",
                "Discount Rate"
            ],

            "Distribution": [
                "Triangular",
                "Triangular",
                "Triangular",
                "Triangular",
                "Triangular",
                "Triangular",
                "Triangular"
            ],

            "Minimum": [
                "$2.00M",
                "0.03",
                "5%",
                "5%",
                "$140K",
                "$30K",
                "4%"
            ],

            "Most Likely": [
                "$4.44M",
                "0.10",
                "20%",
                "20%",
                "$185K",
                "$45K",
                "8%"
            ],

            "Maximum": [
                "$7.00M",
                "0.30",
                "40%",
                "40%",
                "$280K",
                "$70K",
                "15%"
            ]
        }
    )


    st.dataframe(
        distribution_table,
        use_container_width=True,
        hide_index=True
    )


    st.warning(
        """
        **Data provenance:** The 5,000 Monte Carlo observations
        are synthetically generated scenarios. They do not
        represent 5,000 observed organisations. Machine-learning
        performance therefore describes predictive performance
        within this simulated analytical environment.
        """
    )


# ============================================================
# 03 — ZERO TRUST ASSESSMENT
# ============================================================

elif page == "03  Zero Trust Assessment":

    page_header(
        "CYBERSECURITY ANALYTICS",
        "Zero Trust Maturity Assessment",
        """
        Assess current and target organisational maturity
        across five Zero Trust pillars and twenty capabilities.
        """
    )


    st.subheader(
        "Organisation Profile"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.text_input(
            "Organisation",
            key="org_name"
        )


    with c2:

        st.selectbox(
            "Industry",
            [
                "Financial Services",
                "Technology",
                "Healthcare",
                "Retail",
                "Government",
                "Manufacturing",
                "Education",
                "Other"
            ],
            key="industry"
        )


    with c3:

        st.number_input(
            "Number of Employees",
            min_value=1,
            step=50,
            key="employees"
        )


    st.session_state.assessment_results[
        "organisation"
    ] = {

        "Organisation":
            st.session_state.org_name,

        "Industry":
            st.session_state.industry,

        "Employees":
            st.session_state.employees
    }


    st.divider()


    st.subheader(
        "Five-Pillar Maturity Assessment"
    )


    st.caption(
        """
        Level 1 = Traditional • Level 2 = Initial •
        Level 3 = Advanced • Level 4 = Optimal
        """
    )


    for pillar, capabilities in PILLARS.items():

        with st.expander(
            f"🛡️ {pillar}",
            expanded=False
        ):

            for capability in capabilities:

                st.markdown(
                    f"**{capability}**"
                )


                left, right = st.columns(2)


                with left:

                    st.select_slider(

                        "Current maturity",

                        options=[
                            1,
                            2,
                            3,
                            4
                        ],

                        format_func=lambda x:
                            (
                                f"{x} — "
                                f"{MATURITY_LEVELS[x]}"
                            ),

                        key=
                        f"current_{pillar}_{capability}"
                    )


                with right:

                    st.select_slider(

                        "Target maturity",

                        options=[
                            1,
                            2,
                            3,
                            4
                        ],

                        format_func=lambda x:
                            (
                                f"{x} — "
                                f"{MATURITY_LEVELS[x]}"
                            ),

                        key=
                        f"target_{pillar}_{capability}"
                    )


                st.divider()


    maturity = calculate_maturity()


    st.subheader(
        "Maturity Analytics"
    )


    m1, m2, m3 = st.columns(3)


    m1.metric(
        "Current Maturity",
        (
            f"{maturity['Current_Maturity']:.2f} / 4"
        )
    )


    m2.metric(
        "Target Maturity",
        (
            f"{maturity['Target_Maturity']:.2f} / 4"
        )
    )


    m3.metric(
        "Maturity Gap",
        (
            f"{maturity['Maturity_Gap']:.2f}"
        )
    )


    pillar_df = (
        maturity[
            "Pillar_Summary"
        ]
    )


    maturity_chart = go.Figure()


    maturity_chart.add_trace(

        go.Bar(
            name="Current",
            x=pillar_df["Pillar"],
            y=pillar_df["Current"]
        )
    )


    maturity_chart.add_trace(

        go.Bar(
            name="Target",
            x=pillar_df["Pillar"],
            y=pillar_df["Target"]
        )
    )


    maturity_chart.update_layout(

        title=
        "Current vs Target Maturity by Zero Trust Pillar",

        barmode="group",

        yaxis=dict(
            range=[0, 4],
            title="Maturity Level"
        ),

        template="plotly_white",

        height=450
    )


    st.plotly_chart(
        maturity_chart,
        use_container_width=True
    )


    st.dataframe(
        maturity[
            "Capability_Data"
        ],
        use_container_width=True,
        hide_index=True
    )


    st.info(
        """
        Maturity values are ordinal assessment scores.
        They are intentionally not converted directly into
        risk-reduction percentages.
        """
    )


# ============================================================
# 04 — CYBER RISK ANALYTICS
# ============================================================

elif page == "04  Cyber Risk Analytics":

    page_header(
        "QUANTITATIVE CYBER RISK",
        "Cyber Risk Analytics",
        """
        Estimate baseline and post-Zero Trust annual
        financial exposure using Single Loss Expectancy,
        Annual Rate of Occurrence and Annual Loss Expectancy.
        """
    )


    left, right = st.columns(2)


    with left:

        st.number_input(

            "Baseline Single Loss Expectancy ($)",

            min_value=0.0,

            step=10_000.0,

            key="baseline_sle"
        )


        st.number_input(

            "Baseline Annual Rate of Occurrence",

            min_value=0.0,

            max_value=1.0,

            step=0.01,

            format="%.2f",

            key="baseline_aro"
        )


    with right:

        st.slider(

            "Expected Incident Frequency Reduction",

            min_value=0.0,

            max_value=0.80,

            step=0.01,

            format="%.0f%%",

            key="frequency_reduction"
        )


        st.slider(

            "Expected Loss Severity Reduction",

            min_value=0.0,

            max_value=0.80,

            step=0.01,

            format="%.0f%%",

            key="severity_reduction"
        )


    risk = calculate_risk()


    r1, r2, r3, r4 = st.columns(4)


    r1.metric(
        "Baseline ALE",
        currency(
            risk["Baseline_ALE"]
        )
    )


    r2.metric(
        "Post-ZT ALE",
        currency(
            risk["Post_ZT_ALE"]
        )
    )


    r3.metric(
        "Avoided Annual Loss",
        currency(
            risk["Avoided_Annual_Loss"]
        )
    )


    r4.metric(
        "ALE Reduction",
        percent(
            risk["ALE_Reduction"]
        )
    )


    risk_df = pd.DataFrame(
        {

            "Risk State": [
                "Baseline ALE",
                "Post-Zero Trust ALE"
            ],

            "Annual Loss": [
                risk["Baseline_ALE"],
                risk["Post_ZT_ALE"]
            ]
        }
    )


    fig = px.bar(

        risk_df,

        x="Risk State",

        y="Annual Loss",

        title=
        "Estimated Annual Cyber Loss Exposure",

        text_auto=".3s"
    )


    fig.update_layout(
        template="plotly_white",
        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Risk Calculation"
    )


    st.latex(
        r"ALE = SLE \times ARO"
    )


    st.latex(
        r"""
        Post\ ALE =
        SLE(1-S_r)
        \times
        ARO(1-F_r)
        """
    )


    st.info(
        """
        Frequency and severity reductions combine
        multiplicatively. A 20% frequency reduction and
        20% severity reduction therefore produce a 36%
        reduction in ALE rather than 40%.
        """
    )


# ============================================================
# 05 — FINANCIAL ANALYTICS
# ============================================================

elif page == "05  Financial Analytics":

    page_header(
        "FINANCIAL MODELLING",
        "Five-Year Business Value Analytics",
        """
        Translate estimated avoided cyber losses into
        financial investment metrics including ROI,
        NPV, IRR and payback.
        """
    )


    left, right = st.columns(2)


    with left:

        st.number_input(

            "Initial Zero Trust Investment ($)",

            min_value=0.0,

            step=5_000.0,

            key="initial_investment"
        )


        st.number_input(

            "Annual Operating Cost ($)",

            min_value=0.0,

            step=1_000.0,

            key="annual_operating_cost"
        )


    with right:

        st.number_input(

            "Annual Maintenance Cost ($)",

            min_value=0.0,

            step=1_000.0,

            key="annual_maintenance_cost"
        )


        st.slider(

            "Discount Rate",

            min_value=0.0,

            max_value=0.30,

            step=0.01,

            format="%.0f%%",

            key="discount_rate"
        )


    finance = calculate_financial()


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "5-Year NPV",
        currency(
            finance["NPV"]
        )
    )


    c2.metric(
        "Simple ROI",
        percent(
            finance["Simple_ROI"]
        )
    )


    c3.metric(
        "IRR",
        percent(
            finance["IRR"]
        )
    )


    if np.isfinite(
        finance[
            "Payback_Years"
        ]
    ):

        payback_text = (
            f"{finance['Payback_Years']:.2f} years"
        )

    else:

        payback_text = (
            "Not achieved"
        )


    c4.metric(
        "Payback Period",
        payback_text
    )


    st.subheader(
        "Financial Value Components"
    )


    f1, f2, f3 = st.columns(3)


    f1.metric(
        "Avoided Annual Loss",
        currency(
            finance[
                "Avoided_Annual_Loss"
            ]
        )
    )


    f2.metric(
        "Annual O&M",
        currency(
            finance[
                "Annual_OM"
            ]
        )
    )


    f3.metric(
        "Net Annual Benefit",
        currency(
            finance[
                "Net_Annual_Benefit"
            ]
        )
    )


    years = list(
        range(
            ANALYSIS_HORIZON + 1
        )
    )


    cash_df = pd.DataFrame(
        {

            "Year":
                years,

            "Cash Flow":
                finance[
                    "Cash_Flows"
                ]
        }
    )


    cash_df[
        "Cumulative Cash Flow"
    ] = (
        cash_df[
            "Cash Flow"
        ].cumsum()
    )


    fig = go.Figure()


    fig.add_trace(

        go.Bar(

            x=cash_df[
                "Year"
            ],

            y=cash_df[
                "Cash Flow"
            ],

            name=
            "Annual Cash Flow"
        )
    )


    fig.add_trace(

        go.Scatter(

            x=cash_df[
                "Year"
            ],

            y=cash_df[
                "Cumulative Cash Flow"
            ],

            mode=
            "lines+markers",

            name=
            "Cumulative Cash Flow"
        )
    )


    fig.update_layout(

        title=
        "Five-Year Investment Cash Flow",

        template=
        "plotly_white",

        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 06 — MONTE CARLO SIMULATION
# ============================================================

elif page == "06  Monte Carlo Simulation":

    page_header(
        "DATA SCIENCE • STOCHASTIC MODELLING",
        "Monte Carlo Simulation Laboratory",
        """
        Explore uncertainty in cyber-risk and financial
        assumptions through 5,000 synthetic probabilistic
        simulation scenarios.
        """
    )


    mc_df = run_monte_carlo()


    positive_probability = (
        mc_df[
            "Positive_NPV"
        ].mean()
    )


    median_npv = (
        mc_df[
            "NPV"
        ].median()
    )


    mean_npv = (
        mc_df[
            "NPV"
        ].mean()
    )


    p05 = (
        mc_df[
            "NPV"
        ].quantile(
            0.05
        )
    )


    p95 = (
        mc_df[
            "NPV"
        ].quantile(
            0.95
        )
    )


    st.session_state.assessment_results[
        "monte_carlo"
    ] = {

        "Positive_NPV_Probability":
            positive_probability,

        "Median_NPV":
            median_npv,

        "Mean_NPV":
            mean_npv,

        "P05_NPV":
            p05,

        "P95_NPV":
            p95
    }


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Simulations",
        f"{len(mc_df):,}"
    )


    c2.metric(
        "Positive NPV",
        percent(
            positive_probability
        )
    )


    c3.metric(
        "Median NPV",
        currency(
            median_npv
        )
    )


    c4.metric(
        "Mean NPV",
        currency(
            mean_npv
        )
    )


    st.subheader(
        "NPV Probability Distribution"
    )


    fig = px.histogram(

        mc_df,

        x="NPV",

        nbins=60,

        title=
        "Distribution of Simulated Five-Year NPV"
    )


    fig.add_vline(

        x=0,

        line_dash="dash",

        annotation_text=
        "NPV = 0"
    )


    fig.update_layout(
        template="plotly_white",
        height=470
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    p1, p2, p3 = st.columns(3)


    p1.metric(
        "5th Percentile",
        currency(p05)
    )


    p2.metric(
        "Median",
        currency(
            median_npv
        )
    )


    p3.metric(
        "95th Percentile",
        currency(p95)
    )


    st.subheader(
        "Risk vs Value"
    )


    scatter_sample = (
        mc_df.sample(
            min(
                1500,
                len(mc_df)
            ),
            random_state=42
        )
    )


    fig = px.scatter(

        scatter_sample,

        x="Baseline_ALE",

        y="NPV",

        title=
        "Baseline Cyber Risk vs Five-Year NPV",

        opacity=0.55
    )


    fig.update_layout(
        template="plotly_white",
        height=450
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Simulation Dataset Preview"
    )


    st.dataframe(
        mc_df.head(100),
        use_container_width=True,
        hide_index=True
    )


    csv_data = (
        mc_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(

        "⬇️ Download Simulation Dataset",

        data=
        csv_data,

        file_name=
        "zero_trust_monte_carlo_dataset.csv",

        mime=
        "text/csv"
    )


    st.warning(
        """
        The 5,000 records are synthetic Monte Carlo
        observations generated from specified probability
        distributions. They are not observations from
        5,000 real organisations.
        """
    )


# ============================================================
# 07 — MACHINE LEARNING LAB
# ============================================================

elif page == "07  Machine Learning Lab":

    page_header(
        "ARTIFICIAL INTELLIGENCE • SUPERVISED LEARNING",
        "Machine Learning Laboratory",
        """
        Use ensemble regression models to predict five-year
        NPV from seven raw cyber-risk and financial variables.
        """
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Synthetic Dataset",
        "5,000"
    )


    c2.metric(
        "Input Features",
        "7"
    )


    c3.metric(
        "Train / Test",
        "70 / 30"
    )


    c4.metric(
        "Prediction Target",
        "5-Year NPV"
    )


    st.subheader(
        "Machine Learning Architecture"
    )


    ml_table = pd.DataFrame(
        {

            "Component": [
                "Training Dataset",
                "Predictors",
                "Target",
                "Train/Test Split",
                "Model 1",
                "Model 2"
            ],

            "Implementation": [
                "Monte Carlo synthetic observations",
                "7 raw input variables",
                "Five-year NPV",
                "70% / 30%",
                "Random Forest Regressor",
                "Gradient Boosting Regressor"
            ]
        }
    )


    st.dataframe(
        ml_table,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Predictor Variables"
    )


    predictors = pd.DataFrame(
        {

            "Feature":
                ML_FEATURES,

            "Type": [
                "Cyber Risk",
                "Cyber Risk",
                "Control Effect",
                "Control Effect",
                "Financial",
                "Financial",
                "Financial"
            ]
        }
    )


    st.dataframe(
        predictors,
        use_container_width=True,
        hide_index=True
    )


    rf_model, gb_model = (
        load_ml_models()
    )


    if (
        rf_model is None
        and
        gb_model is None
    ):

        st.warning(
            """
            **ML models are not connected yet.**

            Put the trained model files inside the
            `models` folder using these names:

            `random_forest_npv_model.pkl`

            `gradient_boosting_npv_model.pkl`

            Once added, this page will automatically
            activate live NPV prediction and feature
            importance.
            """
        )

    else:

        risk = calculate_risk()

        finance = (
            calculate_financial()
        )


        prediction_input = (
            pd.DataFrame(
                {

                    "Baseline_SLE": [
                        risk[
                            "Baseline_SLE"
                        ]
                    ],

                    "Baseline_ARO": [
                        risk[
                            "Baseline_ARO"
                        ]
                    ],

                    "Frequency_Reduction": [
                        risk[
                            "Frequency_Reduction"
                        ]
                    ],

                    "Severity_Reduction": [
                        risk[
                            "Severity_Reduction"
                        ]
                    ],

                    "Initial_Investment": [
                        finance[
                            "Initial_Investment"
                        ]
                    ],

                    "Annual_OM": [
                        finance[
                            "Annual_OM"
                        ]
                    ],

                    "Discount_Rate": [
                        finance[
                            "Discount_Rate"
                        ]
                    ]
                }
            )
        )


        st.subheader(
            "Live AI Prediction"
        )


        predictions = {}


        p1, p2, p3 = st.columns(3)


        if rf_model is not None:

            rf_prediction = float(

                rf_model.predict(
                    prediction_input[
                        ML_FEATURES
                    ]
                )[0]
            )


            predictions[
                "Random Forest"
            ] = rf_prediction


            p1.metric(
                "Random Forest NPV",
                currency(
                    rf_prediction
                )
            )


        if gb_model is not None:

            gb_prediction = float(

                gb_model.predict(
                    prediction_input[
                        ML_FEATURES
                    ]
                )[0]
            )


            predictions[
                "Gradient Boosting"
            ] = gb_prediction


            p2.metric(
                "Gradient Boosting NPV",
                currency(
                    gb_prediction
                )
            )


        p3.metric(
            "Deterministic NPV",
            currency(
                finance["NPV"]
            )
        )


        st.session_state.assessment_results[
            "machine_learning"
        ] = predictions


        comparison_methods = (
            list(
                predictions.keys()
            )
            +
            ["Deterministic"]
        )


        comparison_values = (
            list(
                predictions.values()
            )
            +
            [
                finance[
                    "NPV"
                ]
            ]
        )


        comparison_df = (
            pd.DataFrame(
                {

                    "Method":
                        comparison_methods,

                    "NPV":
                        comparison_values
                }
            )
        )


        fig = px.bar(

            comparison_df,

            x="Method",

            y="NPV",

            title=
            "NPV Prediction Comparison",

            text_auto=".3s"
        )


        fig.update_layout(
            template="plotly_white",
            height=430
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        model_for_importance = (

            gb_model

            if gb_model is not None

            else rf_model
        )


        if hasattr(
            model_for_importance,
            "feature_importances_"
        ):

            importance_df = (
                pd.DataFrame(
                    {

                        "Feature":
                            ML_FEATURES,

                        "Importance":
                            model_for_importance
                            .feature_importances_
                    }
                )
                .sort_values(
                    "Importance",
                    ascending=True
                )
            )


            fig = px.bar(

                importance_df,

                x="Importance",

                y="Feature",

                orientation="h",

                title=
                "Machine Learning Feature Importance"
            )


            fig.update_layout(
                template="plotly_white",
                height=450
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        st.info(
            """
            Feature importance describes relative predictive
            contribution within the trained model. It does not
            establish causal relationships between the
            predictors and business value.
            """
        )


# ============================================================
# 08 — DECISION INTELLIGENCE
# ============================================================

elif page == "08  Decision Intelligence":

    page_header(
        "INTEGRATED ANALYTICS",
        "Decision Intelligence Dashboard",
        """
        Integrate cybersecurity maturity, quantitative risk,
        deterministic finance, uncertainty analysis and
        predictive evidence within one decision-support view.
        """
    )


    maturity = (
        calculate_maturity()
    )

    risk = (
        calculate_risk()
    )

    finance = (
        calculate_financial()
    )

    mc_df = (
        run_monte_carlo()
    )


    positive_probability = (
        mc_df[
            "Positive_NPV"
        ].mean()
    )


    median_npv = (
        mc_df[
            "NPV"
        ].median()
    )


    d1, d2, d3, d4 = (
        st.columns(4)
    )


    d1.metric(
        "Current Maturity",
        (
            f"{maturity['Current_Maturity']:.2f} / 4"
        )
    )


    d2.metric(
        "ALE Reduction",
        percent(
            risk[
                "ALE_Reduction"
            ]
        )
    )


    d3.metric(
        "Deterministic NPV",
        currency(
            finance[
                "NPV"
            ]
        )
    )


    d4.metric(
        "Positive NPV Probability",
        percent(
            positive_probability
        )
    )


    decision_html = """
<div class="dark-panel">
<div class="dark-kicker">DECISION INTELLIGENCE LAYER</div>
<div class="dark-title">Integrated Analytical Evidence</div>
<div class="dark-text">
This layer combines cybersecurity maturity, deterministic financial modelling,
probabilistic uncertainty analysis and machine-learning evidence. The system
supports managerial interpretation and does not replace human decision-making.
</div>
</div>
"""

    st.markdown(
        decision_html,
        unsafe_allow_html=True
    )


    summary_df = pd.DataFrame(
        {

            "Analytical Layer": [
                "Maturity Assessment",
                "Cyber Risk",
                "Financial Model",
                "Monte Carlo"
            ],

            "Primary Result": [

                (
                    f"{maturity['Current_Maturity']:.2f}"
                    f" → "
                    f"{maturity['Target_Maturity']:.2f}"
                ),

                percent(
                    risk[
                        "ALE_Reduction"
                    ]
                ),

                currency(
                    finance[
                        "NPV"
                    ]
                ),

                percent(
                    positive_probability
                )
            ],

            "Interpretation": [

                "Current to target Zero Trust maturity",

                "Estimated annual loss exposure reduction",

                "Deterministic five-year NPV",

                "Simulated scenarios producing positive NPV"
            ]
        }
    )


    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Uncertainty Range"
    )


    u1, u2, u3 = st.columns(3)


    u1.metric(
        "Downside — P05",
        currency(
            mc_df[
                "NPV"
            ].quantile(
                0.05
            )
        )
    )


    u2.metric(
        "Median",
        currency(
            median_npv
        )
    )


    u3.metric(
        "Upside — P95",
        currency(
            mc_df[
                "NPV"
            ].quantile(
                0.95
            )
        )
    )


# ============================================================
# 09 — EXECUTIVE REPORT
# ============================================================

elif page == "09  Executive Report":

    page_header(
        "RESEARCH OUTPUT",
        "Executive Assessment Report",
        """
        Consolidate the major analytical findings into
        an executive-level interpretation of Zero Trust
        business value.
        """
    )


    maturity = (
        calculate_maturity()
    )

    risk = (
        calculate_risk()
    )

    finance = (
        calculate_financial()
    )

    mc_df = (
        run_monte_carlo()
    )


    positive_probability = (
        mc_df[
            "Positive_NPV"
        ].mean()
    )


    median_npv = (
        mc_df[
            "NPV"
        ].median()
    )


    st.subheader(
        st.session_state.org_name
    )


    st.caption(
        f"""
        {st.session_state.industry} •
        {st.session_state.employees:,} employees
        """
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Current → Target Maturity",
        (
            f"{maturity['Current_Maturity']:.2f}"
            f" → "
            f"{maturity['Target_Maturity']:.2f}"
        )
    )


    c2.metric(
        "Avoided Annual Loss",
        currency(
            risk[
                "Avoided_Annual_Loss"
            ]
        )
    )


    c3.metric(
        "5-Year NPV",
        currency(
            finance[
                "NPV"
            ]
        )
    )


    c4, c5, c6 = st.columns(3)


    c4.metric(
        "Simple ROI",
        percent(
            finance[
                "Simple_ROI"
            ]
        )
    )


    c5.metric(
        "Positive NPV Probability",
        percent(
            positive_probability
        )
    )


    c6.metric(
        "Median Simulated NPV",
        currency(
            median_npv
        )
    )


    st.subheader(
        "Executive Interpretation"
    )


    st.write(
        f"""
        The decision-support system evaluates the potential
        business value of Zero Trust for
        **{st.session_state.org_name}**.

        The organisation's current Zero Trust maturity is
        **{maturity['Current_Maturity']:.2f}/4**, compared
        with a target maturity of
        **{maturity['Target_Maturity']:.2f}/4**.

        Under the selected quantitative cyber-risk assumptions,
        estimated annual loss exposure decreases from
        **{currency(risk['Baseline_ALE'])}** to
        **{currency(risk['Post_ZT_ALE'])}**, representing an
        estimated ALE reduction of
        **{percent(risk['ALE_Reduction'])}**.

        The deterministic five-year financial model produces
        an NPV of **{currency(finance['NPV'])}** and a simple
        ROI of **{percent(finance['Simple_ROI'])}**.

        Across **5,000 synthetic Monte Carlo simulations**,
        **{percent(positive_probability)}** of scenarios
        produce a positive five-year NPV. The median simulated
        NPV is **{currency(median_npv)}**.
        """
    )


    st.info(
        """
        The report presents analytical decision support rather
        than a guaranteed financial outcome. Results remain
        dependent on the selected cyber-risk, financial and
        probability-distribution assumptions.
        """
    )


# ============================================================
# 10 — METHODOLOGY & LIMITATIONS
# ============================================================

elif page == "10  Methodology & Limitations":

    page_header(
        "ACADEMIC DOCUMENTATION",
        "Methodology, Reproducibility & Limitations",
        """
        Document the analytical methods, model boundaries,
        reproducibility controls and interpretation limitations
        of the research artefact.
        """
    )


    st.subheader(
        "Analytical Methodology"
    )


    methodology = pd.DataFrame(
        {

            "Research Layer": [
                "Zero Trust Maturity",
                "Cyber Risk",
                "Financial Analytics",
                "Monte Carlo",
                "Machine Learning",
                "Decision Intelligence"
            ],

            "Method": [
                "Ordinal 1–4 assessment",
                "SLE × ARO = ALE",
                "ROI, NPV, IRR and Payback",
                "Triangular probability distributions",
                "Random Forest + Gradient Boosting regression",
                "Integrated evidence synthesis"
            ],

            "Purpose": [
                "Assess cybersecurity posture",
                "Quantify expected annual financial exposure",
                "Estimate business value",
                "Represent parameter uncertainty",
                "Predict five-year NPV",
                "Support managerial interpretation"
            ]
        }
    )


    st.dataframe(
        methodology,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Machine Learning Research Design"
    )


    ml_methodology = pd.DataFrame(
        {

            "Component": [
                "Dataset",
                "Observations",
                "Input Features",
                "Prediction Target",
                "Train/Test Split",
                "Model 1",
                "Model 2",
                "Random Seed"
            ],

            "Implementation": [
                "Synthetic Monte Carlo dataset",
                "5,000",
                "7 raw predictor variables",
                "Five-year NPV",
                "70% / 30%",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "42"
            ]
        }
    )


    st.dataframe(
        ml_methodology,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Target Leakage Control"
    )


    st.success(
        """
        The machine-learning feature set contains only seven
        raw input variables: Baseline SLE, Baseline ARO,
        Frequency Reduction, Severity Reduction, Initial
        Investment, Annual O&M and Discount Rate.

        Derived financial variables such as ALE, avoided loss,
        net annual benefit and ROI are excluded from the model
        inputs to reduce direct target leakage.
        """
    )


    st.subheader(
        "Research Limitations"
    )


    st.markdown(
        """
        - The Monte Carlo dataset is synthetic and does not
          represent observations from 5,000 real organisations.

        - Simulation results depend on the selected probability
          distributions and parameter ranges.

        - Zero Trust maturity is measured using an ordinal
          research assessment scale and is not a direct measure
          of breach probability.

        - Frequency and severity reductions are independent
          modelling assumptions and are not calculated directly
          from maturity scores.

        - Machine-learning models learn relationships within
          the synthetic simulation environment.

        - Strong machine-learning performance within synthetic
          data should not be interpreted as independent
          real-world validation.

        - Feature importance represents predictive contribution
          within the fitted model and does not demonstrate
          causality.

        - Financial results are decision-support estimates and
          should not be interpreted as guaranteed investment
          outcomes.
        """
    )


    st.subheader(
        "Reproducibility Controls"
    )


    r1, r2, r3, r4 = (
        st.columns(4)
    )


    r1.metric(
        "Random Seed",
        "42"
    )


    r2.metric(
        "Simulations",
        "5,000"
    )


    r3.metric(
        "Train / Test",
        "70 / 30"
    )


    r4.metric(
        "Financial Horizon",
        "5 Years"
    )


    st.info(
        """
        A consistent five-year horizon is maintained across
        deterministic financial modelling, Monte Carlo
        simulation and machine-learning NPV prediction to
        support direct comparison between analytical layers.
        """
    )