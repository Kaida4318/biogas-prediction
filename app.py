import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Biogas Production Predictor",
    page_icon="🌱",
    layout="wide"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0a1628; color: #e8f5e9; }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #112240;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #64ffda;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1d4ed8 !important;
        color: white !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #112240, #1a3a5c);
        border: 1px solid #1d4ed8;
        border-radius: 12px;
        padding: 20px;
    }
    [data-testid="stMetricLabel"] { color: #64ffda !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 30px !important; font-weight: 800 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 700;
        padding: 16px;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(29, 78, 216, 0.5);
    }

    .hero {
        background: linear-gradient(135deg, #0f2744 0%, #1d4ed8 50%, #0a1628 100%);
        border-radius: 20px;
        padding: 50px 40px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid #1d4ed8;
    }
    .hero-title { font-size: 52px; font-weight: 900; color: #ffffff; margin: 0; letter-spacing: -2px; }
    .hero-sub { font-size: 18px; color: #64ffda; margin-top: 10px; }

    .stat-card {
        background: linear-gradient(135deg, #112240, #0f2744);
        border: 1px solid #1d4ed8;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 6px 0;
    }
    .stat-number { font-size: 38px; font-weight: 900; color: #64ffda; }
    .stat-label { font-size: 12px; color: #8892b0; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

    .info-card {
        background-color: #112240;
        border: 1px solid #1d4ed8;
        border-radius: 14px;
        padding: 24px;
        margin: 10px 0;
        height: 100%;
    }
    .info-card h4 { color: #64ffda; margin-top: 0; }
    .info-card p, .info-card li { color: #ccd6f6; }

    .result-box {
        background: linear-gradient(135deg, #0f3460, #1d4ed8);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid #64ffda;
    }
    .result-number { font-size: 64px; font-weight: 900; color: #64ffda; }
    .result-label { font-size: 18px; color: #ccd6f6; }

    label { color: #ccd6f6 !important; font-weight: 500 !important; }
    h1, h2, h3 { color: #ccd6f6 !important; }
    hr { border-color: #1d4ed8; opacity: 0.3; }

    .stSuccess { background-color: #0d3321 !important; border: 1px solid #64ffda !important; border-radius: 12px !important; }
    .stInfo { background-color: #112240 !important; border: 1px solid #1d4ed8 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Train Model on Startup
# ============================================================
@st.cache_resource
def train_model():
    df = pd.read_csv('biogas_dataset.csv')
    df.columns = [
        'year', 'month', 'day',
        'pig_manure', 'kitchen_waste', 'chicken_litter',
        'cassava', 'bagasse', 'energy_grass',
        'banana_shafts', 'alcohol_waste', 'municipal_residue',
        'fish_waste', 'water', 'diesel',
        'electricity_use', 'temperature', 'humidity',
        'rainfall', 'cn_ratio', 'digester_temp',
        'biogas_production'
    ]

    features = [
        'pig_manure', 'kitchen_waste', 'chicken_litter',
        'cassava', 'bagasse', 'municipal_residue',
        'temperature', 'digester_temp'
    ]

    X = df[features]
    y = df['biogas_production']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    model.fit(X, y)
    return model, features

model, feature_names = train_model()

# ============================================================
# Model Results
# ============================================================
model_results = pd.DataFrame({
    'Model':    ['Linear Regression', 'Ridge', 'SVR', 'Lasso',
                 'Gradient Boosting', 'XGBoost', 'Random Forest'],
    'R² Score': [0.9561, 0.9561, 0.9547, 0.9543,
                 0.9388, 0.9313, 0.9141],
    'RMSE':     [2.3322, 2.3322, 2.3683, 2.3779,
                 2.7536, 2.9155, 3.2610],
    'MAE':      [1.8617, 1.8617, 1.8806, 1.9035,
                 2.1920, 2.3268, 2.5666]
})

# ============================================================
# Hero Banner
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🌱 Biogas Production Predictor</div>
    <div class="hero-sub">
        Machine Learning powered daily biogas estimation · Linear Regression · R² = 0.9561
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  About",
    "🔮  Predict",
    "📊  Model Performance",
    "📈  EDA Insights"
])


# ============================================================
# TAB 1 — ABOUT
# ============================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">15,298</div>
            <div class="stat-label">Daily Records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">0.9561</div>
            <div class="stat-label">R² Score</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">7</div>
            <div class="stat-label">Models Tested</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">8</div>
            <div class="stat-label">Key Features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📌 What is Biogas?</h4>
            <p>Biogas is a renewable energy source produced when organic 
            waste decomposes in a sealed tank called an <b>anaerobic digester</b> 
            — without oxygen. It can be used for:</p>
            <ul>
                <li>🔥 Cooking fuel</li>
                <li>💡 Generating electricity</li>
                <li>🏠 Heating buildings</li>
                <li>♻️ Replacing fossil fuels</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Project Objective</h4>
            <p>Develop a machine learning model to <b>predict daily biogas 
            production (m³/day)</b> based on feedstock inputs and environmental 
            conditions from a small-scale biogas plant in India.</p>
            <h4>📦 Dataset</h4>
            <p><b>Source:</b> Indian Biogas Production Dataset (Kaggle)<br>
            <b>Records:</b> 15,298 daily measurements (2010–2024)<br>
            <b>License:</b> CC0 Public Domain</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-card">
        <h4>🗺️ Project Workflow</h4>
        <p>
        <b>1. Data Collection</b> → Found and evaluated 8+ open source datasets<br>
        <b>2. Data Cleaning</b> → Handled missing values, removed leakage columns<br>
        <b>3. EDA</b> → Explored distributions, correlations, and outliers<br>
        <b>4. Feature Selection</b> → Selected 8 most impactful features<br>
        <b>5. Model Training</b> → Trained and compared 7 regression models<br>
        <b>6. Evaluation</b> → Learning curves, cross-validation, train/test analysis<br>
        <b>7. Deployment</b> → Deployed as interactive web app via Streamlit Cloud
        </p>
    </div>""", unsafe_allow_html=True)


# ============================================================
# TAB 2 — PREDICT
# ============================================================
with tab2:
    st.markdown("### Enter daily inputs to estimate biogas output")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌾 Feedstock Inputs (kg/day)")
        pig_manure     = st.number_input("🐷 Pig Manure (kg)",         0.0, 100.0, 25.0, help="Daily pig manure input")
        kitchen_waste  = st.number_input("🍽️ Kitchen Food Waste (kg)", 0.0,  50.0, 18.0, help="Daily kitchen waste")
        chicken_litter = st.number_input("🐔 Chicken Litter (kg)",     0.0,  40.0, 12.0, help="Daily chicken waste")
        cassava        = st.number_input("🌿 Cassava (kg)",            0.0,  60.0, 20.0, help="Cassava plant material")

    with col2:
        st.markdown("#### ⚙️ More Inputs & Conditions")
        bagasse           = st.number_input("🌾 Bagasse Feed (kg)",       0.0, 50.0, 15.0, help="Sugarcane bagasse")
        municipal_residue = st.number_input("🏙️ Municipal Residue (kg)",  0.0, 25.0, 10.0, help="Organic municipal waste")
        temperature       = st.slider("🌤️ Ambient Temperature (°C)", 15.0, 45.0, 30.0)
        digester_temp     = st.slider("🌡️ Digester Temperature (°C)", 25.0, 45.0, 36.0)

    st.markdown("---")

    if st.button("🔮  PREDICT BIOGAS PRODUCTION"):
        input_data = pd.DataFrame([[
            pig_manure, kitchen_waste, chicken_litter, cassava,
            bagasse, municipal_residue, temperature, digester_temp
        ]], columns=feature_names)

        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)
        energy_kwh = prediction * 2.0

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Daily Biogas Production</div>
            <div class="result-number">{prediction:.2f} m³</div>
            <div class="result-label">per day</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("📅 Daily",   f"{prediction:.2f} m³")
        with c2:
            st.metric("📆 Monthly", f"{prediction * 30:.1f} m³")
        with c3:
            st.metric("🗓️ Yearly",  f"{prediction * 365:.1f} m³")

        st.info(f"💡 Generates ~**{energy_kwh:.1f} kWh** electricity/day — enough for ~**{energy_kwh/10:.0f} households**")


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================
with tab3:
    st.markdown("### Seven regression models trained and compared")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Results Table")
        st.dataframe(
            model_results.style.highlight_max(
                subset=['R² Score'], color='#0d3321'
            ).highlight_min(
                subset=['RMSE', 'MAE'], color='#0d3321'
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        <div class="info-card">
            <h4>🧠 Why Linear Regression Won</h4>
            <ul>
                <li><b>Linear relationships</b> — feedstock inputs have a near-perfect linear relationship with biogas output</li>
                <li><b>Simple is best</b> — Occam's Razor applies here</li>
                <li><b>No overfitting</b> — gap between train/test only 0.004</li>
                <li><b>Interpretable</b> — easy to explain to stakeholders</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 R² Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0a1628')
        ax.set_facecolor('#112240')

        colors = ['#64ffda' if m == 'Linear Regression' else '#1d4ed8'
                  for m in model_results['Model']]
        bars = ax.barh(model_results['Model'],
                       model_results['R² Score'], color=colors, edgecolor='none')
        ax.set_xlabel('R² Score', color='#ccd6f6')
        ax.set_title('Model R² Score Comparison', color='#ccd6f6', fontsize=14, pad=15)
        ax.set_xlim(0.88, 0.97)
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(x=0.95, color='#64ffda', linestyle='--', alpha=0.3)
        for bar, val in zip(bars, model_results['R² Score']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, color='#ccd6f6')
        plt.tight_layout()
        st.pyplot(fig)


# ============================================================
# TAB 4 — EDA INSIGHTS
# ============================================================
with tab4:
    st.markdown("### Key insights from exploratory data analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Target Variable — Biogas Production</h4>
            <ul>
                <li>Normally distributed — bell shaped</li>
                <li>Range: <b>40 — 125 m³/day</b></li>
                <li>Mean: <b>~79 m³/day</b></li>
                <li>No log transform needed</li>
            </ul>
            <h4>🌾 Key Feedstock Findings</h4>
            <ul>
                <li>Pig manure strongest predictor <b>(r = 0.70)</b></li>
                <li>Kitchen waste second <b>(r = 0.41)</b></li>
                <li>All features normally distributed</li>
                <li>No missing values in dataset</li>
            </ul>
            <h4>🌡️ Environmental Findings</h4>
            <ul>
                <li>Month has <b>zero</b> seasonal effect</li>
                <li>Humidity and rainfall near zero correlation</li>
                <li>Digester temp more important than ambient temp</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 Feature Correlation with Biogas")
        corr_data = pd.DataFrame({
            'Feature':     ['pig_manure', 'kitchen_waste', 'cassava',
                            'municipal_residue', 'chicken_litter',
                            'bagasse', 'temperature', 'digester_temp'],
            'Correlation': [0.70, 0.41, 0.34, 0.27, 0.25, 0.22, 0.12, 0.00]
        }).sort_values('Correlation', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0a1628')
        ax.set_facecolor('#112240')
        colors = ['#64ffda' if c >= 0.4 else '#1d4ed8'
                  for c in corr_data['Correlation']]
        ax.barh(corr_data['Feature'], corr_data['Correlation'],
                color=colors, edgecolor='none')
        ax.set_xlabel('Correlation with Biogas Production', color='#ccd6f6')
        ax.set_title('Feature Correlation with Target', color='#ccd6f6', fontsize=14, pad=15)
        ax.axvline(x=0.4, color='#64ffda', linestyle='--', alpha=0.5, label='Strong (≥0.4)')
        ax.tick_params(colors='#8892b0')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(facecolor='#112240', labelcolor='#ccd6f6')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">0.70</div>
            <div class="stat-label">Pig Manure Correlation</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">~0</div>
            <div class="stat-label">Seasonal Effect</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">0</div>
            <div class="stat-label">Missing Values</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-card">
            <div class="stat-number">0.004</div>
            <div class="stat-label">Train/Test Gap</div>
        </div>""", unsafe_allow_html=True)