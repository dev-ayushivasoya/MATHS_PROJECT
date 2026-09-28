import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Financial Literacy & Allowance Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for excellent UI/UX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(14, 165, 233, 0.25);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    .section-card {
        background: white;
        border-radius: 14px;
        padding: 1.75rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .best-model-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    
    .formula-box {
        background: #f8fafc;
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #a7f3d0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #fed7aa;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .step-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border-left: 4px solid #6366f1;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SAMPLE DATA ====================
@st.cache_data
def generate_survey_data(n=40, seed=42):
    """Generate realistic survey data for engineering students' allowance vs savings."""
    np.random.seed(seed)
    X = np.random.uniform(3000, 18000, n)
    X = np.round(X / 100) * 100
    base_rate = 0.18
    quadratic_boost = 0.0000008 * (X ** 2) / 1000
    noise = np.random.normal(0, 800, n)
    Y = base_rate * X + quadratic_boost * X + noise
    Y = np.clip(Y, 200, X * 0.55)
    Y = np.round(Y / 50) * 50
    df = pd.DataFrame({
        'Student_ID': [f'S{i+1:02d}' for i in range(n)],
        'Allowance_X': X.astype(int),
        'Savings_Y': Y.astype(int)
    })
    return df.sort_values('Allowance_X').reset_index(drop=True)

# ==================== CURVE FITTING ====================
def fit_linear(X, Y):
    """Fit Y = a + bX using normal equations."""
    n = len(X)
    sum_X = np.sum(X)
    sum_Y = np.sum(Y)
    sum_X2 = np.sum(X ** 2)
    sum_XY = np.sum(X * Y)

    A = np.array([[n, sum_X],
                  [sum_X, sum_X2]], dtype=float)
    B = np.array([sum_Y, sum_XY], dtype=float)

    try:
        params = np.linalg.solve(A, B)
        a, b = params
    except np.linalg.LinAlgError:
        a, b = 0.0, 0.0

    Y_pred = a + b * X
    return a, b, Y_pred, {
        'sum_X': sum_X, 'sum_Y': sum_Y, 'sum_X2': sum_X2,
        'sum_XY': sum_XY, 'n': n, 'A': A, 'B': B
    }

def fit_quadratic(X, Y):
    """Fit Y = a + bX + cX² using normal equations."""
    n = len(X)
    sum_X = np.sum(X)
    sum_Y = np.sum(Y)
    sum_X2 = np.sum(X ** 2)
    sum_X3 = np.sum(X ** 3)
    sum_X4 = np.sum(X ** 4)
    sum_XY = np.sum(X * Y)
    sum_X2Y = np.sum((X ** 2) * Y)

    A = np.array([
        [n,      sum_X,  sum_X2],
        [sum_X,  sum_X2, sum_X3],
        [sum_X2, sum_X3, sum_X4]
    ], dtype=float)
    B = np.array([sum_Y, sum_XY, sum_X2Y], dtype=float)

    try:
        params = np.linalg.solve(A, B)
        a, b, c = params
    except np.linalg.LinAlgError:
        a, b, c = 0.0, 0.0, 0.0

    Y_pred = a + b * X + c * (X ** 2)
    sums = {
        'sum_X': sum_X, 'sum_Y': sum_Y, 'sum_X2': sum_X2,
        'sum_X3': sum_X3, 'sum_X4': sum_X4, 'sum_XY': sum_XY,
        'sum_X2Y': sum_X2Y, 'n': n, 'A': A, 'B': B
    }
    return a, b, c, Y_pred, sums

def calculate_metrics(Y, Y_pred, k):
    """Calculate SSE, Standard Error of Estimate, R²."""
    residuals = Y - Y_pred
    sse = np.sum(residuals ** 2)
    n = len(Y)
    syx = np.sqrt(sse / (n - k)) if (n - k) > 0 else np.nan
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - (sse / ss_tot) if ss_tot > 0 else 0.0
    return {'SSE': sse, 'Syx': syx, 'R2': r2, 'residuals': residuals}

def format_matrix(M, decimals=2):
    """Pretty-print a matrix for display."""
    rows = []
    for row in M:
        cells = [f"{v:.{decimals}f}" if abs(v) < 1e10 else f"{v:.4e}" for v in row]
        rows.append("[ " + "  ".join(f"{c:>14}" for c in cells) + " ]")
    return "\n".join(rows)


def build_sum_table(X, Y, model="linear"):
    """Row-by-row summation table (textbook style) plus a separate totals row."""
    # Exact Python ints when data are whole numbers (X⁴ gets very large)
    Xv = [int(t) if float(t).is_integer() else float(t) for t in X]
    Yv = [int(t) if float(t).is_integer() else float(t) for t in Y]

    data = {"X": Xv, "Y": Yv, "X²": [x ** 2 for x in Xv]}
    if model == "quadratic":
        data["X³"] = [x ** 3 for x in Xv]
        data["X⁴"] = [x ** 4 for x in Xv]
    data["XY"] = [x * y for x, y in zip(Xv, Yv)]
    if model == "quadratic":
        data["X²Y"] = [(x ** 2) * y for x, y in zip(Xv, Yv)]

    sym = {"X": "∑X", "Y": "∑Y", "X²": "∑X²", "X³": "∑X³",
           "X⁴": "∑X⁴", "XY": "∑XY", "X²Y": "∑X²Y"}
    fmt = lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"

    rows = pd.DataFrame({c: [fmt(v) for v in vals] for c, vals in data.items()},
                        index=range(1, len(Xv) + 1))
    rows.index.name = "#"
    totals = pd.DataFrame({sym[c]: [fmt(sum(vals))] for c, vals in data.items()},
                          index=[f"Σ (n = {len(Xv)})"])
    return rows, totals

def show_sum_table(X, Y, model):
    rows, totals = build_sum_table(X, Y, model)
    st.dataframe(rows, use_container_width=True, height=380)
    st.markdown("**Column totals (Σ):**")
    st.dataframe(totals, use_container_width=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio(
        "Select Section",
        [
            "🏠 Home",
            "📥 Survey Data",
            "📐 Normal Equations",
            "📈 Curve Fitting",
            "⚖️ Model Comparison",
            "🔮 Savings Predictor",
            "🏦 Banking Insights"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#94a3b8;'>
    <b>Module V</b><br>
    Statistical Techniques<br>
    Bivariate Curve Fitting<br><br>
    Applied Mathematics Thinking-I
    </div>
    """, unsafe_allow_html=True)

# ==================== DATA SOURCE (session state) ====================
if "df" not in st.session_state:
    st.session_state.df = generate_survey_data(n=40, seed=42)
    st.session_state.data_source = "sample"

df = st.session_state.df
X = df['Allowance_X'].values.astype(float)
Y = df['Savings_Y'].values.astype(float)

# Fit models only if we have enough data
if len(df) >= 5:
    a_lin, b_lin, Y_lin, sums_lin = fit_linear(X, Y)
    a_quad, b_quad, c_quad, Y_quad, sums_quad = fit_quadratic(X, Y)
    metrics_lin = calculate_metrics(Y, Y_lin, k=2)
    metrics_quad = calculate_metrics(Y, Y_quad, k=3)
    best_is_quad = (metrics_quad['R2'] > metrics_lin['R2']) and (metrics_quad['Syx'] < metrics_lin['Syx'])
    best_model_name = "Quadratic (Parabolic)" if best_is_quad else "Linear"
    best_R2 = metrics_quad['R2'] if best_is_quad else metrics_lin['R2']
    best_Syx = metrics_quad['Syx'] if best_is_quad else metrics_lin['Syx']
else:
    a_lin = b_lin = a_quad = b_quad = c_quad = 0.0
    Y_lin = Y_quad = np.zeros(len(Y))
    sums_lin = sums_quad = {'n': len(df), 'sum_X': 0, 'sum_Y': 0, 'sum_X2': 0, 'sum_X3': 0, 'sum_X4': 0, 'sum_XY': 0, 'sum_X2Y': 0, 'A': np.eye(2), 'B': np.zeros(2)}
    metrics_lin = metrics_quad = {'SSE': 0, 'Syx': 0, 'R2': 0, 'residuals': np.zeros(len(Y))}
    best_is_quad = False
    best_model_name = "—"
    best_R2 = best_Syx = 0.0

# ==================== PAGES ====================

# ---- HOME ----
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>💰 Financial Literacy & Allowance Expenditure Analysis</h1>
        <p>Bivariate Curve Fitting • Linear vs Quadratic Models • Student Savings Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Project objective ----
    st.markdown("### 🎯 Project Objective")
    st.markdown("""
    <div class="section-card">
        <p style="margin:0;">
        Engineering students receive a monthly allowance, but how much of it do they actually save?
        This project uses <b>survey data</b> on allowance (X) and savings (Y) to build a mathematical model
        that <b>predicts a student's savings from their allowance</b>. Banks and fintech apps can use such a
        model to design student-friendly savings and deposit products.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Roadmap ----
    st.markdown("### 🧭 How to Use This App")
    steps = [
        ("1", "📥", "Survey Data", "Load sample data, upload a CSV, or enter your own values."),
        ("2", "📐", "Normal Equations", "See the summation tables and the matrix setup."),
        ("3", "📈", "Curve Fitting", "View fitted curves and residual plots."),
        ("4", "⚖️", "Model Comparison", "Compare R² and Sᵧ|ₓ to pick the best model."),
        ("5", "🔮", "Savings Predictor", "Enter an allowance and get predicted savings."),
        ("6", "🏦", "Banking Insights", "Turn the results into product ideas."),
    ]
    for row_start in (0, 3):
        cols = st.columns(3)
        for col, (num, icon, title, desc) in zip(cols, steps[row_start:row_start + 3]):
            with col:
                st.markdown(f"""
                <div class="section-card" style="padding:1.25rem; min-height:150px;">
                    <div style="font-size:0.8rem; color:#0ea5e9; font-weight:700; letter-spacing:0.5px;">STEP {num}</div>
                    <div style="font-size:1.15rem; font-weight:600; margin:0.35rem 0;">{icon} {title}</div>
                    <div style="color:#64748b; font-size:0.92rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ---- Financial literacy essentials ----
    st.markdown("### 💡 Financial Literacy Essentials")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="insight-box">
            <b>📊 The 50/30/20 Rule</b><br>
            <span style="font-size:0.92rem;">Spend about 50% on needs, 30% on wants, and put 20% into savings.</span>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="insight-box">
            <b>🛟 Build an Emergency Fund</b><br>
            <span style="font-size:0.92rem;">Aim to keep 3–6 months of essential expenses in easy-to-access savings.</span>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="insight-box">
            <b>⏳ Start Early</b><br>
            <span style="font-size:0.92rem;">Small, regular savings started in college grow much more through compounding.</span>
        </div>
        """, unsafe_allow_html=True)

# ---- SURVEY DATA ----
elif page == "📥 Survey Data":
    st.markdown("""
    <div class="main-header">
        <h1>📥 Survey Data</h1>
        <p>Monthly Allowance (X) vs Monthly Savings (Y)</p>
    </div>
    """, unsafe_allow_html=True)

    data_option = st.radio(
        "Choose how to load data:",
        ["📂 Load Sample Data", "📤 Upload CSV File", "✍️ Enter Data Manually"],
        horizontal=True
    )

    if data_option == "📂 Load Sample Data":
        n_students = st.slider("Number of Students (N)", 35, 60, 40, 1)
        seed = st.number_input("Random Seed", 1, 999, 42)
        if st.button("Load Sample Data", type="primary"):
            st.session_state.df = generate_survey_data(n=n_students, seed=int(seed))
            st.session_state.data_source = "sample"
            st.success(f"Loaded sample data with N = {n_students} students.")
            st.rerun()

    elif data_option == "📤 Upload CSV File":
        st.markdown("Upload a CSV with columns **Allowance_X** and **Savings_Y** (or any two numeric columns for X and Y).")
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                st.write("Preview of uploaded file:")
                st.dataframe(raw.head(10), use_container_width=True)

                cols = list(raw.columns)
                col_x = st.selectbox("Column for Allowance (X)", cols, index=0)
                col_y = st.selectbox("Column for Savings (Y)", cols, index=min(1, len(cols)-1))

                if st.button("Use This Data", type="primary"):
                    new_df = pd.DataFrame({
                        'Student_ID': [f'S{i+1:02d}' for i in range(len(raw))],
                        'Allowance_X': pd.to_numeric(raw[col_x], errors='coerce'),
                        'Savings_Y': pd.to_numeric(raw[col_y], errors='coerce')
                    }).dropna()
                    if len(new_df) < 5:
                        st.error("Need at least 5 valid numeric rows.")
                    else:
                        st.session_state.df = new_df.reset_index(drop=True)
                        st.session_state.data_source = "upload"
                        st.success(f"Loaded {len(new_df)} records from CSV.")
                        st.rerun()
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    elif data_option == "✍️ Enter Data Manually":
        st.markdown("Enter allowance and savings pairs. Add rows as needed.")
        default_rows = 8
        if "manual_rows" not in st.session_state:
            st.session_state.manual_rows = default_rows

        n_rows = st.number_input("Number of rows", min_value=5, max_value=100, value=st.session_state.manual_rows)
        st.session_state.manual_rows = n_rows

        # Editable table via data_editor
        if "manual_df" not in st.session_state or len(st.session_state.manual_df) != n_rows:
            st.session_state.manual_df = pd.DataFrame({
                'Allowance_X': [5000 + i * 1000 for i in range(n_rows)],
                'Savings_Y': [800 + i * 200 for i in range(n_rows)]
            })

        edited = st.data_editor(
            st.session_state.manual_df,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Allowance_X": st.column_config.NumberColumn("Allowance X (₹)", min_value=0, step=100),
                "Savings_Y": st.column_config.NumberColumn("Savings Y (₹)", min_value=0, step=50),
            }
        )

        if st.button("Use Manual Data", type="primary"):
            cleaned = edited.dropna()
            cleaned = cleaned[(cleaned['Allowance_X'] > 0) & (cleaned['Savings_Y'] >= 0)]
            if len(cleaned) < 5:
                st.error("Need at least 5 valid rows.")
            else:
                new_df = pd.DataFrame({
                    'Student_ID': [f'S{i+1:02d}' for i in range(len(cleaned))],
                    'Allowance_X': cleaned['Allowance_X'].astype(int).values,
                    'Savings_Y': cleaned['Savings_Y'].astype(int).values
                })
                st.session_state.df = new_df.reset_index(drop=True)
                st.session_state.data_source = "manual"
                st.session_state.manual_df = edited
                st.success(f"Loaded {len(new_df)} manually entered records.")
                st.rerun()

    # Always show current data
    st.markdown("---")
    st.markdown(f"### Current Dataset  ·  Source: **{st.session_state.data_source}**  ·  N = **{len(df)}**")

    if len(df) == 0:
        st.warning("No data loaded yet. Use one of the options above.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                df.style.format({'Allowance_X': '₹{:,.0f}', 'Savings_Y': '₹{:,.0f}'}),
                use_container_width=True,
                height=360
            )
        with col2:
            st.markdown("### Summary")
            st.metric("Mean Allowance", f"₹{X.mean():,.0f}")
            st.metric("Mean Savings", f"₹{Y.mean():,.0f}")
            st.metric("Savings Rate", f"{(Y.mean()/X.mean()*100):.1f}%")
            st.metric("Correlation", f"{np.corrcoef(X, Y)[0,1]:.3f}")

        st.markdown("### Scatter Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(X, Y, c='#0ea5e9', s=70, alpha=0.75, edgecolors='white', linewidth=0.8)
        ax.set_xlabel('Monthly Allowance X (₹)', fontsize=11)
        ax.set_ylabel('Monthly Savings Y (₹)', fontsize=11)
        ax.set_title('Allowance vs Savings', fontsize=13, fontweight='600')
        ax.grid(True, alpha=0.3, linestyle='--')
        sns.despine()
        st.pyplot(fig)
        plt.close()

# ---- NORMAL EQUATIONS ----
elif page == "📐 Normal Equations":
    st.markdown("""
    <div class="main-header">
        <h1>📐 Normal Equations</h1>
        <p>Summation tables & matrix algebra solution</p>
    </div>
    """, unsafe_allow_html=True)

    if len(df) < 5:
        st.warning("Load at least 5 data points first (Survey Data page).")
        st.stop()

    st.markdown("---")
    st.markdown("### Linear Model — Matrix Form & Solution")

    st.markdown("""
    Normal equations:
    """)
    st.latex(r"\sum Y = n a + b \sum X")
    st.latex(r"\sum XY = a \sum X + b \sum X^{2}")

    st.markdown("In matrix form:")
    st.latex(r"""
    \begin{bmatrix} n & \sum X \\ \sum X & \sum X^{2} \end{bmatrix}
    \begin{bmatrix} a \\ b \end{bmatrix}
    =
    \begin{bmatrix} \sum Y \\ \sum XY \end{bmatrix}
    """)

    st.markdown("**Summation table (Linear model):**")
    show_sum_table(X, Y, "linear")

    st.latex(rf"\hat{{Y}} = {a_lin:.4f} + ({b_lin:.6f})\,X")

    st.markdown("---")
    st.markdown("### Quadratic Model — Matrix Form & Solution")

    st.markdown("Normal equations:")
    st.latex(r"\sum Y = n a + b \sum X + c \sum X^{2}")
    st.latex(r"\sum XY = a \sum X + b \sum X^{2} + c \sum X^{3}")
    st.latex(r"\sum X^{2}Y = a \sum X^{2} + b \sum X^{3} + c \sum X^{4}")

    st.markdown("In matrix form:")
    st.latex(r"""
    \begin{bmatrix}
    n & \sum X & \sum X^{2} \\
    \sum X & \sum X^{2} & \sum X^{3} \\
    \sum X^{2} & \sum X^{3} & \sum X^{4}
    \end{bmatrix}
    \begin{bmatrix} a \\ b \\ c \end{bmatrix}
    =
    \begin{bmatrix} \sum Y \\ \sum XY \\ \sum X^{2}Y \end{bmatrix}
    """)

    st.markdown("**Summation table (Quadratic model):**")
    show_sum_table(X, Y, "quadratic")

    st.latex(rf"\hat{{Y}} = {a_quad:.4f} + ({b_quad:.6f})\,X + ({c_quad:.10f})\,X^{2}")

# ---- CURVE FITTING ----
elif page == "📈 Curve Fitting":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Curve Fitting</h1>
        <p>Fitted parameters and visual comparison</p>
    </div>
    """, unsafe_allow_html=True)

    if len(df) < 5:
        st.warning("Load at least 5 data points first (Survey Data page).")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-card"><h4>Linear Model: Y = a + bX</h4></div>
        """, unsafe_allow_html=True)
        st.latex(rf"\hat{{Y}} = {a_lin:.4f} + ({b_lin:.6f})\,X")
        st.metric("Intercept (a)", f"{a_lin:.4f}")
        st.metric("Slope (b)", f"{b_lin:.6f}")

    with col2:
        st.markdown("""
        <div class="section-card"><h4>Quadratic Model: Y = a + bX + cX²</h4></div>
        """, unsafe_allow_html=True)
        st.latex(rf"\hat{{Y}} = {a_quad:.4f} + ({b_quad:.6f})\,X + ({c_quad:.10f})\,X^{2}")
        st.metric("a", f"{a_quad:.4f}")
        st.metric("b", f"{b_quad:.6f}")
        st.metric("c", f"{c_quad:.10f}")

    st.markdown("---")
    st.markdown("### Fitted Curves vs Actual Data")

    X_smooth = np.linspace(X.min(), X.max(), 200)
    Y_lin_smooth = a_lin + b_lin * X_smooth
    Y_quad_smooth = a_quad + b_quad * X_smooth + c_quad * (X_smooth ** 2)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(X, Y, c='#64748b', s=60, alpha=0.7, edgecolors='white', linewidth=0.7, label='Actual Data', zorder=3)
    ax.plot(X_smooth, Y_lin_smooth, color='#ef4444', linewidth=2.5, label=f'Linear', zorder=2)
    ax.plot(X_smooth, Y_quad_smooth, color='#10b981', linewidth=2.5, label=f'Quadratic', zorder=2)
    ax.set_xlabel('Monthly Allowance X (₹)', fontsize=12)
    ax.set_ylabel('Monthly Savings Y (₹)', fontsize=12)
    ax.set_title('Linear vs Quadratic Curve Fit', fontsize=14, fontweight='600')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    sns.despine()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Residual Plots")
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.scatter(Y_lin, metrics_lin['residuals'], c='#ef4444', alpha=0.7, s=50)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Predicted Y (Linear)')
    ax1.set_ylabel('Residual')
    ax1.set_title('Linear Residuals')
    ax1.grid(True, alpha=0.3)
    ax2.scatter(Y_quad, metrics_quad['residuals'], c='#10b981', alpha=0.7, s=50)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Predicted Y (Quadratic)')
    ax2.set_ylabel('Residual')
    ax2.set_title('Quadratic Residuals')
    ax2.grid(True, alpha=0.3)
    sns.despine()
    st.pyplot(fig2)
    plt.close()

# ---- MODEL COMPARISON ----
elif page == "⚖️ Model Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Model Comparison</h1>
        <p>R² and Standard Error of Estimate — selecting the best curve</p>
    </div>
    """, unsafe_allow_html=True)

    if len(df) < 5:
        st.warning("Load at least 5 data points first (Survey Data page).")
        st.stop()

    metrics_df = pd.DataFrame({
        'Metric': [
            'R² (Coefficient of Determination)',
            'SSE (Residual Sum of Squares)',
            'Sᵧ|ₓ (Standard Error of Estimate)',
            'Parameters (k)',
            'Degrees of Freedom (n−k)'
        ],
        'Linear Model': [
            f"{metrics_lin['R2']:.6f}",
            f"{metrics_lin['SSE']:,.2f}",
            f"{metrics_lin['Syx']:.4f}",
            "2 (a, b)",
            f"{len(X)-2}"
        ],
        'Quadratic Model': [
            f"{metrics_quad['R2']:.6f}",
            f"{metrics_quad['SSE']:,.2f}",
            f"{metrics_quad['Syx']:.4f}",
            "3 (a, b, c)",
            f"{len(X)-3}"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("### Decision")
    if best_is_quad:
        st.markdown(f"""
        <div class="insight-box">
            <span class="best-model-badge">✓ Best Model: Quadratic</span>
            <br><br>
            Higher R² ({metrics_quad['R2']:.4f} > {metrics_lin['R2']:.4f}) and lower Sᵧ|ₓ
            ({metrics_quad['Syx']:.2f} < {metrics_lin['Syx']:.2f}).
            Savings behaviour shows accelerated (non-linear) growth at higher allowance levels.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
            <span class="best-model-badge">✓ Best Model: Linear</span>
            <br><br>
            Linear model is preferred based on R² and/or Sᵧ|ₓ.
            Savings grow approximately proportionally with allowance.
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Linear R²</div>
            <div class="metric-value" style="color:#ef4444">{metrics_lin['R2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem;">
            <div class="metric-label">Linear Sᵧ|ₓ</div>
            <div class="metric-value" style="color:#ef4444">{metrics_lin['Syx']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Quadratic R²</div>
            <div class="metric-value" style="color:#10b981">{metrics_quad['R2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem;">
            <div class="metric-label">Quadratic Sᵧ|ₓ</div>
            <div class="metric-value" style="color:#10b981">{metrics_quad['Syx']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

# ---- SAVINGS PREDICTOR ----
elif page == "🔮 Savings Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Savings Predictor</h1>
        <p>Input monthly allowance → predicted savings & product recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    if len(df) < 5:
        st.warning("Load at least 5 data points first (Survey Data page).")
        st.stop()

    st.markdown(f"""
    <div class="insight-box">
        Active Model: <b>{best_model_name}</b> &nbsp;|&nbsp; R² = <b>{best_R2:.4f}</b> &nbsp;|&nbsp; Sᵧ|ₓ = <b>{best_Syx:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    allowance_input = st.slider(
        "Monthly Allowance (₹)",
        min_value=2000, max_value=25000, value=10000, step=500
    )

    pred_lin = max(0, a_lin + b_lin * allowance_input)
    pred_quad = max(0, a_quad + b_quad * allowance_input + c_quad * (allowance_input ** 2))
    pred_best = pred_quad if best_is_quad else pred_lin

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Linear Prediction</div>
            <div class="metric-value" style="color:#ef4444">₹{pred_lin:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Quadratic Prediction</div>
            <div class="metric-value" style="color:#10b981">₹{pred_quad:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border:2px solid #10b981;">
            <div class="metric-label">★ Recommended (Best Model)</div>
            <div class="metric-value" style="color:#059669">₹{pred_best:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    savings_rate = (pred_best / allowance_input * 100) if allowance_input > 0 else 0
    st.markdown(f"""
    <div style="text-align:center; margin:1.5rem 0; font-size:1.1rem;">
        Predicted Savings Rate: <b style="color:#0ea5e9;">{savings_rate:.1f}%</b> of allowance
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    X_smooth = np.linspace(2000, 25000, 300)
    Y_lin_s = np.maximum(a_lin + b_lin * X_smooth, 0)
    Y_quad_s = np.maximum(a_quad + b_quad * X_smooth + c_quad * (X_smooth ** 2), 0)
    ax.plot(X_smooth, Y_lin_s, color='#ef4444', linewidth=2, alpha=0.8, label='Linear')
    ax.plot(X_smooth, Y_quad_s, color='#10b981', linewidth=2, alpha=0.8, label='Quadratic')
    ax.scatter(X, Y, c='#94a3b8', s=40, alpha=0.5, label='Survey Data')
    ax.scatter([allowance_input], [pred_best], c='#f59e0b', s=180, zorder=5,
               edgecolors='white', linewidth=2, label=f'Prediction: ₹{pred_best:,.0f}')
    ax.axvline(allowance_input, color='#f59e0b', linestyle=':', alpha=0.7)
    ax.set_xlabel('Monthly Allowance (₹)')
    ax.set_ylabel('Predicted Savings (₹)')
    ax.set_title('Prediction on Fitted Curves')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    sns.despine()
    st.pyplot(fig)
    plt.close()

    st.markdown("### 🏦 Recommended Student Savings Products")
    if pred_best < 1000:
        products = [
            ("Digital Piggy Bank / Auto-Save", "Round-up apps (Paytm, PhonePe) — start with small auto-transfers."),
            ("Mini Recurring Deposit", "Bank RD from ₹100–500/month."),
            ("Emergency Cash Stash", "Keep ₹500–1000 liquid in a zero-balance savings account."),
        ]
    elif pred_best < 3000:
        products = [
            ("Student Savings + RD", "Set up ₹1,000–2,000 monthly RD in a student account."),
            ("Liquid Fund SIP", "Small SIP (₹500–1,500) via Groww / Zerodha."),
            ("Goal-based Savings", "Target laptop, courses, or semester fees."),
        ]
    else:
        products = [
            ("Systematic Investment Plan (SIP)", f"₹{int(pred_best*0.6):,}–{int(pred_best*0.8):,}/month in equity or hybrid funds."),
            ("RD + PPF", "Split between bank RD and Public Provident Fund."),
            ("Fixed Deposit Ladder", "Short-tenure FDs (3–12 months) for better rates."),
        ]
    for i, (title, desc) in enumerate(products, 1):
        st.markdown(f"""
        <div class="section-card" style="padding:1rem 1.25rem;">
            <b>{i}. {title}</b><br>
            <span style="color:#64748b; font-size:0.95rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

# ---- BANKING INSIGHTS ----
elif page == "🏦 Banking Insights":
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Banking Insights</h1>
        <p>Model summary for deposit prediction & student products</p>
    </div>
    """, unsafe_allow_html=True)

    if len(df) < 5:
        st.warning("Load at least 5 data points first (Survey Data page).")
        st.stop()

    summary_df = pd.DataFrame({
        'Aspect': [
            'Best Predictive Model',
            'R²',
            'Standard Error Sᵧ|ₓ',
            'Sample Size N',
            'Average Allowance',
            'Average Savings',
            'Avg Savings Rate'
        ],
        'Value': [
            best_model_name,
            f"{best_R2:.4f}",
            f"₹{best_Syx:.2f}",
            str(len(df)),
            f"₹{X.mean():,.0f}",
            f"₹{Y.mean():,.0f}",
            f"{(Y.mean()/X.mean()*100):.1f}%"
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="section-card">
        <h4>Quick Applications</h4>
        <ul>
            <li>Use the best-fit curve to estimate expected student deposits from allowance data.</li>
            <li>Design tiered savings products that match predicted savings levels.</li>
            <li>Identify high-savers (positive residuals) for priority product offers.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Mini Project • Module V: Statistical Techniques (Bivariate Curve Fitting)<br>
    Applied Mathematics Thinking-I • Second Year Engineering
</div>
""", unsafe_allow_html=True)
