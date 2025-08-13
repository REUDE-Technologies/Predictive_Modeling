# app.py — Width prediction dashboard (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# -------------------------------
# Streamlit page settings / style
# -------------------------------
st.set_page_config(page_title="Width Prediction Dashboard", layout="wide")

st.markdown("""
    <style>
        .main-card {
            background: linear-gradient(135deg, #1d2b64, #f8cdda);
            border-radius: 12px;
            padding: 14px;
            color: white;
            margin-bottom: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .metric-box {
            background: rgba(255, 255, 255, 0.10);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='main-card'>
    <h3 style='text-align: center; margin: 0;'>
        Predictive Control for Silicone Printing via Vision + ML
    </h3>
    <p style='text-align:center; margin: 6px 0 0 0;'>Enter process parameters to predict width and view derived physics features.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Utilities
# -------------------------------
def kpa_to_psi(kpa: float) -> float:
    return float(kpa) / 6.89475729  # 1 psi = 6.89475729 kPa

def compute_derived(needle_mm: float, speed_mms: float):
    """Compute area (mm^2), flow (mm^3/s), shear rate (1/s), viscosity (Pa.s) from inputs.
       Simple rheology proxy (power-law) used for demo; align with your dataset if needed."""
    # area of circular inlet: pi * d^2 / 4
    area_mm2 = np.pi * (needle_mm ** 2) / 4.0
    flow_mm3_s = speed_mms * area_mm2

    # approximate wall shear rate for capillary flow: ~ 8V/D (here: 4 * v / (d/2) = 8v/d)
    # v=linear speed (mm/s), d=diameter (mm). Keep units consistent (1/s)
    shear_rate = 8.0 * speed_mms / max(needle_mm, 1e-6)

    # simple power-law viscosity model: mu = K * gamma^(n-1) (illustrative)
    n_index = 0.06
    K_index = 0.9
    viscosity = K_index * (shear_rate ** (n_index - 1.0))

    return round(area_mm2, 4), round(flow_mm3_s, 4), round(shear_rate, 2), round(viscosity, 4)

# --------------------------------
# Train model (fits once per run)
# --------------------------------
@st.cache_resource(show_spinner=True)
def train_model(data_path: str = "Combined_files.csv"):
    df = pd.read_csv(data_path)
    # Drop unnamed index-like columns if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Harmonize column names used in training
    rename_map = {
        'Speed (mm/s)': 'Speed (mm/s)',
        'Pressure (psi)': 'Pressure (psi)',
        'Needle Size': 'Needle Size',
        'Time Period': 'Time Period',
        'Width (um)': 'Width (um)',
        'AreaA1 (mm2)': 'AreaA1 (mm2)',
        'FlowQ1 (mm3/s)': 'FlowQ1 (mm3/s)',
        'ShearS1 (1/s)': 'ShearS1 (1/s)',
        'ViscosN1 (PaS)': 'ViscosN1 (PaS)',
    }
    df = df.rename(columns=rename_map)

    # Required columns
    base_required = ['Width (um)', 'Needle Size', 'Thivex', 'Material',
                     'Pressure (psi)', 'Speed (mm/s)', 'Time Period']
    # Derived/physics columns (if present in your training file)
    derived_cols = ['AreaA1 (mm2)', 'FlowQ1 (mm3/s)', 'ShearS1 (1/s)', 'ViscosN1 (PaS)']
    present_derived = [c for c in derived_cols if c in df.columns]

    # Filter to rows with everything we need
    df_filtered = df.dropna(subset=base_required).copy()

    # Build training features; include derived if present
    features = base_required[1:] + present_derived
    target = 'Width (um)'
    X = df_filtered[features]
    y = df_filtered[target].astype(float)

    # Train/validation split for more realistic metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Set up preprocessing
    numeric_features = ['Pressure (psi)', 'Speed (mm/s)'] + [c for c in present_derived]
    categorical_features = ['Material', 'Time Period', 'Thivex', 'Needle Size']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1,
            max_depth=None, min_samples_split=2, min_samples_leaf=1
        ))
    ])

    model.fit(X_train, y_train)

    # Metrics on train and hold-out test split
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    r2_train = r2_score(y_train, y_hat_train)
    # Robust across sklearn versions: prefer squared=False, fallback to manual sqrt
    try:
        rmse_train = float(mean_squared_error(y_train, y_hat_train, squared=False))
    except TypeError:
        rmse_train = float(np.sqrt(mean_squared_error(y_train, y_hat_train)))
    r2_test = r2_score(y_test, y_hat_test)
    try:
        rmse_test = float(mean_squared_error(y_test, y_hat_test, squared=False))
    except TypeError:
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_hat_test)))

    meta = {
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "r2_test": r2_test,
        "rmse_test": rmse_test
    }
    return model, meta

model, meta = train_model()

# -------------------------------
# Sidebar Inputs (aligned to data)
# -------------------------------
with st.sidebar:
    st.header("📋 Input Parameters")

    # Your dataset commonly uses needles ~ 0.84 mm (18G) / 0.51 mm (21G)
    needle_str = st.radio("Needle Diameter (mm)", ['0.84', '0.51'], horizontal=True)
    needle_mm = float(needle_str)

    material = st.radio("Material", ['DS10', 'DS30', 'SS960'], horizontal=True)

    thivex_str = st.radio("Thivex", ['0%', '1%', '2%'], horizontal=True)
    # Keep percent symbol to match training labels
    thivex = thivex_str

    time_label = st.selectbox(
        "Time Period",
        ["Phase1: First 30 mins", "Phase2: 30 mins to 60 min", "Phase3: After 60 mins"]
    )

    # Your model trains on psi, but many operators think in kPa → convert
    pressure_kpa = st.number_input("Pressure (kPa)", min_value=300, max_value=750, value=448, step=5)
    pressure_psi = round(kpa_to_psi(pressure_kpa), 3)

    speed_mms = st.number_input("Printing Speed (mm/s)", min_value=10, max_value=70, value=20, step=1)

    pattern = st.text_input("Print Pattern (optional)", "Nested Polygon in Circle")

    submitted = st.button("🔍 Predict Width")

# -------------------------------
# Prediction + UI
# -------------------------------
if submitted:
    # Compute physics features (used if your model was trained with them)
    area_mm2, flow_mm3_s, shear_rate, viscosity = compute_derived(needle_mm, speed_mms)

    # Prepare an input row aligned to training feature names
    row = {
        'Needle Size': 18 if abs(needle_mm - 0.84) < 1e-3 else 21,  # map diameter to your dataset's "Needle Size"
        'Thivex': thivex,                       # data uses strings "0", "1", "2" or similar
        'Material': material,
        'Pressure (psi)': pressure_psi,
        'Speed (mm/s)': speed_mms,
        'Time Period': time_label
    }

    # If your training included derived columns, pass the computed values
    if 'AreaA1 (mm2)' in meta['features']:       row['AreaA1 (mm2)'] = area_mm2
    if 'FlowQ1 (mm3/s)' in meta['features']:     row['FlowQ1 (mm3/s)'] = flow_mm3_s
    if 'ShearS1 (1/s)' in meta['features']:      row['ShearS1 (1/s)'] = shear_rate
    if 'ViscosN1 (PaS)' in meta['features']:     row['ViscosN1 (PaS)'] = viscosity

    input_df = pd.DataFrame([row])
    pred_width_um = float(model.predict(input_df)[0])

    # (Optional) difference to internal diameter in µm if you want to display it:
    # map 18G≈0.838 mm ≈ 838 µm; 21G≈0.514 mm ≈ 514 µm (adjust to your exact hardware)
    needle_um = 838 if row['Needle Size'] == 18 else 514
    difference = abs(needle_um - pred_width_um)

    # -------- Metrics row --------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='metric-box'><h5>Inlet Area (mm²)</h5><h3>{area_mm2}</h3></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'><h5>Flow Rate (mm³/s)</h5><h3>{flow_mm3_s}</h3></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'><h5>Shear Rate (1/s)</h5><h3>{shear_rate}</h3></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-box'><h5>Viscosity (Pa·s)</h5><h3>{viscosity}</h3></div>", unsafe_allow_html=True)
    c5.markdown(
        f"<div class='metric-box'>"
        f"<h5>Model</h5>"
        f"<h4>Train: R² {meta['r2_train']:.3f} | RMSE {meta['rmse_train']:.1f}</h4>"
        f"<h4>Test: R² {meta['r2_test']:.3f} | RMSE {meta['rmse_test']:.1f}</h4>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    left, right = st.columns([1,1])
    with left:
        st.subheader("🔮 Predicted Width (µm)")
        st.markdown(f"### {pred_width_um:.1f} µm")

        st.caption("Difference to internal diameter (heuristic mapping 18→838 µm, 21→514 µm)")
        st.write(f"**|ID − Width| = {difference:.1f} µm**")

        st.write("**Inputs used**")
        st.json(row)

    with right:
        st.subheader("ℹ️ Notes")
        st.write("- Pressure converted from kPa → psi for model consistency.")
        st.write("- Physics-derived features computed from your inputs.")
        st.write("- If your training file **doesn’t** contain the derived columns, the model will still run using base features.")

else:
    st.info("Enter your parameters on the left and click **Predict Width**.")

