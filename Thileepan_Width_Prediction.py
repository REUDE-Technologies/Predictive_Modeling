# app.py — Width Prediction Dashboard with EDA, Feature Importance, and SHAP
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import shap

st.set_page_config(page_title="Width Prediction: ML + Physics + Vision", layout="wide")

# -------------------- helpers --------------------
INTERNAL_DIAMETER = {18: 838.0, 21: 514.0}  # µm (adjust if needed)

def compute_physics_columns(df: pd.DataFrame,
                            needle_mm_col: str,
                            speed_col: str,
                            new_col_name: str,
                            formula_key: str = "None",
                            custom_expr: str = ""):
    """
    Adds a new column via preset formulas or a custom expression using pandas.eval.
    Presets expect:
      - needle diameter in mm (needle_mm_col)
      - speed in mm/s (speed_col)
    Preset outputs:
      - AreaA1 (mm^2)   : pi * d^2 / 4
      - FlowQ1 (mm^3/s) : Area * speed
      - ShearS1 (1/s)   : ~ 8 * speed / d
      - ViscosN1 (Pa·s) : K * (shear)^(n - 1), defaults K=0.9, n=0.06
    """
    if new_col_name.strip() == "":
        return df, None

    df = df.copy()

    try:
        if formula_key == "AreaA1 (mm2) = π*d^2/4":
            df[new_col_name] = np.pi * (df[needle_mm_col].astype(float)**2) / 4.0

        elif formula_key == "FlowQ1 (mm3/s) = Area * Speed":
            area = np.pi * (df[needle_mm_col].astype(float)**2) / 4.0
            df[new_col_name] = area * df[speed_col].astype(float)

        elif formula_key == "ShearS1 (1/s) ≈ 8*Speed/d":
            d = df[needle_mm_col].astype(float).replace(0, np.nan)
            df[new_col_name] = 8.0 * df[speed_col].astype(float) / d
            df[new_col_name] = df[new_col_name].fillna(0.0)

        elif formula_key == "ViscosN1 (Pa·s) = K * shear^(n-1)":
            d = df[needle_mm_col].astype(float).replace(0, np.nan)
            shear = 8.0 * df[speed_col].astype(float) / d
            K, n = 0.9, 0.06
            df[new_col_name] = K * np.power(shear.replace(0, 1e-9), (n - 1.0))
            df[new_col_name] = df[new_col_name].fillna(df[new_col_name].median())

        elif formula_key == "Custom (pandas.eval)":
            # Let the user refer to existing columns by name (safe-ish use)
            # Example: "(Pressure_psi * 0.5) + (Speed_mms * 2) - AreaA1"
            df[new_col_name] = pd.eval(custom_expr, engine="python", target=df)

        else:
            # No formula selected; just create an empty column?
            df[new_col_name] = np.nan

        return df, None

    except Exception as e:
        return df, f"Failed to compute new column: {e}"

def build_model(X: pd.DataFrame, y: pd.Series):
    """
    Builds and returns a pipeline (scaler + OHE + RandomForest).
    """
    # Auto-detect numeric and categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1,
            max_depth=None, min_samples_split=2, min_samples_leaf=1
        ))
    ])
    return model, numeric_cols, categorical_cols

def quality_bucket(width_diff):
    """
    Classify the print quality by difference (PredWidth - InternalDiameter).
    """
    if -50 <= width_diff <= 50:
        return "✅ Perfect"
    if -75 <= width_diff <= 75:
        return "🟡 Acceptable"
    if width_diff < -75:
        return "🔵 Over Extrusion"
    return "🔴 Under Extrusion"

# -------------------- UI: banner --------------------
st.markdown("""
<div style="padding:10px;border-radius:10px;background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);color:#fff;">
  <h3 style="margin:0;">Vision-based Width Prediction • ML + Physics Features</h3>
  <p style="margin:6px 0 0 0;">Upload training data, engineer features, train the model, and predict optimal printing outcomes.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

# -------------------- Right Sidebar: Data & Modeling --------------------
with st.sidebar:
    st.header("📂 Data & Modeling")

    uploaded = st.file_uploader("Upload CSV (or leave empty to use 'Combined_files.csv')", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        # fallback to a default filename in working dir
        try:
            df = pd.read_csv("Combined_files.csv")
        except Exception:
            st.error("No file uploaded and 'Combined_files.csv' not found.")
            st.stop()

    st.caption(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    st.subheader("Select Training Columns")
    all_cols = df.columns.tolist()
    y_col = st.selectbox("Target (y)", options=all_cols, index=all_cols.index('Width (um)') if 'Width (um)' in all_cols else 0)
    X_cols = st.multiselect("Features (X)", options=[c for c in all_cols if c != y_col],
                            default=[c for c in ['Speed (mm/s)','Pressure (psi)','Material','Thivex','Time Period','Needle Size',
                                                 'AreaA1 (mm2)','FlowQ1 (mm3/s)','ShearS1 (1/s)','ViscosN1 (PaS)']
                                     if c in all_cols])

    st.divider()
    st.subheader("➕ Add New Column by Formula")
    # The user must specify which columns hold physical quantities
    needle_mm_col = st.selectbox("Column: Needle Diameter (mm) or map it", options=all_cols,
                                 index=all_cols.index('Needle dia (mm)') if 'Needle dia (mm)' in all_cols else 0)
    speed_col = st.selectbox("Column: Speed (mm/s)", options=all_cols,
                             index=(all_cols.index('Speed (mm/s)') if 'Speed (mm/s)' in all_cols else 0))
    new_col_name = st.text_input("New Column Name (e.g., AreaA1 (mm2))", value="")
    formula_choice = st.selectbox("Formula",
                                  options=["None",
                                           "AreaA1 (mm2) = π*d^2/4",
                                           "FlowQ1 (mm3/s) = Area * Speed",
                                           "ShearS1 (1/s) ≈ 8*Speed/d",
                                           "ViscosN1 (Pa·s) = K * shear^(n-1)",
                                           "Custom (pandas.eval)"])
    custom_expr = st.text_input("Custom expression (optional, uses column names)", value="")

    add_col_clicked = st.button("Add Column")

    if add_col_clicked and new_col_name.strip() != "":
        df, err = compute_physics_columns(df, needle_mm_col, speed_col, new_col_name, formula_choice, custom_expr)
        if err:
            st.error(err)
        else:
            st.success(f"Added '{new_col_name}' to the dataframe.")
            # Auto-include the new column if not in X yet
            if new_col_name not in X_cols:
                X_cols.append(new_col_name)

    st.divider()
    run_training = st.button("Train & Analyze")

# -------------------- Train + EDA center area --------------------
if run_training:
    # Clean + subset
    miss_cols = [c for c in [y_col] + X_cols if c not in df.columns]
    if miss_cols:
        st.error(f"Selected columns not found in data: {miss_cols}")
        st.stop()

    data = df.dropna(subset=[y_col] + X_cols).copy()
    if data.empty:
        st.error("No rows left after dropping NA in selected columns.")
        st.stop()

    X = data[X_cols]
    y = data[y_col].astype(float)

    # Train/test split for quick metrics
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe, num_cols, cat_cols = build_model(X_tr, y_tr)
    pipe.fit(X_tr, y_tr)
    y_hat = pipe.predict(X_te)
    r2 = r2_score(y_te, y_hat)
    try:
        rmse = float(mean_squared_error(y_te, y_hat, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))

    # Save in session
    st.session_state["trained_pipeline"] = pipe
    st.session_state["X_cols"] = X_cols
    st.session_state["y_col"] = y_col
    st.session_state["numeric"] = num_cols
    st.session_state["categorical"] = cat_cols
    st.session_state["data"] = data

    # ---- Layout
    m1, m2 = st.columns(2)
    with m1:
        st.metric("R² (test)", f"{r2:.3f}")
    with m2:
        st.metric("RMSE (test, µm)", f"{rmse:.1f}")

    st.markdown("### 📈 Pearson Correlation Heatmap (numeric features only)")
    num_for_corr = data[X_cols + [y_col]].select_dtypes(include=[np.number])
    if num_for_corr.shape[1] >= 2:
        plt.figure(figsize=(6.8, 5.2))
        corr = num_for_corr.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    # ---- Feature importance (using RF model inside pipeline)
    st.markdown("### 🌲 Feature Importance (Random Forest)")
    # Extract trained RF and one-hot feature names
    rf = pipe.named_steps["rf"]
    pre = pipe.named_steps["pre"]
    ohe = pre.named_transformers_["cat"]
    # build full feature names: scaled numeric + OHE cats
    num_names = num_cols
    cat_expanded = []
    if hasattr(ohe, "get_feature_names_out"):
        cat_expanded = ohe.get_feature_names_out(cat_cols).tolist()
    full_names = num_names + cat_expanded
    importances = rf.feature_importances_
    fi = pd.DataFrame({"feature": full_names, "importance": importances}).sort_values("importance", ascending=False)
    st.dataframe(fi.head(30), use_container_width=True)

    plt.figure(figsize=(7.2, 4.8))
    sns.barplot(x="importance", y="feature", data=fi.head(20))
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # ---- SHAP summary
    st.markdown("### 🔎 SHAP Summary (top 100 samples)")
    # Build a small background set
    sample_X = X_tr.sample(min(100, len(X_tr)), random_state=42)
    # Transform sample through preprocessor to get model input
    # For tree models, we can use TreeExplainer directly on pipeline's rf by passing preprocessed array
    bg = pre.transform(sample_X)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(pre.transform(sample_X))
    try:
        st.set_option('deprecation.showPyplotGlobalUse', False)
    except Exception:
        pass
    shap.summary_plot(shap_values, features=pre.transform(sample_X), feature_names=full_names, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

# -------------------- Right Sidebar: Prediction Panel --------------------
with st.sidebar:
    st.divider()
    st.header("🎯 Predict Width")

    # Inputs
    needle_size = st.selectbox("Needle Size (Gauge)", options=[18, 21], index=0)
    material_in = st.selectbox("Material", options=sorted(df['Material'].astype(str).unique()) if 'Material' in df.columns else ["DS10","DS30","SS960"])
    thivex_in = st.selectbox("Thivex", options=sorted(df['Thivex'].astype(str).unique()) if 'Thivex' in df.columns else ["0","1","2"])
    time_in = st.selectbox("Time Period", options=sorted(df['Time Period'].astype(str).unique()) if 'Time Period' in df.columns else ["Phase1: First 30 mins","Phase2: 30 mins to 60 min","Phase3: After 60 mins"])
    press_in = st.number_input("Pressure (psi)", min_value=20.0, max_value=200.0, value=85.0, step=1.0)
    speed_in = st.number_input("Speed (mm/s)", min_value=5.0, max_value=4000.0, value=1200.0, step=10.0)

    pred_btn = st.button("🔮 Predict Line Width")

# -------------------- Run prediction --------------------
if pred_btn:
    if "trained_pipeline" not in st.session_state:
        st.error("Please train the model first (upload/select columns and click 'Train & Analyze').")
        st.stop()

    # pipe = st.session_state["trained_pipeline"]
    pipe = st.session_state["trained_pipeline"]
    data = st.session_state["data"]
    X_cols = st.session_state["X_cols"]
    
    X = data[X_cols]
    y = data[y_col].astype(float)
    pipe, num_cols, cat_cols = build_model(X, y)
    pipe.fit(X, y)
    # pipe = pipe.fit(X, y)
    # X_cols = st.session_state["X_cols"]

    # Build a single-row DataFrame aligned to X_cols
    # Try to populate missing engineered columns if names match
    row = {}
    for c in X_cols:
        if c == 'Needle Size':
            row[c] = needle_size
        elif c == 'Material':
            row[c] = material_in
        elif c == 'Thivex':
            row[c] = thivex_in
        elif c == 'Time Period':
            row[c] = time_in
        elif c == 'Pressure (psi)':
            row[c] = press_in
        elif c == 'Speed (mm/s)':
            row[c] = speed_in
        else:
            # Try to compute physics columns if names match common presets
            if c.lower().startswith("areaa1"):
                # Need an approximate needle diameter (mm) for area
                d_mm = 0.838 if needle_size == 18 else 0.514
                row[c] = np.pi * (d_mm**2) / 4.0
            elif c.lower().startswith("flowq1"):
                d_mm = 0.838 if needle_size == 18 else 0.514
                area = np.pi * (d_mm**2) / 4.0
                row[c] = area * float(speed_in)
            elif c.lower().startswith("shears1"):
                d_mm = 0.838 if needle_size == 18 else 0.514
                row[c] = 8.0 * float(speed_in) / max(d_mm, 1e-6)
            elif c.lower().startswith("viscosn1"):
                d_mm = 0.838 if needle_size == 18 else 0.514
                shear = 8.0 * float(speed_in) / max(d_mm, 1e-6)
                K, n = 0.9, 0.06
                row[c] = K * (shear ** (n - 1.0))
            else:
                # If column is categorical and exists in df, fallback to first mode
                if c in df.columns:
                    row[c] = df[c].iloc[0]
                else:
                    row[c] = 0.0

    x_row = pd.DataFrame([row], columns=X_cols)
    pred_width = float(pipe.predict(x_row)[0])  # µm

    # Classification by difference
    internal_um = INTERNAL_DIAMETER.get(needle_size, 0.0)
    width_diff = pred_width - internal_um
    verdict = quality_bucket(width_diff)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Width (µm)", f"{pred_width:.1f}")
    c2.metric("Internal ID (µm)", f"{internal_um:.0f}")
    c3.metric("Width Difference (µm)", f"{width_diff:+.1f}")

    st.success(f"Print Quality: **{verdict}**")
    with st.expander("Show input row sent to model"):
        st.json(row)

# -------------------- footer --------------------
st.write("")
st.caption("Tip: use the formula tool to add Area/Flow/Shear/Viscosity columns if your CSV lacks them. Then retrain for better accuracy.")
