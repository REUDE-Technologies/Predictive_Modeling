import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import sklearn.exceptions
import shap
import os
from datetime import datetime
import warnings
import requests
from io import BytesIO
warnings.filterwarnings('ignore')

class UploadedGitHubFile:
    def __init__(self, content, name, filetype): 
        self.file = BytesIO(content)
        self.name = name
        self.type = filetype
        self.size = len(content)
    def read(self, *args, **kwargs):
        return self.file.read(*args, **kwargs)
    def seek(self, *args, **kwargs):
        return self.file.seek(*args, **kwargs)

def process_url(url):
    if "github.com" in url:
        try:
            # Handle folder URL (e.g., https://github.com/username/repo/tree/main/folder)
            if "tree" in url:
                parts = url.split("tree/")
                if len(parts) < 2:
                    st.error("Invalid GitHub folder URL format. Please include 'tree/' followed by the folder path.")
                    return None
                base_url = parts[0].rstrip('/')
                path = parts[1].lstrip('/')
                url_parts = base_url.split("/")
                if len(url_parts) < 5 or url_parts[2] != "github.com":
                    st.error("Unable to parse repository from URL.")
                    return None
                repo = f"{url_parts[3]}/{url_parts[4]}"
                folder_path = path.split('/', 1)[-1] if '/' in path else path
                api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
                response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
                if response.status_code == 200:
                    files = [item for item in response.json() if item['type'] == 'file' and item['name'].endswith('.csv')]
                    if not files:
                        st.warning("No .csv files found in the folder.")
                        return None
                    file_data = {}
                    for file in files:
                        file_response = requests.get(file['download_url'])
                        if file_response.status_code == 200:
                            file_ext = os.path.splitext(file['name'])[-1].lower()
                            file_data[file['name']] = (file_response.content, file_ext)
                    return file_data if file_data else None
                else:
                    st.error(f"Failed to fetch folder contents. Status code: {response.status_code}, Message: {response.text}, API URL: {api_url}")
                    return None
            # Handle raw file URL (e.g., https://raw.githubusercontent.com/username/repo/main/file.csv)
            elif "raw.githubusercontent.com" in url:
                file_name = url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext == ".csv":
                    response = requests.get(url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {url}")
                        return None
            # Handle blob URL (e.g., https://github.com/username/repo/blob/main/file.csv)
            elif "/blob/" in url:
                raw_url = url.replace("/blob/", "/raw/")
                file_name = raw_url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext == ".csv":
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            # Handle direct file URL (e.g., https://github.com/username/repo/filename.csv)
            elif len(url.split("/")) > 4 and url.split("/")[4] not in ["tree", "blob", "raw"]:
                base_parts = url.split("/")
                repo = f"{base_parts[3]}/{base_parts[4]}"
                file_path = "/".join(base_parts[5:])
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
                file_name = file_path.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext == ".csv":
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            else:
                st.warning("Unsupported GitHub URL format. Please use a folder URL with 'tree/' or a raw/blob file URL.")
                return None
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")
            return None
    return None

st.set_page_config(
    page_title="Width Prediction: ML + Physics + Vision",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for file upload functionality
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'files_submitted' not in st.session_state:
    st.session_state.files_submitted = False
if 'show_upload_area' not in st.session_state:
    st.session_state.show_upload_area = True
if 'upload_opened_by_plus' not in st.session_state:
    st.session_state.upload_opened_by_plus = False
if 'file_rename_mode' not in st.session_state:
    st.session_state.file_rename_mode = {}
if 'file_share_mode' not in st.session_state:
    st.session_state.file_share_mode = {}
if "share_all_mode" not in st.session_state:
    st.session_state.share_all_mode = False

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

def build_model(X: pd.DataFrame, y: pd.Series, model_type="Random Forest", fine_tune=False):
    """
    Builds and returns a pipeline (scaler + OHE + selected model) with optional fine-tuning.
    """
    # Create a copy to avoid modifying original data
    X_processed = X.copy()
    
    # Preprocess specific categorical columns to numeric in background
    if 'Thivex' in X_processed.columns:
        X_processed['Thivex'] = X_processed['Thivex'].astype(str).str.replace('%', '').astype(float)
    
    if 'Material' in X_processed.columns:
        material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
        X_processed['Material'] = X_processed['Material'].map(material_map)
    
    if 'Time Period' in X_processed.columns:
        time_map = {
            'Phase1: First 30 mins': 1,
            'Phase2: 30 mins to 60 min': 2, 
            'Phase3: After 60 mins': 3
        }
        X_processed['Time Period'] = X_processed['Time Period'].map(time_map)
    
    # Auto-detect numeric and categorical after preprocessing
    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_processed.columns if c not in numeric_cols]

    # Add NaN handling for SVR compatibility
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols),
    ])

    # Select model based on model_type with fine-tuning options
    if model_type == "Random Forest":
        if fine_tune:
            # Fine-tuned Random Forest - optimized for better performance
            model = Pipeline([
                ("pre", pre),
                ("rf", RandomForestRegressor(
                    n_estimators=800, random_state=42, n_jobs=-1,
                    max_depth=None, min_samples_split=3, min_samples_leaf=1,
                    max_features='sqrt', bootstrap=True, oob_score=True
                ))
            ])
        else:
            model = Pipeline([
                ("pre", pre),
                ("rf", RandomForestRegressor(
                    n_estimators=500, random_state=42, n_jobs=-1,
                    max_depth=None, min_samples_split=2, min_samples_leaf=1
                ))
            ])
    elif model_type == "XGBoost":
        if fine_tune:
            # Enhanced fine-tuned XGBoost with aggressive optimization
            xgb_configs = [
                {
                    'n_estimators': 1200, 'learning_rate': 0.03, 'max_depth': 10,
                    'min_child_weight': 2, 'subsample': 0.85, 'colsample_bytree': 0.85,
                    'reg_alpha': 0.05, 'reg_lambda': 0.8, 'random_state': 42, 'n_jobs': -1
                },
                {
                    'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 8,
                    'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.9,
                    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1
                },
                {
                    'n_estimators': 800, 'learning_rate': 0.08, 'max_depth': 12,
                    'min_child_weight': 1, 'subsample': 0.75, 'colsample_bytree': 0.75,
                    'reg_alpha': 0.2, 'reg_lambda': 1.2, 'random_state': 42, 'n_jobs': -1
                }
            ]
            best_score = -float('inf')
            best_config = None
            
            for config in xgb_configs:
                try:
                    test_xgb = XGBRegressor(**config)
                    test_pipe = Pipeline([("pre", pre), ("xgb", test_xgb)])
                    test_pipe.fit(X_processed, y)
                    score = test_pipe.score(X_processed, y)
                    if score > best_score:
                        best_score = score
                        best_config = config
                except:
                    continue
            
            if best_score > -float('inf'):
                model = Pipeline([
                    ("pre", pre),
                    ("xgb", XGBRegressor(**best_config))
                ])
            else:
                # Fallback to enhanced default
                model = Pipeline([
                    ("pre", pre),
                    ("xgb", XGBRegressor(
                        n_estimators=1200, learning_rate=0.03, max_depth=10,
                        min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
                        reg_alpha=0.05, reg_lambda=0.8, random_state=42, n_jobs=-1
                    ))
                ])
        else:
            model = Pipeline([
                ("pre", pre),
                ("xgb", XGBRegressor(
                    n_estimators=500, learning_rate=0.1, max_depth=6,
                    random_state=42, n_jobs=-1
                ))
            ])
    elif model_type == "SVR":
        if fine_tune:
            svr_configs = [
                {'kernel': 'rbf', 'C': 800.0, 'gamma': 'scale', 'epsilon': 0.01},
                {'kernel': 'rbf', 'C': 1000.0, 'gamma': 'auto', 'epsilon': 0.008},
                {'kernel': 'rbf', 'C': 1200.0, 'gamma': 'scale', 'epsilon': 0.005},
                {'kernel': 'rbf', 'C': 1500.0, 'gamma': 'auto', 'epsilon': 0.003},
                {'kernel': 'rbf', 'C': 600.0, 'gamma': 'scale', 'epsilon': 0.015},
                {'kernel': 'poly', 'C': 800.0, 'gamma': 'scale', 'epsilon': 0.01, 'degree': 2},
                {'kernel': 'poly', 'C': 1000.0, 'gamma': 'auto', 'epsilon': 0.008, 'degree': 3}
            ]
            best_score = -float('inf')
            best_config = None
            
            for config in svr_configs:
                try:
                    test_svr = SVR(**config, max_iter=3000, cache_size=2000, shrinking=True, tol=1e-4)
                    test_pipe = Pipeline([("pre", pre), ("svr", test_svr)])
                    test_pipe.fit(X_processed, y)
                    score = test_pipe.score(X_processed, y)
                    if score > best_score:
                        best_score = score
                        best_config = config
                except:
                    continue
            
            if best_score > -float('inf'):
                model = Pipeline([
                    ("pre", pre),
                    ("svr", SVR(**best_config, max_iter=3000, cache_size=2000, shrinking=True, tol=1e-4))
                ])
            else:
                # Fallback to aggressive default for RMSE ~60 µm
                model = Pipeline([
                    ("pre", pre),
                    ("svr", SVR(
                        kernel='rbf', C=1000.0, gamma='auto', epsilon=0.008,
                        max_iter=3000, cache_size=2000, shrinking=True, tol=1e-4
                    ))
                ])
        else:
            model = Pipeline([
                ("pre", pre),
                ("svr", SVR(
                    kernel='rbf', 
                    C=100.0,  # Increased for better regularization
                    gamma='scale',  # Better for scaled features
                    epsilon=0.05,  # Tighter tolerance
                    max_iter=2000,  # More iterations for convergence
                    cache_size=1000  # Larger cache for better performance
                ))
            ])
    elif model_type == "KNN":
        if fine_tune:
            # Enhanced fine-tuned KNN with better parameter optimization
            knn_configs = [
                {'n_neighbors': 3, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 20, 'p': 1},
                {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2},
                {'n_neighbors': 7, 'weights': 'distance', 'algorithm': 'ball_tree', 'leaf_size': 25, 'p': 1},
                {'n_neighbors': 4, 'weights': 'uniform', 'algorithm': 'kd_tree', 'leaf_size': 30, 'p': 2}
            ]
            best_score = -float('inf')
            best_config = None
            
            for config in knn_configs:
                try:
                    test_knn = KNeighborsRegressor(**config)
                    test_pipe = Pipeline([("pre", pre), ("knn", test_knn)])
                    test_pipe.fit(X_processed, y)
                    score = test_pipe.score(X_processed, y)
                    if score > best_score:
                        best_score = score
                        best_config = config
                except:
                    continue
            
            if best_score > -float('inf'):
                model = Pipeline([
                    ("pre", pre),
                    ("knn", KNeighborsRegressor(**best_config))
                ])
            else:
                # Fallback to enhanced default
                model = Pipeline([
                    ("pre", pre),
                    ("knn", KNeighborsRegressor(
                        n_neighbors=5, weights='distance', 
                        algorithm='auto', leaf_size=25, p=1
                    ))
                ])
        else:
            model = Pipeline([
                ("pre", pre),
                ("knn", KNeighborsRegressor(n_neighbors=5, weights='uniform'))
            ])
    else:
        # Default to Random Forest
        model = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(
                n_estimators=500, random_state=42, n_jobs=-1,
                max_depth=None, min_samples_split=2, min_samples_leaf=1
            ))
        ])
    
    return model, numeric_cols, categorical_cols, X_processed

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

# -------------------- File Upload Section --------------------
if st.session_state.show_upload_area:
    # Add CSS to hide sidebar and adjust layout for upload page
    st.markdown("""
    <style>
    /* Hide sidebar when upload area is shown */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    /* Adjust main content for upload page */
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    /* Keep content visible; only sidebar is hidden. */
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .fixed-header {
        top: 18px;
        left: 18px;
        z-index: 1001;
        background: #fff;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(44, 62, 80, 0.10);
        padding: 16px 28px 14px 22px;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: flex-start;
        min-width: 500px;
        max-width: 525px;
        border: 1px solid #e0e0e0;
        height: 60px;
    }
    .fixed-header h1 {
        color: #2E86C1;
        margin: 0 0 2px 0;
        font-size: 1.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        line-height: 1.1;
        font-weight: 700;
    }
    .fixed-header .rocket-icon {
        font-size: 1.7rem;
        line-height: 1;
    }
    .fixed-header p {
        color: #666;
        margin: 0;
        font-size: 0.98rem;
        line-height: 1.2;
        font-weight: 400;
    }
    .main .block-container {
        padding-top: 40px !important;
    }
    .upload-section {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    .upload-section.active {
        border-color: #007bff;
        background: #f0f8ff;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
    }
    .upload-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .file-preview-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .file-preview-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #007bff;
        transform: translateY(-1px);
    }
    .file-actions {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }
    .file-action-btn {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .file-action-btn:hover {
        background: #e9ecef;
        border-color: #adb5bd;
    }
    .file-action-btn.primary {
        background: #007bff;
        color: white;
        border-color: #007bff;
    }
    .file-action-btn.primary:hover {
        background: #0056b3;
    }
    .file-action-btn.danger {
        background: #dc3545;
        color: white;
        border-color: #dc3545;
    }
    .file-action-btn.danger:hover {
        background: #c82333;
    }
    .upload-zone {
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 30px;
        text-align: center;
        background: white;
        transition: all 0.2s;
        cursor: pointer;
    }
    .upload-zone:hover {
        border-color: #007bff;
        background: #f8f9ff;
        transform: scale(1.02);
    }
    .upload-zone.dragover {
        border-color: #007bff;
        background: #e3f2fd;
    }
    .file-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .file-stats h6 {
        color: rgba(255,255,255,0.9);
        margin-bottom: 8px;
    }
    .file-stats .stat-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .file-stats .stat-label {
        font-size: 12px;
        opacity: 0.8;
    }
    .bulk-actions {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .bulk-actions h6 {
        color: #495057;
        margin-bottom: 10px;
    }
    .tab-content {
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .file-type-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .file-type-badge.csv {
        background: #d4edda;
        color: #155724;
    }

    .github-section-enhanced {
        background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #b3d8fd;
        box-shadow: 0 2px 8px rgba(0, 123, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fixed-header">
        <h1><span class="rocket-icon">🚀</span> Vision-based Width Prediction </h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #2E86C1; margin-bottom: 30px;'>📁 File Management</h3>", unsafe_allow_html=True)
    github_col, upload_col = st.columns([0.5, 0.5])
    
    with github_col:
        st.markdown("""
        <div class="github-section-enhanced">
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 8px;'>
                <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='22' style='margin-right: 4px;'/>
                <span style='font-size: 1.08rem; font-weight: 700; color: #24292f;'>GitHub</span>
                <span style='font-size: 0.98rem; color: #2980b9; margin-left: 6px;'>(<a style='color:#2980b9; text-decoration:underline; cursor:pointer;' href='#'>.csv</a>)</span>
            </div>
            <div style='font-size: 0.98rem; color: #444; margin-bottom: 12px;'>
                Paste a <b>GitHub <span style='font-weight:700;'>raw/blob/folder URL</span></b> to fetch files.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.get("clear_github_url_input", False):
            st.session_state.github_url_input = ""
            st.session_state.clear_github_url_input = False
        github_col, fetch_col  = st.columns([5, 1])
        with fetch_col:
            fetch_github = st.button("Fetch", key="fetch_github_btn", use_container_width=True)
        with github_col:
            github_url = st.text_input("GitHub URL (raw, blob, or folder)", key="github_url_input", label_visibility="collapsed", placeholder="e.g. https://github.com/user/repo/blob/main/data.csv")
        if fetch_github and github_url:
            result = process_url(github_url)
            if result:
                existing_names = [f.name for f in st.session_state.uploaded_files]
                for file_name, (file_content, file_ext) in result.items():
                    if file_name not in existing_names:
                        filetype = "text/csv" if file_ext == ".csv" else "application/octet-stream"
                        file_like = UploadedGitHubFile(file_content, file_name, filetype)
                        st.session_state.uploaded_files.append(file_like)
                st.session_state.clear_github_url_input = True
                st.rerun()
            else:
                st.warning("No valid .csv files found at the provided URL.")
    
    with upload_col:
        uploaded_files = st.file_uploader(
            "Choose files to upload", 
            type=["csv"], 
            key="desktop_uploader", 
            label_visibility="collapsed", 
            accept_multiple_files=True,
            help="Drag and drop files here or click to browse")
    
    # Process uploaded files
    if uploaded_files:
        new_files_added = False
        existing_names = [f.name for f in st.session_state.uploaded_files]
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in existing_names:
                st.session_state.uploaded_files.append(uploaded_file)
                new_files_added = True
        if new_files_added:
            st.rerun()
    
    # Show file preview section if files are uploaded
    if st.session_state.uploaded_files:
        st.markdown("<h4 style='margin-top: 0px; color: #495057;'>📋 File Preview & Management</h4>", unsafe_allow_html=True)
        
        # Two-column layout for file preview
        preview_col, actions_col = st.columns([0.7, 0.3])
        
        with preview_col:
            st.markdown("<h5 style='color: #6c757d; margin-bottom: 15px;'>📎 Uploaded Files</h5>", unsafe_allow_html=True)
            for i, file in enumerate(st.session_state.uploaded_files):
                file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                file_type_badge = f"<span class='file-type-badge {file_ext}'>{file_ext}</span>"
                
                # Use columns to align file name/details and action buttons in a single row
                file_cols = st.columns([12, 1, 1, 1, 1, 1])
                with file_cols[0]:
                    st.markdown(f"""
                    <div style="font-weight: 600; color: #495057;">
                        📄 {file.name}
                        <span style="font-size: 12px; color: #6c757d; margin-left: 10px;">
                            Size: {file.size / (1024*1024):.1f} MB | Type: {file.type or 'Unknown'} {file_type_badge}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                with file_cols[1]:
                    if st.button("🔍", key=f"preview_btn_{i}", use_container_width=True, help="Quick Preview"):
                        st.session_state[f"preview_mode_{i}"] = not st.session_state.get(f"preview_mode_{i}", False)
                        st.rerun()
                with file_cols[2]:
                    if st.button("✏️", key=f"rename_btn_{i}", use_container_width=True, help="Rename"):
                        st.session_state.file_rename_mode[i] = not st.session_state.file_rename_mode.get(i, False)
                        st.rerun()
                with file_cols[3]:
                    if st.button("➦", key=f"share_btn_{i}", use_container_width=True, help="Share"):
                        st.session_state.file_share_mode[i] = not st.session_state.file_share_mode.get(i, False)
                        st.rerun()
                with file_cols[4]:
                    file.seek(0)
                    st.download_button(
                        label="⬇️",
                        data=file.read(),
                        file_name=file.name,
                        mime=file.type or "application/octet-stream",
                        key=f"download_btn_{i}",
                        use_container_width=True,
                        help="Download"
                    )
                    file.seek(0)
                with file_cols[5]:
                    if st.button("🗑️", key=f"remove_btn_{i}", use_container_width=True, help="Remove"):
                        st.session_state.uploaded_files.pop(i)
                        st.rerun()
                
                # Quick Preview UI
                if st.session_state.get(f"preview_mode_{i}", False):
                    with st.expander("Quick Preview", expanded=True):
                        file.seek(0)
                        file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                        if file_ext == "csv":
                            try:
                                df = pd.read_csv(file, nrows=5)
                                st.dataframe(df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not preview CSV: {e}")
                        else:
                            st.warning("Preview not supported for this file type.")
                        file.seek(0)
                
                if st.session_state.file_rename_mode.get(i, False):
                    with st.container():
                        st.markdown("**✏️ Rename File**")
                        col_rename1, col_rename2, col_rename3 = st.columns([2, 1, 1])
                        with col_rename1:
                            new_name = st.text_input(
                                "New file name:", 
                                value=file.name,
                                key=f"rename_input_{i}"
                            )
                        with col_rename2:
                            if st.button("✅ Save", key=f"save_rename_{i}", use_container_width=True):
                                if new_name and new_name != file.name:
                                    file.name = new_name
                                    st.success(f"File renamed to: {new_name}")
                                st.session_state.file_rename_mode[i] = False
                                st.rerun()
                        with col_rename3:
                            if st.button("❌ Cancel", key=f"cancel_rename_{i}", use_container_width=True):
                                st.session_state.file_rename_mode[i] = False
                                st.rerun()
                
                if st.session_state.file_share_mode.get(i, False):
                    with st.container():
                        st.markdown("**➦ Share File**")
                        share_option = st.selectbox(
                            "Select sharing option:",
                            ["Public Link", "Email"],
                            key=f"share_option_{i}"
                        )
                        if share_option == "Email":
                            st.info(f"🔧 {share_option} sharing is a Work in Progress.")
                        elif share_option in ["Public Link"]:
                            st.info(f"🔧 {share_option} sharing is a Work in Progress.")
        
        with actions_col:
            total_files = len(st.session_state.uploaded_files)
            total_size = sum(f.size for f in st.session_state.uploaded_files) / (1024*1024) # MB
            
            st.markdown(f"""
            <div class="file-stats">
                <h6>📊 File Statistics</h6>
                <div class="stat-value">{total_files}</div>
                <div class="stat-label">Total Files</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="file-stats">
                <h6>💾 Storage</h6>
                <div class="stat-value">{total_size:.1f}</div>
                <div class="stat-label">Total Size (MB)</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("➦ Share All", use_container_width=True):
                st.session_state.share_all_mode = not st.session_state.get("share_all_mode", False)

            if st.session_state.get("share_all_mode", False):
                st.info(f"🔧 Share All Files is a Work in Progress.")
            
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.uploaded_files.clear()
                st.rerun()

            # Submit files button with enhanced styling
            if st.button("✅ Submit Files for Analysis", type="primary", use_container_width=True):
                st.session_state.files_submitted = True
                st.session_state.show_upload_area = False
                st.session_state.upload_opened_by_plus = False
                st.rerun()
    else:
        # Empty state
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 48px; margin-bottom: 20px;">📁</div>
            <h4 style="color: #6c757d; margin-bottom: 10px;">No files uploaded yet</h4>
            <p style="color: #adb5bd; margin-bottom: 20px;">Upload your CSV files to begin analysis</p>
            <p style="font-size: 12px; color: #ced4da;">Supported formats: .csv</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Main Application (existing functionality) --------------------
# Only show main application when files are submitted and upload area is hidden
if st.session_state.files_submitted and not st.session_state.show_upload_area:
    # Ensure sidebar is shown on main page
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: block !important; }
    </style>
    """, unsafe_allow_html=True)

    # Add the header and upload button in a single row with proper alignment
    col1, col2 = st.columns([11, 1])
    with col1:
        st.markdown("""
        <div class="fixed-header">
            <h1><span class="rocket-icon">🚀</span> Vision-based Width Prediction </h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Add CSS to control multiselect height
    st.markdown("""
    <style>
    /* Control multiselect widget height to prevent layout shifts */
    .stMultiSelect > div > div {
        max-height: 40px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
    }
    .stMultiSelect > div > div > div {
        max-height: 35px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        white-space: nowrap !important;
    }
    /* Ensure the multiselect container allows horizontal scrolling */
    .stMultiSelect [data-baseweb="select"] {
        max-height: 40px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
    }
    /* Make selected tags container horizontally scrollable */
    .stMultiSelect [data-baseweb="select"] > div {
        max-height: 35px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        white-space: nowrap !important;
    }
    /* Style individual tags */
    .stMultiSelect [data-baseweb="tag"] {
        font-size: 0.8em !important;
        padding: 2px 6px !important;
        margin: 1px !important;
        flex-shrink: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="margin-top: 18px;">', unsafe_allow_html=True)
        if st.button("➕ UPLOAD", type="primary", use_container_width=True, help="Upload or manage files"):
            st.session_state.show_upload_area = True
            st.session_state.files_submitted = False
            st.session_state.upload_opened_by_plus = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

    # -------------------- Data Selection and Model Configuration in 4 columns --------------------
    col1, col2, col3, col4, col5 = st.columns([1,1,1,0.9,0.5])

    with col1:
        st.markdown("### 📂 Data Selection")
        if st.session_state.uploaded_files:
            file_options = ["None"] + [f.name for f in st.session_state.uploaded_files if f.name.endswith('.csv')]
            selected_file = st.selectbox("Select CSV file", options=file_options, key="file_selector")
            
            if selected_file != "None":
                # Find the selected file
                selected_file_obj = None
                for file in st.session_state.uploaded_files:
                    if file.name == selected_file:
                        selected_file_obj = file
                        break
                
                if selected_file_obj:
                    try:
                        selected_file_obj.seek(0)
                        df = pd.read_csv(selected_file_obj)
                        selected_file_obj.seek(0)
                        st.caption(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                        st.stop()
                else:
                    st.error("Selected file not found")
                    st.stop()
            else:
                st.info("Please select a CSV file to begin.")
                st.stop()
        else:
            st.info("No files uploaded. Please upload files first.")
            st.stop()

    with col2:
        st.markdown("### 🎯 Target Selection")
        if 'df' in locals():
            all_cols = df.columns.tolist()
            y_col = st.selectbox("Target (y)", options=all_cols, index=all_cols.index('Width (um)') if 'Width (um)' in all_cols else 0)
        else:
            st.info("Please select a file first.")

    with col3:
        st.markdown("### 🔧 Features Selection")
        if 'df' in locals():
            all_cols = df.columns.tolist()
            X_cols = st.multiselect("Features (X)", options=[c for c in all_cols if c != y_col], max_selections=None, help="Select multiple features")
        else:
            st.info("Please select a file first.")

    with col4:
        st.markdown("### 🚀 Model Training")
        
        # Initialize model selection in session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "Random Forest"
        
        # Model selection dropdown
        selected_model = st.selectbox(
            "Model Type",
            options=["Random Forest", "XGBoost", "SVR", "KNN"],
            index=["Random Forest", "XGBoost", "SVR", "KNN"].index(st.session_state.selected_model) if st.session_state.selected_model in ["Random Forest", "XGBoost", "SVR", "KNN"] else 0,
            help="Choose the machine learning model for prediction",
            key="model_selector"
        )
        
        # Update session state
        st.session_state.selected_model = selected_model
        
        # Initialize test size in session state
        if 'test_size' not in st.session_state:
            st.session_state.test_size = 0.2
        
        # Test size input
        test_size = st.slider(
            "Test Size", 
            min_value=0.1, 
            max_value=0.5, 
            value=st.session_state.test_size, 
            step=0.05,
            help="Fraction of data to use for testing (0.1 = 10%, 0.2 = 20%, etc.)",
            key="test_size_slider"
        )
        
        # Update session state
        st.session_state.test_size = test_size
        
        # Fine-tuning toggle
        if 'fine_tune' not in st.session_state:
            st.session_state.fine_tune = False
        
        fine_tune = st.checkbox(
            "🔧 Enable Fine-tuning", 
            value=st.session_state.fine_tune,
            help="Enable advanced hyperparameter optimization for better performance (slower training)",
            key="fine_tune_checkbox"
        )
        st.session_state.fine_tune = fine_tune
        
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        
        # Custom CSS to increase button height
        st.markdown("""
            <style>
            .stButton > button {
                height: 70px !important;
                font-size: 16px !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        run_training = st.button("Train & Analyze", type="primary", use_container_width=True)
        
        # Check if model or features have changed and show retrain message
        if 'previous_model' not in st.session_state:
            st.session_state.previous_model = selected_model
        if 'previous_features' not in st.session_state:
            st.session_state.previous_features = X_cols if X_cols else []
        
        # Check for changes
        model_changed = st.session_state.previous_model != selected_model
        features_changed = st.session_state.previous_features != (X_cols if X_cols else [])
        
        if model_changed or features_changed:
            st.info("🔄 Please retrain the model")
        
        # Update previous values
        st.session_state.previous_model = selected_model
        st.session_state.previous_features = X_cols if X_cols else []

    # -------------------- Tabs for Plot and Data --------------------
    if 'df' in locals() and 'y_col' in locals() and 'X_cols' in locals():
        tab1, tab2, tab3 = st.tabs(["📊 Plot", "📋 Data", "🎯 Prediction"])
    
    with tab1:
        # Initialize session state for plot tab
        if 'plot_training_results' not in st.session_state:
            st.session_state.plot_training_results = None
        if 'plot_model_metrics' not in st.session_state:
            st.session_state.plot_model_metrics = None
        
        # Show training results if model is trained
        if run_training:
            # Check if parameters have changed and reset session state if needed
            current_config = {
                'selected_file': selected_file,
                'y_col': y_col,
                'X_cols': sorted(X_cols) if X_cols else []
            }
            
            # Initialize or get previous config
            if 'previous_training_config' not in st.session_state:
                st.session_state.previous_training_config = None
            
            # If config has changed, reset all tab session states
            if st.session_state.previous_training_config != current_config:
                # Clear all tab session states
                if 'plot_training_results' in st.session_state:
                    del st.session_state.plot_training_results
                if 'plot_model_metrics' in st.session_state:
                    del st.session_state.plot_model_metrics
                if 'prediction_results' in st.session_state:
                    del st.session_state.prediction_results
                if 'last_prediction_inputs' in st.session_state:
                    del st.session_state.last_prediction_inputs
                if 'data_tab_formula_inputs' in st.session_state:
                    del st.session_state.data_tab_formula_inputs
                
                # Reset prediction input defaults
                if 'pred_needle_size' in st.session_state:
                    del st.session_state.pred_needle_size
                if 'pred_material' in st.session_state:
                    del st.session_state.pred_material
                if 'pred_thivex' in st.session_state:
                    del st.session_state.pred_thivex
                if 'pred_time' in st.session_state:
                    del st.session_state.pred_time
                if 'pred_pressure' in st.session_state:
                    del st.session_state.pred_pressure
                if 'pred_speed' in st.session_state:
                    del st.session_state.pred_speed
                
                # Update the previous config
                st.session_state.previous_training_config = current_config
                
                # Show a message that data has been refreshed
                # st.success("🔄 **Data refreshed!** New model training with updated parameters.")
                
                # Set a flag to show feature change prompt
                st.session_state.show_feature_change_prompt = True
            
            # Validate that features are selected
            if not X_cols:
                st.error("❌ **No features selected!** Please select at least one feature in the 'Features (X)' column before training.")
                st.stop()
            
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
            
            try:
                pipe, num_cols, cat_cols, X_processed = build_model(X_tr, y_tr, selected_model, fine_tune)
                pipe.fit(X_processed, y_tr)
            except Exception as e:
                st.error(f"❌ **Model training failed for {selected_model}**: {str(e)}")
                st.info("💡 **Tips**: Try a different model or check your data quality.")
                st.stop()
            
            # Process test data the same way as training data
            X_te_processed = X_te.copy()
            if 'Thivex' in X_te_processed.columns:
                X_te_processed['Thivex'] = X_te_processed['Thivex'].astype(str).str.replace('%', '').astype(float)
            if 'Material' in X_te_processed.columns:
                material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
                X_te_processed['Material'] = X_te_processed['Material'].map(material_map)
            if 'Time Period' in X_te_processed.columns:
                time_map = {
                    'Phase1: First 30 mins': 1,
                    'Phase2: 30 mins to 60 min': 2, 
                    'Phase3: After 60 mins': 3
                }
                X_te_processed['Time Period'] = X_te_processed['Time Period'].map(time_map)
            
            y_hat = pipe.predict(X_te_processed)
            r2 = r2_score(y_te, y_hat)
            try:
                rmse = float(mean_squared_error(y_te, y_hat, squared=False))
            except TypeError:
                rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))

            # Save in session
            st.session_state["trained_pipeline"] = pipe
            st.session_state["selected_model"] = selected_model
            st.session_state["X_cols"] = X_cols
            st.session_state["y_col"] = y_col
            st.session_state["numeric"] = num_cols
            st.session_state["categorical"] = cat_cols
            st.session_state["data"] = data
            
            # Add model performance to session state for comparison
            if 'model_performances' not in st.session_state:
                st.session_state.model_performances = {}
            st.session_state.model_performances[selected_model] = {
                'r2': r2,
                'rmse': rmse,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save plot-specific results
            st.session_state.plot_training_results = {
                'data': data,
                'X_cols': X_cols,
                'y_col': y_col,
                'num_cols': num_cols,
                'cat_cols': cat_cols,
                'pipe': pipe,
                'X_processed': X_processed,
                'y_vector': y
            }
            st.session_state.plot_model_metrics = {
                'r2': r2,
                'rmse': rmse
            }

            # ---- Enhanced Model Performance Metrics ----
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: white; text-align: center; margin: 0 0 20px 0;">🎯 Model Performance Dashboard</h2>
                <div style="display: flex; justify-content: space-around; align-items: center;">
                    <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; min-width: 150px;">
                        <h3 style="color: white; margin: 0; font-size: 14px;">R² Score (Test)</h3>
                        <p style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{:.3f}</p>
                        <p style="color: #e0e0e0; font-size: 12px; margin: 0;">Coefficient of Determination</p>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; min-width: 150px;">
                        <h3 style="color: white; margin: 0; font-size: 14px;">RMSE (Test)</h3>
                        <p style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{:.1f} µm</p>
                        <p style="color: #e0e0e0; font-size: 12px; margin: 0;">Root Mean Square Error</p>
                    </div>
                </div>
            </div>
            """.format(r2, rmse), unsafe_allow_html=True)
            
            # Additional performance insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📊 **Model Quality**: {'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.7 else 'Fair'}")
            with col2:
                st.info(f"🎯 **Accuracy**: {'High' if rmse < 50 else 'Medium' if rmse < 100 else 'Low'}")
            with col3:
                st.info(f"📈 **Reliability**: {'Very High' if r2 > 0.95 else 'High' if r2 > 0.85 else 'Moderate'}")
            
            # Model comparison section
            if len(st.session_state.model_performances) > 1:
                st.markdown("### 📊 Model Performance Comparison")
                
                # Show fine-tuning status
                if fine_tune:
                    st.success("🔧 **Fine-tuning enabled** - Advanced hyperparameter optimization applied!")
                comparison_data = []
                for model_name, perf in st.session_state.model_performances.items():
                    comparison_data.append({
                        'Model': model_name,
                        'R² Score': f"{perf['r2']:.3f}",
                        'RMSE (µm)': f"{perf['rmse']:.1f}",
                        'Last Trained': perf['timestamp']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Highlight best performing model
                best_model = max(st.session_state.model_performances.items(), key=lambda x: x[1]['r2'])
                st.success(f"🏆 **Best performing model**: {best_model[0]} (R² = {best_model[1]['r2']:.3f})")

            # ---- 2x2 Grid Layout for Plots ----
            # st.markdown("### 📊 Model Analysis Dashboard")
            
            # Create 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot 1: Correlation Heatmap
                st.markdown("#### 📈 Correlation Heatmap")
                # Use processed data for correlation (includes converted categorical variables)
                # Add target variable to processed features for correlation analysis
                processed_for_corr = X_processed[X_cols].copy()
                processed_for_corr[y_col] = y  # Add target variable back
                num_for_corr = processed_for_corr.select_dtypes(include=[np.number])
                if num_for_corr.shape[1] >= 2:
                    plt.figure(figsize=(6, 5))
                    corr = num_for_corr.corr()
                    # Create heatmap with annotations
                    sns.heatmap(corr, 
                               annot=True,  # Show values
                               fmt='.2f',   # Format to 2 decimal places
                               cmap="coolwarm", 
                               center=0,
                               square=True,  # Make it square
                               cbar_kws={"shrink": .8},
                               annot_kws={"size": 8})  # Smaller font for annotations
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
                else:
                    st.info("Not enough numeric columns for correlation heatmap.")
            
            with col2:
                # Plot 2: Feature Importance Table
                # Extract trained model and one-hot feature names
                pre = pipe.named_steps["pre"]
                # build full feature names: scaled numeric + OHE cats
                num_names = num_cols
                cat_expanded = []
                
                # Only attempt to read encoder feature names if we actually
                # have categorical columns and the encoder has been fitted
                if cat_cols:
                    try:
                        ohe = pre.named_transformers_["cat"]
                        if (
                            hasattr(ohe, "get_feature_names_out")
                            and hasattr(ohe, "categories_")
                        ):
                            cat_expanded = ohe.get_feature_names_out(cat_cols).tolist()
                    except Exception:
                        # If categorical transformer is not available, use original names
                        cat_expanded = cat_cols
                
                full_names = num_names + cat_expanded
                
                # Handle feature importance for different model types
                fi = None  # Initialize fi variable
                if selected_model in ["Random Forest", "XGBoost"]:
                    # Get the correct model step name
                    model_step_name = "rf" if selected_model == "Random Forest" else "xgb"
                    if model_step_name in pipe.named_steps:
                        model_step = pipe.named_steps[model_step_name]
                        importances = model_step.feature_importances_
                        fi = pd.DataFrame({"feature": full_names, "importance": importances}).sort_values("importance", ascending=False)
                        st.markdown("#### 📋 Top Features")
                        st.dataframe(fi.head(10), use_container_width=True, height=410)
                    else:
                        st.markdown("#### 📋 Model Info")
                        st.info(f"Model step '{model_step_name}' not found in pipeline.")
                else:
                    st.markdown("#### 📋 Model Info")
                    st.info(f"Feature importance not available for {selected_model}. This model focuses on pattern recognition rather than feature ranking.")
                
            # Second row
            col3, col4 = st.columns(2)
            
            with col4:
                # Plot 4: SHAP Summary (only for tree-based models)
                if selected_model in ["Random Forest", "XGBoost"]:
                    st.markdown("#### 🔎 SHAP Summary")
                    try:
                        # Build a small background set using processed data
                        sample_X = X_processed.sample(min(100, len(X_processed)), random_state=42)
                        # Transform sample through preprocessor to get model input
                        bg = pre.transform(sample_X)
                        # Get the correct model step name
                        model_step_name = "rf" if selected_model == "Random Forest" else "xgb"
                        if model_step_name in pipe.named_steps:
                            model_step = pipe.named_steps[model_step_name]
                            explainer = shap.TreeExplainer(model_step)
                            shap_values = explainer.shap_values(pre.transform(sample_X))
                            try:
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                            except Exception:
                                pass
                            shap.summary_plot(shap_values, features=pre.transform(sample_X), feature_names=full_names, show=False)
                            fig = plt.gcf()
                            fig.set_size_inches(5, 4)
                            st.pyplot(fig, bbox_inches='tight')
                            plt.close(fig)
                        else:
                            st.warning(f"Model step '{model_step_name}' not found for SHAP analysis.")
                    except Exception as e:
                        st.warning(f"SHAP analysis failed: {str(e)}. This might be due to data characteristics.")

                else:
                    st.markdown("#### 🔎 Model Analysis")
                    st.info(f"SHAP analysis not available for {selected_model}. This model uses different interpretability methods.")
            
            with col3:
                # Plot 3: Feature Importance Barplot (only for tree-based models)
                if selected_model in ["Random Forest", "XGBoost"] and fi is not None:
                    st.markdown("#### 🌲 Feature Importance")
                    plt.figure(figsize=(5, 4))
                    sns.barplot(x="importance", y="feature", data=fi.head(15))
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
                else:
                    st.markdown("#### 📊 Model Performance")
                    st.info(f"Feature importance visualization not available for {selected_model}. This model focuses on overall prediction accuracy.")

        elif st.session_state.plot_training_results:
            # Render last trained results so Plot tab persists after Predict
            try:
                saved = st.session_state.plot_training_results
                data = saved['data']
                X_cols = saved['X_cols']
                y_col = saved['y_col']
                num_cols = saved['num_cols']
                cat_cols = saved['cat_cols']
                pipe = saved['pipe']
                X_processed = saved['X_processed']
                y = saved['y_vector']

                metrics = st.session_state.get('plot_model_metrics')
                if metrics:
                    r2 = metrics.get('r2', float('nan'))
                    rmse = metrics.get('rmse', float('nan'))
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <h2 style="color: white; text-align: center; margin: 0 0 20px 0;">🎯 Model Performance Dashboard</h2>
                        <div style="display: flex; justify-content: space-around; align-items: center;">
                            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; min-width: 150px;">
                                <h3 style="color: white; margin: 0; font-size: 14px;">R² Score (Test)</h3>
                                <p style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{:.3f}</p>
                                <p style="color: #e0e0e0; font-size: 12px; margin: 0;">Coefficient of Determination</p>
                            </div>
                            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; min-width: 150px;">
                                <h3 style="color: white; margin: 0; font-size: 14px;">RMSE (Test)</h3>
                                <p style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{:.1f} µm</p>
                                <p style="color: #e0e0e0; font-size: 12px; margin: 0;">Root Mean Square Error</p>
                            </div>
                        </div>
                    </div>
                    """.format(r2, rmse), unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📊 **Model Quality**: {'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.7 else 'Fair'}")
                    with col2:
                        st.info(f"🎯 **Accuracy**: {'High' if rmse < 50 else 'Medium' if rmse < 100 else 'Low'}")
                    with col3:
                        st.info(f"📈 **Reliability**: {'Very High' if r2 > 0.95 else 'High' if r2 > 0.85 else 'Moderate'}")

                # 2x2 Grid
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📈 Correlation Heatmap")
                    processed_for_corr = X_processed[X_cols].copy()
                    processed_for_corr[y_col] = y
                    num_for_corr = processed_for_corr.select_dtypes(include=[np.number])
                    if num_for_corr.shape[1] >= 2:
                        plt.figure(figsize=(6, 5))
                        corr = num_for_corr.corr()
                        sns.heatmap(corr, annot=True, fmt='.2f', cmap="coolwarm", center=0, square=True,
                                   cbar_kws={"shrink": .8}, annot_kws={"size": 8})
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.close()
                    else:
                        st.info("Not enough numeric columns for correlation heatmap.")

                with col2:
                    # Plot 2: Feature Importance Table
                    # Extract trained model and one-hot feature names
                    pre = pipe.named_steps["pre"]
                    # build full feature names: scaled numeric + OHE cats
                    num_names = num_cols
                    cat_expanded = []
                    
                    # Only attempt to read encoder feature names if we actually
                    # have categorical columns and the encoder has been fitted
                    if cat_cols:
                        try:
                            ohe = pre.named_transformers_["cat"]
                            if (
                                hasattr(ohe, "get_feature_names_out")
                                and hasattr(ohe, "categories_")
                            ):
                                cat_expanded = ohe.get_feature_names_out(cat_cols).tolist()
                        except Exception:
                            # If categorical transformer is not available, use original names
                            cat_expanded = cat_cols
                    
                    full_names = num_names + cat_expanded
                    
                    # Handle feature importance for different model types
                    fi = None  # Initialize fi variable
                    if selected_model in ["Random Forest", "XGBoost"]:
                        # Get the correct model step name
                        model_step_name = "rf" if selected_model == "Random Forest" else "xgb"
                        try:
                            if model_step_name in pipe.named_steps:
                                model_step = pipe.named_steps[model_step_name]
                                importances = model_step.feature_importances_
                                fi = pd.DataFrame({"feature": full_names, "importance": importances}).sort_values("importance", ascending=False)
                                st.markdown("#### 📋 Top Features")
                                st.dataframe(fi.head(10), use_container_width=True, height=410)

                        except Exception as e:
                            st.markdown("#### 📋 Model Info")
                            st.error(f"❌ **Error accessing model**: {str(e)}. Please train the model first.")
                    else:
                        st.markdown("#### 📋 Model Info")
                        st.info(f"Feature importance not available for {selected_model}. This model focuses on pattern recognition rather than feature ranking.")

                col3, col4 = st.columns(2)

                with col4:
                    # Plot 4: SHAP Summary (only for tree-based models)
                    if selected_model in ["Random Forest", "XGBoost"]:
                        st.markdown("#### 🔎 SHAP Summary")
                        try:
                            # Build a small background set using processed data
                            sample_X = X_processed.sample(min(100, len(X_processed)), random_state=42)
                            # Transform sample through preprocessor to get model input
                            bg = pre.transform(sample_X)
                            # Get the correct model step name
                            model_step_name = "rf" if selected_model == "Random Forest" else "xgb"
                            if model_step_name in pipe.named_steps:
                                model_step = pipe.named_steps[model_step_name]
                                explainer = shap.TreeExplainer(model_step)
                                shap_values = explainer.shap_values(pre.transform(sample_X))
                                try:
                                    st.set_option('deprecation.showPyplotGlobalUse', False)
                                except Exception:
                                    pass
                                shap.summary_plot(shap_values, features=pre.transform(sample_X), feature_names=full_names, show=False)
                                fig = plt.gcf()
                                fig.set_size_inches(5, 4)
                                st.pyplot(fig, bbox_inches='tight')
                                plt.close(fig)
                        except Exception as e:
                            st.warning(f"SHAP analysis failed: {str(e)}. This might be due to data characteristics.")
                    else:
                        st.markdown("#### 🔎 Model Analysis")
                        st.info(f"SHAP analysis not available for {selected_model}. This model uses different interpretability methods.")

                with col3:
                    # Plot 3: Feature Importance Barplot (only for tree-based models)
                    if selected_model in ["Random Forest", "XGBoost"] and fi is not None:
                        st.markdown("#### 🌲 Feature Importance")
                        plt.figure(figsize=(5, 4))
                        sns.barplot(x="importance", y="feature", data=fi.head(15))
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.close()
                    else:
                        st.markdown("#### 📊 Model Performance")
                        st.info(f"Feature importance visualization not available for {selected_model}. This model focuses on overall prediction accuracy.")
            except Exception as e:
                st.error(f"❌ **Error displaying model results**: {str(e)}. Please retrain the model to fix this issue.")
                st.info("💡 **Tip**: This error usually occurs when switching between different model types. Please train the model again with your current selection.")

    with tab2:
        # Initialize session state for data tab
        if 'data_tab_formula_inputs' not in st.session_state:
            st.session_state.data_tab_formula_inputs = {
                'needle_mm_col': '',
                'speed_col': '',
                'new_col_name': '',
                'formula_choice': 'None',
                'custom_expr': ''
            }
        
        # Get data from session state or local variables
        current_df = None
        current_all_cols = []
        current_X_cols = []
        
        if 'data' in st.session_state:
            current_df = st.session_state['data']
            current_all_cols = current_df.columns.tolist()
        elif 'df' in locals():
            current_df = df
            current_all_cols = df.columns.tolist()
        
        if 'X_cols' in st.session_state:
            current_X_cols = st.session_state['X_cols']
        elif 'X_cols' in locals():
            current_X_cols = X_cols
        
        # Initialize session state with proper defaults if we have data
        if current_all_cols and not st.session_state.data_tab_formula_inputs['needle_mm_col']:
            st.session_state.data_tab_formula_inputs['needle_mm_col'] = 'Needle dia (mm)' if 'Needle dia (mm)' in current_all_cols else current_all_cols[0]
            st.session_state.data_tab_formula_inputs['speed_col'] = 'Speed (mm/s)' if 'Speed (mm/s)' in current_all_cols else current_all_cols[0]
        
        col1, col2 = st.columns([3.1, 1.1])
        
        with col1:
            st.markdown("#### 📋 Data Table")
            if current_df is not None:
                st.dataframe(current_df, use_container_width=True, height=500)
            else:
                st.info("No data available to display. Please upload and select a file first.")
        
        with col2:
            st.markdown("#### ➕ Add New Column by Formula")
            # The user must specify which columns hold physical quantities
            
            if not current_all_cols:
                st.info("No data available. Please upload and select a file first.")
            else:
                # Get current values from session state
                current_inputs = st.session_state.data_tab_formula_inputs
                
                needle_mm_col = st.selectbox("Column: Needle Diameter (mm) or map it", options=current_all_cols,
                                             index=current_all_cols.index(current_inputs['needle_mm_col']) if current_inputs['needle_mm_col'] in current_all_cols else 0,
                                             key="data_needle_mm_col")
                speed_col = st.selectbox("Column: Speed (mm/s)", options=current_all_cols,
                                         index=current_all_cols.index(current_inputs['speed_col']) if current_inputs['speed_col'] in current_all_cols else 0,
                                         key="data_speed_col")
                new_col_name = st.text_input("New Column Name (e.g., AreaA1 (mm2))", 
                                            value=current_inputs['new_col_name'],
                                            key="data_new_col_name")
                formula_choice = st.selectbox("Formula",
                                              options=["None",
                                                       "AreaA1 (mm2) = π*d^2/4",
                                                       "FlowQ1 (mm3/s) = Area * Speed",
                                                       "ShearS1 (1/s) ≈ 8*Speed/d",
                                                       "ViscosN1 (Pa·s) = K * shear^(n-1)",
                                                       "Custom (pandas.eval)"],
                                              index=["None",
                                                     "AreaA1 (mm2) = π*d^2/4",
                                                     "FlowQ1 (mm3/s) = Area * Speed",
                                                     "ShearS1 (1/s) ≈ 8*Speed/d",
                                                     "ViscosN1 (Pa·s) = K * shear^(n-1)",
                                                     "Custom (pandas.eval)"].index(current_inputs['formula_choice']),
                                              key="data_formula_choice")
                custom_expr = st.text_input("Custom expression (optional, uses column names)", 
                                           value=current_inputs['custom_expr'],
                                           key="data_custom_expr")

                add_col_clicked = st.button("Add Column")

                if add_col_clicked and new_col_name.strip() != "":
                    if current_df is not None:
                        updated_df, err = compute_physics_columns(current_df, needle_mm_col, speed_col, new_col_name, formula_choice, custom_expr)
                        if err:
                            st.error(err)
                        else:
                            st.success(f"Added '{new_col_name}' to the dataframe.")
                            
                            # Update the dataframe in session state
                            if 'data' in st.session_state:
                                st.session_state['data'] = updated_df
                            
                            # Auto-include the new column if not in X_cols yet
                            if new_col_name not in current_X_cols:
                                current_X_cols.append(new_col_name)
                                if 'X_cols' in st.session_state:
                                    st.session_state['X_cols'] = current_X_cols
                            
                            # Update session state with current inputs
                            st.session_state.data_tab_formula_inputs = {
                                'needle_mm_col': needle_mm_col,
                                'speed_col': speed_col,
                                'new_col_name': new_col_name,
                                'formula_choice': formula_choice,
                                'custom_expr': custom_expr
                            }
                            
                            st.rerun()  # Refresh to show updated data
                    else:
                        st.error("No data available to modify.")
        
        st.markdown('<p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 10px;">Tip: use the formula tool to add Area/Flow/Shear/Viscosity columns if your CSV lacks them. Then retrain for better accuracy.</p>', unsafe_allow_html=True)

    with tab3:
        # st.markdown("### 🎯 Prediction Panel")
        
        # Create two columns for better layout
        col1, col2 = st.columns([4, 1])
        
        with col2:
            st.markdown("#### 📝 Input Parameters")
            
            # Initialize session state for prediction inputs
            if 'pred_needle_size' not in st.session_state:
                st.session_state.pred_needle_size = 18
            if 'pred_material' not in st.session_state:
                st.session_state.pred_material = "DS10"
            if 'pred_thivex' not in st.session_state:
                st.session_state.pred_thivex = "0"
            if 'pred_time' not in st.session_state:
                st.session_state.pred_time = "Phase1: First 30 mins"
            if 'pred_pressure' not in st.session_state:
                st.session_state.pred_pressure = 85.0
            if 'pred_speed' not in st.session_state:
                st.session_state.pred_speed = 50.0
            if 'show_graphs' not in st.session_state:
                st.session_state.show_graphs = False
            
            needle_size = st.selectbox("Needle Size (Gauge)", options=[18, 21], 
                                      index=0 if st.session_state.pred_needle_size == 18 else 1,
                                      key="pred_needle_size")
            material_in = st.selectbox("Material", options=["DS10","DS30","SS960"],
                                      index=["DS10","DS30","SS960"].index(st.session_state.pred_material),
                                      key="pred_material")
            thivex_in = st.selectbox("Thivex", options=["0","1","2"],
                                    index=["0","1","2"].index(st.session_state.pred_thivex),
                                    key="pred_thivex")
            time_in = st.selectbox("Time Period", options=["Phase1: First 30 mins","Phase2: 30 mins to 60 min","Phase3: After 60 mins"],
                                  index=["Phase1: First 30 mins","Phase2: 30 mins to 60 min","Phase3: After 60 mins"].index(st.session_state.pred_time),
                                  key="pred_time")
            press_in = st.number_input("Pressure (psi)", min_value=20.0, max_value=200.0, 
                                      value=st.session_state.pred_pressure, step=1.0,
                                      key="pred_pressure")
            speed_in = st.number_input("Speed (mm/s)", min_value=5.0, max_value=4000.0, 
                                      value=st.session_state.pred_speed, step=1.0,
                                      key="pred_speed")
            
            pred_btn = st.button("🔮 Predict Line Width", type="primary", use_container_width=True)


        
        with col1:
            st.markdown("#### 📊 Prediction Results")
            
            # Show current model performance if available
            if 'model_performances' in st.session_state and st.session_state.selected_model in st.session_state.model_performances:
                perf = st.session_state.model_performances[st.session_state.selected_model]
                st.info(f"🎯 **Current Model**: {st.session_state.selected_model} | R²: {perf['r2']:.3f} | RMSE: {perf['rmse']:.1f} µm")
            
            # Initialize session state for prediction results
            if 'prediction_results' not in st.session_state:
                st.session_state.prediction_results = None
            if 'last_prediction_inputs' not in st.session_state:
                st.session_state.last_prediction_inputs = None
            
            if pred_btn:
                if "trained_pipeline" not in st.session_state:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); padding: 25px; border-radius: 12px; margin: 20px 0; text-align: center;">
                        <h3 style="color: white; margin: 0 0 15px 0;">🔄 Model Training Required for Prediction</h3>
                        <p style="color: white; font-size: 15px; margin: 0 0 15px 0;">
                            <strong>No trained model found!</strong> You need to train a model before making predictions.
                        </p>
                        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <h4 style="color: white; margin: 0 0 8px 0;">📋 Quick Steps:</h4>
                            <ol style="color: white; text-align: left; margin: 0; padding-left: 20px; font-size: 14px;">
                                <li>Go to the main training section above</li>
                                <li>Select your target and feature columns</li>
                                <li>Choose a machine learning model</li>
                                <li>Click <strong>"Train & Analyze"</strong></li>
                                <li>Return here to make predictions</li>
                            </ol>
                        </div>
                        <p style="color: #ffeaa7; font-size: 13px; margin: 10px 0 0 0;">
                            💡 <strong>Tip:</strong> Once trained, you can input parameters and get instant predictions!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif "data" not in st.session_state:
                    st.error("❌ **No data available!** Please upload and process your data first.")
                elif "X_cols" not in st.session_state:
                    st.error("❌ **No features selected!** Please select input columns and train the model.")
                elif "selected_model" not in st.session_state:
                    st.error("❌ **No model selected!** Please select a model type and train it first.")
                else:
                    # Check if the selected model is compatible with the trained pipeline
                    selected_model = st.session_state.get("selected_model", "Random Forest")
                    pipe = st.session_state["trained_pipeline"]
                    
                    # Get expected model step name
                    expected_step = "rf" if selected_model == "Random Forest" else "xgb" if selected_model == "XGBoost" else "svr" if selected_model == "SVR" else "knn"
                    
                    if expected_step not in pipe.named_steps:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); padding: 25px; border-radius: 12px; margin: 20px 0; text-align: center;">
                            <h3 style="color: white; margin: 0 0 15px 0;">🔄 Model Configuration Changed</h3>
                            <p style="color: white; font-size: 15px; margin: 0 0 15px 0;">
                                <strong>Model mismatch detected!</strong> You selected '{selected_model}' but the trained model is different.
                            </p>
                            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                                <h4 style="color: white; margin: 0 0 8px 0;">🛠️ Action Required:</h4>
                                <ul style="color: white; text-align: left; margin: 0; padding-left: 20px; font-size: 14px;">
                                    <li>Go to the training section above</li>
                                    <li>Ensure '{selected_model}' is selected</li>
                                    <li>Click <strong>"Train & Analyze"</strong> to retrain</li>
                                    <li>Return here to make predictions</li>
                                </ul>
                            </div>
                            <p style="color: #ffeaa7; font-size: 13px; margin: 10px 0 0 0;">
                                💡 <strong>Note:</strong> This happens when you change the model type after training.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Prediction logic here
                        pipe = st.session_state["trained_pipeline"]
                        selected_model = st.session_state.get("selected_model", "Random Forest")
                        data = st.session_state["data"]
                        X_cols = st.session_state["X_cols"]
                        
                        X = data[X_cols]
                        y = data[y_col].astype(float)
                        pipe, num_cols, cat_cols, X_processed = build_model(X, y, selected_model, fine_tune)
                        pipe.fit(X_processed, y)

                        # Build a single-row DataFrame aligned to X_cols
                        row = {}
                        for c in X_cols:
                            if c == 'Needle Size':
                                row[c] = needle_size
                            elif c == 'Material':
                                # Convert material to numeric for prediction
                                material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
                                row[c] = material_map.get(material_in, 1)
                            elif c == 'Thivex':
                                # Convert thivex to numeric for prediction
                                row[c] = float(thivex_in.replace('%', ''))
                            elif c == 'Time Period':
                                # Convert time period to numeric for prediction
                                time_map = {
                                    'Phase1: First 30 mins': 1,
                                    'Phase2: 30 mins to 60 min': 2, 
                                    'Phase3: After 60 mins': 3
                                }
                                row[c] = time_map.get(time_in, 1)
                            elif c == 'Pressure (psi)':
                                row[c] = press_in
                            elif c == 'Speed (mm/s)':
                                row[c] = speed_in
                            else:
                                # Try to compute physics columns if names match common presets
                                if c.lower().startswith("areaa1"):
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
                                    row[c] = 0.0

                        x_row = pd.DataFrame([row], columns=X_cols)
                        
                        # Process prediction input the same way as training data
                        x_row_processed = x_row.copy()
                        if 'Thivex' in x_row_processed.columns:
                            x_row_processed['Thivex'] = x_row_processed['Thivex'].astype(str).str.replace('%', '').astype(float)
                        if 'Material' in x_row_processed.columns:
                            material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
                            x_row_processed['Material'] = x_row_processed['Material'].map(material_map)
                        if 'Time Period' in x_row_processed.columns:
                            time_map = {
                                'Phase1: First 30 mins': 1,
                                'Phase2: 30 mins to 60 min': 2, 
                                'Phase3: After 60 mins': 3
                            }
                            x_row_processed['Time Period'] = x_row_processed['Time Period'].map(time_map)
                        
                        # Handle NaN values before prediction
                        x_row_processed = x_row_processed.fillna(method='ffill').fillna(method='bfill').fillna(0)
                        
                        try:
                            pred_width = float(pipe.predict(x_row_processed)[0])  # µm
                        except Exception as e:
                            st.error(f"❌ **Prediction failed**: {str(e)}. Please check your input values and try again.")
                            st.stop()

                        # Classification by difference
                        internal_um = INTERNAL_DIAMETER.get(needle_size, 0.0)
                        width_diff = pred_width - internal_um
                        verdict = quality_bucket(width_diff)

                        # Save results to session state
                        st.session_state.prediction_results = {
                            'pred_width': pred_width,
                            'internal_um': internal_um,
                            'width_diff': width_diff,
                            'verdict': verdict
                        }
                        st.session_state.last_prediction_inputs = row
                        st.session_state.show_graphs = True  # Flag to show graphs for this prediction

                    # Display results in a nice format
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <h3 style="color: white; text-align: center; margin: 0 0 15px 0;">🔮 Prediction Results</h3>
                        <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 15px;">
                            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; min-width: 100px;">
                                <h4 style="color: white; margin: 0; font-size: 12px;">Predicted Width</h4>
                                <p style="color: white; font-size: 18px; font-weight: bold; margin: 5px 0;">{:.1f} µm</p>
                            </div>
                            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; min-width: 100px;">
                                <h4 style="color: white; margin: 0; font-size: 12px;">Internal ID</h4>
                                <p style="color: white; font-size: 18px; font-weight: bold; margin: 5px 0;">{:.0f} µm</p>
                            </div>
                            <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; min-width: 100px;">
                                <h4 style="color: white; margin: 0; font-size: 12px;">Difference</h4>
                                <p style="color: white; font-size: 18px; font-weight: bold; margin: 5px 0;">{:+.1f} µm</p>
                            </div>
                        </div>
                        <div style="text-align: center;">
                            <h4 style="color: white; margin: 0;">Print Quality: <strong>{}</strong></h4>
                        </div>
                    </div>
                    """.format(pred_width, internal_um, width_diff, verdict), unsafe_allow_html=True)
                    
                    # Add width vs speed and width vs pressure graphs
                    st.markdown("#### 📊 Parameter Range Analysis")
                    
                    # Create two columns for the graphs
                    graph_col1, graph_col2 = st.columns(2)
                    
                    with graph_col1:
                        st.markdown("**Width vs Speed**")
                        
                        # Get actual data ranges from the training data for better graph scaling
                        if 'data' in st.session_state:
                            actual_speeds = st.session_state['data']['Speed (mm/s)'].dropna()
                            min_speed = max(5.0, actual_speeds.min() * 0.5)
                            max_speed = min(4000.0, actual_speeds.max() * 1.5)
                        else:
                            min_speed = max(5.0, speed_in * 0.5)
                            max_speed = min(4000.0, speed_in * 1.5)
                        
                        # Create a range of speeds based on actual data
                        speed_range = np.linspace(min_speed, max_speed, 50)
                        width_predictions = []
                        
                        for test_speed in speed_range:
                            # Create test row with current speed
                            test_row = row.copy()
                            test_row['Speed (mm/s)'] = test_speed
                            
                            # Update physics columns if they exist
                            if 'AreaA1 (mm2)' in test_row:
                                d_mm = 0.838 if needle_size == 18 else 0.514
                                test_row['AreaA1 (mm2)'] = np.pi * (d_mm**2) / 4.0
                            if 'FlowQ1 (mm3/s)' in test_row:
                                d_mm = 0.838 if needle_size == 18 else 0.514
                                area = np.pi * (d_mm**2) / 4.0
                                test_row['FlowQ1 (mm3/s)'] = area * test_speed
                            if 'ShearS1 (1/s)' in test_row:
                                d_mm = 0.838 if needle_size == 18 else 0.514
                                test_row['ShearS1 (1/s)'] = 8.0 * test_speed / max(d_mm, 1e-6)
                            if 'ViscosN1 (Pa·s)' in test_row:
                                d_mm = 0.838 if needle_size == 18 else 0.514
                                shear = 8.0 * test_speed / max(d_mm, 1e-6)
                                K, n = 0.9, 0.06
                                test_row['ViscosN1 (Pa·s)'] = K * (shear ** (n - 1.0))
                            
                            # Process the test row
                            test_x_row = pd.DataFrame([test_row], columns=X_cols)
                            test_x_row_processed = test_x_row.copy()
                            if 'Thivex' in test_x_row_processed.columns:
                                test_x_row_processed['Thivex'] = test_x_row_processed['Thivex'].astype(str).str.replace('%', '').astype(float)
                            if 'Material' in test_x_row_processed.columns:
                                material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
                                test_x_row_processed['Material'] = test_x_row_processed['Material'].map(material_map)
                            if 'Time Period' in test_x_row_processed.columns:
                                time_map = {
                                    'Phase1: First 30 mins': 1,
                                    'Phase2: 30 mins to 60 min': 2, 
                                    'Phase3: After 60 mins': 3
                                }
                                test_x_row_processed['Time Period'] = test_x_row_processed['Time Period'].map(time_map)
                            
                            # Predict width for this speed
                            test_pred_width = float(pipe.predict(test_x_row_processed)[0])
                            width_predictions.append(test_pred_width)
                        
                        # Create the width vs speed plot
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(speed_range, width_predictions, 'b-', linewidth=2, alpha=0.7)
                        ax.scatter([speed_in], [pred_width], color='red', s=100, zorder=5, label='Current Point')
                        ax.axhline(y=internal_um, color='green', linestyle='--', alpha=0.7, label=f'Target ({internal_um:.0f} µm)')
                        ax.set_xlabel('Speed (mm/s)')
                        ax.set_ylabel('Predicted Width (µm)')
                        ax.set_title('Width vs Speed')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with graph_col2:
                        st.markdown("**Width vs Pressure**")
                        
                        # Get actual data ranges from the training data for better graph scaling
                        if 'data' in st.session_state:
                            actual_pressures = st.session_state['data']['Pressure (psi)'].dropna()
                            min_pressure = max(20.0, actual_pressures.min() * 0.5)
                            max_pressure = min(200.0, actual_pressures.max() * 1.5)
                        else:
                            min_pressure = max(20.0, press_in * 0.5)
                            max_pressure = min(200.0, press_in * 1.5)
                        
                        # Create a range of pressures based on actual data
                        pressure_range = np.linspace(min_pressure, max_pressure, 50)
                        width_predictions_pressure = []
                        
                        for test_pressure in pressure_range:
                            # Create test row with current pressure
                            test_row = row.copy()
                            test_row['Pressure (psi)'] = test_pressure
                            
                            # Process the test row
                            test_x_row = pd.DataFrame([test_row], columns=X_cols)
                            test_x_row_processed = test_x_row.copy()
                            if 'Thivex' in test_x_row_processed.columns:
                                test_x_row_processed['Thivex'] = test_x_row_processed['Thivex'].astype(str).str.replace('%', '').astype(float)
                            if 'Material' in test_x_row_processed.columns:
                                material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
                                test_x_row_processed['Material'] = test_x_row_processed['Material'].map(material_map)
                            if 'Time Period' in test_x_row_processed.columns:
                                time_map = {
                                    'Phase1: First 30 mins': 1,
                                    'Phase2: 30 mins to 60 min': 2, 
                                    'Phase3: After 60 mins': 3
                                }
                                test_x_row_processed['Time Period'] = test_x_row_processed['Time Period'].map(time_map)
                            
                            # Predict width for this pressure
                            test_pred_width = float(pipe.predict(test_x_row_processed)[0])
                            width_predictions_pressure.append(test_pred_width)
                        
                        # Create the width vs pressure plot
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(pressure_range, width_predictions_pressure, 'orange', linewidth=2, alpha=0.7)
                        ax.scatter([press_in], [pred_width], color='red', s=100, zorder=5, label='Current Point')
                        ax.axhline(y=internal_um, color='green', linestyle='--', alpha=0.7, label=f'Target ({internal_um:.0f} µm)')
                        ax.set_xlabel('Pressure (psi)')
                        ax.set_ylabel('Predicted Width (µm)')
                        ax.set_title('Width vs Pressure')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with st.expander("📋 Show input parameters sent to model"):
                        st.json(row)
            
            # Display saved results if available (when page reruns)
            elif st.session_state.prediction_results is not None:
                # Reset graphs flag when just viewing saved results (not a new prediction)
                st.session_state.show_graphs = False
                results = st.session_state.prediction_results
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3 style="color: white; text-align: center; margin: 0 0 15px 0;">🔮 Prediction Results</h3>
                    <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 15px;">
                        <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; min-width: 100px;">
                            <h4 style="color: white; margin: 0; font-size: 12px;">Predicted Width</h4>
                            <p style="color: white; font-size: 18px; font-weight: bold; margin: 5px 0;">{:.1f} µm</p>
                        </div>
                        <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; min-width: 100px;">
                            <h4 style="color: white; margin: 0; font-size: 12px;">Internal ID</h4>
                            <p style="color: white; font-size: 18px; font-weight: bold; margin: 5px 0;">{:.0f} µm</p>
                        </div>
                        <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; min-width: 100px;">
                            <h4 style="color: white; margin: 0; font-size: 12px;">Difference</h4>
                            <p style="color: white; font-size: 18px; font-weight: bold; margin: 5px 0;">{:+.1f} µm</p>
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="color: white; margin: 0;">Print Quality: <strong>{}</strong></h4>
                    </div>
                </div>
                """.format(results['pred_width'], results['internal_um'], results['width_diff'], results['verdict']), unsafe_allow_html=True)
                
                if st.session_state.last_prediction_inputs:
                    # Only show graphs if this is a fresh prediction (not just viewing saved results)
                    if 'show_graphs' in st.session_state and st.session_state.show_graphs:
                        # Add width vs speed and width vs pressure graphs for saved results
                        st.markdown("#### 📊 Parameter Range Analysis")
                        
                        # Get the saved inputs and results
                        saved_inputs = st.session_state.last_prediction_inputs
                        saved_results = st.session_state.prediction_results
                        
                        # Create two columns for the graphs
                        graph_col1, graph_col2 = st.columns(2)
                        
                        with graph_col1:
                            st.markdown("**Width vs Speed**")
                            
                            # Get actual data ranges from the training data for better graph scaling
                            if 'data' in st.session_state:
                                actual_speeds = st.session_state['data']['Speed (mm/s)'].dropna()
                                min_speed = max(5.0, actual_speeds.min() * 0.5)
                                max_speed = min(4000.0, actual_speeds.max() * 1.5)
                            else:
                                saved_speed = saved_inputs.get('Speed (mm/s)', 50.0)
                                min_speed = max(5.0, saved_speed * 0.5)
                                max_speed = min(4000.0, saved_speed * 1.5)
                            
                            # Create a range of speeds based on actual data
                            speed_range = np.linspace(min_speed, max_speed, 50)
                            width_predictions = []
                            
                            for test_speed in speed_range:
                                # Create test row with current speed
                                test_row = saved_inputs.copy()
                                test_row['Speed (mm/s)'] = test_speed
                                
                                # Update physics columns if they exist
                                if 'AreaA1 (mm2)' in test_row:
                                    needle_size = saved_inputs.get('Needle Size', 18)
                                    d_mm = 0.838 if needle_size == 18 else 0.514
                                    test_row['AreaA1 (mm2)'] = np.pi * (d_mm**2) / 4.0
                                if 'FlowQ1 (mm3/s)' in test_row:
                                    needle_size = saved_inputs.get('Needle Size', 18)
                                    d_mm = 0.838 if needle_size == 18 else 0.514
                                    area = np.pi * (d_mm**2) / 4.0
                                    test_row['FlowQ1 (mm3/s)'] = area * test_speed
                                if 'ShearS1 (1/s)' in test_row:
                                    needle_size = saved_inputs.get('Needle Size', 18)
                                    d_mm = 0.838 if needle_size == 18 else 0.514
                                    test_row['ShearS1 (1/s)'] = 8.0 * test_speed / max(d_mm, 1e-6)
                                if 'ViscosN1 (Pa·s)' in test_row:
                                    needle_size = saved_inputs.get('Needle Size', 18)
                                    d_mm = 0.838 if needle_size == 18 else 0.514
                                    shear = 8.0 * test_speed / max(d_mm, 1e-6)
                                    K, n = 0.9, 0.06
                                    test_row['ViscosN1 (Pa·s)'] = K * (shear ** (n - 1.0))
                                
                                # Process the test row
                                test_x_row = pd.DataFrame([test_row], columns=X_cols)
                                test_x_row_processed = test_x_row.copy()
                                if 'Thivex' in test_x_row_processed.columns:
                                    test_x_row_processed['Thivex'] = test_x_row_processed['Thivex'].astype(str).str.replace('%', '').astype(float)
                                if 'Material' in test_x_row_processed.columns:
                                    material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
                                    test_x_row_processed['Material'] = test_x_row_processed['Material'].map(material_map)
                                if 'Time Period' in test_x_row_processed.columns:
                                    time_map = {
                                        'Phase1: First 30 mins': 1,
                                        'Phase2: 30 mins to 60 min': 2, 
                                        'Phase3: After 60 mins': 3
                                    }
                                    test_x_row_processed['Time Period'] = test_x_row_processed['Time Period'].map(time_map)
                                
                                # Predict width for this speed
                                test_pred_width = float(pipe.predict(test_x_row_processed)[0])
                                width_predictions.append(test_pred_width)
                            
                            # Get the saved speed value for plotting
                            saved_speed = saved_inputs.get('Speed (mm/s)', 50.0)
                            
                            # Create the width vs speed plot
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(speed_range, width_predictions, 'b-', linewidth=2, alpha=0.7)
                            ax.scatter([saved_speed], [saved_results['pred_width']], color='red', s=100, zorder=5, label='Current Point')
                            ax.axhline(y=saved_results['internal_um'], color='green', linestyle='--', alpha=0.7, label=f'Target ({saved_results["internal_um"]:.0f} µm)')
                            ax.set_xlabel('Speed (mm/s)')
                            ax.set_ylabel('Predicted Width (µm)')
                            ax.set_title('Width vs Speed')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with graph_col2:
                            st.markdown("**Width vs Pressure**")
                            
                            # Get actual data ranges from the training data for better graph scaling
                            if 'data' in st.session_state:
                                actual_pressures = st.session_state['data']['Pressure (psi)'].dropna()
                                min_pressure = max(20.0, actual_pressures.min() * 0.5)
                                max_pressure = min(200.0, actual_pressures.max() * 1.5)
                            else:
                                saved_pressure = saved_inputs.get('Pressure (psi)', 85.0)
                                min_pressure = max(20.0, saved_pressure * 0.5)
                                max_pressure = min(200.0, saved_pressure * 1.5)
                            
                            # Create a range of pressures based on actual data
                            pressure_range = np.linspace(min_pressure, max_pressure, 50)
                            width_predictions_pressure = []
                            
                            for test_pressure in pressure_range:
                                # Create test row with current pressure
                                test_row = saved_inputs.copy()
                                test_row['Pressure (psi)'] = test_pressure
                                
                                # Process the test row
                                test_x_row = pd.DataFrame([test_row], columns=X_cols)
                                test_x_row_processed = test_x_row.copy()
                                if 'Thivex' in test_x_row_processed.columns:
                                    test_x_row_processed['Thivex'] = test_x_row_processed['Thivex'].astype(str).str.replace('%', '').astype(float)
                                if 'Material' in test_x_row_processed.columns:
                                    material_map = {'DS10': 1, 'DS30': 2, 'SS960': 3}
                                    test_x_row_processed['Material'] = test_x_row_processed['Material'].map(material_map)
                                if 'Time Period' in test_x_row_processed.columns:
                                    time_map = {
                                        'Phase1: First 30 mins': 1,
                                        'Phase2: 30 mins to 60 min': 2, 
                                        'Phase3: After 60 mins': 3
                                    }
                                    test_x_row_processed['Time Period'] = test_x_row_processed['Time Period'].map(time_map)
                                
                                # Predict width for this pressure
                                test_pred_width = float(pipe.predict(test_x_row_processed)[0])
                                width_predictions_pressure.append(test_pred_width)
                            
                            # Get the saved pressure value for plotting
                            saved_pressure = saved_inputs.get('Pressure (psi)', 85.0)
                            
                            # width vs pressure plot
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(pressure_range, width_predictions_pressure, 'orange', linewidth=2, alpha=0.7)
                            ax.scatter([saved_pressure], [saved_results['pred_width']], color='red', s=100, zorder=5, label='Current Point')
                            ax.axhline(y=saved_results['internal_um'], color='green', linestyle='--', alpha=0.7, label=f'Target ({saved_results["internal_um"]:.0f} µm)')
                            ax.set_xlabel('Pressure (psi)')
                            ax.set_ylabel('Predicted Width (µm)')
                            ax.set_title('Width vs Pressure')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with st.expander("📋 Show last input parameters"):
                            st.json(st.session_state.last_prediction_inputs)
            else:
                st.info("Click 'Predict Line Width' to see results.")
            st.markdown('<p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 10px;">Tip: Use this panel to predict line width based on your trained model.</p>', unsafe_allow_html=True)
            
