#type: ignore
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
import sklearn.exceptions
import shap
import tempfile
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
            
            # Save plot-specific results
            st.session_state.plot_training_results = {
                'data': data,
                'X_cols': X_cols,
                'y_col': y_col,
                'num_cols': num_cols,
                'cat_cols': cat_cols,
                'pipe': pipe
            }
            st.session_state.plot_model_metrics = {
                'r2': r2,
                'rmse': rmse
            }

            # # ---- Enhanced Model Performance Metrics ----
            # st.markdown("""
            # <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
            #     <h2 style="color: white; text-align: center; margin: 0 0 20px 0;">🎯 Model Performance Dashboard</h2>
            #     <div style="display: flex; justify-content: space-around; align-items: center;">
            #         <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; min-width: 150px;">
            #             <h3 style="color: white; margin: 0; font-size: 14px;">R² Score (Test)</h3>
            #             <p style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{:.3f}</p>
            #             <p style="color: #e0e0e0; font-size: 12px; margin: 0;">Coefficient of Determination</p>
            #         </div>
            #         <div style="text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; min-width: 150px;">
            #             <h3 style="color: white; margin: 0; font-size: 14px;">RMSE (Test)</h3>
            #             <p style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{:.1f} µm</p>
            #             <p style="color: #e0e0e0; font-size: 12px; margin: 0;">Root Mean Square Error</p>
            #         </div>
            #     </div>
            # </div>
            # """.format(r2, rmse), unsafe_allow_html=True)
            
            # Additional performance insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📊 **Model Quality**: {'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.7 else 'Fair'}")
            with col2:
                st.info(f"🎯 **Accuracy**: {'High' if rmse < 50 else 'Medium' if rmse < 100 else 'Low'}")
            with col3:
                st.info(f"📈 **Reliability**: {'Very High' if r2 > 0.95 else 'High' if r2 > 0.85 else 'Moderate'}")

            # ---- 2x2 Grid Layout for Plots ----
            # st.markdown("### 📊 Model Analysis Dashboard")
            
            # Create 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot 1: Correlation Heatmap
                st.markdown("#### 📈 Correlation Heatmap")
                num_for_corr = data[X_cols + [y_col]].select_dtypes(include=[np.number])
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
                st.markdown("#### 📋 Top Features")
                st.dataframe(fi.head(10), use_container_width=True, height=410)
                
            # Second row
            col3, col4 = st.columns(2)
            
            with col4:
                # Plot 4: SHAP Summary
                st.markdown("#### 🔎 SHAP Summary")
                # Build a small background set
                sample_X = X_tr.sample(min(100, len(X_tr)), random_state=42)
                # Transform sample through preprocessor to get model input
                bg = pre.transform(sample_X)
                explainer = shap.TreeExplainer(rf)
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
            
            with col3:
                # Plot 3: Feature Importance Barplot
                st.markdown("#### 🌲 Feature Importance")
                plt.figure(figsize=(5, 4))
                sns.barplot(x="importance", y="feature", data=fi.head(15))
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.close()
        else:
            # Display saved results if available (when page reruns)
            if st.session_state.plot_training_results is not None and st.session_state.plot_model_metrics is not None:
                # Restore saved data
                data = st.session_state.plot_training_results['data']
                X_cols = st.session_state.plot_training_results['X_cols']
                y_col = st.session_state.plot_training_results['y_col']
                num_cols = st.session_state.plot_training_results['num_cols']
                cat_cols = st.session_state.plot_training_results['cat_cols']
                pipe = st.session_state.plot_training_results['pipe']
                r2 = st.session_state.plot_model_metrics['r2']
                rmse = st.session_state.plot_model_metrics['rmse']
                
                # Display saved model performance metrics
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

                # Display saved plots
                st.markdown("### 📊 Model Analysis Dashboard")
                
                # Create 2x2 grid
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot 1: Correlation Heatmap
                    st.markdown("#### 📈 Correlation Heatmap")
                    num_for_corr = data[X_cols + [y_col]].select_dtypes(include=[np.number])
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
                    st.markdown("#### 📋 Top Features")
                    st.dataframe(fi.head(10), use_container_width=True, height=410)
                    
                # Second row
                col3, col4 = st.columns(2)
                
                with col4:
                    # Plot 3: SHAP Summary
                    st.markdown("#### 🔎 SHAP Summary")
                    # Build a small background set
                    X_tr = data[X_cols].sample(min(100, len(data)), random_state=42)
                    # Transform sample through preprocessor to get model input
                    bg = pre.transform(X_tr)
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(pre.transform(X_tr))
                    try:
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                    except Exception:
                        pass
                    shap.summary_plot(shap_values, features=pre.transform(X_tr), feature_names=full_names, show=False)
                    fig = plt.gcf()
                    fig.set_size_inches(5, 4)
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close(fig)
                
                with col3:
                    # Plot 4: Feature Importance Barplot
                    st.markdown("#### 🌲 Feature Importance")
                    plt.figure(figsize=(5, 4))
                    sns.barplot(x="importance", y="feature", data=fi.head(15))
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
            else:
                st.info("Click 'Train & Analyze' to see model performance and visualizations.")
        
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
                
                # Add helpful information about the feature
                st.markdown("""
                <div style="background-color: #f0f8ff; border-left: 4px solid #007bff; padding: 10px; margin: 10px 0; border-radius: 4px;">
                    <h6 style="margin: 0 0 5px 0; color: #007bff;">💡 How to use this feature:</h6>
                    <ul style="margin: 0; padding-left: 20px; font-size: 0.9em;">
                        <li>Select the column containing needle diameter values (in mm)</li>
                        <li>Select the column containing speed values (in mm/s)</li>
                        <li>Choose a formula or enter a custom expression</li>
                        <li>Enter a name for the new column</li>
                        <li>Click "Add Column" to create the calculated column</li>
                        <li>After adding columns, retrain your model for better accuracy</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
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
                st.session_state.pred_speed = 1200.0
            
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
                                      value=st.session_state.pred_speed, step=10.0,
                                      key="pred_speed")
            
            pred_btn = st.button("🔮 Predict Line Width", type="primary", use_container_width=True)


        
        with col1:
            st.markdown("#### 📊 Prediction Results")
            
            # Initialize session state for prediction results
            if 'prediction_results' not in st.session_state:
                st.session_state.prediction_results = None
            if 'last_prediction_inputs' not in st.session_state:
                st.session_state.last_prediction_inputs = None
            
            if pred_btn:
                if "trained_pipeline" not in st.session_state:
                    st.error("Please train the model first (upload/select columns and click 'Train & Analyze').")
                else:
                    # Prediction logic here
                    pipe = st.session_state["trained_pipeline"]
                    data = st.session_state["data"]
                    X_cols = st.session_state["X_cols"]
                    
                    X = data[X_cols]
                    y = data[y_col].astype(float)
                    pipe, num_cols, cat_cols = build_model(X, y)
                    pipe.fit(X, y)

                    # Build a single-row DataFrame aligned to X_cols
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
                    pred_width = float(pipe.predict(x_row)[0])  # µm

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
                    
                    with st.expander("📋 Show input parameters sent to model"):
                        st.json(row)
            
            # Display saved results if available (when page reruns)
            elif st.session_state.prediction_results is not None:
                results = st.session_state.prediction_results
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3 style="color: white; text-align: center; margin: 0 0 15px 0;">🔮 Last Prediction Results</h3>
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
                    with st.expander("📋 Show last input parameters"):
                        st.json(st.session_state.last_prediction_inputs)
            else:
                st.info("Click 'Predict Line Width' to see results.")
            st.markdown('<p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 10px;">Tip: Use this panel to predict line width based on your trained model.</p>', unsafe_allow_html=True)
            

    #         # st.markdown("### 📊 Model Analysis Dashboard")
    #         # Create 2x2 grid
    #         col3, col4 = st.columns(2)
    #         with col3:
    #             # Plot 1: Correlation Heatmap
    #             st.markdown("#### 📈 Correlation Heatmap")
    #             num_for_corr = data[X_cols + [y_col]].select_dtypes(include=[np.number])
    #             if num_for_corr.shape[1] >= 2:
    #                 plt.figure(figsize=(6, 5))
    #                 corr = num_for_corr.corr()
    #                 # Create heatmap with annotations
    #                 sns.heatmap(corr, 
    #                             annot=True,  # Show values
    #                             fmt='.2f',   # Format to 2 decimal places
    #                             cmap="coolwarm", 
    #                             center=0,
    #                             square=True,  # Make it square
    #                             cbar_kws={"shrink": .8},
    #                             annot_kws={"size": 8})  # Smaller font for annotations
    #                 plt.tight_layout()
    #                 st.pyplot(plt.gcf())
    #                 plt.close()
    #             else:
    #                 st.info("Not enough numeric columns for correlation heatmap.")
            
    #         with col4:
    #             # Plot 2: Feature Importance Table
    #             # Extract trained RF and one-hot feature names
    #             rf = pipe.named_steps["rf"]
    #             pre = pipe.named_steps["pre"]
    #             ohe = pre.named_transformers_["cat"]
    #             # build full feature names: scaled numeric + OHE cats
    #             num_names = num_cols
    #             cat_expanded = []
    #             if hasattr(ohe, "get_feature_names_out"):
    #                 cat_expanded = ohe.get_feature_names_out(cat_cols).tolist()
    #             full_names = num_names + cat_expanded
    #             importances = rf.feature_importances_
    #             fi = pd.DataFrame({"feature": full_names, "importance": importances}).sort_values("importance", ascending=False)
    #             st.markdown("#### 📋 Top Features")
    #             st.dataframe(fi.head(10), use_container_width=True, height=410)
                
    #         # Second row
    #         col5, col6 = st.columns(2)
            
    #         with col5:
    #             # Plot 4: SHAP Summary
    #             st.markdown("#### 🔎 SHAP Summary")
    #             # Build a small background set
    #             sample_X = X_tr.sample(min(100, len(X_tr)), random_state=42)
    #             # Transform sample through preprocessor to get model input
    #             bg = pre.transform(sample_X)
    #             explainer = shap.TreeExplainer(rf)
    #             shap_values = explainer.shap_values(pre.transform(sample_X))
    #             try:
    #                 st.set_option('deprecation.showPyplotGlobalUse', False)
    #             except Exception:
    #                 pass
    #             shap.summary_plot(shap_values, features=pre.transform(sample_X), feature_names=full_names, show=False)
    #             fig = plt.gcf()
    #             fig.set_size_inches(5, 4)
    #             st.pyplot(fig, bbox_inches='tight')
    #             plt.close(fig)
            
    #         with col6:
    #             # Plot 3: Feature Importance Barplot
    #             st.markdown("#### 🌲 Feature Importance")
    #             plt.figure(figsize=(5, 4))
    #             sns.barplot(x="importance", y="feature", data=fi.head(15))
    #             plt.tight_layout()
    #             st.pyplot(plt.gcf())
    #             plt.close()
    
    # End of tabs section
    # End of main application condition
