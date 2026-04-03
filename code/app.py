import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OpticVantage | Image Processing and Analysis",
    page_icon="🔮",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS (Purple & White Professional Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #4B0082 !important; /* Indigo/Purple */
    }
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Header Styling */
    h1 {
        color: #4B0082;
        text-align: center;
        font-weight: 800;
        padding-bottom: 0px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f3f0f7;
        padding: 12px;
        border-radius: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 10px;
        color: #4B0082;
        font-weight: 600;
        border: 1px solid #e1d9eb;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4B0082 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(75, 0, 130, 0.2);
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #e1d9eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-value { font-size: 1.7rem; color: #4B0082; font-weight: 800; }
    .metric-label { font-size: 0.8rem; color: #7b68ee; text-transform: uppercase; letter-spacing: 1px; }

    /* Clean Dividers */
    hr { margin-top: 1rem; margin-bottom: 1rem; border-color: #e1d9eb; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PROCESSING LOGIC
# ─────────────────────────────────────────────
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return img_rgb, img_gray

def gamma_correction(image, gamma):
    table = np.array([(i / 255.0) ** gamma * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)

def apply_enhancement(img_gray):
    # Professional pipeline: Noise reduction -> Contrast -> Sharpness
    denoised = cv2.fastNlMeansDenoising(img_gray, h=7)
    equalized = cv2.equalizeHist(denoised)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(equalized, -1, kernel)

# ─────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────
st.markdown("<h1>OpticVantage</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#7b68ee;'>Image Processing & Analysis Suite</p>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Core Controls")
    uploaded = st.file_uploader("Upload Source Asset", type=["jpg","jpeg","png","tif"])
    
    st.markdown("---")
    st.markdown("### 🛠️ Image Overrides")
    gamma_val = st.slider("Luminance Curve (Gamma)", 0.1, 3.0, 1.0)
    rotate_ang = st.selectbox("Fixed Axis Rotation", [0, 90, 180, 270])
    scale_val = st.select_slider("Resolution Scaling", options=[0.25, 0.5, 1.0, 1.5, 2.0], value=1.0)
    
    st.markdown("---")
    st.caption("OpticVantage Engine v2.4.0")

if uploaded is None:
    st.info("👋 Welcome. Please upload an image in the sidebar to begin analysis.")
    st.stop()

# LOAD DATA
img_rgb, img_gray = load_image(uploaded)
h, w = img_gray.shape

# METRICS
m1, m2, m3, m4 = st.columns(4)
metrics = [
    (f"{w}×{h}", "Resolution"),
    (str(img_gray.dtype), "Format"),
    (f"{round(w*h/1e6, 2)} MP", "Data Volume"),
    (f"{round(uploaded.size/1024, 1)} KB", "File Weight")
]
for col, (v, l) in zip([m1, m2, m3, m4], metrics):
    col.markdown(f"<div class='metric-card'><div class='metric-value'>{v}</div><div class='metric-label'>{l}</div></div>", unsafe_allow_html=True)

# TABS
tabs = st.tabs([
    "Data Import",
    "Bitstream Sampling",
    "Spatial Geometry",
    "Intensity Mapping",
    "Statistical Profile",
    "Intelligent Pipeline"
])

# 1. DATA IMPORT
with tabs[0]:
    st.markdown("### Source Acquisition")
    c1, c2 = st.columns(2)
    c1.image(img_rgb, caption="Original Chromatic Data", use_container_width=True)
    c2.image(img_gray, caption="Extracted Luminance Channel", use_container_width=True)

# 2. SAMPLING
with tabs[1]:
    st.markdown("### Signal Quantization & Sampling")
    st.write("Analyze how resolution and bit-depth variations impact data integrity.")
    
    # Scale test
    cols = st.columns(3)
    for i, s in enumerate([0.25, 0.5, 1.0]):
        low_res = cv2.resize(img_gray, (0,0), fx=s, fy=s)
        cols[i].image(low_res, caption=f"Sampling Rate: {s*100}%", use_container_width=True)

# 3. GEOMETRY
with tabs[2]:
    st.markdown("### Spatial Coordinate Transformations")
    c1, c2 = st.columns(2)
    
    # Rotation logic
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotate_ang, 1.0)
    rotated = cv2.warpAffine(img_gray, matrix, (w, h))
    
    c1.image(rotated, caption=f"Rotated Asset ({rotate_ang}°)", use_container_width=True)
    
    # Perspective/Shear
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    sheared = cv2.warpAffine(img_gray, M, (w,h))
    c2.image(sheared, caption="Affine Perspective Shift", use_container_width=True)

# 4. INTENSITY
with tabs[3]:
    st.markdown("### Radiometric Enhancements")
    c1, c2 = st.columns(2)
    
    c1.image(gamma_correction(img_gray, gamma_val), caption=f"Gamma Corrected (γ={gamma_val})", use_container_width=True)
    
    # Negative
    c2.image(255 - img_gray, caption="Inverted Polarity (Negative)", use_container_width=True)

# 5. STATISTICS
with tabs[4]:
    st.markdown("### Pixel Intensity Distribution")
    equalized = cv2.equalizeHist(img_gray)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(img_gray.ravel(), 256, [0,256], color='#4B0082', alpha=0.6, label='Original')
    ax.hist(equalized.ravel(), 256, [0,256], color='#7b68ee', alpha=0.4, label='Equalized')
    ax.set_title("Luminance Histogram", color="#4B0082")
    ax.legend()
    st.pyplot(fig)
    st.success("Analysis: Equalization successfully redistributed the global contrast.")

# 6. PIPELINE
with tabs[5]:
    st.markdown("### End-to-End Production Pipeline")
    st.write("Sequence: Adaptive Denoising → Global Equalization → Laplacian Sharpening")
    
    enhanced = apply_enhancement(img_gray)
    
    col_a, col_b = st.columns(2)
    col_a.image(img_gray, caption="Input Data", use_container_width=True)
    col_b.image(enhanced, caption="Optimized Output", use_container_width=True)
    
    # Download
    buf = io.BytesIO()
    Image.fromarray(enhanced).save(buf, format="PNG")
    st.download_button(
        label="📥 Download Production Asset",
        data=buf.getvalue(),
        file_name="opticvantage_processed.png",
        mime="image/png",
        use_container_width=True
    )

# FOOTER
st.markdown("<div style='text-align:center; padding:30px; color:#aaa;'>OpticVantage Analytics Suite | Powered by Python & OpenCV</div>", unsafe_allow_html=True)
