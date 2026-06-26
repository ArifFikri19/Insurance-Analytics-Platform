# ---------------------------------------------------------
# Insurance Quarterly Claim Prediction Project
# Step 9: Streamlit Web Dashboard - Modern UI Edition
# Currency: Malaysian Ringgit (RM)
# Features: Year Filter + Compare Years Side-by-Side
# ---------------------------------------------------------

import os
import sys
import time
import shutil
import base64
from html import escape

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import preprocessing
import feature_engineering
import aggregation
import split_data
import train_models
import tuning
import evaluate_models
import predict

# Force reload modules (ensures latest code is used after pipeline changes)
import importlib
importlib.reload(preprocessing)
importlib.reload(feature_engineering)
importlib.reload(aggregation)
importlib.reload(split_data)
importlib.reload(train_models)
importlib.reload(tuning)
importlib.reload(evaluate_models)
importlib.reload(predict)


def load_image_asset(relative_path, mime_type=None):
    """Load a local image asset as an embeddable image tag."""
    path = os.path.join(_SCRIPT_DIR, relative_path)
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        if mime_type is None:
            ext = os.path.splitext(relative_path)[1].lower()
            mime_type = {
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(ext, "image/png")
        return f'<img src="data:{mime_type};base64,{encoded}" alt="Insurance Analytics logo" />'
    except OSError:
        return ""

# Page config
st.set_page_config(
    page_title="Insurance Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS Styling
st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        padding: 2rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        border-right: 2px solid #e94560;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Custom Headers */
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 3rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    h2 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #e94560;
        font-size: 2rem !important;
        margin-top: 2rem;
        border-left: 4px solid #e94560;
        padding-left: 1rem;
    }
    
    h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #48cae4;
        font-size: 1.5rem !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace;
        font-size: 2rem !important;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif;
        font-size: 0.9rem !important;
        font-weight: 600;
        color: #90cdf4 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Metric Container */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.95) 0%, rgba(15, 52, 96, 0.95) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid rgba(72, 202, 228, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #e94560;
        box-shadow: 0 12px 40px rgba(233, 69, 96, 0.4);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Outfit', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e94560 0%, #ff6b9d 100%);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(233, 69, 96, 0.6);
    }
    
    /* Radio Buttons (Navigation) */
    .stRadio > label {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #90cdf4;
    }
    
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stRadio > div > label:hover {
        background: rgba(72, 202, 228, 0.1);
        border-color: #48cae4;
    }
    
    .stRadio > div > label > div[data-checked="true"] {
        background: linear-gradient(135deg, #e94560 0%, #ff6b9d 100%);
        border-color: #e94560;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: rgba(22, 33, 62, 0.8);
        border: 2px solid rgba(72, 202, 228, 0.3);
        border-radius: 10px;
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
    }
    
    /* Data Tables */
    .dataframe {
        background: rgba(22, 33, 62, 0.8);
        border-radius: 10px;
        overflow: hidden;
        border: 2px solid rgba(72, 202, 228, 0.3);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        padding: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }
    
    .dataframe tbody tr {
        background: rgba(22, 33, 62, 0.6);
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(72, 202, 228, 0.1);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(22, 33, 62, 0.8);
        border: 2px dashed rgba(72, 202, 228, 0.5);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #e94560;
        background: rgba(233, 69, 96, 0.1);
    }
    
    /* Info/Warning/Success Boxes */
    .stAlert {
        background: rgba(22, 33, 62, 0.9);
        border-left: 4px solid;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(22, 33, 62, 0.8);
        border: 2px solid rgba(72, 202, 228, 0.3);
        border-radius: 10px;
        color: #48cae4;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #e94560;
        background: rgba(233, 69, 96, 0.1);
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        background: rgba(22, 33, 62, 0.8);
        border: 2px solid rgba(72, 202, 228, 0.3);
        border-radius: 8px;
        color: white;
        font-family: 'Space Mono', monospace;
        padding: 0.5rem;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48cae4 0%, #0096c7 100%);
        box-shadow: 0 4px 15px rgba(72, 202, 228, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(22, 33, 62, 0.5);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 2px solid transparent;
        border-radius: 8px;
        color: #90cdf4;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(72, 202, 228, 0.1);
        border-color: #48cae4;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #764ba2;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(22, 33, 62, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #e94560 0%, #ff6b9d 100%);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main > div {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Custom Cards */
    .custom-card {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.95) 0%, rgba(15, 52, 96, 0.95) 100%);
        border: 2px solid rgba(72, 202, 228, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: #667eea;
    }
    
    /* Feature Badge */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Premium enterprise theme overrides.
# These styles intentionally do not change app logic or page features.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --iap-bg: #0F172A;
        --iap-panel: rgba(15, 23, 42, 0.78);
        --iap-panel-strong: rgba(30, 41, 59, 0.88);
        --iap-border: rgba(226, 232, 240, 0.16);
        --iap-border-strong: rgba(6, 182, 212, 0.38);
        --iap-text: #FFFFFF;
        --iap-muted: #CBD5E1;
        --iap-soft: #94A3B8;
        --iap-blue: #2563EB;
        --iap-cyan: #06B6D4;
        --iap-emerald: #10B981;
        --iap-amber: #F59E0B;
        --iap-red: #EF4444;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        letter-spacing: 0 !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 24% 8%, rgba(37, 99, 235, 0.18), transparent 32%),
            linear-gradient(135deg, #0F172A 0%, #111827 52%, #0B1120 100%) !important;
        color: var(--iap-text);
    }

    .main, section.main {
        background: transparent !important;
        padding: 1.5rem 2.25rem 2.5rem 2.25rem !important;
    }

    .block-container {
        max-width: 1500px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.92)) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.14) !important;
        box-shadow: 18px 0 48px rgba(0, 0, 0, 0.30) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    h1 {
        color: var(--iap-text) !important;
        background: none !important;
        -webkit-text-fill-color: var(--iap-text) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: clamp(2.1rem, 2.5vw, 3.25rem) !important;
        font-weight: 800 !important;
        line-height: 1.05 !important;
        text-shadow: none !important;
        margin-bottom: 1.15rem !important;
    }

    h2 {
        color: var(--iap-text) !important;
        border-left: 4px solid var(--iap-cyan) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.55rem !important;
        font-weight: 750 !important;
        margin-top: 2rem !important;
    }

    h3 {
        color: var(--iap-cyan) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 750 !important;
    }

    p, label, span, div {
        font-family: 'Inter', sans-serif !important;
    }

    div[data-testid="metric-container"] {
        background:
            linear-gradient(145deg, rgba(30, 41, 59, 0.88), rgba(15, 23, 42, 0.78)) !important;
        border: 1px solid var(--iap-border) !important;
        border-radius: 8px !important;
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(18px) !important;
        padding: 1.25rem 1.35rem !important;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px) !important;
        border-color: var(--iap-border-strong) !important;
        box-shadow: 0 24px 60px rgba(37, 99, 235, 0.22) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--iap-muted) !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--iap-text) !important;
        background: none !important;
        -webkit-text-fill-color: var(--iap-text) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.75rem !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 46px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(226, 232, 240, 0.16) !important;
        background: linear-gradient(135deg, var(--iap-blue), #1D4ED8) !important;
        color: #FFFFFF !important;
        box-shadow: 0 14px 34px rgba(37, 99, 235, 0.28) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 750 !important;
        letter-spacing: 0.02em !important;
        text-transform: none !important;
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        border-color: rgba(6, 182, 212, 0.55) !important;
        box-shadow: 0 18px 42px rgba(6, 182, 212, 0.22) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--iap-cyan), var(--iap-blue)) !important;
        box-shadow: 0 18px 42px rgba(6, 182, 212, 0.26) !important;
    }

    .stButton > button:disabled,
    .stDownloadButton > button:disabled {
        opacity: 0.48 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    [data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stMultiSelect [data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.82) !important;
        border: 1px solid rgba(6, 182, 212, 0.38) !important;
        border-radius: 8px !important;
        color: var(--iap-text) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    }

    [data-baseweb="select"] > div:focus-within,
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: var(--iap-cyan) !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.16) !important;
    }

    [data-testid="stFileUploader"] {
        background:
            linear-gradient(145deg, rgba(30, 41, 59, 0.86), rgba(15, 23, 42, 0.72)) !important;
        border: 1.5px dashed rgba(6, 182, 212, 0.52) !important;
        border-radius: 8px !important;
        padding: 1.35rem !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 20px 52px rgba(0, 0, 0, 0.20) !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--iap-cyan) !important;
        background: rgba(30, 41, 59, 0.92) !important;
    }

    [data-testid="stDataFrame"],
    .dataframe {
        border: 1px solid var(--iap-border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.18) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.76) !important;
        border: 1px solid var(--iap-border) !important;
        border-radius: 8px !important;
        padding: 0.35rem !important;
        gap: 0.35rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        color: var(--iap-muted) !important;
        border: 1px solid transparent !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(37, 99, 235, 0.22) !important;
        border-color: rgba(6, 182, 212, 0.35) !important;
        color: #FFFFFF !important;
    }

    .stRadio > div > label {
        background: rgba(30, 41, 59, 0.58) !important;
        border: 1px solid rgba(226, 232, 240, 0.08) !important;
        border-radius: 8px !important;
        padding: 0.72rem 0.9rem !important;
    }

    .stRadio > div > label:hover {
        background: rgba(37, 99, 235, 0.18) !important;
        border-color: rgba(6, 182, 212, 0.34) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--iap-border) !important;
        border-radius: 8px !important;
        background: rgba(15, 23, 42, 0.58) !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18) !important;
    }

    [data-testid="stExpander"] summary {
        color: var(--iap-text) !important;
        font-weight: 750 !important;
    }

    .stAlert {
        border-radius: 8px !important;
        border: 1px solid var(--iap-border) !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16) !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, var(--iap-cyan), var(--iap-blue)) !important;
        border-radius: 999px !important;
    }

    hr {
        border-color: rgba(226, 232, 240, 0.12) !important;
    }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.65); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--iap-blue), var(--iap-cyan));
        border-radius: 999px;
        border: 2px solid rgba(15, 23, 42, 0.85);
    }

    @media (max-width: 900px) {
        .main, section.main { padding: 1rem !important; }
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# Final layout polish: compact fixed sidebar, consistent spacing, and widget overflow fixes.
st.markdown("""
<style>
    :root {
        --iap-radius: 8px;
        --iap-sidebar-width: 300px;
        --iap-content-max: 1440px;
    }

    .main, section.main {
        padding: 1.25rem 1.75rem 2.25rem !important;
    }

    .block-container {
        max-width: var(--iap-content-max) !important;
        padding: 1.75rem 1.75rem 2.75rem !important;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.95rem !important;
    }

    hr {
        margin: 1.35rem 0 !important;
        border-color: rgba(226, 232, 240, 0.12) !important;
    }

    [data-testid="stSidebar"] {
        width: var(--iap-sidebar-width) !important;
        min-width: var(--iap-sidebar-width) !important;
        max-width: var(--iap-sidebar-width) !important;
        overflow: hidden !important;
        scrollbar-width: none !important;
    }

    [data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 0 !important;
        height: 0 !important;
        display: none !important;
    }

    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        height: 100vh !important;
        overflow: hidden !important;
        padding: 0.15rem 1rem 0.85rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    [data-testid="stSidebarCollapseButton"],
    button[aria-label*="sidebar"],
    button[title*="sidebar"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h1 * {
        font-size: 1.82rem !important;
        line-height: 1.08 !important;
        margin: -0.1rem 0 0.42rem !important;
        white-space: normal !important;
        overflow: visible !important;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.68rem 0 !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.38rem !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        width: 100% !important;
        min-height: 39px !important;
        max-height: 39px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0.46rem 0.68rem !important;
        border-radius: var(--iap-radius) !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        flex: 0 0 auto !important;
        margin-right: 0.48rem !important;
    }

    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] .stRadio label div {
        font-size: 0.88rem !important;
        line-height: 1.15 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > div[style*="text-align: center"][style*="padding"] {
        padding: 0.62rem !important;
        border-radius: var(--iap-radius) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.66rem !important;
        line-height: 1.16 !important;
        margin: 0.08rem 0 !important;
    }

    .sidebar-footer {
        position: fixed !important;
        left: 1rem !important;
        bottom: 0.65rem !important;
        width: calc(var(--iap-sidebar-width) - 2rem) !important;
        text-align: center !important;
        padding: 0.72rem 0.65rem !important;
        background: rgba(37, 99, 235, 0.14) !important;
        border-radius: var(--iap-radius) !important;
        border: 1px solid rgba(102, 126, 234, 0.45) !important;
        box-sizing: border-box !important;
        z-index: 10 !important;
    }

    .sidebar-footer p {
        margin: 0.08rem 0 !important;
        line-height: 1.18 !important;
    }

    [data-testid="stSidebar"] .stRadio {
        margin-bottom: 6.4rem !important;
    }

    /* Reference-style sidebar refresh */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 20% 0%, rgba(37, 99, 235, 0.24), transparent 34%),
            linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.96)) !important;
        border-right: 1px solid rgba(6, 182, 212, 0.22) !important;
        box-shadow: 18px 0 44px rgba(0, 0, 0, 0.28) !important;
    }

    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.45rem 1.05rem 0.9rem !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 35px;
        margin-bottom: 0.2rem;
    }

    .sidebar-brand-left {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
        min-width: 0;
    }

    .sidebar-logo {
        width: 65px;
        height: 65px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
        flex: 0 0 auto;
        overflow: hidden;
    }

    .sidebar-logo svg {
        width: 36px !important;
        height: 36px !important;
        display: block !important;
        flex: 0 0 auto !important;
    }

    .sidebar-logo img {
        width: 65px !important;
        height: 65px !important;
        display: block !important;
        object-fit: contain !important;
    }

    .page-title-with-logo {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 1rem !important;
        margin: 0 0 1.15rem !important;
        color: #FFFFFF !important;
        font-size: clamp(2.1rem, 2.5vw, 3.25rem) !important;
        font-weight: 800 !important;
        line-height: 1.05 !important;
    }

    .page-title-with-logo img {
        width: 70px !important;
        height: 70px !important;
        object-fit: contain !important;
        flex: 0 0 auto !important;
    }

    .sidebar-brand-title {
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.3rem;
        line-height: 1.2;
        letter-spacing: .15rem !important;
    }

    .sidebar-collapse-mark {
        color: #94A3B8;
        font-size: 1rem;
        opacity: 0.85;
    }

    .sidebar-section-label {
        color: #94A3B8;
        font-size: 0.68rem;
        font-weight: 650;
        margin: 0.45rem 0 0.55rem;
        text-transform: none;
    }

    [data-testid="stSidebar"] hr {
        display: none !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.34rem !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        min-height: 38px !important;
        max-height: 38px !important;
        padding: 0.48rem 0.6rem !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        background: transparent !important;
        color: #E5E7EB !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(37, 99, 235, 0.16) !important;
        border-color: transparent !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.28), rgba(6, 182, 212, 0.14)) !important;
        border-color: rgba(6, 182, 212, 0.28) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        display: none !important;
    }

    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] .stRadio label div {
        color: #F8FAFC !important;
        font-size: 1.0rem !important;
        font-weight: 650 !important;
    }

    .sidebar-footer {
        left: 1.05rem !important;
        bottom: 0.9rem !important;
        width: calc(var(--iap-sidebar-width) - 2.1rem) !important;
        padding: 0.78rem 0.8rem !important;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.30), rgba(6, 182, 212, 0.16)) !important;
        border: 1px solid rgba(6, 182, 212, 0.30) !important;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.20) !important;
        text-align: left !important;
    }

    .sidebar-footer p:first-child {
        color: #FFFFFF !important;
        font-size: 0.72rem !important;
        font-weight: 850 !important;
    }

    .sidebar-footer p {
        color: #93C5FD !important;
        font-size: 0.64rem !important;
    }

    .custom-card {
        min-height: 305px !important;
        height: 305px !important;
        display: block !important;
        box-sizing: border-box !important;
        padding: 1.65rem !important;
    }

    .custom-card p {
        min-height: 112px !important;
        margin-bottom: 0.9rem !important;
    }

    .custom-card .feature-badge {
        width: auto !important;
        min-width: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0.48rem 0.95rem !important;
        margin: 0.18rem 0.28rem 0.18rem 0 !important;
        border-radius: 999px !important;
        white-space: nowrap !important;
    }

    .custom-card p + .feature-badge {
        margin-left: 0 !important;
    }

    .custom-card .feature-badge + .feature-badge {
        margin-left: 0.45rem !important;
    }

    .feature-badges {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 0.55rem !important;
        flex-wrap: wrap !important;
        width: 100% !important;
    }

    .feature-badges .feature-badge {
        margin: 0 !important;
    }

    .home-feature-card {
        min-height: 235px !important;
        height: 235px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        overflow: hidden !important;
    }

    .home-feature-card h4 {
        min-height: 38px !important;
        display: flex !important;
        align-items: center !important;
        margin-bottom: 0.9rem !important;
    }

    .home-feature-card ul {
        margin: 0 !important;
        padding-left: 1.35rem !important;
    }

    .equal-kpi-card {
        min-height: 170px !important;
        height: 170px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
    }

    .equal-kpi-card .kpi-value {
        max-width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        font-size: clamp(1.65rem, 2.6vw, 2.5rem) !important;
        line-height: 1.12 !important;
    }

    .sim-result-card {
        min-height: 220px !important;
        height: 220px !important;
    }

    .sim-result-card .kpi-value {
        font-size: clamp(1.45rem, 2.35vw, 2.35rem) !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        text-align: center !important;
    }

    div[data-testid="metric-container"],
    [data-testid="stExpander"],
    [data-testid="stFileUploader"],
    [data-testid="stDataFrame"],
    .dataframe,
    .stAlert,
    [data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {
        border-radius: var(--iap-radius) !important;
        box-sizing: border-box !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: var(--iap-radius) !important;
        min-height: 44px !important;
        padding: 0.68rem 1rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    [data-testid="stFileUploader"] {
        padding: 1rem !important;
        overflow: hidden !important;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section {
        min-height: 88px !important;
        display: flex !important;
        align-items: center !important;
        flex-wrap: wrap !important;
        gap: 0.8rem 1rem !important;
        padding: 1rem 1.15rem !important;
        border-radius: var(--iap-radius) !important;
        overflow: visible !important;
    }

    [data-testid="stFileUploader"] button {
        min-width: 112px !important;
        height: 38px !important;
        padding: 0 0.95rem !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        position: static !important;
        transform: none !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stFileUploaderDropzone"] > div,
    [data-testid="stFileUploader"] section > div {
        min-width: 240px !important;
        flex: 1 1 280px !important;
    }

    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p {
        font-size: 0.95rem !important;
        color: inherit !important;
        opacity: 1 !important;
        max-width: none !important;
        overflow: visible !important;
        white-space: nowrap !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
        line-height: 1.3 !important;
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
    }

    [data-testid="stDataFrame"] {
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] canvas {
        border-radius: var(--iap-radius) !important;
    }

    [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
    [data-testid="stDataFrame"] .stElementToolbar,
    [data-testid="stDataFrame"] [data-testid="stToolbar"],
    [data-testid="stDataFrame"] [role="menu"],
    [data-testid="stDataFrame"] [data-baseweb="menu"],
    [data-testid="stDataFrame"] [data-baseweb="popover"],
    [data-testid="stDataFrame"] button[title="Search"],
    [data-testid="stDataFrame"] button[title="Download"],
    [data-testid="stDataFrame"] button[title="Fullscreen"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    [data-testid="stElementToolbar"],
    .stElementToolbar,
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    div[data-baseweb="popover"] [role="menu"],
    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] [aria-label*="column"],
    div[data-baseweb="popover"] [aria-label*="Column"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    [role="menu"]:has([aria-label*="Sort"]),
    [role="menu"]:has([title*="Sort"]),
    [role="menu"]:has([aria-label*="Pin"]),
    [role="menu"]:has([title*="Pin"]),
    [role="menu"]:has([aria-label*="Autosize"]),
    [role="menu"]:has([title*="Autosize"]),
    div:has(> [role="menuitem"]):has([aria-label*="Sort"]),
    div:has(> [role="menuitem"]):has([title*="Sort"]) {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    .material-icons,
    .material-icons-round,
    .material-symbols-rounded,
    .material-symbols-outlined,
    span[class*="material-icons"],
    span[class*="material-symbols"],
    i[class*="material-icons"],
    i[class*="material-symbols"],
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: "liga" !important;
    }

    .pipeline-status-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.88));
        border: 1px solid rgba(6, 182, 212, 0.38);
        border-radius: var(--iap-radius);
        padding: 1.1rem 1.25rem;
        margin: 0.85rem 0;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
    }

    .pipeline-status-card p {
        margin: 0;
    }

    .pipeline-step-list {
        display: grid;
        gap: 0.45rem;
        margin-top: 0.85rem;
    }

    .pipeline-step-row {
        display: grid;
        grid-template-columns: 92px minmax(150px, 220px) minmax(0, 1fr);
        gap: 0.7rem;
        align-items: center;
        padding: 0.58rem 0.7rem;
        border: 1px solid rgba(226, 232, 240, 0.10);
        border-radius: var(--iap-radius);
        background: rgba(15, 23, 42, 0.62);
    }

    .pipeline-step-status {
        font-weight: 800;
        font-size: 0.82rem;
    }

    .pipeline-step-name {
        color: #FFFFFF;
        font-weight: 750;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .pipeline-step-detail {
        color: #CBD5E1;
        font-size: 0.88rem;
        line-height: 1.35;
        overflow-wrap: break-word;
    }

    .pipeline-loader {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 3px solid rgba(6, 182, 212, 0.22);
        border-top-color: #06B6D4;
        display: inline-block;
        animation: iapSpin 900ms linear infinite;
        vertical-align: -4px;
        margin-right: 0.45rem;
    }

    @keyframes iapSpin {
        to { transform: rotate(360deg); }
    }

    @media (max-width: 900px) {
        .pipeline-step-row {
            grid-template-columns: 1fr;
            gap: 0.25rem;
        }
    }

    @media (max-height: 760px) {
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h1 * {
            font-size: 1.45rem !important;
            margin-bottom: 0.45rem !important;
        }

        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.28rem !important;
        }

        [data-testid="stSidebar"] .stRadio > div > label {
            min-height: 36px !important;
            max-height: 36px !important;
            padding: 0.42rem 0.62rem !important;
        }
    }

    @media (max-width: 900px) {
        .main, section.main {
            padding: 0.85rem !important;
        }

        .block-container {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Set matplotlib style
plt.style.use('dark_background')
sns.set_palette("husl")


# =========================================================
# HELPER: Extract year from Quarter_Label
# =========================================================
def extract_year(quarter_label):
    """Extract year from Quarter_Label like '2020-Q1' or '2020Q1' or 'Q1 2020'."""
    import re
    match = re.search(r'(\d{4})', str(quarter_label))
    if match:
        return int(match.group(1))
    return None


def get_available_years(df, label_col='Quarter_Label'):
    """Get sorted list of available years from a dataframe."""
    if label_col not in df.columns:
        return []
    years = df[label_col].apply(extract_year).dropna().unique().astype(int).tolist()
    return sorted(years)


def filter_by_year(df, year, label_col='Quarter_Label'):
    """Filter dataframe by year."""
    if year == "All Years":
        return df
    df_copy = df.copy()
    df_copy['_Year'] = df_copy[label_col].apply(extract_year)
    filtered = df_copy[df_copy['_Year'] == int(year)].drop('_Year', axis=1)
    return filtered


def year_filter_widget(df, label_col='Quarter_Label', key_prefix=""):
    """Display year filter widget and return filtered dataframe."""
    years = get_available_years(df, label_col)
    options = ["All Years"] + [str(y) for y in years]
    selected = st.selectbox(
        "📅 Select Year",
        options,
        index=0,
        key=f"{key_prefix}_year_filter"
    )
    return filter_by_year(df, selected, label_col), selected


def compare_years_widget(df, label_col='Quarter_Label', key_prefix=""):
    """Display compare years widget and return two filtered dataframes."""
    years = get_available_years(df, label_col)
    if len(years) < 2:
        st.warning("⚠️ Need at least 2 years of data for comparison.")
        return None, None, None, None
    
    col1, col2 = st.columns(2)
    with col1:
        year1 = st.selectbox(
            "📅 Year 1",
            [str(y) for y in years],
            index=0,
            key=f"{key_prefix}_compare_year1"
        )
    with col2:
        year2 = st.selectbox(
            "📅 Year 2",
            [str(y) for y in years],
            index=min(1, len(years)-1),
            key=f"{key_prefix}_compare_year2"
        )
    
    df1 = filter_by_year(df, year1, label_col)
    df2 = filter_by_year(df, year2, label_col)
    
    return df1, df2, year1, year2


# =========================================================
# HELPER FUNCTIONS FOR PLOTLY CHARTS
# =========================================================

CHART_PRIMARY = "#06B6D4"
CHART_SECONDARY = "#2563EB"
CHART_ACCENT = "#8B5CF6"
CHART_SUCCESS = "#10B981"
CHART_WARNING = "#F59E0B"
CHART_DANGER = "#EF4444"
CHART_NEUTRAL = "#94A3B8"
CHART_PALETTE = [CHART_PRIMARY, CHART_SECONDARY, CHART_ACCENT, CHART_SUCCESS, CHART_WARNING]
RISK_PALETTE = {"LOW": CHART_SUCCESS, "MEDIUM": CHART_WARNING, "HIGH": CHART_DANGER}


def apply_chart_layout(fig, bottom_margin=110, legend_y=-0.26):
    """Apply consistent chart spacing so axis titles and legends do not overlap."""
    fig.update_layout(
        margin=dict(l=70, r=40, t=80, b=bottom_margin),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=legend_y,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(15, 23, 42, 0)',
        ),
        xaxis=dict(title_standoff=18),
        yaxis=dict(title_standoff=18),
    )
    return fig


def create_plotly_line_chart(df, x_col, y_col, title, x_label, y_label, color=CHART_PRIMARY):
    """Create an interactive Plotly line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        name=y_label,
        line=dict(color=color, width=3),
        marker=dict(size=10, color=color, line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(6, 182, 212, 0.16)'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_dark',
        plot_bgcolor='rgba(22, 33, 62, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Outfit, sans-serif', color='#90cdf4'),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='rgba(22, 33, 62, 0.9)', font_size=14),
        xaxis=dict(showgrid=True, gridcolor='rgba(72, 202, 228, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(72, 202, 228, 0.1)')
    )
    apply_chart_layout(fig, bottom_margin=95, legend_y=-0.22)
    
    return fig


def create_plotly_bar_chart(labels, values, title, color_scale='Viridis'):
    """Create an interactive Plotly bar chart"""
    fig = go.Figure()
    
    n = len(labels)
    colors = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(n)]
    
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=values,
        texttemplate='%{text:.2s}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
        template='plotly_dark',
        plot_bgcolor='rgba(22, 33, 62, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Outfit, sans-serif', color='#90cdf4'),
        showlegend=False,
        hovermode='x',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(72, 202, 228, 0.1)')
    )
    fig.update_layout(margin=dict(l=70, r=40, t=80, b=80), xaxis=dict(title_standoff=18), yaxis=dict(title_standoff=18))
    
    return fig


def create_plotly_pie_chart(labels, values, title):
    """Create an interactive Plotly pie chart"""
    fig = go.Figure()
    
    colors = [CHART_PALETTE[i % len(CHART_PALETTE)] for i in range(len(labels))]
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=14, family='Outfit, sans-serif'),
        hole=0.4
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Outfit, sans-serif', color='#90cdf4'),
        showlegend=True,
    )
    apply_chart_layout(fig, bottom_margin=105, legend_y=-0.20)
    
    return fig


# =========================================================
# SIDEBAR
# =========================================================
if "pending_nav_page" in st.session_state:
    st.session_state.nav_page = st.session_state.pending_nav_page
    del st.session_state.pending_nav_page

sidebar_logo_svg = load_image_asset("icons/logo.png")

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand-left">
            <div class="sidebar-logo">{sidebar_logo_svg}</div>
            <div class="sidebar-brand-title">Insurance<br>Analytics</div>
        </div>
    </div>
    <div class="sidebar-section-label">Main Menu</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation", 
        [
        "🏠 Home",
        "📤 Upload Data",
        "📈 Quarterly Trends",
        "👤 Customer Risk",
        "🚨 Incident Patterns",
        "💰 Claim Breakdown",
        "🔮 Predictions & Risk",
        "📡 Next Quarter Forecast",
        "⚖️ Model Comparison",
        "🎛️ What-If Simulator"
    ], label_visibility="collapsed", key="nav_page")

    st.markdown("""
    <div class='sidebar-footer'>
        <p>Insurance Analytics Platform v1.0</p>
        <p>Powered by AI & ML</p>
        <p>Currency: Malaysian Ringgit (RM)</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
if 'pipeline_ran' not in st.session_state:
    st.session_state.pipeline_ran = False
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
if 'stop_pipeline_requested' not in st.session_state:
    st.session_state.stop_pipeline_requested = False
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False


def check_outputs():
    """Check if pipeline outputs exist."""
    required = [
        "outputs/quarterly_claims.csv",
        "outputs/predictions_80_20.csv",
        "outputs/customer_risk_scores.csv",
        "outputs/quarterly_incident_patterns.csv",
        "outputs/quarterly_claim_breakdown.csv",
        "outputs/evaluation_master.csv"
    ]
    return all(os.path.exists(f) for f in required)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def show_done_notification(message):
    try:
        st.toast(message, icon="✅")
    except Exception:
        pass
    st.success(message)


def request_pipeline_stop():
    st.session_state.stop_pipeline_requested = True


def render_process_status(status_panel, pipeline_steps, completed_steps, start_time, current_idx=None, failed_idx=None):
    elapsed = time.time() - start_time
    done_count = len(completed_steps)
    total_steps = len(pipeline_steps)
    current_pct = 0

    if current_idx is not None:
        current_pct = pipeline_steps[current_idx][2]
    elif completed_steps:
        current_pct = pipeline_steps[max(completed_steps)][2]

    estimated_remaining = "Calculating..."
    if current_pct > 0 and done_count < total_steps:
        total_estimate = elapsed / (current_pct / 100)
        estimated_remaining = format_duration(total_estimate - elapsed)
    elif done_count == total_steps:
        estimated_remaining = "0s"

    if failed_idx is not None:
        headline = f"Stopped at step {failed_idx + 1}: {pipeline_steps[failed_idx][0]}"
    elif current_idx is not None:
        headline = f"<span class='pipeline-loader'></span>Running step {current_idx + 1}/{total_steps}: {pipeline_steps[current_idx][0]}"
    elif done_count == total_steps:
        headline = "Completed successfully"
    else:
        headline = "Preparing process"

    rows = []
    for idx, (name, detail, _) in enumerate(pipeline_steps):
        if failed_idx == idx:
            status, color = "Failed", "#EF4444"
        elif idx in completed_steps:
            status, color = "Done", "#10B981"
        elif current_idx == idx:
            status, color = "Running", "#06B6D4"
        else:
            status, color = "Waiting", "#94A3B8"
        rows.append(
            f'<div class="pipeline-step-row">'
            f'<div class="pipeline-step-status" style="color:{color};">{status}</div>'
            f'<div class="pipeline-step-name">{idx + 1}. {name}</div>'
            f'<div class="pipeline-step-detail">{detail}</div>'
            f'</div>'
        )

    status_html = (
        '<div class="pipeline-status-card">'
        '<div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">'
        '<div>'
        f'<p style="color:#FFFFFF; font-size:1.05rem; font-weight:800;">{headline}</p>'
        f'<p style="color:#94A3B8; margin-top:0.3rem;">Elapsed: <strong style="color:#FFFFFF;">{format_duration(elapsed)}</strong> | Estimated remaining: <strong style="color:#FFFFFF;">{estimated_remaining}</strong></p>'
        '</div>'
        f'<div style="color:#06B6D4; font-weight:800;">{done_count}/{total_steps} done</div>'
        '</div>'
        '<div class="pipeline-step-list">'
        f'{"".join(rows)}'
        '</div>'
        '</div>'
    )
    status_panel.markdown(status_html, unsafe_allow_html=True)


def check_data_availability(file_path, required_columns=None):
    if not os.path.exists(file_path):
        return False, False, []
    try:
        df = pd.read_csv(file_path)
        if len(df) == 0:
            return True, False, []
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            return True, len(missing) == 0, missing
        return True, True, []
    except Exception as e:
        return False, False, []


def show_data_unavailable_card(page_name, file_paths=None, message=None, action=None, title="Data Not Available"):
    if file_paths is None:
        file_paths = []
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    safe_page = escape(str(page_name))
    safe_title = escape(str(title))
    safe_action = escape(action or "Please run the complete pipeline from 'Upload & Run Pipeline' to generate this data.")
    if message:
        body = escape(str(message))
    elif file_paths:
        file_list = ", ".join(escape(str(path)) for path in file_paths)
        noun = "data file" if len(file_paths) == 1 else "data files"
        body = f"The <strong>{safe_page}</strong> section requires {noun}: <strong style='color:#F43F5E;'>{file_list}</strong>"
    else:
        body = f"The <strong>{safe_page}</strong> section requires generated pipeline data before it can be displayed."

    st.markdown(f"""
    <div style='
        position: relative;
        width: 100%;
        background: linear-gradient(135deg, rgba(255, 159, 64, 0.19), rgba(244, 63, 94, 0.22));
        border: 2px solid #ff9f40;
        padding: 3rem 2.5rem;
        border-radius: 16px;
        min-height: 265px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-shadow: none;
        margin: 1rem 0 1.5rem 0;
    '>
        <div style='max-width: 1120px; width: 100%;'>
            <div style='display:flex; align-items:center; justify-content:center; gap:0.8rem; margin-bottom:1.8rem;'>
                <span style='font-size:1.9rem; line-height:1;'>&#128193;</span>
                <div style='color:#FFFFFF; margin:0; font-size:1.8rem; font-weight:800; line-height:1.2;'>{safe_title}</div>
            </div>
            <p style='color:#7DD3FC; margin:0 auto; font-size:1.2rem; line-height:1.7; font-weight:500;'>
                {body}
            </p>
            <p style='color:#22D3EE; margin:1.8rem 0 0 0; font-size:1.02rem; line-height:1.7;'>
                <span style='color:#FBBF24;'>&#128161;</span> {safe_action}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_data_unavailable_message(page_name, file_path, required_columns=None):
    exists, has_data, missing_cols = check_data_availability(file_path, required_columns)

    if not exists:
        show_data_unavailable_card(page_name, file_path)
        return False

    if not has_data:
        show_data_unavailable_card(
            page_name,
            file_path,
            message=f"The data file is empty: {file_path}",
            title="No Data Available"
        )
        return False

    if missing_cols:
        show_data_unavailable_card(
            page_name,
            file_path,
            message=f"The data file is incomplete. Missing columns: {', '.join(missing_cols)}",
            action="Please rerun the pipeline or review the uploaded dataset column mapping.",
            title="Incomplete Data"
        )
        return False

    return True


# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":
    st.markdown(f"""
    <div class="page-title-with-logo">
        {sidebar_logo_svg}
        <span>Insurance Analytics Platform</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <p style='font-size: 1.2rem; color: #90cdf4; line-height: 1.8;'>
            Welcome to the next-generation <span style='color: #667eea; font-weight: 700;'>Insurance Claim Prediction Dashboard</span>.
            <br>Harness the power of machine learning to predict quarterly claims and identify risks.
            <br><span style='color: #48cae4; font-weight: 600;'>All values displayed in Malaysian Ringgit (RM)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #667eea; margin-top: 0;'>📊 Advanced Analytics</h3>
            <p style='color: #90cdf4; line-height: 1.6;'>
                Deep dive into quarterly trends, seasonal patterns, and historical claim data with interactive visualizations.
            </p>
            <div class='feature-badges'>
                <span class='feature-badge'>Trend Analysis</span>
                <span class='feature-badge'>Seasonality</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #e94560; margin-top: 0;'>🤖 ML Predictions</h3>
            <p style='color: #90cdf4; line-height: 1.6;'>
                Multiple machine learning models including Random Forest, XGBoost, and Ensemble methods for accurate predictions.
            </p>
            <div class='feature-badges'>
                <span class='feature-badge'>Multi-Model</span>
                <span class='feature-badge'>Ensemble</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #48cae4; margin-top: 0;'>⚡ Risk Detection</h3>
            <p style='color: #90cdf4; line-height: 1.6;'>
                Automated risk flagging system that identifies LOW, MEDIUM, and HIGH risk quarters for proactive management.
            </p>
            <div class='feature-badges'>
                <span class='feature-badge'>Auto-Flag</span>
                <span class='feature-badge'>Real-time</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🎯 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='home-feature-card' style='background: rgba(22, 33, 62, 0.6); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;'>
            <h4 style='color: #667eea; margin-top: 0;'>📈 Quarterly Trend Analysis</h4>
            <ul style='color: #90cdf4; line-height: 1.8;'>
                <li>Historical claim trends with interactive charts</li>
                <li>Seasonality detection and patterns</li>
                <li>Year-over-year comparisons</li>
                <li><strong>Year filter & Compare Years</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='home-feature-card' style='background: rgba(22, 33, 62, 0.6); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #e94560; margin-top: 1rem;'>
            <h4 style='color: #e94560; margin-top: 0;'>🚨 Incident Pattern Analysis</h4>
            <ul style='color: #90cdf4; line-height: 1.8;'>
                <li>Most common and costly incident types</li>
                <li>Quarterly incident distribution</li>
                <li>Cost breakdown by incident category</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='home-feature-card' style='background: rgba(22, 33, 62, 0.6); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #48cae4;'>
            <h4 style='color: #48cae4; margin-top: 0;'>👤 Customer Risk Analysis</h4>
            <ul style='color: #90cdf4; line-height: 1.8;'>
                <li>High-risk customer identification</li>
                <li>Risk score distribution</li>
                <li>Customer segmentation by risk level</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='home-feature-card' style='background: rgba(22, 33, 62, 0.6); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #764ba2; margin-top: 1rem;'>
            <h4 style='color: #764ba2; margin-top: 0;'>🎛️ What-If Simulator</h4>
            <ul style='color: #90cdf4; line-height: 1.8;'>
                <li>Interactive feature adjustment</li>
                <li>Real-time prediction updates</li>
                <li>Scenario planning and testing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if check_outputs():
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(46, 213, 115, 0.2), rgba(0, 184, 148, 0.2)); 
                    border: 2px solid #2ed573; padding: 2rem; border-radius: 15px; text-align: center;'>
            <h2 style='color: #2ed573; margin: 0;'>✅ System Ready</h2>
            <p style='color: #90cdf4; margin-top: 1rem; font-size: 1.1rem;'>All pipeline outputs detected.</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.data_ready = True
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255, 159, 64, 0.2), rgba(255, 107, 107, 0.2)); 
                    border: 2px solid #ff9f40; padding: 2rem; border-radius: 15px; text-align: center;'>
            <h2 style='color: #ff9f40; margin: 0;'>⚠️ Pipeline Required</h2>
            <p style='color: #90cdf4; margin-top: 1rem; font-size: 1.1rem;'>Navigate to <strong>'Upload & Run Pipeline'</strong> to begin.</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# UPLOAD & RUN PIPELINE (UPDATED FULL VERSION)
# =========================================================
elif page == "📤 Upload Data":

    st.title("📤 Upload Dataset")

    st.markdown("""
    <div style='background: rgba(72, 202, 228, 0.1);
                border-left: 4px solid #48cae4;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 2rem;'>
        <p style='color: #90cdf4; margin: 0;'>
            Upload your insurance dataset (Excel or CSV format) to run the complete machine learning pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "📁 Choose your dataset file",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True
    )

    generate_png_plots = st.toggle(
        "Generate PNG plot files",
        value=False,
        help="Turn this on only if you need saved PNG charts for reports. Leave off for a faster pipeline."
    )

    # =====================================================
    # UPLOAD FILE SECTION
    # =====================================================
    if uploaded_files:

        import tempfile

        temp_paths = []
        source_candidates = []

        def score_candidate_columns(df):
            mapping = preprocessing.detect_column_mapping(df)
            required_matches = sum(1 for col in preprocessing.REQUIRED_COLUMNS if mapping.get(col) is not None)
            recommended_matches = sum(1 for col in preprocessing.RECOMMENDED_COLUMNS if mapping.get(col) is not None)
            return required_matches, recommended_matches

        def add_source_candidate(file_name, sheet_name, df):
            df = preprocessing.standardize_column_names(df)
            required_matches, recommended_matches = score_candidate_columns(df)
            usable = required_matches == len(preprocessing.REQUIRED_COLUMNS) and len(df) > 0
            source_candidates.append({
                "label": f"{file_name} :: {sheet_name}",
                "file": file_name,
                "sheet": sheet_name,
                "rows": len(df),
                "columns": len(df.columns),
                "required_matches": required_matches,
                "recommended_matches": recommended_matches,
                "usable": usable,
                "df": df,
            })

        try:
            for uploaded_file in uploaded_files:
                file_ext = uploaded_file.name.split(".")[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                    temp_paths.append(tmp_path)

                if file_ext == "csv":
                    add_source_candidate(uploaded_file.name, "CSV", pd.read_csv(tmp_path))
                else:
                    with pd.ExcelFile(tmp_path) as workbook:
                        for sheet_name in workbook.sheet_names:
                            add_source_candidate(
                                uploaded_file.name,
                                sheet_name,
                                pd.read_excel(workbook, sheet_name=sheet_name)
                            )

            st.success(f"Uploaded {len(uploaded_files)} file(s). Found {len(source_candidates)} sheet/table source(s).")
        except Exception as e:
            source_candidates = []
            st.error(f"Upload scan failed: {e}")

        upload_df = None
        tmp_path = None
        column_mapping = {}
        validation_errors = []
        validation_warnings = []
        combine_summary = None

        try:
            if not source_candidates:
                raise ValueError("No readable CSV files or Excel sheets found.")

            source_summary = pd.DataFrame([
                {
                    "Status": "Ready" if c["usable"] else "Needs review",
                    "File": c["file"],
                    "Sheet/Table": c["sheet"],
                    "Rows": c["rows"],
                    "Columns": c["columns"],
                    "Required Fields": f"{c['required_matches']}/{len(preprocessing.REQUIRED_COLUMNS)} found",
                    "Optional Fields": f"{c['recommended_matches']}/{len(preprocessing.RECOMMENDED_COLUMNS)} found",
                }
                for c in source_candidates
            ])
            st.markdown("""
            <div style='display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.65rem; margin: 0.75rem 0 1.15rem;'>
                <div style='background: rgba(37, 99, 235, 0.14); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: 8px; padding: 0.75rem; color: #fff;'><strong>Step 1</strong><br><span style='color:#90cdf4;'>Upload</span></div>
                <div style='background: rgba(37, 99, 235, 0.14); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: 8px; padding: 0.75rem; color: #fff;'><strong>Step 2</strong><br><span style='color:#90cdf4;'>Method</span></div>
                <div style='background: rgba(37, 99, 235, 0.14); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: 8px; padding: 0.75rem; color: #fff;'><strong>Step 3</strong><br><span style='color:#90cdf4;'>Mapping</span></div>
                <div style='background: rgba(37, 99, 235, 0.14); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: 8px; padding: 0.75rem; color: #fff;'><strong>Step 4</strong><br><span style='color:#90cdf4;'>Validate</span></div>
                <div style='background: rgba(37, 99, 235, 0.14); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: 8px; padding: 0.75rem; color: #fff;'><strong>Step 5</strong><br><span style='color:#90cdf4;'>Run</span></div>
            </div>
            """, unsafe_allow_html=True)

            detected_cols = st.columns(4)
            detected_cols[0].metric("Files Uploaded", len(uploaded_files))
            detected_cols[1].metric("Sources Found", len(source_candidates))
            detected_cols[2].metric("Rows Detected", f"{sum(c['rows'] for c in source_candidates):,}")
            detected_cols[3].metric("Ready Sources", f"{sum(1 for c in source_candidates if c['usable'])}/{len(source_candidates)}")

            st.markdown("### Detected Files & Sheets")
            st.caption("Select sources that contain claims data. Use append mode for same-structure files, or join mode when details are split across files.")
            with st.expander("What fields does the app look for?", expanded=False):
                req = ", ".join(preprocessing.REQUIRED_COLUMNS.keys())
                opt = ", ".join([col for col in preprocessing.RECOMMENDED_COLUMNS.keys() if col != "Claim ID"])
                st.markdown(f"""
                **Required fields** are the minimum fields needed to run forecasting:
                {req}

                **Join field** is used when merging related files:
                Claim ID

                **Optional fields** are not required to run, but they improve customer risk, claim breakdown, incident analysis, and dashboard detail:
                {opt}
                """)
            st.table(source_summary.reset_index(drop=True))

            if len(source_candidates) > 1:
                upload_combine_mode = st.radio(
                    "Choose combine method",
                    ["Append rows", "Join files by shared key"],
                    horizontal=True,
                    format_func=lambda mode: "Stack similar files" if mode == "Append rows" else "Merge related files",
                    key="upload_combine_mode",
                    help="Append rows stacks similar files. Join files by shared key merges split tables such as policy details plus claim details."
                )
                if upload_combine_mode == "Append rows":
                    st.caption("Stack similar files when every selected source already contains complete claim rows.")
                else:
                    st.caption("Merge related files when claim details are split across sources and linked by a shared key.")
            else:
                upload_combine_mode = "Append rows"

            default_sources = [c["label"] for c in source_candidates if c["usable"]]
            if not default_sources:
                default_sources = [source_candidates[0]["label"]]

            if len(source_candidates) == 1:
                selected_sources = [source_candidates[0]["label"]]
                st.info(f"Using source: {source_candidates[0]['file']} :: {source_candidates[0]['sheet']}")
            else:
                selected_sources = st.multiselect(
                    "Sources to combine",
                    options=[c["label"] for c in source_candidates],
                    default=default_sources,
                    key="upload_sources_to_combine"
                )
            selected_candidates = [c for c in source_candidates if c["label"] in selected_sources]
            if not selected_candidates:
                raise ValueError("Select at least one sheet/table source to process.")
            if upload_combine_mode == "Join files by shared key" and len(selected_candidates) < 2:
                raise ValueError("Join mode needs at least two selected sources.")

            required_cols = list(preprocessing.REQUIRED_COLUMNS.keys())
            join_cols = ["Claim ID"] if "Claim ID" in preprocessing.RECOMMENDED_COLUMNS else []
            optional_cols = [col for col in preprocessing.RECOMMENDED_COLUMNS.keys() if col not in join_cols]
            standardized_sources = []
            source_validation_rows = []
            source_level_errors = []
            source_mapping_rows = []

            st.markdown("### Per-Source Column Mapping")
            st.caption("Confirm required fields first. Optional fields are hidden by default to keep this section focused.")
            if upload_combine_mode == "Join files by shared key":
                st.info("Merge mode tip: map the shared key in each file to `Claim ID`, then confirm the key column in Join Settings.")
            
            for c in selected_candidates:
                detected_source_mapping = preprocessing.detect_column_mapping(c["df"])
                source_mapping = {}
                mapping_options = ["-- Missing --"] + list(c["df"].columns)

                def render_mapping_fields(fields, columns_per_row=2):
                    if not fields:
                        return
                    map_cols = st.columns(columns_per_row)
                    for i, standard_col in enumerate(fields):
                        detected_col = detected_source_mapping.get(standard_col)
                        default_idx = mapping_options.index(detected_col) if detected_col in mapping_options else 0
                        with map_cols[i % columns_per_row]:
                            selected_col = st.selectbox(
                                standard_col,
                                mapping_options,
                                index=default_idx,
                                key=f"upload_map_{c['label']}_{standard_col}"
                            )
                            source_mapping[standard_col] = None if selected_col == "-- Missing --" else selected_col

                with st.expander(f"{c['file']} :: {c['sheet']} ({c['rows']:,} rows)", expanded=not c["usable"]):
                    st.caption("Confirm this source's column names before it is combined with the others.")
                    st.markdown("**Required fields**")
                    render_mapping_fields(required_cols)
                    if upload_combine_mode == "Join files by shared key":
                        st.markdown("**Join field**")
                        render_mapping_fields(join_cols, columns_per_row=1)
                    with st.expander("Optional fields", expanded=False):
                        optional_fields = optional_cols if upload_combine_mode == "Join files by shared key" else join_cols + optional_cols
                        render_mapping_fields(optional_fields)

                source_df = preprocessing.standardize_uploaded_columns(c["df"], source_mapping)
                src_errors, src_warnings, _ = preprocessing.validate_uploaded_dataset(
                    source_df,
                    source_mapping
                )
                blocking_src_errors = src_errors if upload_combine_mode == "Append rows" else []
                if blocking_src_errors:
                    status = "Error"
                    source_level_errors.extend([f"{c['label']}: {err}" for err in blocking_src_errors])
                elif src_errors and upload_combine_mode == "Join files by shared key":
                    status = "Needs join"
                elif src_warnings:
                    status = "Warning"
                else:
                    status = "OK"

                source_validation_rows.append({
                    "Status": status,
                    "File": c["file"],
                    "Sheet/Table": c["sheet"],
                    "Rows": c["rows"],
                    "Required Matches": c["required_matches"],
                    "Issues": "; ".join(src_errors or src_warnings[:3]) if (src_errors or src_warnings) else "Ready",
                })
                standardized_sources.append(
                    source_df.assign(Source_File=c["file"], Source_Sheet=c["sheet"])
                )
                source_mapping_rows.append({
                    "label": c["label"],
                    "file": c["file"],
                    "sheet": c["sheet"],
                    "df": source_df.assign(Source_File=c["file"], Source_Sheet=c["sheet"]),
                    "raw_columns": list(c["df"].columns),
                })

            if source_level_errors:
                validation_errors.extend(source_level_errors)

            st.markdown("### Source-by-Source Validation")
            st.caption("Each selected file/sheet is checked individually before the combined dataset is processed.")
            st.table(pd.DataFrame(source_validation_rows).reset_index(drop=True))

            if upload_combine_mode == "Append rows":
                upload_df = pd.concat(standardized_sources, ignore_index=True, sort=False)
                combine_summary = {
                    "Mode": "Stack similar files",
                    "Input Rows": sum(len(df) for df in standardized_sources),
                    "Output Rows": len(upload_df),
                    "Duplicate Keys": None,
                    "Join Warnings": 0,
                }
            else:
                st.markdown("### Join Settings")
                st.caption(
                    "Choose which column is the key in each selected file/sheet. The key names can be different "
                    "before mapping, but after mapping they should point to the same identifier."
                )

                join_key_options = {}
                join_key_rows = []
                for source in source_mapping_rows:
                    options = list(source["df"].columns)
                    default_key = "Claim ID" if "Claim ID" in options else ("Customer ID" if "Customer ID" in options else options[0])
                    default_idx = options.index(default_key) if default_key in options else 0
                    key_cols = st.columns([2, 1])
                    with key_cols[0]:
                        st.markdown(
                            f"**{source['file']}**  \n"
                            f"<span style='color:#90cdf4;'>Sheet/Table: {source['sheet']} | Rows: {len(source['df']):,}</span>",
                            unsafe_allow_html=True
                        )
                    with key_cols[1]:
                        join_key_options[source["label"]] = st.selectbox(
                            "Key column",
                            options,
                            index=default_idx,
                            key=f"join_key_{source['label']}"
                        )
                    join_key_rows.append({
                        "File": source["file"],
                        "Sheet/Table": source["sheet"],
                        "Selected Join Key": join_key_options[source["label"]],
                    })

                st.table(pd.DataFrame(join_key_rows).reset_index(drop=True))

                merged_df = source_mapping_rows[0]["df"].copy()
                merged_key = join_key_options[source_mapping_rows[0]["label"]]
                merged_df[merged_key] = merged_df[merged_key].astype(str).str.strip()
                join_issues = []
                input_rows = sum(len(source["df"]) for source in source_mapping_rows)
                duplicate_key_count = int(merged_df[merged_key].duplicated().sum())

                for source in source_mapping_rows[1:]:
                    right_df = source["df"].copy()
                    right_key = join_key_options[source["label"]]
                    right_df[right_key] = right_df[right_key].astype(str).str.strip()

                    if merged_df[merged_key].duplicated().any():
                        join_issues.append(f"{source_mapping_rows[0]['label']}: duplicate join key values found.")
                    if right_df[right_key].duplicated().any():
                        join_issues.append(f"{source['label']}: duplicate join key values found.")
                    duplicate_key_count += int(right_df[right_key].duplicated().sum())

                    before_rows = len(merged_df)
                    merged_df = merged_df.merge(
                        right_df,
                        left_on=merged_key,
                        right_on=right_key,
                        how="inner",
                        suffixes=("", "_joined")
                    )
                    if right_key != merged_key and right_key in merged_df.columns:
                        merged_df = merged_df.drop(columns=[right_key])
                    if len(merged_df) < before_rows:
                        join_issues.append(
                            f"{source['label']}: only {len(merged_df):,} rows matched after join."
                        )

                for col in list(merged_df.columns):
                    joined_col = f"{col}_joined"
                    if joined_col in merged_df.columns:
                        merged_df[col] = merged_df[col].combine_first(merged_df[joined_col])
                        merged_df = merged_df.drop(columns=[joined_col])

                validation_warnings.extend(join_issues)
                upload_df = merged_df
                combine_summary = {
                    "Mode": "Merge related files",
                    "Input Rows": input_rows,
                    "Output Rows": len(upload_df),
                    "Duplicate Keys": duplicate_key_count,
                    "Join Warnings": len(join_issues),
                }

            combined_errors, combined_warnings, standardized_preview = preprocessing.validate_uploaded_dataset(
                upload_df,
                None
            )
            validation_errors.extend(combined_errors)
            validation_warnings.extend(combined_warnings)

            st.markdown("### Validation Summary")
            ready_status = "Ready to run" if not validation_errors else "Needs review"
            summary_cols = st.columns(4)
            summary_cols[0].metric("Status", ready_status)
            summary_cols[1].metric("Final Rows", f"{len(standardized_preview):,}")
            summary_cols[2].metric("Blocking Issues", len(validation_errors))
            summary_cols[3].metric("Warnings", len(validation_warnings))

            if combine_summary:
                combine_cols = st.columns(4)
                combine_cols[0].metric("Combine Method", combine_summary["Mode"])
                combine_cols[1].metric("Input Rows", f"{combine_summary['Input Rows']:,}")
                combine_cols[2].metric(
                    "Matched / Output Rows" if upload_combine_mode == "Join files by shared key" else "Output Rows",
                    f"{combine_summary['Output Rows']:,}"
                )
                combine_cols[3].metric(
                    "Duplicate Keys" if upload_combine_mode == "Join files by shared key" else "Sources Combined",
                    combine_summary["Duplicate Keys"] if upload_combine_mode == "Join files by shared key" else len(standardized_sources)
                )

            if validation_errors:
                st.error("Dataset cannot be processed yet:")
                for err in validation_errors:
                    st.write(f"- {err}")

            if validation_warnings:
                st.warning("Dataset can run, but please review these warnings:")
                for warning in validation_warnings[:8]:
                    st.write(f"- {warning}")
                if len(validation_warnings) > 8:
                    st.write(f"- {len(validation_warnings) - 8} more warning(s)")

            with st.expander("Preview standardized data", expanded=False):
                st.table(standardized_preview.head(10).reset_index(drop=True))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as combined_tmp:
                standardized_preview.to_csv(combined_tmp.name, index=False)
                tmp_path = combined_tmp.name

        except Exception as e:
            validation_errors = [str(e)]
            st.error(f"Upload validation failed: {e}")

        col1, col2 = st.columns(2)

# =================================================
# RUN FULL PIPELINE (UPLOAD)
# =================================================
        with col1:

            if st.button(
                "🚀 Run Full Pipeline",
                type="primary",
                use_container_width=True,
                disabled=bool(validation_errors)
            ):

                st.session_state.stop_pipeline_requested = False
                st.session_state.pipeline_running = True
                process_start = time.time()
                progress = st.progress(0)
                step = st.empty()
                status_panel = st.empty()
                detail_log = st.container()
                st.button(
                    "Stop after current step",
                    key="stop_uploaded_pipeline",
                    help="Streamlit stops safely between pipeline steps. The current model step must finish first.",
                    on_click=request_pipeline_stop,
                    use_container_width=True
                )
                pipeline_steps = [
                    ("Preprocess data", "Clean columns, validate dates and claim amounts, remove duplicates, and prepare time fields.", 12),
                    ("Engineer features", "Create claim ratios, customer signals, policy indicators, and dashboard helper tables.", 25),
                    ("Aggregate quarters", "Build quarterly totals, lag features, rolling averages, seasonality, and trend fields.", 38),
                    ("Split train/test", "Create chronological train and test sets for each configured split.", 50),
                    ("Train models", "Train Linear Regression, Random Forest, XGBoost, and Ensemble models.", 63),
                    ("Tune models", "Run time-series-aware hyperparameter tuning for supported models.", 76),
                    ("Evaluate models", "Calculate test-only metrics and generate model comparison outputs.", 88),
                    ("Generate predictions", "Create prediction files and risk levels for dashboard pages.", 100),
                ]
                completed_steps = []

                def run_stage(idx, fn):
                    if st.session_state.stop_pipeline_requested:
                        raise RuntimeError("Pipeline stopped by user before the next step started.")
                    name, detail, pct = pipeline_steps[idx]
                    step.info(f"Step {idx + 1}/8: {name} - {detail}")
                    render_process_status(status_panel, pipeline_steps, completed_steps, process_start, current_idx=idx)
                    result = fn()
                    completed_steps.append(idx)
                    progress.progress(pct)
                    render_process_status(status_panel, pipeline_steps, completed_steps, process_start)
                    detail_log.success(f"Completed: {name}")
                    return result

                render_process_status(status_panel, pipeline_steps, completed_steps, process_start)

                try:
                    with st.spinner("Preparing workspace and clearing old outputs..."):
                        st.cache_data.clear()

                        # Delete old outputs
                        if os.path.exists("outputs"):
                            shutil.rmtree("outputs")
                        if os.path.exists("models"):
                            shutil.rmtree("models")
                        if os.path.exists("models_tuned"):
                            shutil.rmtree("models_tuned")
                        os.makedirs("outputs/plots", exist_ok=True)
                        os.makedirs("models", exist_ok=True)

                    # STEP 1
                    step.info("⏳ Step 1/8: Preprocessing data...")
                    run_stage(0, lambda: preprocessing.preprocessing_pipeline(tmp_path))

                    # STEP 2
                    step.info("⏳ Step 2/8: Engineering features...")
                    run_stage(1, feature_engineering.feature_engineering_pipeline)

                    # STEP 3
                    step.info("⏳ Step 3/8: Aggregating quarterly data...")
                    run_stage(2, aggregation.aggregation_pipeline)

                    # STEP 4 — RUN ALL THREE SPLITS
                    step.info("⏳ Step 4/8: Splitting train/test data...")
                    run_stage(3, split_data.run_all_splits)

                    # STEP 5
                    step.info("⏳ Step 5/8: Training models...")
                    run_stage(4, train_models.train_models_pipeline)

                    # STEP 6
                    step.info("⏳ Step 6/8: Hyperparameter tuning...")
                    run_stage(5, tuning.tuning_pipeline)

                    # STEP 7
                    step.info("⏳ Step 7/8: Evaluating models...")
                    run_stage(6, lambda: evaluate_models.evaluate_pipeline(generate_plots=generate_png_plots))

                    # STEP 8
                    step.info("⏳ Step 8/8: Running predictions...")
                    run_stage(7, predict.predict_pipeline)

                    def remove_temp_file(path):
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                            except PermissionError:
                                # Windows can briefly keep Excel temp files locked.
                                # The OS temp folder can safely clean these up later.
                                pass

                    remove_temp_file(tmp_path)
                    for path in temp_paths:
                        remove_temp_file(path)

                    st.session_state.pipeline_ran = True
                    st.session_state.data_ready = True

                    st.cache_data.clear()
                    st.session_state.pipeline_running = False
                    render_process_status(status_panel, pipeline_steps, completed_steps, process_start)
                    show_done_notification("Pipeline completed successfully. Dashboard outputs are ready.")
                    step.success("🎉 Pipeline Completed Successfully! Refreshing...")
                    st.session_state.pending_nav_page = "📈 Quarterly Trends"
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.session_state.pipeline_running = False
                    st.error(f"❌ Pipeline Failed: {str(e)}")
                    st.exception(e)

        # =================================================
        # PREVIEW DATA
        # =================================================
        with col2:

            if st.button(
                "🔍 Preview Data Only",
                use_container_width=True
            ):

                try:
                    df = upload_df.copy() if upload_df is not None else pd.DataFrame()

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric("Rows", f"{df.shape[0]:,}")

                    with c2:
                        st.metric("Columns", df.shape[1])

                    with c3:
                        mem = df.memory_usage(
                            deep=True
                        ).sum() / 1024**2
                        st.metric("Memory", f"{mem:.2f} MB")

                    st.dataframe(
                        df.head(10),
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Preview failed: {e}")


# =========================================================
# QUARTERLY TRENDS
# =========================================================
elif page == "📈 Quarterly Trends":
    st.title("📈 Quarterly Trend Analysis")

    required_cols = ['Quarter_Label', 'Quarterly_Total_Claims', 'Quarterly_Claim_Count', 
                     'Quarterly_Avg_Claim', 'Quarterly_Max_Claim', 'Quarter']
    
    if not show_data_unavailable_message("Quarterly Trend Analysis", "outputs/quarterly_claims.csv", required_cols):
        st.stop()
    
    df_full = pd.read_csv("outputs/quarterly_claims.csv")

    # Tabs: Single Year View | Compare Years
    tab1, tab2 = st.tabs(["📊 Single Year View", "🔄 Compare Years"])
    
    with tab1:
        df, selected_year = year_filter_widget(df_full, 'Quarter_Label', key_prefix="trends")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Quarters", len(df))
        with col2:
            st.metric("Avg Quarterly Claims", f"RM{df['Quarterly_Total_Claims'].mean():,.0f}")
        with col3:
            st.metric("Peak Quarter", f"RM{df['Quarterly_Total_Claims'].max():,.0f}")
        with col4:
            if len(df) > 1:
                growth = ((df['Quarterly_Total_Claims'].iloc[-1] / df['Quarterly_Total_Claims'].iloc[0]) - 1) * 100
            else:
                growth = 0
            st.metric("Growth Rate", f"{growth:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        fig = create_plotly_line_chart(
            df, 'Quarter_Label', 'Quarterly_Total_Claims',
            f'Quarterly Claim Trends ({selected_year})',
            'Quarter', 'Total Claims (RM)'
        )
        st.plotly_chart(fig, use_container_width=True)

        seasonal = df.groupby('Quarter')['Quarterly_Total_Claims'].mean()
        quarter_labels = [f'Q{int(q)}' for q in seasonal.index]
        fig2 = create_plotly_bar_chart(
            quarter_labels, seasonal.values,
            f'Average Claims by Quarter - Seasonality ({selected_year})'
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("## 📊 Quarterly Summary Table")
        display_df = df[['Quarter_Label', 'Quarterly_Total_Claims', 'Quarterly_Claim_Count',
                         'Quarterly_Avg_Claim', 'Quarterly_Max_Claim']].copy()
        display_df.columns = ['Quarter', 'Total Claims (RM)', 'Claim Count', 'Avg Claim (RM)', 'Max Claim (RM)']
        
        st.dataframe(
            display_df.style.format({
                'Total Claims (RM)': 'RM{:,.0f}',
                'Avg Claim (RM)': 'RM{:,.0f}',
                'Max Claim (RM)': 'RM{:,.0f}'
            }).background_gradient(subset=['Total Claims (RM)'], cmap='viridis'),
            use_container_width=True
        )

    with tab2:
        st.markdown("### 🔄 Year-over-Year Comparison")
        df1, df2, year1, year2 = compare_years_widget(df_full, 'Quarter_Label', key_prefix="trends_cmp")
        
        if df1 is not None and df2 is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 📅 Year {year1}")
                st.metric("Total Claims", f"RM{df1['Quarterly_Total_Claims'].sum():,.0f}")
                st.metric("Avg per Quarter", f"RM{df1['Quarterly_Total_Claims'].mean():,.0f}")
            with col2:
                st.markdown(f"#### 📅 Year {year2}")
                st.metric("Total Claims", f"RM{df2['Quarterly_Total_Claims'].sum():,.0f}")
                st.metric("Avg per Quarter", f"RM{df2['Quarterly_Total_Claims'].mean():,.0f}")
            
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name=f'Year {year1}',
                x=[f'Q{i+1}' for i in range(len(df1))],
                y=df1['Quarterly_Total_Claims'].values,
                marker=dict(color='#667eea', line=dict(color='white', width=2)),
                text=[f'RM{x:,.0f}' for x in df1['Quarterly_Total_Claims'].values],
                textposition='outside'
            ))
            fig_cmp.add_trace(go.Bar(
                name=f'Year {year2}',
                x=[f'Q{i+1}' for i in range(len(df2))],
                y=df2['Quarterly_Total_Claims'].values,
                marker=dict(color='#e94560', line=dict(color='white', width=2)),
                text=[f'RM{x:,.0f}' for x in df2['Quarterly_Total_Claims'].values],
                textposition='outside'
            ))
            fig_cmp.update_layout(
                barmode='group',
                title=dict(text=f'Claims Comparison: {year1} vs {year2}', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
                xaxis_title='Quarter', yaxis_title='Total Claims (RM)',
                template='plotly_dark',
                plot_bgcolor='rgba(22, 33, 62, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'),
                height=500,
            )
            apply_chart_layout(fig_cmp, bottom_margin=135, legend_y=-0.30)
            st.plotly_chart(fig_cmp, use_container_width=True)
            
            # Percentage Change
            total1 = df1['Quarterly_Total_Claims'].sum()
            total2 = df2['Quarterly_Total_Claims'].sum()
            pct_change = ((total2 - total1) / total1 * 100) if total1 != 0 else 0
            change_color = "#2ecc71" if pct_change >= 0 else "#e74c3c"
            
            st.markdown(f"""
            <div style='background: rgba(22, 33, 62, 0.95); padding: 2rem; border-radius: 15px; border: 2px solid {change_color}; text-align: center;'>
                <p style='margin: 0; color: #90cdf4; font-weight: 600;'>Total Claims Change: {year1} → {year2}</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 3rem; color: {change_color}; font-weight: 700; font-family: "Space Mono", monospace;'>{pct_change:+.1f}%</p>
                <p style='margin: 0.5rem 0 0 0; color: #90cdf4;'>RM{total1:,.0f} → RM{total2:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# CUSTOMER RISK
# =========================================================
elif page == "👤 Customer Risk":
    st.title("👤 Customer Risk Analysis")

    required_cols = ['Customer ID', 'Risk_Score', 'Risk_Category', 'Total_Claim', 
                     'Incident_Count', 'Top_Incident_Type']
    
    if not show_data_unavailable_message("Customer Risk Analysis", "outputs/customer_risk_scores.csv", required_cols):
        st.stop()
    
    df = pd.read_csv("outputs/customer_risk_scores.csv")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        high_risk_count = len(df[df['Risk_Category'] == 'HIGH'])
        st.metric("High Risk Customers", high_risk_count, delta=f"{(high_risk_count/len(df)*100):.1f}%")
    with col3:
        st.metric("Avg Risk Score", f"{df['Risk_Score'].mean():.1f}")
    with col4:
        st.metric("Max Total Claim", f"RM{df['Total_Claim'].max():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = df['Risk_Category'].value_counts()
        fig = create_plotly_pie_chart(risk_counts.index.tolist(), risk_counts.values.tolist(), 'Customer Risk Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df['Risk_Score'], nbinsx=30, marker=dict(color='#667eea', line=dict(color='white', width=1))))
        fig2.add_vline(x=df['Risk_Score'].mean(), line_dash="dash", line_color="#e94560",
                      annotation_text=f"Mean: {df['Risk_Score'].mean():.1f}", annotation_position="top")
        fig2.update_layout(
            title=dict(text='Risk Score Distribution', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
            xaxis_title='Risk Score', yaxis_title='Count',
            template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='Outfit, sans-serif', color='#90cdf4'), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## 🎯 Top 10 Highest Risk Customers")
    top_risk = df.nlargest(10, 'Risk_Score')[['Customer ID', 'Risk_Score', 'Risk_Category', 'Total_Claim', 'Incident_Count', 'Top_Incident_Type']]
    st.dataframe(top_risk.style.background_gradient(subset=['Risk_Score'], cmap='Reds'), use_container_width=True)

    st.markdown("## 🔍 All Customers (Filterable)")
    risk_filter = st.multiselect("Filter by Risk Level", ['LOW', 'MEDIUM', 'HIGH'], default=['HIGH', 'MEDIUM'])
    filtered = df[df['Risk_Category'].isin(risk_filter)]
    st.dataframe(filtered.sort_values('Risk_Score', ascending=False), use_container_width=True)


# =========================================================
# INCIDENT PATTERNS
# =========================================================
elif page == "🚨 Incident Patterns":
    st.title("🚨 Incident Pattern Analysis")

    required_cols = ['Incident Type', 'Count', 'Total_Cost', 'Quarter_Label']
    
    if not show_data_unavailable_message("Incident Pattern Analysis", "outputs/quarterly_incident_patterns.csv", required_cols):
        st.stop()
    
    df_full = pd.read_csv("outputs/quarterly_incident_patterns.csv")

    tab1, tab2 = st.tabs(["📊 Single Year View", "🔄 Compare Years"])
    
    with tab1:
        df, selected_year = year_filter_widget(df_full, 'Quarter_Label', key_prefix="incident")
        
        st.markdown("## 📊 Most Common Incident Types")
        overall = df.groupby('Incident Type').agg({'Count': 'sum', 'Total_Cost': 'sum'}).sort_values('Count', ascending=False)
        top_incidents = overall.head(10).sort_values('Count', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_incidents.index, x=top_incidents['Count'], orientation='h',
            marker=dict(color=top_incidents['Count'], colorscale='Turbo', line=dict(color='white', width=2)),
            text=top_incidents['Count'], textposition='outside'
        ))
        fig.update_layout(
            title=dict(text=f'Top 10 Incident Types by Frequency ({selected_year})', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
            xaxis_title='Total Count', yaxis_title='',
            template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 💸 Highest Cost Incident Types")
        costly = overall.sort_values('Total_Cost', ascending=False).head(10).sort_values('Total_Cost', ascending=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=costly.index, x=costly['Total_Cost'], orientation='h',
            marker=dict(color=costly['Total_Cost'], colorscale='Reds', line=dict(color='white', width=2)),
            text=[f'RM{x:,.0f}' for x in costly['Total_Cost']], textposition='outside'
        ))
        fig2.update_layout(
            title=dict(text=f'Top 10 Incident Types by Total Cost ({selected_year})', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
            xaxis_title='Total Cost (RM)', yaxis_title='',
            template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("## 📅 Incident Distribution by Quarter")
        pivot = df.pivot_table(index='Quarter_Label', columns='Incident Type', values='Count', aggfunc='sum', fill_value=0)
        top_types = overall.nlargest(8, 'Count').index
        pivot_subset = pivot[[col for col in top_types if col in pivot.columns]]
        
        fig3 = go.Figure()
        for column in pivot_subset.columns:
            fig3.add_trace(go.Bar(name=column, x=pivot_subset.index, y=pivot_subset[column], text=pivot_subset[column], textposition='inside'))
        fig3.update_layout(
            barmode='stack',
            title=dict(text=f'Top 8 Incident Types by Quarter ({selected_year})', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
            template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=600,
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02),
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("## 📋 Detailed Incident Data")
        st.dataframe(df.sort_values('Total_Cost', ascending=False).head(50), use_container_width=True)

    with tab2:
        st.markdown("### 🔄 Year-over-Year Incident Comparison")
        df1, df2, year1, year2 = compare_years_widget(df_full, 'Quarter_Label', key_prefix="incident_cmp")
        
        if df1 is not None and df2 is not None:
            overall1 = df1.groupby('Incident Type').agg({'Count': 'sum', 'Total_Cost': 'sum'}).sort_values('Count', ascending=False)
            overall2 = df2.groupby('Incident Type').agg({'Count': 'sum', 'Total_Cost': 'sum'}).sort_values('Count', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 📅 Year {year1}")
                st.metric("Total Incidents", f"{overall1['Count'].sum():,}")
                st.metric("Total Cost", f"RM{overall1['Total_Cost'].sum():,.0f}")
            with col2:
                st.markdown(f"#### 📅 Year {year2}")
                st.metric("Total Incidents", f"{overall2['Count'].sum():,}")
                st.metric("Total Cost", f"RM{overall2['Total_Cost'].sum():,.0f}")
            
            top5 = overall1.head(5).index.tolist()
            vals1 = [overall1.loc[t, 'Total_Cost'] if t in overall1.index else 0 for t in top5]
            vals2 = [overall2.loc[t, 'Total_Cost'] if t in overall2.index else 0 for t in top5]
            
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name=f'Year {year1}', x=top5, y=vals1, marker=dict(color='#667eea'), text=[f'RM{x:,.0f}' for x in vals1], textposition='outside'))
            fig_cmp.add_trace(go.Bar(name=f'Year {year2}', x=top5, y=vals2, marker=dict(color='#e94560'), text=[f'RM{x:,.0f}' for x in vals2], textposition='outside'))
            fig_cmp.update_layout(
                barmode='group',
                title=dict(text=f'Top 5 Incident Costs: {year1} vs {year2}', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
                template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500,
            )
            apply_chart_layout(fig_cmp, bottom_margin=135, legend_y=-0.30)
            st.plotly_chart(fig_cmp, use_container_width=True)


# =========================================================
# CLAIM BREAKDOWN
# =========================================================
elif page == "💰 Claim Breakdown":
    st.title("💰 Claim Amount Breakdown")

    required_cols = ['Quarter_Label', 'Injury Claim', 'Property Claim', 'Vehicle Claim']
    
    if not show_data_unavailable_message("Claim Amount Breakdown", "outputs/quarterly_claim_breakdown.csv", required_cols):
        st.stop()
    
    df_full = pd.read_csv("outputs/quarterly_claim_breakdown.csv")

    tab1, tab2 = st.tabs(["📊 Single Year View", "🔄 Compare Years"])
    
    with tab1:
        df, selected_year = year_filter_widget(df_full, 'Quarter_Label', key_prefix="claim")
        
        st.markdown("## 🥧 Overall Claim Composition")
        total_injury = df['Injury Claim'].sum() if 'Injury Claim' in df.columns else 0
        total_property = df['Property Claim'].sum() if 'Property Claim' in df.columns else 0
        total_vehicle = df['Vehicle Claim'].sum() if 'Vehicle Claim' in df.columns else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Injury Claims", f"RM{total_injury:,.0f}")
        with col2:
            st.metric("Property Claims", f"RM{total_property:,.0f}")
        with col3:
            st.metric("Vehicle Claims", f"RM{total_vehicle:,.0f}")

        if total_injury + total_property + total_vehicle > 0:
            fig = create_plotly_pie_chart(
                ['Injury Claims', 'Property Claims', 'Vehicle Claims'],
                [total_injury, total_property, total_vehicle],
                f'Total Claims Distribution by Type ({selected_year})'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 📊 Quarterly Claim Breakdown (Stacked)")
        claim_cols = [c for c in ['Injury Claim', 'Property Claim', 'Vehicle Claim'] if c in df.columns]
        
        if claim_cols:
            fig2 = go.Figure()
            colors_map = {'Injury Claim': '#e74c3c', 'Property Claim': '#3498db', 'Vehicle Claim': '#2ecc71'}
            for col in claim_cols:
                fig2.add_trace(go.Bar(
                    name=col.replace(' Claim', ''), x=df['Quarter_Label'], y=df[col],
                    marker=dict(color=colors_map.get(col, '#667eea')),
                    text=[f'RM{x:,.0f}' for x in df[col]], textposition='inside'
                ))
            fig2.update_layout(
                barmode='stack',
                title=dict(text=f'Claim Breakdown by Quarter ({selected_year})', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
                xaxis_title='Quarter', yaxis_title='Amount (RM)',
                template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500,
            )
            apply_chart_layout(fig2, bottom_margin=135, legend_y=-0.30)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("## 📈 Percentage Contribution Over Time")
        pct_cols = [c for c in ['Injury Pct', 'Property Pct', 'Vehicle Pct'] if c in df.columns]
        if pct_cols:
            fig3 = go.Figure()
            for col in pct_cols:
                fig3.add_trace(go.Scatter(x=df['Quarter_Label'], y=df[col], mode='lines+markers', name=col.replace(' Pct', ''), line=dict(width=3), marker=dict(size=8)))
            fig3.update_layout(
                title=dict(text=f'Claim Type Contribution % ({selected_year})', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
                xaxis_title='Quarter', yaxis_title='Percentage (%)',
                template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500, hovermode='x unified',
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("## 📋 Detailed Breakdown Table")
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.markdown("### 🔄 Year-over-Year Claim Comparison")
        df1, df2, year1, year2 = compare_years_widget(df_full, 'Quarter_Label', key_prefix="claim_cmp")
        
        if df1 is not None and df2 is not None:
            claim_cols = [c for c in ['Injury Claim', 'Property Claim', 'Vehicle Claim'] if c in df_full.columns]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 📅 Year {year1}")
                for col in claim_cols:
                    st.metric(col, f"RM{df1[col].sum():,.0f}")
            with col2:
                st.markdown(f"#### 📅 Year {year2}")
                for col in claim_cols:
                    st.metric(col, f"RM{df2[col].sum():,.0f}")
            
            categories = [c.replace(' Claim', '') for c in claim_cols]
            vals1 = [df1[c].sum() for c in claim_cols]
            vals2 = [df2[c].sum() for c in claim_cols]
            
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name=f'Year {year1}', x=categories, y=vals1, marker=dict(color='#667eea'), text=[f'RM{x:,.0f}' for x in vals1], textposition='outside'))
            fig_cmp.add_trace(go.Bar(name=f'Year {year2}', x=categories, y=vals2, marker=dict(color='#e94560'), text=[f'RM{x:,.0f}' for x in vals2], textposition='outside'))
            fig_cmp.update_layout(
                barmode='group',
                title=dict(text=f'Claim Comparison: {year1} vs {year2}', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
                template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500,
            )
            apply_chart_layout(fig_cmp, bottom_margin=135, legend_y=-0.30)
            st.plotly_chart(fig_cmp, use_container_width=True)
            
            total1 = sum(vals1)
            total2 = sum(vals2)
            pct_change = ((total2 - total1) / total1 * 100) if total1 != 0 else 0
            change_color = "#2ecc71" if pct_change >= 0 else "#e74c3c"
            st.markdown(f"""
            <div style='background: rgba(22, 33, 62, 0.95); padding: 2rem; border-radius: 15px; border: 2px solid {change_color}; text-align: center;'>
                <p style='margin: 0; color: #90cdf4;'>Total Claims Change: {year1} → {year2}</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 3rem; color: {change_color}; font-weight: 700; font-family: "Space Mono", monospace;'>{pct_change:+.1f}%</p>
                <p style='margin: 0.5rem 0 0 0; color: #90cdf4;'>RM{total1:,.0f} → RM{total2:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# PREDICTIONS & RISK
# =========================================================
elif page == "🔮 Predictions & Risk":
    st.title("🔮 Quarterly Predictions & Risk Levels")

    split_choice = st.selectbox("📊 Select Split", ['80_20', '70_30', '90_10'])
    st.info(
        "This page shows historical train/test predictions. Use the Forecast page for future next-quarter or next-year projections."
    )

    pred_file = f"outputs/predictions_{split_choice}.csv"

    if not os.path.exists(pred_file):
        show_data_unavailable_card("Quarterly Predictions & Risk Levels", pred_file)
        st.stop()

    df_full = pd.read_csv(pred_file)
    if len(df_full) == 0:
        show_data_unavailable_card(
            "Quarterly Predictions & Risk Levels",
            pred_file,
            message=f"The prediction file is empty: {pred_file}",
            title="No Data Available"
        )
        st.stop()

    tab1, tab2 = st.tabs(["📊 Single Year View", "🔄 Compare Years"])
    
    with tab1:
        if 'Quarter_Label' in df_full.columns:
            df, selected_year = year_filter_widget(df_full, 'Quarter_Label', key_prefix="pred")
        else:
            df = df_full
            selected_year = "All Years"

        # Naive Baseline is an evaluation benchmark, not a selectable ML prediction model.
        model_cols = [
            c for c in df.columns
            if '_Prediction' in c and not c.startswith('Naive_Baseline_')
        ]
        model_names = [c.replace('_Prediction', '') for c in model_cols]
        
        # Add "auto" option which picks the best model by RMSE
        model_options = ['auto'] + model_names
        selected_option = st.selectbox(
            "🤖 Select Model", model_options,
            index=0,
            key="pred_model"
        )
        
        # If auto, pick best model (lowest RMSE from evaluation)
        if selected_option == 'auto':
            eval_path = "outputs/evaluation_master.csv"
            if os.path.exists(eval_path):
                eval_df_temp = pd.read_csv(eval_path)
                split_eval = eval_df_temp[eval_df_temp['split'] == split_choice]
                split_eval = split_eval[~split_eval['model'].isin(['Naive_Baseline'])]
                if not split_eval.empty:
                    best_model = split_eval.loc[split_eval['RMSE'].idxmin(), 'model']
                    selected_model = best_model if best_model in model_names else model_names[0]
                else:
                    selected_model = model_names[0]
            else:
                selected_model = 'Ensemble' if 'Ensemble' in model_names else model_names[0]
            st.info(f"🤖 Auto-selected: **{selected_model}** (best RMSE)")
        else:
            selected_model = selected_option
        
        pred_col = f"{selected_model}_Prediction"
        risk_col = f"{selected_model}_Risk"

# --- FIX: Context-Aware Metric Cards ---
        col1, col2, col3, col4 = st.columns(4)
        
        if len(df) > 0:
            # Dynamically identify what quarter we are actually looking at
            target_quarter = df['Quarter_Label'].iloc[-1] if 'Quarter_Label' in df.columns else "Selected Period"
            
            with col1:
                next_pred = df[pred_col].iloc[-1]
                # Dynamic title keeps the dashboard contextually clear
                st.metric(f"Prediction ({target_quarter})", f"RM{next_pred:,.0f}")
                
            with col2:
                last_risk = df[risk_col].iloc[-1] if risk_col in df.columns else "N/A"
                risk_colors = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}
                risk_color = risk_colors.get(last_risk, "#90cdf4")
                st.markdown(f"""
                <div style='background: rgba(22, 33, 62, 0.95); padding: 1rem; border-radius: 10px; border: 2px solid {risk_color}; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>RISK LEVEL ({target_quarter})</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 2rem; color: {risk_color}; font-weight: 700;'>{last_risk}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                last_actual = df['Actual'].iloc[-1] if 'Actual' in df.columns else 0
                st.metric(f"Actual ({target_quarter})", f"RM{last_actual:,.0f}")
                
            with col4:
                if 'Actual' in df.columns and pred_col in df.columns:
                    last_err = abs(df['Actual'].iloc[-1] - df[pred_col].iloc[-1])
                    err_pct = (last_err / df['Actual'].iloc[-1] * 100) if df['Actual'].iloc[-1] != 0 else 0
                    st.metric("Prediction Variance", f"RM{last_err:,.0f}", delta=f"{err_pct:.1f}%", delta_color="inverse")
        else:
            st.warning("⚠️ No data available for the selected filter.")
        
        # Actual vs Predicted — Area Between Chart
        if 'Actual' in df.columns:
            fig = go.Figure()
            
            x_vals = df['Quarter_Label'] if 'Quarter_Label' in df.columns else list(range(len(df)))
            
            # Upper bound (whichever is higher)
            upper = np.maximum(df['Actual'].values, df[pred_col].values)
            lower = np.minimum(df['Actual'].values, df[pred_col].values)
            
            # Error band — fill between actual and predicted
            fig.add_trace(go.Scatter(
                x=x_vals, y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=x_vals, y=lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.3)',
                name='Prediction Gap',
                hoverinfo='skip'
            ))
            
            # Actual line — bold and prominent
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df['Actual'],
                mode='lines+markers',
                name='Actual Claims',
                line=dict(color='#e94560', width=4),
                marker=dict(size=10, color='#e94560', line=dict(width=2, color='white')),
                hovertemplate='Actual: RM%{y:,.0f}<extra></extra>'
            ))
            
            # Predicted line — dashed
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df[pred_col],
                mode='lines+markers',
                name='Predicted Claims',
                line=dict(color='#667eea', width=4, dash='dash'),
                marker=dict(size=10, color='#667eea', symbol='diamond', line=dict(width=2, color='white')),
                hovertemplate='Predicted: RM%{y:,.0f}<extra></extra>'
            ))
            
            # Mark test quarters with vertical shading
            if 'Data_Split' in df.columns:
                test_quarters = df[df['Data_Split'] == 'Test']['Quarter_Label'].tolist()
                if test_quarters:
                    # Add annotation for test region
                    fig.add_vrect(
                        x0=test_quarters[0], x1=test_quarters[-1],
                        fillcolor="rgba(102, 126, 234, 0.1)",
                        layer="below", line_width=2,
                        line_color="rgba(102, 126, 234, 0.5)",
                        line_dash="dash",
                        annotation_text="Test Period (Out-of-Sample)",
                        annotation_position="top left",
                        annotation_font=dict(color="#667eea", size=12)
                    )
            
            fig.update_layout(
                title=dict(
                    text=f'{selected_model}: Actual vs Predicted Claims ({selected_year})',
                    font=dict(size=22, family='Outfit, sans-serif', color='#48cae4')
                ),
                xaxis_title='Quarter',
                yaxis_title='Claims (RM)',
                template='plotly_dark',
                plot_bgcolor='rgba(22, 33, 62, 0.8)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'),
                height=550,
                hovermode='x unified',
                xaxis=dict(showgrid=True, gridcolor='rgba(72, 202, 228, 0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(72, 202, 228, 0.1)', tickformat=',')
            )
            apply_chart_layout(fig, bottom_margin=140, legend_y=-0.30)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add error summary below chart
            if 'Data_Split' in df.columns:
                col_a, col_b, col_c = st.columns(3)
                
                train_mask = df['Data_Split'] == 'Train'
                test_mask = df['Data_Split'] == 'Test'
                
                with col_a:
                    if train_mask.any():
                        train_mape = (abs(df.loc[train_mask, 'Actual'] - df.loc[train_mask, pred_col]) / df.loc[train_mask, 'Actual'] * 100).mean()
                        st.metric("Train Accuracy", f"{100 - train_mape:.1f}%", help="In-sample fit")
                with col_b:
                    if test_mask.any():
                        test_mape = (abs(df.loc[test_mask, 'Actual'] - df.loc[test_mask, pred_col]) / df.loc[test_mask, 'Actual'] * 100).mean()
                        st.metric("Test Accuracy", f"{100 - test_mape:.1f}%", help="True out-of-sample prediction accuracy")
                with col_c:
                    overall_mape = (abs(df['Actual'] - df[pred_col]) / df['Actual'] * 100).mean()
                    st.metric("Overall Accuracy", f"{100 - overall_mape:.1f}%")

    with tab2:
        st.markdown("### 🔄 Year-over-Year Prediction Comparison")
        if 'Quarter_Label' in df_full.columns:
            df1, df2, year1, year2 = compare_years_widget(df_full, 'Quarter_Label', key_prefix="pred_cmp")
            if df1 is not None and df2 is not None:
                model_cols = [
                    c for c in df_full.columns
                    if '_Prediction' in c and not c.startswith('Naive_Baseline_')
                ]
                model_names_cmp = [c.replace('_Prediction', '') for c in model_cols]
                model_options_cmp = ['auto'] + model_names_cmp
                sel_option_cmp = st.selectbox("🤖 Model", model_options_cmp, index=0, key="pred_cmp_model")
                if sel_option_cmp == 'auto':
                    sel_model_cmp = selected_model  # Use same auto-selected model
                else:
                    sel_model_cmp = sel_option_cmp
                pred_col_cmp = f"{sel_model_cmp}_Prediction"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### 📅 Year {year1}")
                    if 'Actual' in df1.columns:
                        st.metric("Total Actual", f"RM{df1['Actual'].sum():,.0f}")
                    st.metric("Total Predicted", f"RM{df1[pred_col_cmp].sum():,.0f}")
                with col2:
                    st.markdown(f"#### 📅 Year {year2}")
                    if 'Actual' in df2.columns:
                        st.metric("Total Actual", f"RM{df2['Actual'].sum():,.0f}")
                    st.metric("Total Predicted", f"RM{df2[pred_col_cmp].sum():,.0f}")
                
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(x=[f'Q{i+1}' for i in range(len(df1))], y=df1[pred_col_cmp].values, mode='lines+markers', name=f'{year1} Predicted', line=dict(color='#667eea', width=3)))
                fig_cmp.add_trace(go.Scatter(x=[f'Q{i+1}' for i in range(len(df2))], y=df2[pred_col_cmp].values, mode='lines+markers', name=f'{year2} Predicted', line=dict(color='#e94560', width=3)))
                if 'Actual' in df1.columns:
                    fig_cmp.add_trace(go.Scatter(x=[f'Q{i+1}' for i in range(len(df1))], y=df1['Actual'].values, mode='lines+markers', name=f'{year1} Actual', line=dict(color='#667eea', width=2, dash='dot')))
                if 'Actual' in df2.columns:
                    fig_cmp.add_trace(go.Scatter(x=[f'Q{i+1}' for i in range(len(df2))], y=df2['Actual'].values, mode='lines+markers', name=f'{year2} Actual', line=dict(color='#e94560', width=2, dash='dot')))
                fig_cmp.update_layout(
                    title=dict(text=f'Predictions: {year1} vs {year2}', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
                    xaxis_title='Quarter', yaxis_title='Claims (RM)',
                    template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500, hovermode='x unified',
                )
                apply_chart_layout(fig_cmp, bottom_margin=140, legend_y=-0.30)
                st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("ℹ️ Quarter_Label column not found. Year comparison requires date labels.")

# =========================================================
# NEXT QUARTER / NEXT YEAR FORECAST
# =========================================================
elif page == "📡 Next Quarter Forecast":
    st.title("📡 Forecast & Projections")
    
    st.markdown("""
    <div style='background: rgba(72, 202, 228, 0.1); border-left: 4px solid #48cae4; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <p style='color: #90cdf4; margin: 0;'>
            Forecast upcoming claims — predict the <strong>next quarter</strong> or the <strong>next year</strong> (4 quarters).
            <br>All values in <strong>Malaysian Ringgit (RM)</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check required files
    feat_path = "models/80_20/feature_columns.txt"
    if not os.path.exists(feat_path):
        show_data_unavailable_card(
            "Forecast & Projections",
            feat_path,
            action="Please run the complete pipeline from 'Upload & Run Pipeline' to train the models first."
        )
        st.stop()
    
    with open(feat_path, 'r') as f:
        feature_cols = f.read().strip().split('\n')
    
    # Load most recent data
    test_path = "test_data_20.csv"
    quarterly_path = "outputs/quarterly_claims.csv"
    
    if os.path.exists(test_path):
        recent_df = pd.read_csv(test_path)
    elif os.path.exists(quarterly_path):
        recent_df = pd.read_csv(quarterly_path)
    else:
        show_data_unavailable_card(
            "Forecast & Projections",
            [test_path, quarterly_path],
            action="Please run the complete pipeline from 'Upload & Run Pipeline' to generate forecasting data."
        )
        st.stop()
    
    # 1. Safely load the last row and its quarter identifier
    last_row = recent_df.iloc[-1]
    last_quarter_label = last_row.get('Quarter_Label', 'Latest')
    
    # 2. Parse last quarter using regex safely
    import re
    match = re.search(r'(\d{4}).*Q(\d)', str(last_quarter_label))
    if match:
        last_year = int(match.group(1))
        last_q = int(match.group(2))
    else:
        last_year = 2017
        last_q = 4
    
    # Generate next quarter labels
    def get_next_quarters(ly, lq, count=4):
        quarters = []
        y, q = ly, lq
        for _ in range(count):
            if q == 4:
                y += 1
                q = 1
            else:
                q += 1
            quarters.append(f"{y}-Q{q}")
        return quarters
    
    next_quarters = get_next_quarters(last_year, last_q, 4)
    next_quarter_label = next_quarters[0]
    
    # Build baseline
    baseline = {}
    for col in feature_cols:
        if col in recent_df.columns:
            val = last_row[col]
            baseline[col] = float(val) if pd.notna(val) else 0.0
        else:
            baseline[col] = 0.0

    history_for_forecast = recent_df
    if os.path.exists(quarterly_path):
        history_for_forecast = pd.read_csv(quarterly_path)

    def safe_float_from(source, col, default=0.0):
        if isinstance(source, dict):
            val = source.get(col, default)
        else:
            val = source.get(col, default) if hasattr(source, 'get') else default
        try:
            return float(val) if pd.notna(val) else default
        except (TypeError, ValueError):
            return default

    def parse_quarter_number(q_label):
        q_match = re.search(r'Q(\d)', str(q_label))
        return int(q_match.group(1)) if q_match else 1

    def historical_quarter_mean(col, q_num, fallback):
        if col in history_for_forecast.columns and 'Quarter' in history_for_forecast.columns:
            q_vals = pd.to_numeric(
                history_for_forecast[history_for_forecast['Quarter'] == q_num][col],
                errors='coerce'
            ).dropna()
            if len(q_vals) > 0:
                return float(q_vals.mean())
        return fallback

    def build_future_feature_row(previous_state, q_label):
        """
        Build a real future feature row by rolling lag features forward and
        updating quarter seasonality. This prevents the forecast from reusing
        the last historical row as-is.
        """
        q_num = parse_quarter_number(q_label)
        future = {col: safe_float_from(previous_state, col, 0.0) for col in feature_cols}

        previous_total = safe_float_from(previous_state, 'Quarterly_Total_Claims')
        if previous_total == 0:
            previous_total = safe_float_from(previous_state, 'Quarterly_Total_Claims_Lag1')

        for base_col in ['Quarterly_Total_Claims', 'Quarterly_Claim_Count', 'Quarterly_Avg_Claim']:
            prev_current = safe_float_from(previous_state, base_col)
            if prev_current == 0:
                prev_current = safe_float_from(previous_state, f'{base_col}_Lag1')

            for lag in range(4, 1, -1):
                lag_col = f'{base_col}_Lag{lag}'
                prev_lag_col = f'{base_col}_Lag{lag - 1}'
                if lag_col in future:
                    future[lag_col] = safe_float_from(previous_state, prev_lag_col)

            lag1_col = f'{base_col}_Lag1'
            if lag1_col in future:
                future[lag1_col] = previous_total if base_col == 'Quarterly_Total_Claims' else prev_current

        rolling_values = [
            previous_total,
            safe_float_from(previous_state, 'Quarterly_Total_Claims_Lag1'),
            safe_float_from(previous_state, 'Quarterly_Total_Claims_Lag2'),
            safe_float_from(previous_state, 'Quarterly_Total_Claims_Lag3'),
        ]
        rolling_values = [v for v in rolling_values if v != 0]
        if rolling_values:
            if 'Quarterly_Total_Claims_Rolling_Mean_4Q' in future:
                future['Quarterly_Total_Claims_Rolling_Mean_4Q'] = float(np.mean(rolling_values))
            if 'Quarterly_Total_Claims_Rolling_Std_4Q' in future:
                future['Quarterly_Total_Claims_Rolling_Std_4Q'] = float(np.std(rolling_values))

        if 'Time_Trend' in future:
            future['Time_Trend'] = safe_float_from(previous_state, 'Time_Trend') + 1
        if 'Quarter_Sin' in future:
            future['Quarter_Sin'] = float(np.sin(2 * np.pi * q_num / 4))
        if 'Quarter_Cos' in future:
            future['Quarter_Cos'] = float(np.cos(2 * np.pi * q_num / 4))
        for q in [1, 2, 3, 4]:
            q_col = f'Q_{q}'
            if q_col in future:
                future[q_col] = 1.0 if q == q_num else 0.0

        # If older saved models still expect current-quarter aggregate columns,
        # estimate them from historical same-quarter averages instead of copying
        # the previous quarter. New retrained models will not use these columns.
        proxy_cols = [
            'Quarterly_Claim_Count',
            'Quarterly_Avg_Claim',
            'Quarterly_Median_Claim',
            'Quarterly_Max_Claim',
            'Quarterly_Min_Claim',
            'Quarterly_Std_Claim',
            'Unique_Customers',
            'Avg_Customer_Age',
            'Avg_Months_As_Customer',
            'Avg_Policy_Premium',
            'Avg_Policy_Deductable',
            'Severe_Incident_Count',
            'High_Deductible_Count',
            'Avg_Injury_Ratio',
            'Avg_Property_Ratio',
            'Avg_Vehicle_Ratio',
        ]
        for col in proxy_cols:
            if col in future:
                future[col] = historical_quarter_mean(col, q_num, future[col])

        return future
    
    # ========== USER CHOICE ==========
    st.markdown("---")
    
    col_choice, col_model, col_split = st.columns(3)
    
    with col_choice:
        forecast_mode = st.selectbox(
            "📅 Forecast Period",
            ["Next Quarter", "Next Year (4 Quarters)"],
            key="forecast_mode"
        )
    with col_model:
        forecast_model = st.selectbox("🤖 Model", ['auto', 'Ensemble', 'XGBoost', 'RandomForest', 'LinearRegression'], key="forecast_model")
    with col_split:
        forecast_split = st.selectbox("📊 Split", ['80_20', '70_30', '90_10'], key="forecast_split")
    
    st.markdown("---")
    
    risk_colors_map = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}
    
    try:
        # =============================================================
        # NEXT QUARTER
        # =============================================================
        if forecast_mode == "Next Quarter":
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); 
                        border: 2px solid #667eea; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;'>
                <p style='margin: 0; font-size: 1rem; color: #90cdf4;'>FORECASTING FOR</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 3rem; color: #667eea; font-weight: 700; font-family: "Space Mono", monospace;'>{next_quarter_label}</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #48cae4;'>Based on last known quarter: {last_quarter_label}</p>
            </div>
            """, unsafe_allow_html=True)
            
            future_baseline = build_future_feature_row(last_row, next_quarter_label)

            # Single prediction
            result = predict.predict_single(
                future_baseline,
                model_name=forecast_model if forecast_model != 'auto' else 'auto',
                split=forecast_split
            )
            
            forecast_pred = result['prediction']
            forecast_risk = result['risk_level']
            thresholds = result['thresholds']
            
            # Main display
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95)); 
                        padding: 3rem; border-radius: 20px; border: 2px solid #667eea; text-align: center; margin-bottom: 2rem;'>
                <p style='margin: 0; font-size: 1rem; color: #90cdf4; font-weight: 600; text-transform: uppercase; letter-spacing: 2px;'>Predicted Total Claims for {next_quarter_label}</p>
                <p style='margin: 1rem 0 0 0; font-size: 3.5rem; color: #667eea; font-weight: 700; font-family: "Space Mono", monospace;'>RM{forecast_pred:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rc = risk_colors_map.get(forecast_risk, "#90cdf4")
                ri = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(forecast_risk, "ℹ️")
                st.markdown(f"""
                <div class='equal-kpi-card' style='background: rgba(22, 33, 62, 0.8); padding: 2rem; border-radius: 15px; border: 2px solid {rc}; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>RISK LEVEL</p>
                    <p class='kpi-value' style='margin: 0.5rem 0 0 0; font-size: 2.5rem; color: {rc}; font-weight: 700;'>{ri} {forecast_risk}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                last_actual = float(last_row.get('Quarterly_Total_Claims', 0)) if pd.notna(last_row.get('Quarterly_Total_Claims', None)) else 0
                if last_actual != 0:
                    change_pct = ((forecast_pred - last_actual) / last_actual) * 100
                    change_color = "#e94560" if change_pct > 0 else "#2ecc71"
                    change_icon = "📈" if change_pct > 0 else "📉"
                else:
                    change_pct = 0
                    change_color = "#90cdf4"
                    change_icon = "➡️"
                
                st.markdown(f"""
                <div class='equal-kpi-card' style='background: rgba(22, 33, 62, 0.8); padding: 2rem; border-radius: 15px; border: 2px solid {change_color}; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>VS LAST QUARTER</p>
                    <p class='kpi-value' style='margin: 0.5rem 0 0 0; font-size: 2.5rem; color: {change_color}; font-weight: 700;'>{change_icon} {change_pct:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='equal-kpi-card' style='background: rgba(22, 33, 62, 0.8); padding: 2rem; border-radius: 15px; border: 2px solid #48cae4; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>LAST QUARTER ACTUAL</p>
                    <p class='kpi-value' style='margin: 0.5rem 0 0 0; font-size: 2rem; color: #48cae4; font-weight: 700; font-family: "Space Mono", monospace;'>RM{last_actual:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Timeline chart
            st.markdown("## 📈 Historical + Forecast")
            
            fig = go.Figure()
            if os.path.exists(quarterly_path):
                hist_df = pd.read_csv(quarterly_path)
                fig.add_trace(go.Scatter(
                    x=hist_df['Quarter_Label'], y=hist_df['Quarterly_Total_Claims'],
                    mode='lines+markers', name='Historical',
                    line=dict(color='#e94560', width=3), marker=dict(size=8, color='#e94560'),
                    hovertemplate='%{x}: RM%{y:,.0f}<extra></extra>'
                ))
                # Connect last historical to forecast
                fig.add_trace(go.Scatter(
                    x=[hist_df['Quarter_Label'].iloc[-1], next_quarter_label],
                    y=[hist_df['Quarterly_Total_Claims'].iloc[-1], forecast_pred],
                    mode='lines', line=dict(color='#667eea', width=2, dash='dot'),
                    showlegend=False, hoverinfo='skip'
                ))
            
            fig.add_trace(go.Scatter(
                x=[next_quarter_label], y=[forecast_pred],
                mode='markers', name='Forecast',
                marker=dict(size=20, color='#667eea', symbol='star', line=dict(width=3, color='white')),
                hovertemplate=f'Forecast: RM{forecast_pred:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text=f'Claims Timeline + {next_quarter_label} Forecast', font=dict(size=22, family='Outfit, sans-serif', color='#48cae4')),
                xaxis_title='Quarter', yaxis_title='Total Claims (RM)',
                template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500, hovermode='x unified',
                yaxis=dict(tickformat=',')
            )
            apply_chart_layout(fig, bottom_margin=135, legend_y=-0.30)
            st.plotly_chart(fig, use_container_width=True)
        
        # =============================================================
        # NEXT YEAR (4 Quarters — Pure ML Recursive Forecast)
        # =============================================================
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); 
                        border: 2px solid #667eea; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;'>
                <p style='margin: 0; font-size: 1rem; color: #90cdf4;'>FORECASTING FOR</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 3rem; color: #667eea; font-weight: 700; font-family: "Space Mono", monospace;'>{next_quarters[0]} to {next_quarters[3]}</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #48cae4;'>4 Quarters Ahead (Recursive ML Forecast) | Based on: {last_quarter_label}</p>
            </div>
            """, unsafe_allow_html=True)
            
            thresholds = None
            quarterly_predictions = []
            forecast_state = last_row.to_dict() if hasattr(last_row, 'to_dict') else dict(last_row)
            
            for q_label in next_quarters:
                future_values = build_future_feature_row(forecast_state, q_label)
                q_result = predict.predict_single(
                    future_values,
                    model_name=forecast_model if forecast_model != 'auto' else 'auto',
                    split=forecast_split
                )
                q_prediction = q_result['prediction']
                thresholds = q_result['thresholds']
                
                # Assign risk
                if q_prediction <= thresholds['low_max']:
                    q_risk = 'LOW'
                elif q_prediction <= thresholds['medium_max']:
                    q_risk = 'MEDIUM'
                else:
                    q_risk = 'HIGH'
                
                quarterly_predictions.append({
                    'quarter': q_label,
                    'prediction': round(q_prediction, 2),
                    'risk': q_risk
                })

                forecast_state.update(future_values)
                forecast_state['Quarterly_Total_Claims'] = q_prediction
            
            # Annual total = sum of 4 quarters
            annual_total = sum(q['prediction'] for q in quarterly_predictions)
            annual_avg = annual_total / 4
            
            # Annual risk
            if annual_avg > thresholds['medium_max']:
                annual_risk = 'HIGH'
            elif annual_avg > thresholds['low_max']:
                annual_risk = 'MEDIUM'
            else:
                annual_risk = 'LOW'
            
            # ===== ANNUAL TOTAL =====
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95)); 
                        padding: 3rem; border-radius: 20px; border: 2px solid #667eea; text-align: center; margin-bottom: 2rem;'>
                <p style='margin: 0; font-size: 1rem; color: #90cdf4; font-weight: 600; text-transform: uppercase; letter-spacing: 2px;'>Predicted Annual Total Claims</p>
                <p style='margin: 1rem 0 0 0; font-size: 3.5rem; color: #667eea; font-weight: 700; font-family: "Space Mono", monospace;'>RM{annual_total:,.0f}</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1rem; color: #48cae4;'>Average per Quarter: RM{annual_avg:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rc = risk_colors_map.get(annual_risk, "#90cdf4")
                ri = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(annual_risk, "ℹ️")
                st.markdown(f"""
                <div style='background: rgba(22, 33, 62, 0.8); padding: 2rem; border-radius: 15px; border: 2px solid {rc}; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>ANNUAL RISK</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 2.5rem; color: {rc}; font-weight: 700;'>{ri} {annual_risk}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                last_actual = float(last_row.get('Quarterly_Total_Claims', 0)) if pd.notna(last_row.get('Quarterly_Total_Claims', None)) else 0
                last_year_approx = last_actual * 4
                if last_year_approx != 0:
                    yr_change = ((annual_total - last_year_approx) / last_year_approx) * 100
                    yr_color = "#e94560" if yr_change > 0 else "#2ecc71"
                    yr_icon = "📈" if yr_change > 0 else "📉"
                else:
                    yr_change = 0
                    yr_color = "#90cdf4"
                    yr_icon = "➡️"
                st.markdown(f"""
                <div style='background: rgba(22, 33, 62, 0.8); padding: 2rem; border-radius: 15px; border: 2px solid {yr_color}; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>VS LAST YEAR (EST.)</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 2.5rem; color: {yr_color}; font-weight: 700;'>{yr_icon} {yr_change:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                highest_q = max(quarterly_predictions, key=lambda x: x['prediction'])
                st.markdown(f"""
                <div style='background: rgba(22, 33, 62, 0.8); padding: 2rem; border-radius: 15px; border: 2px solid #48cae4; text-align: center;'>
                    <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>PEAK QUARTER</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #e94560; font-weight: 700;'>{highest_q["quarter"]}</p>
                    <p style='margin: 0.2rem 0 0 0; font-size: 1rem; color: #48cae4;'>RM{highest_q["prediction"]:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ===== QUARTERLY CARDS =====
            st.markdown("## 📊 Quarterly Breakdown")
            
            q_cols = st.columns(4)
            for i, q_data in enumerate(quarterly_predictions):
                with q_cols[i]:
                    rc = risk_colors_map.get(q_data['risk'], '#90cdf4')
                    ri = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(q_data['risk'], "ℹ️")
                    pct_of_total = (q_data['prediction'] / annual_total * 100) if annual_total != 0 else 25
                    
                    st.markdown(f"""
                    <div style='background: rgba(22, 33, 62, 0.8); padding: 1.5rem; border-radius: 12px; border: 2px solid #667eea; text-align: center;'>
                        <p style='margin: 0; font-size: 1rem; color: #48cae4; font-weight: 600;'>{q_data["quarter"]}</p>
                        <p style='margin: 0.8rem 0 0 0; font-size: 1.8rem; color: #667eea; font-weight: 700; font-family: "Space Mono", monospace;'>RM{q_data["prediction"]:,.0f}</p>
                        <p style='margin: 0.3rem 0 0 0; font-size: 0.85rem; color: {rc};'>{ri} {q_data["risk"]}</p>
                        <p style='margin: 0.5rem 0 0 0; font-size: 0.75rem; color: #90cdf4;'>{pct_of_total:.1f}% of annual</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ===== TIMELINE CHART =====
            st.markdown("## 📈 Historical + Next Year Forecast")
            
            fig = go.Figure()
            
            if os.path.exists(quarterly_path):
                hist_df = pd.read_csv(quarterly_path)
                fig.add_trace(go.Scatter(
                    x=hist_df['Quarter_Label'], y=hist_df['Quarterly_Total_Claims'],
                    mode='lines+markers', name='Historical Actual',
                    line=dict(color='#e94560', width=3), marker=dict(size=8, color='#e94560'),
                    hovertemplate='%{x}: RM%{y:,.0f}<extra></extra>'
                ))
                # Connect last historical to first forecast
                fig.add_trace(go.Scatter(
                    x=[hist_df['Quarter_Label'].iloc[-1], quarterly_predictions[0]['quarter']],
                    y=[hist_df['Quarterly_Total_Claims'].iloc[-1], quarterly_predictions[0]['prediction']],
                    mode='lines', line=dict(color='#667eea', width=2, dash='dot'),
                    showlegend=False, hoverinfo='skip'
                ))
            
            fig.add_trace(go.Scatter(
                x=[q['quarter'] for q in quarterly_predictions],
                y=[q['prediction'] for q in quarterly_predictions],
                mode='lines+markers', name='ML Forecast',
                line=dict(color='#667eea', width=4, dash='dash'),
                marker=dict(size=12, color='#667eea', symbol='diamond', line=dict(width=2, color='white')),
                hovertemplate='%{x}: RM%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text='Historical + Recursive ML Forecast', font=dict(size=22, family='Outfit, sans-serif', color='#48cae4')),
                xaxis_title='Quarter', yaxis_title='Total Claims (RM)',
                template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500, hovermode='x unified',
                yaxis=dict(tickformat=',')
            )
            apply_chart_layout(fig, bottom_margin=135, legend_y=-0.30)
            st.plotly_chart(fig, use_container_width=True)
            
            # ===== BAR CHART =====
            st.markdown("## 📊 Quarterly Comparison")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[q['quarter'] for q in quarterly_predictions],
                y=[q['prediction'] for q in quarterly_predictions],
                marker=dict(color=[risk_colors_map.get(q['risk'], '#667eea') for q in quarterly_predictions], line=dict(color='white', width=2)),
                text=[f"RM{q['prediction']:,.0f}" for q in quarterly_predictions],
                textposition='outside', textfont=dict(size=12)
            ))
            fig2.update_layout(
                title=dict(text='Predicted Claims per Quarter (ML)', font=dict(size=22, family='Outfit, sans-serif', color='#48cae4')),
                template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=400, showlegend=False,
                yaxis=dict(tickformat=',')
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # ===== SUMMARY TABLE =====
            st.markdown("## 📋 Summary")
            summary_data = []
            for q_data in quarterly_predictions:
                summary_data.append({
                    'Quarter': q_data['quarter'],
                    'Predicted Claims (RM)': f"RM{q_data['prediction']:,.0f}",
                    'Risk Level': q_data['risk'],
                    '% of Annual': f"{(q_data['prediction'] / annual_total * 100):.1f}%" if annual_total != 0 else "25%"
                })
            summary_data.append({
                'Quarter': '📊 ANNUAL TOTAL',
                'Predicted Claims (RM)': f"RM{annual_total:,.0f}",
                'Risk Level': annual_risk,
                '% of Annual': '100%'
            })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Methodology note
            st.markdown("""
            <div style='background: rgba(22, 33, 62, 0.6); padding: 1rem; border-radius: 10px; border-left: 4px solid #764ba2; margin-top: 1rem;'>
                <p style='color: #90cdf4; margin: 0; font-size: 0.85rem;'>
                    <strong style='color: #764ba2;'>📌 Methodology:</strong> Recursive multi-step ML forecast. 
                    Each quarter's prediction is generated by the trained model, then fed back as input (lag features) 
                    to predict the next quarter. This is a standard time-series ML technique (autoregressive forecasting).
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # ========== SHARED: Thresholds & Download ==========
        st.markdown("---")
        
        st.markdown(f"""
        <div style='background: rgba(22, 33, 62, 0.6); padding: 1rem; border-radius: 10px; border-left: 4px solid #48cae4;'>
            <p style='color: #90cdf4; margin: 0;'>
                <strong style='color: #48cae4;'>Risk Thresholds:</strong>
                <span style='color: #2ecc71;'>● LOW</span> ≤ RM{thresholds['low_max']:,.0f} | 
                <span style='color: #f1c40f;'>● MEDIUM</span> ≤ RM{thresholds['medium_max']:,.0f} | 
                <span style='color: #e74c3c;'>● HIGH</span> > RM{thresholds['medium_max']:,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        
        if forecast_mode == "Next Quarter":
            dl_data = [{'Quarter': next_quarter_label, 'Predicted Claims (RM)': forecast_pred, 'Risk Level': forecast_risk}]
            filename = f"forecast_{next_quarter_label}.csv"
        else:
            dl_data = []
            for q_data in quarterly_predictions:
                dl_data.append({'Quarter': q_data['quarter'], 'Predicted Claims (RM)': q_data['prediction'], 'Risk Level': q_data['risk']})
            dl_data.append({'Quarter': 'ANNUAL TOTAL', 'Predicted Claims (RM)': annual_total, 'Risk Level': annual_risk})
            filename = f"forecast_{next_quarters[0]}_to_{next_quarters[3]}.csv"
        
        dl_df = pd.DataFrame(dl_data)
        csv = dl_df.to_csv(index=False)
        st.download_button("📥 Download Forecast Report", csv, filename, "text/csv", use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Forecast failed: {e}")
        st.exception(e)
        
# =========================================================
# MODEL COMPARISON
# =========================================================
elif page == "⚖️ Model Comparison":
    st.title("⚖️ Model Performance Comparison")

    required_cols = ['split', 'model', 'MAE', 'RMSE', 'R2', 'MAPE']
    
    if not show_data_unavailable_message("Model Performance Comparison", "outputs/evaluation_master.csv", required_cols):
        st.stop()
    
    eval_df = pd.read_csv("outputs/evaluation_master.csv")

    split_choice = st.selectbox("📊 Select Split", eval_df['split'].unique())
    subset = eval_df[eval_df['split'] == split_choice]

    st.markdown(f"## 📊 Performance Metrics ({split_choice})")
    metrics_display = subset[['model', 'MAE', 'RMSE', 'R2', 'MAPE']].copy().sort_values('RMSE')
    
    st.dataframe(
        metrics_display.style.format({
            'MAE': 'RM{:,.2f}',
            'RMSE': 'RM{:,.2f}',
            'R2': '{:.4f}',
            'MAPE': '{:.2f}%'
        }).background_gradient(subset=['RMSE', 'MAE'], cmap='RdYlGn_r'),
        use_container_width=True
    )

    metric_choice = st.selectbox("📈 Metric to Compare", ['RMSE', 'MAE', 'R2', 'MAPE'])
    
    st.caption("Model comparison metrics are calculated on out-of-sample test rows only.")

    fig = go.Figure()
    n = len(subset)
    colors = px.colors.sample_colorscale('Turbo', [i/(n-1) if n > 1 else 0 for i in range(n)])
    
    if metric_choice in ['RMSE', 'MAE', 'MAPE']:
        best_idx = subset[metric_choice].idxmin()
    else:
        best_idx = subset[metric_choice].idxmax()
    
    bar_colors = ['#2ecc71' if i == best_idx else colors[idx] for idx, i in enumerate(subset.index)]
    
    fig.add_trace(go.Bar(
        x=subset['model'], y=subset[metric_choice],
        marker=dict(color=bar_colors, line=dict(color='white', width=2)),
        text=[f'{v:.2f}' for v in subset[metric_choice]], textposition='outside'
    ))
    fig.update_layout(
        title=dict(text=f'Model Comparison: {metric_choice}', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
        template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 📊 All Metrics Comparison")
    fig2 = go.Figure()
    for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
        fig2.add_trace(go.Bar(name=metric, x=subset['model'], y=subset[metric], text=[f'{v:.2f}' for v in subset[metric]], textposition='outside'))
    fig2.update_layout(
        barmode='group',
        title=dict(text='All Metrics by Model', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
        template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500,
    )
    apply_chart_layout(fig2, bottom_margin=135, legend_y=-0.30)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## 🎯 Feature Importance Analysis")
    if os.path.exists("outputs/"):
        fi_files = [f for f in os.listdir("outputs/") if f.startswith("feature_importance_") and f.endswith('.csv')]
    else:
        fi_files = []
    
    if fi_files:
        fi_file = st.selectbox("📁 Select Feature Importance File", fi_files)
        fi_df = pd.read_csv(f"outputs/{fi_file}")
        top_n = st.slider("Number of Top Features", 5)
        top = fi_df.head(top_n).sort_values('importance', ascending=True)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=top['feature'], x=top['importance'], orientation='h',
            marker=dict(color=top['importance'], colorscale='Viridis', line=dict(color='white', width=2)),
            text=[f'{v:.4f}' for v in top['importance']], textposition='outside'
        ))
        fig3.update_layout(
            title=dict(text=f'Top {top_n} Features', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
            template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=max(400, top_n * 25),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("ℹ️ Feature importance files not found.")


# =========================================================
# WHAT-IF SIMULATOR (FINAL FIXED VERSION)
# =========================================================
elif page == "🎛️ What-If Simulator":
    st.title("🎛️ What-If Prediction Simulator")

    st.markdown("""
    <div style='background: rgba(72, 202, 228, 0.1); border-left: 4px solid #48cae4; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <p style='color: #90cdf4; margin: 0;'>
            Simulate different scenarios to predict next quarter's insurance claims.<br>
            <strong style='color: #48cae4;'>Choose a mode below:</strong> Simple Mode for quick predictions, or Advanced Mode for full control.
            <br>All monetary values in <strong>Malaysian Ringgit (RM)</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("What-if outputs combine the selected model prediction with a scenario adjustment so slider changes stay visible and interpretable.")

    # Check for required files
    pred_file = "outputs/predictions_80_20.csv"
    feat_path = "models/80_20/feature_columns.txt"
    
    missing_files = []
    if not os.path.exists(pred_file):
        missing_files.append(pred_file)
    if not os.path.exists(feat_path):
        missing_files.append(feat_path)
    
    if missing_files:
        show_data_unavailable_card(
            "What-If Prediction Simulator",
            missing_files,
            action="Please run the complete pipeline from 'Upload & Run Pipeline' before using the simulator."
        )
        st.stop()

    # Load feature columns
    with open(feat_path, 'r') as f:
        feature_cols = f.read().strip().split('\n')

    # Load ALL data (train + test) for full feature ranges
    test_path = "test_data_20.csv"
    train_path = "train_data_80.csv"
    all_data = None
    baseline = {}
    
    # Combine train + test for full range of feature statistics
    dfs_to_combine = []
    if os.path.exists(train_path):
        dfs_to_combine.append(pd.read_csv(train_path))
    if os.path.exists(test_path):
        dfs_to_combine.append(pd.read_csv(test_path))
    
    if dfs_to_combine:
        all_data = pd.concat(dfs_to_combine, ignore_index=True)
        # Baseline = last row of test data (most recent quarter)
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            if len(test_df) > 0:
                for col in feature_cols:
                    if col in test_df.columns:
                        val = test_df[col].iloc[-1]
                        baseline[col] = float(val) if pd.notna(val) else 0.0
                    else:
                        baseline[col] = 0.0
            else:
                baseline = {col: 0.0 for col in feature_cols}
        else:
            baseline = {col: 0.0 for col in feature_cols}
    else:
        baseline = {col: 0.0 for col in feature_cols}

    # Also try to load the quarterly claims data for better range understanding
    quarterly_path = "outputs/quarterly_claims.csv"
    if os.path.exists(quarterly_path):
        quarterly_df = pd.read_csv(quarterly_path)
    else:
        quarterly_df = None

    # =====================================================
    # COMPUTE FEATURE STATISTICS FROM ALL DATA (train+test)
    # =====================================================
    feature_stats = {}
    if all_data is not None and len(all_data) > 1:
        for col in feature_cols:
            if col in all_data.columns:
                col_data = all_data[col].dropna()
                col_data = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(col_data) > 0:
                    feature_stats[col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()) if len(col_data) > 1 else 0,
                    }
                else:
                    feature_stats[col] = {'min': 0, 'max': 1, 'mean': 0, 'std': 0}
            else:
                feature_stats[col] = {'min': 0, 'max': 1, 'mean': 0, 'std': 0}
    else:
        for col in feature_cols:
            bv = baseline.get(col, 0)
            feature_stats[col] = {'min': 0, 'max': max(abs(bv) * 2, 1), 'mean': bv, 'std': 0}

    # =====================================================
    # apply_adjustments — FIXED to always produce changes
    # =====================================================
    def apply_adjustments(baseline_values, claims_pct, incidents_pct, severity_pct, growth_pct):
        """
        Apply percentage-based adjustments using ACTUAL DATA RANGES.
        
        pct = 0   → use baseline (last known value)
        pct = +50 → move halfway between baseline and historical max
        pct = +100→ use historical max (or 2x baseline if baseline IS the max)
        pct = -50 → move halfway between baseline and historical min
        pct = -100→ use historical min (or 0.2x baseline if baseline IS the min)
        """
        adjusted = {}
        
        for col in feature_cols:
            base_val = baseline_values.get(col, 0.0)
            stats = feature_stats.get(col, {'min': 0, 'max': base_val * 2 if base_val > 0 else 1, 'mean': base_val, 'std': 0})
            
            # Determine which adjustment percentage to use for this feature
            col_lower = col.lower()
            if any(k in col_lower for k in ['growth', 'change', 'diff', 'trend', 'momentum', 'pct_change', 'rate']):
                pct = growth_pct
            elif any(k in col_lower for k in ['avg', 'mean', 'max_claim', 'severity', 'per_incident', 'average', 'median']):
                pct = severity_pct
            elif any(k in col_lower for k in ['count', 'incident', 'frequency', 'num_', 'number', 'n_claims', 'severe', 'deductible']):
                pct = incidents_pct
            else:
                pct = claims_pct
            
            # Apply interpolation-based adjustment
            if pct == 0:
                adjusted[col] = base_val
            elif pct > 0:
                data_max = stats['max']
                # CRITICAL FIX: Ensure max is meaningfully higher than baseline
                if data_max <= base_val * 1.01:
                    if base_val > 0:
                        data_max = base_val * 2.0
                    elif base_val == 0:
                        data_max = stats['mean'] + stats['std'] * 2 if stats['std'] > 0 else 1.0
                    else:
                        data_max = base_val * 0.5  # negative: max is closer to zero
                adjusted[col] = base_val + (data_max - base_val) * (pct / 100)
            else:
                data_min = stats['min']
                # CRITICAL FIX: Ensure min is meaningfully lower than baseline
                if data_min >= base_val * 0.99:
                    if base_val > 0:
                        data_min = base_val * 0.2
                    elif base_val == 0:
                        data_min = -(stats['mean'] + stats['std'] * 2) if stats['std'] > 0 else -1.0
                    else:
                        data_min = base_val * 2.0  # negative: min is more negative
                adjusted[col] = base_val + (data_min - base_val) * (abs(pct) / 100)
        
        return adjusted

    # =====================================================
    # DEFINE PRESET SCENARIOS
    # =====================================================
    scenarios = {
        "🔄 Business as Usual": {
            "description": "Continue current trends with no significant changes.",
            "adjustments": {"claims_pct": 0, "incidents_pct": 0, "severity_pct": 0, "growth_pct": 0},
            "color": "#667eea"
        },
        "📈 Moderate Growth": {
            "description": "Slight increase — growing portfolio or mild seasonal effect.",
            "adjustments": {"claims_pct": 25, "incidents_pct": 20, "severity_pct": 15, "growth_pct": 20},
            "color": "#48cae4"
        },
        "🔴 High Claim Season": {
            "description": "Monsoon season or festive period — significant spike expected.",
            "adjustments": {"claims_pct": 60, "incidents_pct": 55, "severity_pct": 40, "growth_pct": 50},
            "color": "#e94560"
        },
        "🌊 Crisis / Disaster": {
            "description": "Major flood, pandemic, or catastrophic event.",
            "adjustments": {"claims_pct": 100, "incidents_pct": 90, "severity_pct": 80, "growth_pct": 90},
            "color": "#ff6b6b"
        },
        "📉 Low Activity": {
            "description": "Below-average activity. Fewer claims, lower severity.",
            "adjustments": {"claims_pct": -40, "incidents_pct": -50, "severity_pct": -30, "growth_pct": -40},
            "color": "#2ecc71"
        }
    }

    # =====================================================
    # MODEL SELECTION
    # =====================================================
    st.markdown("---")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
         model_choice = st.selectbox(
            "🤖 Prediction Model", 
            ['auto', 'Ensemble', 'XGBoost', 'RandomForest', 'LinearRegression'], 
            key="sim_model"
        )
    with col_m2:
        split_choice_sim = st.selectbox("📊 Data Split", ['80_20', '70_30', '90_10'], key="sim_split")


    # Helper function for prediction — uses predict_single (proper scaler)
    def scenario_impact_multiplier(claims_pct, incidents_pct, severity_pct, growth_pct):
        """
        Convert business-friendly slider percentages into a smooth claims impact.
        Tree models can return the same prediction for many nearby feature values,
        so the simulator applies this transparent overlay to keep what-if changes visible.
        """
        weighted_pct = (
            (claims_pct * 0.45) +
            (incidents_pct * 0.25) +
            (severity_pct * 0.20) +
            (growth_pct * 0.10)
        )
        return max(0.05, 1 + (weighted_pct / 100))

    def run_prediction(input_values, adjustment_pcts=None):
        """Run prediction and optionally apply a smooth what-if business impact."""
        result = predict.predict_single(
            input_values,
            model_name=model_choice if model_choice != 'auto' else 'auto',
            split=split_choice_sim
        )
        if adjustment_pcts is None:
            return result

        multiplier = scenario_impact_multiplier(*adjustment_pcts)
        adjusted_pred = max(0, result['prediction'] * multiplier)
        result = result.copy()
        result['prediction'] = round(adjusted_pred, 2)
        result['risk_level'] = predict.assign_risk_level(adjusted_pred, result['thresholds'])
        result['impact_multiplier'] = multiplier
        return result

    # =====================================================
    # TABS
    # =====================================================
    sim_tab1, sim_tab2, sim_tab3 = st.tabs(["🎯 Simple Mode", "📊 Scenario Cards", "🔧 Advanced Mode"])

    # -------------------------------------------------
    # TAB 1: SIMPLE MODE (Just sliders, no scenarios)
    # -------------------------------------------------
    with sim_tab1:
        st.markdown("### 🎯 Simple Mode — Adjust by Percentage")
        st.markdown("""
        <p style='color: #90cdf4; margin-bottom: 1.5rem;'>
            Adjust the sliders to simulate changes from last quarter's values.
            <br><strong>0%</strong> = Last quarter | <strong>+100%</strong> = Historical max | <strong>-100%</strong> = Historical min
        </p>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎚️ Adjustment Sliders")
        
        col1, col2 = st.columns(2)
        
        with col1:
            claims_pct = st.slider(
                "💰 Claims Volume",
                min_value=-100, max_value=100,
                value=0,
                step=5,
                help="How much total claims change vs last quarter",
                key="simple_claims"
            )
            
            incidents_pct = st.slider(
                "🚨 Incident Frequency",
                min_value=-100, max_value=100,
                value=0,
                step=5,
                help="How much the number of incidents changes",
                key="simple_incidents"
            )
        
        with col2:
            severity_pct = st.slider(
                "⚡ Claim Severity",
                min_value=-100, max_value=100,
                value=0,
                step=5,
                help="How much the average claim size changes",
                key="simple_severity"
            )
            
            growth_pct = st.slider(
                "📈 Growth Trend",
                min_value=-100, max_value=100,
                value=0,
                step=5,
                help="Overall growth momentum",
                key="simple_growth"
            )

        # Adjustment Summary
        st.markdown("#### 📊 Current Adjustments")
        adj_col1, adj_col2, adj_col3, adj_col4 = st.columns(4)
        
        for col_widget, label, pct in [
            (adj_col1, "Claims", claims_pct),
            (adj_col2, "Incidents", incidents_pct),
            (adj_col3, "Severity", severity_pct),
            (adj_col4, "Growth", growth_pct)
        ]:
            with col_widget:
                pct_color = "#e94560" if pct > 0 else "#2ecc71" if pct < 0 else "#90cdf4"
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; border-radius: 10px; background: rgba(22, 33, 62, 0.8); border: 1px solid {pct_color};'>
                    <p style='margin: 0; font-size: 0.8rem; color: #90cdf4;'>{label}</p>
                    <p style='margin: 0; font-size: 1.5rem; color: {pct_color}; font-weight: 700;'>{pct:+d}%</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Predict Button
        if st.button("🔮 Predict Next Quarter", type="primary", use_container_width=True, key="simple_predict"):
            try:
                # Apply adjustments using actual data ranges
                adjusted_values = apply_adjustments(
                    baseline, claims_pct, incidents_pct, severity_pct, growth_pct
                )

                # Baseline = apply_adjustments with 0% (ensures same dict structure)
                baseline_values = apply_adjustments(
                    baseline, 0, 0, 0, 0
                )

                # Predict with proper scaling, then apply simulator impact.
                result = run_prediction(
                    adjusted_values,
                    adjustment_pcts=(claims_pct, incidents_pct, severity_pct, growth_pct)
                )

                # Baseline for comparison (same path, same dict structure)
                baseline_result = run_prediction(
                    baseline_values,
                    adjustment_pcts=(0, 0, 0, 0)
                )
                
                baseline_pred = baseline_result['prediction']

                pred = result['prediction']
                risk = result['risk_level']
                model_used = result['model']
                thresholds = result['thresholds']

                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class='equal-kpi-card sim-result-card' style='background: rgba(102, 126, 234, 0.3); padding: 2rem; border-radius: 15px; border: 2px solid #667eea; text-align: center;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>PREDICTED CLAIMS</p>
                        <p class='kpi-value' style='margin: 0.5rem 0 0 0; font-size: 2.5rem; color: #667eea; font-weight: 700; font-family: "Space Mono", monospace;'>RM{pred:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    risk_colors_map = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}
                    rc = risk_colors_map.get(risk, "#90cdf4")
                    ri = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(risk, "ℹ️")
                    st.markdown(f"""
                    <div class='equal-kpi-card sim-result-card' style='background: rgba(233, 69, 96, 0.3); padding: 2rem; border-radius: 15px; border: 2px solid {rc}; text-align: center;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>RISK LEVEL</p>
                        <p class='kpi-value' style='margin: 0.5rem 0 0 0; font-size: 2.5rem; color: {rc}; font-weight: 700;'>{ri} {risk}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class='equal-kpi-card sim-result-card' style='background: rgba(72, 202, 228, 0.3); padding: 2rem; border-radius: 15px; border: 2px solid #48cae4; text-align: center;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #90cdf4; font-weight: 600;'>MODEL USED</p>
                        <p class='kpi-value' style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #48cae4; font-weight: 700;'>{model_used}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Comparison
                change_from_baseline = ((pred - baseline_pred) / baseline_pred * 100) if baseline_pred != 0 else 0
                change_color = "#e94560" if change_from_baseline > 0 else "#2ecc71" if change_from_baseline < 0 else "#90cdf4"
                
                st.markdown(f"""
                <div style='background: rgba(22, 33, 62, 0.6); padding: 1.5rem; border-radius: 10px; margin-top: 1.5rem; border: 1px solid rgba(72, 202, 228, 0.3);'>
                    <table style='width: 100%; color: #90cdf4;'>
                        <tr>
                            <td style='text-align: left;'><span style='font-size: 0.85rem;'>Baseline (0%):</span><br><strong style='color: #667eea; font-size: 1.2rem;'>RM{baseline_pred:,.0f}</strong></td>
                            <td style='text-align: center;'><span style='font-size: 0.85rem;'>Change:</span><br><strong style='color: {change_color}; font-size: 1.2rem;'>{change_from_baseline:+.1f}%</strong></td>
                            <td style='text-align: right;'><span style='font-size: 0.85rem;'>Scenario:</span><br><strong style='color: #e94560; font-size: 1.2rem;'>RM{pred:,.0f}</strong></td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style='background: rgba(22, 33, 62, 0.6); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #48cae4;'>
                    <p style='color: #90cdf4; margin: 0;'>
                        <strong style='color: #48cae4;'>Risk Thresholds:</strong>
                        <span style='color: #2ecc71;'>● LOW</span> ≤ RM{thresholds['low_max']:,.0f} | 
                        <span style='color: #f1c40f;'>● MEDIUM</span> ≤ RM{thresholds['medium_max']:,.0f} | 
                        <span style='color: #e74c3c;'>● HIGH</span> > RM{thresholds['medium_max']:,.0f}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)

    # -------------------------------------------------
    # TAB 2: SCENARIO CARDS
    # -------------------------------------------------
    with sim_tab2:
        st.markdown("### 📊 Quick Scenario Comparison")
        st.markdown("<p style='color: #90cdf4;'>Compare all scenarios side-by-side with one click.</p>", unsafe_allow_html=True)

        if st.button("🚀 Run All Scenarios & Compare", use_container_width=True, type="primary", key="run_all"):
            results_data = []
            prediction_values = []
            
            for scenario_name, scenario_info in scenarios.items():
                try:
                    adj = scenario_info['adjustments']
                    adjusted = apply_adjustments(baseline, adj['claims_pct'], adj['incidents_pct'], adj['severity_pct'], adj['growth_pct'])
                    result = run_prediction(
                        adjusted,
                        adjustment_pcts=(adj['claims_pct'], adj['incidents_pct'], adj['severity_pct'], adj['growth_pct'])
                    )
                    
                    results_data.append({
                        'Scenario': scenario_name,
                        'Predicted Claims': f"RM{result['prediction']:,.0f}",
                        'Risk Level': result['risk_level'],
                        'Claims': f"{adj['claims_pct']:+d}%",
                        'Incidents': f"{adj['incidents_pct']:+d}%",
                        'Severity': f"{adj['severity_pct']:+d}%",
                        'Growth': f"{adj['growth_pct']:+d}%"
                    })
                    prediction_values.append({
                        'name': scenario_name,
                        'prediction': result['prediction'],
                        'risk': result['risk_level'],
                        'color': scenario_info['color']
                    })
                except Exception as e:
                    results_data.append({
                        'Scenario': scenario_name,
                        'Predicted Claims': f'Error: {e}',
                        'Risk Level': 'N/A',
                        'Claims': '-', 'Incidents': '-', 'Severity': '-', 'Growth': '-'
                    })
            
            # Results table
            st.markdown("#### 📋 Results Table")
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            
            # Bar chart
            if prediction_values:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[p['name'] for p in prediction_values],
                    y=[p['prediction'] for p in prediction_values],
                    marker=dict(color=[p['color'] for p in prediction_values], line=dict(color='white', width=2)),
                    text=[f"RM{p['prediction']:,.0f}<br>({p['risk']})" for p in prediction_values],
                    textposition='outside', textfont=dict(size=11)
                ))
                fig.update_layout(
                    title=dict(text='Predicted Claims by Scenario', font=dict(size=24, family='Outfit, sans-serif', color='#48cae4')),
                    xaxis_title='Scenario', yaxis_title='Predicted Claims (RM)',
                    template='plotly_dark', plot_bgcolor='rgba(22, 33, 62, 0.8)', paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(family='Outfit, sans-serif', color='#90cdf4'), height=500, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Individual Scenarios")
        
        for scenario_name, scenario_info in scenarios.items():
            adj = scenario_info['adjustments']
            with st.expander(f"{scenario_name} — Claims: {adj['claims_pct']:+d}%, Incidents: {adj['incidents_pct']:+d}%, Severity: {adj['severity_pct']:+d}%"):
                st.markdown(f"<p style='color: #90cdf4;'>{scenario_info['description']}</p>", unsafe_allow_html=True)
                if st.button(f"🔮 Predict", key=f"card_{scenario_name}", use_container_width=True):
                    try:
                        adjusted = apply_adjustments(baseline, adj['claims_pct'], adj['incidents_pct'], adj['severity_pct'], adj['growth_pct'])
                        result = run_prediction(
                            adjusted,
                            adjustment_pcts=(adj['claims_pct'], adj['incidents_pct'], adj['severity_pct'], adj['growth_pct'])
                        )
                        rc = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}.get(result['risk_level'], "#90cdf4")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"<div style='background: rgba(102, 126, 234, 0.2); padding: 1.5rem; border-radius: 10px; border: 2px solid #667eea; text-align: center;'><p style='margin:0;color:#90cdf4;font-size:0.85rem;'>PREDICTED</p><p style='margin:0;color:#667eea;font-size:2rem;font-weight:700;font-family:\"Space Mono\",monospace;'>RM{result['prediction']:,.0f}</p></div>", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"<div style='background: rgba(233, 69, 96, 0.2); padding: 1.5rem; border-radius: 10px; border: 2px solid {rc}; text-align: center;'><p style='margin:0;color:#90cdf4;font-size:0.85rem;'>RISK</p><p style='margin:0;color:{rc};font-size:2rem;font-weight:700;'>{result['risk_level']}</p></div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # -------------------------------------------------
    # TAB 3: ADVANCED MODE
    # -------------------------------------------------
    with sim_tab3:
        st.markdown("### 🔧 Advanced Mode — Full Feature Control")
        st.markdown("""
        <div style='background: rgba(233, 69, 96, 0.1); border-left: 4px solid #e94560; padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;'>
            <p style='color: #90cdf4; margin: 0;'>⚠️ <strong>For technical users.</strong> Directly edit raw feature values.</p>
        </div>
        """, unsafe_allow_html=True)

        lag_feats = [c for c in feature_cols if '_lag' in c.lower() or 'lag_' in c.lower()]
        rolling_feats = [c for c in feature_cols if 'rolling' in c.lower() or 'moving' in c.lower()]
        trend_feats = [
            c for c in feature_cols
            if any(keyword in c.lower() for keyword in ('growth', 'change', 'diff', 'trend', 'momentum', 'rate'))
        ]
        seasonality_feats = [
            c for c in feature_cols
            if c.lower() in {'quarter_sin', 'quarter_cos'} or c.upper() in {'Q_1', 'Q_2', 'Q_3', 'Q_4'}
        ]
        grouped_features = lag_feats + rolling_feats + trend_feats + seasonality_feats
        other_feats = [c for c in feature_cols if c not in grouped_features]

        user_inputs = {}

        with st.expander(f"📊 Lag Features ({len(lag_feats)})", expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(lag_feats):
                with cols[i % 3]:
                    user_inputs[feat] = st.number_input(feat.replace('_', ' '), value=baseline.get(feat, 0.0), step=0.01, key=f"adv_lag_{i}", format="%.2f")

        with st.expander(f"📈 Trend Features ({len(trend_feats)})", expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(trend_feats):
                with cols[i % 3]:
                    user_inputs[feat] = st.number_input(feat.replace('_', ' '), value=baseline.get(feat, 0.0), step=0.01, key=f"adv_trend_{i}", format="%.2f")

        with st.expander(f"🔄 Rolling Features ({len(rolling_feats)})", expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(rolling_feats):
                with cols[i % 3]:
                    user_inputs[feat] = st.number_input(feat.replace('_', ' '), value=baseline.get(feat, 0.0), step=0.01, key=f"adv_roll_{i}", format="%.2f")

        with st.expander(f"📅 Seasonality Features ({len(seasonality_feats)})", expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(seasonality_feats):
                with cols[i % 3]:
                    user_inputs[feat] = st.number_input(feat.replace('_', ' '), value=baseline.get(feat, 0.0), step=0.01, key=f"adv_seasonality_{i}", format="%.2f")

        with st.expander(f"🔧 Other Features ({len(other_feats)})", expanded=False):
            cols = st.columns(3)
            for i, feat in enumerate(other_feats):
                with cols[i % 3]:
                    user_inputs[feat] = st.number_input(feat.replace('_', ' '), value=baseline.get(feat, 0.0), step=0.01, key=f"adv_other_{i}", format="%.2f")

        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True, key="adv_predict"):
            try:
                result = run_prediction(user_inputs)
                pred = result['prediction']
                risk = result['risk_level']
                thresholds = result['thresholds']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='equal-kpi-card sim-result-card' style='background: rgba(102, 126, 234, 0.3); padding: 2rem; border-radius: 15px; border: 2px solid #667eea; text-align: center;'><p style='margin:0;color:#90cdf4;font-size:0.9rem;'>PREDICTED CLAIMS</p><p class='kpi-value' style='margin:0.5rem 0 0 0;color:#667eea;font-size:2.5rem;font-weight:700;font-family:\"Space Mono\",monospace;'>RM{pred:,.2f}</p></div>", unsafe_allow_html=True)
                with col2:
                    rc = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}.get(risk, "#90cdf4")
                    ri = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(risk, "ℹ️")
                    st.markdown(f"<div class='equal-kpi-card sim-result-card' style='background: rgba(233, 69, 96, 0.3); padding: 2rem; border-radius: 15px; border: 2px solid {rc}; text-align: center;'><p style='margin:0;color:#90cdf4;font-size:0.9rem;'>RISK LEVEL</p><p class='kpi-value' style='margin:0.5rem 0 0 0;color:{rc};font-size:2.5rem;font-weight:700;'>{ri} {risk}</p></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='equal-kpi-card sim-result-card' style='background: rgba(72, 202, 228, 0.3); padding: 2rem; border-radius: 15px; border: 2px solid #48cae4; text-align: center;'><p style='margin:0;color:#90cdf4;font-size:0.9rem;'>MODEL</p><p class='kpi-value' style='margin:0.5rem 0 0 0;color:#48cae4;font-size:1.5rem;font-weight:700;'>{result['model']}</p></div>", unsafe_allow_html=True)
                
                st.markdown(f"<div style='background: rgba(22, 33, 62, 0.6); padding: 1rem; border-radius: 10px; margin-top: 1.5rem; border-left: 4px solid #48cae4;'><p style='color: #90cdf4; margin: 0;'><strong style='color: #48cae4;'>Risk Thresholds:</strong> <span style='color: #2ecc71;'>● LOW</span> ≤ RM{thresholds['low_max']:,.0f} | <span style='color: #f1c40f;'>● MEDIUM</span> ≤ RM{thresholds['medium_max']:,.0f} | <span style='color: #e74c3c;'>● HIGH</span> > RM{thresholds['medium_max']:,.0f}</p></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)
