"""
Linear Regression in Finance — Streamlit App v2
Simple & Multiple Linear Regression | OLS | CAPM | Fama-French | Diagnostics
The Mountain Path – World of Finance | Prof. V. Ravichandran
"""
import streamlit as st
from styles import inject_css
from tab_slr         import tab_slr
from tab_mlr         import tab_mlr
from tab_diagnostics import tab_diagnostics
from tab_finance     import tab_finance
from tab_code        import tab_code
from tab_qa          import tab_qa
from tab_vocab       import tab_vocab

st.set_page_config(
    page_title="Linear Regression in Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

NO_SEL = "user-select:none;-webkit-user-select:none"
FH = "'Playfair Display',serif"
FB = "'Source Sans Pro',sans-serif"

# ── Hero Header ───────────────────────────────────────────────────────────────
st.html(f"""
<div style="text-align:center;padding:28px 20px 16px;
            border-bottom:2px solid #FFD700;margin-bottom:24px;{NO_SEL}">
  <div style="color:#8892b0;-webkit-text-fill-color:#8892b0;
              font-family:{FB};font-size:.82rem;letter-spacing:2px;
              text-transform:uppercase;margin-bottom:8px">
    The Mountain Path – World of Finance
  </div>
  <div style="font-family:{FH};font-size:2.3rem;color:#FFD700;
              -webkit-text-fill-color:#FFD700;letter-spacing:1px;
              margin-bottom:6px;font-weight:700">
    Linear Regression in Finance
  </div>
  <div style="color:#8892b0;-webkit-text-fill-color:#8892b0;
              font-family:{FB};font-size:1rem;margin-bottom:4px">
    OLS Estimation · CAPM Beta · Fama-French · Diagnostics · Live NSE Data
  </div>
  <div style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;
              font-family:{FB};font-size:.85rem;font-style:italic;margin-bottom:14px">
    Prof. V. Ravichandran &nbsp;|&nbsp;
    28+ Years Corporate Finance &amp; Banking Experience &nbsp;|&nbsp;
    10+ Years Academic Excellence
  </div>
  <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
    <span style="background:rgba(0,51,102,0.6);color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #1e3a5f">📡 Live NSE Data</span>
    <span style="background:rgba(255,215,0,0.12);color:#FFD700;-webkit-text-fill-color:#FFD700;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #FFD700">📐 CAPM Beta</span>
    <span style="background:rgba(40,167,69,0.18);color:#28a745;-webkit-text-fill-color:#28a745;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #28a745">📊 Fama-French 3-Factor</span>
    <span style="background:rgba(255,159,67,0.15);color:#ff9f43;-webkit-text-fill-color:#ff9f43;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #ff9f43">🔬 OLS Diagnostics</span>
    <span style="background:rgba(162,155,254,0.15);color:#a29bfe;-webkit-text-fill-color:#a29bfe;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #a29bfe">📈 Bond Yield · P/E Models</span>
  </div>
</div>
""")

# ── Tabs ──────────────────────────────────────────────────────────────────────
TABS = st.tabs([
    "📈 SLR — Concepts",
    "📊 MLR — Multi-Factor",
    "🔬 Diagnostics",
    "🏦 Finance Lab",
    "🐍 Python Code",
    "🎓 Q&A / Self-Test",
    "📚 Education Hub",
])

with TABS[0]: tab_slr()
with TABS[1]: tab_mlr()
with TABS[2]: tab_diagnostics()
with TABS[3]: tab_finance()
with TABS[4]: tab_code()
with TABS[5]: tab_qa()
with TABS[6]: tab_vocab()

# ── Footer ────────────────────────────────────────────────────────────────────
st.html(f"""
<div style="text-align:center;padding:20px;color:#8892b0;-webkit-text-fill-color:#8892b0;
            font-family:{FB};font-size:.84rem;border-top:1px solid #1e3a5f;
            margin-top:28px;line-height:2.0;{NO_SEL}">
  <span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:700;
               font-family:{FH};font-size:1rem">
    The Mountain Path – World of Finance
  </span><br>
  <a href="https://www.linkedin.com/in/trichyravis" target="_blank"
     style="color:#FFD700;-webkit-text-fill-color:#FFD700;text-decoration:none;font-weight:600">
    🔗 LinkedIn</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://github.com/trichyravis" target="_blank"
     style="color:#FFD700;-webkit-text-fill-color:#FFD700;text-decoration:none;font-weight:600">
    💻 GitHub</a><br>
  <span style="color:#8892b0;-webkit-text-fill-color:#8892b0">
    Prof. V. Ravichandran &nbsp;|&nbsp;
    28+ Years Corporate Finance &amp; Banking Experience &nbsp;|&nbsp;
    10+ Years Academic Excellence
  </span>
</div>
""")
