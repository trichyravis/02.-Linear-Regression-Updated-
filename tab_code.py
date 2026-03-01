"""
tab_code.py — Python Code Reference (comprehensive, with CAPM & FF)
"""
import streamlit as st
from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, acc_t, txt_s, p, two_col, three_col,
    table_html, section_heading, S, FH, FB, FM, TXT, NO_SEL
)

CODE_BLOCKS = {
    "📐 SLR — CAPM Beta (from scratch)": """\
import numpy as np
import scipy.stats as stats

# ── 1. OLS from scratch ─────────────────────────────────────────
def ols_slr(x, y):
    n = len(x)
    x_bar, y_bar = x.mean(), y.mean()
    beta1 = np.sum((x - x_bar)*(y - y_bar)) / np.sum((x - x_bar)**2)
    beta0 = y_bar - beta1 * x_bar
    y_hat = beta0 + beta1 * x
    resid = y - y_hat
    sse   = np.sum(resid**2)
    sst   = np.sum((y - y_bar)**2)
    r2    = 1 - sse/sst
    mse   = sse / (n - 2)
    se_b1 = np.sqrt(mse / np.sum((x - x_bar)**2))
    se_b0 = np.sqrt(mse * (1/n + x_bar**2 / np.sum((x - x_bar)**2)))
    t_b1  = beta1 / se_b1
    p_b1  = 2*(1 - stats.t.cdf(abs(t_b1), df=n-2))
    dw    = np.sum(np.diff(resid)**2) / sse
    return dict(alpha=beta0, beta=beta1, r2=r2,
                se_alpha=se_b0, se_beta=se_b1,
                t_beta=t_b1, p_beta=p_b1, dw=dw,
                resid=resid, yhat=y_hat)

# ── 2. CAPM Beta using yfinance (live NSE data) ─────────────────
import yfinance as yf
import pandas as pd

def estimate_capm_beta(ticker, period="1y", rf_annual=6.5):
    stock = yf.download(ticker, period=period, auto_adjust=True)["Close"]
    nifty = yf.download("^NSEI", period=period, auto_adjust=True)["Close"]
    df    = pd.DataFrame({"s": stock, "m": nifty}).dropna()
    r_s   = np.log(df["s"] / df["s"].shift(1)).dropna()
    r_m   = np.log(df["m"] / df["m"].shift(1)).dropna()
    rf    = rf_annual / 252 / 100  # daily risk-free rate
    xs_s  = r_s - rf
    xs_m  = r_m - rf
    common = xs_s.index.intersection(xs_m.index)
    res   = ols_slr(xs_m.loc[common].values, xs_s.loc[common].values)
    return res

# ── 3. Example: Reliance Industries ─────────────────────────────
res = estimate_capm_beta("RELIANCE.NS", period="2y")
print(f"Jensen's Alpha: {res['alpha']*252*100:.3f}% p.a.")
print(f"Beta:           {res['beta']:.4f}")
print(f"R²:             {res['r2']:.4f}")
print(f"t(beta):        {res['t_beta']:.3f}  p={res['p_beta']:.4f}")
print(f"Durbin-Watson:  {res['dw']:.4f}")
""",
    "📊 MLR — Fama-French 3-Factor": """\
import numpy as np
import pandas as pd
import scipy.stats as stats

# ── OLS in matrix form ───────────────────────────────────────────
def ols_mlr(Y, X):
    \"\"\"X should include column of ones for intercept\"\"\"
    n, k  = X.shape
    b     = np.linalg.lstsq(X, Y, rcond=None)[0]
    yhat  = X @ b
    resid = Y - yhat
    sse   = np.sum(resid**2)
    sst   = np.sum((Y - Y.mean())**2)
    r2    = 1 - sse/sst
    adj_r2= 1 - (1-r2)*(n-1)/(n-k)
    mse   = sse / (n-k)
    XtXinv= np.linalg.inv(X.T @ X)
    se    = np.sqrt(np.diag(XtXinv) * mse)
    t_stats = b / se
    p_vals  = 2*(1 - stats.t.cdf(np.abs(t_stats), df=n-k))
    f_stat  = (r2/(k-1)) / ((1-r2)/(n-k))
    f_pval  = 1 - stats.f.cdf(f_stat, k-1, n-k)
    dw      = np.sum(np.diff(resid)**2) / sse
    return dict(beta=b, yhat=yhat, resid=resid, r2=r2,
                adj_r2=adj_r2, se=se, t_stats=t_stats,
                p_vals=p_vals, f_stat=f_stat, f_pval=f_pval,
                dw=dw, n=n, k=k)

# ── Fama-French 3-Factor (illustrative) ─────────────────────────
np.random.seed(42)
n = 252  # 1 year daily data
# Simulate factors
mkt = np.random.normal(0.0004, 0.012, n)   # Market excess return
smb = np.random.normal(0.0001, 0.008, n)   # Small Minus Big
hml = np.random.normal(0.0001, 0.007, n)   # High Minus Low

# True factor loadings
TRUE = dict(alpha=0.0002, b_mkt=1.15, b_smb=0.40, b_hml=0.25)
y = (TRUE["alpha"]
     + TRUE["b_mkt"]*mkt
     + TRUE["b_smb"]*smb
     + TRUE["b_hml"]*hml
     + np.random.normal(0, 0.008, n))

X = np.column_stack([np.ones(n), mkt, smb, hml])
res = ols_mlr(y, X)

factor_names = ["α (Intercept)", "β_MKT", "β_SMB", "β_HML"]
print("=" * 60)
print("FAMA-FRENCH 3-FACTOR REGRESSION")
print("=" * 60)
for name, b, se, t, p in zip(factor_names, res["beta"],
                               res["se"], res["t_stats"], res["p_vals"]):
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
    print(f"{name:20s}: {b:.6f}  SE={se:.6f}  t={t:.3f}  p={p:.4f}  {sig}")
print(f"\\nR²={res['r2']:.4f}  Adj.R²={res['adj_r2']:.4f}  "
      f"F={res['f_stat']:.2f} (p={res['f_pval']:.4f})")
""",
    "🔬 Diagnostics — All Tests": """\
import numpy as np
import scipy.stats as stats

def run_diagnostics(y, yhat, X, resid):
    n, k = len(y), X.shape[1]
    sse  = np.sum(resid**2)

    # 1. Jarque-Bera (Normality)
    jb_stat, jb_p = stats.jarque_bera(resid)

    # 2. Durbin-Watson (Autocorrelation)
    dw = np.sum(np.diff(resid)**2) / sse

    # 3. Breusch-Pagan (Heteroscedasticity)
    e2 = resid**2
    bp_b  = np.linalg.lstsq(X, e2, rcond=None)[0]
    e2hat = X @ bp_b
    bp_r2 = 1 - np.sum((e2 - e2hat)**2) / np.sum((e2 - e2.mean())**2)
    bp_stat = n * bp_r2
    bp_p    = 1 - stats.chi2.cdf(bp_stat, df=k-1)

    # 4. RESET Test (Functional Form)
    yhat2    = (yhat / yhat.std())**2
    X_reset  = np.column_stack([X, yhat2])
    b_reset  = np.linalg.lstsq(X_reset, y, rcond=None)[0]
    yhat_r   = X_reset @ b_reset
    resid_r  = y - yhat_r
    sse_r    = np.sum(resid_r**2)
    f_reset  = ((sse - sse_r)/1) / (sse_r/(n-k-1))
    p_reset  = 1 - stats.f.cdf(f_reset, 1, n-k-1)

    # 5. VIF for each predictor (excluding intercept)
    vifs = []
    for j in range(1, k):
        Xi = np.delete(X[:,1:], j-1, axis=1)
        Xi = np.column_stack([np.ones(n), Xi])
        b_vif = np.linalg.lstsq(Xi, X[:,j], rcond=None)[0]
        yhat_vif = Xi @ b_vif
        r2_j = 1 - np.sum((X[:,j]-yhat_vif)**2)/np.sum((X[:,j]-X[:,j].mean())**2)
        vifs.append(1 / (1 - r2_j) if r2_j < 1 else np.inf)

    print("=" * 55)
    print("OLS DIAGNOSTIC SUMMARY")
    print("=" * 55)
    jb_ok  = "PASS" if jb_p  > 0.05 else "FAIL"
    dw_ok  = "OK"   if 1.5 < dw < 2.5 else "CHECK"
    bp_ok  = "PASS" if bp_p  > 0.05 else "FAIL"
    rst_ok = "PASS" if p_reset > 0.05 else "FAIL"
    print(f"Jarque-Bera (Normality):    stat={jb_stat:.3f}  p={jb_p:.4f}  [{jb_ok}]")
    print(f"Durbin-Watson (Autocorr.): DW={dw:.4f}              [{dw_ok}]")
    print(f"Breusch-Pagan (Heterosc.):  stat={bp_stat:.3f}  p={bp_p:.4f}  [{bp_ok}]")
    print(f"RESET (Misspecification):   F={f_reset:.3f}    p={p_reset:.4f}  [{rst_ok}]")
    for j, vif in enumerate(vifs, 1):
        vif_ok = "OK" if vif < 5 else "MODERATE" if vif < 10 else "HIGH"
        print(f"VIF(X{j}):                   {vif:.3f}                [{vif_ok}]")

    return dict(jb=(jb_stat, jb_p), dw=dw, bp=(bp_stat, bp_p),
                reset=(f_reset, p_reset), vifs=vifs)
""",
    "📈 Rolling Beta & Confidence Bands": """\
import numpy as np
import plotly.graph_objects as go

def rolling_beta(r_stock, r_mkt, window=63):
    \"\"\"Compute rolling OLS beta with ±1 SE confidence bands\"\"\"
    n = len(r_stock)
    betas, alphas, se_betas = [], [], []
    for i in range(window, n+1):
        xs = r_stock[i-window:i]
        xm = r_mkt[i-window:i]
        x_bar = xm.mean()
        b1 = np.sum((xm-x_bar)*(xs-xs.mean())) / np.sum((xm-x_bar)**2)
        b0 = xs.mean() - b1*x_bar
        yhat = b0 + b1*xm
        resid = xs - yhat
        mse   = np.sum(resid**2) / (window-2)
        se_b1 = np.sqrt(mse / np.sum((xm-x_bar)**2))
        betas.append(b1); alphas.append(b0); se_betas.append(se_b1)
    return np.array(betas), np.array(alphas), np.array(se_betas)

# Example usage with simulated data
np.random.seed(42)
n = 500
r_mkt = np.random.normal(0, 0.012, n)
r_stk = 0.8 + 1.2*r_mkt + np.random.normal(0, 0.009, n)

betas, alphas, se_betas = rolling_beta(r_stk, r_mkt, window=63)
ci_upper = betas + 1.96*se_betas
ci_lower = betas - 1.96*se_betas

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(betas))), y=betas,
    line=dict(color="#FFD700"), name="Rolling Beta"
))
fig.add_trace(go.Scatter(
    x=list(range(len(betas))) + list(range(len(betas)-1,-1,-1)),
    y=list(ci_upper) + list(ci_lower[::-1]),
    fill="toself", fillcolor="rgba(255,215,0,0.15)",
    line=dict(color="rgba(255,0,0,0)"), name="95% CI"
))
fig.add_hline(y=1, line_dash="dash", line_color="#dc3545",
              annotation_text="β=1 (Market)")
fig.update_layout(title="Rolling Beta with 95% Confidence Bands",
                  xaxis_title="Day", yaxis_title="Beta")
fig.show()
""",
}

def tab_code():
    render_card("🐍 Python Implementation — OLS, CAPM & Fama-French",
        p(f'Production-quality Python code for all regression models. '
          f'Uses {hl("NumPy OLS from scratch")} and {hl("yfinance for live NSE data")}. '
          f'Compatible with pandas, scipy, and plotly.')
    )

    render_ib(table_html(
        ["Module", "Key Functions", "Finance Application", "Libraries"],
        [
            [bdg("SLR from Scratch","gold"),    acc_t("ols_slr()"),    txt_s("CAPM Beta, SCL"), txt_s("numpy, scipy")],
            [bdg("Live CAPM","blue"),            acc_t("estimate_capm_beta()"), txt_s("Nifty 50 stock beta"), txt_s("yfinance, pandas")],
            [bdg("MLR Matrix Form","green"),     acc_t("ols_mlr()"),    txt_s("Fama-French 3-Factor"), txt_s("numpy, scipy")],
            [bdg("Diagnostics","orange"),        acc_t("run_diagnostics()"), txt_s("JB, DW, BP, RESET, VIF"), txt_s("scipy, numpy")],
            [bdg("Rolling Beta","purple"),       acc_t("rolling_beta()"), txt_s("Time-varying systematic risk"), txt_s("numpy, plotly")],
        ]
    ), "blue")

    selected = st.selectbox("📖 Select Code Module", list(CODE_BLOCKS.keys()), index=0)
    code = CODE_BLOCKS[selected]
    st.code(code, language="python")

    render_card("📦 Requirements & Setup",
        fml("# Install all dependencies\npip install streamlit numpy pandas scipy plotly yfinance statsmodels\n\n"
            "# Run the app\nstreamlit run app.py\n\n"
            "# Quick test (CAPM beta for Infosys, 2 years)\npython -c \"\nimport yfinance as yf, numpy as np\n"
            "s = yf.download('INFY.NS', period='2y', auto_adjust=True)['Close']\n"
            "m = yf.download('^NSEI', period='2y', auto_adjust=True)['Close']\n"
            "rs = np.log(s/s.shift(1)).dropna(); rm = np.log(m/m.shift(1)).dropna()\n"
            "idx = rs.index.intersection(rm.index)\n"
            "b = np.cov(rs[idx],rm[idx])[0,1] / np.var(rm[idx])\n"
            "print(f'Infosys Beta: {b:.4f}')\n\"")
    )
