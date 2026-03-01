"""
tab_mlr.py — Multiple Linear Regression (Plotly, interactive, Fama-French theory)
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, acc_t, txt_s, teal_t, p, two_col, three_col, four_col,
    table_html, metric_row, section_heading, stat_box, hero_row,
    alert_box, formula_block, S, FH, FB, FM, TXT, NO_SEL
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
    font=dict(family="Source Sans Pro", color="#e6f1ff", size=12),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    legend=dict(bgcolor="rgba(17,34,64,0.8)", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(t=50, b=40, l=60, r=30),
)

def run_ols_full(y, X):
    n, k = len(y), X.shape[1]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ b; resid = y - yhat
    sse = np.sum(resid**2); sst = np.sum((y - y.mean())**2)
    r2 = 1 - sse/sst; adj_r2 = 1 - (1-r2)*(n-1)/(n-k)
    mse = sse / (n-k)
    XtXinv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtXinv) * mse)
    t_stats = b / se; p_vals = 2*(1 - stats.t.cdf(np.abs(t_stats), n-k))
    f_stat = (r2/(k-1))/((1-r2)/(n-k)); f_pval = 1 - stats.f.cdf(f_stat, k-1, n-k)
    vif = [np.nan] + [1/(1 - np.linalg.lstsq(
        np.column_stack([np.ones(n), np.delete(X[:,1:], j, axis=1)]),
        X[:,j+1], rcond=None)[3]) if X.shape[1] > 2 else 1.0
        for j in range(k-1)]
    return dict(beta=b, yhat=yhat, resid=resid, r2=r2, adj_r2=adj_r2,
                se=se, t_stats=t_stats, p_vals=p_vals,
                f_stat=f_stat, f_pval=f_pval, n=n, k=k, mse=mse)


def tab_mlr():
    render_card("📊 Multiple Linear Regression — More than One Predictor",
        p(f'MLR extends SLR to {hl("k predictors")}. In finance, MLR underpins '
          f'{hl("Fama-French")} factor models, credit scoring, and multi-driver valuation.')
        + two_col(
            ib(f'<div style="font-family:{FH};color:#FFD700;-webkit-text-fill-color:#FFD700;'
               f'font-size:1rem;margin-bottom:8px">📐 The MLR Model</div>'
               + fml("Y = β₀ + β₁X₁ + β₂X₂ + … + βₖXₖ + ε\n\n"
                     "In matrix form:  Y = Xβ + ε\n"
                     "OLS solution:  β̂ = (X'X)⁻¹ X'Y\n"
                     "Fitted values: Ŷ = X(X'X)⁻¹X'Y = HY\n"
                     "where H = X(X'X)⁻¹X' is the Hat matrix"), "gold"),
            ib(f'<div style="font-family:{FH};color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
               f'font-size:1rem;margin-bottom:8px">📊 Fama-French 3-Factor Example</div>'
               + fml("Rᵢ − Rf = α + β₁(Rₘ−Rf) + β₂·SMB + β₃·HML + ε\n\n"
                     "β₁ = Market beta (systematic risk)\n"
                     "β₂ = SMB loading (size premium exposure)\n"
                     "β₃ = HML loading (value premium exposure)\n"
                     "α  = Abnormal return (Jensen's alpha)"), "blue")
        )
    )

    render_card("🔑 MLR Goodness of Fit & Inference",
        three_col(
            ib(fml("R² = SSR/SST = 1 − SSE/SST\n\nAlways increases when\nwe add more predictors!\n(Even irrelevant ones)"), "gold"),
            ib(fml("Adj. R² = 1 − (1−R²)(n−1)/(n−k)\n\nPenalises for extra predictors.\nUse for model comparison.\nPreferred over R² in MLR."), "blue"),
            ib(fml("F-test: H₀: β₁=β₂=…=βₖ=0\nF = (R²/k) / ((1−R²)/(n−k−1))\n~F(k, n−k−1) under H₀\n\nReject if p < 0.05"), "green")
        )
    )

    # ── Interactive MLR Demo: Fama-French ────────────────────────
    render_card("🎯 Interactive Fama-French Simulation",
        p(f'Simulate a {hl("Fama-French 3-Factor")} regression and observe how factor loadings '
          f'affect model fit. Adjust true betas and noise.')
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: b_mkt = st.slider("True β_MKT", 0.4, 2.0, 1.1, 0.05)
    with c2: b_smb = st.slider("True β_SMB", -0.5, 1.5, 0.4, 0.05)
    with c3: b_hml = st.slider("True β_HML", -0.5, 1.0, 0.2, 0.05)
    with c4: noise = st.slider("Idio. Noise %", 0.5, 3.5, 1.2, 0.1)

    np.random.seed(99)
    n = 250
    mkt = np.random.normal(0.0004, 0.012, n)
    smb = np.random.normal(0.0001, 0.008, n)
    hml = np.random.normal(0.0001, 0.007, n)
    eps = np.random.normal(0, noise/100/np.sqrt(252), n)
    y   = 0.0002 + b_mkt*mkt + b_smb*smb + b_hml*hml + eps

    X_ff = np.column_stack([np.ones(n), mkt, smb, hml])
    # Also run CAPM only for comparison
    X_capm = np.column_stack([np.ones(n), mkt])
    res_ff   = run_ols_full(y, X_ff)
    res_capm = run_ols_full(y, X_capm)

    hero_row([
        ("α (FF Alpha)", f"{res_ff['beta'][0]*100:.4f}%", f"p={res_ff['p_vals'][0]:.4f}", "", "#28a745"),
        ("β_MKT", f"{res_ff['beta'][1]:.4f}", f"True: {b_mkt:.2f}", "", "#FFD700"),
        ("β_SMB", f"{res_ff['beta'][2]:.4f}", f"True: {b_smb:.2f}", "", "#ff9f43"),
        ("β_HML", f"{res_ff['beta'][3]:.4f}", f"True: {b_hml:.2f}", "", "#a29bfe"),
        ("Adj. R² (FF)", f"{res_ff['adj_r2']:.4f}", f"vs CAPM: {res_capm['adj_r2']:.4f}", "", "#ADD8E6"),
    ])

    c_left, c_right = st.columns(2)
    with c_left:
        # Predicted vs Actual
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y*100, y=res_ff["yhat"]*100, mode="markers",
                                 marker=dict(color="#ADD8E6", size=4, opacity=0.5), name="Obs"))
        mn = min(y.min(), res_ff["yhat"].min())*100
        mx = max(y.max(), res_ff["yhat"].max())*100
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                 line=dict(color="#FFD700", width=2, dash="dash"), name="45°"))
        fig.update_layout(**PLOTLY_LAYOUT, title="Actual vs Fitted Returns",
                          xaxis_title="Actual (%)", yaxis_title="Fitted (%)", height=330)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        # Model comparison R² bar
        models = ["CAPM (1F)", "FF 3-Factor"]
        r2s = [res_capm["r2"], res_ff["r2"]]
        adj_r2s = [res_capm["adj_r2"], res_ff["adj_r2"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="R²", x=models, y=r2s, marker_color="#003366",
                              text=[f"{v:.4f}" for v in r2s], textposition="outside"))
        fig2.add_trace(go.Bar(name="Adj. R²", x=models, y=adj_r2s, marker_color="#FFD700",
                              text=[f"{v:.4f}" for v in adj_r2s], textposition="outside"))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Model Comparison: R²",
                           yaxis_title="R²", barmode="group", height=330)
        st.plotly_chart(fig2, use_container_width=True)

    # Coefficient table
    section_heading("📋 FF 3-Factor OLS Results")
    names = ["α (Intercept)", "β_MKT (Market)", "β_SMB (Size)", "β_HML (Value)"]
    rows = []
    for name, b, se, t, pv in zip(names, res_ff["beta"], res_ff["se"],
                                   res_ff["t_stats"], res_ff["p_vals"]):
        sig = bdg("***","green") if pv<0.01 else bdg("**","teal") if pv<0.05 else bdg("*","orange") if pv<0.1 else bdg("ns","red")
        rows.append([name, f"{b:.6f}", f"{se:.6f}", f"{t:.3f}", f"{pv:.4f}", sig])
    render_ib(table_html(["Factor","Estimate","Std Error","t-stat","p-value","Sig."], rows), "blue")
    render_ib(ib(f'F-statistic: {hl(str(round(res_ff["f_stat"],3)))} (p={res_ff["f_pval"]:.4f})  '
                 f'| Adj. R² = {hl(str(round(res_ff["adj_r2"],4)))}  '
                 f'| n = {res_ff["n"]} observations', "gold"), "gold")

    # ── Multicollinearity section ─────────────────────────────────
    render_card("⚠️ Multicollinearity — Detection & Remedies",
        p(f'In MLR, {hl("multicollinearity")} occurs when predictors are highly correlated, '
          f'making it hard to isolate individual effects.')
        + two_col(
            ib(fml("VIF (Variance Inflation Factor):\nVIF_j = 1 / (1 − R²_j)\n\n"
                   "R²_j = R² from regressing Xⱼ on all other Xᵢ\n\n"
                   "Rule of thumb:\nVIF < 5    → Acceptable\nVIF 5–10   → Moderate concern\nVIF > 10   → Serious multicollinearity"), "gold"),
            ib(fml("Detection methods:\n① Correlation matrix: |rᵢⱼ| > 0.8 → concern\n② VIF > 10\n③ High R² but low t-stats\n④ Sensitive β when Xᵢ added/removed\n\nRemedies:\n① Drop one correlated variable\n② Ridge regression (L2 penalty)\n③ Principal Component Analysis (PCA)\n④ Increase sample size"), "blue")
        )
    )

    # VIF simulation
    section_heading("🔢 VIF Simulation — Effect of Correlation on Stability")
    corr_val = st.slider("Correlation between X₁ and X₂", 0.0, 0.99, 0.50, 0.05)
    np.random.seed(5)
    n_mc = 200
    cov = [[1, corr_val],[corr_val, 1]]
    X12 = np.random.multivariate_normal([0,0], cov, n_mc)
    y_mc= 2 + 1.5*X12[:,0] + 0.8*X12[:,1] + np.random.normal(0, 1, n_mc)
    X_mc = np.column_stack([np.ones(n_mc), X12])
    res_mc = run_ols_full(y_mc, X_mc)
    vif1_r2 = run_ols_full(X12[:,0], np.column_stack([np.ones(n_mc), X12[:,1]]))["r2"]
    vif1 = 1 / (1 - vif1_r2)

    hero_row([
        ("Corr(X₁,X₂)", f"{corr_val:.2f}", "", "", "#FFD700"),
        ("VIF (X₁)", f"{vif1:.2f}", "Threshold: 10", "up" if vif1 > 5 else "", "#ff9f43" if vif1 > 5 else "#28a745"),
        ("SE(β₁)", f"{res_mc['se'][1]:.4f}", "Inflated by multicollinearity", "", "#ADD8E6"),
        ("SE(β₂)", f"{res_mc['se'][2]:.4f}", "", "", "#a29bfe"),
        ("Adj. R²", f"{res_mc['adj_r2']:.4f}", "Model still fits well", "", "#28a745"),
    ])

    if vif1 > 5:
        render_ib(alert_box(f"⚠️ VIF={vif1:.2f} > 5 — Moderate multicollinearity. SE of β̂₁ is inflated. Consider variable selection or Ridge regression.", "red"))
    else:
        render_ib(alert_box(f"✅ VIF={vif1:.2f} < 5 — Acceptable multicollinearity level.", "green"))

    # ── Partial regression ────────────────────────────────────────
    render_card("📐 Partial Regression Plots — Ceteris Paribus Effect",
        p(f'{hl("Partial regression plots")} (added variable plots) show the marginal relationship '
          f'between each predictor and Y after {hl("controlling for all other variables")}. '
          f'This isolates the unique contribution of each factor — critical in Fama-French analysis.')
        + fml("Partial Regression of Y on Xⱼ (holding other Xi constant):\n\n"
              "1. Regress Y on all X except Xⱼ → get residuals eᵧ\n"
              "2. Regress Xⱼ on all other X   → get residuals eⱼ\n"
              "3. Plot eᵧ vs eⱼ → slope = β̂ⱼ in full MLR\n\n"
              "The partial slope βⱼ represents the ceteris paribus effect of Xⱼ on Y")
        + p(f'In Fama-French: {hl("β_SMB")} = marginal effect of size factor '
            f'after controlling for market and value factors.')
    )
