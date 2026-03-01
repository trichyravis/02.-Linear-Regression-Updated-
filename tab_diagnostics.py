"""
tab_diagnostics.py — OLS Regression Diagnostics (Plotly)
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
    alert_box, S, FH, FB, FM, TXT, NO_SEL
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
    font=dict(family="Source Sans Pro", color="#e6f1ff", size=12),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    legend=dict(bgcolor="rgba(17,34,64,0.8)", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(t=50, b=40, l=60, r=30),
)

def tab_diagnostics():
    render_card("🔬 OLS Regression Diagnostics — Testing CLRM Assumptions",
        p(f'Diagnostics validate whether the {hl("CLRM assumptions")} hold. '
          f'Violations require remedies before relying on OLS estimates.')
        + table_html(
            ["Test", "Assumption Tested", "Statistic", "Decision Rule", "Finance Impact"],
            [
                [bdg("Jarque-Bera","blue"),       txt_s("Normality of ε"),         acc_t("JB = n(S²/6 + K²/24)"), txt_s("p > 0.05 → Normal"),      txt_s("Invalid t/F in small samples")],
                [bdg("Durbin-Watson","gold"),      txt_s("No Autocorrelation"),     acc_t("DW = Σ(eᵢ−eᵢ₋₁)²/SSE"), txt_s("~2.0 → No autocorr."),   txt_s("Time-series returns often autocorrelated")],
                [bdg("Breusch-Pagan","green"),     txt_s("Homoscedasticity"),       acc_t("nR² from aux. regression"), txt_s("p > 0.05 → Homoscedastic"), txt_s("Volatility clustering in returns")],
                [bdg("White's Test","orange"),     txt_s("Homoscedasticity (robust)"),acc_t("nR² (non-linear terms)"),txt_s("p > 0.05 → OK"),         txt_s("Prefer robust SE in finance")],
                [bdg("Ramsey RESET","purple"),     txt_s("Functional Form"),        acc_t("F-test on Ŷ², Ŷ³"),    txt_s("p > 0.05 → Correct spec."), txt_s("Non-linear factor exposure")],
                [bdg("VIF","teal"),                txt_s("No Multicollinearity"),   acc_t("1/(1−R²_j)"),           txt_s("VIF < 10 acceptable"),    txt_s("Correlated FF factors inflate SE")],
            ])
    )

    # ── Interactive Diagnostics ────────────────────────────────────
    section_heading("🎛️ Interactive Diagnostic Simulator")
    st.html(f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};'
            f'font-size:.87rem;margin-bottom:12px;{NO_SEL}">'
            f'Choose a DGP (Data Generating Process) to simulate violations and run all diagnostic tests.</div>')

    c1, c2 = st.columns(2)
    with c1:
        dgp = st.selectbox("Data Generating Process",
                           ["✅ Valid OLS (all assumptions met)",
                            "📈 Heteroscedasticity (ARCH effects)",
                            "🔁 Autocorrelation (AR(1) residuals)",
                            "⚡ Non-Normality (fat tails / Student-t)",
                            "📐 Misspecification (quadratic ignored)"])
    with c2:
        n_obs = st.slider("Sample Size", 50, 500, 200, 25)

    np.random.seed(42)
    x = np.sort(np.random.uniform(-3, 3, n_obs))

    if "Valid OLS" in dgp:
        eps = np.random.normal(0, 1, n_obs)
        y   = 2 + 1.5*x + eps
        dgp_label = "Valid OLS"
    elif "Heteroscedasticity" in dgp:
        sigma = 0.3 + 0.5*np.abs(x)
        eps = np.array([np.random.normal(0, s) for s in sigma])
        y   = 2 + 1.5*x + eps
        dgp_label = "Heteroscedastic"
    elif "Autocorrelation" in dgp:
        eps = np.zeros(n_obs)
        eps[0] = np.random.normal()
        for i in range(1, n_obs):
            eps[i] = 0.7*eps[i-1] + np.random.normal(0, 0.7)
        y = 2 + 1.5*x + eps
        dgp_label = "Autocorrelated"
    elif "Non-Normality" in dgp:
        eps = stats.t.rvs(df=3, size=n_obs) * 0.8
        y   = 2 + 1.5*x + eps
        dgp_label = "Fat-Tailed"
    else:
        eps = np.random.normal(0, 1, n_obs)
        y   = 2 + 1.5*x + 0.5*x**2 + eps  # quadratic true model
        dgp_label = "Misspecified"

    X_m = np.column_stack([np.ones(n_obs), x])
    b   = np.linalg.lstsq(X_m, y, rcond=None)[0]
    yhat= X_m @ b; resid = y - yhat
    sse = np.sum(resid**2); sst = np.sum((y - y.mean())**2)
    r2  = 1 - sse/sst
    n, k = n_obs, 2

    # Compute tests
    jb_stat, jb_p = stats.jarque_bera(resid)
    dw = np.sum(np.diff(resid)**2) / sse

    # Breusch-Pagan: regress e² on X
    e2 = resid**2
    bp_b = np.linalg.lstsq(X_m, e2, rcond=None)[0]
    e2hat = X_m @ bp_b
    bp_r2 = 1 - np.sum((e2 - e2hat)**2) / np.sum((e2 - e2.mean())**2)
    bp_stat = n * bp_r2
    bp_p = 1 - stats.chi2.cdf(bp_stat, df=k-1)

    # RESET: add yhat² as predictor
    yhat2 = yhat**2 / yhat.std()
    X_reset = np.column_stack([X_m, yhat2])
    b_reset = np.linalg.lstsq(X_reset, y, rcond=None)[0]
    yhat_r = X_reset @ b_reset; resid_r = y - yhat_r
    sse_r = np.sum(resid_r**2)
    f_reset = ((sse - sse_r)/1) / (sse_r/(n-k-1))
    p_reset = 1 - stats.f.cdf(f_reset, 1, n-k-1)

    # Display results
    def ok_fail(p, threshold=0.05, invert=False):
        if invert:
            return (bdg("✅ PASS","green"), "#28a745") if p < threshold else (bdg("❌ FAIL","red"), "#dc3545")
        return (bdg("✅ PASS","green"), "#28a745") if p > threshold else (bdg("❌ FAIL","red"), "#dc3545")

    jb_badge, jb_col  = ok_fail(jb_p)
    dw_badge = bdg("✅ OK","green") if 1.5 < dw < 2.5 else bdg("⚠️ Check","orange") if 1.0 < dw < 1.5 or 2.5 < dw < 3.0 else bdg("❌ FAIL","red")
    bp_badge, bp_col  = ok_fail(bp_p)
    rst_badge, rst_col= ok_fail(p_reset)

    hero_row([
        ("Jarque-Bera", f"p={jb_p:.4f}", f"Stat={jb_stat:.2f}", "", jb_col),
        ("Durbin-Watson", f"{dw:.4f}", "Target: 1.5–2.5", "", "#FFD700"),
        ("Breusch-Pagan", f"p={bp_p:.4f}", f"Stat={bp_stat:.2f}", "", bp_col),
        ("RESET Test", f"p={p_reset:.4f}", f"F={f_reset:.2f}", "", rst_col),
    ])

    # 4-panel diagnostic plot
    section_heading(f"📈 Diagnostic Plots — {dgp_label} DGP")
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Residuals vs Fitted", "Q-Q Plot (Normality)",
                                       "√|e| vs Fitted (Homoscedasticity)", "ACF of Residuals (Autocorrelation)"])

    # Residuals vs Fitted
    fig.add_trace(go.Scatter(x=yhat, y=resid, mode="markers",
                             marker=dict(color="#ADD8E6", size=4, opacity=0.6), name="Residuals"),
                  row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#FFD700", row=1, col=1)

    # Q-Q
    qq = stats.probplot(resid)
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers",
                             marker=dict(color="#ADD8E6", size=4), name="Q-Q"), row=1, col=2)
    lo, hi = qq[0][0].min(), qq[0][0].max()
    fig.add_trace(go.Scatter(x=[lo,hi], y=[qq[1][1]+qq[1][0]*lo, qq[1][1]+qq[1][0]*hi],
                             mode="lines", line=dict(color="#FFD700", width=2)), row=1, col=2)

    # Scale-Location
    fig.add_trace(go.Scatter(x=yhat, y=np.sqrt(np.abs(resid)), mode="markers",
                             marker=dict(color="#ff9f43", size=4, opacity=0.6)), row=2, col=1)
    fig.add_hline(y=np.sqrt(np.abs(resid)).mean(), line_dash="dash", line_color="#FFD700", row=2, col=1)

    # ACF
    max_lag = min(30, n_obs//5)
    acf_vals = [1.0]
    for lag in range(1, max_lag+1):
        c = np.corrcoef(resid[:-lag], resid[lag:])[0,1]
        acf_vals.append(c)
    ci_95 = 1.96 / np.sqrt(n_obs)
    fig.add_trace(go.Bar(x=list(range(max_lag+1)), y=acf_vals,
                         marker_color=["#dc3545" if abs(v)>ci_95 else "#003366" for v in acf_vals]),
                  row=2, col=2)
    fig.add_hline(y=ci_95,  line_dash="dash", line_color="#FFD700", row=2, col=2)
    fig.add_hline(y=-ci_95, line_dash="dash", line_color="#FFD700", row=2, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=520, showlegend=False)
    fig.update_xaxes(gridcolor="#1e3a5f"); fig.update_yaxes(gridcolor="#1e3a5f")
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────
    render_ib(table_html(
        ["Diagnostic Test","Statistic","p-value","Result","Remedy if Failed"],
        [
            ["Jarque-Bera (Normality)", f"{jb_stat:.3f}", f"{jb_p:.4f}", jb_badge,
             txt_s("Use robust SE; increase n; check outliers")],
            ["Durbin-Watson (Autocorr.)", f"{dw:.4f}", "N/A", dw_badge,
             txt_s("Use Newey-West SE; add lagged variable; GLS")],
            ["Breusch-Pagan (Heterosced.)", f"{bp_stat:.3f}", f"{bp_p:.4f}", bp_badge,
             txt_s("Use White robust SE; WLS; log transform Y")],
            ["RESET (Misspecification)", f"{f_reset:.3f}", f"{p_reset:.4f}", rst_badge,
             txt_s("Add polynomial terms; log transforms; interaction")],
        ]
    ), "blue")

    # ── Remedies Section ──────────────────────────────────────────
    render_card("🛠️ Remedies for OLS Violations — Finance Applications",
        three_col(
            ib(f'{bdg("Heteroscedasticity","orange")}<br><br>'
               + fml("1. White's Robust SE\n   → std errors robust to heterosc.\n\n"
                     "2. WLS (Weighted Least Squares)\n   → weight obs by 1/σᵢ²\n\n"
                     "3. Log transform Y or X\n   → often stabilises variance\n\n"
                     "Finance: Returns vol clustering\n→ use GARCH-corrected errors"), "orange"),
            ib(f'{bdg("Autocorrelation","purple")}<br><br>'
               + fml("1. Newey-West SE\n   → HAC robust standard errors\n\n"
                     "2. Add lagged dependent variable\n   → Y = f(Xₜ, Yₜ₋₁)\n\n"
                     "3. GLS / ARIMA errors\n   → model error structure\n\n"
                     "Finance: Momentum in returns\n→ DW test; LM autocorr test"), "purple"),
            ib(f'{bdg("Non-Normality","teal")}<br><br>'
               + fml("1. Robust regression\n   → M-estimators, LAD\n\n"
                     "2. Bootstrap inference\n   → non-parametric CI\n\n"
                     "3. Box-Cox transform Y\n   → Y^λ to achieve normality\n\n"
                     "Finance: Fat tails in returns\n→ t-distribution; Student-t"), "teal"),
        )
    )

    # ── Leverage, Influence & Cook's D ───────────────────────────
    render_card("🎯 Influential Observations — Leverage & Cook's Distance",
        p(f'In financial data (e.g. CAPM residuals), {hl("outliers")} from market crashes, '
          f'corporate events, or data errors can unduly influence regression estimates.')
        + two_col(
            fml("Leverage (hat values): hᵢᵢ = diag(H)\nH = X(X'X)⁻¹X' (Hat matrix)\nhᵢᵢ ∈ [1/n, 1]\n\n"
                "High leverage: hᵢᵢ > 2k/n\n(obs far from X̄ in predictor space)\nNot always harmful alone"),
            fml("Cook's Distance: Dᵢ = eᵢ²·hᵢᵢ / (k·MSE·(1−hᵢᵢ)²)\n\nCombines leverage + residual\nDᵢ > 4/n → potentially influential\nDᵢ > 1   → highly influential\n\nRemedy: check, winsorise, or remove")
        )
    )
