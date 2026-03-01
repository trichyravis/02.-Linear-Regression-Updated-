"""
tab_slr.py — Simple Linear Regression (Upgraded with Plotly + interactive demo)
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, acc_t, txt_s, teal_t, p, steps_html, two_col, three_col, four_col,
    table_html, metric_row, section_heading, stat_box, hero_row, alert_box,
    formula_block, S, FH, FB, FM, TXT, NO_SEL
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
    font=dict(family="Source Sans Pro", color="#e6f1ff", size=12),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    legend=dict(bgcolor="rgba(17,34,64,0.8)", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(t=50, b=40, l=60, r=30),
)

def tab_slr():
    # ── 1. Concept Card ──────────────────────────────────────────
    render_card("📈 Simple Linear Regression — One Predictor, One Outcome",
        p(f'SLR models the {lb_t("<strong>linear relationship</strong>")} between one independent '
          f'variable (X) and one dependent variable (Y). In finance, the most classic application '
          f'is estimating {hl("CAPM Beta")} — how a stock moves relative to the market.')
        + two_col(
            ib(f'<div style="font-family:{FH};color:#FFD700;-webkit-text-fill-color:#FFD700;'
               f'font-size:1.05rem;margin-bottom:8px">📐 The Regression Model</div>'
               + fml("Y = β₀ + β₁X + ε\n\nY  = Dependent variable (e.g. Stock Return)\n"
                     "X  = Independent variable (e.g. Market Return)\nβ₀ = Intercept (Alpha)\n"
                     "β₁ = Slope (Beta) — change in Y per unit X\nε  = Error term (residual)")
               + p(f'{hl("Fitted line:")} Ŷ = β̂₀ + β̂₁X minimises Σ(Yᵢ − Ŷᵢ)²'),
               "gold"),
            ib(f'<div style="font-family:{FH};color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
               f'font-size:1.05rem;margin-bottom:8px">💹 CAPM Beta Application</div>'
               + p(f'{lb_t("<strong>Security Characteristic Line (SCL):</strong>")}')
               + fml("Rᵢ − Rf = αᵢ + βᵢ(Rₘ − Rf) + εᵢ\n\nRᵢ = Stock excess return\nRₘ = Market excess return\n"
                      "α = Jensens Alpha (intercept)\nβ = Systematic Risk (slope)\nε = Idiosyncratic risk")
               + p(f'{bdg("β > 1","red")} Aggressive &nbsp; {bdg("β = 1","gold")} Neutral &nbsp; {bdg("β < 1","green")} Defensive'),
               "blue"),
        )
    )

    # ── 2. OLS Estimation ─────────────────────────────────────────
    render_card("🔧 OLS Estimation — Ordinary Least Squares",
        p(f'OLS finds β̂₀ and β̂₁ by {hl("minimising the sum of squared residuals (SSR)")}. '
          f'The solution is {hl("analytical")} — no iteration required (Gauss-Markov BLUE).')
        + two_col(
            fml("β̂₁ = Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)²\n"
                "   = Cov(X,Y) / Var(X)\n"
                "   = r × (Sʏ / Sˣ)\n\nβ̂₀ = Ȳ − β̂₁X̄"),
            fml("SST = Σ(Yᵢ−Ȳ)²    (Total variation)\n"
                "SSR = Σ(Ŷᵢ−Ȳ)²    (Explained)\n"
                "SSE = Σ(Yᵢ−Ŷᵢ)²   (Unexplained)\n\n"
                "R² = SSR/SST = 1 − SSE/SST\n"
                "SE(β̂₁) = √(MSE / Σ(Xᵢ−X̄)²)")
        )
        + three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">R² — Goodness of Fit</span><br>'
               + p("Proportion of Y's variance explained by X. R² ∈ [0, 1].")
               + p(f'{hl("0.80")} → X explains 80% of Y variation'), "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">t-Test on β̂₁</span><br>'
               + p(f'H₀: β₁ = 0 (no relationship)<br>t = β̂₁ / SE(β̂₁)<br>Reject if |t| > t_crit (df = n−2)'), "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">F-Test (ANOVA)</span><br>'
               + p(f'H₀: β₁ = 0<br>F = MSR/MSE ~ F(1,n−2)<br>In SLR: F = t² (equivalent)'), "green"),
        )
    )

    # ── 3. CLRM Assumptions ───────────────────────────────────────
    render_card("⚙ CLRM Assumptions — Gauss-Markov Theorem",
        p(f'For OLS to be {hl("BLUE")} (Best Linear Unbiased Estimator), the following must hold:')
        + table_html(
            ["#", "Assumption", "Mathematical Statement", "Violation → Problem"],
            [
                ["1", bdg("Linearity","blue"),      txt_s("E(ε|X) = 0; true model is linear in parameters"), txt_s("Biased, inconsistent β̂")],
                ["2", bdg("No Multicollinearity","gold"), txt_s("No perfect linear combo of regressors (MLR)"), txt_s("Inflated SE, unstable β̂")],
                ["3", bdg("Homoscedasticity","green"), txt_s("Var(εᵢ) = σ² ∀i (constant variance)"), txt_s("Inefficient OLS, incorrect SE")],
                ["4", bdg("No Autocorrelation","orange"), txt_s("Cov(εᵢ,εⱼ) = 0 for i≠j"), txt_s("DW test fails; biased SE in time series")],
                ["5", bdg("Normality","purple"),    txt_s("ε ~ N(0,σ²) for finite sample inference"), txt_s("Invalid t/F tests (small samples)")],
                ["6", bdg("Exogeneity","teal"),     txt_s("Cov(Xᵢ,εᵢ) = 0; X is non-stochastic"), txt_s("Endogeneity bias; need IV")],
            ])
    )

    # ── 4. Interactive OLS Demo ───────────────────────────────────
    render_card("🎯 Interactive CAPM Demo — Adjust Beta & Alpha",
        p(f'Simulate a stock with user-defined {hl("True Beta")} and {hl("Jensen\'s Alpha")}. '
          f'Observe OLS fit quality and statistical properties.')
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: true_beta  = st.slider("True Beta (β)", 0.3, 2.5, 1.2, 0.1)
    with c2: true_alpha = st.slider("True Alpha (α) %", -2.0, 2.0, 0.3, 0.1)
    with c3: noise      = st.slider("Idiosyncratic Noise (σε) %", 0.5, 4.0, 1.5, 0.1)
    with c4: n_pts      = st.slider("No. of Observations", 30, 500, 120, 10)

    np.random.seed(42)
    x_mkt = np.random.normal(0, 1.2, n_pts)
    eps    = np.random.normal(0, noise/100, n_pts)
    y_stk  = true_alpha/100 + true_beta * x_mkt/100 + eps

    X_d = np.column_stack([np.ones(n_pts), x_mkt/100])
    b   = np.linalg.lstsq(X_d, y_stk, rcond=None)[0]
    yhat = X_d @ b
    resid= y_stk - yhat
    sse = np.sum(resid**2); sst = np.sum((y_stk - y_stk.mean())**2)
    r2 = 1 - sse/sst
    mse = sse / (n_pts - 2)
    XtXinv = np.linalg.inv(X_d.T @ X_d)
    se = np.sqrt(np.diag(XtXinv) * mse)
    t_stats = b / se
    p_vals = 2*(1 - stats.t.cdf(np.abs(t_stats), n_pts - 2))

    hero_row([
        ("Est. Alpha (α)", f"{b[0]*100:.4f}%", f"True: {true_alpha:.1f}%", "", "#28a745"),
        ("Est. Beta (β)", f"{b[1]:.4f}", f"True: {true_beta:.2f}", "", "#FFD700"),
        ("R²", f"{r2:.4f}", f"{r2*100:.1f}% explained", "", "#ADD8E6"),
        ("t(α)", f"{t_stats[0]:.3f}", f"p={p_vals[0]:.4f}", "", "#a29bfe"),
        ("t(β)", f"{t_stats[1]:.3f}", f"p={p_vals[1]:.4f}", "", "#ff9f43"),
    ])

    c_l, c_r = st.columns([3, 2])
    with c_l:
        x_line = np.linspace(x_mkt.min(), x_mkt.max(), 100) / 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_mkt, y=y_stk*100, mode="markers",
            marker=dict(color="#ADD8E6", size=5, opacity=0.6), name="Simulated Returns"
        ))
        fig.add_trace(go.Scatter(
            x=x_line*100, y=(b[0]+b[1]*x_line)*100, mode="lines",
            line=dict(color="#FFD700", width=2.5),
            name=f"OLS: α={b[0]*100:.3f}% β={b[1]:.4f}"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="OLS Regression Fit",
                          xaxis_title="Market Return (%)", yaxis_title="Stock Return (%)", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c_r:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=resid*100, nbinsx=30,
                                    marker_color="#003366", opacity=0.85, name="Residuals"))
        x_n = np.linspace(resid.min(), resid.max(), 100)
        y_n = stats.norm.pdf(x_n, 0, resid.std()) * n_pts * (resid.max()-resid.min())/30
        fig2.add_trace(go.Scatter(x=x_n*100, y=y_n, mode="lines",
                                   line=dict(color="#FFD700", width=2), name="Normal"))
        fig2.update_layout(**PLOTLY_LAYOUT, title="Residual Distribution",
                           xaxis_title="Residual (%)", yaxis_title="Freq", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # ── 5. Hypothesis Testing ─────────────────────────────────────
    render_card("🧪 Hypothesis Testing on Regression Coefficients",
        two_col(
            ib(fml("H₀: β₁ = 0  (No linear relationship)\nH₁: β₁ ≠ 0  (Significant linear relationship)\n\n"
                   "Test statistic:\nt = β̂₁ / SE(β̂₁)  ~  t(n−2) under H₀\n\n"
                   "Decision:\n|t| > t_crit(α/2, n−2) → Reject H₀\np < α → Reject H₀"), "gold"),
            ib(fml("95% Confidence Interval for β₁:\nβ̂₁ ± t_crit(0.025, n−2) × SE(β̂₁)\n\n"
                   "If CI excludes 0 → Significant at 5%\n\n"
                   "For CAPM Alpha (H₀: α = 0):\nt = α̂ / SE(α̂)  — tests if stock\n"
                   "earns abnormal return beyond CAPM"), "blue")
        )
        + p(f'{hl("Coefficient of Correlation (r):")} β̂₁ = r × (Sʏ/Sˣ) → '
            f'same sign as r; {hl("r² = R²")} in SLR')
    )

    # ── 6. Key metrics cheat sheet ────────────────────────────────
    render_card("📋 SLR Quick Reference — Finance Context",
        table_html(
            ["Metric", "Formula", "Finance Interpretation", "Good Value"],
            [
                [hl("R²"),         acc_t("SSR/SST = 1 − SSE/SST"), txt_s("% of stock return explained by market"), bdg("CAPM: 0.3–0.7","green")],
                [hl("Adj. R²"),    acc_t("1 − (1−R²)(n−1)/(n−k)"), txt_s("R² penalised for extra predictors"),    bdg("Higher better","teal")],
                [hl("Beta (β̂₁)"),  acc_t("Cov(X,Y)/Var(X)"),        txt_s("Systematic risk; market sensitivity"),  bdg("1.0 = market","gold")],
                [hl("Alpha (β̂₀)"), acc_t("Ȳ − β̂₁X̄"),               txt_s("Jensen's alpha — abnormal return"),     bdg("> 0 = outperform","green")],
                [hl("RMSE"),       acc_t("√(SSE/n)"),                txt_s("Avg prediction error in return units"),  bdg("Lower better","blue")],
                [hl("F-stat"),     acc_t("MSR/MSE"),                 txt_s("Overall model significance"),            bdg("p < 0.05","orange")],
                [hl("DW stat"),    acc_t("Σ(eᵢ−eᵢ₋₁)²/SSE"),       txt_s("Autocorrelation of residuals"),         bdg("~2.0 ideal","purple")],
            ]
        )
    )
