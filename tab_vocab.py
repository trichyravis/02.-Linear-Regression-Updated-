"""
tab_vocab.py — Education Hub (Glossary, Cheat Sheets, Formula Reference)
"""
import streamlit as st
from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, acc_t, txt_s, teal_t, p, two_col, three_col, four_col,
    table_html, section_heading, S, FH, FB, FM, TXT, NO_SEL
)

def tab_vocab():
    render_card("📚 Education Hub — Concepts, Formulas & Finance Glossary",
        p(f'Complete reference for {hl("Linear Regression")} theory, statistical tests, '
          f'and {hl("Indian capital market applications")} (CAPM, Fama-French, NSE context).')
    )

    sub_tabs = st.tabs(["📐 Formula Sheet", "📖 Glossary", "🏦 Finance Context", "🎓 Cheat Sheet"])

    # ── Formula Sheet ─────────────────────────────────────────────
    with sub_tabs[0]:
        section_heading("📐 Core OLS Formulas")
        render_ib(two_col(
            fml("SLR:\nβ̂₁ = Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)² = Cov(X,Y)/Var(X)\nβ̂₀ = Ȳ − β̂₁X̄\n\n"
                "Confidence interval for β̂₁:\nβ̂₁ ± t_α/2(n−2) × SE(β̂₁)\n\n"
                "SE(β̂₁) = √(MSE / Σ(Xᵢ−X̄)²)\nMSE = SSE/(n−2)"),
            fml("MLR (matrix form):\nβ̂ = (X'X)⁻¹X'Y\nŶ = Xβ̂ = HY\nH = X(X'X)⁻¹X'\n\n"
                "Var(β̂) = σ²(X'X)⁻¹\nSE(β̂ⱼ) = √[σ̂²(X'X)⁻¹ⱼⱼ]\n\n"
                "σ̂² = MSE = SSE/(n−k)")
        ), "blue")

        section_heading("📊 Model Fit Metrics")
        render_ib(table_html(
            ["Metric", "Formula", "Interpretation"],
            [
                [hl("R²"),       acc_t("1 − SSE/SST"),                  txt_s("Proportion of Y variation explained. SLR only: r²=R²")],
                [hl("Adj. R²"),  acc_t("1−(1−R²)(n−1)/(n−k)"),         txt_s("Penalises for adding predictors. Compare across models")],
                [hl("AIC"),      acc_t("n·ln(SSE/n) + 2k"),             txt_s("Akaike Info Criterion — lower is better")],
                [hl("BIC"),      acc_t("n·ln(SSE/n) + k·ln(n)"),        txt_s("Bayesian IC — penalises complexity more than AIC")],
                [hl("RMSE"),     acc_t("√(SSE/n)"),                      txt_s("Root mean squared error — same units as Y")],
                [hl("F-stat"),   acc_t("(R²/k) / ((1−R²)/(n−k−1))"),   txt_s("Overall model significance ~ F(k, n−k−1)")],
                [hl("t-stat"),   acc_t("β̂ⱼ / SE(β̂ⱼ)"),                txt_s("Individual coefficient significance ~ t(n−k)")],
                [hl("VIF"),      acc_t("1/(1−R²ⱼ)"),                    txt_s("Variance Inflation Factor: multicollinearity measure")],
                [hl("DW"),       acc_t("Σ(eᵢ−eᵢ₋₁)²/SSE"),             txt_s("Durbin-Watson: ~2 → no autocorrelation")],
            ]
        ), "blue")

        section_heading("📉 CAPM & Factor Model Formulas")
        render_ib(four_col(
            fml("CAPM:\nE(Rᵢ) = Rf + βᵢ[E(Rₘ)−Rf]\n\nβᵢ = Cov(Rᵢ,Rₘ)/Var(Rₘ)\n   = ρᵢₘ·(σᵢ/σₘ)"),
            fml("SCL (OLS):\nRᵢ−Rf = α + β(Rₘ−Rf) + ε\n\nα = Jensen's Alpha\nβ = Systematic Risk\nε = Idio. risk"),
            fml("Fama-French:\nRᵢ−Rf = α + β₁MKT\n       + β₂SMB\n       + β₃HML + ε\n\nAdj-R² > CAPM R²"),
            fml("Risk Decomposition:\nσ²ᵢ = β²σ²ₘ + σ²ε\n\nSystematic = β²σ²ₘ/σ²ᵢ × 100%\nIdiosyncratic = (1−R²) × 100%")
        ), "gold")

    # ── Glossary ──────────────────────────────────────────────────
    with sub_tabs[1]:
        section_heading("📖 Statistical Glossary")
        terms = [
            ("OLS (Ordinary Least Squares)", "β̂ = (X'X)⁻¹X'Y",
             "Method that minimises sum of squared residuals. BLUE under CLRM assumptions.",
             "Estimating CAPM beta by regressing stock returns on market returns"),
            ("R² (Coefficient of Determination)", "R² = SSR/SST ∈ [0,1]",
             "Fraction of variance in Y explained by X(s). In SLR: R² = r².",
             "R²=0.35 for Nifty stock → market explains 35% of daily return variation"),
            ("Adjusted R²", "1 − (1−R²)(n−1)/(n−k)",
             "R² corrected for model complexity. Increases only if new variable adds more than noise.",
             "Fama-French adj.R² > CAPM R² confirms SMB/HML add explanatory power"),
            ("Standard Error (SE)", "SE(β̂₁) = √(MSE/Σ(Xᵢ−X̄)²)",
             "Estimated standard deviation of β̂. Smaller SE → more precise estimate.",
             "SE(beta) large → uncertain beta estimate (short data or noisy returns)"),
            ("t-statistic", "t = β̂/SE(β̂) ~ t(n−k)",
             "Tests H₀: β = 0. |t| > 2 roughly significant at 5% for large n.",
             "t(beta) = 8.5, p < 0.001 → beta highly significant for ICICI Bank"),
            ("p-value", "P(|T| > |t| | H₀ true)",
             "Probability of observing this |t| if H₀: β=0 is true. p < 0.05 → reject H₀.",
             "p(alpha) = 0.32 → Jensen's alpha not statistically different from zero"),
            ("Confidence Interval", "β̂ ± t_α/2 × SE(β̂)",
             "Range that contains true β with (1−α)×100% probability across repeated samples.",
             "95% CI for beta: [0.82, 1.15] → β significantly above zero"),
            ("Heteroscedasticity", "Var(εᵢ) ≠ σ²",
             "Non-constant error variance. Common in financial returns (ARCH/GARCH effects).",
             "Stock return variance is higher during market stress (COVID, GFC)"),
            ("Autocorrelation", "Cov(εᵢ,εᵢ₋ₖ) ≠ 0",
             "Correlated errors across time. Violates CLRM; biases SE in time series.",
             "Momentum in stock returns; DW < 1.5 → positive autocorrelation"),
            ("VIF", "1/(1−R²ⱼ)",
             "Variance Inflation Factor measures multicollinearity. VIF>10 → serious problem.",
             "SMB & HML factors in FF may correlate → VIF check needed"),
        ]
        for term, formula, definition, example in terms:
            with st.expander(f"📌 {term}"):
                render_ib(
                    f'<b style="color:#FFD700;-webkit-text-fill-color:#FFD700">{term}</b><br>'
                    + fml(formula) + "<br>"
                    + p(definition) + "<br>"
                    + ib(f'<b style="color:#28a745;-webkit-text-fill-color:#28a745">Finance Example:</b> {example}', "green"),
                    "blue"
                )

    # ── Finance Context ───────────────────────────────────────────
    with sub_tabs[2]:
        section_heading("🏦 Regression in Indian Capital Markets")
        render_card("📡 CAPM Beta for Nifty 50 Stocks",
            p(f'Beta (β) is the fundamental {hl("systematic risk measure")} used by Indian fund managers, '
              f'analysts, and risk desks. NSE computes beta daily for Nifty 50 constituents.')
            + table_html(
                ["Stock", "Sector", "Typical Beta Range", "Interpretation"],
                [
                    [bdg("HDFC Bank","blue"),      txt_s("BFSI"),          acc_t("0.90 – 1.10"), txt_s("Near-market neutral; stable")],
                    [bdg("Tata Motors","red"),      txt_s("Auto"),          acc_t("1.30 – 1.80"), txt_s("Highly aggressive; cyclical")],
                    [bdg("Infosys","gold"),         txt_s("IT Services"),   acc_t("0.80 – 1.10"), txt_s("USD revenue; sector-specific risk")],
                    [bdg("ITC","green"),            txt_s("FMCG"),          acc_t("0.50 – 0.80"), txt_s("Defensive; low market sensitivity")],
                    [bdg("Adani Ports","orange"),   txt_s("Infrastructure"), acc_t("1.10 – 1.40"), txt_s("Above market; policy-driven")],
                    [bdg("Sun Pharma","purple"),    txt_s("Pharma"),        acc_t("0.60 – 0.90"), txt_s("Defensive; import-linked risks")],
                    [bdg("Bajaj Finance","teal"),   txt_s("NBFC"),          acc_t("1.20 – 1.60"), txt_s("Growth stock; rate-sensitive")],
                ]
            )
        )
        render_card("📊 Fama-French in Indian Context",
            p(f'The Fama-French 3-Factor model has been tested on {hl("BSE 500 and Nifty 500")} stocks. '
              f'Evidence suggests SMB and HML premiums exist in India, though weaker than US markets.')
            + two_col(
                ib(f'{bdg("Size Premium (SMB)","orange")}<br>'
                   + p("Small-cap stocks on NSE/BSE historically outperform large-caps "
                       "(Nifty SmallCap 250 vs Nifty 50). Loading β_SMB > 0 → small-cap tilt."), "orange"),
                ib(f'{bdg("Value Premium (HML)","gold")}<br>'
                   + p("High book-to-market (P/B < 1) stocks outperform growth stocks in India, "
                       "especially post-crisis periods. β_HML > 0 → value tilt."), "gold")
            )
        )

    # ── Cheat Sheet ───────────────────────────────────────────────
    with sub_tabs[3]:
        section_heading("🎓 1-Page Cheat Sheet — Linear Regression")
        render_card("Model Specification",
            three_col(
                ib(fml("SLR: Y = β₀ + β₁X + ε\nMLR: Y = Xβ + ε\nOLS: β̂=(X'X)⁻¹X'Y\nFitted: Ŷ=Xβ̂"), "gold"),
                ib(fml("R² = 1−SSE/SST\nAdj.R²=1−(1−R²)(n−1)/(n−k)\nF = (R²/k)/((1−R²)/(n−k−1))\nt = β̂ⱼ/SE(β̂ⱼ)"), "blue"),
                ib(fml("CAPM: Rᵢ−Rf = α+β(Rₘ−Rf)+ε\nFF3F: +β₂SMB+β₃HML\nVIF=1/(1−R²ⱼ)\nDW=Σ(eᵢ−eᵢ₋₁)²/SSE"), "green"),
            )
        )

        section_heading("Assumptions & Violations")
        render_ib(table_html(
            ["#", "Assumption", "Test", "Remedy"],
            [
                ["1", bdg("Linearity","blue"),       txt_s("RESET, residual plots"),    txt_s("Polynomial; log transform")],
                ["2", bdg("Homoscedasticity","gold"), txt_s("Breusch-Pagan; White"),    txt_s("Robust SE; WLS; GARCH")],
                ["3", bdg("No Autocorr.","orange"),  txt_s("Durbin-Watson; LM test"),   txt_s("Newey-West SE; ARIMA; GLS")],
                ["4", bdg("Normality","purple"),      txt_s("Jarque-Bera; Q-Q plot"),   txt_s("Bootstrap; robust regression")],
                ["5", bdg("No Multicollin.","teal"),  txt_s("VIF; condition number"),   txt_s("Drop variable; Ridge; PCA")],
                ["6", bdg("Exogeneity","red"),        txt_s("Hausman; Wu"),             txt_s("Instrumental Variables (IV)")],
            ]
        ), "blue")

        section_heading("Decision Framework")
        render_ib(
            p(f'1. {hl("Specify model:")} theory-driven (CAPM/FF) or data-driven (stepwise)')
            + p(f'2. {hl("Estimate OLS:")} β̂ = (X\'X)⁻¹X\'Y; check sign/magnitude make economic sense')
            + p(f'3. {hl("Statistical inference:")} t-tests (individual), F-test (overall), CI widths')
            + p(f'4. {hl("Model selection:")} Adj.R², AIC/BIC; add variables only if theory supports')
            + p(f'5. {hl("Diagnostics:")} 4 plots (residuals vs fitted, Q-Q, scale-location, ACF)')
            + p(f'6. {hl("If violation found:")} apply remedy (robust SE, WLS, add lag, IV)')
            + p(f'7. {hl("Interpret:")} β̂ = change in Y per unit X, {hl("ceteris paribus")}')
            + p(f'8. {hl("Finance application:")} beta = systematic risk; alpha = abnormal return'),
            "gold"
        )
