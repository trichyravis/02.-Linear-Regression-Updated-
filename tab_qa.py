"""
tab_qa.py — Interactive Q&A Self-Assessment (upgraded with scoring + progress)
"""
import streamlit as st
import numpy as np
from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, acc_t, txt_s, p, two_col, three_col,
    table_html, section_heading, metric_row, hero_row,
    S, FH, FB, FM, TXT, NO_SEL
)

MCQ_BANK = [
    # ── SLR ───────────────────────────────────────────────────────
    {"id": "SLR-1", "topic": "SLR", "level": "Foundation",
     "question": "In SLR, the OLS estimator β̂₁ (slope) is:",
     "options": ["Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Yᵢ−Ȳ)²", "Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)²",
                 "Cov(X,Y) / Var(Y)", "Σ(Yᵢ−Ȳ)² / Σ(Xᵢ−X̄)²"],
     "answer": 1,
     "explanation": "β̂₁ = Σ(Xᵢ−X̄)(Yᵢ−Ȳ)/Σ(Xᵢ−X̄)² = Cov(X,Y)/Var(X). The denominator is Var(X), NOT Var(Y)."},
    {"id": "SLR-2", "topic": "SLR", "level": "Foundation",
     "question": "If R² = 0.82 in a CAPM regression of Infosys on Nifty, it means:",
     "options": ["Nifty explains 18% of Infosys return variation",
                 "Nifty explains 82% of Infosys return variation",
                 "Infosys beta is 0.82",
                 "Infosys outperforms Nifty by 82%"],
     "answer": 1,
     "explanation": "R² = 0.82 means the market (Nifty) explains 82% of the variance in Infosys returns. The remaining 18% is idiosyncratic (company-specific) risk."},
    {"id": "SLR-3", "topic": "SLR", "level": "Applied",
     "question": "A stock has β = 1.5 and Jensen's α = 0.02% (daily). If Nifty rises 1%, the stock is expected to rise by approximately:",
     "options": ["1.5%", "1.52%", "0.52%", "2.0%"],
     "answer": 1,
     "explanation": "Expected return = α + β × Rₘ = 0.02% + 1.5 × 1% = 1.52%. Alpha is small but adds to the expected return."},
    {"id": "SLR-4", "topic": "SLR", "level": "Advanced",
     "question": "If Durbin-Watson statistic = 0.85 in a CAPM regression, this indicates:",
     "options": ["No autocorrelation", "Negative autocorrelation",
                 "Positive autocorrelation in residuals", "Model is misspecified"],
     "answer": 2,
     "explanation": "DW ≈ 2 means no autocorrelation. DW < 1.5 indicates positive autocorrelation. DW > 2.5 indicates negative autocorrelation. DW=0.85 is strong positive autocorrelation (momentum in returns)."},
    # ── MLR ───────────────────────────────────────────────────────
    {"id": "MLR-1", "topic": "MLR", "level": "Foundation",
     "question": "In MLR, the Adjusted R² differs from R² because it:",
     "options": ["Is always lower than R²",
                 "Penalises adding predictors that don't improve fit",
                 "Equals R² minus the number of variables",
                 "Can be negative"],
     "answer": 1,
     "explanation": "Adj. R² = 1 − (1−R²)(n−1)/(n−k). It penalises each added predictor, so it only increases when the new variable improves fit more than expected by chance."},
    {"id": "MLR-2", "topic": "MLR", "level": "Applied",
     "question": "In Fama-French 3-Factor, β_SMB > 0 implies the stock has:",
     "options": ["Large-cap tilt (negative size premium)", "Value stock characteristics",
                 "Small-cap tilt (positive size premium exposure)", "High market beta"],
     "answer": 2,
     "explanation": "SMB = Small Minus Big. β_SMB > 0 means the stock loads positively on the small-cap factor, indicating small-cap tilt or higher exposure to the size premium."},
    {"id": "MLR-3", "topic": "MLR", "level": "Foundation",
     "question": "VIF = 15 for a predictor in MLR indicates:",
     "options": ["Low multicollinearity (acceptable)", "Moderate multicollinearity",
                 "Serious multicollinearity — SE of β̂ inflated by √15 ≈ 3.9×", "The variable is not significant"],
     "answer": 2,
     "explanation": "VIF = 15 >> 10 indicates serious multicollinearity. SE is inflated by √VIF = √15 ≈ 3.9× compared to an orthogonal model. Remedies: drop correlated variable, Ridge regression, or PCA."},
    {"id": "MLR-4", "topic": "MLR", "level": "Advanced",
     "question": "The F-test in MLR tests:",
     "options": ["Each individual β̂ⱼ is significant",
                 "H₀: β₁ = β₂ = … = βₖ = 0 (all slopes are zero)",
                 "The model has the correct functional form",
                 "There is no autocorrelation"],
     "answer": 1,
     "explanation": "F-test: H₀: β₁=β₂=…=βₖ=0 (joint significance). F = (R²/k)/((1−R²)/(n−k−1)) ~ F(k, n−k−1). Significant F-stat → at least one βⱼ ≠ 0."},
    # ── Diagnostics ───────────────────────────────────────────────
    {"id": "DX-1", "topic": "Diagnostics", "level": "Foundation",
     "question": "Breusch-Pagan test detects violation of:",
     "options": ["Normality", "Autocorrelation", "Heteroscedasticity", "Multicollinearity"],
     "answer": 2,
     "explanation": "Breusch-Pagan: H₀: Var(εᵢ) = σ² (homoscedasticity). Reject H₀ (p < 0.05) → heteroscedasticity. In finance: stock return variance is higher during volatile markets (ARCH effects)."},
    {"id": "DX-2", "topic": "Diagnostics", "level": "Applied",
     "question": "A Q-Q plot where residuals fan out into a curve at the tails indicates:",
     "options": ["Homoscedasticity", "No autocorrelation",
                 "Fat tails / non-normality (leptokurtosis)", "Correct model specification"],
     "answer": 2,
     "explanation": "In Q-Q plots, curvature at the tails indicates non-normality. For returns data, fat tails (leptokurtosis) are common — residuals deviate from the normal line at extremes. Use robust SE or t-distribution."},
    {"id": "DX-3", "topic": "Diagnostics", "level": "Advanced",
     "question": "White's robust standard errors are preferred over OLS SE when:",
     "options": ["All CLRM assumptions hold",
                 "Heteroscedasticity is present (unknown form)",
                 "Autocorrelation is present",
                 "The sample size is very small"],
     "answer": 1,
     "explanation": "White's robust SE (HC3) corrects for heteroscedasticity of unknown form without requiring correct specification of the error structure. Essential in finance where volatility clustering is common."},
    # ── Finance ───────────────────────────────────────────────────
    {"id": "FIN-1", "topic": "Finance", "level": "Foundation",
     "question": "Jensen's Alpha in CAPM regression is:",
     "options": ["The slope coefficient on the market return",
                 "The intercept term representing abnormal return above CAPM",
                 "The standard error of beta",
                 "R² of the regression"],
     "answer": 1,
     "explanation": "Jensen's Alpha = intercept (β₀) in the SCL regression: Rᵢ−Rf = α + β(Rₘ−Rf) + ε. α > 0 → stock earns positive risk-adjusted return above what CAPM predicts. Testing H₀: α=0 checks market efficiency."},
    {"id": "FIN-2", "topic": "Finance", "level": "Applied",
     "question": "A fund manager finds β = 0.65 for HDFC Bank vs Nifty 50. This means:",
     "options": ["HDFC Bank is 65% correlated with Nifty",
                 "HDFC Bank rises 65% when Nifty rises 1%",
                 "HDFC Bank is defensive: a 1% Nifty rise → ~0.65% expected HDFC rise",
                 "HDFC Bank has 65% systematic risk"],
     "answer": 2,
     "explanation": "Beta = 0.65 < 1 → defensive stock. Expected return = α + 0.65 × R_Nifty. A 1% Nifty rise → only 0.65% HDFC rise, making it less volatile than the market."},
    {"id": "FIN-3", "topic": "Finance", "level": "Advanced",
     "question": "Treynor Ratio is defined as:",
     "options": ["(Rp − Rf) / σp", "(Rp − Rf) / βp", "αp / βp", "Rp / σp"],
     "answer": 1,
     "explanation": "Treynor Ratio = (Rp − Rf) / βp. Uses beta (systematic risk) in denominator vs. Sharpe Ratio which uses total volatility. Better for well-diversified portfolios where idiosyncratic risk is eliminated."},
    {"id": "FIN-4", "topic": "Finance", "level": "Advanced",
     "question": "In the Fama-French 3-Factor model, HML captures:",
     "options": ["High beta Minus Low beta stocks",
                 "High Momentum Minus Low momentum",
                 "High book-to-market (value) Minus Low book-to-market (growth) stocks",
                 "High Market cap Minus Low market cap"],
     "answer": 2,
     "explanation": "HML = High book-to-market (value stocks) Minus Low book-to-market (growth stocks). β_HML > 0 → value stock tilt. β_HML < 0 → growth stock tilt. First documented by Fama & French (1992, 1993)."},
]

TOPICS = ["All", "SLR", "MLR", "Diagnostics", "Finance"]
LEVELS = ["All", "Foundation", "Applied", "Advanced"]

def tab_qa():
    render_card("🎓 Q&A Self-Assessment — Linear Regression in Finance",
        p(f'Test your understanding across {hl("SLR, MLR, Diagnostics, and Finance Applications")}. '
          f'Questions span {bdg("Foundation","green")} {bdg("Applied","gold")} {bdg("Advanced","red")} levels.')
    )

    # ── Filters & Session state ───────────────────────────────────
    if "qa_score" not in st.session_state:
        st.session_state.qa_score = 0
        st.session_state.qa_answered = {}
        st.session_state.qa_idx = 0

    c1, c2, c3 = st.columns(3)
    with c1: topic_filter = st.selectbox("📚 Topic", TOPICS)
    with c2: level_filter = st.selectbox("🎯 Level", LEVELS)
    with c3:
        if st.button("🔄 Reset Quiz"):
            st.session_state.qa_score = 0
            st.session_state.qa_answered = {}
            st.session_state.qa_idx = 0
            st.rerun()

    # Filter questions
    qs = [q for q in MCQ_BANK
          if (topic_filter == "All" or q["topic"] == topic_filter)
          and (level_filter == "All" or q["level"] == level_filter)]

    if not qs:
        render_ib("No questions match the selected filters.", "orange")
        return

    answered = st.session_state.qa_answered
    score    = sum(1 for v in answered.values() if v["correct"])
    total_answered = len(answered)
    accuracy = score/total_answered*100 if total_answered > 0 else 0.0

    hero_row([
        ("Questions Available", str(len(qs)), f"{topic_filter} | {level_filter}", "", "#ADD8E6"),
        ("Answered", str(total_answered), f"out of {len(qs)}", "", "#FFD700"),
        ("Correct", str(score), "", "up" if score > 0 else "", "#28a745"),
        ("Accuracy", f"{accuracy:.0f}%",
         "Excellent" if accuracy >= 80 else "Good" if accuracy >= 60 else "Keep trying", "",
         "#28a745" if accuracy >= 80 else "#ff9f43" if accuracy >= 60 else "#dc3545"),
    ])

    # Progress bar
    if total_answered > 0:
        pct = total_answered / len(qs) * 100
        bar_color = "#28a745" if accuracy >= 80 else "#FFD700" if accuracy >= 60 else "#dc3545"
        st.html(f'<div style="background:#1e3a5f;border-radius:6px;height:8px;margin:10px 0;{NO_SEL}">'
                f'<div style="background:{bar_color};border-radius:6px;height:8px;width:{pct:.1f}%"></div></div>'
                f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};'
                f'font-size:.82rem;text-align:right;{NO_SEL}">Progress: {pct:.0f}%</div>')

    # Navigation
    idx = min(st.session_state.qa_idx, len(qs)-1)
    c_prev, c_ctr, c_next = st.columns([1, 6, 1])
    with c_prev:
        if st.button("◀") and idx > 0:
            st.session_state.qa_idx -= 1; st.rerun()
    with c_ctr:
        st.html(f'<div style="text-align:center;color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
                f'font-family:{FB};font-size:.9rem;{NO_SEL}">'
                f'Question {idx+1} of {len(qs)}</div>')
    with c_next:
        if st.button("▶") and idx < len(qs)-1:
            st.session_state.qa_idx += 1; st.rerun()

    q = qs[idx]
    lvl_colors = {"Foundation": "green", "Applied": "gold", "Advanced": "red"}
    lvl_badge  = bdg(q["level"], lvl_colors.get(q["level"], "blue"))
    top_badge  = bdg(q["topic"], "blue")

    render_card(f"Q{idx+1}. {top_badge} {lvl_badge}",
        p(f'<b style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-size:1.05rem">'
          f'{q["question"]}</b>')
    )

    already = q["id"] in answered
    if already:
        user_ans = answered[q["id"]]["choice"]
        correct  = answered[q["id"]]["correct"]
        for i, opt in enumerate(q["options"]):
            if i == q["answer"]:
                icon = "✅"
                variant = "green"
            elif i == user_ans and not correct:
                icon = "❌"
                variant = "red"
            else:
                icon = "⬜"
                variant = "blue"
            render_ib(f'{icon} {opt}', variant)
        if correct:
            render_ib(f'🎉 {hl("Correct!")} {q["explanation"]}', "green")
        else:
            render_ib(f'❌ {rt2("Incorrect.")} {q["explanation"]}', "red")
    else:
        c_opts = st.columns(2)
        for i, opt in enumerate(q["options"]):
            with c_opts[i % 2]:
                if st.button(f"{chr(65+i)}. {opt}", key=f"opt_{q['id']}_{i}"):
                    is_correct = (i == q["answer"])
                    answered[q["id"]] = {"choice": i, "correct": is_correct}
                    st.session_state.qa_answered = answered
                    st.rerun()

    # ── Final summary ─────────────────────────────────────────────
    if total_answered == len(qs) and len(qs) > 0:
        grade = "A+" if accuracy >= 90 else "A" if accuracy >= 80 else "B" if accuracy >= 70 else "C" if accuracy >= 60 else "F"
        grade_color = "#28a745" if accuracy >= 80 else "#FFD700" if accuracy >= 70 else "#ff9f43" if accuracy >= 60 else "#dc3545"
        render_ib(
            f'<div style="text-align:center;{NO_SEL}">'
            f'<div style="font-family:{FH};font-size:1.8rem;color:{grade_color};'
            f'-webkit-text-fill-color:{grade_color};font-weight:700">'
            f'Quiz Complete! Grade: {grade}</div>'
            f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};margin-top:8px">'
            f'{score}/{total_answered} correct ({accuracy:.0f}%)</div>'
            f'</div>', "gold"
        )
