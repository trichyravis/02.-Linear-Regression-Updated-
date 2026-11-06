"""
tab_finance.py — Finance Lab (Live NSE Data)
CAPM Beta estimation, Fama-French 3-Factor, Bond Yield, P/E Valuation
Using yfinance for live Nifty 50 stock data
"""
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, acc_t, txt_s, teal_t, p, two_col, three_col, four_col,
    table_html, hero_row, metric_card, metric_row, section_heading,
    alert_box, formula_block, stat_box, S, FH, FB, FM, TXT, NO_SEL
)

# ── Nifty 50 universe (aligned with VaR app) ─────────────────────
NIFTY_STOCKS = {
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS", "HUL": "HINDUNILVR.NS",
    "ITC": "ITC.NS", "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "Kotak Mahindra": "KOTAKBANK.NS",
    "L&T": "LT.NS", "Axis Bank": "AXISBANK.NS",
    "Asian Paints": "ASIANPAINT.NS", "Wipro": "WIPRO.NS",
    "Maruti Suzuki": "MARUTI.NS", "Nestle India": "NESTLEIND.NS",
    "HCL Tech": "HCLTECH.NS", "Titan": "TITAN.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "ONGC": "ONGC.NS",
    "NTPC": "NTPC.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Tech Mahindra": "TECHM.NS", "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS", "Dr. Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS", "Eicher Motors": "EICHERMOT.NS",
    "Tata Steel": "TATASTEEL.NS", "M&M": "M&M.NS",
    "Hindalco": "HINDALCO.NS", "UltraTech Cement": "ULTRACEMCO.NS",
    "JSW Steel": "JSWSTEEL.NS", "Tata Consumer": "TATACONSUM.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS", "Coal India": "COALINDIA.NS",
    "BPCL": "BPCL.NS", "Apollo Hospitals": "APOLLOHOSP.NS",
}
MARKET = "^NSEI"  # Nifty 50 index

PERIODS = {"3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y",
           "2 Years": "2y", "3 Years": "3y", "5 Years": "5y", "Custom dates": "custom"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
    font=dict(family="Source Sans Pro", color="#e6f1ff", size=12),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    legend=dict(bgcolor="rgba(17,34,64,0.8)", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(t=50, b=40, l=60, r=30),
)

# ── Data helpers ──────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def _close_series(ticker, period):
    """Return one numeric, dated series despite Yahoo's MultiIndex columns."""
    options = dict(interval="1d", auto_adjust=True, progress=False,
                   threads=False, timeout=20)
    if ":" in period:
        start_text, end_text = period.split(":", 1)
        raw = yf.download(ticker, start=start_text,
                          end=(pd.Timestamp(end_text)+pd.Timedelta(days=1)).date(), **options)
    elif period == "3y":
        end = pd.Timestamp.now(tz="Asia/Kolkata").date()
        start = pd.Timestamp(end) - pd.DateOffset(years=3)
        raw = yf.download(ticker, start=start.date(), end=end, **options)
    else:
        raw = yf.download(ticker, period=period, **options)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo returned no daily prices for {ticker}. Check the symbol or retry later; Yahoo may rate-limit requests.")
    if isinstance(raw.columns, pd.MultiIndex):
        levels = [i for i in range(raw.columns.nlevels) if "Close" in raw.columns.get_level_values(i)]
        if not levels:
            raise ValueError(f"No Close field returned for {ticker}.")
        close = raw.xs("Close", axis=1, level=levels[0])
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                close = close[ticker]
            elif close.shape[1] == 1:
                close = close.iloc[:, 0]
            else:
                raise ValueError(f"Ambiguous Close columns for {ticker}.")
    else:
        close = raw["Close"]
    close = pd.to_numeric(close, errors="coerce")
    dates = pd.DatetimeIndex(pd.to_datetime(close.index))
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Kolkata").tz_localize(None)
    close.index = dates.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index().dropna()
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    close = close[(close.index.date < today) & (close > 0)]
    if close.empty:
        raise ValueError(f"No completed daily prices available for {ticker}.")
    return close.rename(ticker)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(ticker, market, period, rf_annual=6.5):
    stock = _close_series(ticker, period)
    benchmark = _close_series(market, period)
    df = pd.concat([stock.rename("stock"), benchmark.rename("market")], axis=1, join="inner").dropna()
    if len(df) < 31:
        raise ValueError("Fewer than 31 common completed closing dates. Select a longer period.")
    df["r_stock"] = df["stock"].pct_change(fill_method=None)
    df["r_market"] = df["market"].pct_change(fill_method=None)
    df["rf"] = (1 + float(rf_annual)/100)**(1/252) - 1
    df["xs_stock"] = df["r_stock"] - df["rf"]
    df["xs_market"] = df["r_market"] - df["rf"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df["xs_market"].var() <= 0:
        raise ValueError("Market returns have no variation; cannot estimate regression.")
    df.attrs["fetched_at"] = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d %b %Y, %H:%M IST")
    return df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ff_proxies(period):
    prices = pd.concat([_close_series(t, period).rename(n) for n, t in
        [("large", "NIFTYBEES.NS"), ("small", "SETFNN50.NS"),
         ("value", "BFSI.NS"), ("growth", "INFY.NS")]], axis=1, join="inner").dropna()
    returns = prices.pct_change(fill_method=None).dropna()
    return pd.DataFrame({"smb": returns["small"]-returns["large"],
                         "hml": returns["value"]-returns["growth"]})

def run_ols(y, X, covariance="Classical", maxlags=5):
    """Run OLS and return full stats dict"""
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    n, k = len(y), X.shape[1]
    if n <= k or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("OLS requires finite data and positive residual degrees of freedom.")
    if np.linalg.matrix_rank(X) < k or np.var(y) <= 0:
        raise ValueError("OLS design is singular or outcome has no variation.")
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ b
    resid = y - yhat
    sse = np.sum(resid**2)
    sst = np.sum((y - y.mean())**2)
    r2  = 1 - sse/sst
    adj_r2 = 1 - (1-r2)*(n-1)/(n-k)
    mse = sse / (n-k)
    XtXinv = np.linalg.inv(X.T @ X)
    cov = XtXinv * mse
    if covariance == "HC3":
        leverage = np.sum((X @ XtXinv) * X, axis=1)
        scaled = resid / np.maximum(1-leverage, 1e-10)
        cov = XtXinv @ (X.T @ ((scaled**2)[:, None] * X)) @ XtXinv
    elif covariance == "HAC":
        score = X * resid[:, None]
        meat = score.T @ score
        lag_count = min(int(maxlags), n-1)
        for lag in range(1, lag_count+1):
            cross = score[lag:].T @ score[:-lag]
            meat += (1-lag/(lag_count+1)) * (cross+cross.T)
        cov = (n/(n-k)) * XtXinv @ meat @ XtXinv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_stats = b / se
    p_vals  = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
    f_stat  = (r2/(k-1)) / ((1-r2)/(n-k))
    f_pval  = 1 - stats.f.cdf(f_stat, k-1, n-k)
    dw = np.sum(np.diff(resid)**2) / sse  # Durbin-Watson
    return dict(beta=b, yhat=yhat, resid=resid, r2=r2, adj_r2=adj_r2,
                se=se, t_stats=t_stats, p_vals=p_vals,
                f_stat=f_stat, f_pval=f_pval, dw=dw, n=n, k=k)


# ══════════════════════════════════════════════════════════════════
# MAIN TAB
# ══════════════════════════════════════════════════════════════════
def tab_finance():

    # ── Sidebar controls ────────────────────────────────────────────
    with st.sidebar:
        st.html(f'<div style="font-family:{FH};color:#FFD700;-webkit-text-fill-color:#FFD700;'
                f'font-size:1.1rem;font-weight:700;padding:8px 0;border-bottom:1px solid #1e3a5f;'
                f'margin-bottom:12px;{NO_SEL}">📊 Finance Lab Controls</div>')
        sel_name = st.selectbox("📡 Search and select NSE stock", list(NIFTY_STOCKS.keys()), index=0, help="Click and type to search. Use the optional ticker field for another Yahoo NSE symbol.")
        sel_ticker = NIFTY_STOCKS[sel_name]
        period_label = st.selectbox("📅 Data Period", list(PERIODS.keys()), index=2)
        period = PERIODS[period_label]
        if period == "custom":
            latest = (pd.Timestamp.now(tz="Asia/Kolkata")-pd.Timedelta(days=1)).date()
            start_date = st.date_input("Start date", value=(pd.Timestamp(latest)-pd.DateOffset(years=1)).date(), min_value=pd.Timestamp("1995-01-01").date(), max_value=latest)
            end_date = st.date_input("End date (inclusive)", value=latest, min_value=pd.Timestamp("1995-01-01").date(), max_value=latest)
            if start_date >= end_date:
                st.warning("Choose a start date earlier than the end date.")
                return
            period = f"{start_date.isoformat()}:{end_date.isoformat()}"
        model_type = st.radio("📐 Model", ["CAPM (SLR)", "Exploratory proxy-factor model (MLR)"], index=0)
        custom_symbol = st.text_input("Optional Yahoo NSE ticker", placeholder="e.g. RELIANCE.NS").strip().upper()
        if custom_symbol:
            if not __import__("re").fullmatch(r"[A-Z0-9&._-]{1,32}\.NS", custom_symbol):
                st.error("Enter a Yahoo NSE symbol ending in .NS.")
                return
            sel_ticker, sel_name = custom_symbol, custom_symbol
        st.caption("Convenience list: verify current Nifty constituent membership separately.")
        rf_annual = st.number_input("Assumed annual risk-free rate (%)", min_value=0.0, max_value=25.0, value=6.5, step=0.1)
        st.caption("User assumption, not a downloaded current RBI/T-bill rate. Converted to a daily effective rate using 252 trading days.")
        st.markdown("**Market data source:** [Yahoo Finance — Nifty 50](https://finance.yahoo.com/quote/%5ENSEI/history/)")
        if st.button("Refresh Yahoo data", key="refresh_yahoo"):
            _close_series.clear()
            fetch_data.clear()
            fetch_ff_proxies.clear()
        covariance = st.selectbox("Coefficient uncertainty", ["HAC", "HC3", "Classical"], help="HAC: weak serial dependence and heteroscedasticity. HC3: heteroscedasticity. Classical: constant-variance uncorrelated errors.")
        hac_lags = st.slider("HAC lag count", 1, 20, 5) if covariance == "HAC" else 5
        conf = st.slider("Confidence Level (%)", 90, 99, 95) / 100

    st.html(f'<div style="font-family:{FH};font-size:.95rem;color:#8892b0;'
            f'-webkit-text-fill-color:#8892b0;margin-bottom:16px;{NO_SEL}">'
            f'🏦 Latest available completed daily prices via Yahoo Finance · Quotes may be delayed · Cache: 15 minutes</div>')

    if not YFINANCE_OK:
        render_ib(alert_box("⚠️ yfinance not installed. Run: pip install yfinance", "red"))
        return

    from urllib.parse import quote
    st.markdown(f"**Data source:** [Yahoo Finance — {sel_ticker}](https://finance.yahoo.com/quote/{quote(sel_ticker, safe='')}/history/) · [Nifty 50 (^NSEI)](https://finance.yahoo.com/quote/%5ENSEI/history/)")
    st.info("Choose a stock and period in the sidebar. This lab downloads actual Yahoo daily history; today's unfinished candle is excluded. It is not an exchange real-time feed.")
    # ── Fetch data ───────────────────────────────────────────────────
    with st.spinner(f"📡 Fetching {sel_name} & Nifty 50 data..."):
        try:
            df = fetch_data(sel_ticker, MARKET, period, rf_annual)
        except Exception as exc:
            st.error(f"Yahoo Finance download failed: {exc}")
            st.info("Click Refresh Yahoo data, try a longer period or a different .NS ticker. No synthetic stock or index data is substituted.")
            return

    if df is None or len(df) < 30:
        render_ib(alert_box("⚠️ Insufficient data. Try a longer period or different stock.", "red"))
        return

    st.success(f"Yahoo data loaded: {sel_ticker} vs {MARKET} | {len(df)} common daily returns | {df.index.min():%d %b %Y}–{df.index.max():%d %b %Y}")
    st.caption(f"Fetched {df.attrs.get('fetched_at', 'unknown')} | Adjusted stock Close vs Nifty price index; educational CAPM approximation, not a consistent total-return benchmark.")
    with st.expander("View downloaded Yahoo prices and calculated returns"):
        st.dataframe(df.tail(20), use_container_width=True)
    st.download_button("Download Yahoo regression data (CSV)", df.to_csv(index_label="Date"), file_name=f"{sel_ticker}_nifty_regression.csv", mime="text/csv")
    last_date = df.index.max()
    age_days = (pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize() - last_date).days
    if age_days > 5:
        st.warning(f"Latest common completed price is {age_days} calendar days old. Check holidays, symbol history and Yahoo availability before interpreting results.")
    st.caption(f"Coefficient covariance: {covariance}. Robust t-tests use an approximate t reference with n−k degrees of freedom. The displayed model F statistic remains the conventional OLS F test; robust covariance does not repair endogeneity.")
    with st.expander("Compare price performance — rebased to 100"):
        st.line_chart(df[["stock", "market"]].div(df[["stock", "market"]].iloc[0]).mul(100))
    ret_s = df["xs_stock"].values
    ret_m = df["xs_market"].values

    # ── CAPM (SLR) ───────────────────────────────────────────────────
    if model_type == "CAPM (SLR)":
        X = np.column_stack([np.ones(len(ret_m)), ret_m])
        res = run_ols(ret_s, X, covariance, hac_lags)
        alpha, beta = res["beta"]
        r2, adj_r2 = res["r2"], res["adj_r2"]
        dw = res["dw"]
        n = res["n"]
        te, tb = res["t_stats"]
        pe, pb = res["p_vals"]
        annual_alpha = alpha * 252
        annual_vol = ret_s.std() * np.sqrt(252)
        sys_risk = beta**2 * ret_m.var() / ret_s.var() * 100
        idio_risk = (1 - r2) * 100

        split = int(n * 0.8)
        if split >= 20 and n - split >= 10:
            train_X, test_X = X[:split], X[split:]
            train_beta = np.linalg.lstsq(train_X, ret_s[:split], rcond=None)[0]
            test_pred = test_X @ train_beta
            rmse = np.sqrt(np.mean((ret_s[split:]-test_pred)**2))*100
            benchmark = np.sqrt(np.mean((ret_s[split:]-ret_s[:split].mean())**2))*100
            with st.expander("Chronological holdout — conditional evaluation"):
                st.write(f"Last {n-split} observations: regression RMSE {rmse:.4f} percentage points; training-mean benchmark {benchmark:.4f} percentage points.")
                st.caption("Uses the realised same-day Nifty return. This evaluates a conditional relationship, not an advance trading forecast. Coefficients are fitted only on the earlier 80% of observations.")
        # Header
        render_card(f"📡 CAPM Beta — {sel_name} vs Nifty 50 ({period_label})",
            p(f'Security Characteristic Line: {hl("Rᵢ − Rf = α + β(Rₘ − Rf) + ε")}'
              f'  |  Data points: {hl(str(n))} trading days')
            + fml(f"Estimated SCL: R_excess = {alpha*100:.4f}% + {beta:.4f} × R_market\n"
                  f"Jensen's Alpha (α) = {alpha*252*100:.2f}% per annum\n"
                  f"Beta (β)           = {beta:.4f} — "
                  + ("Aggressive (β > 1)" if beta > 1 else "Defensive (β < 1)" if beta < 1 else "Unit market sensitivity (β = 1)")
                  + f"\nR² = {r2:.4f} → Systematic risk explains {r2*100:.1f}% of return variation")
        )

        # Hero metrics
        b_col = "#dc3545" if beta > 1.2 else "#28a745" if beta < 0.8 else "#FFD700"
        a_col = "#28a745" if annual_alpha > 0 else "#dc3545"
        hero_row([
            ("Jensen's Alpha (p.a.)", f"{annual_alpha*100:+.2f}%", "Excess return above CAPM", "up" if annual_alpha > 0 else "down", a_col),
            ("Beta (β)", f"{beta:.4f}", "Aggressive >1 | Defensive <1", "", b_col),
            ("R² (Fit)", f"{r2:.4f}", f"{r2*100:.1f}% systematic risk", "", "#FFD700"),
            ("Adj. R²", f"{adj_r2:.4f}", f"n={n} observations", "", "#ADD8E6"),
            ("Durbin-Watson", f"{dw:.3f}", "~2 = no autocorrelation", "", "#a29bfe"),
        ])

        # Coefficient table
        section_heading("📋 OLS Coefficient Summary")
        ci_lo = alpha - stats.t.ppf((1+conf)/2, n-2) * res["se"][0]
        ci_hi = alpha + stats.t.ppf((1+conf)/2, n-2) * res["se"][0]
        bci_lo= beta  - stats.t.ppf((1+conf)/2, n-2) * res["se"][1]
        bci_hi= beta  + stats.t.ppf((1+conf)/2, n-2) * res["se"][1]
        sig_a = bdg("Significant","green") if pe < 0.05 else bdg("Not Sig.","orange")
        sig_b = bdg("Significant","green") if pb < 0.05 else bdg("Not Sig.","orange")
        render_ib(table_html(
            ["Coefficient","Estimate","Std Error","t-stat","p-value",f"{int(conf*100)}% CI","Significance"],
            [
                ["α (Intercept/Alpha)", f"{alpha:.6f}", f"{res['se'][0]:.6f}", f"{te:.3f}",
                 f"{pe:.4f}", f"[{ci_lo:.6f}, {ci_hi:.6f}]", sig_a],
                ["β (Beta/Slope)", f"{beta:.4f}", f"{res['se'][1]:.4f}", f"{tb:.3f}",
                 f"{pb:.4f}", f"[{bci_lo:.4f}, {bci_hi:.4f}]", sig_b],
            ]), "blue")

        # Risk decomposition
        section_heading("⚡ Risk Decomposition")
        render_ib(four_col(
            metric_card("Systematic Risk", f"{sys_risk:.1f}%", "β² × σ²ₘ / σ²ᵢ", "#FFD700"),
            metric_card("Idiosyncratic Risk", f"{idio_risk:.1f}%", "1 - R²", "#ff9f43"),
            metric_card("Ann. Volatility", f"{annual_vol*100:.2f}%", "√252 × σ_daily", "#ADD8E6"),
            metric_card("Treynor Ratio", f"{(ret_s.mean()*252/beta*100 if abs(beta) > 1e-10 else float('nan')):.3f}%", "Annual mean excess return / signed beta", "#a29bfe"),
        ), "blue")

        # ── Scatter plot with regression line ───────────────────────
        section_heading("📈 Security Characteristic Line (Scatter + OLS Fit)")
        x_line = np.linspace(ret_m.min(), ret_m.max(), 100)
        y_line = alpha + beta * x_line

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ret_m*100, y=ret_s*100, mode="markers",
            marker=dict(color="#ADD8E6", size=5, opacity=0.6),
            name="Daily Returns", hovertemplate="Mkt: %{x:.3f}%<br>Stock: %{y:.3f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x_line*100, y=y_line*100, mode="lines",
            line=dict(color="#FFD700", width=2.5),
            name=f"SCL: α={alpha*100:.4f}% β={beta:.4f}"
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#1e3a5f", line_width=1)
        fig.add_vline(x=0, line_dash="dot", line_color="#1e3a5f", line_width=1)
        fig.update_layout(**PLOTLY_LAYOUT, title=f"CAPM SCL — {sel_name} vs Nifty 50",
                          xaxis_title="Market Excess Return (%)",
                          yaxis_title="Stock Excess Return (%)", height=440)
        st.plotly_chart(fig, use_container_width=True)

        # ── Rolling Beta ─────────────────────────────────────────────
        section_heading("🔄 Rolling Beta (63-Day Window)")
        win = 63
        roll_beta, roll_r2 = [], []
        for i in range(win, len(ret_s)+1):
            yi = ret_s[i-win:i]; xi = ret_m[i-win:i]
            Xi = np.column_stack([np.ones(win), xi])
            ri = run_ols(yi, Xi)
            roll_beta.append(ri["beta"][1])
            roll_r2.append(ri["r2"])
        roll_idx = df.index[win-1:]

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.65, 0.35],
                             subplot_titles=["Rolling Beta (63-Day)", "Rolling R² (63-Day)"])
        fig2.add_trace(go.Scatter(x=roll_idx, y=roll_beta,
                                  line=dict(color="#FFD700", width=1.8),
                                  name="Beta", fill="tozeroy",
                                  fillcolor="rgba(255,215,0,0.1)"), row=1, col=1)
        fig2.add_hline(y=1, line_dash="dash", line_color="#dc3545",
                       line_width=1, annotation_text="β=1", row=1, col=1)
        fig2.add_trace(go.Scatter(x=roll_idx, y=roll_r2,
                                  line=dict(color="#ADD8E6", width=1.5),
                                  name="R²", fill="tozeroy",
                                  fillcolor="rgba(173,216,230,0.1)"), row=2, col=1)
        fig2.update_layout(**PLOTLY_LAYOUT, height=420)
        fig2.update_xaxes(gridcolor="#1e3a5f"); fig2.update_yaxes(gridcolor="#1e3a5f")
        st.plotly_chart(fig2, use_container_width=True)

        # ── Residuals analysis ───────────────────────────────────────
        section_heading("📐 Residual Diagnostics")
        resid = res["resid"]
        jb_stat, jb_p = stats.jarque_bera(resid)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=resid*100, nbinsx=40,
                                        marker_color="#004d80", opacity=0.8, name="Residuals"))
            x_n = np.linspace(resid.min(), resid.max(), 200)
            y_n = stats.norm.pdf(x_n, resid.mean(), resid.std()) * len(resid) * (resid.max()-resid.min())/40
            fig3.add_trace(go.Scatter(x=x_n*100, y=y_n, mode="lines",
                                      line=dict(color="#FFD700", width=2), name="Normal"))
            fig3.update_layout(**PLOTLY_LAYOUT, title="Residual Distribution",
                               xaxis_title="Residual (%)", yaxis_title="Count", height=280)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = go.Figure()
            qq = stats.probplot(resid, dist="norm")
            fig4.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1]*100,
                                      mode="markers", marker=dict(color="#ADD8E6", size=4), name="Data"))
            lo, hi = qq[0][0].min(), qq[0][0].max()
            fig4.add_trace(go.Scatter(x=[lo, hi],
                                      y=[(qq[1][1]+qq[1][0]*lo)*100, (qq[1][1]+qq[1][0]*hi)*100],
                                      mode="lines", line=dict(color="#FFD700", width=2), name="Normal Line"))
            fig4.update_layout(**PLOTLY_LAYOUT, title="Q-Q Plot (Normality Check)",
                               xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles (%)", height=280)
            st.plotly_chart(fig4, use_container_width=True)

        jb_ok = bdg("Normality not rejected","green") if jb_p > 0.05 else bdg("Normality rejected","red")
        dw_ok = bdg("DW near 2: heuristic","green") if 1.5 < dw < 2.5 else bdg("Check Autocorrelation","orange")
        render_ib(four_col(
            stat_box("Jarque-Bera Stat", f"{jb_stat:.3f}", "#ff9f43"),
            stat_box("JB p-value", f"{jb_p:.4f}", "#ff9f43"),
            stat_box("Normality", jb_ok, "#e6f1ff"),
            stat_box("Durbin-Watson", f"{dw:.3f} {dw_ok}", "#e6f1ff"),
        ), "blue")

        # ── Theory box ───────────────────────────────────────────────
        render_card("📚 CAPM Theory & SCL Interpretation",
            two_col(
                ib(f'<b style="color:#FFD700;-webkit-text-fill-color:#FFD700">CAPM Model</b><br>'
                   + fml("E(Rᵢ) = Rf + βᵢ[E(Rₘ) − Rf]\n\nSCL: Rᵢ − Rf = α + β(Rₘ − Rf) + ε\n"
                         "β = Cov(Rᵢ,Rₘ) / Var(Rₘ) = ρᵢₘ × (σᵢ/σₘ)"),
                   "gold"),
                ib(f'<b style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6">Beta Interpretation</b><br>'
                   + table_html(
                       ["Beta Range", "Classification", "Implication"],
                       [[bdg("β > 1.3","red"), "Highly Aggressive", "Amplifies market moves"],
                        [bdg("1.0–1.3","orange"), "Aggressive", "Slightly outperforms rising market"],
                        [bdg("β = 1.0","gold"), "Market sensitivity about 1", "Similar market sensitivity"],
                        [bdg("0.7–1.0","teal"), "Defensive", "Less sensitive to market"],
                        [bdg("β < 0.7","green"), "Very Defensive", "Low systematic risk"]]),
                   "blue")
            )
        )

    # ── Fama-French 3-Factor ─────────────────────────────────────────
    else:
        section_heading("📊 Exploratory Proxy-Factor Model")
        st.html(f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};'
                f'font-size:.88rem;margin-bottom:12px;{NO_SEL}">'
                f'Note: SMB & HML proxied using Nifty-related ETFs (data availability on NSE)'
                f'</div>')

        st.warning("Exploratory ETF/stock proxy spreads are not genuine Fama-French SMB/HML factors. Do not interpret them as research-grade size/value factors.")
        try:
            with st.spinner("Fetching exploratory proxy data..."):
                ff_df = fetch_ff_proxies(period)
            aligned = df[["xs_stock", "xs_market"]].join(ff_df, how="inner").dropna()
            if len(aligned) < 30:
                raise ValueError("Fewer than 30 matching factor/stock return dates.")
        except Exception as exc:
            st.error(f"Proxy data unavailable: {exc}. No simulated factors were substituted.")
            _bond_yield_section()
            _pe_valuation_section()
            return
        smb, hml = aligned["smb"].to_numpy(), aligned["hml"].to_numpy()
        mkt, ret_s_ff = aligned["xs_market"].to_numpy(), aligned["xs_stock"].to_numpy()
        X_ff = np.column_stack([np.ones(len(mkt)), mkt, smb, hml])
        res_ff = run_ols(ret_s_ff, X_ff, covariance, hac_lags)
        alpha_ff, beta_mkt, beta_smb, beta_hml = res_ff["beta"]

        render_card("Exploratory Proxy-Factor Regression Results",
            p(f'Model: {hl("Rᵢ − Rf = α + β_MKT(MKT) + β_SMB(SMB) + β_HML(HML) + ε")}'
              f'  |  n = {hl(str(res_ff["n"]))} trading days')
            + fml(f"α (Intercept)  = {alpha_ff*100:.5f}%  (p = {res_ff['p_vals'][0]:.4f})\n"
                  f"β_MKT (Market) = {beta_mkt:.4f}  (p = {res_ff['p_vals'][1]:.4f})\n"
                  f"β_SMB (Size)   = {beta_smb:.4f}  (p = {res_ff['p_vals'][2]:.4f})  "
                  f"→ {'Small-cap tilt' if beta_smb > 0 else 'Large-cap tilt'}\n"
                  f"β_HML (Value)  = {beta_hml:.4f}  (p = {res_ff['p_vals'][3]:.4f})  "
                  f"→ {'Value tilt' if beta_hml > 0 else 'Growth tilt'}\n\n"
                  f"R² = {res_ff['r2']:.4f}  |  Adj. R² = {res_ff['adj_r2']:.4f}  "
                  f"|  F-stat = {res_ff['f_stat']:.2f}  (p = {res_ff['f_pval']:.4f})")
        )

        hero_row([
            ("Market Beta (β_MKT)", f"{beta_mkt:.4f}", "Systematic market exposure", "", "#FFD700"),
            ("Size Beta (β_SMB)", f"{beta_smb:.4f}",
             "Small-cap tilt" if beta_smb > 0 else "Large-cap tilt", "", "#ff9f43"),
            ("Value Beta (β_HML)", f"{beta_hml:.4f}",
             "Value tilt" if beta_hml > 0 else "Growth tilt", "", "#a29bfe"),
            ("R² (3-Factor)", f"{res_ff['r2']:.4f}", "Explained variance", "", "#28a745"),
        ])

        # Factor contribution chart
        section_heading("📊 Factor Contribution Analysis")
        mkt_contrib = beta_mkt * np.std(mkt) * 100
        smb_contrib = beta_smb * np.std(smb) * 100
        hml_contrib = beta_hml * np.std(hml) * 100
        total = abs(mkt_contrib) + abs(smb_contrib) + abs(hml_contrib)

        fig_ff = go.Figure(go.Bar(
            x=["Market (MKT)", "Size (SMB)", "Value (HML)"],
            y=[mkt_contrib, smb_contrib, hml_contrib],
            marker_color=["#FFD700", "#ff9f43", "#a29bfe"],
            text=[f"{v:.4f}%" for v in [mkt_contrib, smb_contrib, hml_contrib]],
            textposition="outside", textfont=dict(color="#e6f1ff")
        ))
        fig_ff.update_layout(**PLOTLY_LAYOUT,
                             title="Factor Beta × Factor Volatility Contribution",
                             yaxis_title="Contribution (%)", height=350)
        st.plotly_chart(fig_ff, use_container_width=True)

        # Coefficient table
        section_heading("📋 Full Coefficient Table")
        coeff_names = ["α (Intercept)", "β_MKT (Market)", "β_SMB (Size)", "β_HML (Value)"]
        rows_ff = []
        for name, b, se, t, pv in zip(coeff_names, res_ff["beta"], res_ff["se"],
                                       res_ff["t_stats"], res_ff["p_vals"]):
            sig = bdg("***","green") if pv < 0.01 else bdg("**","teal") if pv < 0.05 else bdg("*","orange") if pv < 0.1 else bdg("ns","red")
            rows_ff.append([name, f"{b:.6f}", f"{se:.6f}", f"{t:.3f}", f"{pv:.4f}", sig])
        render_ib(table_html(["Factor","Estimate","Std Error","t-stat","p-value","Sig."], rows_ff), "blue")

        render_card("📚 Fama-French 3-Factor Theory",
            p(f'Extension of CAPM that adds {hl("SMB (Small Minus Big)")} and {hl("HML (High Minus Low)")} factors.')
            + fml("E(Rᵢ) − Rf = α + β_MKT[E(Rₘ)−Rf] + β_SMB·SMB + β_HML·HML\n\n"
                  "SMB = Return(Small-cap) − Return(Large-cap)  → Size premium\n"
                  "HML = Return(Value stocks) − Return(Growth stocks)  → Value premium\n"
                  "α   = Abnormal return unexplained by 3 factors (Jensen's Alpha)")
            + two_col(
                ib(f'{bdg("β_SMB > 0","orange")} → Small-cap tilt (higher exposure to size premium)<br>'
                   f'{bdg("β_SMB < 0","blue")} → Large-cap tilt (lower size premium exposure)', "orange"),
                ib(f'{bdg("β_HML > 0","gold")} → Value tilt (high book-to-market ratio exposure)<br>'
                   f'{bdg("β_HML < 0","purple")} → Growth tilt (low book-to-market exposure)', "gold")
            )
        )

    # ── Static illustrative models ────────────────────────────────────
    _bond_yield_section()
    _pe_valuation_section()


# ── Bond Yield Section ────────────────────────────────────────────
def _bond_yield_section():
    with st.expander("📈 Bond Yield Regression — YTM vs Duration, Coupon, Rating"):
        section_heading("Bond Yield Determinants (Illustrative OLS)")
        np.random.seed(7)
        n = 80
        duration = np.random.uniform(1, 15, n)
        coupon   = np.random.uniform(4, 12, n)
        rating   = np.random.choice([0, 1, 2], n)  # 0=AAA, 1=AA, 2=A
        ytm = 4.5 + 0.25*duration - 0.10*coupon + 0.80*rating + np.random.normal(0, 0.3, n)
        ytm = np.clip(ytm, 3, 13)

        X_bond = np.column_stack([np.ones(n), duration, coupon, rating])
        res_bond = run_ols(ytm, X_bond)
        a0, a1, a2, a3 = res_bond["beta"]

        render_ib(
            p(f'OLS: {hl("YTM = α₀ + α₁·Duration + α₂·Coupon + α₃·Rating_Dummy + ε")}')
            + fml(f"Ŷ(YTM) = {a0:.3f} + {a1:.4f}·Duration + {a2:.4f}·Coupon + {a3:.4f}·Rating\n\n"
                  f"Duration effect (+1yr → YTM +{a1:.4f}%): Longer bonds demand higher yield\n"
                  f"Coupon effect   (+1%  → YTM {a2:.4f}%): Higher coupon → lower required yield\n"
                  f"Rating effect   (AAA→AA → YTM +{a3:.4f}%): Credit spread for downgrade\n"
                  f"R² = {res_bond['r2']:.4f}  |  Adj. R² = {res_bond['adj_r2']:.4f}"),
            "gold"
        )

        # Scatter YTM vs Duration
        fig_bond = go.Figure()
        colors_map = {0: "#28a745", 1: "#FFD700", 2: "#dc3545"}
        labels_map = {0: "AAA", 1: "AA", 2: "A"}
        for r in [0, 1, 2]:
            mask = rating == r
            fig_bond.add_trace(go.Scatter(
                x=duration[mask], y=ytm[mask], mode="markers",
                name=f"Rating {labels_map[r]}",
                marker=dict(color=colors_map[r], size=7, opacity=0.8)
            ))
        x_d = np.linspace(1, 15, 100)
        fig_bond.add_trace(go.Scatter(
            x=x_d, y=a0 + a1*x_d + a2*np.mean(coupon) + a3*1,
            mode="lines", line=dict(color="#ADD8E6", width=2, dash="dash"),
            name="OLS Fit (AA, avg coupon)"
        ))
        fig_bond.update_layout(**PLOTLY_LAYOUT, title="Bond YTM vs Duration (by Rating)",
                               xaxis_title="Duration (Years)", yaxis_title="YTM (%)", height=350)
        st.plotly_chart(fig_bond, use_container_width=True)


# ── P/E Valuation Section ─────────────────────────────────────────
def _pe_valuation_section():
    with st.expander("💰 P/E Valuation Regression — P/E vs ROE, Growth, Beta"):
        section_heading("P/E Multiple Determinants (Illustrative OLS)")
        np.random.seed(13)
        n = 60
        roe    = np.random.uniform(8, 35, n)
        growth = np.random.uniform(5, 25, n)
        beta   = np.random.uniform(0.4, 1.8, n)
        pe = 5 + 0.8*roe + 0.9*growth - 8*beta + np.random.normal(0, 4, n)
        pe = np.clip(pe, 5, 80)

        X_pe = np.column_stack([np.ones(n), roe, growth, beta])
        res_pe = run_ols(pe, X_pe)
        b0, b1, b2, b3 = res_pe["beta"]

        render_ib(
            p(f'OLS: {hl("P/E = β₀ + β₁·ROE + β₂·Growth + β₃·Beta + ε")}')
            + fml(f"Ŷ(P/E) = {b0:.3f} + {b1:.4f}·ROE + {b2:.4f}·Growth + {b3:.4f}·Beta\n\n"
                  f"ROE effect    (+1% → P/E +{b1:.3f}×): Higher profitability → premium valuation\n"
                  f"Growth effect (+1% → P/E +{b2:.3f}×): Higher growth → higher multiple\n"
                  f"Beta effect   (+0.1 → P/E {b3*0.1:.3f}×): Higher risk → lower valuation multiple\n"
                  f"R² = {res_pe['r2']:.4f}  |  Adj. R² = {res_pe['adj_r2']:.4f}"),
            "purple"
        )

        # 3D scatter simulation
        fig_pe = go.Figure()
        fig_pe.add_trace(go.Scatter(
            x=roe, y=pe, mode="markers",
            marker=dict(color=growth, colorscale="Viridis", size=8, opacity=0.8,
                        colorbar=dict(title="Growth %", thickness=12)),
            name="Stocks", hovertemplate="ROE: %{x:.1f}%<br>P/E: %{y:.1f}×<extra></extra>"
        ))
        x_r = np.linspace(roe.min(), roe.max(), 100)
        fig_pe.add_trace(go.Scatter(
            x=x_r, y=b0 + b1*x_r + b2*np.mean(growth) + b3*1.0,
            mode="lines", line=dict(color="#FFD700", width=2),
            name="OLS Fit (avg growth, β=1)"
        ))
        fig_pe.update_layout(**PLOTLY_LAYOUT, title="P/E Multiple vs ROE (colour = Growth Rate)",
                             xaxis_title="ROE (%)", yaxis_title="P/E Multiple (×)", height=350)
        st.plotly_chart(fig_pe, use_container_width=True)

        metric_row([
            ("Const (β₀)", f"{b0:.2f}×", "Base P/E", "#8892b0"),
            ("ROE coeff (β₁)", f"{b1:.3f}×", "Per 1% ROE", "#28a745"),
            ("Growth coeff (β₂)", f"{b2:.3f}×", "Per 1% growth", "#FFD700"),
            ("Beta coeff (β₃)", f"{b3:.3f}×", "Per unit beta", "#dc3545"),
        ])
