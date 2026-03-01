"""
components.py — Mountain Path design system (aligned with Nifty VaR framework).
All HTML via st.html() with 100% inline styles + user-select:none.
"""
import streamlit as st

# ── Color & Font palette ──────────────────────────────────────────────────────
S = {
    "txt":  "#e6f1ff", "gold": "#FFD700", "lb":   "#ADD8E6",
    "grn":  "#28a745", "red":  "#dc3545", "acc":  "#64ffda",
    "mut":  "#8892b0", "card": "#112240", "blue": "#003366",
    "mid":  "#004d80", "dark": "#0a1628", "bdr":  "#1e3a5f",
    "org":  "#ff9f43", "pur":  "#a29bfe", "teal": "#00cec9",
    "amb":  "#fdcb6e",
}
FH = "'Playfair Display',serif"
FB = "'Source Sans Pro',sans-serif"
FM = "'JetBrains Mono',monospace"
TXT = f"color:#e6f1ff;font-family:{FB};line-height:1.65;-webkit-text-fill-color:#e6f1ff"
NO_SEL = "user-select:none;-webkit-user-select:none"

_IB = {
    "blue":   ("rgba(0,51,102,0.6)",    "#ADD8E6"),
    "gold":   ("rgba(255,215,0,0.13)",  "#FFD700"),
    "green":  ("rgba(40,167,69,0.2)",   "#28a745"),
    "red":    ("rgba(220,53,69,0.2)",   "#dc3545"),
    "orange": ("rgba(255,159,67,0.15)", "#ff9f43"),
    "purple": ("rgba(162,155,254,0.15)","#a29bfe"),
    "teal":   ("rgba(0,206,201,0.15)",  "#00cec9"),
}
_BADGE = {
    "blue":   ("#004d80","#ffffff"), "gold":   ("#FFD700","#0a1628"),
    "green":  ("#28a745","#ffffff"), "red":    ("#dc3545","#ffffff"),
    "orange": ("#ff9f43","#0a1628"), "purple": ("#a29bfe","#0a1628"),
    "teal":   ("#00cec9","#0a1628"),
}

# ── Core render functions ─────────────────────────────────────────────────────

def render_card(title: str, body_html: str, accent_color: str = "#FFD700"):
    h2 = (f'<h2 style="font-family:{FH};font-size:1.35rem;color:{accent_color};'
          f'-webkit-text-fill-color:{accent_color};border-bottom:1px solid #1e3a5f;'
          f'padding-bottom:8px;margin:0 0 14px 0;{NO_SEL}">{title}</h2>')
    st.html(f'<div style="background:#112240;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:22px;margin-bottom:18px;{TXT};{NO_SEL}">{h2}{body_html}</div>')

def section_heading(title, color="#ADD8E6"):
    st.html(f'<h3 style="font-family:{FH};font-size:1.1rem;color:{color};'
            f'-webkit-text-fill-color:{color};margin:18px 0 10px 0;'
            f'border-left:3px solid #FFD700;padding-left:10px;{NO_SEL}">{title}</h3>')

def ib(content, variant="blue"):
    bg, bc = _IB.get(variant, _IB["blue"])
    return (f'<div style="background:{bg};border-left:4px solid {bc};border-radius:8px;'
            f'padding:13px 15px;margin:10px 0;{TXT};{NO_SEL}">{content}</div>')

def render_ib(content, variant="blue"): st.html(ib(content, variant))

def fml(content):
    return (f'<div style="background:#0d1f3a;border-left:4px solid #FFD700;border-radius:6px;'
            f'padding:13px 17px;margin:10px 0;font-family:{FM};font-size:.88rem;'
            f'color:#64ffda;-webkit-text-fill-color:#64ffda;line-height:1.85;'
            f'white-space:pre-wrap;overflow-x:auto;{NO_SEL}">{content}</div>')

def bdg(text, variant="blue"):
    bg, fg = _BADGE.get(variant, _BADGE["blue"])
    return (f'<span style="background:{bg};color:{fg};-webkit-text-fill-color:{fg};'
            f'display:inline-block;padding:2px 10px;border-radius:20px;font-size:.77rem;'
            f'font-weight:700;margin:2px;font-family:{FB};{NO_SEL}">{text}</span>')

def metric_card(label, value, sub="", accent="#FFD700"):
    return (f'<div style="background:#112240;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:16px 14px;text-align:center;{NO_SEL}">'
            f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};'
            f'font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">{label}</div>'
            f'<div style="font-family:{FM};font-size:1.4rem;font-weight:700;'
            f'color:{accent};-webkit-text-fill-color:{accent}">{value}</div>'
            f'{"" if not sub else f"""<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};font-size:.78rem;margin-top:4px">{sub}</div>"""}'
            f'</div>')

def metric_row(metrics):
    """metrics: list of (label, value, sub, accent) tuples"""
    cols = "".join(f'<div>{metric_card(l, v, s, a)}</div>'
                   for l, v, s, a in metrics)
    n = len(metrics)
    st.html(f'<div style="display:grid;grid-template-columns:repeat({n},1fr);gap:12px;'
            f'margin:14px 0;{NO_SEL}">{cols}</div>')

def stat_box(label, value, color="#ADD8E6"):
    return (f'<div style="background:#0d1f3a;border-radius:6px;padding:10px 14px;'
            f'text-align:center;{NO_SEL}">'
            f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-size:.78rem;'
            f'font-family:{FB};margin-bottom:4px">{label}</div>'
            f'<div style="color:{color};-webkit-text-fill-color:{color};font-family:{FM};'
            f'font-weight:700;font-size:1.05rem">{value}</div>'
            f'</div>')

def hero_metric(label, value, sub="", trend="", accent="#FFD700"):
    trend_html = ""
    if trend == "up":
        trend_html = f'<span style="color:#28a745;-webkit-text-fill-color:#28a745">▲</span>'
    elif trend == "down":
        trend_html = f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545">▼</span>'
    return (f'<div style="background:linear-gradient(135deg,#112240,#0d1b2a);'
            f'border:1px solid #1e3a5f;border-top:3px solid {accent};border-radius:10px;'
            f'padding:18px 16px;text-align:center;{NO_SEL}">'
            f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};'
            f'font-size:.82rem;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">{label}</div>'
            f'<div style="font-family:{FM};font-size:1.6rem;font-weight:700;'
            f'color:{accent};-webkit-text-fill-color:{accent}">{trend_html}{value}</div>'
            f'{"" if not sub else f"""<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};font-size:.8rem;margin-top:5px">{sub}</div>"""}'
            f'</div>')

def hero_row(metrics):
    """metrics: list of (label, value, sub, trend, accent)"""
    cols = "".join(f'<div>{hero_metric(l, v, s, t, a)}</div>'
                   for l, v, s, t, a in metrics)
    n = len(metrics)
    st.html(f'<div style="display:grid;grid-template-columns:repeat({n},1fr);gap:14px;'
            f'margin:16px 0;{NO_SEL}">{cols}</div>')

# ── Text helpers ──────────────────────────────────────────────────────────────
def hl(t):    return f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">{t}</span>'
def gt(t):    return f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">{t}</span>'
def rt2(t):   return f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">{t}</span>'
def org(t):   return f'<span style="color:#ff9f43;-webkit-text-fill-color:#ff9f43;font-weight:600">{t}</span>'
def pur(t):   return f'<span style="color:#a29bfe;-webkit-text-fill-color:#a29bfe;font-weight:600">{t}</span>'
def lb_t(t):  return f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6">{t}</span>'
def acc_t(t): return f'<span style="color:#64ffda;-webkit-text-fill-color:#64ffda;font-family:{FM}">{t}</span>'
def txt_s(t): return f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">{t}</span>'
def teal_t(t):return f'<span style="color:#00cec9;-webkit-text-fill-color:#00cec9;font-weight:600">{t}</span>'
def p(content): return f'<p style="{TXT};margin-bottom:7px">{content}</p>'

# ── Layout helpers ────────────────────────────────────────────────────────────
def steps_html(steps):
    rows = ""
    for i, (title, body) in enumerate(steps, 1):
        rows += (f'<div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start;{NO_SEL}">'
                 f'<div style="background:#FFD700;color:#0a1628;-webkit-text-fill-color:#0a1628;'
                 f'border-radius:50%;min-width:28px;height:28px;display:flex;align-items:center;'
                 f'justify-content:center;font-weight:700;font-size:.85rem;font-family:{FB}">{i}</div>'
                 f'<div style="{TXT};flex:1">'
                 f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">{title}</span><br>'
                 f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">{body}</span>'
                 f'</div></div>')
    return rows

def two_col(left, right):
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0">'
            f'<div>{left}</div><div>{right}</div></div>')

def three_col(a, b, c):
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:10px 0">'
            f'<div>{a}</div><div>{b}</div><div>{c}</div></div>')

def four_col(a, b, c, d):
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin:10px 0">'
            f'<div>{a}</div><div>{b}</div><div>{c}</div><div>{d}</div></div>')

def table_html(headers, rows, stripe=True):
    ths = "".join(f'<th style="background:#003366;color:#FFD700;-webkit-text-fill-color:#FFD700;'
                  f'padding:9px 12px;text-align:left;font-weight:600;font-family:{FB}">{h}</th>'
                  for h in headers)
    trs = ""
    for i, row in enumerate(rows):
        bg = "background:rgba(0,51,102,0.15);" if stripe and i % 2 == 0 else ""
        cells = "".join(f'<td style="{bg}padding:8px 12px;border-bottom:1px solid #1e3a5f;'
                        f'color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-family:{FB}">{c}</td>'
                        for c in row)
        trs += f'<tr>{cells}</tr>'
    return (f'<div style="overflow-x:auto;margin:10px 0;{NO_SEL}">'
            f'<table style="width:100%;border-collapse:collapse;font-size:.88rem">'
            f'<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>')

def progress_bar(label, value, max_val=100, color="#FFD700"):
    pct = min(100, max(0, (value / max_val) * 100))
    return (f'<div style="margin:8px 0;{NO_SEL}">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
            f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-family:{FB};font-size:.85rem">{label}</span>'
            f'<span style="color:{color};-webkit-text-fill-color:{color};font-family:{FM};font-size:.85rem;font-weight:600">{value:.2f}</span>'
            f'</div>'
            f'<div style="background:#1e3a5f;border-radius:4px;height:6px">'
            f'<div style="background:{color};border-radius:4px;height:6px;width:{pct:.1f}%;'
            f'transition:width .5s"></div></div></div>')

def alert_box(message, variant="blue"):
    bg, bc = _IB.get(variant, _IB["blue"])
    icons = {"blue": "ℹ️", "gold": "⭐", "green": "✅", "red": "⚠️", "orange": "🔔", "purple": "🔮"}
    icon = icons.get(variant, "ℹ️")
    return (f'<div style="background:{bg};border:1px solid {bc};border-radius:8px;'
            f'padding:12px 16px;margin:10px 0;{TXT};{NO_SEL}">'
            f'{icon} {message}</div>')

def formula_block(title, formula, note=""):
    note_html = f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};font-size:.82rem;margin-top:8px">{note}</div>' if note else ""
    return (f'<div style="background:#0d1f3a;border:1px solid #1e3a5f;border-radius:8px;'
            f'padding:16px;margin:10px 0;{NO_SEL}">'
            f'<div style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-family:{FH};'
            f'font-size:.9rem;font-weight:700;margin-bottom:8px">{title}</div>'
            f'{fml(formula)}{note_html}</div>')
