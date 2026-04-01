#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
TPI · Analyse Financière — Interface Streamlit unifiée MQ + ACT
===============================================================
Prérequis : avoir exécuté pipeline_MQ.py ET pipeline_ACT.py
Lancement  : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import statsmodels.formula.api as smf
import requests, os

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TPI · Analyse Financière",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.metric-card {
    background: #0f1117; border: 1px solid #2a2d35;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 8px;
}
.metric-label { font-size: 11px; color: #6b7280; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 500; }
.score-badge { display: inline-block; padding: 6px 14px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500; }
.section-title { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #6b7280; margin: 24px 0 12px; border-bottom: 1px solid #2a2d35; padding-bottom: 6px; }
.note-box { background: #0f1a2e; border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 10px 14px; font-size: 12px; color: #93c5fd; margin-top: 8px; line-height: 1.6; }
.warn-box { background: #1c1a0e; border: 1px solid #713f12; border-radius: 6px;
    padding: 10px 14px; font-size: 12px; color: #fde68a; margin-top: 8px; line-height: 1.6; }
.dataset-badge-mq  { background:#1e3a5f; color:#93c5fd; padding:4px 12px;
    border-radius:4px; font-size:12px; font-weight:600; }
.dataset-badge-act { background:#1a3320; color:#86efac; padding:4px 12px;
    border-radius:4px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── CHEMINS ───────────────────────────────────────────────────────────────────
LOCAL_DIR = '/Users/hamidi/Desktop/'
GITHUB_RELEASE_URL = "https://github.com/Ihssane-Hamidi/TPI_MQ/releases/download/v1.0/"

FILES = {
    'mq_metriques':       'mq2_metriques.parquet',
    'mq_prix':            'mq2_prix_journaliers.parquet',
    'act_metriques':      'act_metriques.parquet',
    'act_prix':           'act_prix_journaliers.parquet',
}

def get_parquet(key):
    filename = FILES[key]
    local    = os.path.join(LOCAL_DIR, filename)
    if os.path.exists(local):   return local
    if os.path.exists(filename): return filename
    url = GITHUB_RELEASE_URL + filename
    with st.spinner(f"Téléchargement {filename}..."):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename

# ── CONSTANTES ────────────────────────────────────────────────────────────────
PERIODS_LABELS = {
    '2023':'2023', '2024':'2024', '2025':'2025', '2023_2025':'2023–2025'
}
QUINTILE_COLORS = {
    'Q1':'#ef4444','Q2':'#f97316','Q3':'#a3a3a3','Q4':'#86efac','Q5':'#16a34a'
}
PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
@st.cache_data
def load_mq():
    return pd.read_parquet(get_parquet('mq_metriques'))

@st.cache_data
def load_mq_prix():
    px = pd.read_parquet(get_parquet('mq_prix'))
    px.index = pd.to_datetime(px.index)
    return px

@st.cache_data
def load_act():
    return pd.read_parquet(get_parquet('act_metriques'))

@st.cache_data
def load_act_prix():
    px = pd.read_parquet(get_parquet('act_prix'))
    px.index = pd.to_datetime(px.index)
    return px

@st.cache_data
def load_brent():
    raw = yf.download('BZ=F', start='2023-01-01', end='2025-10-31',
                      auto_adjust=True, progress=False)
    b = raw['Close'].iloc[:,0] if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
    b.index = pd.to_datetime(b.index)
    return b.dropna()

# ── UTILITAIRES ───────────────────────────────────────────────────────────────
def detect_oil_rallies(brent, threshold=0.15, window=60):
    prices   = brent.dropna()
    roll_ret = prices.pct_change(window)
    rallies, in_rally = [], False
    rally_start = peak_price = peak_date = None
    for date, val in roll_ret.items():
        price = prices.loc[date]
        if not in_rally:
            if pd.notna(val) and val >= threshold:
                idx         = max(0, prices.index.get_loc(date) - window)
                rally_start = prices.index[idx]
                in_rally, peak_price, peak_date = True, price, date
        else:
            if price > peak_price:
                peak_price, peak_date = price, date
            elif price < peak_price * 0.92:
                rallies.append((rally_start, peak_date))
                in_rally = False
    if in_rally and rally_start:
        rallies.append((rally_start, prices.index[-1]))
    return rallies

def add_oil_rectangles(fig, rallies, first_only=False):
    for i, (s, e) in enumerate(rallies):
        fig.add_vrect(
            x0=str(s), x1=str(e),
            fillcolor='rgba(59,130,246,0.13)', line_width=0,
            annotation_text="↑ Brent" if (not first_only or i==0) else "",
            annotation_position="top left",
            annotation_font_size=9, annotation_font_color='#93c5fd',
        )
    return fig

def score_color(pct):
    if pct >= 0.67:   return '#14532d','#86efac','Score élevé'
    elif pct >= 0.33: return '#713f12','#fde68a','Score moyen'
    else:             return '#7f1d1d','#fca5a5','Score faible'

def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))

def sig_stars(p):
    if p<0.01: return '★★★'
    if p<0.05: return '★★'
    if p<0.10: return '★'
    return 'ns'

def metric_card(col, label, val, ok=True):
    color = '#86efac' if ok else '#fca5a5'
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{val}</div>
    </div>""", unsafe_allow_html=True)

def plot_rendements_societe(px, ticker, brent, rallies, company_name):
    """Graphique rendements journaliers + Brent axe secondaire."""
    ret_daily  = px.pct_change().dropna()
    brent_al   = brent.reindex(ret_daily.index, method='ffill').dropna()
    brent_norm = brent_al / brent_al.iloc[0] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = add_oil_rectangles(fig, rallies, first_only=True)
    fig.add_hline(y=0, line_width=0.5, line_color='rgba(255,255,255,0.15)')

    bar_col = ['rgba(239,68,68,0.8)' if v < 0 else 'rgba(52,211,153,0.8)'
               for v in ret_daily.values]
    fig.add_trace(go.Bar(
        x=ret_daily.index, y=ret_daily.values,
        marker_color=bar_col, name='Rendement journalier',
        hovertemplate='%{x|%d %b %Y}<br>%{y:.2%}<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=brent_norm.index, y=brent_norm.values,
        name='Brent (base 100)',
        line=dict(color='rgba(59,130,246,0.7)', width=1.5),
        hovertemplate='%{x|%d %b %Y}<br>Brent: %{y:.1f}<extra></extra>',
    ), secondary_y=True)

    fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    fig.update_yaxes(tickformat='.1%', title_text='Rendement',
                     gridcolor='rgba(255,255,255,0.05)', secondary_y=False)
    fig.update_yaxes(title_text='Brent (base 100)',
                     gridcolor='rgba(0,0,0,0)', secondary_y=True)
    return fig

def plot_metriques_periode(row, periods, labels):
    """Barres rendement / volatilité / Sharpe par période."""
    rendements = [row.get(f'Rendement_{p}', np.nan) for p in periods]
    vols       = [row.get(f'Volatilite_{p}', np.nan) for p in periods]
    sharpes    = [row.get(f'Sharpe_{p}', np.nan)     for p in periods]

    r_col = ['rgba(239,68,68,0.8)' if (v and v<0) else 'rgba(52,211,153,0.8)' for v in rendements]
    s_col = ['rgba(239,68,68,0.8)' if (v and v<0) else 'rgba(251,191,36,0.8)' for v in sharpes]

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=('Rendement','Volatilité annualisée','Sharpe'))
    fig.add_trace(go.Bar(x=labels, y=rendements, marker_color=r_col,
                         hovertemplate='%{x}<br>%{y:.1%}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=vols, marker_color='rgba(99,179,237,0.7)',
                         hovertemplate='%{x}<br>%{y:.1%}<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=sharpes, marker_color=s_col,
                         hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'), row=1, col=3)
    fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
    fig.update_yaxes(tickformat='.0%', row=1, col=1)
    fig.update_yaxes(tickformat='.0%', row=1, col=2)
    return fig

def page_societe(valid, prices, brent, rallies,
                 score_col, score_label, quintile_col, pct_col,
                 secteur_col, extra_info_fn, company_col='Company Name'):
    """Page société commune MQ et ACT."""
    st.sidebar.markdown('<p class="section-title">Sélection</p>', unsafe_allow_html=True)
    companies    = sorted(valid[company_col].unique().tolist())
    company_name = st.sidebar.selectbox("Entreprise", companies, label_visibility="collapsed")
    row          = valid[valid[company_col] == company_name].iloc[0]
    ticker       = row['ticker']

    bg, fg, label = score_color(float(row[pct_col]))
    score_val     = row[score_col]
    score_fmt     = f"{score_val:.1%}" if score_val <= 1 else f"{score_val:.1f}/100"

    st.sidebar.markdown('<p class="section-title">Score</p>', unsafe_allow_html=True)
    extra_html = extra_info_fn(row)
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{score_label}</div>
        <div class="metric-value" style="color:{fg}">{score_fmt}</div>
        <div style="margin-top:8px">
            <span class="score-badge" style="background:{bg};color:{fg}">
                {label} · {row[pct_col]:.0%} du panel
            </span>
        </div>
        <div style="margin-top:10px;font-size:12px;color:#6b7280;line-height:1.9">
            {extra_html}
        </div>
    </div>""", unsafe_allow_html=True)

    if ticker not in prices.columns:
        st.warning(f"Prix non disponibles pour {ticker}")
        return

    px = prices[ticker].dropna()
    if len(px) < 50:
        st.warning(f"Données insuffisantes pour {ticker} ({len(px)} jours)")
        return

    st.markdown(f"### {company_name}  ·  `{ticker}`")
    st.markdown('<p class="section-title">Rendements journaliers · Zones bleues = hausses Brent</p>',
                unsafe_allow_html=True)
    st.plotly_chart(plot_rendements_societe(px, ticker, brent, rallies, company_name),
                    use_container_width=True)

    st.markdown('<p class="section-title">Performance par période</p>', unsafe_allow_html=True)
    periods = list(PERIODS_LABELS.keys())
    labels  = list(PERIODS_LABELS.values())
    st.plotly_chart(plot_metriques_periode(row, periods, labels), use_container_width=True)

def page_panel(valid, prices, brent, rallies, quintile_col, company_col='Company Name'):
    """Page panel quintiles commune."""
    st.markdown("### Panel · Rendements cumulés par quintile")
    st.markdown('<p class="section-title">Q1 (scores faibles) → Q5 (scores élevés) · Zones bleues = hausses Brent</p>',
                unsafe_allow_html=True)

    brent_norm = brent / brent.iloc[0] * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = add_oil_rectangles(fig, rallies, first_only=True)
    fig.add_hline(y=0, line_width=0.5, line_color='rgba(255,255,255,0.15)')

    for q in ['Q1','Q2','Q3','Q4','Q5']:
        tickers_q = valid[valid[quintile_col]==q]['ticker'].dropna().tolist()
        tickers_q = [t for t in tickers_q if t in prices.columns]
        if not tickers_q: continue
        cum_par_ticker = (1 + prices[tickers_q].pct_change()).cumprod() - 1
        cum_r = cum_par_ticker.mean(axis=1).dropna()
        fig.add_trace(go.Scatter(
            x=cum_r.index, y=cum_r.values,
            name=f'{q} (n={len(tickers_q)})',
            line=dict(color=QUINTILE_COLORS[q], width=2),
            hovertemplate='%{x|%d %b %Y}<br>%{y:.1%}<extra>'+q+'</extra>',
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=brent_norm.index, y=brent_norm.values,
        name='Brent (base 100)',
        line=dict(color='rgba(59,130,246,0.5)', width=1.5, dash='dot'),
    ), secondary_y=True)

    fig.update_layout(**PLOTLY_LAYOUT, height=470,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    fig.update_yaxes(tickformat='.0%', title_text='Rendement cumulé',
                     gridcolor='rgba(255,255,255,0.05)', secondary_y=False)
    fig.update_yaxes(title_text='Brent (base 100)', gridcolor='rgba(0,0,0,0)', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

def page_ols(valid, score_col, score_label, secteur_col, total_panel, quintile_col):
    """Page OLS commune avec 3 modèles."""
    st.markdown(f"### Régression OLS · {score_label} → Performance financière")

    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox("Période", list(PERIODS_LABELS.keys()),
                              format_func=lambda x: PERIODS_LABELS[x])
    with c2:
        dep_choice = st.selectbox("Variable expliquée",
            ['Rendement','Volatilite','Sharpe','MaxDrawdown'],
            format_func=lambda x: {'Rendement':'Rendement','Volatilite':'Volatilité',
                                    'Sharpe':'Sharpe','MaxDrawdown':'Max Drawdown'}[x])
    with c3:
        model_choice = st.selectbox("Modèle",
            ['simple','interaction','fama_french'],
            format_func=lambda x: {
                'simple':      'Modèle 1 · Simple (Score + Secteur)',
                'interaction': 'Modèle 2 · Interaction Score × Secteur',
                'fama_french': 'Modèle 3 · Fama-French (Score + Secteur + Taille + B/M)',
            }[x])

    dep_var = f'{dep_choice}_{period}'
    base    = ['Company Name' if 'Company Name' in valid.columns else valid.columns[0],
               score_col, secteur_col, quintile_col, dep_var]
    if model_choice == 'fama_french':
        base += ['LogMarketCap','BookToMarket']

    df_ols = valid[[c for c in base if c in valid.columns]].dropna().copy()
    df_ols[dep_var] = winsorize(df_ols[dep_var])
    n_ols   = len(df_ols)

    if n_ols < 30:
        st.warning("Pas assez d'observations.")
        return

    st.markdown(f"""<div class="note-box">
        Régression sur <b>{n_ols}</b> entreprises ({n_ols/total_panel:.0%} du panel)
        après winsorisation 1–99ème percentile · Erreurs robustes HC3<br>
        ⚠ <b>Biais de sélection</b> : entreprises non cotées exclues.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    formulas = {
        'simple':      f'{dep_var} ~ {score_col} + C({secteur_col})',
        'interaction': f'{dep_var} ~ {score_col} * C({secteur_col})',
        'fama_french': f'{dep_var} ~ {score_col} + C({secteur_col}) + LogMarketCap + BookToMarket',
    }
    # Renommer score_col pour statsmodels (pas d'espaces ni slash)
    df_ols = df_ols.rename(columns={score_col: 'Score_principal'})
    formula = formulas[model_choice].replace(score_col, 'Score_principal')

    model  = smf.ols(formula, data=df_ols).fit(cov_type='HC3')
    coef   = model.params['Score_principal']
    pval   = model.pvalues['Score_principal']
    r2_adj = model.rsquared_adj
    n      = int(model.nobs)

    cols = st.columns(4)
    for col, lbl, val, ok in [
        (cols[0], 'Coefficient score',  f'{coef:+.3f}',            coef>0),
        (cols[1], 'p-value',            f'{pval:.3f} {sig_stars(pval)}', pval<0.05),
        (cols[2], 'R² ajusté',          f'{r2_adj:.3f}',           True),
        (cols[3], 'Observations',       str(n),                    True),
    ]:
        metric_card(col, lbl, val, ok)

    # Graphique coefficients par secteur (modèle interaction)
    if model_choice == 'interaction':
        st.markdown('<p class="section-title">Effet du score par secteur</p>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="note-box">
            Coefficient = effet du score dans le secteur de référence (premier alphabétique).<br>
            Effet dans les autres secteurs = coefficient global + terme d'interaction.
        </div>""", unsafe_allow_html=True)
        st.markdown("")

        secteurs = sorted(df_ols[secteur_col].unique().tolist())
        ref      = secteurs[0]
        coef_par_secteur = []
        for s in secteurs:
            if s == ref:
                c  = model.params.get('Score_principal', np.nan)
                se = model.bse.get('Score_principal', np.nan)
                p  = model.pvalues.get('Score_principal', np.nan)
            else:
                k  = f'Score_principal:C({secteur_col})[T.{s}]'
                if k in model.params:
                    c  = model.params['Score_principal'] + model.params[k]
                    se = model.bse[k]
                    p  = model.pvalues[k]
                else:
                    c, se, p = np.nan, np.nan, np.nan
            coef_par_secteur.append({
                'Secteur': s, 'Coef': c, 'SE': se, 'pval': p,
                'sig': sig_stars(p) if pd.notna(p) else '',
                'n': len(df_ols[df_ols[secteur_col]==s]),
            })

        df_coef = pd.DataFrame(coef_par_secteur).dropna(subset=['Coef']).sort_values('Coef')
        bar_col = ['rgba(52,211,153,0.8)' if c>0 else 'rgba(239,68,68,0.8)'
                   for c in df_coef['Coef']]

        fig_c = go.Figure(go.Bar(
            x=df_coef['Coef'], y=df_coef['Secteur'], orientation='h',
            marker_color=bar_col,
            error_x=dict(type='data', array=1.96*df_coef['SE'],
                         color='rgba(255,255,255,0.4)', thickness=1.5, width=4),
            text=[f"{c:+.3f} {s} (n={n})"
                  for c,s,n in zip(df_coef['Coef'],df_coef['sig'],df_coef['n'])],
            textposition='outside',
            hovertemplate='%{y}<br>Coef: %{x:+.3f}<extra></extra>',
        ))
        fig_c.add_vline(x=0, line_width=1, line_color='rgba(255,255,255,0.3)')
        fig_c.update_layout(**PLOTLY_LAYOUT, height=max(350, len(df_coef)*40),
                            showlegend=False,
                            xaxis=dict(title=f'Coefficient ({dep_choice})',
                                       gridcolor='rgba(255,255,255,0.05)'),
                            yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown("""<div class="warn-box">
            ★★★ p&lt;0.01 · ★★ p&lt;0.05 · ★ p&lt;0.10 · ns = non significatif<br>
            Barres d'erreur = intervalle de confiance 95%.
        </div>""", unsafe_allow_html=True)

    # Scatter
    st.markdown(f'<p class="section-title">Score vs {dep_choice} · coloré par quintile</p>',
                unsafe_allow_html=True)
    x_rng = np.linspace(df_ols['Score_principal'].min(), df_ols['Score_principal'].max(), 100)
    y_fit = model.params['Intercept'] + model.params['Score_principal'] * x_rng

    fig4 = go.Figure()
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = df_ols[df_ols[quintile_col]==q]
        if sub.empty: continue
        fig4.add_trace(go.Scatter(
            x=sub['Score_principal'], y=sub[dep_var],
            mode='markers', name=q,
            marker=dict(color=QUINTILE_COLORS[q], size=6, opacity=0.65),
            hovertemplate='%{text}<br>Score: %{x:.2f}<br>'+dep_choice+': %{y:.3f}<extra>'+q+'</extra>',
            text=sub.iloc[:,0],
        ))
    fig4.add_trace(go.Scatter(
        x=x_rng, y=y_fit, mode='lines', name='OLS fit',
        line=dict(color='rgba(251,191,36,0.9)', width=2, dash='dot'),
    ))
    fig4.update_layout(**PLOTLY_LAYOUT, height=400,
                       xaxis=dict(title=score_label, gridcolor='rgba(255,255,255,0.05)'),
                       yaxis=dict(title=f'{dep_choice} · {PERIODS_LABELS[period]}',
                                  gridcolor='rgba(255,255,255,0.05)'),
                       legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Résultats complets"):
        st.text(model.summary().as_text())

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
try:
    df_mq      = load_mq()
    prices_mq  = load_mq_prix()
    df_act     = load_act()
    prices_act = load_act_prix()
    brent      = load_brent()
except Exception as e:
    st.error(f"Erreur chargement : {e}\nLancez d'abord pipeline_MQ.py et pipeline_ACT.py")
    st.stop()

rallies = detect_oil_rallies(brent)

# Préparation MQ
valid_mq = df_mq[df_mq['ticker'].notna() & (df_mq['ticker']!='None')
                 & df_mq['Rendement_2023_2025'].notna()].copy()
valid_mq = valid_mq.rename(columns={valid_mq.columns[0]: 'Company Name'}) \
    if 'Company Name' not in valid_mq.columns else valid_mq

# Préparation ACT
col_nom_act = df_act.columns[0]
col_score_act = df_act.columns[5]   # Performance Score /100
col_secteur_act = df_act.columns[3] # Secteur
valid_act = df_act[df_act['ticker'].notna() & (df_act['ticker']!='None')
                   & df_act['Rendement_2023_2025'].notna()].copy()
valid_act = valid_act.rename(columns={col_nom_act: 'Company Name'})

# ── NAVIGATION ────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 TPI · Analyse Financière")
st.sidebar.markdown("---")

dataset = st.sidebar.radio(
    "Référentiel",
    ["🔵 MQ — Management Quality", "🟢 ACT — Transition Carbone"],
    label_visibility="visible",
)
is_mq = dataset.startswith("🔵")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Société", "Panel Quintiles", "Régression OLS", "Analyse Brent"],
    label_visibility="collapsed",
)

# Badge dataset
badge_class = "dataset-badge-mq" if is_mq else "dataset-badge-act"
badge_text  = "Management Quality" if is_mq else "ACT — Transition Carbone"
st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
st.markdown("")


def calc_rendement_brent(prices, tickers, rallies):
    """
    Calcule le rendement cumulé de chaque ticker
    sur l'ensemble des fenêtres de hausse Brent (option A — agrégé).
    """
    records = {}
    for t in tickers:
        if t not in prices.columns:
            records[t] = np.nan
            continue
        px     = prices[t].dropna()
        cumret = 1.0
        n_days = 0
        for s, e in rallies:
            window = px.loc[s:e].dropna()
            if len(window) < 5:
                continue
            cumret *= (window.iloc[-1] / window.iloc[0])
            n_days += len(window)
        records[t] = float(cumret - 1) if n_days >= 10 else np.nan
    return records


def page_brent(valid, prices, rallies, score_col, score_label,
               secteur_col, quintile_col, total_panel, dataset_label):
    """Page Analyse Brent — comparaison modèle général vs pendant hausse Brent."""

    st.markdown(f"### Analyse Brent · {dataset_label}")
    st.markdown(
        f'<p class="section-title">Comparaison des coefficients OLS — modèle général vs pendant les hausses Brent</p>',
        unsafe_allow_html=True,
    )

    # Sélecteurs
    c1, c2 = st.columns(2)
    with c1:
        period_gen = st.selectbox(
            "Période modèle général",
            list(PERIODS_LABELS.keys()),
            format_func=lambda x: PERIODS_LABELS[x],
            key="brent_period",
        )
    with c2:
        model_type = st.selectbox(
            "Modèle",
            ["simple", "fama_french"],
            format_func=lambda x: {
                "simple":      "Simple (Score + Secteur)",
                "fama_french": "Fama-French (Score + Secteur + Taille + B/M)",
            }[x],
            key="brent_model",
        )

    dep_gen = f"Rendement_{period_gen}"

    # ── Calcul rendement pendant hausse Brent ─────────────────────────────
    with st.spinner("Calcul des rendements pendant les hausses Brent..."):
        tickers_valid = valid["ticker"].dropna().tolist()
        rdt_brent     = calc_rendement_brent(prices, tickers_valid, rallies)

    valid2 = valid.copy()
    valid2["Rendement_Brent"] = valid2["ticker"].map(rdt_brent)
    n_brent = valid2["Rendement_Brent"].notna().sum()

    st.markdown(f"""<div class="note-box">
        Rendement cumulé calculé sur <b>{len(rallies)} période(s) de hausse Brent</b>
        détectées · <b>{n_brent}</b> entreprises avec données suffisantes (≥10 jours de cotation dans les fenêtres)<br>
        Périodes : {" · ".join([f"{s.strftime('%b %Y')} → {e.strftime('%b %Y')}" for s,e in rallies])}
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    

    # ── Régressions ───────────────────────────────────────────────────────
    base_cols = [score_col, secteur_col, quintile_col]
    if model_type == "fama_french":
        base_cols += ["LogMarketCap", "BookToMarket"]

    def run_ols(df_in, dep_var, label):
        df_r = df_in[[c for c in base_cols + [dep_var] if c in df_in.columns]].dropna().copy()
        df_r[dep_var] = winsorize(df_r[dep_var])
        df_r = df_r.rename(columns={score_col: "Score_principal"})
        if model_type == "fama_french":
            formula = f"{dep_var} ~ Score_principal + C({secteur_col}) + LogMarketCap + BookToMarket"
        else:
            formula = f"{dep_var} ~ Score_principal + C({secteur_col})"
        m = smf.ols(formula, data=df_r).fit(cov_type="HC3")
        return m, len(df_r)

    model_gen,   n_gen   = run_ols(valid2, dep_gen,             "Modèle général")
    model_brent, n_brent2 = run_ols(valid2, "Rendement_Brent", "Pendant hausse Brent")

    # ── Tableau comparatif des coefficients clés ──────────────────────────
    st.markdown('<p class="section-title">Comparaison des coefficients</p>',
                unsafe_allow_html=True)

    coef_gen   = model_gen.params.get("Score_principal", np.nan)
    pval_gen   = model_gen.pvalues.get("Score_principal", np.nan)
    r2_gen     = model_gen.rsquared_adj

    coef_br    = model_brent.params.get("Score_principal", np.nan)
    pval_br    = model_brent.pvalues.get("Score_principal", np.nan)
    r2_br      = model_brent.rsquared_adj

    delta      = coef_br - coef_gen

    # Métriques côte à côte
    col1, col2, col3, col4 = st.columns(4)
    for col, lbl, val, ok in [
        (col1, f"Coef. Score — Général ({PERIODS_LABELS[period_gen]})",
         f"{coef_gen:+.3f}  {sig_stars(pval_gen)}", pval_gen < 0.05),
        (col2, "Coef. Score — Pendant Brent-up",
         f"{coef_br:+.3f}  {sig_stars(pval_br)}", pval_br < 0.05),
        (col3, "Δ Coefficient (Brent − Général)",
         f"{delta:+.3f}", delta > 0),
        (col4, "R² ajusté (général / Brent)",
         f"{r2_gen:.3f} / {r2_br:.3f}", True),
    ]:
        metric_card(col, lbl, val, ok)

    # Interprétation automatique
    if abs(delta) > 0.05 and pval_br < 0.1:
        msg = (f"Le score {score_label} a un effet {'plus fort' if delta > 0 else 'plus faible'} "
               f"pendant les hausses Brent (Δ = {delta:+.3f}). "
               f"{'Les entreprises mieux notées surperforment davantage pendant les chocs pétroliers.' if delta > 0 else 'Les entreprises mieux notées sont moins sensibles aux hausses pétrolières — effet protecteur.'}")
        st.markdown(f'<div class="note-box">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">Le coefficient du score ne change pas significativement pendant les hausses Brent (Δ = {delta:+.3f}). Le score climatique ne modifie pas la sensibilité aux chocs pétroliers.</div>',
                    unsafe_allow_html=True)
    st.markdown("")

    # ── Graphique comparaison coefficients par secteur ────────────────────
    st.markdown('<p class="section-title">Coefficients par secteur — général vs Brent-up</p>',
                unsafe_allow_html=True)

    secteurs = sorted(valid2[secteur_col].dropna().unique().tolist())
    ref      = secteurs[0]

    def get_sector_effects(model):
        effects = {}
        for s in secteurs:
            if s == ref:
                effects[s] = model.params.get("Score_principal", np.nan)
            else:
                k = f"Score_principal:C({secteur_col})[T.{s}]"
                base = model.params.get("Score_principal", np.nan)
                effects[s] = base + model.params.get(k, 0) if k in model.params else base
        return effects

    # Modèle interaction pour avoir les effets par secteur
    def run_ols_interaction(df_in, dep_var):
        df_r = df_in[[c for c in [score_col, secteur_col, dep_var]
                      if c in df_in.columns]].dropna().copy()
        df_r[dep_var] = winsorize(df_r[dep_var])
        df_r = df_r.rename(columns={score_col: "Score_principal"})
        formula = f"{dep_var} ~ Score_principal * C({secteur_col})"
        try:
            return smf.ols(formula, data=df_r).fit(cov_type="HC3")
        except:
            return None

    m_gen_int   = run_ols_interaction(valid2, dep_gen)
    m_brent_int = run_ols_interaction(valid2, "Rendement_Brent")

    if m_gen_int and m_brent_int:
        eff_gen   = get_sector_effects(m_gen_int)
        eff_brent = get_sector_effects(m_brent_int)

        df_plot = pd.DataFrame({
            "Secteur":  secteurs,
            "Général":  [eff_gen.get(s, np.nan) for s in secteurs],
            "Brent-up": [eff_brent.get(s, np.nan) for s in secteurs],
        }).dropna().sort_values("Général")

        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            name=f"Général ({PERIODS_LABELS[period_gen]})",
            x=df_plot["Secteur"], y=df_plot["Général"],
            marker_color="rgba(99,179,237,0.8)",
        ))
        fig_s.add_trace(go.Bar(
            name="Pendant hausse Brent",
            x=df_plot["Secteur"], y=df_plot["Brent-up"],
            marker_color="rgba(251,191,36,0.8)",
        ))
        fig_s.add_hline(y=0, line_width=0.8, line_color="rgba(255,255,255,0.3)")
        fig_s.update_layout(
            **PLOTLY_LAYOUT, height=380, barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Coefficient du score", gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # ── Comparaison par quintile ───────────────────────────────────────────
    st.markdown('<p class="section-title">Rendement moyen par quintile — général vs Brent-up</p>',
                unsafe_allow_html=True)

    rows_q = []
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        sub = valid2[valid2[quintile_col] == q]
        if sub.empty:
            continue
        rows_q.append({
            "Quintile": q,
            "N": len(sub),
            f"Rdt {PERIODS_LABELS[period_gen]} (moy.)": f"{sub[dep_gen].mean():.1%}" if dep_gen in sub else "N/A",
            "Rdt pendant Brent-up (moy.)": f"{sub['Rendement_Brent'].mean():.1%}",
            "Δ (Brent − Général)": f"{(sub['Rendement_Brent'].mean() - sub[dep_gen].mean()):+.1%}" if dep_gen in sub else "N/A",
        })

    if rows_q:
        st.dataframe(pd.DataFrame(rows_q), use_container_width=True, hide_index=True)

        # Graphique quintiles
        df_q = pd.DataFrame(rows_q)
        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            name=f"Général ({PERIODS_LABELS[period_gen]})",
            x=df_q["Quintile"],
            y=[float(v.strip("%"))/100 for v in df_q[f"Rdt {PERIODS_LABELS[period_gen]} (moy.)"]],
            marker_color=[QUINTILE_COLORS[q] for q in df_q["Quintile"]],
            opacity=0.5,
        ))
        fig_q.add_trace(go.Bar(
            name="Pendant hausse Brent",
            x=df_q["Quintile"],
            y=[float(v.strip("%"))/100 for v in df_q["Rdt pendant Brent-up (moy.)"]],
            marker_color=[QUINTILE_COLORS[q] for q in df_q["Quintile"]],
            marker_line=dict(width=2, color="white"),
        ))
        fig_q.add_hline(y=0, line_width=0.8, line_color="rgba(255,255,255,0.3)")
        fig_q.update_layout(
            **PLOTLY_LAYOUT, height=320, barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            yaxis=dict(tickformat=".0%", title="Rendement moyen",
                       gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_q, use_container_width=True)

    # Résultats complets
    with st.expander("Résultats complets — modèle général"):
        st.text(model_gen.summary().as_text())
    with st.expander("Résultats complets — pendant hausse Brent"):
        st.text(model_brent.summary().as_text())


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING PAGES
# ══════════════════════════════════════════════════════════════════════════════

if page == "Accueil":

    if is_mq:
        total  = len(df_mq)
        avec   = len(valid_mq)
        avec_f = valid_mq['MarketCap'].notna().sum() if 'MarketCap' in valid_mq.columns else 0
        st.markdown("# TPI Management Quality · Analyse Financière")
        st.markdown("**Édition 2025** · Données boursières 2023–2025")
    else:
        total  = len(df_act)
        avec   = len(valid_act)
        avec_f = valid_act['MarketCap'].notna().sum() if 'MarketCap' in valid_act.columns else 0
        st.markdown("# ACT — Assessing low Carbon Transition · Analyse Financière")
        st.markdown("**Évaluation 2025** · Données boursières 2023–2025")

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,sub in [
        (c1, "Entreprises (total)",      f"{total:,}",  "Panel complet"),
        (c2, "Avec données boursières",  f"{avec:,}",   f"{avec/total:.0%} du panel"),
        (c3, "Avec fondamentaux",        f"{avec_f:,}", "MarketCap + Book/Market"),
        (c4, "Hausses Brent détectées",  f"{len(rallies)}", "Seuil +15% / 60j"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:#86efac">{val}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Méthodologie</p>', unsafe_allow_html=True)
    if is_mq:
        st.markdown(f"""
- **Score MQ** : moyenne des réponses Oui/Non aux 23 questions (Q1L0 → Q23L5)
- **Quintiles** : calculés sur le score global continu, entreprises cotées uniquement ({avec} entreprises)
- **Régression OLS** : winsorisée au 1er–99ème percentile · erreurs robustes HC3
- **⚠ Biais de sélection** : analyse limitée aux entreprises cotées ({avec/total:.0%} du panel TPI MQ)
""")
    else:
        st.markdown(f"""
- **Score ACT** : Performance Score /100 — évaluation de la trajectoire de décarbonation
- **Narrative Score** : note A (meilleur) à E — évaluation qualitative
- **Trend Score** : tendance +/=/− de la performance dans le temps
- **Quintiles** : calculés sur le Performance Score, entreprises cotées uniquement ({avec} entreprises)
- **Régression OLS** : winsorisée au 1er–99ème percentile · erreurs robustes HC3
- **⚠ Biais de sélection** : analyse limitée aux entreprises cotées ({avec/total:.0%} du panel ACT)
""")

    st.markdown('<p class="section-title">Périodes de hausse Brent détectées</p>', unsafe_allow_html=True)
    if rallies:
        rally_df = pd.DataFrame(rallies, columns=['Début','Fin'])
        rally_df['Durée (jours)'] = (rally_df['Fin']-rally_df['Début']).dt.days
        rally_df['Début'] = rally_df['Début'].dt.strftime('%d/%m/%Y')
        rally_df['Fin']   = rally_df['Fin'].dt.strftime('%d/%m/%Y')
        st.dataframe(rally_df, use_container_width=True, hide_index=True)

elif page == "Société":
    if is_mq:
        def mq_extra(row):
            return (f"Niveau : {row.get('Level','N/A')}<br>"
                    f"Quintile : {row.get('Quintile_MQ','N/A')}<br>"
                    f"Secteur : {row.get('Sector','N/A')}<br>"
                    f"Macro-secteur : {row.get('Macro_Secteur','N/A')}<br>"
                    f"Géographie : {row.get('Geography','N/A')}")
        page_societe(
            valid=valid_mq, prices=prices_mq, brent=brent, rallies=rallies,
            score_col='Score_global_MQ', score_label='Score global MQ',
            quintile_col='Quintile_MQ', pct_col='MQ_percentile',
            secteur_col='Macro_Secteur', extra_info_fn=mq_extra,
        )
    else:
        col_I_name = df_act.columns[5]
        col_J_name = df_act.columns[6]
        col_K_name = df_act.columns[7]
        def act_extra(row):
            return (f"Narrative : {row.get(col_J_name,'N/A')}<br>"
                    f"Trend : {row.get(col_K_name,'N/A')}<br>"
                    f"Secteur : {row.get(col_secteur_act,'N/A')}<br>"
                    f"Méthodologie : {row.get(df_act.columns[4],'N/A')}<br>"
                    f"Quintile : {row.get('Quintile_ACT','N/A')}")
        page_societe(
            valid=valid_act, prices=prices_act, brent=brent, rallies=rallies,
            score_col=col_score_act, score_label='Performance Score /100',
            quintile_col='Quintile_ACT', pct_col='Score_percentile',
            secteur_col=col_secteur_act, extra_info_fn=act_extra,
        )

elif page == "Panel Quintiles":
    if is_mq:
        page_panel(valid_mq, prices_mq, brent, rallies, 'Quintile_MQ')
    else:
        page_panel(valid_act, prices_act, brent, rallies, 'Quintile_ACT')

    # Tableau résumé
    st.markdown('<p class="section-title">Résumé statistique par quintile</p>', unsafe_allow_html=True)
    valid  = valid_mq if is_mq else valid_act
    q_col  = 'Quintile_MQ' if is_mq else 'Quintile_ACT'
    s_col  = 'Score_global_MQ' if is_mq else col_score_act
    rows_t = []
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = valid[valid[q_col]==q]
        if sub.empty: continue
        rows_t.append({
            'Quintile': q, 'N': len(sub),
            'Score moyen': f"{sub[s_col].mean():.1f}" if not is_mq else f"{sub[s_col].mean():.1%}",
            'Rdt 2023':  f"{sub['Rendement_2023'].mean():.1%}",
            'Rdt 2024':  f"{sub['Rendement_2024'].mean():.1%}",
            'Rdt 2025':  f"{sub['Rendement_2025'].mean():.1%}",
            'Rdt 23–25': f"{sub['Rendement_2023_2025'].mean():.1%}",
            'Vol':       f"{sub['Volatilite_2023_2025'].mean():.1%}",
            'Sharpe':    f"{sub['Sharpe_2023_2025'].mean():.2f}",
            'MDD':       f"{sub['MaxDrawdown_2023_2025'].mean():.1%}",
        })
    st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)

elif page == "Analyse Brent":
    if is_mq:
        page_brent(
            valid=valid_mq, prices=prices_mq, rallies=rallies,
            score_col="Score_global_MQ", score_label="Score MQ",
            secteur_col="Macro_Secteur", quintile_col="Quintile_MQ",
            total_panel=len(df_mq), dataset_label="Management Quality",
        )
    else:
        page_brent(
            valid=valid_act, prices=prices_act, rallies=rallies,
            score_col=col_score_act, score_label="Performance Score ACT",
            secteur_col=col_secteur_act, quintile_col="Quintile_ACT",
            total_panel=len(df_act), dataset_label="ACT — Transition Carbone",
        )

elif page == "Régression OLS":

    if is_mq:
        page_ols(
            valid=valid_mq,
            score_col='Score_global_MQ', score_label='Score global MQ',
            secteur_col='Macro_Secteur',
            total_panel=len(df_mq),
            quintile_col='Quintile_MQ',
        )
    else:
        page_ols(
            valid=valid_act,
            score_col=col_score_act, score_label='Performance Score ACT /100',
            secteur_col=col_secteur_act,
            total_panel=len(df_act),
            quintile_col='Quintile_ACT',
        )


# In[ ]:




