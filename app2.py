#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
MQ Financial Analysis — Interface Streamlit
============================================
Prérequis : avoir exécuté mq_pipeline.py
Lancement  : streamlit run mq_app.py
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
    page_title="TPI MQ · Analyse Financière",
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
.mq-badge { display: inline-block; padding: 6px 14px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500; }
.section-title { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #6b7280; margin: 24px 0 12px; border-bottom: 1px solid #2a2d35; padding-bottom: 6px; }
.note-box { background: #0f1a2e; border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 10px 14px; font-size: 12px; color: #93c5fd; margin-top: 8px; line-height: 1.6; }
.warn-box { background: #1c1a0e; border: 1px solid #713f12; border-radius: 6px;
    padding: 10px 14px; font-size: 12px; color: #fde68a; margin-top: 8px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── CHEMINS ───────────────────────────────────────────────────────────────────
LOCAL_DIR  = '/Users/hamidi/Desktop/'
METRIQUES_F = 'mq_metriques.parquet'
PRIX_F      = 'mq_prix_journaliers.parquet'

# ⚠ Mettre à jour avec ton URL GitHub Release après upload
GITHUB_RELEASE_URL = "https://github.com/Ihssane-Hamidi/TPI_MQ/releases/download/v1.0/"

def get_parquet(filename):
    local = os.path.join(LOCAL_DIR, filename)
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
    '2023': '2023', '2024': '2024',
    '2025': '2025', '2023_2025': '2023–2025',
}
QUINTILE_COLORS = {
    'Q1': '#ef4444', 'Q2': '#f97316', 'Q3': '#a3a3a3',
    'Q4': '#86efac', 'Q5': '#16a34a',
}
PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── UTILITAIRES ───────────────────────────────────────────────────────────────
@st.cache_data
def load_metriques():
    return pd.read_parquet(get_parquet(METRIQUES_F))

@st.cache_data
def load_prix():
    px = pd.read_parquet(get_parquet(PRIX_F))
    px.index = pd.to_datetime(px.index)
    return px

@st.cache_data
def load_brent():
    raw = yf.download('BZ=F', start='2023-01-01', end='2025-10-31',
                      auto_adjust=True, progress=False)
    b = raw['Close'].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
    b.index = pd.to_datetime(b.index)
    return b.dropna()

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
            annotation_text="↑ Brent" if (not first_only or i == 0) else "",
            annotation_position="top left",
            annotation_font_size=9, annotation_font_color='#93c5fd',
        )
    return fig

def mq_color(pct):
    if pct >= 0.67:   return '#14532d', '#86efac', 'Score élevé'
    elif pct >= 0.33: return '#713f12', '#fde68a', 'Score moyen'
    else:             return '#7f1d1d', '#fca5a5', 'Score faible'

def winsorize(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))

def sig_stars(p):
    if p < 0.01: return '★★★'
    if p < 0.05: return '★★'
    if p < 0.10: return '★'
    return 'ns'

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
try:
    df     = load_metriques()
    prices = load_prix()
    brent  = load_brent()
except Exception as e:
    st.error(f"Erreur chargement : {e}\nLancez d'abord mq_pipeline.py")
    st.stop()

rallies   = detect_oil_rallies(brent)
valid     = df[df['ticker'].notna() & (df['ticker'] != 'None')
               & df['Rendement_2023_2025'].notna()].copy()
companies = sorted(valid['Company Name'].unique().tolist())

total_mq  = len(df)
avec_fin  = len(valid)
avec_fund = valid['MarketCap'].notna().sum() if 'MarketCap' in valid.columns else 0

# ── NAVIGATION ────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 TPI MQ · Finance")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Société", "Panel Quintiles", "Régression OLS"],
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "Accueil":

    st.markdown("# TPI Management Quality · Analyse Financière")
    st.markdown("**Édition 2025** · Données boursières 2023–2025")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "Entreprises MQ 2025",     f"{total_mq:,}",  "Panel total"),
        (c2, "Avec données boursières", f"{avec_fin:,}",  f"{avec_fin/total_mq:.0%} du panel"),
        (c3, "Avec fondamentaux",       f"{avec_fund:,}", "MarketCap + Book/Market"),
        (c4, "Hausses Brent détectées", f"{len(rallies)}", "Seuil +15% / 60j"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:#86efac">{val}</div>
            <div style="font-size:11px;color:#6b7280;margin-top:4px">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Méthodologie</p>', unsafe_allow_html=True)
    st.markdown(f"""
- **Score MQ** : moyenne des réponses Oui/Non aux 23 questions (Q1L0 → Q23L5)
- **Quintiles** : calculés sur les entreprises cotées uniquement ({avec_fin} entreprises)
- **Hausses Brent** : +15% sur 60 jours ouvrés glissants, fin quand recul de 8% depuis le pic
- **Régression OLS** : winsorisée au 1er–99ème percentile · erreurs robustes HC3
- **⚠ Biais de sélection** : analyse limitée aux entreprises cotées ({avec_fin/total_mq:.0%} du panel TPI)
""")

    st.markdown('<p class="section-title">Périodes de hausse Brent détectées</p>',
                unsafe_allow_html=True)
    if rallies:
        rally_df = pd.DataFrame(rallies, columns=['Début', 'Fin'])
        rally_df['Durée (jours)'] = (rally_df['Fin'] - rally_df['Début']).dt.days
        rally_df['Début'] = rally_df['Début'].dt.strftime('%d/%m/%Y')
        rally_df['Fin']   = rally_df['Fin'].dt.strftime('%d/%m/%Y')
        st.dataframe(rally_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SOCIÉTÉ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Société":

    st.sidebar.markdown('<p class="section-title">Sélection</p>', unsafe_allow_html=True)
    company_name = st.sidebar.selectbox("Entreprise", companies, label_visibility="collapsed")

    row    = valid[valid['Company Name'] == company_name].iloc[0]
    ticker = row['ticker']

    bg, fg, label = mq_color(float(row['MQ_percentile']))
    st.sidebar.markdown('<p class="section-title">Score MQ</p>', unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Score global</div>
        <div class="metric-value" style="color:{fg}">{row['Score_global_MQ']:.1%}</div>
        <div style="margin-top:8px">
            <span class="mq-badge" style="background:{bg};color:{fg}">
                {label} · {row['MQ_percentile']:.0%} du panel
            </span>
        </div>
        <div style="margin-top:10px;font-size:12px;color:#6b7280;line-height:1.9">
            Niveau : {row['Level']}<br>
            Quintile : {row['Quintile_MQ']}<br>
            Secteur : {row['Sector']}<br>
            Macro-secteur : {row['Macro_Secteur']}<br>
            Géographie : {row['Geography']}
        </div>
    </div>""", unsafe_allow_html=True)

    if ticker not in prices.columns:
        st.warning(f"Prix non disponibles pour {ticker}")
        st.stop()

    px = prices[ticker].dropna()
    if len(px) < 50:
        st.warning(f"Données insuffisantes pour {ticker} ({len(px)} jours)")
        st.stop()

    ret_daily = px.pct_change().dropna()

    # Graphique rendements + Brent axe secondaire
    st.markdown(f"### {company_name}  ·  `{ticker}`")
    st.markdown('<p class="section-title">Rendements journaliers · Zones bleues = hausses Brent</p>',
                unsafe_allow_html=True)

    brent_al   = brent.reindex(ret_daily.index, method='ffill').dropna()
    brent_norm = brent_al / brent_al.iloc[0] * 100

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1 = add_oil_rectangles(fig1, rallies, first_only=True)
    fig1.add_hline(y=0, line_width=0.5, line_color='rgba(255,255,255,0.15)')

    bar_col = ['rgba(239,68,68,0.8)' if v < 0 else 'rgba(52,211,153,0.8)'
               for v in ret_daily.values]
    fig1.add_trace(go.Bar(
        x=ret_daily.index, y=ret_daily.values,
        marker_color=bar_col, name='Rendement journalier',
        hovertemplate='%{x|%d %b %Y}<br>%{y:.2%}<extra></extra>',
    ), secondary_y=False)
    fig1.add_trace(go.Scatter(
        x=brent_norm.index, y=brent_norm.values,
        name='Brent (base 100)',
        line=dict(color='rgba(59,130,246,0.7)', width=1.5),
        hovertemplate='%{x|%d %b %Y}<br>Brent: %{y:.1f}<extra></extra>',
    ), secondary_y=True)

    fig1.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=True,
                       legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0))
    fig1.update_yaxes(tickformat='.1%', title_text='Rendement',
                      gridcolor='rgba(255,255,255,0.05)', secondary_y=False)
    fig1.update_yaxes(title_text='Brent (base 100)',
                      gridcolor='rgba(0,0,0,0)', secondary_y=True)
    st.plotly_chart(fig1, use_container_width=True)

    # Métriques par période
    st.markdown('<p class="section-title">Performance par période</p>', unsafe_allow_html=True)
    periods    = list(PERIODS_LABELS.keys())
    labs       = list(PERIODS_LABELS.values())
    rendements = [row.get(f'Rendement_{p}', np.nan) for p in periods]
    vols       = [row.get(f'Volatilite_{p}', np.nan) for p in periods]
    sharpes    = [row.get(f'Sharpe_{p}', np.nan) for p in periods]

    r_col = ['rgba(239,68,68,0.8)' if (v and v < 0) else 'rgba(52,211,153,0.8)' for v in rendements]
    s_col = ['rgba(239,68,68,0.8)' if (v and v < 0) else 'rgba(251,191,36,0.8)' for v in sharpes]

    fig2 = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                         subplot_titles=('Rendement', 'Volatilité annualisée', 'Sharpe'))
    fig2.add_trace(go.Bar(x=labs, y=rendements, marker_color=r_col,
                          hovertemplate='%{x}<br>%{y:.1%}<extra></extra>'), row=1, col=1)
    fig2.add_trace(go.Bar(x=labs, y=vols, marker_color='rgba(99,179,237,0.7)',
                          hovertemplate='%{x}<br>%{y:.1%}<extra></extra>'), row=1, col=2)
    fig2.add_trace(go.Bar(x=labs, y=sharpes, marker_color=s_col,
                          hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'), row=1, col=3)
    fig2.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
    fig2.update_yaxes(tickformat='.0%', row=1, col=1)
    fig2.update_yaxes(tickformat='.0%', row=1, col=2)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PANEL QUINTILES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Panel Quintiles":

    st.markdown("### Panel · Rendements cumulés par quintile MQ")
    st.markdown('<p class="section-title">Q1 (scores faibles) → Q5 (scores élevés) · Zones bleues = hausses Brent</p>',
                unsafe_allow_html=True)

    # Brent normalisé pour axe secondaire
    brent_norm_panel = brent / brent.iloc[0] * 100

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3 = add_oil_rectangles(fig3, rallies, first_only=True)
    fig3.add_hline(y=0, line_width=0.5, line_color='rgba(255,255,255,0.15)')

    for q in ['Q1','Q2','Q3','Q4','Q5']:
        tickers_q = valid[valid['Quintile_MQ'] == q]['ticker'].dropna().tolist()
        tickers_q = [t for t in tickers_q if t in prices.columns]
        if not tickers_q: continue

        # Rendement cumulé par ticker puis moyenne — plus robuste
        cum_par_ticker = (1 + prices[tickers_q].pct_change()).cumprod() - 1
        cum_r = cum_par_ticker.mean(axis=1).dropna()

        fig3.add_trace(go.Scatter(
            x=cum_r.index, y=cum_r.values,
            name=f'{q} (n={len(tickers_q)})',
            line=dict(color=QUINTILE_COLORS[q], width=2),
            hovertemplate='%{x|%d %b %Y}<br>%{y:.1%}<extra>' + q + '</extra>',
        ), secondary_y=False)

    fig3.add_trace(go.Scatter(
        x=brent_norm_panel.index, y=brent_norm_panel.values,
        name='Brent (base 100)',
        line=dict(color='rgba(59,130,246,0.5)', width=1.5, dash='dot'),
        hovertemplate='%{x|%d %b %Y}<br>Brent: %{y:.1f}<extra></extra>',
    ), secondary_y=True)

    fig3.update_layout(
        **PLOTLY_LAYOUT, height=480,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
    )
    fig3.update_yaxes(tickformat='.0%', title_text='Rendement cumulé',
                      gridcolor='rgba(255,255,255,0.05)', secondary_y=False)
    fig3.update_yaxes(title_text='Brent (base 100)',
                      gridcolor='rgba(0,0,0,0)', secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

    # Tableau résumé
    st.markdown('<p class="section-title">Résumé statistique par quintile</p>', unsafe_allow_html=True)
    rows = []
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = valid[valid['Quintile_MQ'] == q]
        rows.append({
            'Quintile': q, 'N': len(sub),
            'Score MQ': f"{sub['Score_global_MQ'].mean():.1%}",
            'Rdt 2023': f"{sub['Rendement_2023'].mean():.1%}",
            'Rdt 2024': f"{sub['Rendement_2024'].mean():.1%}",
            'Rdt 2025': f"{sub['Rendement_2025'].mean():.1%}",
            'Rdt 23–25': f"{sub['Rendement_2023_2025'].mean():.1%}",
            'Vol':      f"{sub['Volatilite_2023_2025'].mean():.1%}",
            'Sharpe':   f"{sub['Sharpe_2023_2025'].mean():.2f}",
            'MDD':      f"{sub['MaxDrawdown_2023_2025'].mean():.1%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RÉGRESSION OLS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Régression OLS":

    st.markdown("### Régression OLS · Score MQ → Performance financière")

    # ── Sélecteurs ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        period_choice = st.selectbox("Période",
            options=list(PERIODS_LABELS.keys()),
            format_func=lambda x: PERIODS_LABELS[x])
    with c2:
        dep_choice = st.selectbox("Variable expliquée",
            options=['Rendement','Volatilite','Sharpe','MaxDrawdown'],
            format_func=lambda x: {'Rendement':'Rendement','Volatilite':'Volatilité',
                                    'Sharpe':'Sharpe','MaxDrawdown':'Max Drawdown'}[x])
    with c3:
        model_choice = st.selectbox("Modèle",
            options=['simple','interaction','fama_french'],
            format_func=lambda x: {
                'simple':       'Modèle 1 · Simple (MQ + Secteur)',
                'interaction':  'Modèle 2 · Interaction MQ × Secteur',
                'fama_french':  'Modèle 3 · Fama-French (MQ + Secteur + Taille + B/M)',
            }[x])

    dep_var = f'{dep_choice}_{period_choice}'

    # ── Préparation données ───────────────────────────────────────────────────
    base_cols = ['Company Name','Score_global_MQ','Macro_Secteur','Quintile_MQ', dep_var]
    if model_choice == 'fama_french':
        base_cols += ['LogMarketCap','BookToMarket']

    df_ols = valid[base_cols].dropna().copy()
    df_ols[dep_var] = winsorize(df_ols[dep_var])  # winsorisation systématique

    n_total   = len(valid)
    n_ols     = len(df_ols)
    pct_panel = n_ols / total_mq

    if n_ols < 30:
        st.warning("Pas assez d'observations après winsorisation.")
        st.stop()

    # ── Formules ──────────────────────────────────────────────────────────────
    formulas = {
        'simple':      f'{dep_var} ~ Score_global_MQ + C(Macro_Secteur)',
        'interaction': f'{dep_var} ~ Score_global_MQ * C(Macro_Secteur)',
        'fama_french': f'{dep_var} ~ Score_global_MQ + C(Macro_Secteur) + LogMarketCap + BookToMarket',
    }
    formula = formulas[model_choice]
    model   = smf.ols(formula, data=df_ols).fit(cov_type='HC3')

    coef   = model.params['Score_global_MQ']
    pval   = model.pvalues['Score_global_MQ']
    r2_adj = model.rsquared_adj
    n      = int(model.nobs)

    # ── Note méthodologique ───────────────────────────────────────────────────
    st.markdown(f"""<div class="note-box">
        Régression estimée sur <b>{n_ols}</b> entreprises ({pct_panel:.0%} du panel MQ total de {total_mq})
        après winsorisation au 1er–99ème percentile · Erreurs robustes HC3<br>
        ⚠ <b>Biais de sélection</b> : les entreprises non cotées sont exclues — les résultats
        ne sont pas généralisables à l'ensemble du panel TPI.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    # ── Métriques clés ────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, lbl, val, ok in [
        (mc1, 'Coefficient MQ',  f'{coef:+.3f}',              coef > 0),
        (mc2, 'p-value',         f'{pval:.3f} {sig_stars(pval)}', pval < 0.05),
        (mc3, 'R² ajusté',       f'{r2_adj:.3f}',             True),
        (mc4, 'Observations',    str(n),                       True),
    ]:
        color = '#86efac' if ok else '#fca5a5'
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    # ── Graphique coefficients MQ par secteur (modèle interaction) ────────────
    if model_choice == 'interaction':
        st.markdown('<p class="section-title">Effet du score MQ par macro-secteur</p>',
                    unsafe_allow_html=True)

        st.markdown("""<div class="note-box">
            Le coefficient MQ global est l'effet dans le secteur de référence (Consumer).<br>
            L'effet dans chaque autre secteur = coefficient MQ global + terme d'interaction du secteur.
        </div>""", unsafe_allow_html=True)
        st.markdown("")

        # Secteur de référence (absent des dummies = premier alphabétique)
        secteurs = sorted(df_ols['Macro_Secteur'].unique().tolist())
        ref      = secteurs[0]

        # Reconstruction des coefficients par secteur
        coef_par_secteur = []
        for s in secteurs:
            if s == ref:
                c   = model.params.get('Score_global_MQ', np.nan)
                se  = model.bse.get('Score_global_MQ', np.nan)
                p   = model.pvalues.get('Score_global_MQ', np.nan)
            else:
                inter_key = f'Score_global_MQ:C(Macro_Secteur)[T.{s}]'
                if inter_key in model.params:
                    c  = model.params['Score_global_MQ'] + model.params[inter_key]
                    se = model.bse[inter_key]
                    p  = model.pvalues[inter_key]
                else:
                    c, se, p = np.nan, np.nan, np.nan
            coef_par_secteur.append({
                'Secteur': s,
                'Coef':    c,
                'SE':      se,
                'pval':    p,
                'sig':     sig_stars(p) if pd.notna(p) else '',
                'n':       len(df_ols[df_ols['Macro_Secteur'] == s]),
            })

        df_coef = pd.DataFrame(coef_par_secteur).dropna(subset=['Coef'])
        df_coef = df_coef.sort_values('Coef')

        bar_colors = ['rgba(52,211,153,0.8)' if c > 0 else 'rgba(239,68,68,0.8)'
                      for c in df_coef['Coef']]
        error_x = dict(
            type='data',
            array=1.96 * df_coef['SE'],
            color='rgba(255,255,255,0.4)',
            thickness=1.5, width=4,
        )

        fig_coef = go.Figure(go.Bar(
            x=df_coef['Coef'],
            y=df_coef['Secteur'],
            orientation='h',
            marker_color=bar_colors,
            error_x=error_x,
            text=[f"{c:+.3f} {s} (n={n})"
                  for c, s, n in zip(df_coef['Coef'], df_coef['sig'], df_coef['n'])],
            textposition='outside',
            hovertemplate='%{y}<br>Coef MQ: %{x:+.3f}<extra></extra>',
        ))
        fig_coef.add_vline(x=0, line_width=1, line_color='rgba(255,255,255,0.3)')
        fig_coef.update_layout(
            **PLOTLY_LAYOUT, height=400, showlegend=False,
            xaxis=dict(title='Coefficient MQ (effet sur ' + dep_choice + ')',
                       gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown("""<div class="warn-box">
            ★★★ p&lt;0.01 · ★★ p&lt;0.05 · ★ p&lt;0.10 · ns = non significatif<br>
            Les barres d'erreur représentent l'intervalle de confiance à 95%.
            Un secteur avec peu d'observations (n faible) aura un intervalle large — interpréter avec précaution.
        </div>""", unsafe_allow_html=True)

    # ── Scatter MQ vs variable dépendante ─────────────────────────────────────
    st.markdown(f'<p class="section-title">Score MQ vs {dep_choice} · coloré par quintile</p>',
                unsafe_allow_html=True)

    x_rng = np.linspace(df_ols['Score_global_MQ'].min(),
                        df_ols['Score_global_MQ'].max(), 100)
    y_fit = model.params['Intercept'] + model.params['Score_global_MQ'] * x_rng

    fig4 = go.Figure()
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = df_ols[df_ols['Quintile_MQ'] == q]
        if sub.empty: continue
        fig4.add_trace(go.Scatter(
            x=sub['Score_global_MQ'], y=sub[dep_var],
            mode='markers', name=q,
            marker=dict(color=QUINTILE_COLORS[q], size=6, opacity=0.6),
            text=sub['Company Name'],
            hovertemplate='%{text}<br>Score MQ: %{x:.1%}<br>'
                          + dep_choice + ': %{y:.3f}<extra>' + q + '</extra>',
        ))
    fig4.add_trace(go.Scatter(
        x=x_rng, y=y_fit, mode='lines', name='OLS fit',
        line=dict(color='rgba(251,191,36,0.9)', width=2, dash='dot'),
    ))
    fig4.update_layout(
        **PLOTLY_LAYOUT, height=420,
        xaxis=dict(tickformat='.0%', title='Score global MQ',
                   gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title=f'{dep_choice} · {PERIODS_LABELS[period_choice]}',
                   gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Résultats complets ────────────────────────────────────────────────────
    with st.expander("Résultats complets de la régression"):
        st.text(model.summary().as_text())


# In[ ]:




