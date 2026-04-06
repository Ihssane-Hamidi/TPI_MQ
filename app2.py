#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import plotly.express as px



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

/* 1. Reset global plus petit */
html, body, [class*="st-"], .stMarkdown { 
    font-family: 'IBM Plex Sans', sans-serif; 
    font-size: 13px; 
}

/* 2. Tes cartes métriques */
.metric-card {
   background: #0f1117; border: 1px solid #2a2d35;
   border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
}
.metric-label { 
    font-size: 10px; color: #6b7280; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 2px; 
}
.metric-value { 
    font-family: 'IBM Plex Mono', monospace; font-size: 19px; font-weight: 500; 
}

/* 3. Ajustement des titres de section */
.section-title { 
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #6b7280; margin: 16px 0 8px; border-bottom: 1px solid #2a2d35; padding-bottom: 4px; 
}

/* 4. Les badges et boîtes */
.score-badge { padding: 4px 10px; font-size: 11px; }
.note-box, .warn-box { font-size: 11px; padding: 8px 12px; }
.dataset-badge-mq, .dataset-badge-act { font-size: 11px; padding: 2px 10px; }
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
   local = '/Users/hamidi/Desktop/brent.parquet'
   url   = GITHUB_RELEASE_URL + 'brent.parquet'
   
   # 1. Fichier local
   if os.path.exists(local):
       df = pd.read_parquet(local)
       return df['Close']
   
   # 2. Fallback GitHub
   try:
       r = requests.get(url, stream=True)
       r.raise_for_status()
       with open('brent.parquet', 'wb') as f:
           for chunk in r.iter_content(chunk_size=8192):
               f.write(chunk)
       df = pd.read_parquet('brent.parquet')
       return df['Close']
   except Exception as e:
       st.error(f"❌ Brent introuvable : {e}")
       return pd.Series(dtype=float)
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
       px_q = prices[tickers_q].copy()
       first_common = px_q.dropna(how='all').index[0]
       px_q = px_q.loc[first_common:].ffill()

       cum_par_ticker = (1 + px_q.pct_change()).cumprod() - 1
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



   
def prepare_ols_data(df, score_col, secteur_col, min_obs=8):
   """Nettoie et stabilise les données pour la régression."""
   df_clean = df.copy()
   
   # 1. Regroupement des secteurs rares (évite les secteurs fantômes)
   counts = df_clean[secteur_col].value_counts()
   rare_sectors = counts[counts < min_obs].index
   if len(rare_sectors) > 0:
       df_clean[secteur_col] = df_clean[secteur_col].replace(rare_sectors, 'Autres/Divers')
   
   # 2. Standardisation du score (réduit le Condition Number)
   # On centre sur 0 avec un écart-type de 1
   std_val = df_clean[score_col].std()
   if std_val > 0:
       df_clean['Score_std'] = (df_clean[score_col] - df_clean[score_col].mean()) / std_val
   else:
       df_clean['Score_std'] = 0
       
   return df_clean

   
def page_ols(valid, score_col, score_label, secteur_col, total_panel, quintile_col):
   """Page OLS complète — Design original + Moteur robuste."""
   
   st.markdown(f"### Régression OLS · {score_label} → Performance financière")

   # ── SÉLECTEURS (Inchangés) ───────────────────────────────────────────
   c1, c2, c3 = st.columns(3)
   with c1:
       period = st.selectbox("Période", list(PERIODS_LABELS.keys()),
                             format_func=lambda x: PERIODS_LABELS[x], key="ols_per")
   with c2:
       dep_choice = st.selectbox("Variable expliquée",
           ['Rendement','Volatilite','Sharpe','MaxDrawdown'],
           format_func=lambda x: {'Rendement':'Rendement','Volatilite':'Volatilité',
                                   'Sharpe':'Sharpe','MaxDrawdown':'Max Drawdown'}[x], key="ols_dep")
   with c3:
       model_choice = st.selectbox("Modèle",
           ['simple','interaction','fama_french'],
           format_func=lambda x: {
               'simple':      'Modèle 1 · Simple (Score + Secteur)',
               'interaction': 'Modèle 2 · Interaction Score × Secteur',
               'fama_french': 'Modèle 3 · Fama-French (Score + Secteur + Taille + B/M)',
           }[x], key="ols_mod")

   dep_var = f'{dep_choice}_{period}'
   
   # ── PRÉPARATION DES DONNÉES (La correction Maths) ────────────────────
   # 1. On filtre les colonnes nécessaires
   cols_needed = [score_col, secteur_col, dep_var, 'LogMarketCap', 'BookToMarket', quintile_col]
   df_ols = valid[[c for c in cols_needed if c in valid.columns]].dropna().copy()
   
   # 2. On stabilise (Regroupement secteurs + Standardisation Score_std)
   df_ols = prepare_ols_data(df_ols, score_col, secteur_col)
   df_ols[dep_var] = winsorize(df_ols[dep_var])
   
   n_ols = len(df_ols)
   n_secteurs = df_ols[secteur_col].nunique()

   # ── SÉCURITÉ ANTI-CRASH (Pour 2023) ──────────────────────────────────
   if n_ols < (n_secteurs + 15):
       st.markdown(f"""<div class="warn-box">
           <b>Données insuffisantes pour {PERIODS_LABELS[period]}</b><br>
           Le modèle nécessite plus d'entreprises ({n_ols} actuelles) pour estimer {n_secteurs} secteurs de façon fiable.
       </div>""", unsafe_allow_html=True)
       return

   # ── VOTRE DESIGN ORIGINAL (Note-box) ────────────────────────────────
   st.markdown(f"""<div class="note-box">
       Régression sur <b>{n_ols}</b> entreprises ({n_ols/total_panel:.0%} du panel initial)<br>
       Méthode : OLS avec erreurs robustes (HC3) · Score standardisé (Z-score) · Secteurs regroupés (N < 8).
   </div>""", unsafe_allow_html=True)
   st.markdown("")

   # ── CONSTRUCTION DE LA FORMULE ──────────────────────────────────────
   # Note : on utilise 'Score_std' au lieu de score_col
   if model_choice == 'interaction':
       formula = f"{dep_var} ~ Score_std * C({secteur_col})"
   elif model_choice == 'fama_french':
       formula = f"{dep_var} ~ Score_std + C({secteur_col}) + LogMarketCap + BookToMarket"
   else:
       formula = f"{dep_var} ~ Score_std + C({secteur_col})"

   # ── EXÉCUTION ET AFFICHAGE ──────────────────────────────────────────
   try:
       model = smf.ols(formula, data=df_ols).fit(cov_type='HC3')
       
       # 1. Metric Cards (Design original)
       coef = model.params.get('Score_std', 0)
       pval = model.pvalues.get('Score_std', 1)
       
       m_cols = st.columns(4)
       metric_card(m_cols[0], 'Coef. Score (std)', f'{coef:+.3f}', pval < 0.05)
       metric_card(m_cols[1], 'p-value', f'{pval:.3f} {sig_stars(pval)}', pval < 0.05)
       metric_card(m_cols[2], 'R² ajusté', f'{model.rsquared_adj:.3f}', True)
       metric_card(m_cols[3], 'Observations', str(int(model.nobs)), True)

       # 2. Graphique Interaction (si applicable)
       if model_choice == 'interaction':
           st.markdown('<p class="section-title">Effet du score par secteur (Pentes)</p>', unsafe_allow_html=True)
           # Insère ici ton code Plotly de barres horizontales (secteurs)
           # Utilise les coefficients du modèle pour chaque secteur

       # 3. Scatter Plot (Design original)
       st.markdown(f'<p class="section-title">Visualisation : Score standardisé vs {dep_choice}</p>', unsafe_allow_html=True)
       
       # --- EXEMPLE DE TON CODE PLOTLY (à adapter avec Score_std) ---
       import plotly.express as px
       fig = px.scatter(df_ols, x='Score_std', y=dep_var, color=quintile_col,
                        trendline="ols", color_discrete_map=QUINTILE_COLORS)
       fig.update_layout(PLOTLY_LAYOUT, height=450)
       st.plotly_chart(fig, use_container_width=True)

       # 4. Détails techniques
       with st.expander("Consulter le rapport statistique complet (Summary)"):
           st.text(model.summary().as_text())

   except Exception as e:
       st.error(f"Le calcul a échoué en raison d'une instabilité numérique : {e}")

def page_strategique(valid, prices, brent, rallies, narrative_col='Narrative', trend_col='Trend'):
   """Page Stratégique avec Graphique et Tableau Statistique complet (Rdt, Vol, Sharpe, MDD)."""
   
   st.markdown("### 📊 Analyse Stratégique")
   st.markdown('<div class="note-box">Analyse combinée des performances par profil qualitatif (Narrative) et dynamique (Trend).</div>', unsafe_allow_html=True)
   
   c1, _ = st.columns([1, 2])
   with c1:
       view_mode = st.selectbox("Axe d'analyse :", ["Par Catégorie Narrative", "Par Tendance (Trend)"], index=0)
   
   # Configuration selon le choix
   if "Narrative" in view_mode:
       target_col = narrative_col
       categories = sorted([str(x) for x in valid[target_col].dropna().unique()])
       colors_map = {"A": "#2ecc71", "B": "#3498db", "C": "#f1c40f", "D": "#e67e22", "E": "#e74c3c"}
   else:
       target_col = trend_col
       categories = sorted([str(x) for x in valid[target_col].dropna().unique()])
       colors_map = {"+": "#27ae60", "=": "#95a5a6", "-": "#c0392b"}

   # --- 1. LE GRAPHIQUE (Cumulatif) ---
   brent_norm = brent / brent.iloc[0] * 100
   fig = make_subplots(specs=[[{"secondary_y": True}]])
   fig = add_oil_rectangles(fig, rallies, first_only=True)
   
   for cat in categories:
       tickers_cat = valid[valid[target_col].astype(str) == cat]['ticker'].dropna().tolist()
       tickers_cat = [t for t in tickers_cat if t in prices.columns]
       if not tickers_cat: continue
       
       cum_par_ticker = (1 + prices[tickers_cat].pct_change()).cumprod() - 1
       cum_r = cum_par_ticker.mean(axis=1).dropna()
       
       fig.add_trace(go.Scatter(
           x=cum_r.index, y=cum_r.values,
           name=f'{cat} (n={len(tickers_cat)})',
           line=dict(color=colors_map.get(cat, "#ffffff"), width=2.5),
           hovertemplate='<b>'+cat+'</b><br>%{y:.1%}<extra></extra>'
       ), secondary_y=False)

   fig.add_trace(go.Scatter(x=brent_norm.index, y=brent_norm.values, name='Brent',
                            line=dict(color='rgba(59,130,246,0.4)', width=1.5, dash='dot')), secondary_y=True)
   
   fig.update_layout(**PLOTLY_LAYOUT, height=500)
   st.plotly_chart(fig, use_container_width=True)

   # --- 2. LE TABLEAU STATISTIQUE (Ce qui te manquait) ---
   st.markdown(f'<p class="section-title">Résumé statistique : {view_mode}</p>', unsafe_allow_html=True)
   
   rows_stats = []
   for cat in categories:
       sub = valid[valid[target_col].astype(str) == cat]
       if sub.empty: continue
       
       # On calcule les moyennes des colonnes financières déjà présentes dans 'valid'
       rows_stats.append({
           'Catégorie': cat,
           'N': len(sub),
           'Rdt 2023':  f"{sub['Rendement_2023'].mean():.1%}",
           'Rdt 2024':  f"{sub['Rendement_2024'].mean():.1%}",
           'Rdt 2025':  f"{sub['Rendement_2025'].mean():.1%}",
           'Rdt Total': f"{sub['Rendement_2023_2025'].mean():.1%}",
           'Vol':       f"{sub['Volatilite_2023_2025'].mean():.1%}",
           'Sharpe':    f"{sub['Sharpe_2023_2025'].mean():.2f}",
           'MDD':       f"{sub['MaxDrawdown_2023_2025'].mean():.1%}",
       })
   
   if rows_stats:
       st.dataframe(pd.DataFrame(rows_stats), use_container_width=True, hide_index=True)
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
   ["Accueil", "Société", "Panel Quintiles","Analyse Blocs Narrative /Trend","Score Composite Propriétaire", "Régression OLS", "Analyse Brent"],
   label_visibility="collapsed",
)

# Badge dataset
badge_class = "dataset-badge-mq" if is_mq else "dataset-badge-act"
badge_text  = "Management Quality" if is_mq else "ACT — Transition Carbone"
st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
st.markdown("")


def page_composite_proprietaire(valid_act, prices_act, brent, rallies):
   st.markdown("### 🛠️ Simulateur de Score Composite (Stratégie I, J, K)")
   st.markdown('<p class="section-title">1. Pondération des variables de transition</p>', unsafe_allow_html=True)

   # Définition dynamique des colonnes pour éviter les erreurs d'index
   col_perf  = valid_act.columns[5] # Performance Score
   col_narr  = valid_act.columns[6] # Narrative (A-E)
   col_trend = valid_act.columns[7] # Trend (+, =, -)
   col_sect  = valid_act.columns[3] # Secteur

   # --- UI DE PONDÉRATION ---
   with st.expander("Paramétrage du Modèle (Coefficients)", expanded=True):
       c1, c2, c3 = st.columns(3)
       w_i = c1.number_input("Poids Performance (I)", value=100, step=10)
       w_j = c2.number_input("Poids Narrative (J)", value=100, step=10)
       w_k = c3.number_input("Poids Trend (K)", value=100, step=10)

   # --- CALCUL DU SCORE COMPOSITE ---
   df_c = valid_act.copy()
   
   # Mappage qualitatif -> quantitatif (Base 100)
   map_j = {'A': 100, 'B': 75, 'C': 50, 'D': 25, 'E': 0}
   map_k = {'+': 100, '=': 50, '-': 0}
   
   df_c['val_j'] = df_c[col_narr].map(map_j).fillna(50)
   df_c['val_k'] = df_c[col_trend].map(map_k).fillna(50)
   
   total_w = w_i + w_j + w_k
   if total_w > 0:
       # Formule du Score Composite
       df_c['Composite_Score'] = (df_c[col_perf]*w_i + df_c['val_j']*w_j + df_c['val_k']*w_k) / total_w
   else:
       df_c['Composite_Score'] = 0

   # Standardisation (Z-Score) pour la lecture de l'Alpha en OLS
   if df_c['Composite_Score'].std() > 0:
       df_c['Score_std'] = (df_c['Composite_Score'] - df_c['Composite_Score'].mean()) / df_c['Composite_Score'].std()
   else:
       df_c['Score_std'] = 0

   # --- STRESS TEST BRENT ---
   # Calcul du rendement spécifique pendant les périodes de hausse du Brent
   # On utilise ta fonction existante calc_metriques_brent
   rdt_brent_dict, _ = calc_metriques_brent(prices_act, df_c['ticker'].tolist(), rallies)
   df_c['Rdt_Brent'] = df_c['ticker'].map(rdt_brent_dict)

   # --- RÉGRESSIONS OLS ---
   st.markdown('<p class="section-title">2. Analyse de l\'Alpha & Résilience au Choc Pétrolier</p>', unsafe_allow_html=True)
   
   # On prépare les données (nettoyage des NaN pour les modèles)
   data_glob = df_c.dropna(subset=['Rendement_2023_2025', 'Score_std', col_sect])
   data_brent = df_c.dropna(subset=['Rdt_Brent', 'Score_std', col_sect])

   if len(data_glob) > 10 and len(data_brent) > 10:
       m_glob = smf.ols(f"Rendement_2023_2025 ~ Score_std + C({col_sect})", data=data_glob).fit()
       m_brent = smf.ols(f"Rdt_Brent ~ Score_std + C({col_sect})", data=data_brent).fit()

       # Affichage des résultats
       res1, res2, res3 = st.columns(3)
       
       alpha_g = m_glob.params['Score_std']
       alpha_b = m_brent.params['Score_std']
       sig_g = m_glob.pvalues['Score_std']
       sig_b = m_brent.pvalues['Score_std']

       # Utilisation de ta fonction metric_card si elle existe, sinon st.metric
       metric_card(res1, "Alpha Global (std)", f"{alpha_g:+.4f}", sig_g < 0.05)
       metric_card(res2, "Alpha Brent (std)", f"{alpha_b:+.4f}", sig_b < 0.05)
       
       diff = alpha_b - alpha_g
       metric_card(res3, "Gain de Résilience", f"{diff:+.4f}", diff > 0)

       # Tabs pour les détails
       with st.expander("Consulter les rapports statistiques détaillés"):
           t1, t2 = st.tabs(["Régression Période Totale", "Régression Période Brent-Up"])
           t1.code(m_glob.summary().as_text())
           t2.code(m_brent.summary().as_text())
   else:
       st.warning("Échantillon trop faible pour générer les régressions.")

   # --- TABLEAU RÉCAPITULATIF ---
   st.markdown('<p class="section-title">3. Aperçu des meilleurs scores composites</p>', unsafe_allow_html=True)
   top_df = df_c.sort_values('Composite_Score', ascending=False).head(15)
   st.dataframe(top_df[[df_c.columns[0], 'ticker', col_perf, col_narr, col_trend, 'Composite_Score']], use_container_width=True, hide_index=True)

def calc_metriques_brent(prices, tickers, rallies):
   """
   Calcule le rendement cumulé ET la volatilité annualisée
   sur l'ensemble des fenêtres de hausse Brent (agrégé).
   """
   rdt_records = {}
   vol_records = {}
   for t in tickers:
       if t not in prices.columns:
           rdt_records[t] = np.nan
           vol_records[t] = np.nan
           continue
       px        = prices[t].dropna()
       cumret    = 1.0
       all_rets  = []
       for s, e in rallies:
           window = px.loc[s:e].dropna()
           if len(window) < 5:
               continue
           cumret *= (window.iloc[-1] / window.iloc[0])
           all_rets.extend(window.pct_change().dropna().tolist())
       n_days = len(all_rets)
       if n_days >= 10:
           rdt_records[t] = float(cumret - 1)
           vol_records[t] = float(np.std(all_rets) * np.sqrt(21))
       else:
           rdt_records[t] = np.nan
           vol_records[t] = np.nan
   return rdt_records, vol_records


def page_brent(valid, prices, rallies, score_col, score_label, secteur_col, quintile_col, total_panel, dataset_label):
   """Analyse Brent complète : Réintégrée avec Graphiques Secteurs et Quintiles."""
   
   st.markdown(f"### Analyse Brent · {dataset_label}")
   st.markdown('<p class="section-title">Comparaison des coefficients OLS — modèle général vs pendant les hausses Brent</p>', unsafe_allow_html=True)

   # 1. SÉLECTEURS
   c1, c2, c3 = st.columns(3)
   with c1:
       period_gen = st.selectbox("Période modèle général", list(PERIODS_LABELS.keys()), 
                                 format_func=lambda x: PERIODS_LABELS[x], key="b_p")
   with c2:
       dep_choice_b = st.selectbox("Variable", ["Rendement", "Volatilite"], key="b_d")
   with c3:
       model_type = st.selectbox("Modèle", 
           ["simple", "interaction", "fama_french"],
           format_func=lambda x: {
               "simple": "Simple (Score + Secteur)",
               "interaction": "Interaction (Score × Secteur)",
               "fama_french": "Fama-French (Score + Secteur + Taille + B/M)"
           }[x], key="b_m")

   dep_gen   = f"{dep_choice_b}_{period_gen}"
   dep_brent = f"{dep_choice_b}_Brent"

   # 2. CALCULS ET PRÉPARATION (Stabilité maths)
   with st.spinner("Calcul des métriques..."):
       tickers_valid = valid["ticker"].dropna().tolist()
       rdt_brent, vol_brent = calc_metriques_brent(prices, tickers_valid, rallies)

   valid2 = valid.copy()
   valid2["Rendement_Brent"]  = valid2["ticker"].map(rdt_brent)
   valid2["Volatilite_Brent"] = valid2["ticker"].map(vol_brent)
   valid2 = prepare_ols_data(valid2, score_col, secteur_col)



# 3. MOTEUR DE CALCUL POUR LES 3 MODÈLES
   def run_ols_custom(dep_v, force_model=None):
       m_to_use = force_model if force_model else model_type
       df_r = valid2.dropna(subset=[dep_v, 'Score_std', secteur_col]).copy()
       df_r[dep_v] = winsorize(df_r[dep_v])
       
       if len(df_r) < (df_r[secteur_col].nunique() + 10): return None
       
       if m_to_use == "interaction":
           formula = f"{dep_v} ~ Score_std * C({secteur_col})"
       elif m_to_use == "fama_french":
           formula = f"{dep_v} ~ Score_std + C({secteur_col}) + LogMarketCap + BookToMarket"
       else: # simple
           formula = f"{dep_v} ~ Score_std + C({secteur_col})"
       
       try:
           return smf.ols(formula, data=df_r).fit(cov_type="HC3")
       except: return None

   m_gen = run_ols_custom(dep_gen)
   m_br  = run_ols_custom(dep_brent)



   # 5. AFFICHAGE DES METRIC CARDS (Design original)
   if m_gen and m_br:
       c_gen, p_gen = m_gen.params['Score_std'], m_gen.pvalues['Score_std']
       c_br, p_br   = m_br.params['Score_std'], m_br.pvalues['Score_std']
       delta = c_br - c_gen

       cols = st.columns(4)
       metric_card(cols[0], "Coef. Général", f"{c_gen:+.3f} {sig_stars(p_gen)}", p_gen < 0.05)
       metric_card(cols[1], "Coef. Brent-up", f"{c_br:+.3f} {sig_stars(p_br)}", p_br < 0.05)
       metric_card(cols[2], "Δ (Brent - Gén)", f"{delta:+.3f}", delta > 0)
       metric_card(cols[3], "R² (Gén / Br)", f"{m_gen.rsquared_adj:.2f} / {m_br.rsquared_adj:.2f}", True)

       if abs(delta) > 0.05 and p_br < 0.1:
           st.markdown(f'<div class="note-box">Le score {score_label} semble avoir un impact {"accru" if delta > 0 else "atténué"} lors des chocs pétroliers.</div>', unsafe_allow_html=True)
   else:
       st.markdown(f'<div class="warn-box">Données insuffisantes en {period_gen} ou pendant les hausses Brent pour calculer les coefficients.</div>', unsafe_allow_html=True)
   # 5. SECTION VISUALISATION : SCATTER PLOT (Hausse Brent)
   st.markdown(f'<p class="section-title">Visualisation : Score standardisé vs {dep_choice_b} (Période Brent-up)</p>', unsafe_allow_html=True)
   
   # On filtre pour n'avoir que les données valides pour le graphique
   df_scatter = valid2.dropna(subset=[dep_brent, 'Score_std', quintile_col]).copy()

   if not df_scatter.empty:
       import plotly.express as px
       
       # Création du graphique
       fig_scat = px.scatter(
           df_scatter, 
           x='Score_std', 
           y=dep_brent, 
           color=quintile_col,
           trendline="ols", # Trace la droite de régression
           color_discrete_map=QUINTILE_COLORS,
           hover_name="ticker",
           labels={
               "Score_std": "Score ESG (Standardisé)",
               dep_brent: f"{dep_choice_b} (pendant la hausse)"
           }
       )
       
       # Application de ton design global
       fig_scat.update_layout(PLOTLY_LAYOUT, height=450)
       
       # Affichage
       st.plotly_chart(fig_scat, use_container_width=True)
   else:
       st.info("Échantillon trop restreint pour afficher le nuage de points.")
   # 6. GRAPHIQUE : COEFFICIENTS PAR SECTEUR (Interaction)
   st.markdown('<p class="section-title">Coefficients par secteur — général vs Brent-up</p>', unsafe_allow_html=True)
   m_gen_int = run_ols_custom(dep_gen, "interaction")
   m_br_int  = run_ols_custom(dep_brent, "interaction")

   if m_gen_int and m_br_int:
       sectors_list = sorted(valid2[secteur_col].unique())
       data_sect = []
       
       for s in sectors_list:
           # Pente générale
           p_gen = m_gen_int.params.get('Score_std', 0)
           inter_name_gen = f'Score_std:C({secteur_col})[T.{s}]'
           if inter_name_gen in m_gen_int.params:
               p_gen += m_gen_int.params[inter_name_gen]
           
           # Pente Brent
           p_br = m_br_int.params.get('Score_std', 0)
           inter_name_br = f'Score_std:C({secteur_col})[T.{s}]'
           if inter_name_br in m_br_int.params:
               p_br += m_br_int.params[inter_name_br]
               
           data_sect.append({"Secteur": s, "Modèle": "Général", "Coefficient": p_gen})
           data_sect.append({"Secteur": s, "Modèle": "Brent-up", "Coefficient": p_br})
       
       df_plot_s = pd.DataFrame(data_sect)
       import plotly.express as px
       fig_s = px.bar(df_plot_s, x="Coefficient", y="Secteur", color="Modèle", 
                      barmode="group", orientation='h',
                      color_discrete_map={"Général": "#636EFA", "Brent-up": "#EF553B"})
       fig_s.update_layout(PLOTLY_LAYOUT, height=600, margin=dict(l=20, r=20, t=20, b=20))
       st.plotly_chart(fig_s, use_container_width=True)
   else:
       st.info("Le graphique des secteurs n'est pas disponible pour cette sélection (échantillon trop faible).")

   # 7. SECTION QUINTILES (Comparaison visuelle directe)
   st.markdown('<p class="section-title">Performance moyenne par quintile — général vs Brent-up</p>', unsafe_allow_html=True)
   
   q_data = []
   quintiles_present = sorted(valid2[quintile_col].dropna().unique())
   
   for q in quintiles_present:
       sub = valid2[valid2[quintile_col] == q]
       q_data.append({
           "Quintile": q,
           "Type": "Général",
           "Valeur": sub[dep_gen].mean()
       })
       q_data.append({
           "Quintile": q,
           "Type": "Brent-up",
           "Valeur": sub[dep_brent].mean()
       })
   
   if q_data:
       df_q = pd.DataFrame(q_data)
       fig_q = px.bar(df_q, x="Quintile", y="Valeur", color="Type", 
                      barmode="group", color_discrete_map={"Général": "#AB63FA", "Brent-up": "#FFA15A"})
       fig_q.update_layout(PLOTLY_LAYOUT, yaxis_tickformat='.1%', height=400)
       st.plotly_chart(fig_q, use_container_width=True)

       # Tableau récapitulatif
       df_table = df_q.pivot(index="Quintile", columns="Type", values="Valeur")
       df_table["Différence"] = df_table["Brent-up"] - df_table["Général"]
       st.table(df_table.style.format("{:.2%}"))

   # 8. RÉSULTATS DÉTAILLÉS (Summary)
   with st.expander("Consulter les rapports statistiques complets (OLS Summary)"):
       col_left, col_right = st.columns(2)
       with col_left:
           st.write("**Modèle Général**")
           if m_gen: st.text(m_gen.summary().as_text())
           else: st.write("Non disponible")
       with col_right:
           st.write("**Modèle Brent-up**")
           if m_br: st.text(m_br.summary().as_text())
           else: st.write("Non disponible")

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
elif page == "Analyse Blocs Narrative /Trend":
   if is_mq:
       st.info("📊 **Analyse Stratégique** : Cette section (Narrative & Trend) est spécifique au panel **ACT**. Le panel Management Quality ne contient pas ces indicateurs qualitatifs.")
   else:
       # Récupération dynamique des noms de colonnes J et K pour ACT
       col_J_name = df_act.columns[6]  # Narrative
       col_K_name = df_act.columns[7]  # Trend
       
       page_strategique(
           valid=valid_act, 
           prices=prices_act, 
           brent=brent, 
           rallies=rallies, 
           narrative_col=col_J_name, 
           trend_col=col_K_name
       )
# Dans ton bloc if/elif de navigation à la fin du fichier :

# Dans ton bloc if/elif de navigation à la fin du fichier :

elif page == "Score Composite Propriétaire":
   # On force l'utilisation des données ACT
   if not is_mq:
       # On vérifie que les variables nécessaires existent
       try:
           page_composite_proprietaire(valid_act, prices_act, brent, rallies)
       except Exception as e:
           st.error(f"Erreur lors du calcul du score : {e}")
   else:
       st.info("💡 Cette analyse est optimisée pour le référentiel **ACT**. Veuillez basculer le menu latéral sur ACT.")


# In[ ]:





# In[ ]:




