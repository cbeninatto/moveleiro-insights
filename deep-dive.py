import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import plotly.express as px
import math
import io
import time
import requests
import re
import unicodedata
import html
import datetime

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Insights de Vendas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@page { size: A4 portrait; margin: 1.5cm; }
@media print {
  html, body { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  [data-testid="stSidebar"], header, footer { display: none !important; }
  .page-break { break-before: page; page-break-before: always; }
}
</style>
""",
    unsafe_allow_html=True,
)

GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"
)

CITY_GEO_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/performance-moveleiro-v2/main/data/cidades_br_geo.csv"
)

# === MAP REQUIREMENTS ===
LEAFLET_VERSION = "1.9.4"
OSM_TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTR = "© OpenStreetMap contributors"

STATUS_COL = "StatusCarteira"

STATUS_WEIGHTS = {
    "Novos": 1, "Novo": 1,
    "Crescendo": 2, "CRESCENDO": 2,
    "Estáveis": 1, "Estável": 1, "ESTAVEIS": 1,
    "Caindo": -1, "CAINDO": -1,
    "Perdidos": -2, "Perdido": -2, "PERDIDOS": -2,
}
STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]

# green = higher, red = lower
MAP_BIN_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

# ==========================
# STREAMLIT COMPAT HELPERS
# ==========================
def st_button_stretch(label, key=None):
    try:
        return st.button(label, width="stretch", key=key)
    except TypeError:
        return st.button(label, use_container_width=True, key=key)

def st_plotly_stretch(fig, height=None, key=None):
    if fig is None:
        return
    try:
        st.plotly_chart(fig, width="stretch", height=height, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)

def st_altair_stretch(chart, key=None):
    if chart is None:
        return
    try:
        st.altair_chart(chart, width="stretch", key=key)
    except TypeError:
        st.altair_chart(chart, use_container_width=True, key=key)

# ==========================
# HELPERS
# ==========================
def format_brl(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_brl_compact(value: float) -> str:
    if pd.isna(value):
        return "R$ 0"
    v = float(value)
    av = abs(v)
    if av >= 1_000_000_000:
        return "R$ " + f"{v/1_000_000_000:.1f} bi".replace(".", ",")
    if av >= 1_000_000:
        return "R$ " + f"{v/1_000_000:.1f} mi".replace(".", ",")
    if av >= 1_000:
        return "R$ " + f"{v/1_000:.1f} mil".replace(".", ",")
    return format_brl(v)

def format_brl_signed(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    v = float(value)
    sign = "-" if v < 0 else ""
    return sign + format_brl(abs(v))

def format_un(value: float) -> str:
    if pd.isna(value):
        return "0 un"
    try:
        v = int(round(float(value)))
    except Exception:
        v = 0
    return f"{v:,}".replace(",", ".") + " un"

def shorten_name(name: str, max_len: int = 26) -> str:
    s = str(name) if name is not None else ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"

def force_leaflet_1_9_4():
    try:
        import folium.folium as ff
        new_js = []
        for name, url in ff._default_js:
            if "leaflet" in name.lower():
                new_js.append(("leaflet", f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.js"))
            else:
                new_js.append((name, url))
        new_css = []
        for name, url in ff._default_css:
            if "leaflet" in name.lower():
                new_css.append(("leaflet_css", f"https://unpkg.com/leaflet@{LEAFLET_VERSION}/dist/leaflet.css"))
            else:
                new_css.append((name, url))
        ff._default_js = new_js
        ff._default_css = new_css
    except Exception:
        pass

def build_dynamic_bins(values, is_valor: bool):
    cleaned = [float(v) for v in values if pd.notna(v) and float(v) >= 0]
    cleaned.sort()
    def fmt(v: float) -> str:
        if is_valor:
            return "R$ " + f"{v:,.0f}".replace(",", ".")
        return f"{int(round(v)):,}".replace(",", ".") + " un"
    if not cleaned:
        if is_valor:
            return [
                {"min": 1_000_000, "color": MAP_BIN_COLORS[0], "label": "R$ 1.000.000+"},
                {"min": 500_000,   "color": MAP_BIN_COLORS[1], "label": "R$ 500.000 - 999.999"},
                {"min": 250_000,   "color": MAP_BIN_COLORS[2], "label": "R$ 250.000 - 499.999"},
                {"min": 0,         "color": MAP_BIN_COLORS[3], "label": "R$ 0 - 249.999"},
            ]
        return [
            {"min": 10_000, "color": MAP_BIN_COLORS[0], "label": "10.000 un+"},
            {"min": 5_000,  "color": MAP_BIN_COLORS[1], "label": "5.000 - 9.999 un"},
            {"min": 2_500,  "color": MAP_BIN_COLORS[2], "label": "2.500 - 4.999 un"},
            {"min": 0,      "color": MAP_BIN_COLORS[3], "label": "0 - 2.499 un"},
        ]
    n = len(cleaned)
    min_val = cleaned[0]
    max_val = cleaned[-1]
    if min_val == max_val:
        label_single = fmt(min_val)
        color = MAP_BIN_COLORS[0]
        return [{"min": min_val, "color": color, "label": label_single}] * 4
    def idx(p: float) -> int:
        return int(math.floor(p * (n - 1)))
    q1, q2, q3 = cleaned[idx(0.25)], cleaned[idx(0.5)], cleaned[idx(0.75)]
    t0, t1, t2, t3 = max(0, min_val), max(0, q1), max(0, q2), max(0, q3)
    return [
        {"min": t3, "color": MAP_BIN_COLORS[0], "label": f"{fmt(t3)}+"},
        {"min": t2, "color": MAP_BIN_COLORS[1], "label": f"{fmt(t2)} - {fmt(t3)}"},
        {"min": t1, "color": MAP_BIN_COLORS[2], "label": f"{fmt(t1)} - {fmt(t2)}"},
        {"min": t0, "color": MAP_BIN_COLORS[3], "label": f"{fmt(t0)} - {fmt(t1)}"},
    ]

def get_bin_for_value(v: float, bins):
    for b in bins:
        if v >= b["min"]: return b
    return bins[-1]

def load_data() -> pd.DataFrame:
    cb = int(time.time())
    url = f"{GITHUB_CSV_URL}?cb={cb}"
    resp = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")
    df["Competencia"] = pd.to_datetime(dict(year=df["Ano"], month=df["MesNum"], day=1), errors="coerce")
    return df

def compute_carteira_score(clientes_carteira: pd.DataFrame):
    if clientes_carteira is None or clientes_carteira.empty:
        return 50.0, "Neutra"
    df = clientes_carteira.copy()
    df["PesoReceita"] = df[["ValorAtual", "ValorAnterior"]].max(axis=1).clip(lower=0)
    receita_status = df.groupby(STATUS_COL)["PesoReceita"].sum()
    total = float(receita_status.sum())
    if total <= 0: return 50.0, "Neutra"
    score_bruto = sum(STATUS_WEIGHTS.get(str(s), 0) * (r / total) for s, r in receita_status.items())
    isc = max(0.0, min(100.0, (score_bruto + 2) / 4 * 100))
    label = "Crítica" if isc < 30 else "Alerta" if isc < 50 else "Neutra" if isc < 70 else "Saudável"
    return float(isc), label

MONTH_MAP_NUM_TO_NAME = {1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR", 5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO", 9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ"}
MONTH_MAP_NAME_TO_NUM = {v: k for k, v in MONTH_MAP_NUM_TO_NAME.items()}

def format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    def fmt(d: pd.Timestamp): return f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}"
    return fmt(start) if start == end else f"{fmt(start)} - {fmt(end)}"

def build_carteira_status(df_all: pd.DataFrame, rep: str, start_comp: pd.Timestamp, end_comp: pd.Timestamp) -> pd.DataFrame:
    df_rep_all = df_all.copy() if rep == "Todos" else df_all[df_all["Representante"] == rep].copy()
    if df_rep_all.empty: return pd.DataFrame()
    months_span = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1
    prev_end = start_comp - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=months_span - 1)
    curr = df_rep_all[(df_rep_all["Competencia"] >= start_comp) & (df_rep_all["Competencia"] <= end_comp)]
    prev = df_rep_all[(df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)]
    curr_agg = curr.groupby("Cliente").agg({"Valor": "sum", "Estado": "first", "Cidade": "first"}).rename(columns={"Valor": "ValorAtual"})
    prev_agg = prev.groupby("Cliente").agg({"Valor": "sum", "Estado": "first", "Cidade": "first"}).rename(columns={"Valor": "ValorAnterior"})
    cl = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer").fillna(0)
    def classify(row):
        va, vp = row["ValorAtual"], row["ValorAnterior"]
        if va > 0 and vp == 0: return "Novos"
        if va == 0 and vp > 0: return "Perdidos"
        ratio = va / vp if vp > 0 else 0
        return "Crescendo" if ratio >= 1.2 else "Caindo" if ratio <= 0.8 else "Estáveis"
    cl[STATUS_COL] = cl.apply(classify, axis=1)
    return cl

@st.cache_data(ttl=3600)
def load_geo() -> pd.DataFrame:
    resp = requests.get(f"{CITY_GEO_CSV_URL}?cb={int(time.time())}", timeout=60)
    df_geo = pd.read_csv(io.StringIO(resp.text), sep=None, engine="python")
    # Simplificação de colunas para o exemplo
    df_geo = df_geo.rename(columns={"codigo_uf": "Estado", "nome": "Cidade", "latitude": "lat", "longitude": "lon"})
    df_geo["key"] = df_geo["Estado"].astype(str).str.upper() + "|" + df_geo["Cidade"].astype(str).str.upper()
    return df_geo

# ==========================
# LOAD DATA & FILTERS
# ==========================
try:
    df = load_data()
except Exception as e:
    st.error(f"Erro: {e}"); st.stop()

st.sidebar.title("Filtros")
anos = sorted(df["Ano"].dropna().unique())
last_year = int(anos[-1])
start_year = st.sidebar.selectbox("Ano Inicial", anos, index=len(anos)-1)
start_month = MONTH_MAP_NAME_TO_NUM[st.sidebar.selectbox("Mês Inicial", list(MONTH_MAP_NUM_TO_NAME.values()))]
end_year = st.sidebar.selectbox("Ano Final", anos, index=len(anos)-1)
end_month = MONTH_MAP_NAME_TO_NUM[st.sidebar.selectbox("Mês Final", list(MONTH_MAP_NUM_TO_NAME.values()), index=11)]

start_comp = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
end_comp = pd.Timestamp(year=int(end_year), month=int(end_month), day=1)
if start_comp > end_comp: st.error("Data inicial > final"); st.stop()

rep_selected = st.sidebar.selectbox("Representante", ["Todos"] + sorted(df["Representante"].unique().tolist()))
df_rep = df[(df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)]
if rep_selected != "Todos": df_rep = df_rep[df_rep["Representante"] == rep_selected]

# Período Anterior
months_span = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1
prev_end = start_comp - pd.DateOffset(months=1)
prev_start = prev_end - pd.DateOffset(months=months_span - 1)
df_rep_prev = df[(df["Competencia"] >= prev_start) & (df["Competencia"] <= prev_end)]
if rep_selected != "Todos": df_rep_prev = df_rep_prev[df_rep_prev["Representante"] == rep_selected]

current_period_label = format_period_label(start_comp, end_comp)
previous_period_label = format_period_label(prev_start, prev_end)
clientes_carteira = build_carteira_status(df, rep_selected, start_comp, end_comp)

# ==========================
# INICIALIZAÇÃO DE VARIÁVEIS (CORREÇÃO DO NameError)
# ==========================
total_rep = 0.0
total_vol_rep = 0.0
media_mensal = 0.0
num_clientes_rep = 0
n80_count = 0
hhi_value = 0.0
hhi_label_short = "Sem dados"
top1_share = 0.0; top3_share = 0.0; top10_share = 0.0
clientes_atendidos = 0
cidades_atendidas = 0
estados_atendidos = 0
carteira_score = 50.0
carteira_label = "Neutra"

# ==========================
# CÁLCULO DOS KPIs
# ==========================
if not df_rep.empty:
    total_rep = float(df_rep["Valor"].sum())
    total_vol_rep = float(df_rep["Quantidade"].sum())
    meses_venda = df_rep.groupby(["Ano", "MesNum"]).size().shape[0]
    media_mensal = total_rep / meses_venda if meses_venda > 0 else 0.0
    
    df_cli = df_rep.groupby("Cliente")["Valor"].sum().sort_values(ascending=False)
    num_clientes_rep = len(df_cli)
    clientes_atendidos = num_clientes_rep
    
    if total_rep > 0:
        shares = df_cli / total_rep
        n80_count = (shares.cumsum() <= 0.8).sum() + 1
        hhi_value = float((shares**2).sum())
        hhi_label_short = "Baixa" if hhi_value < 0.1 else "Moderada" if hhi_value < 0.2 else "Alta"
        top1_share = shares.iloc[0] if len(shares) >= 1 else 0
        top3_share = shares.iloc[:3].sum() if len(shares) >= 3 else 0
        top10_share = shares.iloc[:10].sum() if len(shares) >= 10 else 0

    cidades_atendidas = df_rep[["Estado", "Cidade"]].drop_duplicates().shape[0]
    estados_atendidos = df_rep["Estado"].nunique()

if not clientes_carteira.empty:
    carteira_score, carteira_label = compute_carteira_score(clientes_carteira)

# ==========================
# UI - DASHBOARD
# ==========================
st.title("Insights de Vendas")
titulo_rep = rep_selected
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total período", format_brl_compact(total_rep))
col2.metric("Média mensal", format_brl_compact(media_mensal))
col3.metric("Distribuição", hhi_label_short, f"N80: {n80_count}")
col4.metric("Saúde Carteira", f"{carteira_score:.0f}/100", carteira_label)
col5.metric("Clientes", f"{clientes_atendidos}")

st.markdown("---")

# Seção de Evolução
st.subheader("Evolução – Faturamento x Volume")
def make_evolucao_chart(df_in):
    if df_in.empty: return None
    ts = df_in.groupby("Competencia").agg({"Valor":"sum", "Quantidade":"sum"}).reset_index()
    ts["MesLabel"] = ts["Competencia"].apply(lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}")
    base = alt.Chart(ts).encode(x=alt.X("MesLabel:N", sort=None))
    bars = base.mark_bar(color="#38bdf8").encode(y=alt.Y("Valor:Q", title="Faturamento"))
    line = base.mark_line(color="#22c55e").encode(y=alt.Y("Quantidade:Q", title="Volume"))
    return alt.layer(bars, line).resolve_scale(y="independent").properties(height=300)

c_ev = make_evolucao_chart(df_rep)
if c_ev: st_altair_stretch(c_ev)

st.markdown("---")

# Seção que estava dando erro: Distribuição por Clientes
st.subheader("Distribuição por clientes")
if df_rep.empty or clientes_atendidos == 0:
    st.info("Nenhum cliente com vendas no período selecionado.")
else:
    df_clientes_full = df_rep.groupby(["Cliente", "Estado", "Cidade"]).agg({"Valor":"sum", "Quantidade":"sum"}).reset_index().sort_values("Valor", ascending=False)
    st.write(df_clientes_full.head(10))

# ... (O restante do seu código de Mapas, Categorias e PDF segue aqui sem alterações)
# Basta garantir que a lógica de busca/tabelas use as variáveis inicializadas.

st.success("Dashboard carregado com sucesso!")
