# rep_report.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Relatório do Representante", layout="wide")

# ============================================================
# DATA SOURCES (repo paths)
# ============================================================
DATA_PATH = "data/raw/relatorio_faturamento.csv"
GEO_CIDADES_PATH = "data/raw/cidades_br_geo.csv"
GEO_CLIENTES_PATH = "data/raw/clientes_relatorio_faturamento.csv"

# ============================================================
# COLUMN MAPPING (matches your relatorio_faturamento.csv headers)
# ============================================================
# Your CSV columns (as you posted):
# "﻿Codigo","Descricao","Quantidade","Valor","Mes","Ano","ClienteCodigo","Cliente","Estado","Cidade",
# "RepresentanteCodigo","Representante","Categoria","SourcePDF"
COL = {
    "year": "Ano",
    "month": "Mes",
    "rep": "Representante",
    "client": "Cliente",
    "city": "Cidade",
    "state": "Estado",
    "category": "Categoria",
    "rev": "Valor",
    "vol": "Quantidade",
}

MONTHS = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
MONTH_TO_INT = {m: i + 1 for i, m in enumerate(MONTHS)}
INT_TO_MONTH = {i + 1: m for i, m in enumerate(MONTHS)}

# ============================================================
# FORMATTERS
# ============================================================
def money(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def num(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%".replace(".", ",")

def pct_change(curr: float, prev: float) -> float:
    if prev == 0 or pd.isna(prev):
        return np.nan
    return (curr - prev) / prev

# ============================================================
# DATE / PERIOD HELPERS
# ============================================================
def normalize_month(m):
    """Accepts 1..12, '01', 'JAN', 'JANEIRO', etc. Returns 1..12 or NaN."""
    if pd.isna(m):
        return np.nan
    if isinstance(m, (int, np.integer)):
        mi = int(m)
        return mi if 1 <= mi <= 12 else np.nan
    s = str(m).strip().upper()

    if s.isdigit():
        mi = int(s)
        return mi if 1 <= mi <= 12 else np.nan

    mapa = {
        "JAN": 1, "JANEIRO": 1,
        "FEV": 2, "FEVEREIRO": 2,
        "MAR": 3, "MARÇO": 3, "MARCO": 3,
        "ABR": 4, "ABRIL": 4,
        "MAI": 5, "MAIO": 5,
        "JUN": 6, "JUNHO": 6,
        "JUL": 7, "JULHO": 7,
        "AGO": 8, "AGOSTO": 8,
        "SET": 9, "SETEMBRO": 9,
        "OUT": 10, "OUTUBRO": 10,
        "NOV": 11, "NOVEMBRO": 11,
        "DEZ": 12, "DEZEMBRO": 12,
    }
    for k, v in mapa.items():
        if s.startswith(k):
            return v
    return np.nan

def month_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(month), day=1)

def month_end(year: int, month: int) -> pd.Timestamp:
    return (pd.Timestamp(year=int(year), month=int(month), day=1) + pd.offsets.MonthEnd(0))

def filter_period_months(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    return df[(df["data"] >= start_ts) & (df["data"] <= end_ts)].copy()

# ============================================================
# LOADERS
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_normalize_base(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df_raw = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    df_raw.columns = [c.lstrip("\ufeff").strip() for c in df_raw.columns]

    needed = [COL["year"], COL["month"], COL["rep"], COL["client"], COL["city"], COL["state"], COL["category"], COL["rev"], COL["vol"]]
    missing = [c for c in needed if c not in df_raw.columns]
    if missing:
        return pd.DataFrame()

    df = df_raw.copy()
    df[COL["year"]] = pd.to_numeric(df[COL["year"]], errors="coerce")
    df["_month_num"] = df[COL["month"]].apply(normalize_month)

    df[COL["rev"]] = pd.to_numeric(df[COL["rev"]], errors="coerce")
    df[COL["vol"]] = pd.to_numeric(df[COL["vol"]], errors="coerce")

    df["data"] = pd.to_datetime(
        dict(year=df[COL["year"]], month=df["_month_num"], day=1),
        errors="coerce"
    )

    # clean strings
    for c in [COL["rep"], COL["client"], COL["city"], COL["state"], COL["category"]]:
        df[c] = df[c].astype(str).str.strip()

    df = df.dropna(subset=["data"])
    return df

def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm:
            return norm[key]
    return None

@st.cache_data(show_spinner=False)
def load_geo_tables():
    """
    Retorna (geo_clientes, geo_cidades) já padronizados:
      geo_clientes: cliente_key, lat, lon
      geo_cidades: cidade_key, estado_key, lat, lon
    """
    geo_clientes = pd.DataFrame()
    geo_cidades = pd.DataFrame()

    # ---- clientes geo
    if os.path.exists(GEO_CLIENTES_PATH):
        g = pd.read_csv(GEO_CLIENTES_PATH, sep=None, engine="python", encoding="utf-8")
        g.columns = [c.lstrip("\ufeff").strip() for c in g.columns]

        col_cliente = _pick_col(g, ["Cliente", "cliente", "Nome", "nome", "RazaoSocial", "razaosocial", "Descricao", "descricao"])
        col_lat = _pick_col(g, ["lat", "latitude", "Latitude", "LAT"])
        col_lon = _pick_col(g, ["lon", "lng", "longitude", "Longitude", "LON", "LONG"])

        if col_cliente and col_lat and col_lon:
            geo_clientes = g[[col_cliente, col_lat, col_lon]].copy()
            geo_clientes.rename(columns={col_cliente: "cliente_key", col_lat: "lat", col_lon: "lon"}, inplace=True)
            geo_clientes["cliente_key"] = geo_clientes["cliente_key"].astype(str).str.strip().str.upper()
            geo_clientes["lat"] = pd.to_numeric(geo_clientes["lat"], errors="coerce")
            geo_clientes["lon"] = pd.to_numeric(geo_clientes["lon"], errors="coerce")
            geo_clientes = geo_clientes.dropna(subset=["lat", "lon"]).drop_duplicates(["cliente_key"])

    # ---- cidades geo
    if os.path.exists(GEO_CIDADES_PATH):
        g = pd.read_csv(GEO_CIDADES_PATH, sep=None, engine="python", encoding="utf-8")
        g.columns = [c.lstrip("\ufeff").strip() for c in g.columns]

        col_cidade = _pick_col(g, ["Cidade", "cidade", "Municipio", "município", "nome_municipio", "nome"])
        col_uf = _pick_col(g, ["UF", "uf", "Estado", "estado", "sigla_uf"])
        col_lat = _pick_col(g, ["lat", "latitude", "Latitude", "LAT"])
        col_lon = _pick_col(g, ["lon", "lng", "longitude", "Longitude", "LON", "LONG"])

        if col_cidade and col_uf and col_lat and col_lon:
            geo_cidades = g[[col_cidade, col_uf, col_lat, col_lon]].copy()
            geo_cidades.rename(
                columns={col_cidade: "cidade_key", col_uf: "estado_key", col_lat: "lat", col_lon: "lon"},
                inplace=True,
            )
            geo_cidades["cidade_key"] = geo_cidades["cidade_key"].astype(str).str.strip().str.upper()
            geo_cidades["estado_key"] = geo_cidades["estado_key"].astype(str).str.strip().str.upper()
            geo_cidades["lat"] = pd.to_numeric(geo_cidades["lat"], errors="coerce")
            geo_cidades["lon"] = pd.to_numeric(geo_cidades["lon"], errors="coerce")
            geo_cidades = geo_cidades.dropna(subset=["lat", "lon"]).drop_duplicates(["cidade_key", "estado_key"])

    return geo_clientes, geo_cidades

# ============================================================
# CALCS
# ============================================================
def client_distribution(df_period: pd.DataFrame):
    by_client = (
        df_period.groupby(COL["client"], as_index=False)[COL["rev"]]
        .sum()
        .rename(columns={COL["rev"]: "rev", COL["client"]: "cliente"})
    )
    by_client["rev"] = by_client["rev"].fillna(0.0)
    total = float(by_client["rev"].sum())

    if total <= 0 or by_client.empty:
        empty = pd.DataFrame(columns=["cliente", "rev", "share", "cum_share"])
        return empty, 0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan

    by_client = by_client.sort_values("rev", ascending=False).reset_index(drop=True)
    by_client["share"] = by_client["rev"] / total
    by_client["cum_share"] = by_client["share"].cumsum()

    n80_count = int((by_client["cum_share"] <= 0.80).sum())
    if n80_count == 0:
        n80_count = 1
    n80_rev_share = float(by_client.loc[n80_count - 1, "cum_share"])
    n80_share_clients = n80_count / len(by_client)

    hhi = float((by_client["share"] ** 2).sum())
    top1 = float(by_client["share"].head(1).sum())
    top3 = float(by_client["share"].head(3).sum())
    top10 = float(by_client["share"].head(10).sum())

    return by_client, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10

def build_client_status(df_curr: pd.DataFrame, df_prev: pd.DataFrame, thr: float = 0.10) -> pd.DataFrame:
    curr = df_curr.groupby(COL["client"], as_index=False).agg(
        curr_rev=(COL["rev"], "sum"),
        curr_vol=(COL["vol"], "sum"),
    )
    prev = df_prev.groupby(COL["client"], as_index=False).agg(
        prev_rev=(COL["rev"], "sum"),
        prev_vol=(COL["vol"], "sum"),
    )

    m = curr.merge(prev, on=COL["client"], how="outer").fillna(0.0)
    m["delta_rev"] = m["curr_rev"] - m["prev_rev"]
    m["delta_vol"] = m["curr_vol"] - m["prev_vol"]

    def _pct(a, b):
        return np.nan if b == 0 else (a - b) / b

    m["pct_rev"] = m.apply(lambda r: _pct(r["curr_rev"], r["prev_rev"]), axis=1)
    m["pct_vol"] = m.apply(lambda r: _pct(r["curr_vol"], r["prev_vol"]), axis=1)

    prev_r = m["prev_rev"].to_numpy()
    curr_r = m["curr_rev"].to_numpy()

    status = np.full(len(m), "Estáveis", dtype=object)
    status[(prev_r == 0) & (curr_r > 0)] = "Novos"
    status[(prev_r > 0) & (curr_r == 0)] = "Perdidos"
    status[(prev_r > 0) & (curr_r > prev_r * (1 + thr))] = "Crescendo"
    status[(prev_r > 0) & (curr_r < prev_r * (1 - thr))] = "Caindo"

    m["status"] = status
    return m.rename(columns={COL["client"]: "cliente"})

def carteira_health_score(status_df: pd.DataFrame):
    counts = status_df["status"].value_counts().to_dict()
    total = max(1, len(status_df))
    share = {k: counts.get(k, 0) / total for k in ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]}

    score = (
        50
        + 25 * share["Crescendo"]
        + 10 * share["Novos"]
        + 15 * share["Estáveis"]
        - 25 * share["Caindo"]
        - 40 * share["Perdidos"]
    )
    return float(np.clip(score, 0, 100)), counts, share

# ============================================================
# LOAD BASE DATA
# ============================================================
df = load_and_normalize_base(DATA_PATH)

st.title("Relatório do Representante")

if df.empty:
    st.error("Não consegui carregar/normalizar o CSV do repo.")
    st.write("Caminho esperado:", DATA_PATH)
    st.write("Existe no container?:", os.path.exists(DATA_PATH))
    st.stop()

# available years
years_available = sorted(df["data"].dt.year.unique().tolist())
max_date = df["data"].max()
default_curr_year = int(max_date.year)
default_curr_month = int(max_date.month)
default_prev_year = default_curr_year - 1
default_prev_month = default_curr_month

# reps
reps = sorted(df[COL["rep"]].dropna().unique().tolist())
rep_options = ["Todos"] + reps

# ============================================================
# SIDEBAR (match screenshot layout)
# ============================================================
with st.sidebar:
    st.markdown("## Filtros – Insights de Vendas")

    st.markdown("### Período atual")
    st.caption("Período inicial")
    a1, a2 = st.columns(2)
    with a1:
        curr_start_month_lbl = st.selectbox("Mês", MONTHS, index=default_curr_month - 1, key="curr_start_month")
    with a2:
        curr_start_year = st.selectbox(
            "Ano",
            years_available,
            index=years_available.index(default_curr_year),
            key="curr_start_year",
        )

    st.caption("Período final")
    b1, b2 = st.columns(2)
    with b1:
        curr_end_month_lbl = st.selectbox("Mês ", MONTHS, index=default_curr_month - 1, key="curr_end_month")
    with b2:
        curr_end_year = st.selectbox(
            "Ano ",
            years_available,
            index=years_available.index(default_curr_year),
            key="curr_end_year",
        )

    st.markdown("### Período anterior (manual)")
    st.caption("Período anterior – inicial")
    c1, c2 = st.columns(2)
    with c1:
        prev_start_month_lbl = st.selectbox("Mês (anterior – início)", MONTHS, index=default_prev_month - 1, key="prev_start_month")
    with c2:
        prev_start_year = st.selectbox(
            "Ano (anterior – início)",
            years_available,
            index=years_available.index(default_prev_year) if default_prev_year in years_available else 0,
            key="prev_start_year",
        )

    st.caption("Período anterior – final")
    d1, d2 = st.columns(2)
    with d1:
        prev_end_month_lbl = st.selectbox("Mês (anterior – fim)", MONTHS, index=default_prev_month - 1, key="prev_end_month")
    with d2:
        prev_end_year = st.selectbox(
            "Ano (anterior – fim)",
            years_available,
            index=years_available.index(default_prev_year) if default_prev_year in years_available else 0,
            key="prev_end_year",
        )

    st.caption("Representante")
    rep_selected = st.selectbox("Representante", rep_options, index=0, key="rep_selected")

# build timestamps
curr_start = month_start(curr_start_year, MONTH_TO_INT[curr_start_month_lbl])
curr_end = month_end(curr_end_year, MONTH_TO_INT[curr_end_month_lbl])
prev_start = month_start(prev_start_year, MONTH_TO_INT[prev_start_month_lbl])
prev_end = month_end(prev_end_year, MONTH_TO_INT[prev_end_month_lbl])

# validate
if curr_start > curr_end:
    st.error("Período atual inválido: início maior que fim.")
    st.stop()

if prev_start > prev_end:
    st.error("Período anterior inválido: início maior que fim.")
    st.stop()

# filter rep
df_work = df.copy()
if rep_selected != "Todos":
    df_work = df_work[df_work[COL["rep"]] == rep_selected].copy()

# slice periods
df_curr = filter_period_months(df_work, curr_start, curr_end)
df_prev = filter_period_months(df_work, prev_start, prev_end)

# ============================================================
# COMPUTE METRICS
# ============================================================
dist_df, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10 = client_distribution(df_curr)
status_df = build_client_status(df_curr, df_prev, thr=0.10)
health_score, health_counts, health_share = carteira_health_score(status_df)

curr_rev = float(df_curr[COL["rev"]].fillna(0).sum())
curr_vol = float(df_curr[COL["vol"]].fillna(0).sum())
prev_rev = float(df_prev[COL["rev"]].fillna(0).sum())
prev_vol = float(df_prev[COL["vol"]].fillna(0).sum())

clients_attended = int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["client"]].nunique())
cities_attended = int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["city"]].nunique())

# ============================================================
# 1) PERFORMANCE DASHBOARD
# ============================================================
st.subheader("1. Performance Dashboard")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Faturamento Total", money(curr_rev), pct(pct_change(curr_rev, prev_rev)))
k2.metric("Volume Total", num(curr_vol), pct(pct_change(curr_vol, prev_vol)))
k3.metric("N80 (qtd clientes)", f"{n80_count}", pct(n80_share_clients))
k4.metric("Saúde da Carteira (0-100)", f"{health_score:.0f}")
k5.metric("Clientes Atendidos", f"{clients_attended}")
k6.metric("Cidades Atendidas", f"{cities_attended}")

# ============================================================
# 2) EVOLUÇÃO
# ============================================================
st.subheader("2. Evolução (comparativo)")

df_curr_m = df_curr.copy()
df_curr_m["mes"] = df_curr_m["data"].dt.to_period("M").dt.to_timestamp()
evo_curr = df_curr_m.groupby("mes", as_index=False).agg(faturamento=(COL["rev"], "sum"), volume=(COL["vol"], "sum"))
evo_curr["periodo"] = "Atual"

df_prev_m = df_prev.copy()
df_prev_m["mes"] = df_prev_m["data"].dt.to_period("M").dt.to_timestamp()
evo_prev = df_prev_m.groupby("mes", as_index=False).agg(faturamento=(COL["rev"], "sum"), volume=(COL["vol"], "sum"))
evo_prev["periodo"] = "Anterior"

evo = pd.concat([evo_curr, evo_prev], ignore_index=True)

left, right = st.columns([2, 1])

with left:
    base = alt.Chart(evo).encode(x=alt.X("mes:T", title="Mês"))
    line_rev = base.mark_line().encode(y=alt.Y("faturamento:Q", title="Faturamento"), color="periodo:N")
    line_vol = base.mark_line(strokeDash=[4, 2]).encode(y=alt.Y("volume:Q", title="Volume"), color="periodo:N")
    st.altair_chart((line_rev + line_vol).interactive(), use_container_width=True)

with right:
    st.markdown("**Categorias vendidas (período atual)**")
    cat_curr = (
        df_curr.groupby(COL["category"], as_index=False)[COL["rev"]]
        .sum()
        .sort_values(COL["rev"], ascending=False)
        .head(12)
    )
    if cat_curr.empty:
        st.info("Sem categorias no período.")
    else:
        bar = alt.Chart(cat_curr).mark_bar().encode(
            x=alt.X(f"{COL['rev']}:Q", title="Faturamento"),
            y=alt.Y(f"{COL['category']}:N", sort="-x", title="Categoria"),
        )
        st.altair_chart(bar, use_container_width=True)

# ============================================================
# 3) MAPA DE CLIENTES (offline using repo geo files)
# ============================================================
st.subheader("3. Mapa de Clientes")

geo_clientes, geo_cidades = load_geo_tables()

df_map = df_curr.copy()

# normalize keys from main df
df_map["_cliente_key"] = df_map[COL["client"]].astype(str).str.strip().str.upper()
df_map["_cidade_key"] = df_map[COL["city"]].astype(str).str.strip().str.upper()
df_map["_estado_key"] = df_map[COL["state"]].astype(str).str.strip().str.upper()

# 1) merge by client if possible
if not geo_clientes.empty:
    df_map = df_map.merge(
        geo_clientes,
        left_on="_cliente_key",
        right_on="cliente_key",
        how="left",
    )
else:
    df_map["lat"] = np.nan
    df_map["lon"] = np.nan

# 2) fallback: merge by city/uf for missing coords
if not geo_cidades.empty:
    miss = df_map["lat"].isna() | df_map["lon"].isna()
    if miss.any():
        fb = df_map.loc[miss, ["_cidade_key", "_estado_key"]].copy()
        fb = fb.merge(
            geo_cidades,
            left_on=["_cidade_key", "_estado_key"],
            right_on=["cidade_key", "estado_key"],
            how="left",
        )
        df_map.loc[miss, "lat"] = fb["lat"].values
        df_map.loc[miss, "lon"] = fb["lon"].values

# aggregate points by city to avoid overplot
pts = (
    df_map.dropna(subset=["lat", "lon"])
          .groupby(["_cidade_key", "_estado_key", "lat", "lon"], as_index=False)
          .agg(
              faturamento=(COL["rev"], "sum"),
              volume=(COL["vol"], "sum"),
              clientes=(COL["client"], "nunique"),
          )
)

if pts.empty:
    st.warning(
        "Não encontrei coordenadas para plotar. "
        "Verifique se os arquivos geo possuem colunas de latitude/longitude."
    )
    with st.expander("Debug", expanded=False):
        st.write("Existe clientes geo?:", os.path.exists(GEO_CLIENTES_PATH))
        st.write("Existe cidades geo?:", os.path.exists(GEO_CIDADES_PATH))
        st.write("geo_clientes cols:", list(geo_clientes.columns) if not geo_clientes.empty else "geo_clientes vazio")
        st.write("geo_cidades cols:", list(geo_cidades.columns) if not geo_cidades.empty else "geo_cidades vazio")
else:
    st.map(pts.rename(columns={"lat": "latitude", "lon": "longitude"}))

    st.markdown("#### Cidades / Clientes no período")
    show = pts.copy()
    show["Cidade"] = show["_cidade_key"].str.title()
    show["UF"] = show["_estado_key"]
    show["Clientes"] = show["clientes"]
    show["Faturamento"] = show["faturamento"].map(money)
    show["Volume"] = show["volume"].map(num)
    st.dataframe(show[["Cidade", "UF", "Clientes", "Faturamento", "Volume"]], use_container_width=True)

# ============================================================
# 4) DISTRIBUIÇÃO POR CLIENTES
# ============================================================
st.subheader("4. Distribuição por clientes")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Índice de Concentração (HHI)", f"{hhi:.4f}" if not pd.isna(hhi) else "—")
m2.metric("Top 1 cliente", pct(top1))
m3.metric("Top 3 clientes", pct(top3))
m4.metric("Top 10 clientes", pct(top10))

if dist_df.empty:
    st.info("Sem vendas no período.")
else:
    dshow = dist_df.copy()
    dshow["Faturamento"] = dshow["rev"].map(money)
    dshow["Share"] = dshow["share"].map(pct)
    dshow["Share acumulado"] = dshow["cum_share"].map(pct)
    st.dataframe(dshow[["cliente", "Faturamento", "Share", "Share acumulado"]], use_container_width=True)

# ============================================================
# 5) SAÚDE DA CARTEIRA – DETALHES
# ============================================================
st.subheader("5. Saúde da carteira – Detalhes")

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Novos", str(health_counts.get("Novos", 0)), pct(health_share.get("Novos", np.nan)))
s2.metric("Crescendo", str(health_counts.get("Crescendo", 0)), pct(health_share.get("Crescendo", np.nan)))
s3.metric("Estáveis", str(health_counts.get("Estáveis", 0)), pct(health_share.get("Estáveis", np.nan)))
s4.metric("Caindo", str(health_counts.get("Caindo", 0)), pct(health_share.get("Caindo", np.nan)))
s5.metric("Perdidos", str(health_counts.get("Perdidos", 0)), pct(health_share.get("Perdidos", np.nan)))

# ============================================================
# 6) STATUS DOS CLIENTES
# ============================================================
st.subheader("6. Status dos clientes")

t = status_df.copy()
t["Fat (Atual)"] = t["curr_rev"]
t["Fat (Anterior)"] = t["prev_rev"]
t["Δ Fat (R$)"] = t["delta_rev"]
t["Δ Fat (%)"] = t["pct_rev"]
t["Vol (Atual)"] = t["curr_vol"]
t["Vol (Anterior)"] = t["prev_vol"]
t["Δ Vol"] = t["delta_vol"]
t["Δ Vol (%)"] = t["pct_vol"]

order = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
t["status"] = pd.Categorical(t["status"], categories=order, ordered=True)
t = t.sort_values(["status", "curr_rev"], ascending=[True, False]).reset_index(drop=True)

for s in order:
    sub = t[t["status"] == s].copy()
    st.markdown(f"### {s} ({len(sub)})")
    if sub.empty:
        st.caption("—")
        continue

    sub_disp = sub[[
        "cliente",
        "Fat (Atual)", "Fat (Anterior)", "Δ Fat (R$)", "Δ Fat (%)",
        "Vol (Atual)", "Vol (Anterior)", "Δ Vol", "Δ Vol (%)",
    ]].copy()

    for c in ["Fat (Atual)", "Fat (Anterior)", "Δ Fat (R$)"]:
        sub_disp[c] = sub_disp[c].map(money)
    for c in ["Vol (Atual)", "Vol (Anterior)", "Δ Vol"]:
        sub_disp[c] = sub_disp[c].map(num)
    for c in ["Δ Fat (%)", "Δ Vol (%)"]:
        sub_disp[c] = sub_disp[c].map(pct)

    st.dataframe(sub_disp, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.caption(
    f"Período atual: {curr_start.strftime('%d/%m/%Y')} a {curr_end.strftime('%d/%m/%Y')} | "
    f"Período anterior: {prev_start.strftime('%d/%m/%Y')} a {prev_end.strftime('%d/%m/%Y')} | "
    f"Representante: {rep_selected}"
)
