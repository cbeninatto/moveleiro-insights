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

# ==========================
# CONFIG (same as insights.py)
# ==========================
st.set_page_config(
    page_title="Relatório do Representante – Insights",
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

# ==========================
# DATA (repo raw URLs)
# ==========================
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/moveleiro-insights/main/data/raw/relatorio_faturamento.csv"
)

CITY_GEO_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/moveleiro-insights/main/data/raw/cidades_br_geo.csv"
)

# ==========================
# MAP REQUIREMENTS (same)
# ==========================
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

# green = higher, red = lower (same palette)
MAP_BIN_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

MONTH_MAP_NUM_TO_NAME = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR",
    5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO",
    9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
}
MONTH_MAP_NAME_TO_NUM = {v: k for k, v in MONTH_MAP_NUM_TO_NAME.items()}

# ==========================
# HELPERS (same look)
# ==========================
def format_brl(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    return "R$ " + f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

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
    """Force Leaflet 1.9.4 globally for Folium maps."""
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
        return [
            {"min": min_val, "color": color, "label": label_single},
            {"min": min_val, "color": color, "label": label_single},
            {"min": min_val, "color": color, "label": label_single},
            {"min": 0,       "color": color, "label": label_single},
        ]

    def idx(p: float) -> int:
        return int(math.floor(p * (n - 1)))

    q1 = cleaned[idx(0.25)]
    q2 = cleaned[idx(0.5)]
    q3 = cleaned[idx(0.75)]

    t0 = max(0, min_val)
    t1 = max(t0, q1)
    t2 = max(t1, q2)
    t3 = max(t2, q3)

    return [
        {"min": t3, "color": MAP_BIN_COLORS[0], "label": f"{fmt(t3)}+"},
        {"min": t2, "color": MAP_BIN_COLORS[1], "label": f"{fmt(t2)} - {fmt(t3)}"},
        {"min": t1, "color": MAP_BIN_COLORS[2], "label": f"{fmt(t1)} - {fmt(t2)}"},
        {"min": t0, "color": MAP_BIN_COLORS[3], "label": f"{fmt(t0)} - {fmt(t1)}"},
    ]

def get_bin_for_value(v: float, bins):
    for b in bins:
        if v >= b["min"]:
            return b
    return bins[-1]

def format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    def fmt(d: pd.Timestamp) -> str:
        return f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}"
    if start.year == end.year and start.month == end.month:
        return fmt(start)
    return f"{fmt(start)} - {fmt(end)}"

def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def html_table(css: str, df_show: pd.DataFrame, table_class: str):
    if css:
        st.markdown(css, unsafe_allow_html=True)

    cols = list(df_show.columns)
    out = f"<table class='{table_class}'><thead><tr>"
    out += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    out += "</tr></thead><tbody>"
    for _, r in df_show.iterrows():
        out += "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>"
    out += "</tbody></table>"
    st.markdown(out, unsafe_allow_html=True)

# ==========================
# LOADERS
# ==========================
def load_data() -> pd.DataFrame:
    cb = int(time.time())
    url = f"{GITHUB_CSV_URL}?cb={cb}"
    resp = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]

    expected = [
        "Codigo", "Descricao", "Quantidade", "Valor", "Mes", "Ano",
        "ClienteCodigo", "Cliente", "Estado", "Cidade",
        "RepresentanteCodigo", "Representante", "Categoria", "SourcePDF",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError("CSV do GitHub não tem as colunas esperadas: " + ", ".join(missing))

    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")

    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce",
    )
    return df

@st.cache_data(show_spinner=True, ttl=3600)
def load_geo() -> pd.DataFrame:
    cb = int(time.time())
    url = f"{CITY_GEO_CSV_URL}?cb={cb}"
    resp = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=60)
    resp.raise_for_status()

    df_geo = pd.read_csv(io.StringIO(resp.text), sep=None, engine="python")
    df_geo.columns = [c.lstrip("\ufeff").strip() for c in df_geo.columns]
    original_cols = list(df_geo.columns)

    norm_map = {_norm_col(c): c for c in df_geo.columns}

    def pick(candidates):
        for cand in candidates:
            if cand in norm_map:
                return norm_map[cand]
        for nk, orig in norm_map.items():
            for cand in candidates:
                if cand in nk:
                    return orig
        return None

    estado_col = pick(["estado", "uf", "siglauf", "ufsigla", "unidadefederativa", "estadouf", "ufestado", "coduf"])
    cidade_col = pick(["cidade", "municipio", "nomemunicipio", "nmmunicipio", "nomecidade", "cidadenome", "municipionome"])
    lat_col = pick(["lat", "latitude", "y", "coordy", "coordenaday"])
    lon_col = pick(["lon", "lng", "long", "longitude", "x", "coordx", "coordenadax"])

    missing = []
    if estado_col is None: missing.append("Estado/UF")
    if cidade_col is None: missing.append("Cidade/Municipio")
    if lat_col is None: missing.append("Latitude (lat)")
    if lon_col is None: missing.append("Longitude (lon)")
    if missing:
        raise ValueError(
            "cidades_br_geo.csv está com colunas diferentes das esperadas.\n"
            f"Faltando: {', '.join(missing)}\n"
            f"Colunas encontradas: {', '.join(map(str, original_cols))}"
        )

    df_geo = df_geo[[estado_col, cidade_col, lat_col, lon_col]].rename(
        columns={estado_col: "Estado", cidade_col: "Cidade", lat_col: "lat", lon_col: "lon"}
    )

    df_geo["lat"] = df_geo["lat"].astype(str).str.replace(",", ".", regex=False)
    df_geo["lon"] = df_geo["lon"].astype(str).str.replace(",", ".", regex=False)
    df_geo["lat"] = pd.to_numeric(df_geo["lat"], errors="coerce")
    df_geo["lon"] = pd.to_numeric(df_geo["lon"], errors="coerce")
    df_geo = df_geo.dropna(subset=["lat", "lon"])

    df_geo["key"] = (
        df_geo["Estado"].astype(str).str.strip().str.upper()
        + "|"
        + df_geo["Cidade"].astype(str).str.strip().str.upper()
    )
    return df_geo[["key", "Estado", "Cidade", "lat", "lon"]]

# ==========================
# CARTEIRA STATUS + SCORE
# ==========================
def build_carteira_status_manual_prev(
    df_all: pd.DataFrame,
    rep: str,
    start_comp: pd.Timestamp,
    end_comp: pd.Timestamp,
    prev_start: pd.Timestamp,
    prev_end: pd.Timestamp
) -> pd.DataFrame:
    if rep is None or rep == "Todos":
        df_rep_all = df_all.copy()
    else:
        df_rep_all = df_all[df_all["Representante"] == rep].copy()

    if df_rep_all.empty:
        return pd.DataFrame(columns=[
            "Cliente", "Estado", "Cidade",
            "ValorAtual", "ValorAnterior",
            "QuantidadeAtual", "QuantidadeAnterior",
            STATUS_COL
        ])

    mask_curr = (df_rep_all["Competencia"] >= start_comp) & (df_rep_all["Competencia"] <= end_comp)
    mask_prev = (df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)

    df_curr = df_rep_all.loc[mask_curr].copy()
    df_prev = df_rep_all.loc[mask_prev].copy()

    curr_agg = (
        df_curr.groupby("Cliente", as_index=False)
        .agg({"Valor": "sum", "Quantidade": "sum", "Estado": "first", "Cidade": "first"})
        .rename(columns={"Valor": "ValorAtual", "Quantidade": "QuantidadeAtual", "Estado": "EstadoAtual", "Cidade": "CidadeAtual"})
    )
    prev_agg = (
        df_prev.groupby("Cliente", as_index=False)
        .agg({"Valor": "sum", "Quantidade": "sum", "Estado": "first", "Cidade": "first"})
        .rename(columns={"Valor": "ValorAnterior", "Quantidade": "QuantidadeAnterior", "Estado": "EstadoAnterior", "Cidade": "CidadeAnterior"})
    )

    clientes = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer")
    for c in ["ValorAtual", "ValorAnterior", "QuantidadeAtual", "QuantidadeAnterior"]:
        clientes[c] = pd.to_numeric(clientes.get(c, 0.0), errors="coerce").fillna(0.0)

    clientes["Estado"] = clientes["EstadoAtual"].combine_first(clientes["EstadoAnterior"]).fillna("")
    clientes["Cidade"] = clientes["CidadeAtual"].combine_first(clientes["CidadeAnterior"]).fillna("")

    def classify(row):
        va, vp = float(row["ValorAtual"]), float(row["ValorAnterior"])
        if va > 0 and vp == 0:
            return "Novos"
        if va == 0 and vp > 0:
            return "Perdidos"
        if va > 0 and vp > 0:
            ratio = va / vp if vp != 0 else 0.0
            if ratio >= 1.2:
                return "Crescendo"
            elif ratio <= 0.8:
                return "Caindo"
            else:
                return "Estáveis"
        return "Estáveis"

    clientes[STATUS_COL] = clientes.apply(classify, axis=1)
    clientes = clientes[(clientes["ValorAtual"] > 0) | (clientes["ValorAnterior"] > 0)]

    return clientes[[
        "Cliente", "Estado", "Cidade",
        "ValorAtual", "ValorAnterior",
        "QuantidadeAtual", "QuantidadeAnterior",
        STATUS_COL
    ]]

def compute_carteira_score(clientes_carteira: pd.DataFrame):
    if clientes_carteira is None or clientes_carteira.empty:
        return 50.0, "Neutra"

    df = clientes_carteira.copy()
    for col in ["ValorAtual", "ValorAnterior"]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)

    df["PesoReceita"] = df[["ValorAtual", "ValorAnterior"]].max(axis=1).clip(lower=0)

    if STATUS_COL not in df.columns:
        return 50.0, "Neutra"

    receita_status = df.groupby(STATUS_COL)["PesoReceita"].sum()
    total = float(receita_status.sum())
    if total <= 0:
        return 50.0, "Neutra"

    score_bruto = 0.0
    for status, receita in receita_status.items():
        w = STATUS_WEIGHTS.get(str(status), 0)
        score_bruto += w * (receita / total)

    isc = (score_bruto + 2) / 4 * 100
    isc = max(0.0, min(100.0, isc))

    base_anterior = df[df["ValorAnterior"] > 0].copy()
    base_total = float(base_anterior["PesoReceita"].sum())
    perdidos_mask = df[STATUS_COL].astype(str).str.upper().isin(["PERDIDOS", "PERDIDO"])
    receita_perdida = float(df.loc[perdidos_mask, "PesoReceita"].sum())
    churn = receita_perdida / base_total if base_total > 0 else 0.0

    if churn > 0.20 and isc >= 70:
        isc = 69.0

    if isc < 30:
        label = "Crítica"
    elif isc < 50:
        label = "Alerta"
    elif isc < 70:
        label = "Neutra"
    else:
        label = "Saudável"

    return float(isc), label

# ==========================
# EVOLUÇÃO (split charts, compare periods)
# ==========================
def make_compare_timeseries_chart(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    value_col: str,
    title_axis: str,
    height: int = 300,
):
    # build monthly sums for each period, then overlay (two lines)
    def agg(df_in: pd.DataFrame, period_name: str):
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=["Competencia", "Valor", "MesLabelBr", "Periodo"])
        ts = df_in.groupby("Competencia", as_index=False)[value_col].sum().sort_values("Competencia")
        ts = ts.rename(columns={value_col: "Valor"})
        ts["MesLabelBr"] = ts["Competencia"].apply(lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}")
        ts["Periodo"] = period_name
        return ts

    a = agg(df_curr, "Atual")
    b = agg(df_prev, "Anterior")
    base = pd.concat([a, b], ignore_index=True)
    if base.empty:
        return None

    # Keep x order chronological (by Competencia)
    order = (
        base.sort_values("Competencia")
        .drop_duplicates(subset=["MesLabelBr"])
        ["MesLabelBr"].tolist()
    )

    # Tooltip formatting
    if title_axis.lower().startswith("fatur"):
        base["Fmt"] = base["Valor"].map(format_brl)
    else:
        base["Fmt"] = base["Valor"].map(format_un)

    chart = (
        alt.Chart(base)
        .mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=70))
        .encode(
            x=alt.X("MesLabelBr:N", sort=order, axis=alt.Axis(title=None)),
            y=alt.Y("Valor:Q", axis=alt.Axis(title=title_axis)),
            color=alt.Color("Periodo:N", legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip("Periodo:N", title="Período"),
                alt.Tooltip("MesLabelBr:N", title="Mês"),
                alt.Tooltip("Fmt:N", title=title_axis),
            ],
        )
        .properties(height=height)
    )
    return chart

# ==========================
# KPI helper for colored sublabels (Distribuição por clientes)
# ==========================
st.markdown(
    """
<style>
.kpi-sub {
  font-size: 0.80rem;
  opacity: 0.85;
  margin-top: -0.35rem;
}
.kpi-sub b { font-weight: 600; }
.badge {
  display: inline-block;
  padding: 0.08rem 0.35rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  line-height: 1.2;
  border: 1px solid rgba(255,255,255,0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

def color_for_n80_ratio(r: float) -> str:
    # lower is better (fewer clients needed for 80%)
    if r <= 0.20: return MAP_BIN_COLORS[0]  # green
    if r <= 0.35: return MAP_BIN_COLORS[1]  # yellow
    if r <= 0.50: return MAP_BIN_COLORS[2]  # orange
    return MAP_BIN_COLORS[3]               # red

def color_for_hhi(hhi: float) -> str:
    # lower is better (less concentration)
    if hhi < 0.10: return MAP_BIN_COLORS[0]
    if hhi < 0.20: return MAP_BIN_COLORS[1]
    if hhi < 0.30: return MAP_BIN_COLORS[2]
    return MAP_BIN_COLORS[3]

# ==========================
# LOAD DATA
# ==========================
try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar dados do GitHub: {e}")
    st.stop()

if df.empty:
    st.warning("O arquivo de dados está vazio.")
    st.stop()

# ==========================
# SIDEBAR – FILTERS
# ==========================
st.sidebar.title("Filtros – Insights de Vendas")

anos_disponiveis = sorted(df["Ano"].dropna().unique())
if not anos_disponiveis:
    st.error("Não foi possível identificar anos na base de dados.")
    st.stop()

last_year = int(anos_disponiveis[-1])

meses_ano_default = df.loc[df["Ano"] == last_year, "MesNum"].dropna().unique()
default_start_month_num = int(meses_ano_default.min()) if len(meses_ano_default) else 1
default_end_month_num = int(meses_ano_default.max()) if len(meses_ano_default) else 12

month_names = [MONTH_MAP_NUM_TO_NAME[m] for m in range(1, 13)]

st.sidebar.markdown("### Período atual")

st.sidebar.caption("Período inicial")
col_mi, col_ai = st.sidebar.columns(2)
with col_mi:
    start_month_name = st.selectbox(
        "Mês",
        options=month_names,
        index=month_names.index(MONTH_MAP_NUM_TO_NAME[default_start_month_num]),
        key="start_month",
    )
with col_ai:
    start_year = st.selectbox(
        "Ano",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(last_year),
        key="start_year",
    )

st.sidebar.caption("Período final")
col_mf, col_af = st.sidebar.columns(2)
with col_mf:
    end_month_name = st.selectbox(
        "Mês ",
        options=month_names,
        index=month_names.index(MONTH_MAP_NUM_TO_NAME[default_end_month_num]),
        key="end_month",
    )
with col_af:
    end_year = st.selectbox(
        "Ano ",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(last_year),
        key="end_year",
    )

start_month = MONTH_MAP_NAME_TO_NUM[start_month_name]
end_month = MONTH_MAP_NAME_TO_NUM[end_month_name]

start_comp = pd.Timestamp(year=int(start_year), month=int(start_month), day=1)
end_comp = pd.Timestamp(year=int(end_year), month=int(end_month), day=1)

if start_comp > end_comp:
    st.sidebar.error("Período inicial não pode ser maior que o período final.")
    st.stop()

# Manual previous period selector (manual)
st.sidebar.markdown("### Período anterior (manual)")

months_span_default = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1
_prev_end_default = start_comp - pd.DateOffset(months=1)
_prev_start_default = _prev_end_default - pd.DateOffset(months=months_span_default - 1)

prev_default_start_month = int(_prev_start_default.month)
prev_default_start_year = int(_prev_start_default.year)
prev_default_end_month = int(_prev_end_default.month)
prev_default_end_year = int(_prev_end_default.year)

st.sidebar.caption("Período anterior – inicial")
col_pmi, col_pai = st.sidebar.columns(2)
with col_pmi:
    prev_start_month_name = st.selectbox(
        "Mês (anterior – início)",
        options=month_names,
        index=month_names.index(MONTH_MAP_NUM_TO_NAME.get(prev_default_start_month, 1)),
        key="prev_start_month",
    )
with col_pai:
    prev_start_year = st.selectbox(
        "Ano (anterior – início)",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(prev_default_start_year) if prev_default_start_year in anos_disponiveis else 0,
        key="prev_start_year",
    )

st.sidebar.caption("Período anterior – final")
col_pmf, col_paf = st.sidebar.columns(2)
with col_pmf:
    prev_end_month_name = st.selectbox(
        "Mês (anterior – fim)",
        options=month_names,
        index=month_names.index(MONTH_MAP_NUM_TO_NAME.get(prev_default_end_month, 1)),
        key="prev_end_month",
    )
with col_paf:
    prev_end_year = st.selectbox(
        "Ano (anterior – fim)",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(prev_default_end_year) if prev_default_end_year in anos_disponiveis else 0,
        key="prev_end_year",
    )

prev_start_month = MONTH_MAP_NAME_TO_NUM[prev_start_month_name]
prev_end_month = MONTH_MAP_NAME_TO_NUM[prev_end_month_name]

prev_start = pd.Timestamp(year=int(prev_start_year), month=int(prev_start_month), day=1)
prev_end = pd.Timestamp(year=int(prev_end_year), month=int(prev_end_month), day=1)

if prev_start > prev_end:
    st.sidebar.error("Período anterior: início não pode ser maior que o fim.")
    st.stop()

current_period_label = format_period_label(start_comp, end_comp)
previous_period_label = format_period_label(prev_start, prev_end)

# Filter data
mask_period = (df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)
df_period = df.loc[mask_period].copy()
if df_period.empty:
    st.warning("Nenhuma venda no período selecionado.")
    st.stop()

reps_period = sorted(df_period["Representante"].dropna().unique())
if not reps_period:
    st.error("Não há representantes com vendas no período selecionado.")
    st.stop()

rep_options = ["Todos"] + reps_period
rep_selected = st.sidebar.selectbox("Representante", rep_options, key="rep_selected")

df_rep = df_period.copy() if rep_selected == "Todos" else df_period[df_period["Representante"] == rep_selected].copy()

mask_prev_period = (df["Competencia"] >= prev_start) & (df["Competencia"] <= prev_end)
df_prev_period = df.loc[mask_prev_period].copy()
df_rep_prev = df_prev_period.copy() if rep_selected == "Todos" else df_prev_period[df_prev_period["Representante"] == rep_selected].copy()

clientes_carteira = build_carteira_status_manual_prev(df, rep_selected, start_comp, end_comp, prev_start, prev_end)

# ==========================
# HEADER
# ==========================
st.title("Relatório do Representante – Insights")
titulo_rep = "Todos" if rep_selected == "Todos" else rep_selected
st.subheader(f"Representante: **{titulo_rep}**")
st.caption(f"Período atual: {current_period_label}  |  Período anterior: {previous_period_label}")
st.markdown("---")

# ==========================
# TOP KPIs (remove Média mensal)
# ==========================
col1, col2, col3, col4, col5 = st.columns(5)

total_rep = float(df_rep["Valor"].sum())
total_vol_rep = float(df_rep["Quantidade"].sum())

# Distribuição por clientes (N80 + HHI label)
if not df_rep.empty and total_rep > 0:
    df_clientes_tot = df_rep.groupby("Cliente", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False)
    num_clientes_rep = int(df_clientes_tot["Cliente"].nunique())

    shares = df_clientes_tot["Valor"] / total_rep
    cum_share = shares.cumsum()

    n80_count = 0
    for i, v in enumerate(cum_share, start=1):
        n80_count = i
        if v >= 0.8:
            break

    hhi_value = float((shares ** 2).sum())
    if hhi_value < 0.10:
        hhi_label_short = "Baixa"
    elif hhi_value < 0.20:
        hhi_label_short = "Moderada"
    else:
        hhi_label_short = "Alta"
else:
    num_clientes_rep = 0
    n80_count = 0
    hhi_value = 0.0
    hhi_label_short = "Sem dados"

clientes_atendidos = int(num_clientes_rep)
cidades_atendidas = int(df_rep[["Estado", "Cidade"]].dropna().drop_duplicates().shape[0]) if not df_rep.empty else 0
estados_atendidos = int(df_rep["Estado"].dropna().nunique()) if not df_rep.empty else 0

if not clientes_carteira.empty:
    carteira_score, carteira_label = compute_carteira_score(clientes_carteira)
else:
    carteira_score, carteira_label = 50.0, "Neutra"

col1.metric("Total período", format_brl_compact(total_rep))
col2.metric("Volume período", format_un(total_vol_rep))
col3.metric("Distribuição por clientes", hhi_label_short, f"N80: {n80_count} clientes")
col4.metric("Saúde da carteira", f"{carteira_score:.0f} / 100", carteira_label)
col5.metric("Clientes atendidos", f"{clientes_atendidos}")

st.markdown("---")

# ==========================
# EVOLUÇÃO – split into 2 charts (each compares both periods)
# ==========================
st.subheader("Evolução – Faturamento e Volume (comparativo)")

c_a, c_b = st.columns(2)

with c_a:
    st.caption("Faturamento (Atual x Anterior)")
    chart_fat = make_compare_timeseries_chart(df_rep, df_rep_prev, "Valor", "Faturamento (R$)", height=320)
    if chart_fat is None:
        st.info("Sem dados para exibir.")
    else:
        st.altair_chart(chart_fat, width="stretch")

with c_b:
    st.caption("Volume (Atual x Anterior)")
    chart_vol = make_compare_timeseries_chart(df_rep, df_rep_prev, "Quantidade", "Volume (un)", height=320)
    if chart_vol is None:
        st.info("Sem dados para exibir.")
    else:
        st.altair_chart(chart_vol, width="stretch")

# ==========================
# CATEGORIAS VENDIDAS
# - Current: table includes Valor Anterior and % Anterior
# - Previous snapshot: show only Categoria, Valor, %
# - No "PERÍODO ANTERIOR" tabs
# ==========================
st.markdown("---")
st.subheader("Categorias vendidas")

def categorias_snapshot(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["Categoria", "Valor", "%"])
    cat = df_in.groupby("Categoria", as_index=False)["Valor"].sum()
    total = float(cat["Valor"].sum()) if not cat.empty else 0.0
    total = total if total > 0 else 1.0
    cat["%"] = cat["Valor"] / total
    cat = cat.sort_values("Valor", ascending=False)
    cat["Valor"] = cat["Valor"].map(format_brl)
    cat["%"] = cat["%"].map(lambda x: f"{x:.1%}")
    return cat[["Categoria", "Valor", "%"]]

def render_categorias_current_vs_prev(df_curr: pd.DataFrame, df_prev_in: pd.DataFrame, key_suffix: str):
    if df_curr is None or df_curr.empty:
        st.info("Não há vendas no período.")
        return

    curr_cat = df_curr.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "ValorAtual"})
    prev_cat = df_prev_in.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "ValorAnterior"}) if df_prev_in is not None and not df_prev_in.empty else pd.DataFrame(columns=["Categoria", "ValorAnterior"])

    cat = pd.merge(curr_cat, prev_cat, on="Categoria", how="outer")
    cat["ValorAtual"] = pd.to_numeric(cat["ValorAtual"], errors="coerce").fillna(0.0)
    cat["ValorAnterior"] = pd.to_numeric(cat["ValorAnterior"], errors="coerce").fillna(0.0)

    total_curr = float(cat["ValorAtual"].sum())
    total_prev = float(cat["ValorAnterior"].sum())

    total_curr = total_curr if total_curr > 0 else 1.0
    total_prev_safe = total_prev if total_prev > 0 else 1.0

    cat["% Atual"] = cat["ValorAtual"] / total_curr
    cat["% Anterior"] = cat["ValorAnterior"] / total_prev_safe

    cat = cat.sort_values("ValorAtual", ascending=False)

    left, right = st.columns([1.0, 1.25])

    with left:
        st.caption("Período atual – Participação por categoria")

        df_pie = cat[["Categoria", "ValorAtual"]].copy().rename(columns={"ValorAtual": "Valor"})
        df_pie = df_pie.sort_values("Valor", ascending=False).reset_index(drop=True)

        if len(df_pie) > 10:
            top10 = df_pie.head(10).copy()
            others_val = float(df_pie.iloc[10:]["Valor"].sum())
            top10 = pd.concat([top10, pd.DataFrame([{"Categoria": "Outras", "Valor": others_val}])], ignore_index=True)
            df_pie = top10

        df_pie["Share"] = df_pie["Valor"] / float(df_pie["Valor"].sum()) if float(df_pie["Valor"].sum()) > 0 else 0.0
        df_pie["Legenda"] = df_pie.apply(lambda r: f"{r['Categoria']} {r['Share']*100:.1f}%", axis=1)

        def make_text_cat(row):
            if row["Share"] >= 0.07:
                return f"{row['Categoria']}<br>{row['Share']*100:.1f}%"
            return ""

        df_pie["Text"] = df_pie.apply(make_text_cat, axis=1)
        order_leg = df_pie.sort_values("Share", ascending=False)["Legenda"].tolist()

        fig_cat = px.pie(df_pie, values="Valor", names="Legenda", hole=0.35, category_orders={"Legenda": order_leg})
        fig_cat.update_traces(
            text=df_pie["Text"],
            textposition="inside",
            textinfo="text",
            insidetextorientation="radial",
        )
        fig_cat.update_layout(showlegend=False, height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cat, width="stretch", key=f"pie_cat_{key_suffix}")

    with right:
        st.caption("Período atual – Resumo (inclui comparação de participação vs. período anterior)")

        cat_disp = cat.copy()
        cat_disp["Valor Atual"] = cat_disp["ValorAtual"].map(format_brl)
        cat_disp["Valor Anterior"] = cat_disp["ValorAnterior"].map(format_brl)
        cat_disp["% Atual"] = cat_disp["% Atual"].map(lambda x: f"{x:.1%}")
        cat_disp["% Anterior"] = cat_disp["% Anterior"].map(lambda x: f"{x:.1%}")

        cat_disp = cat_disp[["Categoria", "Valor Atual", "Valor Anterior", "% Atual", "% Anterior"]]

        css = """
<style>
table.cat-resumo { width: 100%; border-collapse: collapse; }
table.cat-resumo th, table.cat-resumo td {
  padding: 0.25rem 0.5rem;
  font-size: 0.84rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  text-align: left;
  white-space: nowrap;
}
</style>
"""
        html_table(css, cat_disp, "cat-resumo")

render_categorias_current_vs_prev(df_rep, df_rep_prev, "curr")

st.markdown("#### Período anterior – snapshot (Categoria, Valor, %)")

p_left, p_right = st.columns([1.0, 1.25])

with p_left:
    st.caption(f"Período anterior ({previous_period_label}) – Participação por categoria")
    prev_snap = categorias_snapshot(df_rep_prev)

    if prev_snap.empty:
        st.info("Sem dados no período anterior.")
    else:
        # build pie from previous snapshot (need raw numeric)
        prev_raw = df_rep_prev.groupby("Categoria", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False)
        if len(prev_raw) > 10:
            top10 = prev_raw.head(10).copy()
            others_val = float(prev_raw.iloc[10:]["Valor"].sum())
            prev_raw = pd.concat([top10, pd.DataFrame([{"Categoria": "Outras", "Valor": others_val}])], ignore_index=True)

        prev_raw["Share"] = prev_raw["Valor"] / float(prev_raw["Valor"].sum()) if float(prev_raw["Valor"].sum()) > 0 else 0.0
        prev_raw["Legenda"] = prev_raw.apply(lambda r: f"{r['Categoria']} {r['Share']*100:.1f}%", axis=1)

        def make_text_prev(row):
            if row["Share"] >= 0.07:
                return f"{row['Categoria']}<br>{row['Share']*100:.1f}%"
            return ""

        prev_raw["Text"] = prev_raw.apply(make_text_prev, axis=1)
        order_leg = prev_raw.sort_values("Share", ascending=False)["Legenda"].tolist()

        fig_prev = px.pie(prev_raw, values="Valor", names="Legenda", hole=0.35, category_orders={"Legenda": order_leg})
        fig_prev.update_traces(
            text=prev_raw["Text"],
            textposition="inside",
            textinfo="text",
            insidetextorientation="radial",
        )
        fig_prev.update_layout(showlegend=False, height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_prev, width="stretch", key="pie_cat_prev_snapshot")

with p_right:
    st.caption(f"Período anterior ({previous_period_label}) – Tabela (Categoria, Valor, %)")
    prev_snap = categorias_snapshot(df_rep_prev)
    if prev_snap.empty:
        st.info("Sem dados no período anterior.")
    else:
        css = """
<style>
table.cat-prev { width: 100%; border-collapse: collapse; }
table.cat-prev th, table.cat-prev td {
  padding: 0.25rem 0.5rem;
  font-size: 0.84rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  text-align: left;
  white-space: nowrap;
}
</style>
"""
        html_table(css, prev_snap, "cat-prev")

# ==========================
# MAPA DE CLIENTES
# - Default to Volume
# - Remove option toggle
# - In "Principais clientes" table add Volume column
# ==========================
st.markdown("---")
st.subheader("Mapa de Clientes")

if "selected_city_tooltip" not in st.session_state:
    st.session_state["selected_city_tooltip"] = None

if df_rep.empty:
    st.info("Não há vendas no período selecionado.")
else:
    try:
        force_leaflet_1_9_4()
        df_geo = load_geo()

        df_cities = df_rep.groupby(["Estado", "Cidade"], as_index=False).agg(
            Valor=("Valor", "sum"),
            Quantidade=("Quantidade", "sum"),
            Clientes=("Cliente", "nunique"),
        )
        df_cities["key"] = (
            df_cities["Estado"].astype(str).str.strip().str.upper()
            + "|"
            + df_cities["Cidade"].astype(str).str.strip().str.upper()
        )

        df_map = df_cities.merge(df_geo, on="key", how="inner", suffixes=("_fat", "_geo"))

        if df_map.empty:
            st.info("Não há coordenadas de cidades para exibir no mapa.")
        else:
            df_map["Tooltip"] = df_map["Cidade_fat"].astype(str) + " - " + df_map["Estado_fat"].astype(str)

            # FIXED: default metric = Volume, no toggle
            metric_col = "Quantidade"
            metric_label = "Volume (un)"

            if df_map[metric_col].max() <= 0:
                st.info("Sem dados para exibir no mapa nesse período.")
            else:
                bins = build_dynamic_bins(df_map[metric_col].tolist(), is_valor=False)
                df_map["bin_color"] = df_map[metric_col].apply(lambda v: get_bin_for_value(float(v), bins)["color"])
                legend_entries = [(b["color"], b["label"]) for b in bins]

                col_map, col_stats = st.columns([0.8, 1.2])

                with col_map:
                    center = [df_map["lat"].mean(), df_map["lon"].mean()]
                    m = folium.Map(location=center, zoom_start=5, tiles=None)
                    folium.TileLayer(tiles=OSM_TILE_URL, attr=OSM_ATTR, name="OpenStreetMap", control=False).add_to(m)

                    for _, row in df_map.iterrows():
                        color = row["bin_color"]
                        metric_val_str = format_un(row["Quantidade"])
                        popup_html = (
                            f"<b>{html.escape(str(row['Cidade_fat']))} - {html.escape(str(row['Estado_fat']))}</b><br>"
                            f"{metric_label}: {html.escape(metric_val_str)}<br>"
                            f"Clientes: {int(row['Clientes'])}"
                        )

                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=6,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.8,
                            popup=folium.Popup(popup_html, max_width=320),
                            tooltip=row["Tooltip"],
                        ).add_to(m)

                    map_data = st_folium(m, width=None, height=800, key="folium_map_main")

                    if legend_entries:
                        legend_html = "<div style='font-size:0.8rem; margin-top:0.5rem;'>"
                        legend_html += f"<b>Legenda – {metric_label}</b><br>"
                        for color, label_range in legend_entries:
                            legend_html += (
                                f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
                                f"margin-right:4px;border-radius:2px;'></span>"
                                f"{html.escape(label_range)}<br>"
                            )
                        legend_html += "</div>"
                        st.markdown(legend_html, unsafe_allow_html=True)

                selected_label = None
                if isinstance(map_data, dict):
                    selected_label = map_data.get("last_object_clicked_tooltip")

                if selected_label:
                    st.session_state["selected_city_tooltip"] = selected_label
                else:
                    selected_label = st.session_state.get("selected_city_tooltip")

                with col_stats:
                    st.markdown("**Cobertura**")
                    cov1, cov2, cov3 = st.columns(3)
                    cov1.metric("Cidades atendidas", f"{cidades_atendidas}")
                    cov2.metric("Estados atendidos", f"{estados_atendidos}")
                    cov3.metric("Clientes atendidos", f"{clientes_atendidos}")

                    st.markdown("**Principais clientes**")
                    df_top_clients = (
                        df_rep.groupby(["Cliente", "Estado", "Cidade"], as_index=False)
                        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
                        .sort_values("Valor", ascending=False)
                        .head(15)
                    )
                    df_top_clients["Faturamento"] = df_top_clients["Valor"].map(format_brl)
                    df_top_clients["Volume"] = df_top_clients["Quantidade"].map(format_un)
                    df_top_display = df_top_clients[["Cliente", "Cidade", "Estado", "Faturamento", "Volume"]]

                    css = """
<style>
table.principais-clientes { width: 100%; border-collapse: collapse; }
table.principais-clientes th, table.principais-clientes td {
  padding: 0.25rem 0.5rem;
  font-size: 0.85rem;
  text-align: left;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  vertical-align: top;
  white-space: nowrap;
}
</style>
"""
                    html_table(css, df_top_display, "principais-clientes")

                    if selected_label:
                        row_city = df_map[df_map["Tooltip"] == selected_label].head(1)
                        if not row_city.empty:
                            cidade_sel = row_city["Cidade_fat"].iloc[0]
                            estado_sel = row_city["Estado_fat"].iloc[0]
                            df_city_clients = df_rep[(df_rep["Cidade"] == cidade_sel) & (df_rep["Estado"] == estado_sel)].copy()

                            if not df_city_clients.empty:
                                df_city_agg = (
                                    df_city_clients.groupby("Cliente", as_index=False)
                                    .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
                                    .sort_values("Valor", ascending=False)
                                )
                                df_city_agg["Faturamento"] = df_city_agg["Valor"].map(format_brl)
                                df_city_agg["QuantidadeFmt"] = df_city_agg["Quantidade"].map(format_un)
                                display_city = df_city_agg[["Cliente", "QuantidadeFmt", "Faturamento"]].rename(
                                    columns={"QuantidadeFmt": "Quantidade"}
                                )

                                st.markdown(f"**Clientes em {cidade_sel} - {estado_sel}**")

                                st.markdown(
                                    """
<style>
table.city-table { width: 100%; border-collapse: collapse; }
table.city-table th, table.city-table td {
  padding: 0.25rem 0.5rem;
  font-size: 0.85rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  white-space: nowrap;
}
table.city-table th:nth-child(2), table.city-table th:nth-child(3) { text-align: center; }
table.city-table td { text-align: left; }
</style>
""",
                                    unsafe_allow_html=True,
                                )

                                # No expander: always show list (same feel as the map panel)
                                html_table("", display_city, "city-table")

    except Exception as e:
        st.info(f"Mapa de clientes ainda não disponível: {e}")

# ==========================
# DISTRIBUIÇÃO POR CLIENTES
# - Color the N80 and HHI sublabels based on thresholds
# - No "PERÍODO ANTERIOR" tab
# ==========================
st.markdown("---")
st.subheader("Distribuição por clientes")

def render_clientes_dist(df_in: pd.DataFrame, total_val: float, caption_prefix: str, key_suffix: str):
    if df_in.empty:
        st.info("Nenhum cliente com vendas no período selecionado.")
        return

    df_clientes_full = (
        df_in.groupby(["Cliente", "Estado", "Cidade"], as_index=False)
        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
        .sort_values("Valor", ascending=False)
    )
    total_safe = total_val if total_val > 0 else float(df_clientes_full["Valor"].sum())
    total_safe = total_safe if total_safe > 0 else 1.0

    df_clientes_full["Share"] = df_clientes_full["Valor"] / total_safe

    # KPIs of distribution
    df_clientes_tot_local = df_clientes_full.groupby("Cliente", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False)
    clientes_atendidos_local = int(df_clientes_tot_local["Cliente"].nunique())

    shares_local = (df_clientes_tot_local["Valor"] / float(df_clientes_tot_local["Valor"].sum())) if float(df_clientes_tot_local["Valor"].sum()) > 0 else pd.Series([])
    if len(shares_local) > 0:
        cum_share = shares_local.cumsum()
        n80_local = 0
        for i, v in enumerate(cum_share, start=1):
            n80_local = i
            if v >= 0.8:
                break
        hhi_local = float((shares_local ** 2).sum())
        top1 = float(shares_local.iloc[:1].sum())
        top3 = float(shares_local.iloc[:3].sum())
        top10 = float(shares_local.iloc[:10].sum())
        if hhi_local < 0.10:
            hhi_lbl = "Baixa"
        elif hhi_local < 0.20:
            hhi_lbl = "Moderada"
        else:
            hhi_lbl = "Alta"
    else:
        n80_local = 0
        hhi_local = 0.0
        top1 = top3 = top10 = 0.0
        hhi_lbl = "Sem dados"

    n80_ratio = (n80_local / clientes_atendidos_local) if clientes_atendidos_local > 0 else 0.0
    n80_color = color_for_n80_ratio(n80_ratio)
    hhi_color = color_for_hhi(hhi_local)

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("N80", f"{n80_local}")
        st.markdown(
            f"<div class='kpi-sub'><span class='badge' style='color:{n80_color}; border-color:rgba(255,255,255,0.12);'>"
            f"{n80_ratio:.0%} da carteira</span></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.metric("Índice de concentração", hhi_lbl, f"HHI {hhi_local:.3f}")
        st.markdown(
            f"<div class='kpi-sub'><span class='badge' style='color:{hhi_color}; border-color:rgba(255,255,255,0.12);'>"
            f"Concentração: {hhi_lbl}</span></div>",
            unsafe_allow_html=True,
        )
    k3.metric("Top 1 cliente", f"{top1:.1%}")
    k4.metric("Top 3 clientes", f"{top3:.1%}")
    k5.metric("Top 10 clientes", f"{top10:.1%}")

    col_pie, col_tbl = st.columns([1.10, 1.50])

    with col_pie:
        st.caption(f"{caption_prefix} – Participação dos clientes (Top 10 destacados)")

        df_pie = df_clientes_full[["Cliente", "Valor"]].copy()
        df_pie = df_pie.groupby("Cliente", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False)
        df_pie["Rank"] = range(1, len(df_pie) + 1)
        df_pie["Grupo"] = df_pie.apply(lambda r: r["Cliente"] if r["Rank"] <= 10 else "Outros", axis=1)

        dist_df = df_pie.groupby("Grupo", as_index=False)["Valor"].sum()
        dist_df["Share"] = dist_df["Valor"] / total_safe
        dist_df = dist_df.sort_values("Share", ascending=False)
        dist_df["Legenda"] = dist_df.apply(lambda r: f"{r['Grupo']} {r['Share']*100:.1f}%", axis=1)

        def make_text(row):
            if row["Share"] >= 0.07:
                return f"{row['Grupo']}<br>{row['Share']*100:.1f}%"
            return ""

        dist_df["Text"] = dist_df.apply(make_text, axis=1)
        order_legenda = dist_df["Legenda"].tolist()

        fig = px.pie(dist_df, values="Valor", names="Legenda", category_orders={"Legenda": order_legenda})
        fig.update_traces(
            text=dist_df["Text"],
            textposition="inside",
            textinfo="text",
            insidetextorientation="radial",
        )
        fig.update_layout(showlegend=False, height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", key=f"pie_clients_{key_suffix}")

    with col_tbl:
        st.caption(f"{caption_prefix} – Resumo (mais detalhado)")

        df_tbl = df_clientes_full.copy()
        df_tbl["Faturamento"] = df_tbl["Valor"].map(format_brl)
        df_tbl["% Faturamento"] = df_tbl["Share"].map(lambda x: f"{x:.1%}")
        df_tbl["Volume"] = df_tbl["Quantidade"].map(format_un)

        df_tbl = df_tbl.head(15)[["Cliente", "Cidade", "Estado", "Faturamento", "% Faturamento", "Volume"]]

        css = """
<style>
table.clientes-resumo { width: 100%; border-collapse: collapse; }
table.clientes-resumo th, table.clientes-resumo td {
  padding: 0.25rem 0.5rem;
  font-size: 0.84rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}
</style>
"""
        html_table(css, df_tbl, "clientes-resumo")

render_clientes_dist(df_rep, total_rep, "Período atual", "curr")

# ==========================
# SAÚDE DA CARTEIRA – DETALHES
# (no previous-period tab)
# ==========================
st.markdown("---")
st.subheader("Saúde da carteira – Detalhes")

c_score1, c_score2 = st.columns([0.35, 0.65])
with c_score1:
    st.metric("Pontuação – Saúde da carteira", f"{carteira_score:.0f} / 100", carteira_label)
with c_score2:
    st.caption(
        "A pontuação reflete a distribuição de receita entre os status (Novos/Crescendo/Estáveis/Caindo/Perdidos) "
        "no comparativo entre período atual e anterior (manual)."
    )

# ==========================
# STATUS DOS CLIENTES
# (no previous-period tab)
# ==========================
st.markdown("---")
st.subheader("Status dos clientes")

search_cliente = st.text_input(
    "Buscar cliente",
    value="",
    placeholder="Digite parte do nome do cliente",
    key="buscar_cliente_curr",
)

table_css = """
<style>
table.status-table { width: 100%; border-collapse: collapse; margin-bottom: 0.75rem; }
table.status-table th, table.status-table td {
    padding: 0.2rem 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    font-size: 0.85rem;
    text-align: left;
    white-space: nowrap;
}
table.status-table th { font-weight: 600; }
</style>
"""
st.markdown(table_css, unsafe_allow_html=True)

def safe_pct(curr, prev):
    prev = float(prev)
    curr = float(curr)
    if prev > 0:
        return f"{((curr-prev)/prev):+.1%}"
    if curr > 0 and prev == 0:
        return "Novo"
    return "0.0%"

for status_name in STATUS_ORDER:
    df_status = clientes_carteira[clientes_carteira[STATUS_COL] == status_name].copy() if not clientes_carteira.empty else pd.DataFrame()
    if search_cliente and not df_status.empty:
        df_status = df_status[df_status["Cliente"].astype(str).str.contains(search_cliente, case=False, na=False)]
    if df_status.empty:
        continue

    df_status = df_status.sort_values("ValorAtual", ascending=False).copy()

    df_status["FatAtualFmt"] = df_status["ValorAtual"].map(format_brl)
    df_status["FatAnteriorFmt"] = df_status["ValorAnterior"].map(format_brl)
    df_status["VolAtualFmt"] = df_status["QuantidadeAtual"].map(format_un)
    df_status["VolAnteriorFmt"] = df_status["QuantidadeAnterior"].map(format_un)

    df_status["Δ Fat (R$)"] = (df_status["ValorAtual"] - df_status["ValorAnterior"]).map(format_brl_signed)
    df_status["Δ Fat (%)"] = df_status.apply(lambda r: safe_pct(r["ValorAtual"], r["ValorAnterior"]), axis=1)
    df_status["Δ Vol (%)"] = df_status.apply(lambda r: safe_pct(r["QuantidadeAtual"], r["QuantidadeAnterior"]), axis=1)

    display_df = df_status[[
        "Cliente", "Estado", "Cidade",
        "FatAtualFmt", "FatAnteriorFmt", "Δ Fat (R$)", "Δ Fat (%)",
        "VolAtualFmt", "VolAnteriorFmt", "Δ Vol (%)"
    ]].rename(
        columns={
            "FatAtualFmt": f"Fat {current_period_label}",
            "FatAnteriorFmt": f"Fat {previous_period_label}",
            "VolAtualFmt": f"Vol {current_period_label}",
            "VolAnteriorFmt": f"Vol {previous_period_label}",
        }
    )

    cols_status = list(display_df.columns)

    html_status = "<h5>" + html.escape(status_name) + "</h5>"
    html_status += "<table class='status-table'><thead><tr>"
    html_status += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_status)
    html_status += "</tr></thead><tbody>"
    for _, row in display_df.iterrows():
        html_status += "<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in cols_status) + "</tr>"
    html_status += "</tbody></table>"

    st.markdown(html_status, unsafe_allow_html=True)
