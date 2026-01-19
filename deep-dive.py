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
    "cbeninatto/performance-moveleiro-v2/main/data/raw/relatorio_faturamento.csv"
)

CITY_GEO_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "cbeninatto/performance-moveleiro-v2/main/raw/data/cidades_br_geo.csv"
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

MONTH_MAP_NUM_TO_NAME = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR",
    5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO",
    9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
}
MONTH_MAP_NAME_TO_NUM = {v: k for k, v in MONTH_MAP_NUM_TO_NAME.items()}

# ==========================
# HELPERS
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

def load_data() -> pd.DataFrame:
    cb = int(time.time())
    url = f"{GITHUB_CSV_URL}?cb={cb}"
    resp = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))

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

@st.cache_data(show_spinner=True, ttl=3600)
def load_geo() -> pd.DataFrame:
    cb = int(time.time())
    url = f"{CITY_GEO_CSV_URL}?cb={cb}"
    resp = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=60)
    resp.raise_for_status()

    df_geo = pd.read_csv(io.StringIO(resp.text), sep=None, engine="python")
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
        return pd.DataFrame(columns=["Cliente", "Estado", "Cidade", "ValorAtual", "ValorAnterior", STATUS_COL])

    mask_curr = (df_rep_all["Competencia"] >= start_comp) & (df_rep_all["Competencia"] <= end_comp)
    mask_prev = (df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)

    df_curr = df_rep_all.loc[mask_curr].copy()
    df_prev = df_rep_all.loc[mask_prev].copy()

    curr_agg = (
        df_curr.groupby("Cliente", as_index=False)
        .agg({"Valor": "sum", "Estado": "first", "Cidade": "first"})
        .rename(columns={"Valor": "ValorAtual", "Estado": "EstadoAtual", "Cidade": "CidadeAtual"})
    )
    prev_agg = (
        df_prev.groupby("Cliente", as_index=False)
        .agg({"Valor": "sum", "Estado": "first", "Cidade": "first"})
        .rename(columns={"Valor": "ValorAnterior", "Estado": "EstadoAnterior", "Cidade": "CidadeAnterior"})
    )

    clientes = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer")
    clientes["ValorAtual"] = clientes["ValorAtual"].fillna(0.0)
    clientes["ValorAnterior"] = clientes["ValorAnterior"].fillna(0.0)

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

    return clientes[["Cliente", "Estado", "Cidade", "ValorAtual", "ValorAnterior", STATUS_COL]]

def make_evolucao_chart(df_in: pd.DataFrame, chart_height: int = 300):
    if df_in is None or df_in.empty:
        return None

    ts = df_in.groupby("Competencia", as_index=False)[["Valor", "Quantidade"]].sum().sort_values("Competencia")
    if ts.empty:
        return None

    ts["MesLabelBr"] = ts["Competencia"].apply(lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}")
    ts["VolumeFmt"] = ts["Quantidade"].map(format_un)
    ts["FaturamentoFmt"] = ts["Valor"].map(format_brl)

    x_order = ts["MesLabelBr"].tolist()

    base = alt.Chart(ts).encode(
        x=alt.X("MesLabelBr:N", sort=x_order, axis=alt.Axis(title=None))
    )

    bars = base.mark_bar(color="#38bdf8").encode(
        y=alt.Y("Valor:Q", axis=alt.Axis(title="Faturamento (R$)")),
        tooltip=[
            alt.Tooltip("MesLabelBr:N", title="Mês"),
            alt.Tooltip("FaturamentoFmt:N", title="Faturamento"),
            alt.Tooltip("VolumeFmt:N", title="Volume"),
        ],
    )

    line = base.mark_line(
        color="#22c55e",
        strokeWidth=3,
        point=alt.OverlayMarkDef(color="#22c55e", filled=True, size=70),
    ).encode(
        y=alt.Y("Quantidade:Q", axis=alt.Axis(title="Volume (un)", orient="right")),
        tooltip=[
            alt.Tooltip("MesLabelBr:N", title="Mês"),
            alt.Tooltip("FaturamentoFmt:N", title="Faturamento"),
            alt.Tooltip("VolumeFmt:N", title="Volume"),
        ],
    )

    return alt.layer(bars, line).resolve_scale(y="independent").properties(height=chart_height)

def render_ranking_table(df_rank: pd.DataFrame, value_col: str, value_is_money: bool):
    # df_rank expected cols: Representante, Valor/Quantidade, Clientes, Share, Ranking
    if df_rank.empty:
        st.info("Sem dados para ranquear.")
        return

    view = df_rank.copy()
    view = view.sort_values("Ranking", ascending=True)

    view["%"] = view["Share"].map(lambda x: f"{x:.1%}")
    if value_is_money:
        view[value_col] = view[value_col].map(format_brl)
    else:
        view[value_col] = view[value_col].map(format_un)

    view = view.rename(columns={
        "Ranking": "Ranking",
        "Representante": "Representante",
        value_col: "Faturamento" if value_is_money else "Volume",
        "Clientes": "Clientes",
        "%": "%",
    })

    cols = ["Ranking", "Representante", ("Faturamento" if value_is_money else "Volume"), "%", "Clientes"]
    view = view[cols]

    st.markdown(
        """
<style>
table.rank-table { width: 100%; border-collapse: collapse; }
table.rank-table th, table.rank-table td {
  padding: 0.25rem 0.5rem;
  font-size: 0.86rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  text-align: left;
  white-space: nowrap;
}
table.rank-table th:nth-child(1), table.rank-table td:nth-child(1) { width: 9%; text-align: right; }
table.rank-table th:nth-child(4), table.rank-table td:nth-child(4) { width: 12%; text-align: right; }
table.rank-table th:nth-child(5), table.rank-table td:nth-child(5) { width: 12%; text-align: right; }
</style>
""",
        unsafe_allow_html=True,
    )

    cols_h = list(view.columns)
    html_tbl = "<table class='rank-table'><thead><tr>"
    html_tbl += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_h)
    html_tbl += "</tr></thead><tbody>"
    for _, r in view.iterrows():
        html_tbl += "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols_h) + "</tr>"
    html_tbl += "</tbody></table>"
    st.markdown(html_tbl, unsafe_allow_html=True)

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

# Manual previous period selector (always manual)
st.sidebar.markdown("### Período anterior (manual)")

# defaults suggested = previous window ending right before current start
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
st.title("Insights de Vendas")
titulo_rep = "Todos" if rep_selected == "Todos" else rep_selected
st.subheader(f"Representante: **{titulo_rep}**")
st.caption(f"Período atual: {current_period_label}  |  Período anterior: {previous_period_label}")
st.markdown("---")

# ==========================
# TOP KPIs (5 columns)
# ==========================
col1, col2, col3, col4, col5 = st.columns(5)

total_rep = float(df_rep["Valor"].sum())
total_vol_rep = float(df_rep["Quantidade"].sum())

if not df_rep.empty:
    meses_rep = df_rep.groupby([df_rep["Ano"], df_rep["MesNum"]])["Valor"].sum().reset_index(name="ValorMes")
    meses_com_venda = int((meses_rep["ValorMes"] > 0).sum())
else:
    meses_com_venda = 0

media_mensal = total_rep / meses_com_venda if meses_com_venda > 0 else 0.0

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

    top1_share = float(shares.iloc[:1].sum())
    top3_share = float(shares.iloc[:3].sum())
    top10_share = float(shares.iloc[:10].sum())
else:
    num_clientes_rep = 0
    n80_count = 0
    hhi_value = 0.0
    hhi_label_short = "Sem dados"
    top1_share = 0.0
    top3_share = 0.0
    top10_share = 0.0

clientes_atendidos = int(num_clientes_rep)
cidades_atendidas = int(df_rep[["Estado", "Cidade"]].dropna().drop_duplicates().shape[0]) if not df_rep.empty else 0
estados_atendidos = int(df_rep["Estado"].dropna().nunique()) if not df_rep.empty else 0

if not clientes_carteira.empty:
    carteira_score, carteira_label = compute_carteira_score(clientes_carteira)
else:
    carteira_score, carteira_label = 50.0, "Neutra"

col1.metric("Total período", format_brl_compact(total_rep))
col2.metric("Média mensal", format_brl_compact(media_mensal))
col3.metric("Distribuição por clientes", hhi_label_short, f"N80: {n80_count} clientes")
col4.metric("Saúde da carteira", f"{carteira_score:.0f} / 100", carteira_label)
col5.metric("Clientes atendidos", f"{clientes_atendidos}")

st.markdown("---")

# ==========================
# DESTAQUES DO PERÍODO
# ==========================
st.subheader("Destaques do período")

if df_rep.empty:
    st.info("Não há vendas no período selecionado.")
else:
    mensal_rep = df_rep.groupby(["Ano", "MesNum"], as_index=False)[["Valor", "Quantidade"]].sum()
    mensal_rep["Competencia"] = pd.to_datetime(dict(year=mensal_rep["Ano"], month=mensal_rep["MesNum"], day=1))
    mensal_rep["MesLabel"] = mensal_rep["Competencia"].apply(lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}")

    best_fat = mensal_rep.loc[mensal_rep["Valor"].idxmax()]
    worst_fat = mensal_rep.loc[mensal_rep["Valor"].idxmin()]
    best_vol = mensal_rep.loc[mensal_rep["Quantidade"].idxmax()]
    worst_vol = mensal_rep.loc[mensal_rep["Quantidade"].idxmin()]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Faturamento**")
        st.write(f"• Melhor mês: **{best_fat['MesLabel']}** — {format_brl(best_fat['Valor'])}")
        st.write(f"• Pior mês: **{worst_fat['MesLabel']}** — {format_brl(worst_fat['Valor'])}")
    with c2:
        st.markdown("**Volume**")
        st.write(f"• Melhor mês: **{best_vol['MesLabel']}** — {format_un(best_vol['Quantidade'])}")
        st.write(f"• Pior mês: **{worst_vol['MesLabel']}** — {format_un(worst_vol['Quantidade'])}")

# ==========================
# (NEW) PERFORMANCE DE REPRESENTANTES
# ==========================
st.markdown("---")
st.subheader("Performance de Representantes")

# Current period rep summary (always computed from df_period)
rep_summary_curr = (
    df_period.groupby("Representante", as_index=False)
    .agg(
        Faturamento=("Valor", "sum"),
        Volume=("Quantidade", "sum"),
        Clientes=("Cliente", "nunique"),
    )
)

rep_summary_prev = (
    df_prev_period.groupby("Representante", as_index=False)
    .agg(
        Faturamento=("Valor", "sum"),
        Volume=("Quantidade", "sum"),
        Clientes=("Cliente", "nunique"),
    )
) if not df_prev_period.empty else pd.DataFrame(columns=["Representante", "Faturamento", "Volume", "Clientes"])

if rep_selected == "Todos":
    if rep_summary_curr.empty or float(rep_summary_curr["Faturamento"].sum()) <= 0:
        st.info("Sem dados para ranquear representantes no período atual.")
    else:
        total_team_fat = float(rep_summary_curr["Faturamento"].sum())
        total_team_vol = float(rep_summary_curr["Volume"].sum())

        # Ranking por Faturamento
        fat_rank = rep_summary_curr.copy()
        fat_rank["Share"] = fat_rank["Faturamento"] / (total_team_fat if total_team_fat > 0 else 1.0)
        fat_rank = fat_rank.sort_values("Faturamento", ascending=False).reset_index(drop=True)
        fat_rank["Ranking"] = fat_rank.index + 1
        fat_rank = fat_rank.rename(columns={"Clientes": "Clientes"})

        # Ranking por Volume
        vol_rank = rep_summary_curr.copy()
        vol_rank["Share"] = vol_rank["Volume"] / (total_team_vol if total_team_vol > 0 else 1.0)
        vol_rank = vol_rank.sort_values("Volume", ascending=False).reset_index(drop=True)
        vol_rank["Ranking"] = vol_rank.index + 1
        vol_rank = vol_rank.rename(columns={"Clientes": "Clientes"})

        left, right = st.columns(2)
        with left:
            st.caption("Ranking por Faturamento")
            render_ranking_table(
                fat_rank[["Ranking", "Representante", "Faturamento", "Share", "Clientes"]],
                value_col="Faturamento",
                value_is_money=True,
            )

            with st.expander("PERÍODO ANTERIOR", expanded=False):
                if rep_summary_prev.empty or float(rep_summary_prev["Faturamento"].sum()) <= 0:
                    st.info("Sem dados no período anterior.")
                else:
                    total_prev_fat = float(rep_summary_prev["Faturamento"].sum())
                    fat_rank_prev = rep_summary_prev.copy()
                    fat_rank_prev["Share"] = fat_rank_prev["Faturamento"] / (total_prev_fat if total_prev_fat > 0 else 1.0)
                    fat_rank_prev = fat_rank_prev.sort_values("Faturamento", ascending=False).reset_index(drop=True)
                    fat_rank_prev["Ranking"] = fat_rank_prev.index + 1
                    st.caption(f"Período anterior: {previous_period_label}")
                    render_ranking_table(
                        fat_rank_prev[["Ranking", "Representante", "Faturamento", "Share", "Clientes"]],
                        value_col="Faturamento",
                        value_is_money=True,
                    )

        with right:
            st.caption("Ranking por Volume")
            render_ranking_table(
                vol_rank[["Ranking", "Representante", "Volume", "Share", "Clientes"]],
                value_col="Volume",
                value_is_money=False,
            )

            with st.expander("PERÍODO ANTERIOR", expanded=False):
                if rep_summary_prev.empty or float(rep_summary_prev["Volume"].sum()) <= 0:
                    st.info("Sem dados no período anterior.")
                else:
                    total_prev_vol = float(rep_summary_prev["Volume"].sum())
                    vol_rank_prev = rep_summary_prev.copy()
                    vol_rank_prev["Share"] = vol_rank_prev["Volume"] / (total_prev_vol if total_prev_vol > 0 else 1.0)
                    vol_rank_prev = vol_rank_prev.sort_values("Volume", ascending=False).reset_index(drop=True)
                    vol_rank_prev["Ranking"] = vol_rank_prev.index + 1
                    st.caption(f"Período anterior: {previous_period_label}")
                    render_ranking_table(
                        vol_rank_prev[["Ranking", "Representante", "Volume", "Share", "Clientes"]],
                        value_col="Volume",
                        value_is_money=False,
                    )
else:
    # Specific rep selected: compare vs leader and average
    if rep_summary_curr.empty:
        st.info("Sem dados do time para comparar.")
    else:
        rep_row = rep_summary_curr[rep_summary_curr["Representante"] == rep_selected].head(1)
        if rep_row.empty:
            st.info("Sem dados do representante no período atual.")
        else:
            rep_fat = float(rep_row["Faturamento"].iloc[0])
            rep_vol = float(rep_row["Volume"].iloc[0])
            rep_cli = int(rep_row["Clientes"].iloc[0])

            leader_fat_row = rep_summary_curr.sort_values("Faturamento", ascending=False).head(1)
            leader_vol_row = rep_summary_curr.sort_values("Volume", ascending=False).head(1)

            leader_fat = float(leader_fat_row["Faturamento"].iloc[0]) if not leader_fat_row.empty else 0.0
            leader_vol = float(leader_vol_row["Volume"].iloc[0]) if not leader_vol_row.empty else 0.0
            leader_fat_name = str(leader_fat_row["Representante"].iloc[0]) if not leader_fat_row.empty else "-"
            leader_vol_name = str(leader_vol_row["Representante"].iloc[0]) if not leader_vol_row.empty else "-"

            avg_fat = float(rep_summary_curr["Faturamento"].mean()) if len(rep_summary_curr) else 0.0
            avg_vol = float(rep_summary_curr["Volume"].mean()) if len(rep_summary_curr) else 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faturamento (rep)", format_brl_compact(rep_fat))
            c2.metric("Líder (faturamento)", shorten_name(leader_fat_name, 22), f"{(rep_fat/leader_fat):.0%} do líder" if leader_fat > 0 else "")
            c3.metric("Média do time (faturamento)", format_brl_compact(avg_fat), f"{(rep_fat/avg_fat):.0%} da média" if avg_fat > 0 else "")
            c4.metric("Clientes (rep)", f"{rep_cli}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Volume (rep)", format_un(rep_vol))
            c6.metric("Líder (volume)", shorten_name(leader_vol_name, 22), f"{(rep_vol/leader_vol):.0%} do líder" if leader_vol > 0 else "")
            c7.metric("Média do time (volume)", format_un(avg_vol), f"{(rep_vol/avg_vol):.0%} da média" if avg_vol > 0 else "")
            c8.metric("Participação no time (fat)", f"{(rep_fat/rep_summary_curr['Faturamento'].sum()):.1%}" if rep_summary_curr["Faturamento"].sum() > 0 else "0%")

            with st.expander("PERÍODO ANTERIOR", expanded=False):
                prev_row = rep_summary_prev[rep_summary_prev["Representante"] == rep_selected].head(1)
                if prev_row.empty:
                    st.info("Sem dados do representante no período anterior.")
                else:
                    prev_fat = float(prev_row["Faturamento"].iloc[0])
                    prev_vol = float(prev_row["Volume"].iloc[0])
                    prev_cli = int(prev_row["Clientes"].iloc[0])

                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Faturamento (anterior)", format_brl_compact(prev_fat), format_brl_signed(rep_fat - prev_fat))
                    p2.metric("Volume (anterior)", format_un(prev_vol))
                    p3.metric("Clientes (anterior)", f"{prev_cli}")
                    p4.metric("Participação (anterior)", f"{(prev_fat/rep_summary_prev['Faturamento'].sum()):.1%}" if rep_summary_prev["Faturamento"].sum() > 0 else "0%")

# ==========================
# EVOLUÇÃO – FATURAMENTO x VOLUME (moved right after Destaques)
# ==========================
st.markdown("---")
st.subheader("Evolução – Faturamento x Volume")

fat_curr = float(df_rep["Valor"].sum())
vol_curr = float(df_rep["Quantidade"].sum())
fat_prev = float(df_rep_prev["Valor"].sum()) if not df_rep_prev.empty else 0.0
vol_prev = float(df_rep_prev["Quantidade"].sum()) if not df_rep_prev.empty else 0.0

st.markdown(
    f"**Período atual:** {current_period_label} &nbsp;&nbsp;•&nbsp;&nbsp; "
    f"**Faturamento:** {format_brl_compact(fat_curr)} &nbsp;&nbsp;•&nbsp;&nbsp; "
    f"**Volume:** {format_un(vol_curr)}"
)
chart_curr = make_evolucao_chart(df_rep, chart_height=300)
if chart_curr is None:
    st.info("Sem dados para exibir no período atual.")
else:
    st.altair_chart(chart_curr, width="stretch")

with st.expander("PERÍODO ANTERIOR", expanded=False):
    st.markdown(
        f"**Período anterior:** {previous_period_label} &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"**Faturamento:** {format_brl_compact(fat_prev)} &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"**Volume:** {format_un(vol_prev)}"
    )
    chart_prev = make_evolucao_chart(df_rep_prev, chart_height=300)
    if chart_prev is None:
        st.info("Sem dados para exibir no período anterior.")
    else:
        st.altair_chart(chart_prev, width="stretch")

# ==========================
# CATEGORIAS VENDIDAS (moved right after Evolução)
# ==========================
st.markdown("---")
st.subheader("Categorias vendidas")

def render_categorias(df_curr: pd.DataFrame, df_prev_in: pd.DataFrame, caption_prefix: str, key_suffix: str):
    if df_curr.empty:
        st.info("Não há vendas no período.")
        return

    curr_cat = (
        df_curr.groupby("Categoria", as_index=False)["Valor"]
        .sum()
        .rename(columns={"Valor": "ValorAtual"})
    )
    prev_cat = (
        df_prev_in.groupby("Categoria", as_index=False)["Valor"]
        .sum()
        .rename(columns={"Valor": "ValorAnterior"})
    ) if df_prev_in is not None and not df_prev_in.empty else pd.DataFrame(columns=["Categoria", "ValorAnterior"])

    cat = pd.merge(curr_cat, prev_cat, on="Categoria", how="outer")
    cat["ValorAtual"] = pd.to_numeric(cat["ValorAtual"], errors="coerce").fillna(0.0)
    cat["ValorAnterior"] = pd.to_numeric(cat["ValorAnterior"], errors="coerce").fillna(0.0)

    total_cat = float(cat["ValorAtual"].sum())
    if total_cat <= 0:
        st.info("Sem faturamento para exibir categorias.")
        return

    cat["%"] = cat["ValorAtual"] / total_cat
    cat["Variação (R$)"] = cat["ValorAtual"] - cat["ValorAnterior"]

    def pct_growth(row):
        prevv = float(row["ValorAnterior"])
        currv = float(row["ValorAtual"])
        if prevv > 0:
            return (currv - prevv) / prevv
        if currv > 0 and prevv == 0:
            return None
        return 0.0

    cat["% Crescimento"] = cat.apply(pct_growth, axis=1)
    cat = cat.sort_values("ValorAtual", ascending=False)

    col_pie_cat, col_tbl_cat = st.columns([1.0, 1.25])

    with col_pie_cat:
        st.caption(f"{caption_prefix} – Participação por categoria")

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

    with col_tbl_cat:
        st.caption(f"{caption_prefix} – Resumo (com crescimento vs. período anterior)")

        cat_disp = cat.copy()
        cat_disp["Valor"] = cat_disp["ValorAtual"].map(format_brl)
        cat_disp["%"] = cat_disp["%"].map(lambda x: f"{x:.1%}")
        cat_disp["Variação (R$)"] = cat_disp["Variação (R$)"].map(format_brl_signed)

        def fmt_growth(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "Novo"
            return f"{v:+.1%}"

        cat_disp["Crescimento vs anterior"] = cat_disp["% Crescimento"].apply(fmt_growth)
        cat_disp = cat_disp[["Categoria", "Valor", "%", "Variação (R$)", "Crescimento vs anterior"]]

        st.markdown(
            """
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
""",
            unsafe_allow_html=True,
        )

        cols_k = list(cat_disp.columns)
        html_k = "<table class='cat-resumo'><thead><tr>"
        html_k += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_k)
        html_k += "</tr></thead><tbody>"
        for _, r in cat_disp.iterrows():
            html_k += "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols_k) + "</tr>"
        html_k += "</tbody></table>"
        st.markdown(html_k, unsafe_allow_html=True)

# Main (current vs manual previous)
render_categorias(df_rep, df_rep_prev, caption_prefix="Período atual", key_suffix="curr")

with st.expander("PERÍODO ANTERIOR", expanded=False):
    # Show previous snapshot (still compares to current just to keep the same layout)
    st.caption(f"Período anterior: {previous_period_label}")
    render_categorias(df_rep_prev, df_rep, caption_prefix="Período anterior", key_suffix="prev")

# ==========================
# MAPA DE CLIENTES
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

            metric_choice = st.radio("Métrica do mapa", ["Faturamento", "Volume"], horizontal=True, key="map_metric_choice")
            metric_col = "Valor" if metric_choice == "Faturamento" else "Quantidade"
            metric_label = "Faturamento (R$)" if metric_col == "Valor" else "Volume (un)"

            if df_map[metric_col].max() <= 0:
                st.info("Sem dados para exibir no mapa nesse período.")
            else:
                bins = build_dynamic_bins(df_map[metric_col].tolist(), is_valor=(metric_col == "Valor"))
                df_map["bin_color"] = df_map[metric_col].apply(lambda v: get_bin_for_value(float(v), bins)["color"])
                legend_entries = [(b["color"], b["label"]) for b in bins]

                col_map, col_stats = st.columns([0.8, 1.2])

                with col_map:
                    center = [df_map["lat"].mean(), df_map["lon"].mean()]
                    m = folium.Map(location=center, zoom_start=5, tiles=None)
                    folium.TileLayer(tiles=OSM_TILE_URL, attr=OSM_ATTR, name="OpenStreetMap", control=False).add_to(m)

                    for _, row in df_map.iterrows():
                        color = row["bin_color"]
                        metric_val_str = format_brl(row["Valor"]) if metric_col == "Valor" else format_un(row["Quantidade"])
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
                        df_rep.groupby(["Cliente", "Estado", "Cidade"], as_index=False)["Valor"]
                        .sum()
                        .sort_values("Valor", ascending=False)
                        .head(15)
                    )
                    df_top_clients["Faturamento"] = df_top_clients["Valor"].map(format_brl)
                    df_top_display = df_top_clients[["Cliente", "Cidade", "Estado", "Faturamento"]]

                    st.markdown(
                        """
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
""",
                        unsafe_allow_html=True,
                    )

                    cols_top = list(df_top_display.columns)
                    html_top = "<table class='principais-clientes'><thead><tr>"
                    html_top += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_top)
                    html_top += "</tr></thead><tbody>"
                    for _, r in df_top_display.iterrows():
                        html_top += "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols_top) + "</tr>"
                    html_top += "</tbody></table>"
                    st.markdown(html_top, unsafe_allow_html=True)

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

                                with st.expander("Ver lista de clientes da cidade", expanded=True):
                                    cols_city = list(display_city.columns)
                                    html_city = "<table class='city-table'><thead><tr>"
                                    html_city += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_city)
                                    html_city += "</tr></thead><tbody>"
                                    for _, rr in display_city.iterrows():
                                        html_city += "<tr>" + "".join(f"<td>{html.escape(str(rr[c]))}</td>" for c in cols_city) + "</tr>"
                                    html_city += "</tbody></table>"
                                    st.markdown(html_city, unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Mapa de clientes ainda não disponível: {e}")

# ==========================
# DISTRIBUIÇÃO POR ESTADOS
# ==========================
st.markdown("---")
st.subheader("Distribuição por estados")

def render_estados(df_in: pd.DataFrame, caption_prefix: str, key_suffix: str):
    if df_in.empty:
        st.info("Não há vendas no período selecionado.")
        return

    estados_df = df_in.groupby("Estado", as_index=False)[["Valor", "Quantidade"]].sum().sort_values("Valor", ascending=False)

    total_valor_all = float(estados_df["Valor"].sum())
    total_qtd_all = float(estados_df["Quantidade"].sum())

    if total_valor_all <= 0:
        st.info("Não há faturamento para distribuir por estados nesse período.")
        return

    estados_top = estados_df.head(10).copy()
    estados_top["% Faturamento"] = estados_top["Valor"] / total_valor_all
    estados_top["% Volume"] = estados_top["Quantidade"] / total_qtd_all if total_qtd_all > 0 else 0

    estados_display = estados_top.copy()
    estados_display["Faturamento"] = estados_display["Valor"].map(format_brl)
    estados_display["Volume"] = estados_display["Quantidade"].map(format_un)
    estados_display["% Faturamento"] = estados_display["% Faturamento"].map(lambda x: f"{x:.1%}")
    estados_display["% Volume"] = estados_display["% Volume"].map(lambda x: f"{x:.1%}")
    estados_display = estados_display[["Estado", "Faturamento", "% Faturamento", "Volume", "% Volume"]]

    c_left, c_right = st.columns([1.0, 1.25])

    with c_left:
        st.caption(f"{caption_prefix} – Top 10 estados por faturamento")
        fig_states = px.pie(
            estados_top.sort_values("Valor", ascending=False),
            values="Valor",
            names="Estado",
            hole=0.35,
        )
        fig_states.update_traces(textposition="inside", textinfo="percent+label")
        fig_states.update_layout(showlegend=False, height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_states, width="stretch", key=f"pie_states_{key_suffix}")

    with c_right:
        st.caption(f"{caption_prefix} – Resumo")
        st.markdown(
            """
<style>
table.estados-resumo { width: 100%; border-collapse: collapse; }
table.estados-resumo th, table.estados-resumo td {
  padding: 0.25rem 0.5rem;
  font-size: 0.84rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  text-align: left;
  white-space: nowrap;
}
</style>
""",
            unsafe_allow_html=True,
        )
        cols_e = list(estados_display.columns)
        html_e = "<table class='estados-resumo'><thead><tr>"
        html_e += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_e)
        html_e += "</tr></thead><tbody>"
        for _, r in estados_display.iterrows():
            html_e += "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols_e) + "</tr>"
        html_e += "</tbody></table>"
        st.markdown(html_e, unsafe_allow_html=True)

render_estados(df_rep, "Período atual", "curr")

with st.expander("PERÍODO ANTERIOR", expanded=False):
    st.caption(f"Período anterior: {previous_period_label}")
    render_estados(df_rep_prev, "Período anterior", "prev")

# ==========================
# DISTRIBUIÇÃO POR CLIENTES
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

    k1, k2, k3, k4, k5 = st.columns(5)
    n80_ratio = (n80_local / clientes_atendidos_local) if clientes_atendidos_local > 0 else 0.0
    k1.metric("N80", f"{n80_local}", f"{n80_ratio:.0%} da carteira")
    k2.metric("Índice de concentração", hhi_lbl, f"HHI {hhi_local:.3f}")
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

        st.markdown(
            """
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
""",
            unsafe_allow_html=True,
        )

        cols_c = list(df_tbl.columns)
        html_c = "<table class='clientes-resumo'><thead><tr>"
        html_c += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_c)
        html_c += "</tr></thead><tbody>"
        for _, r in df_tbl.iterrows():
            html_c += "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols_c) + "</tr>"
        html_c += "</tbody></table>"
        st.markdown(html_c, unsafe_allow_html=True)

# Current
render_clientes_dist(df_rep, total_rep, "Período atual", "curr")

with st.expander("PERÍODO ANTERIOR", expanded=False):
    total_prev_rep = float(df_rep_prev["Valor"].sum()) if not df_rep_prev.empty else 0.0
    st.caption(f"Período anterior: {previous_period_label}")
    render_clientes_dist(df_rep_prev, total_prev_rep, "Período anterior", "prev")

# ==========================
# SAÚDE DA CARTEIRA – DETALHES
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

def render_carteira_detalhes(clientes_df: pd.DataFrame, curr_label: str, prev_label: str, key_suffix: str):
    if clientes_df is None or clientes_df.empty:
        st.info("Não há clientes com movimento nos períodos para calcular a carteira.")
        return

    status_counts = (
        clientes_df.groupby(STATUS_COL)["Cliente"]
        .nunique()
        .reset_index()
        .rename(columns={"Cliente": "QtdClientes", STATUS_COL: "Status"})
    )

    fat_status = (
        clientes_df.groupby(STATUS_COL)[["ValorAtual", "ValorAnterior"]]
        .sum()
        .reset_index()
        .rename(columns={STATUS_COL: "Status"})
    )
    fat_status["Faturamento"] = fat_status["ValorAtual"] - fat_status["ValorAnterior"]
    fat_status = fat_status[["Status", "Faturamento"]]
    status_counts = status_counts.merge(fat_status, on="Status", how="left")

    total_clientes = int(status_counts["QtdClientes"].sum())
    status_counts["%Clientes"] = status_counts["QtdClientes"] / total_clientes if total_clientes > 0 else 0
    status_counts["Status"] = pd.Categorical(status_counts["Status"], categories=STATUS_ORDER, ordered=True)
    status_counts = status_counts.sort_values("Status")

    col_pie_s, col_table_s = st.columns([1, 1.2])

    with col_pie_s:
        st.caption("Distribuição de clientes por status")
        if total_clientes == 0:
            st.info("Nenhum cliente com status definido.")
        else:
            chart_pie = (
                alt.Chart(status_counts)
                .mark_arc(outerRadius=120)
                .encode(
                    theta=alt.Theta("QtdClientes:Q"),
                    color=alt.Color(
                        "Status:N",
                        legend=alt.Legend(title="Status"),
                        scale=alt.Scale(
                            domain=["Perdidos", "Caindo", "Estáveis", "Crescendo", "Novos"],
                            range=["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Status:N", title="Status"),
                        alt.Tooltip("QtdClientes:Q", title="Clientes"),
                        alt.Tooltip("%Clientes:Q", title="% Clientes", format=".1%"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_pie, width="stretch")

    with col_table_s:
        st.caption("Resumo por status")

        status_disp = status_counts.copy()
        status_disp["%Clientes"] = status_disp["%Clientes"].map(lambda x: f"{x:.1%}")
        status_disp["Faturamento"] = status_disp["Faturamento"].map(format_brl_signed)
        status_disp = status_disp[["Status", "QtdClientes", "%Clientes", "Faturamento"]]

        st.markdown(
            """
<style>
table.status-resumo { width: 100%; border-collapse: collapse; }
table.status-resumo th, table.status-resumo td {
  padding: 0.25rem 0.5rem;
  font-size: 0.84rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  text-align: left;
  white-space: nowrap;
}
</style>
""",
            unsafe_allow_html=True,
        )
        cols_sr = list(status_disp.columns)
        html_sr = "<table class='status-resumo'><thead><tr>"
        html_sr += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_sr)
        html_sr += "</tr></thead><tbody>"
        for _, r in status_disp.iterrows():
            html_sr += "<tr>" + "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols_sr) + "</tr>"
        html_sr += "</tbody></table>"
        st.markdown(html_sr, unsafe_allow_html=True)

        st.markdown(
            (
                f"<span style='font-size:0.8rem;opacity:0.8;'>"
                f"<b>Obs.:</b> A coluna <b>Faturamento</b> mostra a diferença de faturamento "
                f"entre o período atual (<b>{curr_label}</b>) e o período anterior "
                f"(<b>{prev_label}</b>). Valores positivos indicam crescimento; "
                f"valores negativos indicam queda.</span>"
            ),
            unsafe_allow_html=True,
        )

render_carteira_detalhes(clientes_carteira, current_period_label, previous_period_label, "curr")

with st.expander("PERÍODO ANTERIOR", expanded=False):
    # Show "previous as current" vs "current as reference" (labels will clarify)
    clientes_carteira_prev_view = build_carteira_status_manual_prev(
        df, rep_selected,
        prev_start, prev_end,
        start_comp, end_comp
    )
    st.caption(f"Período anterior: {previous_period_label} (comparado ao período atual: {current_period_label})")
    render_carteira_detalhes(clientes_carteira_prev_view, previous_period_label, current_period_label, "prev")

# ==========================
# PAGE BREAK (print)
# ==========================
st.markdown("<div class='page-break'></div>", unsafe_allow_html=True)

# ==========================
# STATUS DOS CLIENTES
# ==========================
st.markdown("### Status dos clientes")

search_cliente = st.text_input("Buscar cliente", value="", placeholder="Digite parte do nome do cliente", key="buscar_cliente_curr")

table_css = """
<style>
table.status-table { width: 100%; border-collapse: collapse; margin-bottom: 0.75rem; }
table.status-table col:nth-child(1) { width: 30%; }
table.status-table col:nth-child(2) { width: 10%; }
table.status-table col:nth-child(3) { width: 15%; }
table.status-table col:nth-child(4) { width: 10%; }
table.status-table col:nth-child(5) { width: 17.5%; }
table.status-table col:nth-child(6) { width: 17.5%; }
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

# Current period status table (computed vs manual previous)
for status_name in STATUS_ORDER:
    df_status = clientes_carteira[clientes_carteira[STATUS_COL] == status_name].copy() if not clientes_carteira.empty else pd.DataFrame()
    if search_cliente and not df_status.empty:
        df_status = df_status[df_status["Cliente"].astype(str).str.contains(search_cliente, case=False, na=False)]
    if df_status.empty:
        continue

    df_status["FaturamentoAtualFmt"] = df_status["ValorAtual"].map(format_brl)
    df_status["FaturamentoAnteriorFmt"] = df_status["ValorAnterior"].map(format_brl)
    df_status = df_status.sort_values("ValorAtual", ascending=False)

    display_df = df_status[
        ["Cliente", "Estado", "Cidade", STATUS_COL, "FaturamentoAtualFmt", "FaturamentoAnteriorFmt"]
    ].rename(
        columns={
            STATUS_COL: "Status",
            "FaturamentoAtualFmt": f"Faturamento {current_period_label}",
            "FaturamentoAnteriorFmt": f"Faturamento {previous_period_label}",
        }
    )

    cols_status = list(display_df.columns)

    html_status = "<h5>" + html.escape(status_name) + "</h5>"
    html_status += "<table class='status-table'><colgroup><col><col><col><col><col><col></colgroup><thead><tr>"
    html_status += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_status)
    html_status += "</tr></thead><tbody>"
    for _, row in display_df.iterrows():
        html_status += "<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in cols_status) + "</tr>"
    html_status += "</tbody></table>"

    st.markdown(html_status, unsafe_allow_html=True)

with st.expander("PERÍODO ANTERIOR", expanded=False):
    st.caption(f"Status dos clientes – referência: {previous_period_label}")

    # Previous period status (previous as 'current', current as reference for classification)
    clientes_carteira_prev = build_carteira_status_manual_prev(
        df, rep_selected,
        prev_start, prev_end,
        start_comp, end_comp
    )

    if clientes_carteira_prev.empty:
        st.info("Sem dados suficientes para montar o status no período anterior.")
    else:
        search_cliente_prev = st.text_input(
            "Buscar cliente (período anterior)",
            value="",
            placeholder="Digite parte do nome do cliente",
            key="buscar_cliente_prev",
        )

        for status_name in STATUS_ORDER:
            df_status = clientes_carteira_prev[clientes_carteira_prev[STATUS_COL] == status_name].copy()
            if search_cliente_prev:
                df_status = df_status[df_status["Cliente"].astype(str).str.contains(search_cliente_prev, case=False, na=False)]
            if df_status.empty:
                continue

            df_status["FatPrevFmt"] = df_status["ValorAtual"].map(format_brl)
            df_status["FatCurrRefFmt"] = df_status["ValorAnterior"].map(format_brl)
            df_status = df_status.sort_values("ValorAtual", ascending=False)

            display_df = df_status[
                ["Cliente", "Estado", "Cidade", STATUS_COL, "FatPrevFmt", "FatCurrRefFmt"]
            ].rename(
                columns={
                    STATUS_COL: "Status",
                    "FatPrevFmt": f"Faturamento {previous_period_label}",
                    "FatCurrRefFmt": f"Faturamento {current_period_label}",
                }
            )

            cols_status = list(display_df.columns)

            html_status = "<h5>" + html.escape(status_name) + "</h5>"
            html_status += "<table class='status-table'><colgroup><col><col><col><col><col><col></colgroup><thead><tr>"
            html_status += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_status)
            html_status += "</tr></thead><tbody>"
            for _, row in display_df.iterrows():
                html_status += "<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in cols_status) + "</tr>"
            html_status += "</tbody></table>"

            st.markdown(html_status, unsafe_allow_html=True)
