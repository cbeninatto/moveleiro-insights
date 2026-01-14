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
from datetime import datetime

# PDF / charts export
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# For Altair -> PNG without Chrome
try:
    import vl_convert as vlc
except Exception:
    vlc = None

# For matplotlib fallback (Plotly static export may require Chrome in Kaleido)
import matplotlib.pyplot as plt


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
/* keep streamlit print CSS, but PDF export will be separate */
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
    "Novos": 1,
    "Novo": 1,
    "Crescendo": 2,
    "CRESCENDO": 2,
    "Estáveis": 1,
    "Estável": 1,
    "ESTAVEIS": 1,
    "Caindo": -1,
    "CAINDO": -1,
    "Perdidos": -2,
    "Perdido": -2,
    "PERDIDOS": -2,
}
STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]

# green = higher, red = lower
MAP_BIN_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]


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
                {"min": 500_000, "color": MAP_BIN_COLORS[1], "label": "R$ 500.000 - 999.999"},
                {"min": 250_000, "color": MAP_BIN_COLORS[2], "label": "R$ 250.000 - 499.999"},
                {"min": 0, "color": MAP_BIN_COLORS[3], "label": "R$ 0 - 249.999"},
            ]
        return [
            {"min": 10_000, "color": MAP_BIN_COLORS[0], "label": "10.000 un+"},
            {"min": 5_000, "color": MAP_BIN_COLORS[1], "label": "5.000 - 9.999 un"},
            {"min": 2_500, "color": MAP_BIN_COLORS[2], "label": "2.500 - 4.999 un"},
            {"min": 0, "color": MAP_BIN_COLORS[3], "label": "0 - 2.499 un"},
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
            {"min": 0, "color": color, "label": label_single},
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
        "Codigo",
        "Descricao",
        "Quantidade",
        "Valor",
        "Mes",
        "Ano",
        "ClienteCodigo",
        "Cliente",
        "Estado",
        "Cidade",
        "RepresentanteCodigo",
        "Representante",
        "Categoria",
        "SourcePDF",
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

    df_ = clientes_carteira.copy()
    for col in ["ValorAtual", "ValorAnterior"]:
        df_[col] = pd.to_numeric(df_.get(col, 0.0), errors="coerce").fillna(0.0)

    df_["PesoReceita"] = df_[["ValorAtual", "ValorAnterior"]].max(axis=1).clip(lower=0)

    if STATUS_COL not in df_.columns:
        return 50.0, "Neutra"

    receita_status = df_.groupby(STATUS_COL)["PesoReceita"].sum()
    total = float(receita_status.sum())
    if total <= 0:
        return 50.0, "Neutra"

    score_bruto = 0.0
    for status, receita in receita_status.items():
        w = STATUS_WEIGHTS.get(str(status), 0)
        score_bruto += w * (receita / total)

    isc = (score_bruto + 2) / 4 * 100
    isc = max(0.0, min(100.0, isc))

    base_anterior = df_[df_["ValorAnterior"] > 0].copy()
    base_total = float(base_anterior["PesoReceita"].sum())
    perdidos_mask = df_[STATUS_COL].astype(str).str.upper().isin(["PERDIDOS", "PERDIDO"])
    receita_perdida = float(df_.loc[perdidos_mask, "PesoReceita"].sum())
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


MONTH_MAP_NUM_TO_NAME = {
    1: "JAN",
    2: "FEV",
    3: "MAR",
    4: "ABR",
    5: "MAI",
    6: "JUN",
    7: "JUL",
    8: "AGO",
    9: "SET",
    10: "OUT",
    11: "NOV",
    12: "DEZ",
}
MONTH_MAP_NAME_TO_NUM = {v: k for k, v in MONTH_MAP_NUM_TO_NAME.items()}


def format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    def fmt(d: pd.Timestamp) -> str:
        return f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}"

    if start.year == end.year and start.month == end.month:
        return fmt(start)
    return f"{fmt(start)} - {fmt(end)}"


def build_carteira_status(df_all: pd.DataFrame, rep: str, start_comp: pd.Timestamp, end_comp: pd.Timestamp) -> pd.DataFrame:
    if rep is None or rep == "Todos":
        df_rep_all = df_all.copy()
    else:
        df_rep_all = df_all[df_all["Representante"] == rep].copy()

    if df_rep_all.empty:
        return pd.DataFrame(columns=["Cliente", "Estado", "Cidade", "ValorAtual", "ValorAnterior", STATUS_COL])

    months_span = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1
    prev_end_ = start_comp - pd.DateOffset(months=1)
    prev_start_ = prev_end_ - pd.DateOffset(months=months_span - 1)

    mask_curr = (df_rep_all["Competencia"] >= start_comp) & (df_rep_all["Competencia"] <= end_comp)
    mask_prev = (df_rep_all["Competencia"] >= prev_start_) & (df_rep_all["Competencia"] <= prev_end_)

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
        va, vp = row["ValorAtual"], row["ValorAnterior"]
        if va > 0 and vp == 0:
            return "Novos"
        if va == 0 and vp > 0:
            return "Perdidos"
        if va > 0 and vp > 0:
            ratio = va / vp if vp != 0 else 0
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
    cidade_col = pick(
        ["cidade", "municipio", "nomemunicipio", "nmmunicipio", "nomecidade", "cidadenome", "municipionome"]
    )
    lat_col = pick(["lat", "latitude", "y", "coordy", "coordenaday"])
    lon_col = pick(["lon", "lng", "long", "longitude", "x", "coordx", "coordenadax"])

    missing = []
    if estado_col is None:
        missing.append("Estado/UF")
    if cidade_col is None:
        missing.append("Cidade/Municipio")
    if lat_col is None:
        missing.append("Latitude (lat)")
    if lon_col is None:
        missing.append("Longitude (lon)")
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
        df_geo["Estado"].astype(str).str.strip().str.upper() + "|" + df_geo["Cidade"].astype(str).str.strip().str.upper()
    )
    return df_geo[["key", "Estado", "Cidade", "lat", "lon"]]


def section_prev_toggle(section_key: str) -> bool:
    """
    Right-aligned button 'PERÍODO ANTERIOR' per section (toggle visibility).
    """
    if section_key not in st.session_state:
        st.session_state[section_key] = False
    _l, _r = st.columns([0.82, 0.18], vertical_alignment="center")
    with _r:
        if st.button("PERÍODO ANTERIOR", key=f"{section_key}__btn", use_container_width=True):
            st.session_state[section_key] = not st.session_state[section_key]
    return bool(st.session_state[section_key])


# --------------------------
# Chart helpers for PDF
# --------------------------
def altair_to_png_bytes(chart: alt.Chart, scale: float = 2.0) -> bytes | None:
    if chart is None or vlc is None:
        return None
    try:
        spec = chart.to_dict()
        return vlc.vegalite_to_png(spec, scale=scale)
    except Exception:
        return None


def plotly_to_png_bytes(fig, scale: float = 2.0) -> bytes | None:
    # This may fail if Kaleido/Chrome is missing
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception:
        return None


def matplotlib_pie_png_bytes(labels, values, title: str = "", scale: float = 2.0) -> bytes:
    # Chrome-free fallback
    buf = io.BytesIO()
    try:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=int(120 * scale))
        ax.pie(values, labels=None, autopct=lambda p: f"{p:.0f}%" if p >= 6 else "")
        ax.axis("equal")
        if title:
            ax.set_title(title)
        # Add a legend with labels (cleaner than labeling wedges)
        ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plt.close("all")
    return buf.getvalue()


def build_pdf_report(
    *,
    titulo: str,
    subtitulo: str,
    periodo_atual_label: str,
    periodo_anterior_label: str,
    kpis_atual: dict,
    kpis_anterior: dict | None,
    charts_png: dict,  # {name: bytes}
    tables: dict,  # {name: pd.DataFrame}
) -> bytes:
    """
    Landscape A4 PDF (ReportLab). Includes charts as images.
    """
    buff = io.BytesIO()
    doc = SimpleDocTemplate(
        buff,
        pagesize=landscape(A4),
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
        title=titulo,
        author="Insights de Vendas",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, leading=18, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, leading=14, spaceAfter=6)
    p = ParagraphStyle("p", parent=styles["BodyText"], fontSize=9, leading=11)

    elems = []
    elems.append(Paragraph(titulo, h1))
    elems.append(Paragraph(subtitulo, p))
    elems.append(Spacer(1, 6))

    # KPIs table (Atual)
    elems.append(Paragraph(f"Período atual: <b>{periodo_atual_label}</b>", h2))
    kpi_rows = [[Paragraph("<b>Métrica</b>", p), Paragraph("<b>Valor</b>", p)]]
    for k, v in kpis_atual.items():
        kpi_rows.append([Paragraph(html.escape(str(k)), p), Paragraph(html.escape(str(v)), p)])
    tbl = Table(kpi_rows, colWidths=[7.0 * cm, 6.5 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#374151")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
            ]
        )
    )
    elems.append(tbl)
    elems.append(Spacer(1, 10))

    # KPIs table (Anterior)
    if kpis_anterior:
        elems.append(Paragraph(f"Período anterior: <b>{periodo_anterior_label}</b>", h2))
        kpi_rows2 = [[Paragraph("<b>Métrica</b>", p), Paragraph("<b>Valor</b>", p)]]
        for k, v in kpis_anterior.items():
            kpi_rows2.append([Paragraph(html.escape(str(k)), p), Paragraph(html.escape(str(v)), p)])
        tbl2 = Table(kpi_rows2, colWidths=[7.0 * cm, 6.5 * cm])
        tbl2.setStyle(tbl._cellstyles and tbl._argW and tbl._argH and tbl._cellvalues and tbl._bkgrndcmds and tbl._linecmds and tbl._spanCmds and tbl._nosplitCmds and tbl._rowHeights and tbl._colWidths and tbl._repeatRows and tbl._repeatCols and tbl._splitByRow and tbl._splitInRow and tbl._splitInCol and tbl._nrows and tbl._ncols and tbl._longTableOptimize and tbl._rowSplitRange and tbl._minRowHeights and tbl._maxRowHeights and tbl._rowSplitRange and tbl._minRowHeights and tbl._maxRowHeights)  # safe no-op
        # Apply a fresh style (simpler)
        tbl2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#374151")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
                ]
            )
        )
        elems.append(tbl2)
        elems.append(Spacer(1, 10))

    # Charts block
    if charts_png:
        elems.append(Paragraph("Gráficos", h2))
        # Put charts in a 2-column grid
        img_cells = []
        row = []
        max_w = 12.5 * cm
        max_h = 8.0 * cm

        for name, png_bytes in charts_png.items():
            if not png_bytes:
                continue
            img_buf = io.BytesIO(png_bytes)
            img = RLImage(img_buf)
            img._restrictSize(max_w, max_h)

            cap = Paragraph(f"<b>{html.escape(str(name))}</b>", p)
            cell = [cap, Spacer(1, 2), img]
            row.append(cell)
            if len(row) == 2:
                img_cells.append(row)
                row = []
        if row:
            # fill remaining
            row.append([""])
            img_cells.append(row)

        # Flatten each cell to a mini-table (caption + img)
        grid_data = []
        for r in img_cells:
            grid_row = []
            for cell in r:
                if cell == "":
                    grid_row.append("")
                else:
                    mini = Table([[cell[0]], [cell[2]]], colWidths=[max_w])
                    mini.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
                    grid_row.append(mini)
            grid_data.append(grid_row)

        grid = Table(grid_data, colWidths=[13.3 * cm, 13.3 * cm])
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        elems.append(grid)
        elems.append(Spacer(1, 8))

    # Tables block
    if tables:
        elems.append(PageBreak())
        elems.append(Paragraph("Tabelas", h2))
        for tname, df_tbl in tables.items():
            if df_tbl is None or df_tbl.empty:
                continue
            elems.append(Paragraph(f"<b>{html.escape(str(tname))}</b>", p))
            # Convert to reportlab table
            cols = list(df_tbl.columns)
            data = [cols] + df_tbl.astype(str).values.tolist()
            t = Table(data, repeatRows=1)
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#374151")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            elems.append(t)
            elems.append(Spacer(1, 10))

    doc.build(elems)
    return buff.getvalue()


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

    base = alt.Chart(ts).encode(x=alt.X("MesLabelBr:N", sort=x_order, axis=alt.Axis(title=None)))

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
st.sidebar.markdown("### Período atual")

anos_disponiveis = sorted(df["Ano"].dropna().unique())
if not anos_disponiveis:
    st.error("Não foi possível identificar anos na base de dados.")
    st.stop()

last_year = int(anos_disponiveis[-1])

meses_ano_default = df.loc[df["Ano"] == last_year, "MesNum"].dropna().unique()
default_start_month_num = int(meses_ano_default.min()) if len(meses_ano_default) else 1
default_end_month_num = int(meses_ano_default.max()) if len(meses_ano_default) else 12

month_names = [MONTH_MAP_NUM_TO_NAME[m] for m in range(1, 13)]

st.sidebar.caption("Período atual (início)")
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

st.sidebar.caption("Período atual (fim)")
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

# ==========================
# PERÍODO ANTERIOR (SEMPRE MANUAL)
# ==========================
st.sidebar.markdown("### Período anterior (manual)")

st.sidebar.caption("Período anterior (início)")
col_pmi, col_pai = st.sidebar.columns(2)
with col_pmi:
    prev_start_month_name = st.selectbox(
        "Mês (ant. início)",
        options=month_names,
        index=month_names.index(start_month_name),
        key="prev_start_month",
    )
with col_pai:
    prev_start_year = st.selectbox(
        "Ano (ant. início)",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(int(start_year)),
        key="prev_start_year",
    )

st.sidebar.caption("Período anterior (fim)")
col_pmf, col_paf = st.sidebar.columns(2)
with col_pmf:
    prev_end_month_name = st.selectbox(
        "Mês (ant. fim)",
        options=month_names,
        index=month_names.index(end_month_name),
        key="prev_end_month",
    )
with col_paf:
    prev_end_year = st.selectbox(
        "Ano (ant. fim)",
        options=[int(a) for a in anos_disponiveis],
        index=list(anos_disponiveis).index(int(end_year)),
        key="prev_end_year",
    )

prev_start_month = MONTH_MAP_NAME_TO_NUM[prev_start_month_name]
prev_end_month = MONTH_MAP_NAME_TO_NUM[prev_end_month_name]

prev_start = pd.Timestamp(year=int(prev_start_year), month=int(prev_start_month), day=1)
prev_end = pd.Timestamp(year=int(prev_end_year), month=int(prev_end_month), day=1)

if prev_start > prev_end:
    st.sidebar.error("Período anterior (início) não pode ser maior que o fim.")
    st.stop()

current_period_label = format_period_label(start_comp, end_comp)
previous_period_label = format_period_label(prev_start, prev_end)

# ==========================
# FILTER DATA
# ==========================
mask_period = (df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)
df_period = df.loc[mask_period].copy()
if df_period.empty:
    st.warning("Nenhuma venda no período atual selecionado.")
    st.stop()

reps_period = sorted(df_period["Representante"].dropna().unique())
if not reps_period:
    st.error("Não há representantes com vendas no período atual selecionado.")
    st.stop()

rep_options = ["Todos"] + reps_period
rep_selected = st.sidebar.selectbox("Representante", rep_options)

df_rep = df_period.copy() if rep_selected == "Todos" else df_period[df_period["Representante"] == rep_selected].copy()

mask_prev_period = (df["Competencia"] >= prev_start) & (df["Competencia"] <= prev_end)
df_prev_period = df.loc[mask_prev_period].copy()
df_rep_prev = df_prev_period.copy() if rep_selected == "Todos" else df_prev_period[df_prev_period["Representante"] == rep_selected].copy()

clientes_carteira = build_carteira_status(df, rep_selected, start_comp, end_comp)
clientes_carteira_prev = build_carteira_status(df, rep_selected, prev_start, prev_end)

if not clientes_carteira.empty:
    carteira_score, carteira_label = compute_carteira_score(clientes_carteira)
else:
    carteira_score, carteira_label = 50.0, "Neutra"

if not clientes_carteira_prev.empty:
    carteira_score_prev, carteira_label_prev = compute_carteira_score(clientes_carteira_prev)
else:
    carteira_score_prev, carteira_label_prev = 50.0, "Neutra"

# ==========================
# HEADER + GERAR PDF (top right)
# ==========================
st.title("Insights de Vendas")

_rep_left, _rep_right = st.columns([0.78, 0.22], vertical_alignment="center")
with _rep_left:
    titulo_rep = "Todos" if rep_selected == "Todos" else rep_selected
    st.subheader(f"Representante: **{titulo_rep}**")
    st.caption(f"Período atual: {current_period_label}  •  Período anterior: {previous_period_label}")

with _rep_right:
    # single button in the whole app, unique key
    gen_pdf_now = st.button("📄 Gerar PDF", key="pdf_gen_btn__top_right", use_container_width=True)

st.markdown("---")


# ==========================
# TOP KPIs (5 columns) - CURRENT
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

    hhi_value = float((shares**2).sum())
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
cidades_atendidas = int(df_rep[["Estado", "Cidade"]].dropna().drop_duplicates().shape[0])
estados_atendidos = int(df_rep["Estado"].dropna().nunique())

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
show_prev_destaques = section_prev_toggle("sec_prev_destaques")

if df_rep.empty:
    st.info("Não há vendas no período atual selecionado.")
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

if show_prev_destaques:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")
    if df_rep_prev.empty:
        st.info("Não há vendas no período anterior selecionado.")
    else:
        mensal_prev = df_rep_prev.groupby(["Ano", "MesNum"], as_index=False)[["Valor", "Quantidade"]].sum()
        mensal_prev["Competencia"] = pd.to_datetime(dict(year=mensal_prev["Ano"], month=mensal_prev["MesNum"], day=1))
        mensal_prev["MesLabel"] = mensal_prev["Competencia"].apply(
            lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}"
        )

        best_fat_p = mensal_prev.loc[mensal_prev["Valor"].idxmax()]
        worst_fat_p = mensal_prev.loc[mensal_prev["Valor"].idxmin()]
        best_vol_p = mensal_prev.loc[mensal_prev["Quantidade"].idxmax()]
        worst_vol_p = mensal_prev.loc[mensal_prev["Quantidade"].idxmin()]

        c1p, c2p = st.columns(2)
        with c1p:
            st.markdown("**Faturamento**")
            st.write(f"• Melhor mês: **{best_fat_p['MesLabel']}** — {format_brl(best_fat_p['Valor'])}")
            st.write(f"• Pior mês: **{worst_fat_p['MesLabel']}** — {format_brl(worst_fat_p['Valor'])}")
        with c2p:
            st.markdown("**Volume**")
            st.write(f"• Melhor mês: **{best_vol_p['MesLabel']}** — {format_un(best_vol_p['Quantidade'])}")
            st.write(f"• Pior mês: **{worst_vol_p['MesLabel']}** — {format_un(worst_vol_p['Quantidade'])}")

st.markdown("---")


# ==========================
# PERFORMANCE DE REPRESENTANTES (after Destaques)
# ==========================
st.subheader("Performance de Representantes")
show_prev_rep_perf = section_prev_toggle("sec_prev_rep_perf")

df_team_curr = df_period.copy()
rep_perf = (
    df_team_curr.groupby("Representante", as_index=False)
    .agg(Faturamento=("Valor", "sum"), Volume=("Quantidade", "sum"), Clientes=("Cliente", "nunique"))
)
rep_perf["Faturamento"] = pd.to_numeric(rep_perf["Faturamento"], errors="coerce").fillna(0.0)
rep_perf["Volume"] = pd.to_numeric(rep_perf["Volume"], errors="coerce").fillna(0.0)
rep_perf["Clientes"] = pd.to_numeric(rep_perf["Clientes"], errors="coerce").fillna(0).astype(int)

if rep_perf.empty:
    st.info("Sem dados de representantes no período atual.")
else:
    if rep_selected == "Todos":
        top_n = 15
        total_team_fat = float(rep_perf["Faturamento"].sum()) or 1.0
        total_team_vol = float(rep_perf["Volume"].sum()) or 1.0

        rep_fat = rep_perf.sort_values("Faturamento", ascending=False).head(top_n).copy()
        rep_fat["Ranking"] = range(1, len(rep_fat) + 1)
        rep_fat["%"] = rep_fat["Faturamento"] / total_team_fat
        rep_fat["Faturamento"] = rep_fat["Faturamento"].map(format_brl_compact)
        rep_fat["%"] = rep_fat["%"].map(lambda x: f"{x:.1%}")
        rep_fat = rep_fat[["Ranking", "Representante", "Faturamento", "%", "Clientes"]]

        rep_vol = rep_perf.sort_values("Volume", ascending=False).head(top_n).copy()
        rep_vol["Ranking"] = range(1, len(rep_vol) + 1)
        rep_vol["%"] = rep_vol["Volume"] / total_team_vol
        rep_vol["Volume"] = rep_vol["Volume"].map(format_un)
        rep_vol["%"] = rep_vol["%"].map(lambda x: f"{x:.1%}")
        rep_vol = rep_vol[["Ranking", "Representante", "Volume", "%", "Clientes"]]

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Ranking por Faturamento**")
            st.dataframe(rep_fat, use_container_width=True, hide_index=True)
        with cB:
            st.markdown("**Ranking por Volume**")
            st.dataframe(rep_vol, use_container_width=True, hide_index=True)
    else:
        row_sel = rep_perf[rep_perf["Representante"] == rep_selected].copy()
        if row_sel.empty:
            st.info("Representante selecionado não tem dados no período atual.")
        else:
            sel_fat = float(row_sel["Faturamento"].iloc[0])
            sel_vol = float(row_sel["Volume"].iloc[0])
            sel_cli = int(row_sel["Clientes"].iloc[0])

            leader_row = rep_perf.sort_values("Faturamento", ascending=False).head(1)
            leader_name = str(leader_row["Representante"].iloc[0])
            leader_fat = float(leader_row["Faturamento"].iloc[0])
            leader_vol = float(leader_row["Volume"].iloc[0])

            avg_fat = float(rep_perf["Faturamento"].mean())
            avg_vol = float(rep_perf["Volume"].mean())

            rep_rank = rep_perf.copy()
            rep_rank["RankFat"] = rep_rank["Faturamento"].rank(method="min", ascending=False).astype(int)
            rep_rank["RankVol"] = rep_rank["Volume"].rank(method="min", ascending=False).astype(int)
            rank_fat = int(rep_rank.loc[rep_rank["Representante"] == rep_selected, "RankFat"].iloc[0])
            rank_vol = int(rep_rank.loc[rep_rank["Representante"] == rep_selected, "RankVol"].iloc[0])
            team_size = int(rep_rank["Representante"].nunique())

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Faturamento (rep)", format_brl_compact(sel_fat))
            k2.metric("Volume (rep)", format_un(sel_vol))
            k3.metric("Clientes (rep)", f"{sel_cli}")
            k4.metric("Ranking Faturamento", f"{rank_fat} / {team_size}")
            k5.metric("Ranking Volume", f"{rank_vol} / {team_size}")

            comp = pd.DataFrame(
                [
                    {"Comparação": "Líder (Faturamento)", "Valor": format_brl_compact(leader_fat), "Referência": leader_name},
                    {"Comparação": "Média do time (Faturamento)", "Valor": format_brl_compact(avg_fat), "Referência": f"Média de {team_size} reps"},
                    {"Comparação": "Líder (Volume)", "Valor": format_un(leader_vol), "Referência": leader_name},
                    {"Comparação": "Média do time (Volume)", "Valor": format_un(avg_vol), "Referência": f"Média de {team_size} reps"},
                ]
            )
            st.dataframe(comp, use_container_width=True, hide_index=True)

if show_prev_rep_perf:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")

    df_team_prev = df_prev_period.copy()
    rep_perf_prev = (
        df_team_prev.groupby("Representante", as_index=False)
        .agg(Faturamento=("Valor", "sum"), Volume=("Quantidade", "sum"), Clientes=("Cliente", "nunique"))
    )
    rep_perf_prev["Faturamento"] = pd.to_numeric(rep_perf_prev["Faturamento"], errors="coerce").fillna(0.0)
    rep_perf_prev["Volume"] = pd.to_numeric(rep_perf_prev["Volume"], errors="coerce").fillna(0.0)
    rep_perf_prev["Clientes"] = pd.to_numeric(rep_perf_prev["Clientes"], errors="coerce").fillna(0).astype(int)

    if rep_perf_prev.empty:
        st.info("Sem dados no período anterior.")
    else:
        if rep_selected == "Todos":
            top_n = 15
            total_team_fat_p = float(rep_perf_prev["Faturamento"].sum()) or 1.0
            total_team_vol_p = float(rep_perf_prev["Volume"].sum()) or 1.0

            rep_fat_p = rep_perf_prev.sort_values("Faturamento", ascending=False).head(top_n).copy()
            rep_fat_p["Ranking"] = range(1, len(rep_fat_p) + 1)
            rep_fat_p["%"] = rep_fat_p["Faturamento"] / total_team_fat_p
            rep_fat_p["Faturamento"] = rep_fat_p["Faturamento"].map(format_brl_compact)
            rep_fat_p["%"] = rep_fat_p["%"].map(lambda x: f"{x:.1%}")
            rep_fat_p = rep_fat_p[["Ranking", "Representante", "Faturamento", "%", "Clientes"]]

            rep_vol_p = rep_perf_prev.sort_values("Volume", ascending=False).head(top_n).copy()
            rep_vol_p["Ranking"] = range(1, len(rep_vol_p) + 1)
            rep_vol_p["%"] = rep_vol_p["Volume"] / total_team_vol_p
            rep_vol_p["Volume"] = rep_vol_p["Volume"].map(format_un)
            rep_vol_p["%"] = rep_vol_p["%"].map(lambda x: f"{x:.1%}")
            rep_vol_p = rep_vol_p[["Ranking", "Representante", "Volume", "%", "Clientes"]]

            cA_p, cB_p = st.columns(2)
            with cA_p:
                st.markdown("**Ranking por Faturamento**")
                st.dataframe(rep_fat_p, use_container_width=True, hide_index=True)
            with cB_p:
                st.markdown("**Ranking por Volume**")
                st.dataframe(rep_vol_p, use_container_width=True, hide_index=True)
        else:
            row_sel_p = rep_perf_prev[rep_perf_prev["Representante"] == rep_selected].copy()
            if row_sel_p.empty:
                st.info("Representante selecionado não tem dados no período anterior.")
            else:
                sel_fat_p = float(row_sel_p["Faturamento"].iloc[0])
                sel_vol_p = float(row_sel_p["Volume"].iloc[0])
                sel_cli_p = int(row_sel_p["Clientes"].iloc[0])

                leader_row_p = rep_perf_prev.sort_values("Faturamento", ascending=False).head(1)
                leader_name_p = str(leader_row_p["Representante"].iloc[0])
                leader_fat_p = float(leader_row_p["Faturamento"].iloc[0])
                leader_vol_p = float(leader_row_p["Volume"].iloc[0])

                avg_fat_p = float(rep_perf_prev["Faturamento"].mean())
                avg_vol_p = float(rep_perf_prev["Volume"].mean())

                rep_rank_p = rep_perf_prev.copy()
                rep_rank_p["RankFat"] = rep_rank_p["Faturamento"].rank(method="min", ascending=False).astype(int)
                rep_rank_p["RankVol"] = rep_rank_p["Volume"].rank(method="min", ascending=False).astype(int)
                rank_fat_p = int(rep_rank_p.loc[rep_rank_p["Representante"] == rep_selected, "RankFat"].iloc[0])
                rank_vol_p = int(rep_rank_p.loc[rep_rank_p["Representante"] == rep_selected, "RankVol"].iloc[0])
                team_size_p = int(rep_rank_p["Representante"].nunique())

                k1p, k2p, k3p, k4p, k5p = st.columns(5)
                k1p.metric("Faturamento (rep)", format_brl_compact(sel_fat_p))
                k2p.metric("Volume (rep)", format_un(sel_vol_p))
                k3p.metric("Clientes (rep)", f"{sel_cli_p}")
                k4p.metric("Ranking Faturamento", f"{rank_fat_p} / {team_size_p}")
                k5p.metric("Ranking Volume", f"{rank_vol_p} / {team_size_p}")

                comp_p = pd.DataFrame(
                    [
                        {"Comparação": "Líder (Faturamento)", "Valor": format_brl_compact(leader_fat_p), "Referência": leader_name_p},
                        {"Comparação": "Média do time (Faturamento)", "Valor": format_brl_compact(avg_fat_p), "Referência": f"Média de {team_size_p} reps"},
                        {"Comparação": "Líder (Volume)", "Valor": format_un(leader_vol_p), "Referência": leader_name_p},
                        {"Comparação": "Média do time (Volume)", "Valor": format_un(avg_vol_p), "Referência": f"Média de {team_size_p} reps"},
                    ]
                )
                st.dataframe(comp_p, use_container_width=True, hide_index=True)

st.markdown("---")


# ==========================
# EVOLUÇÃO – FATURAMENTO x VOLUME (after Destaques)
# ==========================
st.subheader("Evolução – Faturamento x Volume")
show_prev_evolucao = section_prev_toggle("sec_prev_evolucao")

fat_curr = float(df_rep["Valor"].sum())
vol_curr = float(df_rep["Quantidade"].sum())
fat_prev = float(df_rep_prev["Valor"].sum())
vol_prev = float(df_rep_prev["Quantidade"].sum())

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

if show_prev_evolucao:
    st.markdown("---")
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

st.markdown("---")


# ==========================
# CATEGORIAS VENDIDAS (after Destaques)
# ==========================
st.subheader("Categorias vendidas")
show_prev_categorias = section_prev_toggle("sec_prev_categorias")

# CURRENT
if df_rep.empty:
    st.info("Não há vendas no período atual selecionado.")
    cat = pd.DataFrame()
    fig_cat = None
else:
    curr_cat = df_rep.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "ValorAtual"})
    prev_cat = df_rep_prev.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "ValorAnterior"})

    cat = pd.merge(curr_cat, prev_cat, on="Categoria", how="outer")
    cat["ValorAtual"] = pd.to_numeric(cat["ValorAtual"], errors="coerce").fillna(0.0)
    cat["ValorAnterior"] = pd.to_numeric(cat["ValorAnterior"], errors="coerce").fillna(0.0)

    total_cat = float(cat["ValorAtual"].sum())
    fig_cat = None

    if total_cat <= 0:
        st.info("Sem faturamento para exibir categorias no período atual.")
    else:
        cat["%"] = cat["ValorAtual"] / total_cat
        cat["Variação (R$)"] = cat["ValorAtual"] - cat["ValorAnterior"]

        def pct_growth(row):
            prevv = float(row["ValorAnterior"])
            currv = float(row["ValorAtual"])
            if prevv > 0:
                return (currv - prevv) / prevv
            if currv > 0 and prevv == 0:
                return None  # "Novo"
            return 0.0

        cat["% Crescimento"] = cat.apply(pct_growth, axis=1)
        cat = cat.sort_values("ValorAtual", ascending=False)

        col_pie_cat, col_tbl_cat = st.columns([1.0, 1.25])

        with col_pie_cat:
            st.caption("Participação por categoria")

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
            fig_cat.update_traces(text=df_pie["Text"], textposition="inside", textinfo="text", insidetextorientation="radial")
            fig_cat.update_layout(showlegend=False, height=560, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cat, width="stretch")

        with col_tbl_cat:
            st.caption("Resumo – Categorias (com crescimento vs. período anterior)")

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
            st.dataframe(cat_disp, use_container_width=True, hide_index=True)

# PREVIOUS (optional)
if show_prev_categorias:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")

    if df_rep_prev.empty:
        st.info("Não há vendas no período anterior selecionado.")
    else:
        prev_only = df_rep_prev.groupby("Categoria", as_index=False)["Valor"].sum()
        prev_only["Valor"] = pd.to_numeric(prev_only["Valor"], errors="coerce").fillna(0.0)
        total_prev = float(prev_only["Valor"].sum())

        if total_prev <= 0:
            st.info("Sem faturamento para exibir categorias no período anterior.")
        else:
            prev_only = prev_only.sort_values("Valor", ascending=False)
            prev_only["%"] = prev_only["Valor"] / total_prev

            col_pie_p, col_tbl_p = st.columns([1.0, 1.25])
            with col_pie_p:
                st.caption("Participação por categoria")

                df_pie_p = prev_only.copy()
                if len(df_pie_p) > 10:
                    top10p = df_pie_p.head(10).copy()
                    others_val_p = float(df_pie_p.iloc[10:]["Valor"].sum())
                    top10p = pd.concat([top10p, pd.DataFrame([{"Categoria": "Outras", "Valor": others_val_p}])], ignore_index=True)
                    df_pie_p = top10p

                df_pie_p["Share"] = df_pie_p["Valor"] / float(df_pie_p["Valor"].sum()) if float(df_pie_p["Valor"].sum()) > 0 else 0.0
                df_pie_p["Legenda"] = df_pie_p.apply(lambda r: f"{r['Categoria']} {r['Share']*100:.1f}%", axis=1)

                def make_text_cat_p(row):
                    return f"{row['Categoria']}<br>{row['Share']*100:.1f}%" if row["Share"] >= 0.07 else ""

                df_pie_p["Text"] = df_pie_p.apply(make_text_cat_p, axis=1)
                order_leg_p = df_pie_p.sort_values("Share", ascending=False)["Legenda"].tolist()

                fig_cat_p = px.pie(df_pie_p, values="Valor", names="Legenda", hole=0.35, category_orders={"Legenda": order_leg_p})
                fig_cat_p.update_traces(text=df_pie_p["Text"], textposition="inside", textinfo="text", insidetextorientation="radial")
                fig_cat_p.update_layout(showlegend=False, height=560, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cat_p, width="stretch")

            with col_tbl_p:
                st.caption("Resumo – Categorias")
                tbl_p = prev_only.copy()
                tbl_p["Valor"] = tbl_p["Valor"].map(format_brl)
                tbl_p["%"] = tbl_p["%"].map(lambda x: f"{x:.1%}")
                tbl_p = tbl_p[["Categoria", "Valor", "%"]]
                st.dataframe(tbl_p, use_container_width=True, hide_index=True)

st.markdown("---")


# ==========================
# MAPA DE CLIENTES
# ==========================
st.subheader("Mapa de Clientes")
show_prev_mapa = section_prev_toggle("sec_prev_mapa")

if "selected_city_tooltip" not in st.session_state:
    st.session_state["selected_city_tooltip"] = None

def render_map_block(df_base: pd.DataFrame, label_periodo: str):
    if df_base.empty:
        st.info(f"Não há vendas no período selecionado ({label_periodo}).")
        return

    try:
        force_leaflet_1_9_4()
        df_geo = load_geo()

        df_cities = df_base.groupby(["Estado", "Cidade"], as_index=False).agg(
            Valor=("Valor", "sum"),
            Quantidade=("Quantidade", "sum"),
            Clientes=("Cliente", "nunique"),
        )
        df_cities["key"] = (
            df_cities["Estado"].astype(str).str.strip().str.upper() + "|" + df_cities["Cidade"].astype(str).str.strip().str.upper()
        )

        df_map = df_cities.merge(df_geo, on="key", how="inner", suffixes=("_fat", "_geo"))

        if df_map.empty:
            st.info("Não há coordenadas de cidades para exibir no mapa.")
            return

        df_map["Tooltip"] = df_map["Cidade_fat"].astype(str) + " - " + df_map["Estado_fat"].astype(str)

        metric_choice = st.radio(f"Métrica do mapa ({label_periodo})", ["Faturamento", "Volume"], horizontal=True, key=f"map_metric_{label_periodo}")
        metric_col = "Valor" if metric_choice == "Faturamento" else "Quantidade"
        metric_label = "Faturamento (R$)" if metric_col == "Valor" else "Volume (un)"

        if df_map[metric_col].max() <= 0:
            st.info("Sem dados para exibir no mapa nesse período.")
            return

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

            map_data = st_folium(m, width=None, height=800)

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
            cov1.metric("Cidades atendidas", f"{int(df_base[['Estado','Cidade']].dropna().drop_duplicates().shape[0])}")
            cov2.metric("Estados atendidos", f"{int(df_base['Estado'].dropna().nunique())}")
            cov3.metric("Clientes atendidos", f"{int(df_base['Cliente'].dropna().nunique())}")

            st.markdown("**Principais clientes**")
            df_top_clients = (
                df_base.groupby(["Cliente", "Estado", "Cidade"], as_index=False)["Valor"]
                .sum()
                .sort_values("Valor", ascending=False)
                .head(15)
            )
            df_top_clients["Faturamento"] = df_top_clients["Valor"].map(format_brl)
            st.dataframe(df_top_clients[["Cliente", "Cidade", "Estado", "Faturamento"]], use_container_width=True, hide_index=True)

            if selected_label:
                row_city = df_map[df_map["Tooltip"] == selected_label].head(1)
                if not row_city.empty:
                    cidade_sel = row_city["Cidade_fat"].iloc[0]
                    estado_sel = row_city["Estado_fat"].iloc[0]
                    df_city_clients = df_base[(df_base["Cidade"] == cidade_sel) & (df_base["Estado"] == estado_sel)].copy()

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
                        st.dataframe(display_city, use_container_width=True, hide_index=True)

    except Exception as e:
        st.info(f"Mapa de clientes ainda não disponível: {e}")

# Current map
render_map_block(df_rep, current_period_label)

# Prev map (optional)
if show_prev_mapa:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")
    render_map_block(df_rep_prev, previous_period_label)

st.markdown("---")


# ==========================
# DISTRIBUIÇÃO POR ESTADOS
# ==========================
st.subheader("Distribuição por estados")
show_prev_estados = section_prev_toggle("sec_prev_estados")

def render_estados_block(df_base: pd.DataFrame, label_periodo: str):
    if df_base.empty:
        st.info(f"Não há vendas no período selecionado ({label_periodo}).")
        return None, None, None

    estados_df = df_base.groupby("Estado", as_index=False)[["Valor", "Quantidade"]].sum().sort_values("Valor", ascending=False)
    total_valor_all = float(estados_df["Valor"].sum())
    total_qtd_all = float(estados_df["Quantidade"].sum())

    if total_valor_all <= 0:
        st.info("Não há faturamento para distribuir por estados nesse período.")
        return estados_df, None, None

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

    fig_states = None
    with c_left:
        st.caption(f"Top 10 estados por faturamento – {label_periodo}")
        fig_states = px.pie(
            estados_top.sort_values("Valor", ascending=False),
            values="Valor",
            names="Estado",
            hole=0.35,
        )
        fig_states.update_traces(textposition="inside", textinfo="percent+label")
        fig_states.update_layout(showlegend=False, height=560, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_states, width="stretch")

    with c_right:
        st.markdown("**Resumo – Top 10 estados**")
        st.dataframe(estados_display, use_container_width=True, hide_index=True)

    return estados_display, estados_top, fig_states

estados_display_curr, estados_top_curr, fig_states_curr = render_estados_block(df_rep, current_period_label)

if show_prev_estados:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")
    estados_display_prev, estados_top_prev, fig_states_prev = render_estados_block(df_rep_prev, previous_period_label)
else:
    estados_display_prev, estados_top_prev, fig_states_prev = None, None, None

st.markdown("---")


# ==========================
# DISTRIBUIÇÃO POR CLIENTES
# ==========================
st.subheader("Distribuição por clientes")
show_prev_clientes = section_prev_toggle("sec_prev_clientes")

def render_clientes_block(df_base: pd.DataFrame, label_periodo: str):
    if df_base.empty:
        st.info(f"Nenhum cliente com vendas no período ({label_periodo}).")
        return None, None, None, None

    total_base = float(df_base["Valor"].sum())
    if total_base <= 0:
        st.info(f"Nenhum faturamento no período ({label_periodo}).")
        return None, None, None, None

    df_clientes_full = (
        df_base.groupby(["Cliente", "Estado", "Cidade"], as_index=False)
        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
        .sort_values("Valor", ascending=False)
    )
    df_clientes_full["Share"] = df_clientes_full["Valor"] / (total_base if total_base > 0 else 1.0)

    # N80 + HHI
    shares = df_clientes_full["Share"]
    cum_share = shares.cumsum()
    n80_count_local = 0
    for i, v in enumerate(cum_share, start=1):
        n80_count_local = i
        if v >= 0.8:
            break
    hhi_value_local = float((shares**2).sum())
    if hhi_value_local < 0.10:
        hhi_label_local = "Baixa"
    elif hhi_value_local < 0.20:
        hhi_label_local = "Moderada"
    else:
        hhi_label_local = "Alta"

    top1_share_local = float(shares.iloc[:1].sum()) if len(shares) else 0.0
    top3_share_local = float(shares.iloc[:3].sum()) if len(shares) else 0.0
    top10_share_local = float(shares.iloc[:10].sum()) if len(shares) else 0.0
    clientes_atendidos_local = int(df_clientes_full["Cliente"].nunique())

    k1, k2, k3, k4, k5 = st.columns(5)
    n80_ratio = (n80_count_local / clientes_atendidos_local) if clientes_atendidos_local > 0 else 0.0
    k1.metric("N80", f"{n80_count_local}", f"{n80_ratio:.0%} da carteira")
    k2.metric("Índice de concentração", hhi_label_local, f"HHI {hhi_value_local:.3f}")
    k3.metric("Top 1 cliente", f"{top1_share_local:.1%}")
    k4.metric("Top 3 clientes", f"{top3_share_local:.1%}")
    k5.metric("Top 10 clientes", f"{top10_share_local:.1%}")

    col_pie, col_tbl = st.columns([1.10, 1.50])

    fig_clients = None
    with col_pie:
        st.caption(f"Participação dos clientes (Top 10 destacados) – {label_periodo}")

        df_pie = df_clientes_full[["Cliente", "Valor"]].copy()
        df_pie = df_pie.groupby("Cliente", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False)
        df_pie["Rank"] = range(1, len(df_pie) + 1)
        df_pie["Grupo"] = df_pie.apply(lambda r: r["Cliente"] if r["Rank"] <= 10 else "Outros", axis=1)

        dist_df = df_pie.groupby("Grupo", as_index=False)["Valor"].sum()
        dist_df["Share"] = dist_df["Valor"] / total_base
        dist_df = dist_df.sort_values("Share", ascending=False)
        dist_df["Legenda"] = dist_df.apply(lambda r: f"{r['Grupo']} {r['Share']*100:.1f}%", axis=1)

        def make_text(row):
            if row["Share"] >= 0.07:
                return f"{row['Grupo']}<br>{row['Share']*100:.1f}%"
            return ""

        dist_df["Text"] = dist_df.apply(make_text, axis=1)
        order_legenda = dist_df["Legenda"].tolist()

        fig_clients = px.pie(dist_df, values="Valor", names="Legenda", category_orders={"Legenda": order_legenda})
        fig_clients.update_traces(text=dist_df["Text"], textposition="inside", textinfo="text", insidetextorientation="radial")
        fig_clients.update_layout(showlegend=False, height=560, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_clients, width="stretch")

    with col_tbl:
        st.caption(f"Resumo – clientes (mais detalhado) – {label_periodo}")

        df_tbl = df_clientes_full.copy()
        df_tbl["Faturamento"] = df_tbl["Valor"].map(format_brl)
        df_tbl["% Faturamento"] = df_tbl["Share"].map(lambda x: f"{x:.1%}")
        df_tbl["Volume"] = df_tbl["Quantidade"].map(format_un)
        df_tbl = df_tbl.head(15)[["Cliente", "Cidade", "Estado", "Faturamento", "% Faturamento", "Volume"]]
        st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    # Return small items useful for PDF
    kpis_local = {
        "Total período": format_brl_compact(total_base),
        "Clientes atendidos": str(clientes_atendidos_local),
        "N80": f"{n80_count_local} ({n80_ratio:.0%} da carteira)",
        "HHI": f"{hhi_value_local:.3f} ({hhi_label_local})",
        "Top 1": f"{top1_share_local:.1%}",
        "Top 3": f"{top3_share_local:.1%}",
        "Top 10": f"{top10_share_local:.1%}",
    }
    return df_tbl, fig_clients, dist_df, kpis_local

clientes_tbl_curr, fig_clients_curr, dist_df_curr, clientes_kpis_curr = render_clientes_block(df_rep, current_period_label)

if show_prev_clientes:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")
    clientes_tbl_prev, fig_clients_prev, dist_df_prev, clientes_kpis_prev = render_clientes_block(df_rep_prev, previous_period_label)
else:
    clientes_tbl_prev, fig_clients_prev, dist_df_prev, clientes_kpis_prev = None, None, None, None

st.markdown("---")


# ==========================
# SAÚDE DA CARTEIRA – DETALHES
# ==========================
st.subheader("Saúde da carteira – Detalhes")
show_prev_carteira = section_prev_toggle("sec_prev_carteira")

def render_carteira_block(clientes_carteira_df: pd.DataFrame, score: float, label: str, label_periodo: str):
    c_score1, c_score2 = st.columns([0.35, 0.65])
    with c_score1:
        st.metric("Pontuação – Saúde da carteira", f"{score:.0f} / 100", label)
    with c_score2:
        st.caption(
            f"A pontuação reflete a distribuição de receita entre os status no comparativo do período ({label_periodo})."
        )

    if clientes_carteira_df.empty:
        st.info("Não há clientes com movimento para calcular a carteira.")
        return None, None

    status_counts = (
        clientes_carteira_df.groupby(STATUS_COL)["Cliente"]
        .nunique()
        .reset_index()
        .rename(columns={"Cliente": "QtdClientes", STATUS_COL: "Status"})
    )

    fat_status = (
        clientes_carteira_df.groupby(STATUS_COL)[["ValorAtual", "ValorAnterior"]]
        .sum()
        .reset_index()
        .rename(columns={STATUS_COL: "Status"})
    )
    fat_status["Faturamento"] = fat_status["ValorAtual"] - fat_status["ValorAnterior"]
    fat_status = fat_status[["Status", "Faturamento"]]
    status_counts = status_counts.merge(fat_status, on="Status", how="left")

    total_clientes_ = int(status_counts["QtdClientes"].sum())
    status_counts["%Clientes"] = status_counts["QtdClientes"] / total_clientes_ if total_clientes_ > 0 else 0
    status_counts["Status"] = pd.Categorical(status_counts["Status"], categories=STATUS_ORDER, ordered=True)
    status_counts = status_counts.sort_values("Status")

    col_pie_s, col_table_s = st.columns([1, 1.2])

    chart_pie = None
    with col_pie_s:
        st.caption("Distribuição de clientes por status")
        if total_clientes_ == 0:
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
        st.dataframe(status_disp, use_container_width=True, hide_index=True)

    return status_disp, chart_pie

status_tbl_curr, chart_status_curr = render_carteira_block(clientes_carteira, carteira_score, carteira_label, current_period_label)

if show_prev_carteira:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")
    status_tbl_prev, chart_status_prev = render_carteira_block(clientes_carteira_prev, carteira_score_prev, carteira_label_prev, previous_period_label)
else:
    status_tbl_prev, chart_status_prev = None, None

st.markdown("---")


# ==========================
# STATUS DOS CLIENTES
# ==========================
st.markdown("### Status dos clientes")
show_prev_status_clientes = section_prev_toggle("sec_prev_status_clientes")

search_cliente = st.text_input("Buscar cliente", value="", placeholder="Digite parte do nome do cliente")

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

def render_status_list(clientes_df: pd.DataFrame, label_a: str, label_b: str):
    if clientes_df.empty:
        st.info("Sem dados para status de clientes.")
        return

    for status_name in STATUS_ORDER:
        df_status = clientes_df[clientes_df[STATUS_COL] == status_name].copy()
        if search_cliente:
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
                "FaturamentoAtualFmt": f"Faturamento {label_a}",
                "FaturamentoAnteriorFmt": f"Faturamento {label_b}",
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

# Current list
render_status_list(clientes_carteira, current_period_label, "Período anterior automático (base)")

# Previous list optional
if show_prev_status_clientes:
    st.markdown("---")
    st.markdown(f"**Período anterior: {previous_period_label}**")
    render_status_list(clientes_carteira_prev, previous_period_label, "Período anterior automático (base)")


# ==========================
# PDF GENERATION
# ==========================
pdf_bytes = None
pdf_error = None

if gen_pdf_now:
    try:
        # KPIs dicts for PDF
        kpis_pdf_curr = {
            "Total período": format_brl_compact(total_rep),
            "Média mensal": format_brl_compact(media_mensal),
            "Clientes atendidos": str(clientes_atendidos),
            "Cidades atendidas": str(cidades_atendidas),
            "Estados atendidos": str(estados_atendidos),
            "Saúde da carteira": f"{carteira_score:.0f}/100 ({carteira_label})",
        }
        kpis_pdf_prev = {
            "Total período": format_brl_compact(float(df_rep_prev["Valor"].sum())),
            "Média mensal": format_brl_compact(float(df_rep_prev["Valor"].sum()) / max(1, int(df_rep_prev.groupby(["Ano","MesNum"])["Valor"].sum().gt(0).sum()))),
            "Clientes atendidos": str(int(df_rep_prev["Cliente"].dropna().nunique())),
            "Cidades atendidas": str(int(df_rep_prev[["Estado", "Cidade"]].dropna().drop_duplicates().shape[0])),
            "Estados atendidos": str(int(df_rep_prev["Estado"].dropna().nunique())),
            "Saúde da carteira": f"{carteira_score_prev:.0f}/100 ({carteira_label_prev})",
        }

        # Charts: build PNG bytes (Altair via vl-convert; Plotly via chrome-less fallback if needed)
        charts_png = {}

        # Evolução current (Altair)
        if chart_curr is not None:
            charts_png["Evolução – Período atual"] = altair_to_png_bytes(chart_curr) or b""

        if chart_prev is not None:
            charts_png["Evolução – Período anterior"] = altair_to_png_bytes(chart_prev) or b""

        # Estados pie (Plotly -> fallback to matplotlib)
        if estados_top_curr is not None and len(estados_top_curr) > 0:
            if fig_states_curr is not None:
                png = plotly_to_png_bytes(fig_states_curr)
                if not png:
                    labels = estados_top_curr["Estado"].astype(str).tolist()
                    values = estados_top_curr["Valor"].astype(float).tolist()
                    png = matplotlib_pie_png_bytes(labels, values, title="Estados (Atual)")
                charts_png["Estados – Período atual"] = png

        if estados_top_prev is not None and len(estados_top_prev) > 0:
            if fig_states_prev is not None:
                png = plotly_to_png_bytes(fig_states_prev)
                if not png:
                    labels = estados_top_prev["Estado"].astype(str).tolist()
                    values = estados_top_prev["Valor"].astype(float).tolist()
                    png = matplotlib_pie_png_bytes(labels, values, title="Estados (Anterior)")
                charts_png["Estados – Período anterior"] = png

        # Categorias pie (Plotly -> fallback)
        if fig_cat is not None:
            png = plotly_to_png_bytes(fig_cat)
            if not png and "df_pie" in locals():
                labels = df_pie["Categoria"].astype(str).tolist()
                values = df_pie["Valor"].astype(float).tolist()
                png = matplotlib_pie_png_bytes(labels, values, title="Categorias (Atual)")
            charts_png["Categorias – Período atual"] = png or b""

        # Clientes pie (Plotly -> fallback)
        if fig_clients_curr is not None:
            png = plotly_to_png_bytes(fig_clients_curr)
            if not png and dist_df_curr is not None and not dist_df_curr.empty:
                labels = dist_df_curr["Grupo"].astype(str).tolist()
                values = dist_df_curr["Valor"].astype(float).tolist()
                png = matplotlib_pie_png_bytes(labels, values, title="Clientes (Atual)")
            charts_png["Clientes – Período atual"] = png or b""

        if fig_clients_prev is not None:
            png = plotly_to_png_bytes(fig_clients_prev)
            if not png and dist_df_prev is not None and not dist_df_prev.empty:
                labels = dist_df_prev["Grupo"].astype(str).tolist()
                values = dist_df_prev["Valor"].astype(float).tolist()
                png = matplotlib_pie_png_bytes(labels, values, title="Clientes (Anterior)")
            charts_png["Clientes – Período anterior"] = png or b""

        # Status pie (Altair)
        if chart_status_curr is not None:
            charts_png["Status carteira – Período atual"] = altair_to_png_bytes(chart_status_curr) or b""
        if show_prev_carteira and chart_status_prev is not None:
            charts_png["Status carteira – Período anterior"] = altair_to_png_bytes(chart_status_prev) or b""

        # Tables for PDF
        tables_pdf = {}
        if rep_selected == "Todos":
            # include ranking tables current
            try:
                tables_pdf["Ranking por Faturamento (Atual)"] = rep_fat.copy()
                tables_pdf["Ranking por Volume (Atual)"] = rep_vol.copy()
            except Exception:
                pass
            if show_prev_rep_perf:
                try:
                    tables_pdf["Ranking por Faturamento (Anterior)"] = rep_fat_p.copy()
                    tables_pdf["Ranking por Volume (Anterior)"] = rep_vol_p.copy()
                except Exception:
                    pass

        if clientes_tbl_curr is not None and not clientes_tbl_curr.empty:
            tables_pdf["Clientes – Top 15 (Atual)"] = clientes_tbl_curr.copy()
        if show_prev_clientes and clientes_tbl_prev is not None and not clientes_tbl_prev.empty:
            tables_pdf["Clientes – Top 15 (Anterior)"] = clientes_tbl_prev.copy()

        if status_tbl_curr is not None and not status_tbl_curr.empty:
            tables_pdf["Saúde da carteira – Resumo por status (Atual)"] = status_tbl_curr.copy()
        if show_prev_carteira and status_tbl_prev is not None and not status_tbl_prev.empty:
            tables_pdf["Saúde da carteira – Resumo por status (Anterior)"] = status_tbl_prev.copy()

        titulo_pdf = "Insights de Vendas"
        subtitulo_pdf = f"Representante: {titulo_rep}  •  Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        pdf_bytes = build_pdf_report(
            titulo=titulo_pdf,
            subtitulo=subtitulo_pdf,
            periodo_atual_label=current_period_label,
            periodo_anterior_label=previous_period_label,
            kpis_atual=kpis_pdf_curr,
            kpis_anterior=kpis_pdf_prev,
            charts_png={k: v for k, v in charts_png.items() if v},
            tables=tables_pdf,
        )
    except Exception as e:
        pdf_error = str(e)

if pdf_error:
    st.error(f"Não foi possível gerar o PDF: {pdf_error}")

if pdf_bytes:
    filename = f"Insights_{titulo_rep}_{str(start_year)}{start_month:02d}-{str(end_year)}{end_month:02d}.pdf"
    st.download_button(
        "⬇️ Baixar PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=False,
        key="pdf_download_btn",
    )
