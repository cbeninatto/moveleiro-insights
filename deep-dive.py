import streamlit as st
import pandas as pd
import datetime
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
# PDF EXPORT (REPORT)
# ==========================
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage

try:
    import plotly.io as pio
except Exception:  # pragma: no cover
    pio = None

try:
    import vl_convert as vlc
except Exception:  # pragma: no cover
    vlc = None


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

MONTH_MAP_NUM_TO_NAME = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR",
    5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO",
    9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
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
    prev_end = start_comp - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=months_span - 1)

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
# PDF EXPORT HELPERS
# ==========================
def _safe_str(x) -> str:
    return "" if x is None else str(x)

def _plotly_to_png_bytes(fig, width_px: int = 1400, scale: int = 2):
    """Best-effort Plotly -> PNG bytes. Requires kaleido in the environment."""
    if fig is None or pio is None:
        return None

def _altair_to_png_bytes(chart, width_px: int = 1400, scale: float = 2.0):
    """Best-effort Altair/Vega-Lite -> PNG bytes. Requires vl-convert-python."""
    if chart is None or vlc is None:
        return None
    try:
        spec = chart.to_dict()
        # Ensure width for print
        spec.setdefault("width", width_px)
        return vlc.vegalite_to_png(spec, scale=scale)
    except Exception:
        return None

def _chart_to_png_bytes(obj, width_px: int = 1400, scale: float = 2.0):
    """Convert Plotly or Altair charts to PNG bytes (best effort)."""
    if obj is None:
        return None
    # Plotly
    if hasattr(obj, "to_image"):
        try:
            return obj.to_image(format="png", width=width_px, scale=int(scale))
        except Exception:
            return None
    # Altair
    if hasattr(obj, "to_dict"):
        return _altair_to_png_bytes(obj, width_px=width_px, scale=scale)
    return None
    try:
        return fig.to_image(format="png", width=width_px, scale=scale)
    except Exception:
        return None

def build_pdf_report(
    rep_name: str,
    current_period_label: str,
    previous_period_label: str,
    kpis: dict,
    highlights: dict,
    estados_table: pd.DataFrame | None = None,
    categorias_table: pd.DataFrame | None = None,
    clientes_table: pd.DataFrame | None = None,
    carteira_status_table: pd.DataFrame | None = None,
    chart_items: list[tuple[str, bytes]] | None = None,
):
    """Build a print-ready PDF (landscape A4) and return bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=1.2*cm, rightMargin=1.2*cm, topMargin=1.0*cm, bottomMargin=1.0*cm,
        title="Insights de Vendas - Relatório",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, leading=18, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, leading=14, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyS", parent=styles["BodyText"], fontSize=9.5, leading=12))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5, leading=10, textColor=colors.grey))

    story = []
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph("Insights de Vendas — Relatório", styles["H1"]))
    story.append(Paragraph(f"Representante: <b>{_safe_str(rep_name)}</b>", styles["BodyS"]))
    story.append(Paragraph(f"Período selecionado: <b>{_safe_str(current_period_label)}</b>  |  Período anterior: <b>{_safe_str(previous_period_label)}</b>", styles["BodyS"]))
    story.append(Paragraph(f"Gerado em: {now}", styles["Caption"]))
    story.append(Spacer(1, 10))

    # KPIs
    story.append(Paragraph("Resumo (KPIs)", styles["H2"]))
    kpi_rows = [["Métrica", "Valor"]] + [[k, v] for k, v in kpis.items()]
    t = Table(kpi_rows, colWidths=[8.0*cm, 18.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 9.5),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#e5e7eb")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Highlights
    story.append(Paragraph("Destaques do período", styles["H2"]))
    bullets = []
    if highlights:
        for k, v in highlights.items():
            bullets.append(f"• <b>{_safe_str(k)}:</b> {_safe_str(v)}")
    story.append(Paragraph("<br/>".join(bullets) if bullets else "Sem destaques para o período.", styles["BodyS"]))
    story.append(Spacer(1, 10))
    # Charts (PNG) — 2 per page (landscape)
    if chart_items:
        story.append(Paragraph("Gráficos", styles["H2"]))
        # page width for content ~ 26cm; 2 charts side by side
        chart_w = 12.7*cm
        chart_h = 7.6*cm

        # chunk by 2
        chunk = []
        for title, png in chart_items:
            if not png:
                continue
            chunk.append((title, png))
            if len(chunk) == 2:
                row_imgs = []
                row_caps = []
                for ttitle, tpng in chunk:
                    row_imgs.append(RLImage(io.BytesIO(tpng), width=chart_w, height=chart_h))
                    row_caps.append(Paragraph(ttitle, styles["Caption"]))
                tbl = Table([row_imgs, row_caps], colWidths=[chart_w, chart_w])
                tbl.setStyle(TableStyle([
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING", (0,0), (-1,-1), 0),
                    ("RIGHTPADDING", (0,0), (-1,-1), 0),
                    ("TOPPADDING", (0,0), (-1,-1), 2),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 6))
                chunk = []

        # last odd chart
        if len(chunk) == 1:
            ttitle, tpng = chunk[0]
            tbl = Table([[RLImage(io.BytesIO(tpng), width=chart_w, height=chart_h), ""] ,
                         [Paragraph(ttitle, styles["Caption"]), ""]], colWidths=[chart_w, chart_w])
            tbl.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 0),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.append(tbl)

        story.append(PageBreak())

    def _add_df_table(title: str, df_in: pd.DataFrame | None, max_rows: int = 25):
        if df_in is None or df_in.empty:
            return
        story.append(Paragraph(title, styles["H2"]))
        df_show = df_in.head(max_rows).copy()
        data = [list(df_show.columns)] + df_show.astype(str).values.tolist()
        # simple auto widths: spread evenly
        col_count = len(df_show.columns)
        col_w = (26.0*cm) / max(col_count, 1)
        tbl = Table(data, colWidths=[col_w]*col_count, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 8.8),
            ("FONTSIZE", (0,1), (-1,-1), 8.3),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#e5e7eb")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9fafb")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))

    _add_df_table("Resumo – Top estados", estados_table, max_rows=15)
    _add_df_table("Resumo – Categorias", categorias_table, max_rows=25)
    _add_df_table("Resumo – Top clientes", clientes_table, max_rows=25)
    _add_df_table("Saúde da carteira – Resumo", carteira_status_table, max_rows=10)

    doc.build(story)
    return buf.getvalue()

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
st.sidebar.markdown("### Período")

anos_disponiveis = sorted(df["Ano"].dropna().unique())
if not anos_disponiveis:
    st.error("Não foi possível identificar anos na base de dados.")
    st.stop()

last_year = int(anos_disponiveis[-1])

meses_ano_default = df.loc[df["Ano"] == last_year, "MesNum"].dropna().unique()
default_start_month_num = int(meses_ano_default.min()) if len(meses_ano_default) else 1
default_end_month_num = int(meses_ano_default.max()) if len(meses_ano_default) else 12

month_names = [MONTH_MAP_NUM_TO_NAME[m] for m in range(1, 13)]

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

months_span_for_carteira = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1
prev_end = start_comp - pd.DateOffset(months=1)
prev_start = prev_end - pd.DateOffset(months=months_span_for_carteira - 1)

current_period_label = format_period_label(start_comp, end_comp)
previous_period_label = format_period_label(prev_start, prev_end)

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
rep_selected = st.sidebar.selectbox("Representante", rep_options)

df_rep = df_period.copy() if rep_selected == "Todos" else df_period[df_period["Representante"] == rep_selected].copy()

mask_prev_period = (df["Competencia"] >= prev_start) & (df["Competencia"] <= prev_end)
df_prev_period = df.loc[mask_prev_period].copy()
df_rep_prev = df_prev_period.copy() if rep_selected == "Todos" else df_prev_period[df_prev_period["Representante"] == rep_selected].copy()

clientes_carteira = build_carteira_status(df, rep_selected, start_comp, end_comp)

# ==========================
# HEADER
# ==========================
st.title("Insights de Vendas")
titulo_rep = "Todos" if rep_selected == "Todos" else rep_selected
# Representative row + PDF button
_rep_left, _rep_right = st.columns([0.76, 0.24], vertical_alignment="center")
with _rep_left:
    st.subheader(f"Representante: **{titulo_rep}**")
with _rep_right:
    _gen = st.button("📄 Gerar PDF", use_container_width=True, key="pdf_gen_btn")


# ==========================
# PDF EXPORT (REPORT) - HEADER RIGHT BUTTON
# ==========================
if "pdf_report_bytes" not in st.session_state:
    st.session_state["pdf_report_bytes"] = None
    st.session_state["pdf_report_name"] = None
    st.session_state["pdf_report_error"] = None

_rep_left, _rep_right = st.columns([0.76, 0.24], vertical_alignment="center")
with _rep_left:
    # repeat the representative line here to align with the button (hide the original spacing)
    pass

with _rep_right:
    _gen = st.button("📄 Gerar PDF", use_container_width=True, key="pdf_gen_btn")

    if _gen:
        st.session_state["pdf_report_error"] = None
        st.session_state["pdf_report_bytes"] = None
        st.session_state["pdf_report_name"] = None

        if "df_rep" not in globals() or df_rep is None or df_rep.empty:
            st.session_state["pdf_report_error"] = "Sem dados do período atual para gerar o relatório. Ajuste os filtros e tente novamente."
        else:
            # --- Compute KPIs from df_rep (safe) ---
            total_rep_pdf = float(df_rep["Valor"].sum()) if "Valor" in df_rep.columns else 0.0

            try:
                mensal_pdf = (
                    df_rep.groupby(["Ano", "MesNum"], as_index=False)[["Valor", "Quantidade"]]
                    .sum()
                    .sort_values(["Ano", "MesNum"])
                )
                media_mensal_pdf = float(mensal_pdf["Valor"].mean()) if not mensal_pdf.empty else 0.0
            except Exception:
                mensal_pdf = pd.DataFrame()
                media_mensal_pdf = 0.0

            clientes_atendidos_pdf = int(df_rep["Cliente"].nunique()) if "Cliente" in df_rep.columns else 0
            cidades_atendidas_pdf = int(df_rep["Cidade"].nunique()) if "Cidade" in df_rep.columns else 0
            estados_atendidos_pdf = int(df_rep["Estado"].nunique()) if "Estado" in df_rep.columns else 0

            # --- N80 (on the fly) ---
            n80_count_pdf = 0
            try:
                if "Cliente" in df_rep.columns and "Valor" in df_rep.columns and clientes_atendidos_pdf > 0:
                    by_client = df_rep.groupby("Cliente", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False)
                    by_client["share"] = by_client["Valor"] / max(by_client["Valor"].sum(), 1e-9)
                    by_client["cum_share"] = by_client["share"].cumsum()
                    n80_count_pdf = int((by_client["cum_share"] <= 0.80).sum())
                    if n80_count_pdf < len(by_client):
                        n80_count_pdf += 1
            except Exception:
                n80_count_pdf = 0

            # --- HHI (on the fly) ---
            hhi_value_pdf = 0.0
            hhi_label_short_pdf = "—"
            try:
                if "Cliente" in df_rep.columns and "Valor" in df_rep.columns and total_rep_pdf > 0:
                    shares = (df_rep.groupby("Cliente")["Valor"].sum() / total_rep_pdf).clip(lower=0)
                    hhi_value_pdf = float((shares ** 2).sum())
                    if hhi_value_pdf < 0.10:
                        hhi_label_short_pdf = "Baixa"
                    elif hhi_value_pdf < 0.18:
                        hhi_label_short_pdf = "Média"
                    else:
                        hhi_label_short_pdf = "Alta"
            except Exception:
                pass

            # --- Period labels / rep name ---
            rep_name_pdf = titulo_rep
            current_period_label_pdf = globals().get("current_period_label", "Período atual")
            previous_period_label_pdf = globals().get("previous_period_label", "Período anterior")

            # Prepare tables for the report
            estados_tbl_pdf = None
            categorias_tbl_pdf = None
            clientes_tbl_pdf = None
            carteira_tbl_pdf = None

            try:
                estados_df_pdf = df_rep.groupby("Estado", as_index=False)[["Valor", "Quantidade"]].sum().sort_values("Valor", ascending=False)
                if not estados_df_pdf.empty:
                    estados_df_pdf["Faturamento"] = estados_df_pdf["Valor"].map(format_brl)
                    estados_df_pdf["Volume"] = estados_df_pdf["Quantidade"].map(format_un)
                    estados_tbl_pdf = estados_df_pdf[["Estado", "Faturamento", "Volume"]].head(15)
            except Exception:
                estados_tbl_pdf = None

            try:
                if "df_rep_prev" in globals() and df_rep_prev is not None and not df_rep_prev.empty and "Categoria" in df_rep.columns and "Categoria" in df_rep_prev.columns:
                    curr_cat_pdf = df_rep.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "ValorAtual"})
                    prev_cat_pdf = df_rep_prev.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "ValorAnterior"})
                    cat_pdf = pd.merge(curr_cat_pdf, prev_cat_pdf, on="Categoria", how="outer").fillna(0.0)
                else:
                    curr_cat_pdf = df_rep.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "ValorAtual"})
                    cat_pdf = curr_cat_pdf.copy()
                    cat_pdf["ValorAnterior"] = 0.0

                cat_pdf["Valor"] = cat_pdf["ValorAtual"].map(format_brl)
                cat_pdf["Variação (R$)"] = (cat_pdf["ValorAtual"] - cat_pdf["ValorAnterior"]).map(format_brl_signed)
                categorias_tbl_pdf = cat_pdf.sort_values("ValorAtual", ascending=False)[["Categoria", "Valor", "Variação (R$)"]].head(25)
            except Exception:
                categorias_tbl_pdf = None

            try:
                cli_pdf = (
                    df_rep.groupby(["Cliente", "Estado", "Cidade"], as_index=False)
                    .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
                    .sort_values("Valor", ascending=False)
                )
                cli_pdf["Faturamento"] = cli_pdf["Valor"].map(format_brl)
                cli_pdf["Volume"] = cli_pdf["Quantidade"].map(format_un)
                clientes_tbl_pdf = cli_pdf[["Cliente", "Cidade", "Estado", "Faturamento", "Volume"]].head(25)
            except Exception:
                clientes_tbl_pdf = None

            try:
                clientes_carteira_local = globals().get("clientes_carteira", None)
                if clientes_carteira_local is not None and hasattr(clientes_carteira_local, "empty") and not clientes_carteira_local.empty:
                    status_col = globals().get("STATUS_COL", "Status")
                    status_order = globals().get("STATUS_ORDER", None)
                    status_counts_pdf = (
                        clientes_carteira_local.groupby(status_col)["Cliente"]
                        .nunique()
                        .reset_index()
                        .rename(columns={status_col: "Status", "Cliente": "Clientes"})
                    )
                    if status_order is not None:
                        status_counts_pdf["Status"] = pd.Categorical(status_counts_pdf["Status"], categories=status_order, ordered=True)
                        status_counts_pdf = status_counts_pdf.sort_values("Status")
                    carteira_tbl_pdf = status_counts_pdf
            except Exception:
                carteira_tbl_pdf = None

            kpis_pdf = {
                "Total período": format_brl(total_rep_pdf),
                "Média mensal": format_brl(media_mensal_pdf),
                "Clientes atendidos": str(clientes_atendidos_pdf),
                "Cidades atendidas": str(cidades_atendidas_pdf),
                "Estados atendidos": str(estados_atendidos_pdf),
                "N80": f"{n80_count_pdf} ({(n80_count_pdf/clientes_atendidos_pdf if clientes_atendidos_pdf else 0):.0%} da carteira)",
                "Concentração (HHI)": f"{hhi_label_short_pdf} ({hhi_value_pdf:.3f})",
            }

            highlights_pdf = {}
            try:
                if mensal_pdf is not None and not mensal_pdf.empty:
                    mensal_pdf = mensal_pdf.copy()
                    mensal_pdf["Competencia"] = pd.to_datetime(dict(year=mensal_pdf["Ano"], month=mensal_pdf["MesNum"], day=1))
                    mensal_pdf["MesLabel"] = mensal_pdf["Competencia"].apply(lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}")
                    best_fat_pdf = mensal_pdf.loc[mensal_pdf["Valor"].idxmax()]
                    worst_fat_pdf = mensal_pdf.loc[mensal_pdf["Valor"].idxmin()]
                    best_vol_pdf = mensal_pdf.loc[mensal_pdf["Quantidade"].idxmax()]
                    worst_vol_pdf = mensal_pdf.loc[mensal_pdf["Quantidade"].idxmin()]
                    highlights_pdf = {
                        "Melhor mês (Faturamento)": f"{best_fat_pdf['MesLabel']} — {format_brl(best_fat_pdf['Valor'])}",
                        "Pior mês (Faturamento)": f"{worst_fat_pdf['MesLabel']} — {format_brl(worst_fat_pdf['Valor'])}",
                        "Melhor mês (Volume)": f"{best_vol_pdf['MesLabel']} — {format_un(best_vol_pdf['Quantidade'])}",
                        "Pior mês (Volume)": f"{worst_vol_pdf['MesLabel']} — {format_un(worst_vol_pdf['Quantidade'])}",
                    }
            except Exception:
                highlights_pdf = {}

            # Build charts for the PDF (do NOT rely on globals defined later)
            chart_items = []
            try:
                import altair as alt
            except Exception:
                alt = None

            try:
                import plotly.express as px
            except Exception:
                px = None

            # 1) Monthly trend (Altair) - Faturamento
            if alt is not None and mensal_pdf is not None and not mensal_pdf.empty:
                try:
                    tmp = mensal_pdf.copy()
                    tmp["Competencia"] = pd.to_datetime(dict(year=tmp["Ano"], month=tmp["MesNum"], day=1))
                    c1 = (
                        alt.Chart(tmp)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Competencia:T", title="Mês"),
                            y=alt.Y("Valor:Q", title="Faturamento (R$)"),
                            tooltip=["Ano:O", "MesNum:O", "Valor:Q"],
                        )
                        .properties(width=650, height=260, title="Histórico — Faturamento")
                    )
                    png = _chart_to_png_bytes(c1, width_px=1400, scale=2.0)
                    if png:
                        chart_items.append(("Histórico — Faturamento", png))
                except Exception:
                    pass

                # 2) Monthly trend (Altair) - Volume
                try:
                    tmp = mensal_pdf.copy()
                    tmp["Competencia"] = pd.to_datetime(dict(year=tmp["Ano"], month=tmp["MesNum"], day=1))
                    c2 = (
                        alt.Chart(tmp)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Competencia:T", title="Mês"),
                            y=alt.Y("Quantidade:Q", title="Volume (un)"),
                            tooltip=["Ano:O", "MesNum:O", "Quantidade:Q"],
                        )
                        .properties(width=650, height=260, title="Histórico — Volume")
                    )
                    png = _chart_to_png_bytes(c2, width_px=1400, scale=2.0)
                    if png:
                        chart_items.append(("Histórico — Volume", png))
                except Exception:
                    pass

            # 3) Estados (Plotly) - Top 10
            if px is not None and "Estado" in df_rep.columns and "Valor" in df_rep.columns:
                try:
                    st_df = df_rep.groupby("Estado", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False).head(10)
                    fig_st = px.bar(st_df, x="Estado", y="Valor", title="Top 10 — Estados (Faturamento)")
                    png = _chart_to_png_bytes(fig_st, width_px=1400, scale=2.0)
                    if png:
                        chart_items.append(("Top 10 — Estados (Faturamento)", png))
                except Exception:
                    pass

            # 4) Categorias (Plotly) - share
            if px is not None and "Categoria" in df_rep.columns and "Valor" in df_rep.columns:
                try:
                    cat_df = df_rep.groupby("Categoria", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False).head(10)
                    fig_cat_pdf = px.pie(cat_df, names="Categoria", values="Valor", title="Top 10 — Categorias (participação)")
                    png = _chart_to_png_bytes(fig_cat_pdf, width_px=1400, scale=2.0)
                    if png:
                        chart_items.append(("Top 10 — Categorias (participação)", png))
                except Exception:
                    pass

            # 5) Clientes (Plotly) - Top 10
            if px is not None and "Cliente" in df_rep.columns and "Valor" in df_rep.columns:
                try:
                    cli_df = df_rep.groupby("Cliente", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False).head(10)
                    fig_cli = px.bar(cli_df, x="Cliente", y="Valor", title="Top 10 — Clientes (Faturamento)")
                    png = _chart_to_png_bytes(fig_cli, width_px=1400, scale=2.0)
                    if png:
                        chart_items.append(("Top 10 — Clientes (Faturamento)", png))
                except Exception:
                    pass

            try:
                pdf_bytes = build_pdf_report(
                    rep_name=rep_name_pdf,
                    current_period_label=current_period_label_pdf,
                    previous_period_label=previous_period_label_pdf,
                    kpis=kpis_pdf,
                    highlights=highlights_pdf,
                    estados_table=estados_tbl_pdf,
                    categorias_table=categorias_tbl_pdf,
                    clientes_table=clientes_tbl_pdf,
                    carteira_status_table=carteira_tbl_pdf,
                    chart_items=chart_items if chart_items else None,
                )
                safe_rep = re.sub(r"[^A-Za-z0-9_-]+", "_", str(rep_name_pdf))[:40]
                start_comp = globals().get("start_comp", None)
                end_comp = globals().get("end_comp", None)
                if start_comp is not None and end_comp is not None and hasattr(start_comp, "strftime"):
                    suffix = f"{start_comp.strftime('%Y%m')}-{end_comp.strftime('%Y%m')}"
                else:
                    suffix = "periodo"
                file_name = f"relatorio_insights_{safe_rep}_{suffix}.pdf"

                st.session_state["pdf_report_bytes"] = pdf_bytes
                st.session_state["pdf_report_name"] = file_name
            except Exception as ex:
                st.session_state["pdf_report_error"] = str(ex)

# Show download below header (if generated)
if st.session_state.get("pdf_report_error"):
    st.error(f"Não foi possível gerar o PDF: {st.session_state['pdf_report_error']}")
if st.session_state.get("pdf_report_bytes") and st.session_state.get("pdf_report_name"):
    st.download_button(
        "⬇️ Baixar PDF",
        data=st.session_state["pdf_report_bytes"],
        file_name=st.session_state["pdf_report_name"],
        mime="application/pdf",
        use_container_width=False,
    )

st.caption(f"Período selecionado: {current_period_label}")
st.markdown("---")

# ==========================
# EXPORT PDF (REPORT)
# ==========================

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

st.markdown("---")

# ==========================
# MAPA DE CLIENTES
# ==========================
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

            metric_choice = st.radio("Métrica do mapa", ["Faturamento", "Volume"], horizontal=True)
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

st.markdown("---")

# ==========================
# DISTRIBUIÇÃO POR ESTADOS
# ==========================
st.subheader("Distribuição por estados")

if df_rep.empty:
    st.info("Não há vendas no período selecionado.")
else:
    estados_df = df_rep.groupby("Estado", as_index=False)[["Valor", "Quantidade"]].sum().sort_values("Valor", ascending=False)

    total_valor_all = float(estados_df["Valor"].sum())
    total_qtd_all = float(estados_df["Quantidade"].sum())

    if total_valor_all <= 0:
        st.info("Não há faturamento para distribuir por estados nesse período.")
    else:
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
            st.caption("Top 10 estados por faturamento – % do faturamento total")
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

        cidades_top = (
            df_rep.groupby(["Cidade", "Estado"], as_index=False)
            .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
            .sort_values("Valor", ascending=False)
            .head(15)
            .copy()
        )
        cidades_top["% Faturamento"] = cidades_top["Valor"] / total_valor_all if total_valor_all > 0 else 0
        cidades_top["% Volume"] = cidades_top["Quantidade"] / total_qtd_all if total_qtd_all > 0 else 0

        city_client_agg = (
            df_rep.groupby(["Cidade", "Estado", "Cliente"], as_index=False)["Valor"]
            .sum()
            .sort_values(["Cidade", "Estado", "Valor"], ascending=[True, True, False])
        )
        city_client_agg["Rank"] = city_client_agg.groupby(["Cidade", "Estado"]).cumcount() + 1
        city_client_agg = city_client_agg[city_client_agg["Rank"] <= 3].copy()

        top_cli = city_client_agg.pivot_table(index=["Cidade", "Estado"], columns="Rank", values="Cliente", aggfunc="first")
        top_val = city_client_agg.pivot_table(index=["Cidade", "Estado"], columns="Rank", values="Valor", aggfunc="first")

        top3_wide = pd.DataFrame(index=top_cli.index).reset_index()
        for i in [1, 2, 3]:
            top3_wide[f"TopCliente_{i}"] = top_cli[i].values if i in top_cli.columns else ""
            top3_wide[f"TopValor_{i}"] = top_val[i].values if i in top_val.columns else float("nan")

        cidades_disp = cidades_top.merge(top3_wide, on=["Cidade", "Estado"], how="left")

        cidades_disp["Faturamento"] = cidades_disp["Valor"].map(format_brl)
        cidades_disp["Volume"] = cidades_disp["Quantidade"].map(format_un)
        cidades_disp["% Faturamento"] = cidades_disp["% Faturamento"].map(lambda x: f"{x:.1%}")
        cidades_disp["% Volume"] = cidades_disp["% Volume"].map(lambda x: f"{x:.1%}")

        def client_cell(name, val):
            if name is None or str(name).strip() == "" or pd.isna(name):
                return ""
            full = str(name).strip()
            short = shorten_name(full, 28)
            return (
                f"<span title='{html.escape(full)}'>{html.escape(short)}</span>"
                f"<br><span style='opacity:.85'>{html.escape(format_brl_compact(val))}</span>"
            )

        cidades_disp["Top 1 cliente"] = cidades_disp.apply(lambda r: client_cell(r.get("TopCliente_1"), r.get("TopValor_1")), axis=1)
        cidades_disp["Top 2 cliente"] = cidades_disp.apply(lambda r: client_cell(r.get("TopCliente_2"), r.get("TopValor_2")), axis=1)
        cidades_disp["Top 3 cliente"] = cidades_disp.apply(lambda r: client_cell(r.get("TopCliente_3"), r.get("TopValor_3")), axis=1)

        cidades_disp = cidades_disp[
            ["Cidade", "Estado", "Faturamento", "% Faturamento", "Volume", "% Volume", "Top 1 cliente", "Top 2 cliente", "Top 3 cliente"]
        ]

        st.markdown("**Principais cidades por faturamento**")
        st.markdown(
            """
<style>
table.cidades-top { width: 100%; border-collapse: collapse; }
table.cidades-top th, table.cidades-top td {
  padding: 0.22rem 0.45rem;
  font-size: 0.80rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  text-align: left;
  vertical-align: top;
}
table.cidades-top th:nth-child(-n+6), table.cidades-top td:nth-child(-n+6) { white-space: nowrap; }
table.cidades-top th:nth-child(n+7), table.cidades-top td:nth-child(n+7) { white-space: normal; }
table.cidades-top th:nth-child(7), table.cidades-top th:nth-child(8), table.cidades-top th:nth-child(9) { width: 18%; }
</style>
""",
            unsafe_allow_html=True,
        )

        cols_ct = list(cidades_disp.columns)
        html_ct = "<table class='cidades-top'><thead><tr>"
        html_ct += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_ct)
        html_ct += "</tr></thead><tbody>"
        for _, r in cidades_disp.iterrows():
            html_ct += "<tr>"
            for c in cols_ct:
                val = r[c]
                if c.startswith("Top "):
                    html_ct += f"<td>{val if isinstance(val, str) else ''}</td>"
                else:
                    html_ct += f"<td>{html.escape(str(val))}</td>"
            html_ct += "</tr>"
        html_ct += "</tbody></table>"
        st.markdown(html_ct, unsafe_allow_html=True)

st.markdown("---")

# ==========================
# PAGE BREAK (print)
# ==========================
st.markdown("<div class='page-break'></div>", unsafe_allow_html=True)

# ==========================
# EVOLUÇÃO – FATURAMENTO x VOLUME
# ==========================
st.subheader("Evolução – Faturamento x Volume")

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
# CATEGORIAS VENDIDAS (Δ renamed to Variação)
# ==========================
st.markdown("---")
st.subheader("Categorias vendidas")

if df_rep.empty:
    st.info("Não há vendas no período selecionado.")
else:
    curr_cat = (
        df_rep.groupby("Categoria", as_index=False)["Valor"]
        .sum()
        .rename(columns={"Valor": "ValorAtual"})
    )
    prev_cat = (
        df_rep_prev.groupby("Categoria", as_index=False)["Valor"]
        .sum()
        .rename(columns={"Valor": "ValorAnterior"})
    )

    cat = pd.merge(curr_cat, prev_cat, on="Categoria", how="outer")
    cat["ValorAtual"] = pd.to_numeric(cat["ValorAtual"], errors="coerce").fillna(0.0)
    cat["ValorAnterior"] = pd.to_numeric(cat["ValorAnterior"], errors="coerce").fillna(0.0)

    total_cat = float(cat["ValorAtual"].sum())
    if total_cat <= 0:
        st.info("Sem faturamento para exibir categorias nesse período.")
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
            fig_cat.update_traces(
                text=df_pie["Text"],
                textposition="inside",
                textinfo="text",
                insidetextorientation="radial",
            )
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

st.markdown("---")

# ==========================
# DISTRIBUIÇÃO POR CLIENTES
# ==========================
st.subheader("Distribuição por clientes")

if df_rep.empty or clientes_atendidos == 0:
    st.info("Nenhum cliente com vendas no período selecionado.")
else:
    df_clientes_full = (
        df_rep.groupby(["Cliente", "Estado", "Cidade"], as_index=False)
        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
        .sort_values("Valor", ascending=False)
    )
    total_rep_safe = total_rep if total_rep > 0 else 1.0
    df_clientes_full["Share"] = df_clientes_full["Valor"] / total_rep_safe

    k1, k2, k3, k4, k5 = st.columns(5)
    n80_ratio = (n80_count / clientes_atendidos) if clientes_atendidos > 0 else 0.0
    k1.metric("N80", f"{n80_count}", f"{n80_ratio:.0%} da carteira")
    k2.metric("Índice de concentração", hhi_label_short, f"HHI {hhi_value:.3f}")
    k3.metric("Top 1 cliente", f"{top1_share:.1%}")
    k4.metric("Top 3 clientes", f"{top3_share:.1%}")
    k5.metric("Top 10 clientes", f"{top10_share:.1%}")

    col_pie, col_tbl = st.columns([1.10, 1.50])

    with col_pie:
        st.caption("Participação dos clientes (Top 10 destacados)")

        df_pie = df_clientes_full[["Cliente", "Valor"]].copy()
        df_pie = df_pie.groupby("Cliente", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False)
        df_pie["Rank"] = range(1, len(df_pie) + 1)
        df_pie["Grupo"] = df_pie.apply(lambda r: r["Cliente"] if r["Rank"] <= 10 else "Outros", axis=1)

        dist_df = df_pie.groupby("Grupo", as_index=False)["Valor"].sum()
        dist_df["Share"] = dist_df["Valor"] / total_rep_safe
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
        fig.update_layout(showlegend=False, height=560, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with col_tbl:
        st.caption("Resumo – clientes (mais detalhado)")

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

st.markdown("---")

# ==========================
# SAÚDE DA CARTEIRA – DETALHES
# ==========================
st.subheader("Saúde da carteira – Detalhes")

c_score1, c_score2 = st.columns([0.35, 0.65])
with c_score1:
    st.metric("Pontuação – Saúde da carteira", f"{carteira_score:.0f} / 100", carteira_label)
with c_score2:
    st.caption(
        "A pontuação reflete a distribuição de receita entre os status (Novos/Crescendo/Estáveis/Caindo/Perdidos) "
        "no comparativo entre período atual e anterior."
    )

if clientes_carteira.empty:
    st.info("Não há clientes com movimento nos períodos atual / anterior para calcular a carteira.")
else:
    status_counts = (
        clientes_carteira.groupby(STATUS_COL)["Cliente"]
        .nunique()
        .reset_index()
        .rename(columns={"Cliente": "QtdClientes", STATUS_COL: "Status"})
    )

    fat_status = (
        clientes_carteira.groupby(STATUS_COL)[["ValorAtual", "ValorAnterior"]]
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
                f"entre o período atual (<b>{current_period_label}</b>) e o período anterior "
                f"(<b>{previous_period_label}</b>). Valores positivos indicam crescimento; "
                f"valores negativos indicam queda.</span>"
            ),
            unsafe_allow_html=True,
        )

st.markdown("---")

# ==========================
# PAGE BREAK (print)
# ==========================
st.markdown("<div class='page-break'></div>", unsafe_allow_html=True)

# ==========================
# STATUS DOS CLIENTES
# ==========================
st.markdown("### Status dos clientes")

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

for status_name in STATUS_ORDER:
    df_status = clientes_carteira[clientes_carteira[STATUS_COL] == status_name].copy()
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
