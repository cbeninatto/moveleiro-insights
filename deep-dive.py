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
# PDF EXPORT (ReportLab + chart capture)
# ==========================
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, 
    Image as RLImage, KeepInFrame
)

try:
    import vl_convert as vlc  # vl-convert-python
except Exception:
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
MAP_BIN_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

# ==========================
# HELPERS
# ==========================
def format_brl(value: float) -> str:
    if pd.isna(value): return "R$ 0,00"
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_brl_compact(value: float) -> str:
    if pd.isna(value): return "R$ 0"
    v = float(value)
    av = abs(v)
    if av >= 1_000_000_000: return "R$ " + f"{v/1_000_000_000:.1f} bi".replace(".", ",")
    if av >= 1_000_000: return "R$ " + f"{v/1_000_000:.1f} mi".replace(".", ",")
    if av >= 1_000: return "R$ " + f"{v/1_000:.1f} mil".replace(".", ",")
    return format_brl(v)

def format_brl_signed(value: float) -> str:
    if pd.isna(value): return "R$ 0,00"
    v = float(value); sign = "-" if v < 0 else ""
    return sign + format_brl(abs(v))

def format_un(value: float) -> str:
    if pd.isna(value): return "0 un"
    try: v = int(round(float(value)))
    except: v = 0
    return f"{v:,}".replace(",", ".") + " un"

def st_plotly_stretch(fig, height=None, key=None):
    if fig: st.plotly_chart(fig, use_container_width=True, height=height, key=key)

def st_altair_stretch(chart, key=None):
    if chart: st.altair_chart(chart, use_container_width=True, key=key)

def build_dynamic_bins(values, is_valor: bool):
    cleaned = sorted([float(v) for v in values if pd.notna(v) and float(v) >= 0])
    def fmt(v): return format_brl(v) if is_valor else format_un(v)
    if not cleaned:
        return [{"min": 0, "color": MAP_BIN_COLORS[3], "label": "0"}]
    n = len(cleaned)
    q1, q2, q3 = cleaned[int(0.25*(n-1))], cleaned[int(0.5*(n-1))], cleaned[int(0.75*(n-1))]
    return [
        {"min": q3, "color": MAP_BIN_COLORS[0], "label": f"{fmt(q3)}+"},
        {"min": q2, "color": MAP_BIN_COLORS[1], "label": f"{fmt(q2)} - {fmt(q3)}"},
        {"min": q1, "color": MAP_BIN_COLORS[2], "label": f"{fmt(q1)} - {fmt(q2)}"},
        {"min": 0,  "color": MAP_BIN_COLORS[3], "label": f"0 - {fmt(q1)}"},
    ]

def get_bin_for_value(v: float, bins):
    for b in bins:
        if v >= b["min"]: return b
    return bins[-1]

@st.cache_data(ttl=3600)
def load_data():
    resp = requests.get(f"{GITHUB_CSV_URL}?cb={int(time.time())}", timeout=60)
    df = pd.read_csv(io.StringIO(resp.text))
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")
    df["Competencia"] = pd.to_datetime(dict(year=df["Ano"], month=df["MesNum"], day=1), errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_geo():
    resp = requests.get(f"{CITY_GEO_CSV_URL}?cb={int(time.time())}", timeout=60)
    df_geo = pd.read_csv(io.StringIO(resp.text))
    df_geo["key"] = df_geo["Estado"].astype(str).str.upper() + "|" + df_geo["Cidade"].astype(str).str.upper()
    return df_geo

def compute_carteira_score(clientes_carteira):
    if clientes_carteira.empty: return 50.0, "Neutra"
    df = clientes_carteira.copy()
    df["Peso"] = df[["ValorAtual", "ValorAnterior"]].max(axis=1)
    total = df["Peso"].sum()
    if total <= 0: return 50.0, "Neutra"
    score_bruto = sum(STATUS_WEIGHTS.get(s, 0) * (v/total) for s, v in df.groupby(STATUS_COL)["Peso"].sum().items())
    isc = max(0.0, min(100.0, (score_bruto + 2) / 4 * 100))
    label = "Crítica" if isc < 30 else "Alerta" if isc < 50 else "Neutra" if isc < 70 else "Saudável"
    return float(isc), label

def build_carteira_status(df_all, rep, start_comp, end_comp):
    df_rep_all = df_all if rep == "Todos" else df_all[df_all["Representante"] == rep]
    if df_rep_all.empty: return pd.DataFrame()
    months_span = (end_comp.year - start_comp.year) * 12 + (end_comp.month - start_comp.month) + 1
    prev_end = start_comp - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=months_span - 1)
    curr = df_rep_all[(df_rep_all["Competencia"] >= start_comp) & (df_rep_all["Competencia"] <= end_comp)]
    prev = df_rep_all[(df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)]
    curr_agg = curr.groupby("Cliente").agg({"Valor":"sum", "Estado":"first", "Cidade":"first"}).rename(columns={"Valor":"ValorAtual"})
    prev_agg = prev.groupby("Cliente").agg({"Valor":"sum"}).rename(columns={"Valor":"ValorAnterior"})
    cl = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer").fillna(0)
    def classify(r):
        if r.ValorAtual > 0 and r.ValorAnterior == 0: return "Novos"
        if r.ValorAtual == 0 and r.ValorAnterior > 0: return "Perdidos"
        ratio = r.ValorAtual / r.ValorAnterior if r.ValorAnterior > 0 else 1
        return "Crescendo" if ratio >= 1.2 else "Caindo" if ratio <= 0.8 else "Estáveis"
    cl[STATUS_COL] = cl.apply(classify, axis=1)
    return cl

# ==========================
# PDF GENERATION CORE
# ==========================
def _chart_to_png(obj, width_px=1600, scale=2.0):
    if not obj: return None
    try:
        if hasattr(obj, "to_image"): return obj.to_image(format="png", width=width_px, scale=int(scale))
        if hasattr(obj, "to_dict") and vlc: 
            spec = obj.to_dict(); spec.setdefault("width", width_px)
            return vlc.vegalite_to_png(spec, scale=scale)
    except: return None
    return None

def build_pdf_report(rep_name, current_label, previous_label, kpis, highlights, chart_items, tables):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1*cm, rightMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1_C", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=10))
    styles.add(ParagraphStyle(name="KPI_T", parent=styles["BodyText"], fontSize=10, textColor=colors.grey))
    styles.add(ParagraphStyle(name="KPI_V", parent=styles["BodyText"], fontSize=14, leading=16, fontName="Helvetica-Bold"))
    
    story = [Paragraph(f"Insights de Vendas - {rep_name}", styles["H1_C"]), Paragraph(f"Período: {current_label} | Gerado em: {datetime.datetime.now().strftime('%d/%m/%Y')}", styles["BodyText"]), Spacer(1, 10)]
    
    # KPI Row
    kpi_cards = []
    for k, v in list(kpis.items())[:5]:
        kpi_cards.append(Table([[Paragraph(k, styles["KPI_T"])], [Paragraph(v, styles["KPI_V"])]], colWidths=[5*cm]))
    story.append(Table([kpi_cards], colWidths=[5.4*cm]*5)); story.append(Spacer(1, 15))
    
    # Highlights
    story.append(Paragraph("<b>Destaques</b>", styles["Heading2"]))
    for line in highlights: story.append(Paragraph(f"• {line}", styles["BodyText"]))
    story.append(Spacer(1, 15))

    # Charts
    for title, png in chart_items:
        if png:
            story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
            story.append(RLImage(io.BytesIO(png), width=26*cm, height=8*cm))
            story.append(Spacer(1, 10))

    # Tables
    for title, df, rows in tables:
        if not df.empty:
            story.append(PageBreak())
            story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
            data = [df.columns.to_list()] + df.head(rows).values.tolist()
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.black),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
            story.append(t)

    doc.build(story)
    return buf.getvalue()

# ==========================
# MAIN APP LOGIC
# ==========================
df = load_data()
anos = sorted(df["Ano"].dropna().unique())
rep_options = ["Todos"] + sorted(df["Representante"].unique().tolist())

st.sidebar.title("Filtros")
sel_year = st.sidebar.selectbox("Ano", anos, index=len(anos)-1)
sel_month = st.sidebar.selectbox("Mês", list(MONTH_MAP_NUM_TO_NAME.values()), index=datetime.datetime.now().month-1)
rep_selected = st.sidebar.selectbox("Representante", rep_options)

start_comp = pd.Timestamp(year=sel_year, month=MONTH_MAP_NAME_TO_NUM[sel_month], day=1)
end_comp = start_comp # Para este exemplo, mês fixo. Ajuste conforme sua necessidade de range.
df_rep = df[df["Competencia"] == start_comp]
if rep_selected != "Todos": df_rep = df_rep[df_rep["Representante"] == rep_selected]

# --- INITIALIZATION (FIXES NameError) ---
total_rep = 0.0; total_vol_rep = 0.0; media_mensal = 0.0; num_clientes_rep = 0
n80_count = 0; hhi_value = 0.0; hhi_label_short = "Sem dados"; top1_share = 0.0; top3_share = 0.0; top10_share = 0.0
clientes_atendidos = 0; cidades_atendidas = 0; estados_atendidos = 0; carteira_score = 50.0; carteira_label = "Neutra"
highlights_lines = []; fig_cat = None; cat_disp = pd.DataFrame(); fig_states = None; estados_display = pd.DataFrame()
fig_clients = None; df_tbl_clients = pd.DataFrame(); status_disp = pd.DataFrame(); chart_pie = None
chart_curr = None; chart_prev = None; clientes_carteira = pd.DataFrame()

# --- CALCULATIONS ---
if not df_rep.empty:
    total_rep = df_rep["Valor"].sum()
    total_vol_rep = df_rep["Quantidade"].sum()
    clientes_atendidos = df_rep["Cliente"].nunique()
    cidades_atendidas = df_rep["Cidade"].nunique()
    estados_atendidos = df_rep["Estado"].nunique()
    
    df_cli = df_rep.groupby("Cliente")["Valor"].sum().sort_values(ascending=False)
    if total_rep > 0:
        shares = df_cli / total_rep
        n80_count = (shares.cumsum() <= 0.8).sum() + 1
        hhi_value = (shares**2).sum()
        hhi_label_short = "Baixa" if hhi_value < 0.1 else "Moderada" if hhi_value < 0.2 else "Alta"
    
    highlights_lines = [f"Total Faturado: {format_brl(total_rep)}", f"Volume: {format_un(total_vol_rep)}"]
    clientes_carteira = build_carteira_status(df, rep_selected, start_comp, end_comp)
    carteira_score, carteira_label = compute_carteira_score(clientes_carteira)

# ==========================
# DASHBOARD UI
# ==========================
st.title(f"Insights: {rep_selected}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total", format_brl_compact(total_rep))
c2.metric("Clientes", f"{clientes_atendidos}")
c3.metric("N80", f"{n80_count}")
c4.metric("HHI", hhi_label_short)
c5.metric("Saúde", f"{carteira_score:.0f}", carteira_label)

st.markdown("---")

# Evolução (Placeholder logic)
st.subheader("Evolução")
ts = df_rep.groupby("Competencia").agg({"Valor":"sum", "Quantidade":"sum"}).reset_index()
if not ts.empty:
    chart_curr = alt.Chart(ts).mark_bar().encode(x="Competencia:T", y="Valor:Q").properties(height=300)
    st_altair_stretch(chart_curr)

# Distribuição por Clientes
st.subheader("Distribuição por clientes")
if df_rep.empty or clientes_atendidos == 0:
    st.info("Nenhum cliente com vendas no período selecionado.")
else:
    df_tbl_clients = df_rep.groupby("Cliente").agg({"Valor":"sum", "Cidade":"first"}).reset_index().sort_values("Valor", ascending=False).head(15)
    st.table(df_tbl_clients)

# ==========================
# PDF TRIGGER
# ==========================
if st.button("📄 Gerar PDF"):
    chart_items = []
    png_ev = _chart_to_png(chart_curr)
    if png_ev: chart_items.append(("Evolução de Vendas", png_ev))
    
    kpis_pdf = {"Faturamento": format_brl_compact(total_rep), "Clientes": str(clientes_atendidos), "Score": f"{carteira_score:.0f}"}
    tables = [("Principais Clientes", df_tbl_clients, 15)]
    
    pdf_bytes = build_pdf_report(rep_selected, sel_month, "", kpis_pdf, highlights_lines, chart_items, tables)
    st.download_button("⬇️ Baixar Relatório", pdf_bytes, f"Relatorio_{rep_selected}.pdf", "application/pdf")
