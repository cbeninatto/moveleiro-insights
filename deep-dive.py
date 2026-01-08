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

# PDF Libraries
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, KeepInFrame
)

# Optional PNG conversion for PDF
try:
    import vl_convert as vlc
except Exception:
    vlc = None

# ==========================
# CONFIG & CONSTANTS
# ==========================
st.set_page_config(page_title="Insights de Vendas", layout="wide", initial_sidebar_state="collapsed")

GITHUB_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"
CITY_GEO_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/main/data/cidades_br_geo.csv"
STATUS_COL = "StatusCarteira"
STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
STATUS_WEIGHTS = {"Novos": 1, "Crescendo": 2, "Estáveis": 1, "Caindo": -1, "Perdidos": -2}
MAP_BIN_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

# ==========================
# HELPERS & BUSINESS LOGIC
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

def format_un(value: float) -> str:
    if pd.isna(value): return "0 un"
    return f"{int(round(float(value))):,}".replace(",", ".") + " un"

def format_brl_signed(value: float) -> str:
    v = float(value or 0)
    return ("-" if v < 0 else "") + format_brl(abs(v))

@st.cache_data(ttl=3600)
def load_data():
    resp = requests.get(f"{GITHUB_CSV_URL}?cb={int(time.time())}")
    df = pd.read_csv(io.StringIO(resp.text))
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    df["Competencia"] = pd.to_datetime(dict(year=df["Ano"], month=df["Mes"], day=1))
    return df

@st.cache_data(ttl=3600)
def load_geo():
    resp = requests.get(f"{CITY_GEO_CSV_URL}?cb={int(time.time())}")
    df_geo = pd.read_csv(io.StringIO(resp.text))
    df_geo["key"] = df_geo["Estado"].str.upper() + "|" + df_geo["Cidade"].str.upper()
    return df_geo

# ==========================
# PDF ENGINE
# ==========================
_PDF_STYLES_BASE = getSampleStyleSheet()
_PDF_STYLES = {
    "H1": ParagraphStyle("H1", parent=_PDF_STYLES_BASE["Heading1"], fontSize=14, leading=16, spaceAfter=4),
    "H2": ParagraphStyle("H2", parent=_PDF_STYLES_BASE["Heading2"], fontSize=11, leading=13, spaceBefore=6, spaceAfter=4),
    "Body": ParagraphStyle("Body", parent=_PDF_STYLES_BASE["BodyText"], fontSize=9.2, leading=11.2),
    "Small": ParagraphStyle("Small", parent=_PDF_STYLES_BASE["BodyText"], fontSize=8.4, leading=10.2),
    "Caption": ParagraphStyle("Caption", parent=_PDF_STYLES_BASE["BodyText"], fontSize=8.0, leading=9.5, textColor=colors.grey),
}

def _chart_to_png(obj, width_px=1600, scale=2.0):
    if obj is None: return None
    try:
        if hasattr(obj, "to_image"): return obj.to_image(format="png", width=width_px, scale=int(scale))
        if hasattr(obj, "to_dict") and vlc: return vlc.vegalite_to_png(obj.to_dict(), scale=scale)
    except: return None

def _kpi_card(title, value):
    p_title = Paragraph(f"<b>{html.escape(str(title))}</b>", _PDF_STYLES["Small"])
    p_value = Paragraph(f"<b>{html.escape(str(value))}</b>", _PDF_STYLES["H1"])
    tbl = Table([[p_title], [p_value]], colWidths=[5.2 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0b1220")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#111827")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl

def _df_to_rl_table(df, max_rows=15):
    if df.empty: return Paragraph("Sem dados", _PDF_STYLES["Caption"])
    df_sub = df.head(max_rows).astype(str)
    data = [df_sub.columns.tolist()] + df_sub.values.tolist()
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 7),
    ]))
    return t

def build_pdf_report_dashboard_layout(rep_name, current_label, kpis, highlights, 
                                      fig_states, chart_curr, fig_cat, fig_clients, chart_pie):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), margins=(1*cm, 1*cm, 1*cm, 1*cm))
    story = []
    
    story.append(Paragraph(f"Insights: {rep_name}", _PDF_STYLES["H1"]))
    story.append(Paragraph(f"Período: {current_label}", _PDF_STYLES["Caption"]))
    story.append(Spacer(1, 10))
    
    # KPI Row
    cards = [_kpi_card(k, v) for k, v in list(kpis.items())[:5]]
    story.append(Table([cards], colWidths=[5.4 * cm]*5))
    story.append(Spacer(1, 10))
    
    # Highlights
    story.append(Paragraph("Destaques", _PDF_STYLES["H2"]))
    for h in highlights: story.append(Paragraph(f"• {h}", _PDF_STYLES["Body"]))
    
    # Charts (Images)
    imgs = []
    for f in [chart_curr, fig_cat, fig_states, fig_clients]:
        png = _chart_to_png(f)
        if png: imgs.append(RLImage(io.BytesIO(png), width=12*cm, height=7*cm))
    
    if imgs:
        grid = [imgs[i:i+2] for i in range(0, len(imgs), 2)]
        story.append(Table(grid))
        
    doc.build(story)
    return buf.getvalue()

# ==========================
# MAIN APP LOGIC
# ==========================
df = load_data()
MONTH_MAP = {1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR", 5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO", 9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ"}

# Sidebar
st.sidebar.header("Filtros")
anos = sorted(df["Ano"].unique())
start_y = st.sidebar.selectbox("Ano Inicial", anos, index=len(anos)-1)
end_y = st.sidebar.selectbox("Ano Final", anos, index=len(anos)-1)
rep_selected = st.sidebar.selectbox("Representante", ["Todos"] + sorted(df["Representante"].unique()))

# Filtering
start_comp = pd.Timestamp(year=start_y, month=1, day=1)
end_comp = pd.Timestamp(year=end_y, month=12, day=1)
current_period_label = f"{start_y} - {end_y}"

df_rep = df[(df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)].copy()
if rep_selected != "Todos":
    df_rep = df_rep[df_rep["Representante"] == rep_selected]

# ==========================
# KPI CALCULATIONS (FIXED: Defined before PDF/UI)
# ==========================
if not df_rep.empty:
    total_rep = float(df_rep["Valor"].sum())
    total_vol = float(df_rep["Quantidade"].sum())
    clientes_atendidos = int(df_rep["Cliente"].nunique())
    cidades_atendidas = int(df_rep["Cidade"].nunique())
    estados_atendidos = int(df_rep["Estado"].nunique())
    media_mensal = total_rep / 12 # Simplificado
    
    # N80 / HHI
    df_c = df_rep.groupby("Cliente")["Valor"].sum().sort_values(ascending=False)
    shares = df_c / total_rep
    n80_count = (shares.cumsum() <= 0.8).sum() + 1
    hhi_label_short = "Alta" if (shares**2).sum() > 0.2 else "Baixa"
    
    highlights_lines = [
        f"Faturamento Total: {format_brl_compact(total_rep)}",
        f"Volume Total: {format_un(total_vol)}",
        f"Cobertura: {cidades_atendidas} cidades em {estados_atendidos} estados."
    ]
else:
    total_rep = media_mensal = n80_count = clientes_atendidos = cidades_atendidas = estados_atendidos = 0
    hhi_label_short = "N/A"
    highlights_lines = ["Sem dados no período."]

carteira_score, carteira_label = 75, "Saudável" # Mockup para exemplo

# ==========================
# UI & CHART GENERATION
# ==========================
st.title("Insights de Vendas")
col_btn_l, col_btn_r = st.columns([0.8, 0.2])
with col_btn_l: st.subheader(f"Representante: {rep_selected}")
with col_btn_r: gerar_pdf = st.button("📄 Gerar PDF", use_container_width=True)

# Metric row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total", format_brl_compact(total_rep))
m2.metric("Média", format_brl_compact(media_mensal))
m3.metric("Clientes", clientes_atendidos)
m4.metric("N80", n80_count)
m5.metric("Saúde", f"{carteira_score}/100")

# Charts (Define objects for both UI and PDF)
fig_cat = px.pie(df_rep, values="Valor", names="Categoria", title="Categorias") if not df_rep.empty else None
fig_states = px.pie(df_rep, values="Valor", names="Estado", title="Estados") if not df_rep.empty else None
fig_clients = px.bar(df_rep.groupby("Cliente")["Valor"].sum().nlargest(10).reset_index(), x="Cliente", y="Valor") if not df_rep.empty else None

# Evolução Chart
chart_curr = alt.Chart(df_rep).mark_line().encode(x='month(Competencia):T', y='sum(Valor):Q')

# Display UI
st.altair_chart(chart_curr, use_container_width=True)
c_left, c_right = st.columns(2)
with c_left: st.plotly_chart(fig_cat, use_container_width=True) if fig_cat else st.write("Sem dados")
with c_right: st.plotly_chart(fig_states, use_container_width=True) if fig_states else st.write("Sem dados")

# ==========================
# PDF TRIGGER (After KPIs/Charts are defined)
# ==========================
if gerar_pdf:
    kpis_pdf = {
        "Total": format_brl_compact(total_rep),
        "Média": format_brl_compact(media_mensal),
        "Clientes": str(clientes_atendidos),
        "N80": str(n80_count),
        "Saúde": carteira_label
    }
    
    pdf_bytes = build_pdf_report_dashboard_layout(
        rep_selected, current_period_label, kpis_pdf, highlights_lines,
        fig_states, chart_curr, fig_cat, fig_clients, None
    )
    
    st.download_button("⬇️ Clique para Baixar PDF", data=pdf_bytes, 
                       file_name=f"Relatorio_{rep_selected}.pdf", mime="application/pdf")
