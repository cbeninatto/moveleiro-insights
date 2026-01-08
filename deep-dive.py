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
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, KeepInFrame
)

# Tentativa de importação para conversão de gráficos
try:
    import vl_convert as vlc
except ImportError:
    vlc = None

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA (Sempre no topo)
# ==========================================
st.set_page_config(
    page_title="Insights de Vendas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# 2. CONSTANTES E ESTILOS
# ==========================================
GITHUB_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"
CITY_GEO_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/main/data/cidades_br_geo.csv"
STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
STATUS_COL = "StatusCarteira"
MAP_BIN_COLORS = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

_PDF_STYLES_BASE = getSampleStyleSheet()
_PDF_STYLES = {
    "H1": ParagraphStyle("H1", parent=_PDF_STYLES_BASE["Heading1"], fontSize=14, leading=16, spaceAfter=4),
    "H2": ParagraphStyle("H2", parent=_PDF_STYLES_BASE["Heading2"], fontSize=11, leading=13, spaceBefore=6, spaceAfter=4),
    "Body": ParagraphStyle("Body", parent=_PDF_STYLES_BASE["BodyText"], fontSize=9.2, leading=11.2),
    "Small": ParagraphStyle("Small", parent=_PDF_STYLES_BASE["BodyText"], fontSize=8.4, leading=10.2),
    "Caption": ParagraphStyle("Caption", parent=_PDF_STYLES_BASE["BodyText"], fontSize=8.0, leading=9.5, textColor=colors.grey),
}

# ==========================================
# 3. HELPERS DE CONVERSÃO E PDF
# ==========================================
def _chart_to_png(obj, width_px=1600, scale=2.0):
    if obj is None: return None
    try:
        # Plotly
        if hasattr(obj, "to_image"):
            return obj.to_image(format="png", width=width_px, scale=int(scale))
        # Altair
        if hasattr(obj, "to_dict") and vlc:
            spec = obj.to_dict()
            spec.setdefault("width", width_px)
            return vlc.vegalite_to_png(spec, scale=scale)
    except Exception:
        return None
    return None

def _rl_img(png_bytes, w_cm, h_cm):
    if not png_bytes:
        return Paragraph("<i>(gráfico indisponível)</i>", _PDF_STYLES["Caption"])
    return RLImage(io.BytesIO(png_bytes), width=w_cm * cm, height=h_cm * cm)

# 

# (Aqui você manteria as funções de lógica de negócio: format_brl, load_data, build_carteira_status, etc.)
# ... (Funções de formatação e carregamento omitidas por brevidade, mas devem permanecer no seu arquivo)

# ==========================================
# 4. LÓGICA DE PDF (Layout Dashboard)
# ==========================================
def build_pdf_report_dashboard_layout(
    rep_name, current_period_label, previous_period_label, kpis, 
    highlights_lines, fig_states, estados_table, chart_curr, chart_prev, 
    fig_cat, cat_table, fig_clients, clients_table, carteira_chart, 
    carteira_table, df_rep_for_map, status_tables_by_group
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), margins=(1*cm, 1*cm, 0.8*cm, 0.8*cm))
    story = []
    
    # Renderização de imagens (Centralizada para evitar repetição)
    pngs = {
        "states": _chart_to_png(fig_states),
        "evol_curr": _chart_to_png(chart_curr),
        "evol_prev": _chart_to_png(chart_prev),
        "cat": _chart_to_png(fig_cat),
        "clients": _chart_to_png(fig_clients),
        "carteira": _chart_to_png(carteira_chart)
    }
    
    # --- Montagem do PDF conforme seu layout de grid ---
    # (O conteúdo da sua função build_pdf_report_dashboard_layout entra aqui)
    
    doc.build(story)
    return buf.getvalue()

# ==========================================
# 5. EXECUÇÃO DO DASHBOARD (Lógica Principal)
# ==========================================

# Carregamento de dados inicial
df = load_data() 

# Sidebar e Filtros
# ... (Seu código de filtros aqui)

# --- Cabeçalho e Trigger de PDF ---
st.title("Insights de Vendas")
titulo_rep = "Todos" if rep_selected == "Todos" else rep_selected

h_left, h_right = st.columns([0.78, 0.22], vertical_alignment="center")
with h_left:
    st.subheader(f"Representante: **{titulo_rep}**")
with h_right:
    # Capturamos o clique do botão em uma variável
    btn_gerar = st.button("📄 Gerar PDF", use_container_width=True, key="pdf_gen_btn")

# --- Renderização dos Gráficos no Streamlit ---
# (O código que gera fig_states, chart_curr, fig_cat, etc.)

# --- Processamento do PDF (Após a criação dos objetos de gráfico) ---
if btn_gerar:
    with st.spinner("Gerando relatório formatado..."):
        try:
            # Preparação de Tabelas e KPIs para o PDF
            kpis_pdf = {
                "Total período": format_brl_compact(total_rep),
                "Média mensal": format_brl_compact(media_mensal),
                "Clientes atendidos": str(clientes_atendidos),
                "Saúde da carteira": f"{carteira_score:.0f}/100"
            }
            
            # Chamada da função unificada
            pdf_bytes = build_pdf_report_dashboard_layout(
                rep_name=titulo_rep,
                current_period_label=current_period_label,
                # ... outros argumentos ...
            )
            
            st.session_state["pdf_report_bytes"] = pdf_bytes
            st.session_state["pdf_report_name"] = f"Relatorio_{titulo_rep}.pdf"
        except Exception as e:
            st.error(f"Erro na geração: {e}")

# Exibição do botão de download se o PDF existir
if st.session_state.get("pdf_report_bytes"):
    st.download_button(
        label="⬇️ Baixar Relatório PDF",
        data=st.session_state["pdf_report_bytes"],
        file_name=st.session_state["pdf_report_name"],
        mime="application/pdf",
        use_container_width=True
    )
