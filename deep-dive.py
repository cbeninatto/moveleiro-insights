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
# PDF EXPORT (REWRITTEN to match your “printed dashboard” layout)
# Replace your existing PDF helpers + build_pdf_report() + PDF generation block with THIS section.
# ==========================
import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, KeepInFrame
)

try:
    import vl_convert as vlc  # vl-convert-python (Altair/Vega-Lite -> PNG)
except Exception:
    vlc = None


# ---------- Chart -> PNG bytes ----------
def _plotly_to_png_bytes(fig, width_px: int = 1600, scale: int = 2):
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", width=width_px, scale=scale)  # needs kaleido
    except Exception:
        return None

def _altair_to_png_bytes(chart, width_px: int = 1600, scale: float = 2.0):
    if chart is None or vlc is None:
        return None
    try:
        spec = chart.to_dict()
        spec.setdefault("width", width_px)
        return vlc.vegalite_to_png(spec, scale=scale)
    except Exception:
        return None

def _chart_to_png(obj, width_px: int = 1600, scale: float = 2.0):
    if obj is None:
        return None
    if hasattr(obj, "to_image"):
        return _plotly_to_png_bytes(obj, width_px=width_px, scale=int(scale))
    if hasattr(obj, "to_dict"):
        return _altair_to_png_bytes(obj, width_px=width_px, scale=scale)
    return None

def _rl_img(png_bytes: bytes, w_cm: float, h_cm: float):
    if not png_bytes:
        return Paragraph("<i>(gráfico indisponível)</i>", _PDF_STYLES["Caption"])
    img = RLImage(io.BytesIO(png_bytes), width=w_cm * cm, height=h_cm * cm)
    return img


# ---------- PDF styles ----------
_PDF_STYLES_BASE = getSampleStyleSheet()
_PDF_STYLES = {
    "H1": ParagraphStyle("H1", parent=_PDF_STYLES_BASE["Heading1"], fontSize=14, leading=16, spaceAfter=4),
    "H2": ParagraphStyle("H2", parent=_PDF_STYLES_BASE["Heading2"], fontSize=11, leading=13, spaceBefore=6, spaceAfter=4),
    "Body": ParagraphStyle("Body", parent=_PDF_STYLES_BASE["BodyText"], fontSize=9.2, leading=11.2),
    "Small": ParagraphStyle("Small", parent=_PDF_STYLES_BASE["BodyText"], fontSize=8.4, leading=10.2),
    "Caption": ParagraphStyle("Caption", parent=_PDF_STYLES_BASE["BodyText"], fontSize=8.0, leading=9.5, textColor=colors.grey),
}


# ---------- Small layout helpers ----------
def _kpi_card(title: str, value: str, sub: str = ""):
    title_p = Paragraph(f"<b>{html.escape(str(title))}</b>", _PDF_STYLES["Small"])
    value_p = Paragraph(f"<b>{html.escape(str(value))}</b>", ParagraphStyle("KPIValue", parent=_PDF_STYLES["Body"], fontSize=12, leading=13))
    sub_p = Paragraph(html.escape(str(sub)) if sub else "&nbsp;", _PDF_STYLES["Caption"])

    tbl = Table([[title_p], [value_p], [sub_p]], colWidths=[(5.2 * cm)])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0b1220")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#111827")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
    ]))
    return tbl

def _section_box(title: str, inner_flowables):
    head = Table([[Paragraph(f"<b>{html.escape(title)}</b>", _PDF_STYLES["Small"])]], colWidths=[26.3 * cm])
    head.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    frame = KeepInFrame(26.3 * cm, 1000, inner_flowables, mergeSpace=True)
    body = Table([[frame]], colWidths=[26.3 * cm])
    body.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return [head, body]

def _df_to_rl_table(df_in: pd.DataFrame, max_rows: int, col_widths_cm: list[float] | None = None):
    if df_in is None or df_in.empty:
        return Paragraph("<i>(sem dados)</i>", _PDF_STYLES["Caption"])

    df = df_in.head(max_rows).copy()
    data = [list(df.columns)] + df.astype(str).values.tolist()

    ncol = len(df.columns)
    if col_widths_cm and len(col_widths_cm) == ncol:
        colWidths = [w * cm for w in col_widths_cm]
    else:
        # fit into ~13 cm if used as right-side table; adjust by caller
        colWidths = None

    tbl = Table(data, colWidths=colWidths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.2),
        ("FONTSIZE", (0, 1), (-1, -1), 7.9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl

def _build_pdf_static_map_fig(df_rep_in: pd.DataFrame):
    """
    Folium doesn't export nicely. For the PDF, we build a static Plotly geo scatter.
    Uses your city geo file (load_geo()) and the same city aggregation.
    """
    if df_rep_in is None or df_rep_in.empty:
        return None
    try:
        df_geo_local = load_geo()
        df_cities = df_rep_in.groupby(["Estado", "Cidade"], as_index=False).agg(
            Valor=("Valor", "sum"),
            Quantidade=("Quantidade", "sum"),
            Clientes=("Cliente", "nunique"),
        )
        df_cities["key"] = (
            df_cities["Estado"].astype(str).str.strip().str.upper()
            + "|"
            + df_cities["Cidade"].astype(str).str.strip().str.upper()
        )
        df_map = df_cities.merge(df_geo_local, on="key", how="inner")
        if df_map.empty:
            return None

        # nicer hover text
        df_map["Hover"] = df_map.apply(
            lambda r: f"{r['Cidade']} - {r['Estado']}<br>"
                      f"Faturamento: {format_brl(float(r['Valor']))}<br>"
                      f"Clientes: {int(r['Clientes'])}",
            axis=1
        )

        # Plotly geo scatter (no tokens needed)
        fig = px.scatter_geo(
            df_map,
            lat="lat", lon="lon",
            size="Valor",
            size_max=18,
            hover_name="Cidade",
            hover_data={"Hover": True, "lat": False, "lon": False},
            projection="natural earth",
        )
        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=520,
            showlegend=False,
            geo=dict(
                scope="south america",
                showland=True,
                landcolor="rgb(245,245,245)",
                showcountries=True,
                countrycolor="rgb(220,220,220)",
            ),
        )
        return fig
    except Exception:
        return None


# ---------- Main PDF builder (matches your reference: pages with fixed grid) ----------
def build_pdf_report_dashboard_layout(
    rep_name: str,
    current_period_label: str,
    previous_period_label: str,
    kpis: dict,
    highlights_lines: list[str],
    # figures/charts (already created in the app)
    fig_states,
    estados_table: pd.DataFrame,
    chart_curr,
    chart_prev,
    fig_cat,
    cat_table: pd.DataFrame,
    fig_clients,
    clients_table: pd.DataFrame,
    carteira_chart,
    carteira_table: pd.DataFrame,
    df_rep_for_map: pd.DataFrame,
    status_tables_by_group: list[tuple[str, pd.DataFrame]],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
        title="Insights de Vendas",
    )

    story = []
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    # Pre-render PNGs
    png_states = _chart_to_png(fig_states, width_px=1600, scale=2.0)
    png_evol_curr = _chart_to_png(chart_curr, width_px=1600, scale=2.0)
    png_evol_prev = _chart_to_png(chart_prev, width_px=1600, scale=2.0)
    png_cat = _chart_to_png(fig_cat, width_px=1600, scale=2.0)
    png_clients = _chart_to_png(fig_clients, width_px=1600, scale=2.0)
    png_carteira = _chart_to_png(carteira_chart, width_px=1600, scale=2.0)

    fig_map = _build_pdf_static_map_fig(df_rep_for_map)
    png_map = _chart_to_png(fig_map, width_px=1600, scale=2.0)

    # ----------------------
    # PAGE 1 — Summary
    # ----------------------
    story.append(Paragraph("Insights de Vendas", _PDF_STYLES["H1"]))
    story.append(Paragraph(f"Representante: <b>{html.escape(str(rep_name))}</b>", _PDF_STYLES["Body"]))
    story.append(Paragraph(
        f"Período: <b>{html.escape(current_period_label)}</b> &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"Anterior: <b>{html.escape(previous_period_label)}</b> &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"Gerado em: {now}",
        _PDF_STYLES["Caption"],
    ))
    story.append(Spacer(1, 6))

    # KPI row (5 cards)
    kpi_items = list(kpis.items())
    # Ensure exactly 5 visible cards (pad if needed)
    while len(kpi_items) < 5:
        kpi_items.append(("", "",))
    cards = []
    for i in range(5):
        title = kpi_items[i][0] if i < len(kpi_items) else ""
        value = kpi_items[i][1] if i < len(kpi_items) else ""
        cards.append(_kpi_card(title, value))
    kpi_row = Table([cards], colWidths=[5.2 * cm] * 5)
    kpi_row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                 ("TOPPADDING", (0, 0), (-1, -1), 0),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    story.append(kpi_row)
    story.append(Spacer(1, 6))

    # Destaques (2 columns)
    left_lines = [l for l in highlights_lines if "Faturamento" in l]
    right_lines = [l for l in highlights_lines if "Volume" in l]
    if not left_lines and highlights_lines:
        left_lines = highlights_lines[:2]
        right_lines = highlights_lines[2:]

    highlights_left = Paragraph("<br/>".join([f"• {html.escape(x)}" for x in left_lines]) or "—", _PDF_STYLES["Body"])
    highlights_right = Paragraph("<br/>".join([f"• {html.escape(x)}" for x in right_lines]) or "—", _PDF_STYLES["Body"])
    hl_tbl = Table([[highlights_left, highlights_right]], colWidths=[13.15 * cm, 13.15 * cm])
    hl_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend(_section_box("Destaques do período", [hl_tbl]))
    story.append(Spacer(1, 6))

    # Evolução charts stacked (full width, two rows)
    evol_tbl = Table(
        [
            [_rl_img(png_evol_curr, w_cm=26.3, h_cm=6.0)],
            [_rl_img(png_evol_prev, w_cm=26.3, h_cm=6.0)],
        ],
        colWidths=[26.3 * cm],
    )
    evol_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend(_section_box("Evolução – Faturamento x Volume", [evol_tbl]))
    story.append(Spacer(1, 6))

    # Categorias: donut left + table right
    cat_table_rl = _df_to_rl_table(cat_table, max_rows=14)
    cat_row = Table(
        [[_rl_img(png_cat, w_cm=12.9, h_cm=7.0), cat_table_rl]],
        colWidths=[12.9 * cm, 13.4 * cm],
    )
    cat_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend(_section_box("Categorias vendidas", [cat_row]))

    story.append(PageBreak())

    # ----------------------
    # PAGE 2 — Map + Estados
    # ----------------------
    # Map block: map left, coverage+top clients right (we reuse clients_table as a decent “principais clientes”)
    right_side = []
    # Small “Cobertura” line (compact)
    cobertura_txt = Paragraph(
        f"<b>Cobertura</b><br/>"
        f"• Cidades: {html.escape(str(kpis.get('Cidades atendidas', '—')))}<br/>"
        f"• Estados: {html.escape(str(kpis.get('Estados atendidos', '—')))}<br/>"
        f"• Clientes: {html.escape(str(kpis.get('Clientes atendidos', '—')))}",
        _PDF_STYLES["Body"],
    )
    right_side.append(cobertura_txt)
    right_side.append(Spacer(1, 6))
    right_side.append(Paragraph("<b>Principais clientes</b>", _PDF_STYLES["Small"]))
    right_side.append(Spacer(1, 4))
    right_side.append(_df_to_rl_table(clients_table, max_rows=12))

    map_row = Table(
        [[_rl_img(png_map, w_cm=16.9, h_cm=12.0), KeepInFrame(9.4 * cm, 12.0 * cm, right_side, mergeSpace=True)]],
        colWidths=[16.9 * cm, 9.4 * cm],
    )
    map_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend(_section_box("Mapa de Clientes", [map_row]))
    story.append(Spacer(1, 8))

    # Estados: donut left + table right
    estados_table_rl = _df_to_rl_table(estados_table, max_rows=10)
    estados_row = Table(
        [[_rl_img(png_states, w_cm=12.9, h_cm=7.2), estados_table_rl]],
        colWidths=[12.9 * cm, 13.4 * cm],
    )
    estados_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend(_section_box("Distribuição por estados", [estados_row]))

    story.append(PageBreak())

    # ----------------------
    # PAGE 3 — Clientes + Carteira
    # ----------------------
    clients_row = Table(
        [[_rl_img(png_clients, w_cm=12.9, h_cm=7.2), _df_to_rl_table(clients_table, max_rows=15)]],
        colWidths=[12.9 * cm, 13.4 * cm],
    )
    clients_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend(_section_box("Distribuição por clientes", [clients_row]))
    story.append(Spacer(1, 8))

    carteira_row = Table(
        [[_rl_img(png_carteira, w_cm=12.9, h_cm=7.2), _df_to_rl_table(carteira_table, max_rows=12)]],
        colWidths=[12.9 * cm, 13.4 * cm],
    )
    carteira_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend(_section_box("Saúde da carteira – Detalhes", [carteira_row]))

    # ----------------------
    # Pages 4+ — Status tables
    # ----------------------
    story.append(PageBreak())
    story.append(Paragraph("Status dos clientes", _PDF_STYLES["H2"]))
    story.append(Spacer(1, 6))

    for status_name, df_status in status_tables_by_group:
        if df_status is None or df_status.empty:
            continue
        story.append(Paragraph(html.escape(status_name), _PDF_STYLES["H2"]))
        story.append(Spacer(1, 4))
        story.append(_df_to_rl_table(df_status, max_rows=60))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buf.getvalue()


# ==========================
# PDF GENERATION TRIGGER (keep at the end of the script)
# Replace your existing "PDF GENERATION" block with this one.
# ==========================
if "pdf_report_bytes" not in st.session_state:
    st.session_state["pdf_report_bytes"] = None
    st.session_state["pdf_report_name"] = None
    st.session_state["pdf_report_error"] = None

# IMPORTANT: keep your header button with key="pdf_gen_btn"
# gerar_pdf = st.button("📄 Gerar PDF", use_container_width=True, key="pdf_gen_btn")

if gerar_pdf:
    st.session_state["pdf_report_error"] = None
    st.session_state["pdf_report_bytes"] = None
    st.session_state["pdf_report_name"] = None

    # KPI cards (pick the same 5 you show in the UI)
    kpis_pdf = {
        "Total período": format_brl_compact(total_rep),
        "Média mensal": format_brl_compact(media_mensal),
        "Distribuição por clientes": f"{hhi_label_short} (N80: {n80_count})",
        "Saúde da carteira": f"{carteira_score:.0f} / 100 ({carteira_label})",
        "Clientes atendidos": str(clientes_atendidos),
        # (We also pass these through for “Cobertura” sidebar in the PDF)
        "Cidades atendidas": str(cidades_atendidas),
        "Estados atendidos": str(estados_atendidos),
    }

    # Carteira table (use what you already render on screen)
    carteira_table_pdf = status_disp.copy() if isinstance(status_disp, pd.DataFrame) else pd.DataFrame()

    # Status pages tables (build from clientes_carteira like you do in UI)
    status_tables_by_group = []
    if isinstance(clientes_carteira, pd.DataFrame) and not clientes_carteira.empty:
        for status_name in STATUS_ORDER:
            df_status = clientes_carteira[clientes_carteira[STATUS_COL] == status_name].copy()
            if df_status.empty:
                continue
            df_status["FaturamentoAtualFmt"] = df_status["ValorAtual"].map(format_brl)
            df_status["FaturamentoAnteriorFmt"] = df_status["ValorAnterior"].map(format_brl)
            df_status = df_status.sort_values("ValorAtual", ascending=False)

            df_out = df_status[["Cliente", "Estado", "Cidade", "FaturamentoAtualFmt", "FaturamentoAnteriorFmt"]].rename(
                columns={
                    "FaturamentoAtualFmt": f"Faturamento {current_period_label}",
                    "FaturamentoAnteriorFmt": f"Faturamento {previous_period_label}",
                }
            )
            status_tables_by_group.append((status_name, df_out))

    try:
        pdf_bytes = build_pdf_report_dashboard_layout(
            rep_name=titulo_rep,
            current_period_label=current_period_label,
            previous_period_label=previous_period_label,
            kpis=kpis_pdf,
            highlights_lines=highlights_lines,

            fig_states=fig_states,
            estados_table=estados_display if isinstance(estados_display, pd.DataFrame) else pd.DataFrame(),

            chart_curr=chart_curr,
            chart_prev=chart_prev,

            fig_cat=fig_cat,
            cat_table=cat_disp if isinstance(cat_disp, pd.DataFrame) else pd.DataFrame(),

            fig_clients=fig_clients,
            clients_table=df_tbl_clients if isinstance(df_tbl_clients, pd.DataFrame) else pd.DataFrame(),

            carteira_chart=chart_pie,
            carteira_table=carteira_table_pdf,

            df_rep_for_map=df_rep,
            status_tables_by_group=status_tables_by_group,
        )

        safe_rep = re.sub(r"[^A-Za-z0-9_-]+", "_", str(titulo_rep))[:40]
        st.session_state["pdf_report_name"] = f"relatorio_insights_{safe_rep}_{start_comp.strftime('%Y%m')}-{end_comp.strftime('%Y%m')}.pdf"
        st.session_state["pdf_report_bytes"] = pdf_bytes

    except Exception as ex:
        st.session_state["pdf_report_error"] = str(ex)

if st.session_state.get("pdf_report_error"):
    st.error(f"Não foi possível gerar o PDF: {st.session_state['pdf_report_error']}")

if st.session_state.get("pdf_report_bytes") and st.session_state.get("pdf_report_name"):
    st.download_button(
        "⬇️ Baixar PDF",
        data=st.session_state["pdf_report_bytes"],
        file_name=st.session_state["pdf_report_name"],
        mime="application/pdf",
        key="pdf_download_btn",
    )


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
# PDF helpers (chart -> PNG)
# ==========================
def _plotly_to_png_bytes(fig, width_px: int = 1600, scale: int = 2):
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", width=width_px, scale=scale)
    except Exception:
        return None

def _altair_to_png_bytes(chart, width_px: int = 1600, scale: float = 2.0):
    if chart is None or vlc is None:
        return None
    try:
        spec = chart.to_dict()
        spec.setdefault("width", width_px)
        return vlc.vegalite_to_png(spec, scale=scale)
    except Exception:
        return None

def _chart_to_png(obj, width_px: int = 1600, scale: float = 2.0):
    if obj is None:
        return None
    if hasattr(obj, "to_image"):
        return _plotly_to_png_bytes(obj, width_px=width_px, scale=int(scale))
    if hasattr(obj, "to_dict"):
        return _altair_to_png_bytes(obj, width_px=width_px, scale=scale)
    return None

def build_pdf_report(
    rep_name: str,
    current_period_label: str,
    previous_period_label: str,
    kpis: dict,
    highlights_lines: list[str],
    chart_items: list[tuple[str, bytes]],
    tables: list[tuple[str, pd.DataFrame, int]]  # (title, df, max_rows)
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
        title="Insights de Vendas",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, leading=18, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, leading=14, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyS", parent=styles["BodyText"], fontSize=9.5, leading=12))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5, leading=10, textColor=colors.grey))

    story = []
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    story.append(Paragraph("Insights de Vendas — Relatório", styles["H1"]))
    story.append(Paragraph(f"Representante: <b>{html.escape(str(rep_name))}</b>", styles["BodyS"]))
    story.append(Paragraph(
        f"Período selecionado: <b>{html.escape(str(current_period_label))}</b> &nbsp;&nbsp;•&nbsp;&nbsp; "
        f"Período anterior: <b>{html.escape(str(previous_period_label))}</b>",
        styles["BodyS"],
    ))
    story.append(Paragraph(f"Gerado em: {now}", styles["Caption"]))
    story.append(Spacer(1, 10))

    # KPIs
    story.append(Paragraph("Resumo (KPIs)", styles["H2"]))
    rows = [["Métrica", "Valor"]] + [[k, v] for k, v in kpis.items()]
    t = Table(rows, colWidths=[8.0 * cm, 18.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Highlights
    story.append(Paragraph("Destaques do período", styles["H2"]))
    if highlights_lines:
        story.append(Paragraph("<br/>".join([f"• {html.escape(str(x))}" for x in highlights_lines]), styles["BodyS"]))
    else:
        story.append(Paragraph("Sem destaques.", styles["BodyS"]))
    story.append(Spacer(1, 10))

    # Charts — 2 per page
    if chart_items:
        story.append(Paragraph("Gráficos", styles["H2"]))

        chart_w = 12.7 * cm
        chart_h = 7.6 * cm

        chunk = []
        for title, png in chart_items:
            if not png:
                continue
            chunk.append((title, png))
            if len(chunk) == 2:
                imgs = [
                    RLImage(io.BytesIO(chunk[0][1]), width=chart_w, height=chart_h),
                    RLImage(io.BytesIO(chunk[1][1]), width=chart_w, height=chart_h),
                ]
                caps = [Paragraph(html.escape(chunk[0][0]), styles["Caption"]), Paragraph(html.escape(chunk[1][0]), styles["Caption"])]
                tbl = Table([imgs, caps], colWidths=[chart_w, chart_w])
                tbl.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 6))
                chunk = []

        if len(chunk) == 1:
            title, png = chunk[0]
            tbl = Table(
                [[RLImage(io.BytesIO(png), width=chart_w, height=chart_h), ""],
                 [Paragraph(html.escape(title), styles["Caption"]), ""]],
                colWidths=[chart_w, chart_w],
            )
            tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(tbl)

        story.append(PageBreak())

    # Tables
    def add_df(title: str, df_in: pd.DataFrame, max_rows: int):
        if df_in is None or df_in.empty:
            return
        story.append(Paragraph(title, styles["H2"]))
        df_show = df_in.head(max_rows).copy()
        data = [list(df_show.columns)] + df_show.astype(str).values.tolist()
        col_count = len(df_show.columns)
        col_w = (26.0 * cm) / max(col_count, 1)
        tbl = Table(data, colWidths=[col_w] * col_count, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.8),
            ("FONTSIZE", (0, 1), (-1, -1), 8.3),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))

    for title, df_tbl, max_rows in tables:
        add_df(title, df_tbl, max_rows)

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
# HEADER + PDF button (top right, same row as title)
# ==========================
st.title("Insights de Vendas")
titulo_rep = "Todos" if rep_selected == "Todos" else rep_selected

h_left, h_right = st.columns([0.78, 0.22], vertical_alignment="center")
with h_left:
    st.subheader(f"Representante: **{titulo_rep}**")
with h_right:
    gerar_pdf = st.button("📄 Gerar PDF", use_container_width=True, key="pdf_gen_btn")

st.caption(f"Período selecionado: {current_period_label}")
st.markdown("---")


# ==========================
# TOP KPIs (5 columns)
# ==========================
col1, col2, col3, col4, col5 = st.columns(5)

total_rep = float(df_rep["Valor"].sum()) if not df_rep.empty else 0.0
total_vol_rep = float(df_rep["Quantidade"].sum()) if not df_rep.empty else 0.0

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

# IMPORTANT: always define these (prevents NameError later)
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

highlights_lines = []
mensal_rep = None

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

    highlights_lines = [
        f"Melhor mês (Faturamento): {best_fat['MesLabel']} — {format_brl(best_fat['Valor'])}",
        f"Pior mês (Faturamento): {worst_fat['MesLabel']} — {format_brl(worst_fat['Valor'])}",
        f"Melhor mês (Volume): {best_vol['MesLabel']} — {format_un(best_vol['Quantidade'])}",
        f"Pior mês (Volume): {worst_vol['MesLabel']} — {format_un(worst_vol['Quantidade'])}",
    ]

st.markdown("---")

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

fat_curr = float(df_rep["Valor"].sum()) if not df_rep.empty else 0.0
vol_curr = float(df_rep["Quantidade"].sum()) if not df_rep.empty else 0.0
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
# CATEGORIAS VENDIDAS
# ==========================
st.markdown("---")
st.subheader("Categorias vendidas")

fig_cat = None
cat_disp = pd.DataFrame()

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

fig_states = None
estados_display = pd.DataFrame()

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

st.markdown("---")

# ==========================
# DISTRIBUIÇÃO POR CLIENTES
# ==========================
st.subheader("Distribuição por clientes")

fig_clients = None
df_tbl_clients = pd.DataFrame()

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

        fig_clients = px.pie(dist_df, values="Valor", names="Legenda", category_orders={"Legenda": order_legenda})
        fig_clients.update_traces(text=dist_df["Text"], textposition="inside", textinfo="text", insidetextorientation="radial")
        fig_clients.update_layout(showlegend=False, height=560, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_clients, width="stretch")

    with col_tbl:
        df_tbl = df_clientes_full.copy()
        df_tbl["Faturamento"] = df_tbl["Valor"].map(format_brl)
        df_tbl["% Faturamento"] = df_tbl["Share"].map(lambda x: f"{x:.1%}")
        df_tbl["Volume"] = df_tbl["Quantidade"].map(format_un)
        df_tbl_clients = df_tbl.head(15)[["Cliente", "Cidade", "Estado", "Faturamento", "% Faturamento", "Volume"]]

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

        cols_c = list(df_tbl_clients.columns)
        html_c = "<table class='clientes-resumo'><thead><tr>"
        html_c += "".join(f"<th>{html.escape(str(c))}</th>" for c in cols_c)
        html_c += "</tr></thead><tbody>"
        for _, r in df_tbl_clients.iterrows():
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

status_counts = pd.DataFrame()
chart_pie = None
status_disp = pd.DataFrame()

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
                        alt.Tooltip("%Clientes:Q", title="% Clientes", format=".1%")
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


# ==========================
# PDF GENERATION (runs after everything so charts exist)
# ==========================
if "pdf_report_bytes" not in st.session_state:
    st.session_state["pdf_report_bytes"] = None
    st.session_state["pdf_report_name"] = None
    st.session_state["pdf_report_error"] = None

if gerar_pdf:
    st.session_state["pdf_report_error"] = None
    st.session_state["pdf_report_bytes"] = None
    st.session_state["pdf_report_name"] = None

    # Build chart PNGs
    chart_items = []

    # 1) Estados (Plotly)
    png = _chart_to_png(fig_states, width_px=1600, scale=2.0)
    if png:
        chart_items.append(("Distribuição por estados (Top 10)", png))

    # 2) Evolução (Altair) atual e anterior
    if chart_curr is not None:
        png = _chart_to_png(chart_curr.properties(width=760, height=280), width_px=1600, scale=2.0)
        if png:
            chart_items.append((f"Evolução — Período atual ({current_period_label})", png))

    if chart_prev is not None:
        png = _chart_to_png(chart_prev.properties(width=760, height=280), width_px=1600, scale=2.0)
        if png:
            chart_items.append((f"Evolução — Período anterior ({previous_period_label})", png))

    # 3) Categorias (Plotly pie)
    png = _chart_to_png(fig_cat, width_px=1600, scale=2.0)
    if png:
        chart_items.append(("Participação por categoria (Top 10 + Outras)", png))

    # 4) Clientes (Plotly pie)
    png = _chart_to_png(fig_clients, width_px=1600, scale=2.0)
    if png:
        chart_items.append(("Participação dos clientes (Top 10 + Outros)", png))

    # 5) Carteira status (Altair)
    if chart_pie is not None:
        png = _chart_to_png(chart_pie.properties(width=760, height=280), width_px=1600, scale=2.0)
        if png:
            chart_items.append(("Distribuição de clientes por status", png))

    # KPI dict
    kpis_pdf = {
        "Total período": format_brl_compact(total_rep),
        "Média mensal": format_brl_compact(media_mensal),
        "Clientes atendidos": str(clientes_atendidos),
        "Cidades atendidas": str(cidades_atendidas),
        "Estados atendidos": str(estados_atendidos),
        "Distribuição (HHI)": f"{hhi_label_short} (HHI {hhi_value:.3f})",
        "N80": f"{n80_count} clientes",
        "Saúde da carteira": f"{carteira_score:.0f} / 100 ({carteira_label})",
    }

    # Tables for PDF
    tables = []
    if isinstance(estados_display, pd.DataFrame) and not estados_display.empty:
        tables.append(("Resumo — Top 10 estados", estados_display, 10))
    if isinstance(cat_disp, pd.DataFrame) and not cat_disp.empty:
        tables.append(("Resumo — Categorias", cat_disp, 25))
    if isinstance(df_tbl_clients, pd.DataFrame) and not df_tbl_clients.empty:
        tables.append(("Resumo — Clientes (Top 15)", df_tbl_clients, 15))
    if isinstance(status_disp, pd.DataFrame) and not status_disp.empty:
        tables.append(("Resumo — Status da carteira", status_disp, 10))

    try:
        pdf_bytes = build_pdf_report(
            rep_name=titulo_rep,
            current_period_label=current_period_label,
            previous_period_label=previous_period_label,
            kpis=kpis_pdf,
            highlights_lines=highlights_lines,
            chart_items=chart_items,
            tables=tables,
        )
        safe_rep = re.sub(r"[^A-Za-z0-9_-]+", "_", str(titulo_rep))[:40]
        st.session_state["pdf_report_name"] = f"relatorio_insights_{safe_rep}_{start_comp.strftime('%Y%m')}-{end_comp.strftime('%Y%m')}.pdf"
        st.session_state["pdf_report_bytes"] = pdf_bytes
    except Exception as ex:
        st.session_state["pdf_report_error"] = str(ex)

if st.session_state.get("pdf_report_error"):
    st.error(f"Não foi possível gerar o PDF: {st.session_state['pdf_report_error']}")

if st.session_state.get("pdf_report_bytes") and st.session_state.get("pdf_report_name"):
    st.download_button(
        "⬇️ Baixar PDF",
        data=st.session_state["pdf_report_bytes"],
        file_name=st.session_state["pdf_report_name"],
        mime="application/pdf",
    )
    # (Optional small hint, no extra explanation text in UI)
    st.caption("PDF pronto para download.")
