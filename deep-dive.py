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
