# ==========================
# PDF EXPORT (COMPLETE REWRITE - Replace ALL PDF sections with this)
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
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# PDF chart conversion helpers (fallback to None if dependencies missing)
def _safe_plotly_to_png(fig, width_px=800, height_px=400):
    """Safe Plotly to PNG - works even without kaleido"""
    if fig is None:
        return None
    try:
        import plotly.io as pio
        return fig.to_image(format="png", width=width_px, height=height_px, engine="kaleido")
    except:
        try:
            # Fallback: save to temp file and read
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.write_image(f.name, engine="kaleido", width=width_px, height=height_px)
                with open(f.name, "rb") as img_file:
                    png_bytes = img_file.read()
            return png_bytes
        except:
            return None

def _safe_altair_to_png(chart, width_px=800, height_px=400):
    """Safe Altair to PNG - skips if vl_convert not available"""
    if chart is None:
        return None
    try:
        import vl_convert as vlc
        spec = chart.to_dict()
        spec["width"] = width_px
        spec["height"] = height_px
        return vlc.vegalite_to_png(spec, scale=2.0)
    except:
        return None

def chart_to_png(obj, width_px=800, height_px=400):
    """Universal chart to PNG converter"""
    if obj is None:
        return None
    if hasattr(obj, 'to_image'):  # Plotly
        return _safe_plotly_to_png(obj, width_px, height_px)
    elif hasattr(obj, 'to_dict'):  # Altair
        return _safe_altair_to_png(obj, width_px, height_px)
    return None

# PDF Styles
_PDF_STYLES = getSampleStyleSheet()
_PDF_STYLES.update({
    'Title': ParagraphStyle('Title', parent=_PDF_STYLES['Title'], fontSize=18, leading=22, spaceAfter=12, alignment=TA_CENTER),
    'H1': ParagraphStyle('H1', parent=_PDF_STYLES['Heading1'], fontSize=14, leading=16, spaceAfter=8),
    'H2': ParagraphStyle('H2', parent=_PDF_STYLES['Heading2'], fontSize=12, leading=14, spaceAfter=6),
    'Body': ParagraphStyle('Body', parent=_PDF_STYLES['BodyText'], fontSize=10, leading=12),
    'Small': ParagraphStyle('Small', parent=_PDF_STYLES['BodyText'], fontSize=9, leading=11),
    'Caption': ParagraphStyle('Caption', parent=_PDF_STYLES['BodyText'], fontSize=8, leading=10, textColor=colors.grey),
})

def create_kpi_card(title, value, subtitle=""):
    """Create styled KPI card for PDF"""
    title_p = Paragraph(f"<b>{html.escape(str(title))}</b>", _PDF_STYLES['Small'])
    value_p = Paragraph(f"<b style='font-size:13px'>{html.escape(str(value))}</b>", _PDF_STYLES['Body'])
    subtitle_p = Paragraph(html.escape(str(subtitle)), _PDF_STYLES['Caption']) if subtitle else Paragraph("&nbsp;", _PDF_STYLES['Caption'])
    
    tbl = Table([[title_p], [value_p], [subtitle_p]], colWidths=[5.2*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1f2937")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#374151")),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    return tbl

def df_to_pdf_table(df, max_rows=15, title_cols=True):
    """Convert DataFrame to styled PDF table"""
    if df is None or df.empty:
        return Paragraph("<i>Sem dados</i>", _PDF_STYLES['Caption'])
    
    df_show = df.head(max_rows).copy().astype(str)
    data = [list(df_show.columns)] + df_show.values.tolist() if title_cols else df_show.values.tolist()
    
    tbl = Table(data, colWidths=None, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#111827")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f9fafa")]),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return tbl

def build_pdf_report(rep_name, current_period, previous_period, kpis, highlights, 
                    fig_states, chart_curr, chart_prev, fig_cat, fig_clients, chart_pie,
                    estados_table, cat_table, clients_table, carteira_table, df_rep):
    """Build complete PDF report"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), rightMargin=1.5*cm, 
                          leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    
    story = []
    
    # Header
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph("📊 INSIGHTS DE VENDAS", _PDF_STYLES['Title']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Representante: <b>{html.escape(rep_name)}</b>", _PDF_STYLES['H1']))
    story.append(Paragraph(f"Período: <b>{current_period}</b> | Anterior: <b>{previous_period}</b> | Gerado: {now}", _PDF_STYLES['Caption']))
    story.append(Spacer(1, 12))
    
    # KPIs Row
    kpi_items = list(kpis.items())[:5]  # Take first 5
    kpi_cards = [create_kpi_card(k, v) for k, v in kpi_items]
    while len(kpi_cards) < 5:
        kpi_cards.append(Paragraph("<i>Sem dados</i>", _PDF_STYLES['Caption']))
    
    kpi_table = Table([kpi_cards], colWidths=[5.2*cm]*5)
    kpi_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))
    
    # Highlights
    if highlights:
        hl_text = "<br/>".join([f"• {html.escape(h)}" for h in highlights[:6]])
        story.extend([
            Paragraph("Destaques do Período", _PDF_STYLES['H1']),
            Paragraph(hl_text, _PDF_STYLES['Body']),
            Spacer(1, 12)
        ])
    
    # Charts (convert to PNG first)
    charts = [
        (chart_curr, "Evolução - Período Atual", 26.3, 6.0),
        (chart_prev, "Evolução - Período Anterior", 26.3, 6.0),
        (fig_cat, "Categorias Vendidas", 12.5, 7.0),
        (fig_states, "Estados", 12.5, 7.0),
        (fig_clients, "Clientes", 12.5, 7.0),
        (chart_pie, "Saúde da Carteira", 12.5, 7.0),
    ]
    
    for chart, title, w_cm, h_cm in charts:
        png = chart_to_png(chart)
        if png:
            img = RLImage(io.BytesIO(png), width=w_cm*cm, height=h_cm*cm)
            story.extend([
                Paragraph(title, _PDF_STYLES['H2']),
                img,
                Spacer(1, 8)
            ])
    
    # Tables
    tables_data = [
        (estados_table, "Top Estados", 10),
        (cat_table, "Categorias", 12),
        (clients_table, "Top Clientes", 12),
        (carteira_table, "Carteira por Status", 10)
    ]
    
    for table_df, title, max_rows in tables_data:
        if not table_df.empty:
            story.extend([
                Paragraph(title, _PDF_STYLES['H2']),
                df_to_pdf_table(table_df, max_rows),
                Spacer(1, 12)
            ])
    
    doc.build(story)
    return buf.getvalue()

# ==========================
# PDF BUTTON (Replace your existing PDF button block)
# ==========================
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
    st.session_state.pdf_name = None
    st.session_state.pdf_error = None

# Create PDF button in header (you already have this)
if gerar_pdf:
    with st.spinner("Gerando PDF..."):
        try:
            st.session_state.pdf_error = None
            
            # Safe KPI dict
            kpis_safe = {
                "Total": format_brl_compact(total_rep),
                "Média Mensal": format_brl_compact(media_mensal),
                "Clientes": str(clientes_atendidos),
                "Cidades": str(cidades_atendidas),
                "Estados": str(estados_atendidos)
            }
            
            # Safe tables
            tables_safe = {
                'estados': estados_display if 'estados_display' in locals() and isinstance(estados_display, pd.DataFrame) else pd.DataFrame(),
                'cat': cat_disp if 'cat_disp' in locals() and isinstance(cat_disp, pd.DataFrame) else pd.DataFrame(),
                'clients': df_tbl_clients if 'df_tbl_clients' in locals() and isinstance(df_tbl_clients, pd.DataFrame) else pd.DataFrame(),
                'carteira': status_disp if 'status_disp' in locals() and isinstance(status_disp, pd.DataFrame) else pd.DataFrame()
            }
            
            pdf_bytes = build_pdf_report(
                rep_name=titulo_rep,
                current_period=current_period_label,
                previous_period=previous_period_label,
                kpis=kpis_safe,
                highlights=highlights_lines,
                fig_states=fig_states if 'fig_states' in locals() else None,
                chart_curr=chart_curr if 'chart_curr' in locals() else None,
                chart_prev=chart_prev if 'chart_prev' in locals() else None,
                fig_cat=fig_cat if 'fig_cat' in locals() else None,
                fig_clients=fig_clients if 'fig_clients' in locals() else None,
                chart_pie=chart_pie if 'chart_pie' in locals() else None,
                estados_table=tables_safe['estados'],
                cat_table=tables_safe['cat'],
                clients_table=tables_safe['clients'],
                carteira_table=tables_safe['carteira'],
                df_rep=df_rep
            )
            
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(titulo_rep))[:30]
            st.session_state.pdf_name = f"insights_vendas_{safe_name}_{start_comp.strftime('%Y%m')}.pdf"
            st.session_state.pdf_bytes = pdf_bytes
            
        except Exception as e:
            st.session_state.pdf_error = f"Erro: {str(e)}"
            st.error(f"Erro ao gerar PDF: {st.session_state.pdf_error}")

# Download button
if st.session_state.get('pdf_bytes') and st.session_state.get('pdf_name'):
    st.download_button(
        label="⬇️ Baixar PDF Gerado",
        data=st.session_state.pdf_bytes,
        file_name=st.session_state.pdf_name,
        mime="application/pdf"
    )
elif st.session_state.get('pdf_error'):
    st.error(st.session_state.pdf_error)
