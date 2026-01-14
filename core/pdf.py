# core/pdf.py
import io
import html
from datetime import datetime
import plotly.io as pio
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

def _plotly_to_png_bytes(fig, scale: int = 2) -> tuple[bytes | None, str | None]:
    """Return (png_bytes, error_message)."""
    if fig is None:
        return None, "fig is None"
    try:
        b = pio.to_image(fig, format="png", scale=scale)  # requires kaleido
        return b, None
    except Exception as e:
        return None, f"plotly export failed: {e}"

def _altair_to_png_bytes(chart, scale: float = 2.0) -> tuple[bytes | None, str | None]:
    """Return (png_bytes, error_message). Uses Altair's saver (vl-convert-python)."""
    if chart is None:
        return None, "chart is None"
    try:
        buf = io.BytesIO()
        try:
            chart.save(buf, format="png", scale=scale)
        except TypeError:
            chart.save(buf, format="png")
        return buf.getvalue(), None
    except Exception as e:
        return None, f"altair export failed: {e}"

def _rl_image_from_png(png_bytes: bytes, max_w: float, max_h: float) -> RLImage | None:
    """Scale image to fit within max_w/max_h (points)."""
    if not png_bytes:
        return None
    bio = io.BytesIO(png_bytes)
    img = RLImage(bio)
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0:
        return None
    s = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * s
    img.drawHeight = ih * s
    return img

def build_pdf_report(
    rep_title: str,
    current_period_label: str,
    previous_period_label: str,
    kpis: dict[str, str],
    highlights_lines: list[str],
    charts_png: dict[str, bytes | None],
) -> bytes:
    buff = io.BytesIO()
    doc = SimpleDocTemplate(
        buff,
        pagesize=landscape(A4),
        leftMargin=1.0 * cm,
        rightMargin=1.0 * cm,
        topMargin=0.9 * cm,
        bottomMargin=0.9 * cm,
        title="Insights de Vendas",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, leading=18, spaceAfter=4))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=9, leading=11))
    styles.add(ParagraphStyle("Muted", parent=styles["BodyText"], fontSize=9, leading=11, textColor=colors.HexColor("#666666")))
    styles.add(ParagraphStyle("KpiTitle", parent=styles["BodyText"], fontSize=8.8, leading=10, textColor=colors.HexColor("#444444")))
    styles.add(ParagraphStyle("KpiValue", parent=styles["BodyText"], fontSize=10.5, leading=12, textColor=colors.HexColor("#111111")))
    styles["BodyText"].alignment = TA_LEFT

    story = []

    # Header
    story.append(Paragraph("Insights de Vendas", styles["H1"]))
    story.append(Paragraph(f"<b>Representante:</b> {html.escape(rep_title)}", styles["Small"]))
    story.append(
        Paragraph(
            f"<b>Período:</b> {html.escape(current_period_label)} &nbsp;&nbsp;•&nbsp;&nbsp; "
            f"<b>Anterior:</b> {html.escape(previous_period_label)}",
            styles["Muted"],
        )
    )
    story.append(Spacer(1, 8))

    # KPI cards
    kpi_items = list(kpis.items())
    if kpi_items:
        cards = []
        for title, value in kpi_items:
            cards.append([
                [Paragraph(html.escape(str(title)), styles["KpiTitle"])],
                [Paragraph(f"<b>{html.escape(str(value))}</b>", styles["KpiValue"])],
            ])

        per_row = 4
        rows = [cards[i:i + per_row] for i in range(0, len(cards), per_row)]

        for r in rows:
            row_cells = []
            for card in r:
                t = Table(card, colWidths=[(doc.width / per_row) - 10])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F7F7")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#DDDDDD")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                row_cells.append(t)

            outer = Table([row_cells], colWidths=[doc.width / per_row] * len(row_cells))
            outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(outer)
            story.append(Spacer(1, 6))

    # Destaques
    story.append(Paragraph("Destaques do período", styles["H2"]))
    if highlights_lines:
        for line in highlights_lines:
            story.append(Paragraph("• " + html.escape(line), styles["Small"]))
    else:
        story.append(Paragraph("Sem destaques disponíveis para o período.", styles["Muted"]))

    story.append(Spacer(1, 8))

    # Chart sizes
    max_w_full = doc.width
    max_h_full = 250
    max_w_half = (doc.width - 10) / 2
    max_h_half = 220

    def img_or_blank(key, w, h):
        b = charts_png.get(key)
        img = _rl_image_from_png(b, w, h) if b else None
        return img if img else Paragraph("<font color='#999999'>—</font>", styles["Muted"])

    # Evolução
    story.append(Paragraph("Evolução – Faturamento x Volume (Período atual)", styles["H2"]))
    story.append(img_or_blank("evolucao_curr", max_w_full, max_h_full))
    story.append(Spacer(1, 8))

    # Distributions
    story.append(Paragraph("Distribuições", styles["H2"]))
    grid = Table(
        [
            [Paragraph("<b>Categorias</b>", styles["Small"]), Paragraph("<b>Estados</b>", styles["Small"])],
            [img_or_blank("pie_cat", max_w_half, max_h_half), img_or_blank("pie_states", max_w_half, max_h_half)],
            [Paragraph("<b>Clientes</b>", styles["Small"]), Paragraph("<b>Status da carteira</b>", styles["Small"])],
            [img_or_blank("pie_clients", max_w_half, max_h_half), img_or_blank("pie_status", max_w_half, max_h_half)],
        ],
        colWidths=[max_w_half, max_w_half],
        hAlign="LEFT",
    )
    grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(grid)
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Muted"]))
    doc.build(story)
    return buff.getvalue()
