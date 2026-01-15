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

def _plotly_to_png_bytes(fig, scale: int = 2):
    if fig is None: return None, "fig is None"
    try:
        return pio.to_image(fig, format="png", scale=scale), None
    except Exception as e:
        return None, str(e)

def _rl_image_from_png(png_bytes, max_w, max_h):
    if not png_bytes: return None
    bio = io.BytesIO(png_bytes)
    img = RLImage(bio)
    iw, ih = img.imageWidth, img.imageHeight
    if iw <= 0 or ih <= 0: return None
    s = min(max_w/iw, max_h/ih)
    img.drawWidth = iw * s
    img.drawHeight = ih * s
    return img

def build_pdf_report(rep_title, period_label, kpis, charts_png):
    """Generates the PDF Report."""
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Relatório de Performance: {rep_title}", styles['Heading1']))
    story.append(Paragraph(f"Período: {period_label}", styles['Normal']))
    story.append(Spacer(1, 10))

    # KPIs
    data = [[f"{k}: {v}" for k, v in kpis.items()]]
    t = Table(data, colWidths=[doc.width/len(kpis)]*len(kpis))
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0f2f6")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Charts
    for key, png_bytes in charts_png.items():
        img = _rl_image_from_png(png_bytes, doc.width, 250)
        if img:
            story.append(img)
            story.append(Spacer(1, 10))

    doc.build(story)
    return buff.getvalue()
