import re
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# ---------------- BASIC CONFIG ----------------

st.set_page_config(
    page_title="Carelabel & SKU Label Generator",
    layout="wide",
)

ASSETS_DIR = Path("assets")

BRAND_LOGOS = {
    "Arezzo": ASSETS_DIR / "logo_arezzo.png",
    "Anacapri": ASSETS_DIR / "logo_anacapri.png",
    "Schutz": ASSETS_DIR / "logo_schutz.png",
    "Reserva": ASSETS_DIR / "logo_reserva.png",
}

CARE_ICONS_PATH = ASSETS_DIR / "carelabel_icons.png"


# ---------------- IMAGE HELPERS ----------------

def load_image_base64(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


BRAND_LOGOS_B64 = {name: load_image_base64(p) for name, p in BRAND_LOGOS.items()}
CARE_ICONS_B64 = load_image_base64(CARE_ICONS_PATH)


# ---------------- TRANSLATION / TEXT ----------------

def translate_composition_to_pt(text: str) -> str:
    """
    Very simple EN -> PT-BR translator for compositions.
    Extend as needed.
    """
    if not text:
        return ""

    result = text.strip()

    replacements = [
        (r"polyvinyl chloride\s*\(?\s*pvc\s*\)?", "POLICLORETO DE VINILA (PVC)"),
        (r"\bpvc\b", "POLICLORETO DE VINILA (PVC)"),
        (r"polyurethane", "POLIURETANO (PU)"),
        (r"\bpu\b", "POLIURETANO (PU)"),
        (r"polyester", "POLIÉSTER"),
        (r"polyamide", "POLIAMIDA"),
        (r"nylon", "POLIAMIDA"),
        (r"cotton", "ALGODÃO"),
        (r"filler", "ENCHIMENTO"),
        (r"base fabric", "TECIDO BASE"),
        (r"leather", "COURO"),
        (r"metal", "METAL"),
    ]

    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)

    return result.upper()


def build_carelabel_text(exterior_pt: str, forro_pt: str) -> str:
    """
    Fixed Portuguese body with dynamic EXTERIOR / FORRO.
    """
    text = f"""IMPORTADO POR BTG PACTUAL
COMMODITIES SERTRADING S.A
CNPJ: 04.626.426/0007-00
DISTRIBUIDO POR:
AZZAS 2154 S.A
CNPJ: 16.590.234/0025-43

FABRICADO NA CHINA
SACAREZZO@AREZZO.COM.BR

PRODUTO DE MATERIAL SINTÉTICO
MATÉRIA-PRIMA
EXTERIOR: {exterior_pt}
FORRO: {forro_pt}

PROIBIDO LAVAR NA ÁGUA / NÃO ALVEJAR /
PROIBIDO USAR SECADOR / NÃO PASSAR
A FERRO / NÃO LAVAR A SECO /
LIMPAR COM PANO SECO"""
    return text


# ---------------- PDF GENERATION ----------------

def create_carelabel_pdf(brand: str, full_text: str) -> bytes:
    """
    Single-page carelabel PDF:

    • Physical size: 30 x 80 mm (W x H)
    • Logo centered at the top in a fixed band
    • Text block with font size tuned to match reference
    • Care icons at the bottom with fixed visual size
    """

    # Page size (80 x 30 mm carelabel, vertical)
    width = 30 * mm   # width in mm
    height = 80 * mm  # height in mm

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))

    # Outer border
    border_margin = 0.5 * mm
    c.setLineWidth(0.5)
    c.rect(
        border_margin,
        border_margin,
        width - 2 * border_margin,
        height - 2 * border_margin,
    )

    inner_margin_x = 3 * mm

    # Logo band (top)
    top_margin = 3 * mm          # distance from top border
    logo_max_height = 14 * mm    # height of logo band (tuned to sample)
    logo_max_width = width - 2 * inner_margin_x

    logo_path = BRAND_LOGOS.get(brand)
    if logo_path and logo_path.exists():
        logo_img = ImageReader(str(logo_path))
        iw, ih = logo_img.getSize()
        scale = min(logo_max_width / iw, logo_max_height / ih)
        draw_w = iw * scale
        draw_h = ih * scale

        x_logo = (width - draw_w) / 2.0
        y_logo = height - top_margin - draw_h

        c.drawImage(
            logo_img,
            x_logo,
            y_logo,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        # Text starts a bit below the logo
        text_top_y = y_logo - 2 * mm
    else:
        # If logo not found, reserve same band anyway
        text_top_y = height - (top_margin + logo_max_height + 2 * mm)

    # Care icons (bottom)
    icons_bottom_margin = 3 * mm
    icons_max_height = 7 * mm          # controls icon height (match sample feel)
    icons_max_width = width - 8 * mm   # leaves some side margin

    text_bottom_limit = icons_bottom_margin  # fallback if no icons

    if CARE_ICONS_PATH.exists():
        icons_img = ImageReader(str(CARE_ICONS_PATH))
        iw, ih = icons_img.getSize()
        scale_i = min(icons_max_width / iw, icons_max_height / ih)
        draw_w_i = iw * scale_i
        draw_h_i = ih * scale_i

        x_icons = (width - draw_w_i) / 2.0
        y_icons = icons_bottom_margin  # fixed distance from bottom border

        c.drawImage(
            icons_img,
            x_icons,
            y_icons,
            width=draw_w_i,
            height=draw_h_i,
            preserveAspectRatio=True,
            mask="auto",
        )

        # Text must stay above the icon band
        text_bottom_limit = y_icons + draw_h_i + 2 * mm

    # Text block (middle)
    font_size = 7          # tuned to look like your sample
    leading = 8            # line spacing (points)

    text_obj = c.beginText()
    text_obj.setFont("Helvetica", font_size)
    text_obj.setLeading(leading)
    text_obj.setTextOrigin(inner_margin_x, text_top_y)

    for line in full_text.splitlines():
        # Stop if we are about to overlap the icons
        if text_obj.getY() <= text_bottom_limit:
            break
        text_obj.textLine(line)

    c.drawText(text_obj)

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


def create_sku_labels_pdf(skus: list[str]) -> bytes:
    """
    Multi-page PDF.
    Each page = 50 x 10 mm, border, SKU centered.
    """
    width = 50 * mm
    height = 10 * mm

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))

    for sku in skus:
        sku = sku.strip()
        if not sku:
            continue

        # Border
        border_margin = 0.5 * mm
        c.setLineWidth(0.5)
        c.rect(
            border_margin,
            border_margin,
            width - 2 * border_margin,
            height - 2 * border_margin,
        )

        # SKU centered
        c.setFont("Helvetica", 10)
        c.drawCentredString(width / 2.0, height / 2.0 - 3, sku)

        c.showPage()

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# ---------------- HTML PREVIEWS (UI ONLY) ----------------

def carelabel_preview_html(full_text: str, brand: str) -> str:
    """
    Carelabel preview (approximate on-screen view).
    """
    logo_b64 = BRAND_LOGOS_B64.get(brand)
    icons_b64 = CARE_ICONS_B64

    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" '
        f'style="max-width:140px; max-height:90px; margin-bottom:6px;" />'
        if logo_b64
        else ""
    )

    icons_html = (
        f'<img src="data:image/png;base64,{icons_b64}" '
        f'style="width:75%; max-height:60px; margin-top:8px;" />'
        if icons_b64
        else ""
    )

    return f"""
    <div style="
        border:1px solid #000;
        padding:8px 10px;
        width:260px;
        min-height:520px;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        ">
        <div style="text-align:center; margin-bottom:8px;">
            {logo_html}
        </div>
        <div style="font-size:9px; line-height:1.35; white-space:pre-wrap;">
            {full_text}
        </div>
        <div style="margin-top:8px; text-align:center;">
            {icons_html}
        </div>
    </div>
    """


def sku_label_preview_html(sku: str) -> str:
    """
    Simple bordered horizontal SKU preview.
    """
    return f"""
    <div style="
        border:1px solid #000;
        width:300px;
        height:60px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        font-size:20px;
        letter-spacing:2px;
        margin-bottom:8px;
        ">
        {sku}
    </div>
    """


# ---------------- SIDEBAR ----------------

st.sidebar.title("Carelabel Generator")

brand = st.sidebar.selectbox("Brand", list(BRAND_LOGOS.keys()))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Carelabel PDF: 80×30 mm (vertical).\n"
    "SKU labels PDF: 10×50 mm (horizontal, 1 SKU por página)."
)


# ---------------- MAIN UI ----------------

st.title("Carelabel & SKU Label Generator")

tab_care, tab_sku = st.tabs(["Carelabel (80×30 mm)", "SKU labels (10×50 mm)"])


# ---- CARELABEL TAB ----
with tab_care:
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.subheader("Carelabel – Composição")

        family_code = st.text_input(
            "Product family (para nome do arquivo)",
            value="",
            help="Ex.: C500390016 – todas as cores/SKUs desta família usam a mesma carelabel.",
        )

        st.write("### Composition")
        exterior_en = st.text_input(
            "EXTERIOR",
            value="100% PVC",
            help="English ou Português. Ex.: '75% Polyester, 25% Polyvinyl Chloride (PVC)'",
        )
        forro_en = st.text_input(
            "FORRO / LINING",
            value="100% Polyester",
            help="English ou Português. Ex.: '100% Polyester'",
        )

        already_pt = st.checkbox(
            "Composition already in Portuguese (skip auto-translation)",
            value=False,
        )

        generate_care = st.button("Generate carelabel PDF")

    with col_right:
        st.subheader("Preview & PDF")

        if generate_care:
            # Store family in session for optional use in SKU tab
            st.session_state["family_code"] = family_code.strip()

            if already_pt:
                exterior_pt = exterior_en.strip().upper()
                forro_pt = forro_en.strip().upper()
            else:
                exterior_pt = translate_composition_to_pt(exterior_en)
                forro_pt = translate_composition_to_pt(forro_en)

            full_text = build_carelabel_text(exterior_pt, forro_pt)

            # HTML preview
            st.markdown(
                carelabel_preview_html(full_text, brand),
                unsafe_allow_html=True,
            )

            # PDF
            pdf_bytes = create_carelabel_pdf(brand, full_text)
            pdf_name_base = family_code.strip() or "CARELABEL"
            st.download_button(
                "Download carelabel PDF",
                data=pdf_bytes,
                file_name=f"{pdf_name_base} - CARE LABEL.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Preencha a composição e clique em **Generate carelabel PDF**.")


# ---- SKU LABELS TAB ----
with tab_sku:
    if "sku_count" not in st.session_state:
        st.session_state["sku_count"] = 4  # start with 4 fields

    col_left, col_right = st.columns([1.1, 1.6])

    with col_left:
        st.subheader("SKUs para esta carelabel")

        # Suggest family from carelabel tab, if filled
        default_family = st.session_state.get("family_code", "")
        family_code_sku = st.text_input(
            "Product family (para nome do PDF)",
            value=default_family,
            help="Ex.: C500390016 – usada apenas para nome do PDF.",
        )

        if st.button("Add another SKU field"):
            st.session_state["sku_count"] += 1

        sku_values = []
        for i in range(st.session_state["sku_count"]):
            sku_val = st.text_input(
                f"SKU {i + 1}",
                key=f"sku_{i+1}",
                placeholder="Ex.: C5003900160001",
            )
            if sku_val.strip():
                sku_values.append(sku_val.strip())

        generate_skus = st.button("Generate SKU labels PDF")

    with col_right:
        st.subheader("Preview & PDF")

        if generate_skus:
            if not sku_values:
                st.warning("Informe pelo menos um SKU.")
            else:
                # HTML previews
                for sku in sku_values:
                    st.markdown(sku_label_preview_html(sku), unsafe_allow_html=True)

                # PDF
                sku_pdf = create_sku_labels_pdf(sku_values)
                sku_pdf_name = family_code_sku.strip() or "SKUS"
                st.download_button(
                    "Download SKU labels PDF",
                    data=sku_pdf,
                    file_name=f"{sku_pdf_name} - SKU LABELS.pdf",
                    mime="application/pdf",
                )
        else:
            st.info(
                "Digite os SKUs (vários, se quiser) e clique em "
                "**Generate SKU labels PDF**."
            )
