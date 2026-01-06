import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import io
import re
import os
import datetime

# ==========================
# PDF EXPORT (REPORT)
# ==========================
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
)

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
st.set_page_config(page_title="Insights de Vendas", layout="wide")

# ==========================
# HELPERS
# ==========================
MONTH_MAP_NUM_TO_NAME = {
    1: "JAN",
    2: "FEV",
    3: "MAR",
    4: "ABR",
    5: "MAI",
    6: "JUN",
    7: "JUL",
    8: "AGO",
    9: "SET",
    10: "OUT",
    11: "NOV",
    12: "DEZ",
}

STATUS_COL = "Status"
STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]


def format_brl(x):
    try:
        return f"R$ {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


def format_brl_signed(x):
    try:
        v = float(x)
        s = "+" if v >= 0 else "-"
        v = abs(v)
        return f"{s} R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


def format_un(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "—"


def _safe_str(x) -> str:
    return "" if x is None else str(x)


def _plotly_to_png_bytes(fig, width_px: int = 1400, scale: int = 2):
    """Best-effort Plotly -> PNG bytes. Requires kaleido in the environment."""
    if fig is None or pio is None:
        return None
    try:
        return fig.to_image(format="png", width=width_px, scale=scale)
    except Exception:
        return None


def _altair_to_png_bytes(chart, width_px: int = 1400, scale: float = 2.0):
    """Best-effort Altair/Vega-Lite -> PNG bytes. Requires vl-convert-python."""
    if chart is None or vlc is None:
        return None
    try:
        spec = chart.to_dict()
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
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
        title="Insights de Vendas - Relatório",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1", parent=styles["Heading1"], fontSize=16, leading=18, spaceAfter=8
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontSize=12,
            leading=14,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(name="BodyS", parent=styles["BodyText"], fontSize=9.5, leading=12)
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=10,
            textColor=colors.grey,
        )
    )

    story = []
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph("Insights de Vendas — Relatório", styles["H1"]))
    story.append(Paragraph(f"Representante: <b>{_safe_str(rep_name)}</b>", styles["BodyS"]))
    story.append(
        Paragraph(
            f"Período selecionado: <b>{_safe_str(current_period_label)}</b>  |  Período anterior: <b>{_safe_str(previous_period_label)}</b>",
            styles["BodyS"],
        )
    )
    story.append(Paragraph(f"Gerado em: {now}", styles["Caption"]))
    story.append(Spacer(1, 10))

    # KPIs
    story.append(Paragraph("Resumo (KPIs)", styles["H2"]))
    kpi_rows = [["Métrica", "Valor"]] + [[k, v] for k, v in kpis.items()]
    t = Table(kpi_rows, colWidths=[8.0 * cm, 18.0 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
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
        chart_w = 12.7 * cm
        chart_h = 7.6 * cm

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
                tbl.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 6))
                chunk = []

        if len(chunk) == 1:
            ttitle, tpng = chunk[0]
            tbl = Table(
                [[RLImage(io.BytesIO(tpng), width=chart_w, height=chart_h), ""], [Paragraph(ttitle, styles["Caption"]), ""]],
                colWidths=[chart_w, chart_w],
            )
            tbl.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(tbl)

        story.append(PageBreak())

    def _add_df_table(title: str, df_in: pd.DataFrame | None, max_rows: int = 25):
        if df_in is None or df_in.empty:
            return
        story.append(Paragraph(title, styles["H2"]))
        df_show = df_in.head(max_rows).copy()
        data = [list(df_show.columns)] + df_show.astype(str).values.tolist()
        col_count = len(df_show.columns)
        col_w = (26.0 * cm) / max(col_count, 1)
        tbl = Table(data, colWidths=[col_w] * col_count, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8.8),
                    ("FONTSIZE", (0, 1), (-1, -1), 8.3),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
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
@st.cache_data(show_spinner=False)
def load_data():
    # NOTE: adjust this path to your repo file if needed
    # The app expects a CSV with at least: Competencia, Valor, Quantidade, Representante, Cliente, Estado, Cidade, Categoria
    candidates = [
        "data/vendas.csv",
        "vendas.csv",
        "data.csv",
        "dados.csv",
        "output.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            df0 = pd.read_csv(c)
            return df0
    return pd.DataFrame()


df = load_data()

if df is None or df.empty:
    st.error("Nenhum arquivo de dados encontrado. Verifique o caminho do CSV.")
    st.stop()

# Normalize / ensure expected columns
# Try to parse Competencia / Month-Year fields
if "Competencia" in df.columns:
    df["Competencia"] = pd.to_datetime(df["Competencia"], errors="coerce")
else:
    # Try compose from Ano/MesNum if exists
    if "Ano" in df.columns and "MesNum" in df.columns:
        df["Competencia"] = pd.to_datetime(dict(year=df["Ano"], month=df["MesNum"], day=1), errors="coerce")
    else:
        st.error("Coluna 'Competencia' não encontrada e não foi possível inferir por Ano/MesNum.")
        st.stop()

# Create Ano / MesNum if missing
if "Ano" not in df.columns:
    df["Ano"] = df["Competencia"].dt.year
if "MesNum" not in df.columns:
    df["MesNum"] = df["Competencia"].dt.month

# Ensure numeric fields
for c in ["Valor", "Quantidade"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# Ensure required categoricals exist
for c in ["Representante", "Cliente", "Estado", "Cidade", "Categoria"]:
    if c not in df.columns:
        df[c] = ""

# ==========================
# SIDEBAR FILTERS
# ==========================
st.sidebar.header("Filtros")

min_date = df["Competencia"].min()
max_date = df["Competencia"].max()
if pd.isna(min_date) or pd.isna(max_date):
    st.error("Datas inválidas na coluna Competencia.")
    st.stop()

start_comp, end_comp = st.sidebar.date_input(
    "Período (competência)",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
start_comp = pd.to_datetime(start_comp)
end_comp = pd.to_datetime(end_comp)

mask_period = (df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)
df_period = df.loc[mask_period].copy()

if df_period.empty:
    st.warning("Nenhuma venda no período selecionado.")
    st.stop()

# Previous period: same length immediately before
period_days = (end_comp - start_comp).days
prev_end = start_comp - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=period_days)

mask_prev_period = (df["Competencia"] >= prev_start) & (df["Competencia"] <= prev_end)
df_prev_period = df.loc[mask_prev_period].copy()

current_period_label = f"{start_comp.strftime('%d/%m/%Y')} → {end_comp.strftime('%d/%m/%Y')}"
previous_period_label = f"{prev_start.strftime('%d/%m/%Y')} → {prev_end.strftime('%d/%m/%Y')}"

reps_period = sorted(df_period["Representante"].dropna().unique())
if not reps_period:
    st.error("Não há representantes com vendas no período selecionado.")
    st.stop()

rep_options = ["Todos"] + reps_period
rep_selected = st.sidebar.selectbox("Representante", rep_options)

df_rep = df_period.copy() if rep_selected == "Todos" else df_period[df_period["Representante"] == rep_selected].copy()

# ==========================
# BASE METRICS (safe defaults)
# ==========================
clientes_atendidos = int(df_rep["Cliente"].nunique()) if ("Cliente" in df_rep.columns) else 0
cidades_atendidas = int(df_rep["Cidade"].nunique()) if ("Cidade" in df_rep.columns) else 0
estados_atendidos = int(df_rep["Estado"].nunique()) if ("Estado" in df_rep.columns) else 0
total_rep = float(df_rep["Valor"].sum()) if ("Valor" in df_rep.columns) else 0.0

df_rep_prev = df_prev_period.copy() if rep_selected == "Todos" else df_prev_period[df_prev_period["Representante"] == rep_selected].copy()

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

if "pdf_report_bytes" not in st.session_state:
    st.session_state["pdf_report_bytes"] = None
    st.session_state["pdf_report_name"] = None
    st.session_state["pdf_report_error"] = None

if _gen:
    st.session_state["pdf_report_error"] = None
    st.session_state["pdf_report_bytes"] = None
    st.session_state["pdf_report_name"] = None

    if df_rep is None or df_rep.empty:
        st.session_state["pdf_report_error"] = "Sem dados do período atual para gerar o relatório. Ajuste os filtros e tente novamente."
    else:
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

        # N80
        n80_count_pdf = 0
        try:
            if "Cliente" in df_rep.columns and "Valor" in df_rep.columns and clientes_atendidos_pdf > 0:
                by_client = (
                    df_rep.groupby("Cliente", as_index=False)["Valor"]
                    .sum()
                    .sort_values("Valor", ascending=False)
                )
                by_client["share"] = by_client["Valor"] / max(by_client["Valor"].sum(), 1e-9)
                by_client["cum_share"] = by_client["share"].cumsum()
                n80_count_pdf = int((by_client["cum_share"] <= 0.80).sum())
                if n80_count_pdf < len(by_client):
                    n80_count_pdf += 1
        except Exception:
            n80_count_pdf = 0

        # HHI
        hhi_value_pdf = 0.0
        hhi_label_short_pdf = "—"
        try:
            if "Cliente" in df_rep.columns and "Valor" in df_rep.columns and total_rep_pdf > 0:
                shares = (df_rep.groupby("Cliente")["Valor"].sum() / total_rep_pdf).clip(lower=0)
                hhi_value_pdf = float((shares**2).sum())
                if hhi_value_pdf < 0.10:
                    hhi_label_short_pdf = "Baixa"
                elif hhi_value_pdf < 0.18:
                    hhi_label_short_pdf = "Média"
                else:
                    hhi_label_short_pdf = "Alta"
        except Exception:
            pass

        rep_name_pdf = titulo_rep
        current_period_label_pdf = current_period_label
        previous_period_label_pdf = previous_period_label

        # Tables
        estados_tbl_pdf = None
        categorias_tbl_pdf = None
        clientes_tbl_pdf = None
        carteira_tbl_pdf = None

        try:
            estados_df_pdf = (
                df_rep.groupby("Estado", as_index=False)[["Valor", "Quantidade"]]
                .sum()
                .sort_values("Valor", ascending=False)
            )
            if not estados_df_pdf.empty:
                estados_df_pdf["Faturamento"] = estados_df_pdf["Valor"].map(format_brl)
                estados_df_pdf["Volume"] = estados_df_pdf["Quantidade"].map(format_un)
                estados_tbl_pdf = estados_df_pdf[["Estado", "Faturamento", "Volume"]].head(15)
        except Exception:
            estados_tbl_pdf = None

        try:
            if df_rep_prev is not None and not df_rep_prev.empty:
                curr_cat_pdf = (
                    df_rep.groupby("Categoria", as_index=False)["Valor"]
                    .sum()
                    .rename(columns={"Valor": "ValorAtual"})
                )
                prev_cat_pdf = (
                    df_rep_prev.groupby("Categoria", as_index=False)["Valor"]
                    .sum()
                    .rename(columns={"Valor": "ValorAnterior"})
                )
                cat_pdf = pd.merge(curr_cat_pdf, prev_cat_pdf, on="Categoria", how="outer").fillna(0.0)
            else:
                curr_cat_pdf = (
                    df_rep.groupby("Categoria", as_index=False)["Valor"]
                    .sum()
                    .rename(columns={"Valor": "ValorAtual"})
                )
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
                tmp = mensal_pdf.copy()
                tmp["Competencia"] = pd.to_datetime(dict(year=tmp["Ano"], month=tmp["MesNum"], day=1))
                tmp["MesLabel"] = tmp["Competencia"].apply(lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}")
                best_fat = tmp.loc[tmp["Valor"].idxmax()]
                worst_fat = tmp.loc[tmp["Valor"].idxmin()]
                best_vol = tmp.loc[tmp["Quantidade"].idxmax()]
                worst_vol = tmp.loc[tmp["Quantidade"].idxmin()]
                highlights_pdf = {
                    "Melhor mês (Faturamento)": f"{best_fat['MesLabel']} — {format_brl(best_fat['Valor'])}",
                    "Pior mês (Faturamento)": f"{worst_fat['MesLabel']} — {format_brl(worst_fat['Valor'])}",
                    "Melhor mês (Volume)": f"{best_vol['MesLabel']} — {format_un(best_vol['Quantidade'])}",
                    "Pior mês (Volume)": f"{worst_vol['MesLabel']} — {format_un(worst_vol['Quantidade'])}",
                }
        except Exception:
            highlights_pdf = {}

        # Build charts for PDF (from df_rep, not globals)
        chart_items = []

        # Altair: monthly faturamento
        if mensal_pdf is not None and not mensal_pdf.empty:
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

            # Altair: monthly volume
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

        # Plotly: Top 10 estados
        try:
            st_df = df_rep.groupby("Estado", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False).head(10)
            fig_st = px.bar(st_df, x="Estado", y="Valor", title="Top 10 — Estados (Faturamento)")
            png = _chart_to_png_bytes(fig_st, width_px=1400, scale=2.0)
            if png:
                chart_items.append(("Top 10 — Estados (Faturamento)", png))
        except Exception:
            pass

        # Plotly: Top 10 categorias
        try:
            cat_df = df_rep.groupby("Categoria", as_index=False)["Valor"].sum().sort_values("Valor", ascending=False).head(10)
            fig_cat = px.pie(cat_df, names="Categoria", values="Valor", title="Top 10 — Categorias (participação)")
            png = _chart_to_png_bytes(fig_cat, width_px=1400, scale=2.0)
            if png:
                chart_items.append(("Top 10 — Categorias (participação)", png))
        except Exception:
            pass

        # Plotly: Top 10 clientes
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
            file_name = f"relatorio_insights_{safe_rep}_{start_comp.strftime('%Y%m')}-{end_comp.strftime('%Y%m')}.pdf"
            st.session_state["pdf_report_bytes"] = pdf_bytes
            st.session_state["pdf_report_name"] = file_name
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
        use_container_width=False,
    )

st.caption(f"Período selecionado: {current_period_label}")
st.markdown("---")

# ==========================
# TOP KPIs (5 columns)
# ==========================
total_rep = float(df_rep["Valor"].sum()) if "Valor" in df_rep.columns else 0.0

mensal_rep = (
    df_rep.groupby(["Ano", "MesNum"], as_index=False)[["Valor", "Quantidade"]]
    .sum()
    .sort_values(["Ano", "MesNum"])
)

media_mensal = float(mensal_rep["Valor"].mean()) if not mensal_rep.empty else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total período", format_brl(total_rep))
col2.metric("Média mensal", format_brl(media_mensal))
col3.metric("Clientes atendidos", f"{clientes_atendidos}")
col4.metric("Cidades atendidas", f"{cidades_atendidos}")
col5.metric("Estados atendidos", f"{estados_atendidos}")

st.markdown("---")

# ==========================
# DISTRIBUIÇÃO POR CLIENTES (EXCERPT)
# ==========================
st.subheader("Distribuição por clientes")

# FIX: clientes_atendidos is guaranteed defined above now
if df_rep.empty or clientes_atendidos == 0:
    st.info("Nenhum cliente com vendas no período selecionado.")
else:
    df_clientes_full = (
        df_rep.groupby("Cliente", as_index=False)
        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
        .sort_values("Valor", ascending=False)
    )

    df_clientes_full["Share"] = df_clientes_full["Valor"] / max(df_clientes_full["Valor"].sum(), 1e-9)
    df_clientes_full["SharePct"] = (df_clientes_full["Share"] * 100.0).round(2)

    top_n = min(30, len(df_clientes_full))
    df_top = df_clientes_full.head(top_n).copy()

    fig_clients = px.bar(df_top, x="Cliente", y="Valor", title="Top clientes por faturamento")
    st.plotly_chart(fig_clients, width="stretch")

    st.dataframe(
        df_top.assign(Faturamento=df_top["Valor"].map(format_brl), Volume=df_top["Quantidade"].map(format_un))[
            ["Cliente", "Faturamento", "Volume", "SharePct"]
        ],
        width="stretch",
    )

# ==========================
# (REST OF YOUR APP)
# ==========================
# NOTE:
# The remainder of your original script continues below.
# If you want me to include the remaining ~250+ lines exactly as-is from your repo,
# paste the bottom portion after this point and I will merge it cleanly in one message.
