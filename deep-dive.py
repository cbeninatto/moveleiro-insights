import math
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Nova Visão – Deep Dive",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RAW_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/refs/heads/main/data/relatorio_faturamento.csv"
GEO_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/refs/heads/main/data/cidades_br_geo.csv"
GITHUB_COMMIT_API = "https://api.github.com/repos/cbeninatto/performance-moveleiro-v2/commits"
DATA_PATH_IN_REPO = "data/relatorio_faturamento.csv"

PT_MONTHS = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR", 5: "MAI", 6: "JUN",
    7: "JUL", 8: "AGO", 9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ"
}

STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
STATUS_COLORS = {
    "Perdidos": "#ef4444",   # red
    "Caindo": "#f97316",     # orange
    "Estáveis": "#eab308",   # yellow
    "Crescendo": "#22c55e",  # green
    "Novos": "#3b82f6",      # blue
}

# Same palette used in dashboard_deep_dive_map.html (inverted so green=alto, red=baixo)
MAP_BIN_COLORS_LOW_TO_HIGH = ["#ef4444", "#f97316", "#eab308", "#22c55e"]  # red -> green


# =========================================================
# CSS (inclui regras de impressão A4 e quebras de página)
# =========================================================
st.markdown(
    """
<style>
:root{
  --card: rgba(255,255,255,0.04);
  --stroke: rgba(255,255,255,0.08);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --muted2: rgba(255,255,255,0.55);
}
.block-container{ padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1400px; }
.small-muted{ color: var(--muted2); font-size: 0.9rem; }

.kpi-grid{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
  margin-top: 10px;
  margin-bottom: 14px;
}
.kpi-card{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  min-width: 0;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.kpi-label{ color: var(--muted); font-size: 0.92rem; margin-bottom: 6px; }
.kpi-value{
  color: var(--text);
  font-weight: 800;
  line-height: 1.05;
  font-size: clamp(22px, 2.2vw, 36px);
  overflow: visible;
  text-overflow: unset;
  white-space: normal;
  word-break: break-word;
}
.kpi-sub{ margin-top: 8px; color: var(--muted2); font-size: 0.92rem; }
.kpi-pill{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(34,197,94,0.12);
  color: rgba(34,197,94,0.95);
  border: 1px solid rgba(34,197,94,0.25);
  border-radius: 999px;
  padding: 4px 10px;
  font-weight: 650;
  font-size: 0.86rem;
}

.section-card{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}
.hr{ height:1px; background: rgba(255,255,255,0.08); margin: 10px 0 14px 0; }

/* HTML tables */
.table-wrap table{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
  table-layout: fixed;
}
.table-wrap th, .table-wrap td{
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding: 8px 10px;
  vertical-align: top;
  overflow: hidden;
  text-overflow: ellipsis;
}
.table-wrap th{
  color: rgba(255,255,255,0.85);
  font-weight: 800;
  background: rgba(255,255,255,0.03);
  text-align: left;
}
.table-nowrap td, .table-nowrap th{ white-space: nowrap; }

/* Print */
.page-break{ height: 1px; margin: 0; }
@media print{
  @page{ size: A4; margin: 10mm; }
  header, footer, #MainMenu, [data-testid="stSidebar"], [data-testid="stToolbar"]{ display:none !important; }
  html, body{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
  .block-container{ max-width: 780px !important; padding-top: 0.2rem !important; }
  .page-break{ break-after: page; page-break-after: always; }
  .kpi-card, .section-card, .stPlotlyChart, .stAltairChart{ break-inside: avoid; page-break-inside: avoid; }
  iframe{ max-width: 100% !important; }
  /* prevent column overlap */
  div[data-testid="stHorizontalBlock"]{ flex-wrap: wrap !important; }
  div[data-testid="stColumn"]{ min-width: 0 !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Helpers
# =========================================================
def brl(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "R$ 0,00"
    s = f"{float(x):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "0,0%"
    return f"{float(x)*100:.1f}%".replace(".", ",")


def month_label(y: int, m: int) -> str:
    return f"{PT_MONTHS.get(int(m), str(m))} {str(int(y))[-2:]}"


def iter_months(start_y: int, start_m: int, end_y: int, end_m: int):
    y, m = int(start_y), int(start_m)
    while (y < end_y) or (y == end_y and m <= end_m):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def period_key(y: int, m: int) -> int:
    return int(y) * 100 + int(m)


def get_latest_commit_sha(path_in_repo: str) -> str | None:
    try:
        r = requests.get(
            GITHUB_COMMIT_API,
            params={"path": path_in_repo, "per_page": 1, "page": 1},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("sha")
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False, ttl=300)
def load_geo(sha: str | None) -> pd.DataFrame:
    url = GEO_CSV_URL if not sha else f"{GEO_CSV_URL}?cb={sha}"
    r = requests.get(url, timeout=30, headers={"Cache-Control": "no-cache"})
    r.raise_for_status()
    g = pd.read_csv(StringIO(r.text))

    cols = {c.lower(): c for c in g.columns}
    estado_c = cols.get("estado") or cols.get("uf")
    cidade_c = cols.get("cidade")
    lat_c = cols.get("lat") or cols.get("latitude")
    lon_c = cols.get("lon") or cols.get("lng") or cols.get("longitude")

    if not (estado_c and cidade_c and lat_c and lon_c):
        raise KeyError("cidades_br_geo.csv deve conter colunas Estado, Cidade, lat, lon")

    g = g[[estado_c, cidade_c, lat_c, lon_c]].copy()
    g.columns = ["Estado", "Cidade", "lat", "lon"]
    g["Estado"] = g["Estado"].astype(str).str.strip().str.upper()
    g["Cidade"] = g["Cidade"].astype(str).str.strip().str.upper()
    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g = g.dropna(subset=["lat", "lon"])
    return g


@st.cache_data(show_spinner=False, ttl=120)
def load_sales(sha: str | None, bust: int) -> pd.DataFrame:
    # cache-busting para pegar atualizações do CSV imediatamente
    if sha:
        url = f"{RAW_CSV_URL}?cb={sha}-{bust}"
    else:
        url = f"{RAW_CSV_URL}?cb={int(datetime.utcnow().timestamp())}-{bust}"

    r = requests.get(url, timeout=60, headers={"Cache-Control": "no-cache"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    # Coerção de tipos
    for c in ["Quantidade", "Valor", "Mes", "Ano"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Ano" not in df.columns or "Mes" not in df.columns:
        raise KeyError("CSV precisa ter colunas Ano e Mes.")

    df["Ano"] = df["Ano"].astype("Int64")
    df["MesNum"] = df["Mes"].astype("Int64")
    df["CompetenciaKey"] = df["Ano"].astype(int) * 100 + df["MesNum"].astype(int)

    df["Competencia"] = pd.to_datetime(
        df["Ano"].astype(str) + "-" + df["MesNum"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    for c in ["Cliente", "Cidade", "Estado", "Representante"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def filter_df_period(df: pd.DataFrame, sy: int, sm: int, ey: int, em: int) -> pd.DataFrame:
    start_key = period_key(sy, sm)
    end_key = period_key(ey, em)
    return df[(df["CompetenciaKey"] >= start_key) & (df["CompetenciaKey"] <= end_key)].copy()


def compute_basic_metrics(df_period: pd.DataFrame, sy: int, sm: int, ey: int, em: int) -> dict:
    total = float(df_period["Valor"].sum()) if len(df_period) else 0.0
    months = list(iter_months(sy, sm, ey, em))
    n_months = len(months) if months else 1

    by_month = (
        df_period.groupby(["Ano", "MesNum"], as_index=False)
        .agg(Faturamento=("Valor", "sum"))
    )
    months_with_sales = int((by_month["Faturamento"] > 0).sum())
    clientes = int(df_period["Cliente"].nunique()) if "Cliente" in df_period.columns else 0

    return {
        "total": total,
        "avg_month": total / n_months,
        "months_with_sales": months_with_sales,
        "months_total": n_months,
        "clientes": clientes,
    }


def compute_destaques(df_period: pd.DataFrame):
    if df_period.empty:
        return None
    m = (
        df_period.groupby(["Ano", "MesNum"], as_index=False)
        .agg(Faturamento=("Valor", "sum"), Volume=("Quantidade", "sum"))
        .sort_values(["Ano", "MesNum"])
    )
    best_fat = m.loc[m["Faturamento"].idxmax()]
    worst_fat = m.loc[m["Faturamento"].idxmin()]
    best_vol = m.loc[m["Volume"].idxmax()]
    worst_vol = m.loc[m["Volume"].idxmin()]
    return {
        "best_fat": (int(best_fat["Ano"]), int(best_fat["MesNum"]), float(best_fat["Faturamento"])),
        "worst_fat": (int(worst_fat["Ano"]), int(worst_fat["MesNum"]), float(worst_fat["Faturamento"])),
        "best_vol": (int(best_vol["Ano"]), int(best_vol["MesNum"]), float(best_vol["Volume"])),
        "worst_vol": (int(worst_vol["Ano"]), int(worst_vol["MesNum"]), float(worst_vol["Volume"])),
    }


def compute_client_concentration(df_period: pd.DataFrame):
    if df_period.empty:
        return {
            "n80": 0, "hhi": 0.0, "label": "Sem dados",
            "top1": 0.0, "top3": 0.0, "top10": 0.0,
            "pie_df": pd.DataFrame(columns=["Cliente", "Valor", "Legenda", "TextInside"])
        }

    c = df_period.groupby("Cliente", as_index=False).agg(Valor=("Valor", "sum"))
    c = c.sort_values("Valor", ascending=False).reset_index(drop=True)
    total = float(c["Valor"].sum()) if len(c) else 0.0
    if total <= 0:
        return {
            "n80": 0, "hhi": 0.0, "label": "Sem faturamento",
            "top1": 0.0, "top3": 0.0, "top10": 0.0,
            "pie_df": pd.DataFrame(columns=["Cliente", "Valor", "Legenda", "TextInside"])
        }

    c["share"] = c["Valor"] / total
    c["cum"] = c["share"].cumsum()
    n80 = int((c["cum"] <= 0.8).sum())
    if n80 < len(c):
        n80 += 1

    hhi = float((c["share"] ** 2).sum())

    if hhi < 0.10:
        label = "Baixa concentração"
    elif hhi < 0.18:
        label = "Concentração moderada"
    else:
        label = "Alta concentração"

    top1 = float(c["share"].iloc[0]) if len(c) else 0.0
    top3 = float(c["share"].head(3).sum()) if len(c) else 0.0
    top10 = float(c["share"].head(10).sum()) if len(c) else 0.0

    # Pie: Top 10 + Outros (representa todos os demais clientes)
    top = c.head(10).copy()
    rest_val = float(c["Valor"].iloc[10:].sum()) if len(c) > 10 else 0.0
    if rest_val > 0:
        top = pd.concat([top, pd.DataFrame([{"Cliente": "Outros", "Valor": rest_val}])], ignore_index=True)

    top["Percent"] = top["Valor"] / total
    top["Legenda"] = top.apply(lambda r: f"{str(r['Cliente'])[:28]} {pct(float(r['Percent']))}", axis=1)

    def inside_text(row):
        p = float(row["Percent"])
        if row["Cliente"] == "Outros":
            return f"Outros\\n{pct(p)}"
        if p >= 0.06:
            return f"{str(row['Cliente'])[:20]}\\n{pct(p)}"
        return ""

    top["TextInside"] = top.apply(inside_text, axis=1)

    # Ordenação por valor (desc) e "Outros" por último
    if "Outros" in top["Cliente"].values:
        top_no_outros = top[top["Cliente"] != "Outros"].sort_values("Valor", ascending=False)
        outros = top[top["Cliente"] == "Outros"]
        top = pd.concat([top_no_outros, outros], ignore_index=True)
    else:
        top = top.sort_values("Valor", ascending=False).reset_index(drop=True)

    return {"n80": n80, "hhi": hhi, "label": label, "top1": top1, "top3": top3, "top10": top10, "pie_df": top}


def compute_states_distribution(df_period: pd.DataFrame):
    if df_period.empty:
        return pd.DataFrame(columns=["Estado", "Valor", "Percent"])
    s = df_period.groupby("Estado", as_index=False).agg(Valor=("Valor", "sum"))
    total = float(s["Valor"].sum()) if len(s) else 0.0
    s["Percent"] = (s["Valor"] / total) if total > 0 else 0.0
    return s.sort_values("Valor", ascending=False).head(10).reset_index(drop=True)


def carteira_status(df_all: pd.DataFrame, sy: int, sm: int, ey: int, em: int, representante: str | None):
    """Status da carteira comparando o período selecionado vs. mesmos meses do ano anterior."""
    cur = filter_df_period(df_all, sy, sm, ey, em)
    prev = filter_df_period(df_all, sy - 1, sm, ey - 1, em)

    if representante and representante != "Todos":
        cur = cur[cur["Representante"] == representante]
        prev = prev[prev["Representante"] == representante]

    cur_cli = cur.groupby("Cliente", as_index=False).agg(FaturamentoAtual=("Valor", "sum"))
    prev_cli = prev.groupby("Cliente", as_index=False).agg(FaturamentoAnterior=("Valor", "sum"))

    dim = df_all.copy()
    if representante and representante != "Todos":
        dim = dim[dim["Representante"] == representante]
    dim = (
        dim.sort_values("CompetenciaKey")
        .groupby("Cliente", as_index=False)
        .agg(Estado=("Estado", "last"), Cidade=("Cidade", "last"))
    )

    d = cur_cli.merge(prev_cli, on="Cliente", how="outer").merge(dim, on="Cliente", how="left")
    d["FaturamentoAtual"] = d["FaturamentoAtual"].fillna(0.0)
    d["FaturamentoAnterior"] = d["FaturamentoAnterior"].fillna(0.0)

    def status_row(r):
        curv = float(r["FaturamentoAtual"])
        prevv = float(r["FaturamentoAnterior"])
        if prevv <= 0 and curv > 0:
            return "Novos"
        if prevv > 0 and curv <= 0:
            return "Perdidos"
        if prevv <= 0 and curv <= 0:
            return "Estáveis"
        chg = (curv - prevv) / prevv if prevv else 0.0
        if chg >= 0.15:
            return "Crescendo"
        if chg <= -0.15:
            return "Caindo"
        return "Estáveis"

    d["Status"] = d.apply(status_row, axis=1)
    d["DeltaFaturamento"] = d["FaturamentoAtual"] - d["FaturamentoAnterior"]

    total_cli = len(d) or 1
    summary = (
        d.groupby("Status", as_index=False)
        .agg(Clientes=("Cliente", "count"), FaturamentoDelta=("DeltaFaturamento", "sum"))
    )
    summary["PercentClientes"] = summary["Clientes"] / total_cli
    summary["Status"] = pd.Categorical(summary["Status"], categories=STATUS_ORDER, ordered=True)
    summary = summary.sort_values("Status")

    return d, summary


def compute_carteira_score(summary_df: pd.DataFrame) -> int:
    """Score 0–100 baseado na composição e dinâmica da carteira."""
    if summary_df is None or summary_df.empty:
        return 0

    counts = {str(row["Status"]): int(row["Clientes"]) for _, row in summary_df.iterrows()}
    total = sum(counts.values()) or 1

    novos = counts.get("Novos", 0) / total
    crescendo = counts.get("Crescendo", 0) / total
    estaveis = counts.get("Estáveis", 0) / total
    caindo = counts.get("Caindo", 0) / total
    perdidos = counts.get("Perdidos", 0) / total

    score = 60.0
    score += 25.0 * crescendo
    score += 10.0 * novos
    score += 5.0 * estaveis
    score -= 20.0 * caindo
    score -= 40.0 * perdidos

    return int(round(max(0.0, min(100.0, score))))


def render_kpis(metrics: dict, conc: dict, score: int):
    pill = f"HHI {conc['hhi']:.3f} • N80: {conc['n80']} clientes"
    html = f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total período</div>
        <div class="kpi-value">{brl(metrics['total'])}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Média mensal</div>
        <div class="kpi-value">{brl(metrics['avg_month'])}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Distribuição por clientes</div>
        <div class="kpi-value">{conc['label']}</div>
        <div class="kpi-sub"><span class="kpi-pill">{pill}</span></div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Saúde da carteira</div>
        <div class="kpi-value">{score} / 100</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Clientes atendidos</div>
        <div class="kpi-value">{metrics['clientes']}</div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def folium_add_legend(m, bins):
    rows = "".join(
        [f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;'>"
         f"<span style='width:12px;height:12px;border-radius:3px;background:{b['color']};display:inline-block;'></span>"
         f"<span style='font-size:12px;color:#111;'>{b['label']}</span></div>" for b in bins]
    )
    legend_html = f"""
    <div style="
      position: fixed; bottom: 18px; left: 18px; z-index: 9999;
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 10px;
      padding: 10px 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.18);
      ">
      <div style="font-weight:800;font-size:12px;margin-bottom:6px;color:#111;">Legenda</div>
      {rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def build_city_map(df_period: pd.DataFrame, geo: pd.DataFrame, metric: str):
    g = (
        df_period.groupby(["Estado", "Cidade"], as_index=False)
        .agg(
            Faturamento=("Valor", "sum"),
            Volume=("Quantidade", "sum"),
            Clientes=("Cliente", "nunique"),
        )
    )
    if g.empty:
        m = folium.Map(location=[-14.2, -51.9], zoom_start=4, tiles="cartodbpositron")
        return m, g, []

    g["Estado"] = g["Estado"].astype(str).str.strip().str.upper()
    g["Cidade"] = g["Cidade"].astype(str).str.strip().str.upper()

    gg = g.merge(geo, on=["Estado", "Cidade"], how="left").dropna(subset=["lat", "lon"]).copy()
    if gg.empty:
        m = folium.Map(location=[-14.2, -51.9], zoom_start=4, tiles="cartodbpositron")
        return m, g, []

    val_col = "Faturamento" if metric == "Faturamento" else "Volume"
    vals = gg[val_col].astype(float)

    q1, q2, q3 = vals.quantile([0.25, 0.50, 0.75]).tolist()
    thresholds = [float(q1), float(q2), float(q3)]

    def bin_idx(v):
        if v <= thresholds[0]:
            return 0
        if v <= thresholds[1]:
            return 1
        if v <= thresholds[2]:
            return 2
        return 3

    def fmt_range(lo, hi):
        if metric == "Faturamento":
            return f"{brl(lo)} – {brl(hi)}"
        return f"{lo:,.0f} – {hi:,.0f}".replace(",", ".")

    lo0, hi0 = float(vals.min()), thresholds[0]
    lo1, hi1 = thresholds[0], thresholds[1]
    lo2, hi2 = thresholds[1], thresholds[2]
    lo3, hi3 = thresholds[2], float(vals.max())

    bins = [
        {"label": fmt_range(lo0, hi0), "color": MAP_BIN_COLORS_LOW_TO_HIGH[0]},
        {"label": fmt_range(lo1, hi1), "color": MAP_BIN_COLORS_LOW_TO_HIGH[1]},
        {"label": fmt_range(lo2, hi2), "color": MAP_BIN_COLORS_LOW_TO_HIGH[2]},
        {"label": fmt_range(lo3, hi3), "color": MAP_BIN_COLORS_LOW_TO_HIGH[3]},
    ]

    center = [float(gg["lat"].mean()), float(gg["lon"].mean())]
    m = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")

    vmax = float(vals.max()) if float(vals.max()) > 0 else 1.0
    for _, r in gg.iterrows():
        v = float(r[val_col])
        b = bin_idx(v)
        color = MAP_BIN_COLORS_LOW_TO_HIGH[b]
        radius = 7 + (10 * (v / vmax))
        tooltip = f"{r['Cidade'].title()} - {r['Estado']} • {metric}: {brl(v) if metric=='Faturamento' else f'{v:,.0f}'.replace(',','.')}"
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=float(radius),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
            tooltip=tooltip,
        ).add_to(m)

    folium_add_legend(m, bins)
    return m, gg, bins


def html_table(df: pd.DataFrame, nowrap_cols=None, center_headers=None, col_widths=None) -> str:
    nowrap_cols = set(nowrap_cols or [])
    center_headers = set(center_headers or [])

    colgroup = ""
    if col_widths and len(col_widths) == len(df.columns):
        colgroup = "<colgroup>" + "".join([f"<col style='width:{w};'>" for w in col_widths]) + "</colgroup>"

    ths = ""
    for c in df.columns:
        align = "center" if c in center_headers else "left"
        ths += f"<th style='text-align:{align};'>{c}</th>"

    rows = ""
    for _, row in df.iterrows():
        tds = ""
        for c in df.columns:
            val = row[c]
            style = "white-space:nowrap;" if c in nowrap_cols else ""
            tds += f"<td style='{style}'>{val}</td>"
        rows += f"<tr>{tds}</tr>"

    return f"<div class='table-wrap table-nowrap'><table>{colgroup}<thead><tr>{ths}</tr></thead><tbody>{rows}</tbody></table></div>"


# =========================================================
# Load data (auto-detect updates)
# =========================================================
if "cache_bust" not in st.session_state:
    st.session_state["cache_bust"] = 0

with st.sidebar:
    st.markdown("### Deep Dive")
    if st.button("🔄 Atualizar dados"):
        st.session_state["cache_bust"] += 1
        st.cache_data.clear()

sha = get_latest_commit_sha(DATA_PATH_IN_REPO)

try:
    df = load_sales(sha, st.session_state["cache_bust"])
except Exception as e:
    st.error(f"Erro ao carregar dados do GitHub: {e}")
    st.stop()

try:
    geo = load_geo(sha)
except Exception as e:
    st.warning(f"Mapa: não foi possível carregar cidades_br_geo.csv ({e}).")
    geo = pd.DataFrame(columns=["Estado", "Cidade", "lat", "lon"])


# =========================================================
# Sidebar filters (período + representante com 'Todos')
# =========================================================
years = sorted([int(y) for y in df["Ano"].dropna().unique().tolist()])
if not years:
    st.error("Sem dados de Ano/Mes no CSV.")
    st.stop()

max_year = max(years)
months_max_year = sorted(df.loc[df["Ano"] == max_year, "MesNum"].dropna().unique().astype(int).tolist())
default_sm = min(months_max_year) if months_max_year else 1
default_em = max(months_max_year) if months_max_year else 12

with st.sidebar:
    st.markdown("### Filtros")
    cA, cB = st.columns(2)
    with cA:
        start_year = st.selectbox("Ano inicial", years, index=years.index(max_year))
        start_month = st.selectbox("Mês inicial", list(range(1, 13)), index=default_sm - 1, format_func=lambda x: PT_MONTHS[x])
    with cB:
        end_year = st.selectbox("Ano final", years, index=years.index(max_year))
        end_month = st.selectbox("Mês final", list(range(1, 13)), index=default_em - 1, format_func=lambda x: PT_MONTHS[x])

    if period_key(end_year, end_month) < period_key(start_year, start_month):
        start_year, end_year = end_year, start_year
        start_month, end_month = end_month, start_month

    df_period_for_rep = filter_df_period(df, start_year, start_month, end_year, end_month)
    reps = sorted(df_period_for_rep["Representante"].dropna().unique().tolist())
    reps = ["Todos"] + reps
    representante = st.selectbox("Representante", reps, index=0)

df_period = df_period_for_rep.copy()
if representante != "Todos":
    df_period = df_period[df_period["Representante"] == representante]

period_label = f"{PT_MONTHS[start_month]} {start_year} até {PT_MONTHS[end_month]} {end_year}"
rep_title = "Todos os representantes" if representante == "Todos" else representante


# =========================================================
# Compute data for sections
# =========================================================
metrics = compute_basic_metrics(df_period, start_year, start_month, end_year, end_month)
destaques = compute_destaques(df_period)
conc = compute_client_concentration(df_period)
det_status, sum_status = carteira_status(df, start_year, start_month, end_year, end_month, representante)
score = compute_carteira_score(sum_status)


# =========================================================
# PAGE 1
# =========================================================
st.markdown(f"# Representante: {rep_title}")
st.markdown(f"<div class='small-muted'>Período selecionado: {period_label}</div>", unsafe_allow_html=True)

render_kpis(metrics, conc, score)

st.markdown("## Destaques do período")
if not destaques:
    st.info("Sem dados para destacar no período selecionado.")
else:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Faturamento")
        bf = destaques["best_fat"]
        wf = destaques["worst_fat"]
        st.write(f"**Melhor mês:** {month_label(bf[0], bf[1])} • {brl(bf[2])}")
        st.write(f"**Pior mês:** {month_label(wf[0], wf[1])} • {brl(wf[2])}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### Volume")
        bv = destaques["best_vol"]
        wv = destaques["worst_vol"]
        st.write(f"**Melhor mês:** {month_label(bv[0], bv[1])} • {bv[2]:,.0f}".replace(",", "."))
        st.write(f"**Pior mês:** {month_label(wv[0], wv[1])} • {wv[2]:,.0f}".replace(",", "."))
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## Mapa de Clientes")

map_metric = st.radio("Métrica do mapa", ["Faturamento", "Volume"], horizontal=True, key="map_metric")

col_map, col_right = st.columns([1.0, 1.25], gap="large")

with col_map:
    m, cities_df, bins = build_city_map(df_period, geo, map_metric)
    map_out = st_folium(m, height=720, width="100%", returned_objects=["last_object_clicked"])

with col_right:
    st.markdown("#### Cobertura")
    cc1, cc2, cc3 = st.columns(3, gap="medium")
    cidades_atendidas = int(df_period[["Estado", "Cidade"]].drop_duplicates().shape[0]) if not df_period.empty else 0
    estados_atendidos = int(df_period["Estado"].nunique()) if not df_period.empty else 0
    with cc1:
        st.metric("Cidades", cidades_atendidas)
    with cc2:
        st.metric("Estados", estados_atendidos)
    with cc3:
        st.metric("Clientes", metrics["clientes"])

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown("#### Principais clientes")
    top_clients = (
        df_period.groupby(["Cliente", "Cidade", "Estado"], as_index=False)
        .agg(Faturamento=("Valor", "sum"))
        .sort_values("Faturamento", ascending=False)
        .head(15)
    )
    top_disp = top_clients.copy()
    top_disp["Faturamento"] = top_disp["Faturamento"].apply(brl)

    st.markdown(
        html_table(
            top_disp[["Cliente", "Cidade", "Estado", "Faturamento"]],
            nowrap_cols=["Faturamento"],
            col_widths=["46%", "22%", "12%", "20%"],
        ),
        unsafe_allow_html=True,
    )

    clicked = (map_out or {}).get("last_object_clicked")
    if clicked and not cities_df.empty:
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("#### Ver lista de clientes da cidade")
        try:
            lat = float(clicked.get("lat"))
            lon = float(clicked.get("lng"))
            tmp = cities_df.copy()
            tmp["dist"] = (tmp["lat"] - lat) ** 2 + (tmp["lon"] - lon) ** 2
            row = tmp.sort_values("dist").head(1)
            if not row.empty:
                estado = row.iloc[0]["Estado"]
                cidade = row.iloc[0]["Cidade"]
                st.markdown(f"<div class='small-muted'><b>{cidade.title()}</b> – {estado}</div>", unsafe_allow_html=True)

                city_rows = df_period.copy()
                city_rows["CidadeU"] = city_rows["Cidade"].astype(str).str.strip().str.upper()
                city_rows["EstadoU"] = city_rows["Estado"].astype(str).str.strip().str.upper()
                city_rows = city_rows[(city_rows["CidadeU"] == str(cidade).upper()) & (city_rows["EstadoU"] == str(estado).upper())]

                city_clients = (
                    city_rows.groupby("Cliente", as_index=False)
                    .agg(Quantidade=("Quantidade", "sum"), Faturamento=("Valor", "sum"))
                    .sort_values("Faturamento", ascending=False)
                )
                city_disp = city_clients.copy()
                city_disp["Quantidade"] = city_disp["Quantidade"].map(lambda x: f"{float(x):,.0f}".replace(",", "."))
                city_disp["Faturamento"] = city_disp["Faturamento"].apply(brl)

                st.markdown(
                    html_table(
                        city_disp[["Cliente", "Quantidade", "Faturamento"]],
                        nowrap_cols=["Quantidade", "Faturamento"],
                        center_headers=["Quantidade", "Faturamento"],
                        col_widths=["54%", "18%", "28%"],
                    ),
                    unsafe_allow_html=True,
                )
        except Exception:
            st.caption("Clique em um ponto no mapa para ver a lista de clientes daquela cidade.")
    else:
        st.caption("Clique em um ponto no mapa para ver a lista de clientes daquela cidade.")

st.markdown("## Distribuição por estados")
states = compute_states_distribution(df_period)
if states.empty:
    st.info("Sem dados por estado no período.")
else:
    st.markdown("#### Top 10 estados por faturamento – % do faturamento total")
    fig_states = px.bar(
        states,
        x="Percent",
        y="Estado",
        orientation="h",
        text=states["Percent"].map(lambda x: pct(float(x))),
        height=520,
    )
    fig_states.update_layout(
        xaxis_title="% do faturamento total",
        yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig_states.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_states, width="stretch")

st.markdown("<div class='page-break'></div>", unsafe_allow_html=True)


# =========================================================
# PAGE 2
# =========================================================
st.markdown("## Evolução – Faturamento x Volume")

evo = (
    df_period.groupby(["Ano", "MesNum"], as_index=False)
    .agg(Faturamento=("Valor", "sum"), Volume=("Quantidade", "sum"))
    .sort_values(["Ano", "MesNum"])
)
if evo.empty:
    st.info("Sem dados de evolução para o período.")
else:
    evo["Label"] = evo.apply(lambda r: month_label(int(r["Ano"]), int(r["MesNum"])), axis=1)

    fig = go.Figure()
    fig.add_bar(x=evo["Label"], y=evo["Faturamento"], name="Faturamento", yaxis="y", opacity=0.85)
    fig.add_trace(
        go.Scatter(
            x=evo["Label"],
            y=evo["Volume"],
            name="Volume",
            yaxis="y2",
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=7, color="#22c55e"),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="", tickangle=0),
        yaxis=dict(title="Faturamento", tickformat="~s"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")

st.markdown("## Distribuição por clientes")

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
with k1:
    st.metric("N80", f"{conc['n80']}")
with k2:
    st.metric("HHI", f"{conc['hhi']:.3f}")
with k3:
    st.metric("Top 1", pct(conc["top1"]))
with k4:
    st.metric("Top 3", pct(conc["top3"]))
with k5:
    st.metric("Top 10", pct(conc["top10"]))

dist_cols = st.columns([1.15, 1.0], gap="large")
with dist_cols[0]:
    top10 = (
        df_period.groupby("Cliente", as_index=False)
        .agg(Faturamento=("Valor", "sum"))
        .sort_values("Faturamento", ascending=False)
        .head(10)
    )
    if top10.empty:
        st.info("Sem dados por cliente.")
    else:
        fig_top10 = px.bar(
            top10.sort_values("Faturamento", ascending=True),
            x="Faturamento",
            y="Cliente",
            orientation="h",
            title="Top 10 clientes por faturamento",
            height=420,
        )
        fig_top10.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_top10, width="stretch")

with dist_cols[1]:
    pie_df = conc["pie_df"]
    if pie_df.empty:
        st.info("Sem dados para pizza.")
    else:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=pie_df["Legenda"],
                    values=pie_df["Valor"],
                    sort=False,
                    text=pie_df["TextInside"],
                    textinfo="text",
                    textposition="inside",
                    insidetextorientation="auto",
                    hovertemplate="%{label}<br>Valor: %{value:,.2f}<br>%{percent}<extra></extra>",
                )
            ]
        )
        fig_pie.update_layout(
            title="Participação dos clientes (Top 10 destacados)",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )
        st.plotly_chart(fig_pie, width="stretch")

st.markdown("## Saúde da carteira – Detalhes")

if sum_status.empty:
    st.info("Sem dados de carteira para o período.")
else:
    sum_disp = sum_status.copy()
    sum_disp["Status"] = sum_disp["Status"].astype(str)
    sum_disp["Clientes"] = sum_disp["Clientes"].astype(int)
    sum_disp["% Clientes"] = sum_disp["PercentClientes"].map(lambda x: pct(float(x)))
    sum_disp["Faturamento (Δ)"] = sum_disp["FaturamentoDelta"].map(brl)
    sum_disp = sum_disp[["Status", "Clientes", "% Clientes", "Faturamento (Δ)"]]

    st.markdown("#### Resumo por status")
    st.markdown(
        html_table(
            sum_disp,
            nowrap_cols=["Clientes", "% Clientes", "Faturamento (Δ)"],
            col_widths=["22%", "14%", "16%", "48%"],
        ),
        unsafe_allow_html=True,
    )
    st.caption("Obs.: **Faturamento (Δ)** = Atual − Anterior (meses selecionados vs. mesmos meses do ano anterior).")

    pie_status = sum_status.copy()
    pie_status["Status"] = pie_status["Status"].astype(str)
    pie_status["color"] = pie_status["Status"].map(lambda s: STATUS_COLORS.get(s, "#94a3b8"))

    fig_status = go.Figure(
        data=[
            go.Pie(
                labels=pie_status["Status"],
                values=pie_status["Clientes"],
                sort=False,
                marker=dict(colors=pie_status["color"]),
                textinfo="percent+label",
            )
        ]
    )
    fig_status.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_status, width="stretch")

st.markdown("<div class='page-break'></div>", unsafe_allow_html=True)


# =========================================================
# PAGE 3
# =========================================================
st.markdown("## Status dos clientes")

search = st.text_input("Pesquisar cliente", value="", placeholder="Digite parte do nome do cliente...")
d = det_status.copy()
if search.strip():
    d = d[d["Cliente"].str.contains(search.strip(), case=False, na=False)]

cur_period_name = f"Faturamento {PT_MONTHS[start_month]} {str(start_year)[-2:]} – {PT_MONTHS[end_month]} {str(end_year)[-2:]}"
prev_period_name = f"Faturamento {PT_MONTHS[start_month]} {str(start_year-1)[-2:]} – {PT_MONTHS[end_month]} {str(end_year-1)[-2:]}"

d_disp = d.copy()
d_disp[cur_period_name] = d_disp["FaturamentoAtual"].map(brl)
d_disp[prev_period_name] = d_disp["FaturamentoAnterior"].map(brl)
d_disp["Δ Faturamento"] = d_disp["DeltaFaturamento"].map(brl)

base_cols = ["Cliente", "Cidade", "Estado", "Status", cur_period_name, prev_period_name, "Δ Faturamento"]
d_disp = d_disp[base_cols].copy()

col_widths_status = ["34%", "16%", "10%", "12%", "12%", "12%", "14%"]

for status in STATUS_ORDER:
    subset = d_disp[d_disp["Status"] == status].copy()
    if subset.empty:
        continue

    st.markdown(f"### {status}")
    st.markdown(f"<div class='small-muted'>Total de clientes: <b>{len(subset)}</b></div>", unsafe_allow_html=True)

    st.markdown(
        html_table(
            subset,
            nowrap_cols=[cur_period_name, prev_period_name, "Δ Faturamento"],
            col_widths=col_widths_status,
        ),
        unsafe_allow_html=True,
    )
