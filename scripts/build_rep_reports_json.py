# scripts/build_rep_reports_json.py
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =========================
# PATHS (repo)
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]

# INPUTS (ajuste se seus caminhos forem diferentes)
RELATORIO_FATURAMENTO = REPO_ROOT / "data" / "raw" / "relatorio_faturamento.csv"
CIDADES_GEO = REPO_ROOT / "data" / "raw" / "cidades_br_geo.csv"
CLIENTES_MAP = REPO_ROOT / "data" / "raw" / "clientes_relatorio_faturamento.csv"
CATEGORIAS_MAP = REPO_ROOT / "data" / "raw" / "categorias_map.csv"

# OUTPUT (GitHub Pages)
OUT_JSON = REPO_ROOT / "reports" / "data" / "rep_reports.json"


# =========================
# CONSTANTS / HELPERS
# =========================
MONTH_MAP_NUM_TO_NAME = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR",
    5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO",
    9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
}

STATUS_COL = "StatusCarteira"

STATUS_WEIGHTS = {
    "Novos": 1, "Novo": 1,
    "Crescendo": 2, "CRESCENDO": 2,
    "Estáveis": 1, "Estável": 1, "ESTAVEIS": 1,
    "Caindo": -1, "CAINDO": -1,
    "Perdidos": -2, "Perdido": -2, "PERDIDOS": -2,
}
STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]


def _norm_col(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_norm_col(c): c for c in df.columns}
    # exact normalized match
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    # contains match
    for nk, orig in norm_map.items():
        for cand in candidates:
            if cand in nk:
                return orig
    return None


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def safe_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(round(float(x)))
    except Exception:
        return 0


def format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    def fmt(d: pd.Timestamp) -> str:
        return f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}"
    if start.year == end.year and start.month == end.month:
        return fmt(start)
    return f"{fmt(start)} - {fmt(end)}"


def previous_window(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Previous window with same month span ending right before start."""
    months_span = (end.year - start.year) * 12 + (end.month - start.month) + 1
    prev_end = start - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=months_span - 1)
    return pd.Timestamp(prev_start.year, prev_start.month, 1), pd.Timestamp(prev_end.year, prev_end.month, 1)


def compute_carteira_score(clientes_carteira: pd.DataFrame) -> Tuple[float, str]:
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


def classify_status(va: float, vp: float) -> str:
    if va > 0 and vp == 0:
        return "Novos"
    if va == 0 and vp > 0:
        return "Perdidos"
    if va > 0 and vp > 0:
        ratio = va / vp if vp != 0 else 0.0
        if ratio >= 1.2:
            return "Crescendo"
        if ratio <= 0.8:
            return "Caindo"
        return "Estáveis"
    return "Estáveis"


def ensure_dirs():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


# =========================
# LOADERS
# =========================
def load_relatorio() -> pd.DataFrame:
    if not RELATORIO_FATURAMENTO.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {RELATORIO_FATURAMENTO}")

    df = pd.read_csv(RELATORIO_FATURAMENTO, encoding="utf-8-sig")

    # expected columns from your streamlit loader
    expected = [
        "Codigo", "Descricao", "Quantidade", "Valor", "Mes", "Ano",
        "ClienteCodigo", "Cliente", "Estado", "Cidade",
        "RepresentanteCodigo", "Representante", "Categoria", "SourcePDF",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"relatorio_faturamento.csv não tem colunas esperadas: {missing}\nColunas: {list(df.columns)}")

    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")
    df["MesNum"] = pd.to_numeric(df["Mes"], errors="coerce").astype("Int64")

    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce",
    )
    df = df.dropna(subset=["Competencia", "Representante", "Cliente"])
    return df


def load_geo() -> pd.DataFrame:
    if not CIDADES_GEO.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CIDADES_GEO}")

    df_geo = pd.read_csv(CIDADES_GEO, sep=None, engine="python", encoding="utf-8-sig")

    estado_col = pick_col(df_geo, ["estado", "uf", "siglauf", "ufsigla", "unidadefederativa", "estadouf", "ufestado", "coduf"])
    cidade_col = pick_col(df_geo, ["cidade", "municipio", "nomemunicipio", "nmmunicipio", "nomecidade", "cidadenome", "municipionome"])
    lat_col = pick_col(df_geo, ["lat", "latitude", "y", "coordy", "coordenaday"])
    lon_col = pick_col(df_geo, ["lon", "lng", "long", "longitude", "x", "coordx", "coordenadax"])

    if not all([estado_col, cidade_col, lat_col, lon_col]):
        raise ValueError(
            "cidades_br_geo.csv está com colunas diferentes das esperadas.\n"
            f"Colunas: {list(df_geo.columns)}"
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


# Optional: if you later want client-level lat/lon (not required for city map)
def load_clientes_map() -> Optional[pd.DataFrame]:
    if not CLIENTES_MAP.exists():
        return None
    try:
        dfc = pd.read_csv(CLIENTES_MAP, sep=None, engine="python", encoding="utf-8-sig")
        return dfc
    except Exception:
        return None


def load_categorias_map() -> Optional[pd.DataFrame]:
    if not CATEGORIAS_MAP.exists():
        return None
    try:
        dfc = pd.read_csv(CATEGORIAS_MAP, sep=None, engine="python", encoding="utf-8-sig")
        return dfc
    except Exception:
        return None


# =========================
# CORE BUILDERS
# =========================
def build_client_status(df_all: pd.DataFrame, rep: str, start: pd.Timestamp, end: pd.Timestamp, prev_start: pd.Timestamp, prev_end: pd.Timestamp) -> pd.DataFrame:
    df_rep_all = df_all if rep == "Todos" else df_all[df_all["Representante"] == rep]
    if df_rep_all.empty:
        return pd.DataFrame(columns=["Cliente", "Estado", "Cidade", "ValorAtual", "ValorAnterior", "QtdAtual", "QtdAnterior", STATUS_COL])

    mask_curr = (df_rep_all["Competencia"] >= start) & (df_rep_all["Competencia"] <= end)
    mask_prev = (df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)

    df_curr = df_rep_all.loc[mask_curr].copy()
    df_prev = df_rep_all.loc[mask_prev].copy()

    curr_agg = (
        df_curr.groupby("Cliente", as_index=False)
        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"), Estado=("Estado", "first"), Cidade=("Cidade", "first"))
        .rename(columns={"Valor": "ValorAtual", "Quantidade": "QtdAtual", "Estado": "EstadoAtual", "Cidade": "CidadeAtual"})
    )
    prev_agg = (
        df_prev.groupby("Cliente", as_index=False)
        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"), Estado=("Estado", "first"), Cidade=("Cidade", "first"))
        .rename(columns={"Valor": "ValorAnterior", "Quantidade": "QtdAnterior", "Estado": "EstadoAnterior", "Cidade": "CidadeAnterior"})
    )

    clientes = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer")
    for c in ["ValorAtual", "ValorAnterior", "QtdAtual", "QtdAnterior"]:
        clientes[c] = pd.to_numeric(clientes.get(c, 0.0), errors="coerce").fillna(0.0)

    clientes["Estado"] = clientes["EstadoAtual"].combine_first(clientes["EstadoAnterior"]).fillna("")
    clientes["Cidade"] = clientes["CidadeAtual"].combine_first(clientes["CidadeAnterior"]).fillna("")

    clientes[STATUS_COL] = clientes.apply(lambda r: classify_status(float(r["ValorAtual"]), float(r["ValorAnterior"])), axis=1)
    clientes = clientes[(clientes["ValorAtual"] > 0) | (clientes["ValorAnterior"] > 0)]
    return clientes[["Cliente", "Estado", "Cidade", "ValorAtual", "ValorAnterior", "QtdAtual", "QtdAnterior", STATUS_COL]]


def build_evolucao(df_rep_curr: pd.DataFrame, df_rep_prev: pd.DataFrame) -> Dict[str, List[dict]]:
    def ts(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=["Competencia", "Valor", "Quantidade"])
        out = df_in.groupby("Competencia", as_index=False)[["Valor", "Quantidade"]].sum().sort_values("Competencia")
        out["label"] = out["Competencia"].apply(lambda d: f"{MONTH_MAP_NUM_TO_NAME[d.month]} {str(d.year)[2:]}")
        return out

    a = ts(df_rep_curr)
    b = ts(df_rep_prev)

    return {
        "curr": [{"label": r["label"], "valor": safe_float(r["Valor"]), "qtd": safe_float(r["Quantidade"])} for _, r in a.iterrows()],
        "prev": [{"label": r["label"], "valor": safe_float(r["Valor"]), "qtd": safe_float(r["Quantidade"])} for _, r in b.iterrows()],
    }


def build_categorias(df_rep_curr: pd.DataFrame, df_rep_prev: pd.DataFrame) -> Dict[str, List[dict]]:
    def agg(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=["Categoria", "Valor"])
        return df_in.groupby("Categoria", as_index=False)["Valor"].sum().rename(columns={"Valor": "Valor"})

    curr = agg(df_rep_curr)
    prev = agg(df_rep_prev)

    # share within each period
    curr_total = float(curr["Valor"].sum()) if not curr.empty else 0.0
    prev_total = float(prev["Valor"].sum()) if not prev.empty else 0.0
    curr["Share"] = curr["Valor"] / (curr_total if curr_total > 0 else 1.0)
    prev["Share"] = prev["Valor"] / (prev_total if prev_total > 0 else 1.0)

    curr = curr.sort_values("Valor", ascending=False)
    prev = prev.sort_values("Valor", ascending=False)

    return {
        "curr": [{"categoria": str(r["Categoria"]), "valor": safe_float(r["Valor"]), "share": safe_float(r["Share"])} for _, r in curr.iterrows()],
        "prev": [{"categoria": str(r["Categoria"]), "valor": safe_float(r["Valor"]), "share": safe_float(r["Share"])} for _, r in prev.iterrows()],
    }


def build_map_cidades(df_rep_curr: pd.DataFrame, df_geo: pd.DataFrame) -> List[dict]:
    if df_rep_curr.empty:
        return []

    df_cities = df_rep_curr.groupby(["Estado", "Cidade"], as_index=False).agg(
        Valor=("Valor", "sum"),
        Quantidade=("Quantidade", "sum"),
        Clientes=("Cliente", "nunique"),
    )
    df_cities["key"] = (
        df_cities["Estado"].astype(str).str.strip().str.upper()
        + "|"
        + df_cities["Cidade"].astype(str).str.strip().str.upper()
    )
    df_map = df_cities.merge(df_geo, on="key", how="inner")
    if df_map.empty:
        return []

    # default metric is Volume; we still ship both values
    out = []
    for _, r in df_map.iterrows():
        out.append({
            "cidade": str(r["Cidade"]),
            "estado": str(r["Estado"]),
            "lat": safe_float(r["lat"]),
            "lon": safe_float(r["lon"]),
            "valor": safe_float(r["Valor"]),
            "qtd": safe_float(r["Quantidade"]),
            "clientes": safe_int(r["Clientes"]),
        })
    return out


def build_client_distribution(df_rep_curr: pd.DataFrame) -> Dict[str, object]:
    if df_rep_curr.empty:
        return {
            "kpis": {"n80": 0, "n80_ratio": 0.0, "hhi": 0.0, "top1": 0.0, "top3": 0.0, "top10": 0.0, "clientes": 0},
            "top15": [],
        }

    df_cli = (
        df_rep_curr.groupby(["Cliente", "Estado", "Cidade"], as_index=False)
        .agg(Valor=("Valor", "sum"), Quantidade=("Quantidade", "sum"))
        .sort_values("Valor", ascending=False)
    )
    total = float(df_cli["Valor"].sum())
    total = total if total > 0 else 1.0
    df_cli["share"] = df_cli["Valor"] / total

    clientes = int(df_cli["Cliente"].nunique())
    shares = df_cli["share"].values.tolist()

    # n80
    cum = 0.0
    n80 = 0
    for i, s in enumerate(shares, start=1):
        cum += s
        n80 = i
        if cum >= 0.8:
            break
    n80_ratio = (n80 / clientes) if clientes > 0 else 0.0

    # hhi
    hhi = float(sum([s * s for s in shares]))

    top1 = float(sum(shares[:1])) if len(shares) >= 1 else 0.0
    top3 = float(sum(shares[:3])) if len(shares) >= 3 else top1
    top10 = float(sum(shares[:10])) if len(shares) >= 10 else float(sum(shares))

    top15 = []
    for _, r in df_cli.head(15).iterrows():
        top15.append({
            "cliente": str(r["Cliente"]),
            "cidade": str(r["Cidade"]),
            "estado": str(r["Estado"]),
            "valor": safe_float(r["Valor"]),
            "qtd": safe_float(r["Quantidade"]),
            "share": safe_float(r["share"]),
        })

    return {
        "kpis": {"n80": n80, "n80_ratio": n80_ratio, "hhi": hhi, "top1": top1, "top3": top3, "top10": top10, "clientes": clientes},
        "top15": top15,
    }


def build_header_kpis(df_rep_curr: pd.DataFrame, df_rep_prev: pd.DataFrame, clientes_carteira: pd.DataFrame) -> Dict[str, object]:
    total_rep = float(df_rep_curr["Valor"].sum()) if not df_rep_curr.empty else 0.0
    total_vol = float(df_rep_curr["Quantidade"].sum()) if not df_rep_curr.empty else 0.0

    clientes_atendidos = int(df_rep_curr["Cliente"].nunique()) if not df_rep_curr.empty else 0
    cidades_atendidas = int(df_rep_curr[["Estado", "Cidade"]].dropna().drop_duplicates().shape[0]) if not df_rep_curr.empty else 0

    carteira_score, carteira_label = compute_carteira_score(clientes_carteira)

    # distribution quick (HHI label)
    dist = build_client_distribution(df_rep_curr)
    hhi = float(dist["kpis"]["hhi"])
    if hhi < 0.10:
        hhi_label = "Baixa"
    elif hhi < 0.20:
        hhi_label = "Moderada"
    else:
        hhi_label = "Alta"

    return {
        "total_valor": total_rep,
        "total_volume": total_vol,
        "clientes_atendidos": clientes_atendidos,
        "cidades_atendidas": cidades_atendidas,
        "carteira_score": carteira_score,
        "carteira_label": carteira_label,
        "hhi": hhi,
        "hhi_label": hhi_label,
        "n80": int(dist["kpis"]["n80"]),
        "n80_ratio": float(dist["kpis"]["n80_ratio"]),
    }


def build_report_for_rep(df_all: pd.DataFrame, df_geo: pd.DataFrame, rep: str, start: pd.Timestamp, end: pd.Timestamp, prev_start: pd.Timestamp, prev_end: pd.Timestamp) -> Dict[str, object]:
    df_rep_all = df_all if rep == "Todos" else df_all[df_all["Representante"] == rep]

    mask_curr = (df_rep_all["Competencia"] >= start) & (df_rep_all["Competencia"] <= end)
    mask_prev = (df_rep_all["Competencia"] >= prev_start) & (df_rep_all["Competencia"] <= prev_end)

    df_curr = df_rep_all.loc[mask_curr].copy()
    df_prev = df_rep_all.loc[mask_prev].copy()

    clientes_carteira = build_client_status(df_all, rep, start, end, prev_start, prev_end)

    payload = {
        "rep": rep,
        "period": {
            "curr_start": str(start.date()),
            "curr_end": str(end.date()),
            "prev_start": str(prev_start.date()),
            "prev_end": str(prev_end.date()),
            "curr_label": format_period_label(start, end),
            "prev_label": format_period_label(prev_start, prev_end),
        },
        "kpis": build_header_kpis(df_curr, df_prev, clientes_carteira),
        "evolucao": build_evolucao(df_curr, df_prev),
        "categorias": build_categorias(df_curr, df_prev),
        "mapa_cidades": build_map_cidades(df_curr, df_geo),
        "distribuicao_clientes": build_client_distribution(df_curr),
        "carteira_detalhes": build_carteira_breakdown(clientes_carteira),
        "status_clientes": build_status_lists(clientes_carteira),
    }
    return payload


def build_carteira_breakdown(clientes_df: pd.DataFrame) -> Dict[str, object]:
    if clientes_df is None or clientes_df.empty:
        return {"resumo": [], "total_clientes": 0}

    status_counts = (
        clientes_df.groupby(STATUS_COL)["Cliente"].nunique().reset_index()
        .rename(columns={"Cliente": "QtdClientes", STATUS_COL: "Status"})
    )

    fat_status = (
        clientes_df.groupby(STATUS_COL)[["ValorAtual", "ValorAnterior"]].sum().reset_index()
        .rename(columns={STATUS_COL: "Status"})
    )
    fat_status["DeltaValor"] = fat_status["ValorAtual"] - fat_status["ValorAnterior"]

    out = status_counts.merge(fat_status[["Status", "DeltaValor"]], on="Status", how="left")
    total_clientes = int(out["QtdClientes"].sum())
    out["PctClientes"] = out["QtdClientes"] / (total_clientes if total_clientes > 0 else 1)

    out["Status"] = pd.Categorical(out["Status"], categories=STATUS_ORDER, ordered=True)
    out = out.sort_values("Status")

    return {
        "total_clientes": total_clientes,
        "resumo": [
            {
                "status": str(r["Status"]),
                "clientes": safe_int(r["QtdClientes"]),
                "pct_clientes": safe_float(r["PctClientes"]),
                "delta_valor": safe_float(r["DeltaValor"]),
            }
            for _, r in out.iterrows()
        ],
    }


def build_status_lists(clientes_df: pd.DataFrame) -> Dict[str, List[dict]]:
    if clientes_df is None or clientes_df.empty:
        return {k: [] for k in STATUS_ORDER}

    # add deltas and pct changes (valor and volume)
    df = clientes_df.copy()
    df["DeltaValor"] = df["ValorAtual"] - df["ValorAnterior"]
    df["DeltaQtd"] = df["QtdAtual"] - df["QtdAnterior"]

    def pct(curr, prev):
        if prev > 0:
            return (curr - prev) / prev
        if curr > 0 and prev == 0:
            return None
        return 0.0

    df["PctValor"] = df.apply(lambda r: pct(float(r["ValorAtual"]), float(r["ValorAnterior"])), axis=1)
    df["PctQtd"] = df.apply(lambda r: pct(float(r["QtdAtual"]), float(r["QtdAnterior"])), axis=1)

    out: Dict[str, List[dict]] = {k: [] for k in STATUS_ORDER}
    for status in STATUS_ORDER:
        dfx = df[df[STATUS_COL] == status].copy()
        if dfx.empty:
            continue
        dfx = dfx.sort_values("ValorAtual", ascending=False)
        for _, r in dfx.iterrows():
            out[status].append({
                "cliente": str(r["Cliente"]),
                "estado": str(r["Estado"]),
                "cidade": str(r["Cidade"]),
                "valor_atual": safe_float(r["ValorAtual"]),
                "valor_anterior": safe_float(r["ValorAnterior"]),
                "qtd_atual": safe_float(r["QtdAtual"]),
                "qtd_anterior": safe_float(r["QtdAnterior"]),
                "delta_valor": safe_float(r["DeltaValor"]),
                "delta_qtd": safe_float(r["DeltaQtd"]),
                "pct_valor": r["PctValor"] if r["PctValor"] is None or not (isinstance(r["PctValor"], float) and math.isnan(r["PctValor"])) else None,
                "pct_qtd": r["PctQtd"] if r["PctQtd"] is None or not (isinstance(r["PctQtd"], float) and math.isnan(r["PctQtd"])) else None,
                "status": status,
            })
    return out


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    df = load_relatorio()
    df_geo = load_geo()

    anos = sorted(df["Ano"].dropna().unique().tolist())
    if not anos:
        raise ValueError("Não foi possível identificar anos no relatorio_faturamento.csv")

    last_year = int(anos[-1])
    meses_last_year = df.loc[df["Ano"] == last_year, "MesNum"].dropna().unique().tolist()
    if meses_last_year:
        start_m = int(min(meses_last_year))
        end_m = int(max(meses_last_year))
    else:
        start_m, end_m = 1, 12

    start = pd.Timestamp(year=last_year, month=start_m, day=1)
    end = pd.Timestamp(year=last_year, month=end_m, day=1)
    prev_start, prev_end = previous_window(start, end)

    reps = sorted(df["Representante"].dropna().unique().tolist())
    rep_list = ["Todos"] + reps

    reports = []
    for rep in rep_list:
        reports.append(build_report_for_rep(df, df_geo, rep, start, end, prev_start, prev_end))

    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_files": {
                "relatorio_faturamento": str(RELATORIO_FATURAMENTO.relative_to(REPO_ROOT)),
                "cidades_geo": str(CIDADES_GEO.relative_to(REPO_ROOT)),
            },
            "default_period": {
                "curr_start": str(start.date()),
                "curr_end": str(end.date()),
                "prev_start": str(prev_start.date()),
                "prev_end": str(prev_end.date()),
                "curr_label": format_period_label(start, end),
                "prev_label": format_period_label(prev_start, prev_end),
            },
        },
        "reports": reports,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"OK: gerado {OUT_JSON} ({len(reports)} reps)")

    # sanity: show first rep points count
    first = reports[0]
    print("exemplo:", first["rep"], "pontos no mapa:", len(first.get("mapa_cidades", [])))


if __name__ == "__main__":
    main()
