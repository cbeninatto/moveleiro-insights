# report_generator.py
# Robust report generator:
# - Works even if matplotlib/fpdf2 are missing (falls back to console-only report)
# - No flet dependency
# - Fixes fpdf2 DeprecationWarning (no ln=)

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# ----------------------------
# Optional imports (safe)
# ----------------------------
HAS_MPL = True
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    HAS_MPL = False

HAS_FPDF = True
try:
    from fpdf import FPDF, XPos, YPos  # fpdf2
except Exception:
    HAS_FPDF = False


# ============================================================
# 1) CONFIG — adjust to your real column names
# ============================================================
COL = {
    "date": "data",
    "rep": "representante",
    "client": "cliente",
    "city": "cidade",
    "state": "estado",
    "category": "categoria",
    "rev": "faturamento",
    "vol": "volume",
}

DEFAULT_INPUT = r"C:\Users\Cesar\CB Database\Documents\OPENFIELD\APPS\INSIGHTS\data\base_final.parquet"
DEFAULT_OUTDIR = r"C:\Users\Cesar\CB Database\Documents\OPENFIELD\APPS\INSIGHTS\reports"


# ============================================================
# 2) Helpers
# ============================================================
def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def previous_period(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    days = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start, prev_end


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%".replace(".", ",")


def pct_change(curr: float, prev: float) -> float:
    if prev == 0:
        return float("nan")
    return (curr - prev) / prev


def load_df(path: str, csv_sep: Optional[str] = None, csv_encoding: str = "utf-8") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    ext = os.path.splitext(path.lower())[1]
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        if csv_sep:
            df = pd.read_csv(path, sep=csv_sep, encoding=csv_encoding, engine="python")
        else:
            df = pd.read_csv(path, sep=None, encoding=csv_encoding, engine="python")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Extensão não suportada: {ext} (use .parquet/.csv/.xlsx)")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required = [COL["date"], COL["rep"], COL["client"], COL["city"], COL["state"], COL["category"], COL["rev"], COL["vol"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltam colunas no arquivo: {missing}\nColunas encontradas: {list(df.columns)}")


def filter_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df[COL["date"]] >= start) & (df[COL["date"]] <= end)].copy()


# ============================================================
# 3) Business logic — N80, status, score
# ============================================================
def client_distribution(df_period: pd.DataFrame):
    by_client = (
        df_period.groupby(COL["client"], as_index=False)[COL["rev"]]
        .sum()
        .rename(columns={COL["rev"]: "rev", COL["client"]: "cliente"})
    )

    by_client["rev"] = by_client["rev"].fillna(0.0)
    total = float(by_client["rev"].sum())

    if total <= 0 or by_client.empty:
        empty = pd.DataFrame(columns=["cliente", "rev", "share", "cum_share"])
        return empty, 0, 0.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan")

    by_client = by_client.sort_values("rev", ascending=False).reset_index(drop=True)
    by_client["share"] = by_client["rev"] / total
    by_client["cum_share"] = by_client["share"].cumsum()

    n80_count = int((by_client["cum_share"] <= 0.80).sum())
    if n80_count == 0:
        n80_count = 1
    n80_rev_share = float(by_client.loc[n80_count - 1, "cum_share"])
    n80_share_clients = n80_count / len(by_client)

    hhi = float((by_client["share"] ** 2).sum())
    top1 = float(by_client["share"].head(1).sum())
    top3 = float(by_client["share"].head(3).sum())
    top10 = float(by_client["share"].head(10).sum())

    return by_client, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10


def build_client_status(df_curr: pd.DataFrame, df_prev: pd.DataFrame, thr: float = 0.10) -> pd.DataFrame:
    curr = df_curr.groupby(COL["client"], as_index=False).agg(
        curr_rev=(COL["rev"], "sum"),
        curr_vol=(COL["vol"], "sum"),
    )
    prev = df_prev.groupby(COL["client"], as_index=False).agg(
        prev_rev=(COL["rev"], "sum"),
        prev_vol=(COL["vol"], "sum"),
    )

    m = curr.merge(prev, on=COL["client"], how="outer").fillna(0.0)
    m["delta_rev"] = m["curr_rev"] - m["prev_rev"]
    m["delta_vol"] = m["curr_vol"] - m["prev_vol"]

    # pct changes (avoid division by zero)
    m["pct_rev"] = m.apply(lambda r: pct_change(r["curr_rev"], r["prev_rev"]), axis=1)
    m["pct_vol"] = m.apply(lambda r: pct_change(r["curr_vol"], r["prev_vol"]), axis=1)

    prev_r = m["prev_rev"].to_numpy()
    curr_r = m["curr_rev"].to_numpy()

    status = np.full(len(m), "Estáveis", dtype=object)
    status[(prev_r == 0) & (curr_r > 0)] = "Novos"
    status[(prev_r > 0) & (curr_r == 0)] = "Perdidos"
    status[(prev_r > 0) & (curr_r > prev_r * (1 + thr))] = "Crescendo"
    status[(prev_r > 0) & (curr_r < prev_r * (1 - thr))] = "Caindo"

    m["status"] = status
    m = m.rename(columns={COL["client"]: "cliente"})
    return m


def carteira_health_score(status_df: pd.DataFrame) -> Tuple[float, dict, dict]:
    counts = status_df["status"].value_counts().to_dict()
    total = max(1, len(status_df))
    share = {k: counts.get(k, 0) / total for k in ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]}

    score = (
        50
        + 25 * share["Crescendo"]
        + 10 * share["Novos"]
        + 15 * share["Estáveis"]
        - 25 * share["Caindo"]
        - 40 * share["Perdidos"]
    )
    score = float(np.clip(score, 0, 100))
    return score, counts, share


# ============================================================
# 4) Charts (optional)
# ============================================================
def save_evolution_chart(df_curr: pd.DataFrame, df_prev: pd.DataFrame, outpath: str, compare: bool) -> Optional[str]:
    if not HAS_MPL:
        return None

    # Monthly aggregation
    def month_agg(d: pd.DataFrame) -> pd.DataFrame:
        tmp = d.copy()
        tmp["mes"] = tmp[COL["date"]].dt.to_period("M").dt.to_timestamp()
        return tmp.groupby("mes", as_index=False).agg(
            faturamento=(COL["rev"], "sum"),
            volume=(COL["vol"], "sum"),
        )

    evo_curr = month_agg(df_curr)
    evo_prev = month_agg(df_prev) if compare else pd.DataFrame(columns=["mes", "faturamento", "volume"])

    plt.figure()
    plt.plot(evo_curr["mes"], evo_curr["faturamento"], label="Fat (selecionado)")
    if compare and not evo_prev.empty:
        plt.plot(evo_prev["mes"], evo_prev["faturamento"], label="Fat (anterior)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


# ============================================================
# 5) PDF (optional) — with new_x/new_y (no ln=)
# ============================================================
def pdf_cell_ln(pdf: "FPDF", w: float, h: float, txt: str, bold: bool = False):
    if bold:
        pdf.set_font("Helvetica", "B", 12)
    else:
        pdf.set_font("Helvetica", "", 11)
    pdf.cell(w, h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def generate_pdf(
    out_pdf: str,
    rep_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prev_start: pd.Timestamp,
    prev_end: pd.Timestamp,
    compare: bool,
    metrics: dict,
    dist_df: pd.DataFrame,
    status_df: pd.DataFrame,
    evolution_chart_path: Optional[str],
):
    if not HAS_FPDF:
        raise RuntimeError("fpdf2 não está instalado. Instale com: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Relatório do Representante", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Representante: {rep_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(
        0, 6,
        f"Período: {start.strftime('%d/%m/%Y')} a {end.strftime('%d/%m/%Y')}",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    if compare:
        pdf.cell(
            0, 6,
            f"Período anterior: {prev_start.strftime('%d/%m/%Y')} a {prev_end.strftime('%d/%m/%Y')}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )

    pdf.ln(2)

    # 1. Performance
    pdf_cell_ln(pdf, 0, 8, "1. Performance Geral", bold=True)
    pdf_cell_ln(pdf, 0, 6, f"Faturamento total: {fmt_money(metrics['curr_rev'])}")
    pdf_cell_ln(pdf, 0, 6, f"Volume total: {fmt_num(metrics['curr_vol'])}")
    if compare:
        pdf_cell_ln(pdf, 0, 6, f"Variação faturamento vs anterior: {fmt_pct(metrics['rev_chg'])}")
        pdf_cell_ln(pdf, 0, 6, f"Variação volume vs anterior: {fmt_pct(metrics['vol_chg'])}")
    pdf_cell_ln(pdf, 0, 6, f"Clientes atendidos: {metrics['clients_attended']}")
    pdf_cell_ln(pdf, 0, 6, f"Cidades atendidas: {metrics['cities_attended']}")
    pdf_cell_ln(pdf, 0, 6, f"N80: {metrics['n80_count']} clientes ({fmt_pct(metrics['n80_share_clients'])} da carteira) = {fmt_pct(metrics['n80_rev_share'])} do faturamento")
    pdf_cell_ln(pdf, 0, 6, f"Índice de concentração (HHI): {metrics['hhi']:.4f}" if not pd.isna(metrics["hhi"]) else "Índice de concentração (HHI): —")
    pdf_cell_ln(pdf, 0, 6, f"Top 1 / 3 / 10: {fmt_pct(metrics['top1'])} | {fmt_pct(metrics['top3'])} | {fmt_pct(metrics['top10'])}")
    pdf_cell_ln(pdf, 0, 6, f"Saúde da carteira (0-100): {metrics['health_score']:.0f}")

    pdf.ln(2)

    # 2. Evolução
    pdf_cell_ln(pdf, 0, 8, "2. Evolução Mensal", bold=True)
    if evolution_chart_path and os.path.exists(evolution_chart_path):
        pdf.image(evolution_chart_path, w=180)  # auto height
        pdf.ln(2)
    else:
        pdf_cell_ln(pdf, 0, 6, "Gráfico indisponível (matplotlib não instalado ou sem dados).")

    pdf.ln(2)

    # 3. Top clientes (Pareto)
    pdf_cell_ln(pdf, 0, 8, "3. Top 15 Clientes (Pareto)", bold=True)
    top = dist_df.head(15).copy()
    if top.empty:
        pdf_cell_ln(pdf, 0, 6, "Sem vendas no período.")
    else:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(90, 6, "Cliente", border=1)
        pdf.cell(45, 6, "Faturamento", border=1)
        pdf.cell(45, 6, "Share", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        for _, r in top.iterrows():
            pdf.cell(90, 6, str(r["cliente"])[:45], border=1)
            pdf.cell(45, 6, fmt_money(float(r["rev"])), border=1)
            pdf.cell(45, 6, fmt_pct(float(r["share"])), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)

    # 4. Status dos clientes
    pdf_cell_ln(pdf, 0, 8, "4. Detalhes da Carteira (Status)", bold=True)
    order = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
    status_df = status_df.copy()
    status_df["status"] = pd.Categorical(status_df["status"], categories=order, ordered=True)
    status_df = status_df.sort_values(["status", "curr_rev"], ascending=[True, False]).reset_index(drop=True)

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(55, 6, "Cliente", border=1)
    pdf.cell(20, 6, "Status", border=1)
    pdf.cell(28, 6, "Fat (P)", border=1)
    pdf.cell(28, 6, "Fat (A)", border=1)
    pdf.cell(18, 6, "Δ%", border=1)
    pdf.cell(31, 6, "Vol (P)", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 9)
    for _, r in status_df.iterrows():
        pdf.cell(55, 6, str(r["cliente"])[:30], border=1)
        pdf.cell(20, 6, str(r["status"])[:10], border=1)
        pdf.cell(28, 6, fmt_money(float(r["curr_rev"])), border=1)
        pdf.cell(28, 6, fmt_money(float(r["prev_rev"])), border=1)
        pdf.cell(18, 6, fmt_pct(float(r["pct_rev"])) if not pd.isna(r["pct_rev"]) else "—", border=1)
        pdf.cell(31, 6, fmt_num(float(r["curr_vol"])), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Basic page break handling (fpdf also auto-breaks)
        if pdf.get_y() > 270:
            pdf.add_page()

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    pdf.output(out_pdf)


# ============================================================
# 6) Console fallback (always available)
# ============================================================
def print_console_report(rep_name: str, start: pd.Timestamp, end: pd.Timestamp, prev_start: pd.Timestamp, prev_end: pd.Timestamp, compare: bool, metrics: dict):
    print("=" * 70)
    print("RELATÓRIO DO REPRESENTANTE")
    print(f"Representante: {rep_name}")
    print(f"Período: {start.date()} a {end.date()}")
    if compare:
        print(f"Anterior: {prev_start.date()} a {prev_end.date()}")
    print("-" * 70)
    print(f"Faturamento total: {fmt_money(metrics['curr_rev'])}")
    print(f"Volume total:      {fmt_num(metrics['curr_vol'])}")
    if compare:
        print(f"Variação fat:      {fmt_pct(metrics['rev_chg'])}")
        print(f"Variação vol:      {fmt_pct(metrics['vol_chg'])}")
    print(f"Clientes atendidos:{metrics['clients_attended']}")
    print(f"Cidades atendidas: {metrics['cities_attended']}")
    print(f"N80: {metrics['n80_count']} clientes ({fmt_pct(metrics['n80_share_clients'])} da carteira) = {fmt_pct(metrics['n80_rev_share'])} do faturamento")
    print(f"HHI: {metrics['hhi']:.4f}" if not pd.isna(metrics["hhi"]) else "HHI: —")
    print(f"Top 1 / 3 / 10: {fmt_pct(metrics['top1'])} | {fmt_pct(metrics['top3'])} | {fmt_pct(metrics['top10'])}")
    print(f"Saúde da carteira (0-100): {metrics['health_score']:.0f}")
    print("=" * 70)


# ============================================================
# 7) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Gera PDF de relatório por representante (robusto).")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Caminho do arquivo final (parquet/csv/xlsx).")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Pasta de saída para PDFs.")
    parser.add_argument("--rep", default="", help="Nome exato do representante (vazio = primeiro encontrado).")
    parser.add_argument("--start", default="", help="Data início (YYYY-MM-DD). Vazio = 365 dias atrás.")
    parser.add_argument("--end", default="", help="Data fim (YYYY-MM-DD). Vazio = max data do dataset.")
    parser.add_argument("--no-compare", action="store_true", help="Não comparar com período anterior.")
    parser.add_argument("--csv-sep", default="", help="Separador CSV (ex: ;). Vazio = auto.")
    parser.add_argument("--csv-enc", default="utf-8", help="Encoding CSV (ex: latin1).")

    args = parser.parse_args()

    df = load_df(args.input, csv_sep=(args.csv_sep.strip() or None), csv_encoding=args.csv_enc.strip() or "utf-8")
    validate_columns(df)
    df = ensure_datetime(df, COL["date"])

    if df[COL["date"]].isna().all():
        raise ValueError(f"Coluna de data '{COL['date']}' virou NaT. Ajuste COL['date'] ou o formato no arquivo.")

    # Resolve dates
    max_d = df[COL["date"]].max()
    min_d = df[COL["date"]].min()

    if args.end:
        end = pd.to_datetime(args.end)
    else:
        end = pd.to_datetime(max_d)

    if args.start:
        start = pd.to_datetime(args.start)
    else:
        start = pd.to_datetime(max(min_d, end - pd.Timedelta(days=365)))

    if start > end:
        raise ValueError("start > end. Ajuste as datas.")

    prev_start, prev_end = previous_period(start, end)
    compare = not args.no_compare

    # Resolve rep
    reps = sorted(df[COL["rep"]].dropna().unique().tolist())
    if not reps:
        raise ValueError(f"Sem representantes na coluna '{COL['rep']}'.")

    rep_name = args.rep.strip() or reps[0]
    if rep_name not in reps:
        raise ValueError(f"Representante '{rep_name}' não encontrado. Exemplos: {reps[:10]}")

    df_rep = df[df[COL["rep"]] == rep_name].copy()
    df_curr = filter_period(df_rep, start, end)
    df_prev = filter_period(df_rep, prev_start, prev_end) if compare else df_rep.iloc[0:0].copy()

    curr_rev = float(df_curr[COL["rev"]].fillna(0).sum())
    curr_vol = float(df_curr[COL["vol"]].fillna(0).sum())
    prev_rev = float(df_prev[COL["rev"]].fillna(0).sum()) if compare else 0.0
    prev_vol = float(df_prev[COL["vol"]].fillna(0).sum()) if compare else 0.0

    dist_df, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10 = client_distribution(df_curr)
    status_df = build_client_status(df_curr, df_prev if compare else df_rep.iloc[0:0].copy(), thr=0.10)
    health_score, _, _ = carteira_health_score(status_df)

    metrics = {
        "curr_rev": curr_rev,
        "curr_vol": curr_vol,
        "rev_chg": pct_change(curr_rev, prev_rev) if compare else float("nan"),
        "vol_chg": pct_change(curr_vol, prev_vol) if compare else float("nan"),
        "clients_attended": int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["client"]].nunique()),
        "cities_attended": int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["city"]].nunique()),
        "n80_count": n80_count,
        "n80_share_clients": n80_share_clients,
        "n80_rev_share": n80_rev_share,
        "hhi": hhi,
        "top1": top1,
        "top3": top3,
        "top10": top10,
        "health_score": health_score,
    }

    # Always print console summary
    print_console_report(rep_name, start, end, prev_start, prev_end, compare, metrics)

    # Optional chart
    chart_path = None
    if HAS_MPL:
        os.makedirs(args.outdir, exist_ok=True)
        chart_path = os.path.join(args.outdir, f"evolucao_{rep_name[:30].replace(' ', '_')}.png")
        chart_path = save_evolution_chart(df_curr, df_prev, chart_path, compare)

    # PDF output if fpdf2 available
    out_pdf = os.path.join(
        args.outdir,
        f"Relatorio_{rep_name[:50].replace(' ', '_')}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.pdf"
    )

    if HAS_FPDF:
        generate_pdf(
            out_pdf=out_pdf,
            rep_name=rep_name,
            start=start,
            end=end,
            prev_start=prev_start,
            prev_end=prev_end,
            compare=compare,
            metrics=metrics,
            dist_df=dist_df,
            status_df=status_df,
            evolution_chart_path=chart_path,
        )
        print(f"\nPDF gerado em: {out_pdf}")
    else:
        print("\n[INFO] fpdf2 não instalado — pulando geração de PDF.")
        print("       Para gerar PDF: pip install fpdf2")
        if not HAS_MPL:
            print("       Para gráficos:  pip install matplotlib")

if __name__ == "__main__":
    main()
