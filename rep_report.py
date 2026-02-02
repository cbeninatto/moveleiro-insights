# rep_report.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# pydeck optional
try:
    import pydeck as pdk
    HAS_PYDECK = True
except Exception:
    HAS_PYDECK = False

st.set_page_config(page_title="Relatório do Representante", layout="wide")

# ============================================================
# CONFIG
# ============================================================
DEFAULT_PATH = "data/raw/relatorio_faturamento.csv"  # <-- repo relative path

COL = {
    "date": "data",
    "rep": "representante",
    "client": "cliente",
    "city": "cidade",
    "state": "estado",
    "category": "categoria",
    "rev": "faturamento",
    "vol": "volume",
    # optional map
    "lat": "lat",
    "lon": "lon",
}

# ============================================================
# HELPERS
# ============================================================
def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def previous_period(start: pd.Timestamp, end: pd.Timestamp):
    days = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start, prev_end

def filter_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df[COL["date"]] >= start) & (df[COL["date"]] <= end)].copy()

def pct_change(curr, prev):
    if prev == 0 or pd.isna(prev):
        return np.nan
    return (curr - prev) / prev

def money(x):
    if pd.isna(x):
        return "—"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def num(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%".replace(".", ",")

def validate_required_columns(df: pd.DataFrame) -> list[str]:
    required = [
        COL["date"], COL["rep"], COL["client"], COL["city"], COL["state"],
        COL["category"], COL["rev"], COL["vol"]
    ]
    return [c for c in required if c not in df.columns]

@st.cache_data(show_spinner=False)
def load_data(path: str, csv_encoding: str = "utf-8", csv_sep: str | None = None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()

    # running inside repo (Streamlit Cloud): path must be relative and exist
    if not os.path.exists(path):
        return pd.DataFrame()

    ext = os.path.splitext(path.lower())[1]
    try:
        if ext == ".csv":
            # sep=None -> autodetect ; vs ,
            return pd.read_csv(path, sep=(csv_sep or None), encoding=csv_encoding, engine="python")
        elif ext == ".parquet":
            return pd.read_parquet(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ============================================================
# CORE CALCS
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
        return empty, 0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan

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
    return m.rename(columns={COL["client"]: "cliente"})

def carteira_health_score(status_df: pd.DataFrame):
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
    return float(np.clip(score, 0, 100)), counts, share

# ============================================================
# APP
# ============================================================
st.title("Relatório do Representante")

with st.sidebar:
    st.header("Fonte de dados (repo)")
    st.caption("No Streamlit Cloud, use caminho relativo dentro do repo.")
    data_path = st.text_input("Caminho do arquivo", value=DEFAULT_PATH)
    csv_sep = st.text_input("CSV sep (vazio = auto)", value="")
    csv_encoding = st.text_input("CSV encoding", value="utf-8")

    compare_prev = st.checkbox("Comparar com período anterior (mesma duração)", value=True)

df = load_data(
    data_path,
    csv_encoding=(csv_encoding.strip() or "utf-8"),
    csv_sep=(csv_sep.strip() or None),
)

if df.empty:
    st.error("Não consegui carregar o arquivo do repo.")
    st.write("Caminho informado:", data_path)
    st.write("Existe no container?:", os.path.exists(data_path))
    st.info("Confirme que o arquivo está no repo nesse caminho: data/raw/relatorio_faturamento.csv")
    st.stop()

missing = validate_required_columns(df)
if missing:
    st.error("Colunas mínimas não encontradas (ou COL mapping está diferente).")
    st.write("Faltando:", missing)
    st.write("Encontradas:", list(df.columns))
    st.stop()

df = ensure_datetime(df, COL["date"])
if df[COL["date"]].isna().all():
    st.error(f"A coluna '{COL['date']}' não está parseando como data.")
    st.stop()

with st.expander("Preview", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

reps = sorted(df[COL["rep"]].dropna().unique().tolist())
if not reps:
    st.error(f"Sem representantes na coluna '{COL['rep']}'.")
    st.stop()

with st.sidebar:
    st.header("Filtros")
    rep = st.selectbox("Representante", reps)

    min_d = df[COL["date"]].min()
    max_d = df[COL["date"]].max()

    start = st.date_input("Início", value=(max_d - pd.Timedelta(days=365)).date(), min_value=min_d.date(), max_value=max_d.date())
    end = st.date_input("Fim", value=max_d.date(), min_value=min_d.date(), max_value=max_d.date())

start = pd.to_datetime(start)
end = pd.to_datetime(end)
if start > end:
    st.error("Início não pode ser maior que o fim.")
    st.stop()

prev_start, prev_end = previous_period(start, end)

df_rep = df[df[COL["rep"]] == rep].copy()
df_curr = filter_period(df_rep, start, end)
df_prev = filter_period(df_rep, prev_start, prev_end) if compare_prev else df_rep.iloc[0:0].copy()

dist_df, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10 = client_distribution(df_curr)
status_df = build_client_status(df_curr, df_prev, thr=0.10)
health_score, health_counts, health_share = carteira_health_score(status_df)

curr_rev = float(df_curr[COL["rev"]].fillna(0).sum())
curr_vol = float(df_curr[COL["vol"]].fillna(0).sum())
prev_rev = float(df_prev[COL["rev"]].fillna(0).sum()) if compare_prev else np.nan
prev_vol = float(df_prev[COL["vol"]].fillna(0).sum()) if compare_prev else np.nan

clients_attended = int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["client"]].nunique())
cities_attended = int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["city"]].nunique())

# 1) Performance Dashboard
st.subheader("1. Performance Dashboard")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Faturamento Total", money(curr_rev), pct(pct_change(curr_rev, prev_rev)) if compare_prev else None)
c2.metric("Volume Total", num(curr_vol), pct(pct_change(curr_vol, prev_vol)) if compare_prev else None)
c3.metric("N80 (qtd clientes)", f"{n80_count}", pct(n80_share_clients))
c4.metric("Saúde da Carteira (0-100)", f"{health_score:.0f}")
c5.metric("Clientes Atendidos", f"{clients_attended}")
c6.metric("Cidades Atendidas", f"{cities_attended}")

# 2) Evolução
st.subheader("2. Evolução (comparativo)")
df_curr_m = df_curr.copy()
df_curr_m["mes"] = df_curr_m[COL["date"]].dt.to_period("M").dt.to_timestamp()
evo_curr = df_curr_m.groupby("mes", as_index=False).agg(faturamento=(COL["rev"], "sum"), volume=(COL["vol"], "sum"))
evo_curr["periodo"] = "Selecionado"

if compare_prev:
    df_prev_m = df_prev.copy()
    df_prev_m["mes"] = df_prev_m[COL["date"]].dt.to_period("M").dt.to_timestamp()
    evo_prev = df_prev_m.groupby("mes", as_index=False).agg(faturamento=(COL["rev"], "sum"), volume=(COL["vol"], "sum"))
    evo_prev["periodo"] = "Anterior"
    evo = pd.concat([evo_curr, evo_prev], ignore_index=True)
else:
    evo = evo_curr

colA, colB = st.columns([2, 1])

with colA:
    st.markdown("**Faturamento x Volume (mensal)**")
    base = alt.Chart(evo).encode(x=alt.X("mes:T", title="Mês"))
    line_rev = base.mark_line().encode(y=alt.Y("faturamento:Q", title="Faturamento"), color="periodo:N")
    line_vol = base.mark_line(strokeDash=[4, 2]).encode(y=alt.Y("volume:Q", title="Volume"), color="periodo:N")
    st.altair_chart((line_rev + line_vol).interactive(), use_container_width=True)

with colB:
    st.markdown("**Categorias vendidas (período selecionado)**")
    cat_curr = (
        df_curr.groupby(COL["category"], as_index=False)[COL["rev"]]
        .sum()
        .sort_values(COL["rev"], ascending=False)
        .head(12)
    )
    if cat_curr.empty:
        st.info("Sem categorias no período.")
    else:
        bar = alt.Chart(cat_curr).mark_bar().encode(
            x=alt.X(f"{COL['rev']}:Q", title="Faturamento"),
            y=alt.Y(f"{COL['category']}:N", sort="-x", title="Categoria"),
        )
        st.altair_chart(bar, use_container_width=True)

# 3) Mapa (safe)
st.subheader("3. Mapa de Clientes")
has_geo = (COL["lat"] in df_curr.columns) and (COL["lon"] in df_curr.columns)
if not has_geo:
    st.info("Sem colunas lat/lon no arquivo. (Mapa desativado)")
elif not HAS_PYDECK:
    st.info("pydeck não disponível neste ambiente.")
else:
    map_df = (
        df_curr.groupby([COL["client"], COL["city"], COL["state"], COL["lat"], COL["lon"]], as_index=False)
        .agg(faturamento=(COL["rev"], "sum"), volume=(COL["vol"], "sum"))
        .rename(columns={COL["client"]: "cliente", COL["city"]: "cidade", COL["state"]: "estado"})
        .dropna(subset=[COL["lat"], COL["lon"]])
    )
    if map_df.empty:
        st.info("Sem pontos válidos (lat/lon) no período.")
    else:
        view_state = pdk.ViewState(
            latitude=float(map_df[COL["lat"]].mean()),
            longitude=float(map_df[COL["lon"]].mean()),
            zoom=5,
            pitch=0,
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=[COL["lon"], COL["lat"]],
            get_radius=2000,
            pickable=True,
            auto_highlight=True,
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

# 4) Distribuição
st.subheader("4. Distribuição por clientes")
c7, c8, c9, c10 = st.columns(4)
c7.metric("Índice de Concentração (HHI)", f"{hhi:.4f}" if not pd.isna(hhi) else "—")
c8.metric("Top 1 cliente", pct(top1))
c9.metric("Top 3 clientes", pct(top3))
c10.metric("Top 10 clientes", pct(top10))

if dist_df.empty:
    st.info("Sem vendas no período para calcular distribuição.")
else:
    dshow = dist_df.copy()
    dshow["Faturamento"] = dshow["rev"].map(money)
    dshow["Share"] = dshow["share"].map(pct)
    dshow["Share acumulado"] = dshow["cum_share"].map(pct)
    st.dataframe(dshow[["cliente", "Faturamento", "Share", "Share acumulado"]], use_container_width=True)

# 5) Saúde
st.subheader("5. Saúde da carteira – Detalhes")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Novos", str(health_counts.get("Novos", 0)), pct(health_share.get("Novos", np.nan)))
d2.metric("Crescendo", str(health_counts.get("Crescendo", 0)), pct(health_share.get("Crescendo", np.nan)))
d3.metric("Estáveis", str(health_counts.get("Estáveis", 0)), pct(health_share.get("Estáveis", np.nan)))
d4.metric("Caindo", str(health_counts.get("Caindo", 0)), pct(health_share.get("Caindo", np.nan)))
d5.metric("Perdidos", str(health_counts.get("Perdidos", 0)), pct(health_share.get("Perdidos", np.nan)))

# 6) Status
st.subheader("6. Status dos clientes")
t = status_df.copy()
t["Fat (Período)"] = t["curr_rev"]
t["Fat (Anterior)"] = t["prev_rev"]
t["Δ Fat (R$)"] = t["delta_rev"]
t["Δ Fat (%)"] = t["pct_rev"]
t["Vol (Período)"] = t["curr_vol"]
t["Vol (Anterior)"] = t["prev_vol"]
t["Δ Vol"] = t["delta_vol"]
t["Δ Vol (%)"] = t["pct_vol"]

order = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
t["status"] = pd.Categorical(t["status"], categories=order, ordered=True)
t = t.sort_values(["status", "curr_rev"], ascending=[True, False]).reset_index(drop=True)

for s in order:
    sub = t[t["status"] == s].copy()
    st.markdown(f"### {s} ({len(sub)})")
    if sub.empty:
        st.caption("—")
        continue

    sub_disp = sub[[
        "cliente",
        "Fat (Período)", "Fat (Anterior)", "Δ Fat (R$)", "Δ Fat (%)",
        "Vol (Período)", "Vol (Anterior)", "Δ Vol", "Δ Vol (%)",
    ]].copy()

    for c in ["Fat (Período)", "Fat (Anterior)", "Δ Fat (R$)"]:
        sub_disp[c] = sub_disp[c].map(money)
    for c in ["Vol (Período)", "Vol (Anterior)", "Δ Vol"]:
        sub_disp[c] = sub_disp[c].map(num)
    for c in ["Δ Fat (%)", "Δ Vol (%)"]:
        sub_disp[c] = sub_disp[c].map(pct)

    st.dataframe(sub_disp, use_container_width=True)

if compare_prev:
    st.caption(f"Período selecionado: {start.date()} a {end.date()} | Período anterior: {prev_start.date()} a {prev_end.date()}")
else:
    st.caption(f"Período selecionado: {start.date()} a {end.date()}")
