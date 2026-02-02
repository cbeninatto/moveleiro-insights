# rep_report.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Relatório do Representante", layout="wide")

# -----------------------------
# 0) Column mapping (edit here)
# -----------------------------
COL = {
    "date": "data",
    "rep": "representante",
    "client": "cliente",
    "city": "cidade",
    "state": "estado",
    "category": "categoria",
    "rev": "faturamento",
    "vol": "volume",
    "lat": "lat",   # optional
    "lon": "lon",   # optional
}

# -----------------------------
# 1) Helpers
# -----------------------------
def safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)

def pct_change(curr, prev):
    # returns % change; if prev==0 and curr>0 -> inf-like, we keep NaN and handle display
    return safe_div(curr - prev, prev)

def money(x):
    if pd.isna(x):
        return ""
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def num(x):
    if pd.isna(x):
        return ""
    return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(x):
    if pd.isna(x):
        return ""
    return f"{x*100:.1f}%".replace(".", ",")

def ensure_datetime(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def load_data():
    """
    Replace this with your current insights.py data loader.
    Must return a dataframe with columns mapped in COL.
    """
    # Example placeholder:
    # return pd.read_parquet("data/fato_vendas.parquet")
    return pd.DataFrame()

def filter_period(df, start, end):
    d = df[(df[COL["date"]] >= start) & (df[COL["date"]] <= end)].copy()
    return d

def previous_period(start, end):
    # previous period with same length immediately before start
    days = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start, prev_end

# -----------------------------
# 2) Core metrics
# -----------------------------
def client_distribution(df_rep_period):
    """
    Returns:
      - dist_df: client revenue share sorted desc
      - n80_count, n80_share_clients, n80_rev_share
      - hhi, top1, top3, top10
    """
    by_client = (
        df_rep_period.groupby(COL["client"], as_index=False)[COL["rev"]]
        .sum()
        .rename(columns={COL["rev"]: "rev"})
    )
    by_client["rev"] = by_client["rev"].fillna(0.0)
    total = by_client["rev"].sum()

    if total <= 0 or by_client.empty:
        empty = pd.DataFrame(columns=[COL["client"], "rev", "share", "cum_share"])
        return empty, 0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan

    by_client = by_client.sort_values("rev", ascending=False).reset_index(drop=True)
    by_client["share"] = by_client["rev"] / total
    by_client["cum_share"] = by_client["share"].cumsum()

    # N80: minimal clients to reach 80% revenue
    n80_count = int((by_client["cum_share"] <= 0.80).sum())
    if n80_count == 0:
        n80_count = 1
    n80_rev_share = float(by_client.loc[n80_count - 1, "cum_share"])
    n80_share_clients = n80_count / len(by_client)

    # Concentration (HHI)
    hhi = float((by_client["share"] ** 2).sum())

    # Top shares
    top1 = float(by_client["share"].head(1).sum())
    top3 = float(by_client["share"].head(3).sum())
    top10 = float(by_client["share"].head(10).sum())

    dist_df = by_client.rename(columns={COL["client"]: "cliente"})
    return dist_df, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10

def build_client_status(df_curr, df_prev):
    """
    Status rules (simple & corporate-friendly):
      - NOVO: prev_rev==0 and curr_rev>0
      - PERDIDO: prev_rev>0 and curr_rev==0
      - CRESCENDO: curr_rev > prev_rev*(1+thr)
      - CAINDO: curr_rev < prev_rev*(1-thr)
      - ESTÁVEL: otherwise (including both zero -> stable)
    """
    thr = 0.10  # 10% band
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
    m["pct_rev"] = pct_change(m["curr_rev"], m["prev_rev"])
    m["pct_vol"] = pct_change(m["curr_vol"], m["prev_vol"])

    prev_r = m["prev_rev"].values
    curr_r = m["curr_rev"].values

    status = np.full(len(m), "Estáveis", dtype=object)
    status[(prev_r == 0) & (curr_r > 0)] = "Novos"
    status[(prev_r > 0) & (curr_r == 0)] = "Perdidos"
    status[(prev_r > 0) & (curr_r > prev_r * (1 + thr))] = "Crescendo"
    status[(prev_r > 0) & (curr_r < prev_r * (1 - thr))] = "Caindo"

    m["status"] = status
    m = m.rename(columns={COL["client"]: "cliente"})
    return m

def carteira_health_score(status_df):
    """
    Point system (0-100) using status mix.
    You can tune weights to match your business logic.
    """
    counts = status_df["status"].value_counts().to_dict()
    total = max(1, len(status_df))

    share = {k: counts.get(k, 0) / total for k in ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]}

    # weights: positives and negatives
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

# -----------------------------
# 3) UI
# -----------------------------
st.title("Relatório do Representante")

df = load_data()
if df.empty:
    st.warning("Seu loader está retornando um dataframe vazio. Substitua load_data() pelo loader do insights.py.")
    st.stop()

df = ensure_datetime(df, COL["date"])

# Sidebar filters
with st.sidebar:
    st.header("Filtros")
    reps = sorted(df[COL["rep"]].dropna().unique().tolist())
    rep = st.selectbox("Representante", reps)

    min_d = df[COL["date"]].min()
    max_d = df[COL["date"]].max()

    default_end = max_d
    default_start = max(min_d, max_d - pd.Timedelta(days=365))

    start = st.date_input("Início", value=default_start.date(), min_value=min_d.date(), max_value=max_d.date())
    end = st.date_input("Fim", value=default_end.date(), min_value=min_d.date(), max_value=max_d.date())
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    prev_start, prev_end = previous_period(start, end)
    show_prev = st.checkbox("Comparar com período anterior (mesma duração)", value=True)

df_rep = df[df[COL["rep"]] == rep].copy()
df_curr = filter_period(df_rep, start, end)
df_prev = filter_period(df_rep, prev_start, prev_end) if show_prev else df_curr.iloc[0:0].copy()

# Precompute
dist_df, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10 = client_distribution(df_curr)
status_df = build_client_status(df_curr, df_prev if show_prev else df_curr.iloc[0:0])
health_score, health_counts, health_share = carteira_health_score(status_df)

curr_rev = df_curr[COL["rev"]].sum()
curr_vol = df_curr[COL["vol"]].sum()
prev_rev = df_prev[COL["rev"]].sum() if show_prev else np.nan
prev_vol = df_prev[COL["vol"]].sum() if show_prev else np.nan

clients_attended = df_curr[df_curr[COL["rev"]] > 0][COL["client"]].nunique()
cities_attended = df_curr[df_curr[COL["rev"]] > 0][COL["city"]].nunique()

# -----------------------------
# 1) Performance Dashboard
# -----------------------------
st.subheader("1. Performance Dashboard")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Faturamento Total", money(curr_rev), pct(pct_change(curr_rev, prev_rev)) if show_prev else None)
c2.metric("Volume Total", num(curr_vol), pct(pct_change(curr_vol, prev_vol)) if show_prev else None)
c3.metric("N80 (qtd clientes)", f"{n80_count}", pct(n80_share_clients))
c4.metric("Saúde da Carteira (0-100)", f"{health_score:.0f}")
c5.metric("Clientes Atendidos", f"{clients_attended}")
c6.metric("Cidades Atendidas", f"{cities_attended}")

with st.expander("Resumo N80 / Concentração", expanded=False):
    st.write(
        f"- **N80**: {n80_count} clientes (**{pct(n80_share_clients)}** da carteira) "
        f"atingem **{pct(n80_rev_share)}** do faturamento.\n"
        f"- **Concentração (HHI)**: {hhi:.4f}\n"
        f"- **Top 1 / 3 / 10**: {pct(top1)} | {pct(top3)} | {pct(top10)}"
    )

# -----------------------------
# 2) Evolução - vs período anterior
# -----------------------------
st.subheader("2. Evolução")

# Monthly evolution
df_curr_m = df_curr.copy()
df_curr_m["mes"] = df_curr_m[COL["date"]].dt.to_period("M").dt.to_timestamp()
evo_curr = df_curr_m.groupby("mes", as_index=False).agg(
    faturamento=(COL["rev"], "sum"),
    volume=(COL["vol"], "sum"),
)
evo_curr["periodo"] = "Selecionado"

if show_prev:
    df_prev_m = df_prev.copy()
    df_prev_m["mes"] = df_prev_m[COL["date"]].dt.to_period("M").dt.to_timestamp()
    evo_prev = df_prev_m.groupby("mes", as_index=False).agg(
        faturamento=(COL["rev"], "sum"),
        volume=(COL["vol"], "sum"),
    )
    evo_prev["periodo"] = "Anterior"
    evo = pd.concat([evo_curr, evo_prev], ignore_index=True)
else:
    evo = evo_curr

colA, colB = st.columns([2, 1])

with colA:
    st.markdown("**Faturamento x Volume (mensal)**")
    base = alt.Chart(evo).encode(x=alt.X("mes:T", title="Mês"))
    line1 = base.mark_line().encode(y=alt.Y("faturamento:Q", title="Faturamento"), color="periodo:N")
    line2 = base.mark_line(strokeDash=[4, 2]).encode(y=alt.Y("volume:Q", title="Volume"), color="periodo:N")
    st.altair_chart((line1 + line2).interactive(), use_container_width=True)

with colB:
    st.markdown("**Categorias vendidas (período selecionado)**")
    cat_curr = (
        df_curr.groupby(COL["category"], as_index=False)[COL["rev"]]
        .sum()
        .sort_values(COL["rev"], ascending=False)
        .head(12)
    )
    if not cat_curr.empty:
        bar = alt.Chart(cat_curr).mark_bar().encode(
            x=alt.X(f"{COL['rev']}:Q", title="Faturamento"),
            y=alt.Y(f"{COL['category']}:N", sort="-x", title="Categoria")
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Sem categorias no período.")

# -----------------------------
# 3) Mapa de Clientes
# -----------------------------
st.subheader("3. Mapa de Clientes")

has_geo = (COL["lat"] in df_curr.columns) and (COL["lon"] in df_curr.columns)
if not has_geo:
    st.info("Sem colunas de geolocalização (lat/lon). Se você já tem isso em outra tabela, faça merge antes.")
else:
    map_df = (
        df_curr.groupby([COL["client"], COL["city"], COL["state"], COL["lat"], COL["lon"]], as_index=False)
        .agg(faturamento=(COL["rev"], "sum"), volume=(COL["vol"], "sum"))
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=[COL["lon"], COL["lat"]],
        get_radius=2000,
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=float(map_df[COL["lat"]].mean()),
        longitude=float(map_df[COL["lon"]].mean()),
        zoom=5,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={
        "text": "{cliente}\n{cidade}-{estado}\nFat: {faturamento}\nVol: {volume}"
    }))

    st.markdown("**Lista (clientes do mapa)**")
    show_cols = ["cliente", COL["city"], COL["state"], "faturamento", "volume"]
    tmp = map_df.rename(columns={COL["client"]: "cliente"})
    st.dataframe(tmp[show_cols].sort_values("faturamento", ascending=False), use_container_width=True)

# -----------------------------
# 4) Distribuição por clientes
# -----------------------------
st.subheader("4. Distribuição por clientes")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Índice de Concentração (HHI)", f"{hhi:.4f}" if not pd.isna(hhi) else "—")
c8.metric("Top 1 cliente", pct(top1))
c9.metric("Top 3 clientes", pct(top3))
c10.metric("Top 10 clientes", pct(top10))

if not dist_df.empty:
    st.markdown("**Curva acumulada (share de faturamento)**")
    curve = alt.Chart(dist_df).mark_line().encode(
        x=alt.X("cum_share:Q", title="Share acumulado do faturamento"),
        y=alt.Y("rev:Q", title="Faturamento (cliente ordenado)"),
        tooltip=["cliente:N", "rev:Q", "share:Q", "cum_share:Q"]
    )
    st.altair_chart(curve, use_container_width=True)

    with st.expander("Tabela de clientes (ordenado por faturamento)", expanded=False):
        dshow = dist_df.copy()
        dshow["rev"] = dshow["rev"].map(money)
        dshow["share"] = dist_df["share"].map(pct)
        dshow["cum_share"] = dist_df["cum_share"].map(pct)
        st.dataframe(dshow[["cliente", "rev", "share", "cum_share"]], use_container_width=True)
else:
    st.info("Sem vendas no período para calcular distribuição.")

# -----------------------------
# 5) Saúde da carteira – Detalhes
# -----------------------------
st.subheader("5. Saúde da carteira – Detalhes")

# Show breakdown as metrics
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Novos", str(health_counts.get("Novos", 0)), pct(health_share.get("Novos", np.nan)))
d2.metric("Crescendo", str(health_counts.get("Crescendo", 0)), pct(health_share.get("Crescendo", np.nan)))
d3.metric("Estáveis", str(health_counts.get("Estáveis", 0)), pct(health_share.get("Estáveis", np.nan)))
d4.metric("Caindo", str(health_counts.get("Caindo", 0)), pct(health_share.get("Caindo", np.nan)))
d5.metric("Perdidos", str(health_counts.get("Perdidos", 0)), pct(health_share.get("Perdidos", np.nan)))

# -----------------------------
# 6) Status dos clientes (lista)
# -----------------------------
st.subheader("6. Status dos clientes")

# Pretty table
t = status_df.copy()

# Formatting helpers
t["Fat (Período)"] = t["curr_rev"]
t["Fat (Anterior)"] = t["prev_rev"]
t["Δ Fat (R$)"] = t["delta_rev"]
t["Δ Fat (%)"] = t["pct_rev"]

t["Vol (Período)"] = t["curr_vol"]
t["Vol (Anterior)"] = t["prev_vol"]
t["Δ Vol"] = t["delta_vol"]
t["Δ Vol (%)"] = t["pct_vol"]

# Sort order
order = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
t["status"] = pd.Categorical(t["status"], categories=order, ordered=True)

# Choose sorting inside each status (by current revenue desc)
t = t.sort_values(["status", "curr_rev"], ascending=[True, False]).reset_index(drop=True)

# Render per status group
for s in order:
    sub = t[t["status"] == s].copy()
    st.markdown(f"### {s} ({len(sub)})")
    if sub.empty:
        st.caption("—")
        continue

    # Display-friendly formatting
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

st.caption(
    f"Período selecionado: {start.date()} a {end.date()} | "
    f"Período anterior: {prev_start.date()} a {prev_end.date()}" if show_prev else
    f"Período selecionado: {start.date()} a {end.date()}"
)
