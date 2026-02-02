# rep_report.py
# Streamlit - Relatório por Representante (Opção C: lê um arquivo final Parquet/CSV/XLSX)
# Ajuste DEFAULT_PATH e o mapeamento COL conforme seu dataset.

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    import pydeck as pdk
    HAS_PYDECK = True
except Exception:
    HAS_PYDECK = False

st.set_page_config(page_title="Relatório do Representante", layout="wide")

# ============================================================
# 0) CONFIG
# ============================================================
DEFAULT_PATH = r"C:\Users\Cesar\CB Database\Documents\GitHub\performance-moveleiro-v2\data\base_final.parquet"
# Exemplos:
# r"...\base_final.csv"
# r"...\base_final.xlsx"

# Mapeie aqui os nomes REAIS das colunas do seu arquivo final
COL = {
    "date": "data",
    "rep": "representante",
    "client": "cliente",
    "city": "cidade",
    "state": "estado",
    "category": "categoria",
    "rev": "faturamento",
    "vol": "volume",
    "lat": "lat",   # opcional
    "lon": "lon",   # opcional
}

# ============================================================
# 1) HELPERS
# ============================================================
def safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    out = np.full_like(a, np.nan, dtype="float64")
    np.divide(a, b, out=out, where=(b != 0))
    return out

def pct_change(curr, prev):
    return safe_div((curr - prev), prev)

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

def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_data_from_file(path: str, csv_sep: str | None, csv_encoding: str | None) -> pd.DataFrame:
    if not path or not isinstance(path, str):
        return pd.DataFrame()
    if not os.path.exists(path):
        return pd.DataFrame()

    ext = os.path.splitext(path.lower())[1]

    try:
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".csv":
            # Se csv_sep=None, tenta inferir. Se der ruim, defina sep=";" no sidebar.
            if csv_sep:
                df = pd.read_csv(path, sep=csv_sep, encoding=csv_encoding or "utf-8", engine="python")
            else:
                df = pd.read_csv(path, sep=None, encoding=csv_encoding or "utf-8", engine="python")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    return df

def previous_period(start: pd.Timestamp, end: pd.Timestamp):
    days = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start, prev_end

def filter_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    d = df[(df[COL["date"]] >= start) & (df[COL["date"]] <= end)].copy()
    return d

# ============================================================
# 2) CORE CALCS
# ============================================================
def client_distribution(df_rep_period: pd.DataFrame):
    by_client = (
        df_rep_period.groupby(COL["client"], as_index=False)[COL["rev"]]
        .sum()
        .rename(columns={COL["rev"]: "rev"})
    )
    by_client["rev"] = by_client["rev"].fillna(0.0)
    total = by_client["rev"].sum()

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

    dist_df = by_client.rename(columns={COL["client"]: "cliente"})
    return dist_df, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10

def build_client_status(df_curr: pd.DataFrame, df_prev: pd.DataFrame):
    thr = 0.10  # faixa de estabilidade +/- 10%

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
    m["pct_rev"] = pct_change(m["curr_rev"].values, m["prev_rev"].values)
    m["pct_vol"] = pct_change(m["curr_vol"].values, m["prev_vol"].values)

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

def carteira_health_score(status_df: pd.DataFrame):
    counts = status_df["status"].value_counts().to_dict()
    total = max(1, len(status_df))

    share = {
        k: counts.get(k, 0) / total
        for k in ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
    }

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
# 3) UI - SIDEBAR (Opção C)
# ============================================================
st.title("Relatório do Representante")

with st.sidebar:
    st.header("Fonte de dados (Opção C)")
    st.text_input("Caminho do arquivo final", key="data_path", value=DEFAULT_PATH)

    st.caption("Se for CSV, ajuste separador/encoding se necessário.")
    csv_sep = st.text_input("CSV sep (ex: ; ou ,) — vazio = auto", value="")
    csv_encoding = st.text_input("CSV encoding (ex: utf-8, latin1) — vazio = utf-8", value="")

    st.divider()
    st.header("Filtros")
    show_prev = st.checkbox("Comparar com período anterior (mesma duração)", value=True)

# Load data
path = st.session_state.get("data_path", DEFAULT_PATH)
sep = (csv_sep.strip() or None)
enc = (csv_encoding.strip() or None)

df = load_data_from_file(path, sep, enc)

if df.empty:
    st.error("Seu loader está retornando um dataframe vazio.")
    st.write("Caminho:", path)
    st.write("Existe no disco?:", os.path.exists(path))
    st.write("Dica: confira extensão (.parquet/.csv/.xlsx), separador do CSV e encoding.")
    st.stop()

# Basic validation for required columns
missing = [k for k, v in COL.items() if (v not in df.columns) and (k not in ("lat", "lon"))]
if missing:
    st.error("Faltam colunas necessárias no arquivo final (pelo seu COL mapping).")
    st.write("Chaves ausentes:", missing)
    st.write("Colunas do arquivo:", list(df.columns))
    st.write("Ajuste o dicionário COL no topo do script para refletir seus nomes reais.")
    st.stop()

df = ensure_datetime(df, COL["date"])

st.success(f"Dados carregados: {len(df):,} linhas • {df.shape[1]} colunas".replace(",", "."))

with st.expander("Preview do dataframe", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

# Build rep list
if df[COL["rep"]].dropna().empty:
    st.error("Coluna de representante está vazia (ou só NaN). Ajuste COL['rep'] ou seu arquivo final.")
    st.stop()

reps = sorted(df[COL["rep"]].dropna().unique().tolist())

with st.sidebar:
    rep = st.selectbox("Representante", reps)

    min_d = df[COL["date"]].min()
    max_d = df[COL["date"]].max()
    if pd.isna(min_d) or pd.isna(max_d):
        st.error("Coluna de data inválida (tudo NaT). Ajuste COL['date'] e/ou o formato no arquivo.")
        st.stop()

    default_end = max_d
    default_start = max(min_d, max_d - pd.Timedelta(days=365))

    start = st.date_input("Início", value=default_start.date(), min_value=min_d.date(), max_value=max_d.date())
    end = st.date_input("Fim", value=default_end.date(), min_value=min_d.date(), max_value=max_d.date())

start = pd.to_datetime(start)
end = pd.to_datetime(end)

if start > end:
    st.error("Início não pode ser maior que o fim.")
    st.stop()

prev_start, prev_end = previous_period(start, end)

# Filter by rep and periods
df_rep = df[df[COL["rep"]] == rep].copy()
df_curr = filter_period(df_rep, start, end)
df_prev = filter_period(df_rep, prev_start, prev_end) if show_prev else df_curr.iloc[0:0].copy()

# Precompute metrics
dist_df, n80_count, n80_share_clients, n80_rev_share, hhi, top1, top3, top10 = client_distribution(df_curr)
status_df = build_client_status(df_curr, df_prev if show_prev else df_curr.iloc[0:0])
health_score, health_counts, health_share = carteira_health_score(status_df)

curr_rev = float(df_curr[COL["rev"]].fillna(0).sum())
curr_vol = float(df_curr[COL["vol"]].fillna(0).sum())
prev_rev = float(df_prev[COL["rev"]].fillna(0).sum()) if show_prev else np.nan
prev_vol = float(df_prev[COL["vol"]].fillna(0).sum()) if show_prev else np.nan

clients_attended = int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["client"]].nunique())
cities_attended = int(df_curr[df_curr[COL["rev"]].fillna(0) > 0][COL["city"]].nunique())

# ============================================================
# 1) Performance Dashboard
# ============================================================
st.subheader("1. Performance Dashboard")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Faturamento Total", money(curr_rev), pct(pct_change(curr_rev, prev_rev)) if show_prev else None)
c2.metric("Volume Total", num(curr_vol), pct(pct_change(curr_vol, prev_vol)) if show_prev else None)
c3.metric("N80 (qtd clientes)", f"{n80_count}", pct(n80_share_clients) if n80_count else "—")
c4.metric("Saúde da Carteira (0-100)", f"{health_score:.0f}")
c5.metric("Clientes Atendidos", f"{clients_attended}")
c6.metric("Cidades Atendidas", f"{cities_attended}")

with st.expander("Resumo N80 / Concentração", expanded=False):
    st.write(
        f"- **N80**: {n80_count} clientes (**{pct(n80_share_clients)}** da carteira) "
        f"atingem **{pct(n80_rev_share)}** do faturamento.\n"
        f"- **Concentração (HHI)**: {hhi:.4f}" if not pd.isna(hhi) else "- **Concentração (HHI)**: —"
    )
    st.write(f"- **Top 1 / 3 / 10**: {pct(top1)} | {pct(top3)} | {pct(top10)}")

# ============================================================
# 2) Evolução
# ============================================================
st.subheader("2. Evolução (comparativo)")

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
    if evo.empty:
        st.info("Sem dados no período.")
    else:
        base = alt.Chart(evo).encode(x=alt.X("mes:T", title="Mês"))
        line_rev = base.mark_line().encode(
            y=alt.Y("faturamento:Q", title="Faturamento"),
            color="periodo:N",
            tooltip=["mes:T", "periodo:N", "faturamento:Q", "volume:Q"],
        )
        line_vol = base.mark_line(strokeDash=[4, 2]).encode(
            y=alt.Y("volume:Q", title="Volume"),
            color="periodo:N",
            tooltip=["mes:T", "periodo:N", "faturamento:Q", "volume:Q"],
        )
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
            tooltip=[COL["category"], COL["rev"]],
        )
        st.altair_chart(bar, use_container_width=True)

# ============================================================
# 3) Mapa de Clientes
# ============================================================
st.subheader("3. Mapa de Clientes")

has_geo = (COL["lat"] in df_curr.columns) and (COL["lon"] in df_curr.columns)
if not has_geo:
    st.info("Sem colunas de geolocalização (lat/lon). Se você já tem isso em outra tabela, faça merge antes.")
elif not HAS_PYDECK:
    st.info("pydeck não está disponível neste ambiente. Instale/garanta pydeck ou remova a seção do mapa.")
else:
    map_df = (
        df_curr.groupby([COL["client"], COL["city"], COL["state"], COL["lat"], COL["lon"]], as_index=False)
        .agg(faturamento=(COL["rev"], "sum"), volume=(COL["vol"], "sum"))
        .rename(columns={COL["client"]: "cliente"})
    )

    if map_df.empty:
        st.info("Sem dados georreferenciados no período.")
    else:
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

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{cliente}\n{cidade}-{estado}\nFat: {faturamento}\nVol: {volume}"},
            )
        )

        st.markdown("**Lista (clientes do mapa)**")
        show_cols = ["cliente", COL["city"], COL["state"], "faturamento", "volume"]
        st.dataframe(map_df[show_cols].sort_values("faturamento", ascending=False), use_container_width=True)

# ============================================================
# 4) Distribuição por clientes
# ============================================================
st.subheader("4. Distribuição por clientes")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Índice de Concentração (HHI)", f"{hhi:.4f}" if not pd.isna(hhi) else "—")
c8.metric("Top 1 cliente", pct(top1))
c9.metric("Top 3 clientes", pct(top3))
c10.metric("Top 10 clientes", pct(top10))

if dist_df.empty:
    st.info("Sem vendas no período para calcular distribuição.")
else:
    st.markdown("**Tabela (ordenado por faturamento)**")
    dshow = dist_df.copy()
    dshow["Faturamento"] = dshow["rev"].map(money)
    dshow["Share"] = dshow["share"].map(pct)
    dshow["Share acumulado"] = dshow["cum_share"].map(pct)
    st.dataframe(dshow[["cliente", "Faturamento", "Share", "Share acumulado"]], use_container_width=True)

# ============================================================
# 5) Saúde da carteira – Detalhes
# ============================================================
st.subheader("5. Saúde da carteira – Detalhes")

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Novos", str(health_counts.get("Novos", 0)), pct(health_share.get("Novos", np.nan)))
d2.metric("Crescendo", str(health_counts.get("Crescendo", 0)), pct(health_share.get("Crescendo", np.nan)))
d3.metric("Estáveis", str(health_counts.get("Estáveis", 0)), pct(health_share.get("Estáveis", np.nan)))
d4.metric("Caindo", str(health_counts.get("Caindo", 0)), pct(health_share.get("Caindo", np.nan)))
d5.metric("Perdidos", str(health_counts.get("Perdidos", 0)), pct(health_share.get("Perdidos", np.nan)))

# ============================================================
# 6) Status dos clientes
# ============================================================
st.subheader("6. Status dos clientes")

t = status_df.copy()

# colunas do output
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

# Footer
if show_prev:
    st.caption(
        f"Período selecionado: {start.date()} a {end.date()} | "
        f"Período anterior: {prev_start.date()} a {prev_end.date()}"
    )
else:
    st.caption(f"Período selecionado: {start.date()} a {end.date()}")
