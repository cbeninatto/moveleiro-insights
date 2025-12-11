# deep_dive_app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Nova Visão – Deep Dive",
    layout="wide",
)

# Coluna com o status da carteira (ajuste se o nome no seu CSV for diferente)
STATUS_COL = "StatusCarteira"  # ex: Novos / Perdidos / Crescendo / Caindo / Estáveis

# Caminho padrão do CSV no repositório (ajuste se necessário)
DEFAULT_CSV_PATH = "data/relatorio_faturamento.csv"


# ==========================
# HELPERS
# ==========================
def format_brl(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def normalize_month_column(df: pd.DataFrame, month_col: str = "Mes") -> pd.DataFrame:
    """
    Converte a coluna 'Mes' (nome ou número) para um número de 1 a 12 e cria 'MesNum'.
    """
    month_map = {
        "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4,
        "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
        "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
    }

    df = df.copy()

    # Garante inteiros na coluna Ano
    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")

    if df[month_col].dtype == "O":
        upper = df[month_col].astype(str).str.strip().str.upper()

        # Primeiro tenta mapear nomes de meses
        mes_num = upper.map(month_map)

        # Onde for NaN, tenta converter para número direto
        mask_na = mes_num.isna()
        if mask_na.any():
            mes_num.loc[mask_na] = pd.to_numeric(upper.loc[mask_na], errors="coerce")

        df["MesNum"] = mes_num.astype("Int64")
    else:
        df["MesNum"] = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")

    # Cria uma coluna de competência como Timestamp (primeiro dia do mês)
    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce"
    )

    return df


def compute_carteira_score(status_counts: pd.Series) -> tuple[float, str]:
    """
    Recebe um Series com index = status e values = quantidade de clientes.
    Retorna (pontuação 0–100, label textual).
    """
    if status_counts.empty:
        return 50.0, "Neutra"

    weights = {
        "Novo": 1,
        "Novos": 1,
        "Crescendo": 2,
        "CRESCENDO": 2,
        "Caindo": -1,
        "CAINDO": -1,
        "Estável": 1,
        "Estáveis": 1,
        "ESTAVEL": 1,
        "ESTAVEIS": 1,
        "Perdido": -2,
        "Perdidos": -2,
        "PERDIDO": -2,
        "PERDIDOS": -2,
    }

    score_total = 0
    n_clients = 0

    for status, qty in status_counts.items():
        w = weights.get(str(status), 0)
        score_total += w * qty
        n_clients += qty

    if n_clients == 0:
        return 50.0, "Neutra"

    # Cada cliente pode variar entre -2 e +2
    avg = score_total / n_clients  # em [-2, 2]
    score_0_100 = (avg + 2) / 4 * 100  # mapeia -2→0, 2→100
    score_0_100 = max(0, min(100, score_0_100))

    if score_0_100 < 40:
        label = "Crítica"
    elif score_0_100 < 60:
        label = "Neutra"
    else:
        label = "Saudável"

    return score_0_100, label


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """
    Carrega CSV a partir de path ou arquivo enviado.
    Tenta ; primeiro, depois ,.
    """
    if file is None:
        raise FileNotFoundError("Nenhum arquivo informado.")

    try:
        df = pd.read_csv(file, sep=";")
    except Exception:
        df = pd.read_csv(file, sep=",")

    # Normaliza as colunas básicas
    col_map = {c.lower(): c for c in df.columns}
    # Esperados: ano, mes, valor, representante, cliente, estado, cidade
    expected_lower = [
        "ano", "mes", "valor",
        "representante", "cliente", "estado", "cidade"
    ]

    missing = [c for c in expected_lower if c not in col_map]
    if missing:
        st.error(
            "Colunas obrigatórias não encontradas no CSV: "
            + ", ".join(missing)
        )
        st.stop()

    # Renomeia para ter nomes padronizados
    renames = {col_map["ano"]: "Ano",
               col_map["mes"]: "Mes",
               col_map["valor"]: "Valor",
               col_map["representante"]: "Representante",
               col_map["cliente"]: "Cliente",
               col_map["estado"]: "Estado",
               col_map["cidade"]: "Cidade"}
    df = df.rename(columns=renames)

    # Converte valor
    df["Valor"] = (
        df["Valor"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce").fillna(0.0)

    # Normaliza meses / competência
    df = normalize_month_column(df, "Mes")

    return df


# ==========================
# SIDEBAR – CONTROLES
# ==========================
st.sidebar.title("Filtros – Deep Dive")

st.sidebar.markdown("**Fonte de dados**")

uploaded_file = st.sidebar.file_uploader(
    "Envie o CSV processado",
    type=["csv"],
    help="Use o mesmo CSV de faturamento/processado que alimenta o dashboard."
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.sidebar.caption(f"Ou use o caminho padrão: `{DEFAULT_CSV_PATH}`")
    df = load_data(DEFAULT_CSV_PATH)

if df.empty:
    st.warning("O arquivo de dados está vazio.")
    st.stop()

# Seleção de representante
reps = sorted(df["Representante"].dropna().unique())
rep_selected = st.sidebar.selectbox("Representante", reps)

# Período (competências)
valid_comp = df["Competencia"].dropna().sort_values().unique()
if len(valid_comp) == 0:
    st.error("Não foi possível identificar as competências (Ano/Mês).")
    st.stop()

default_start = valid_comp[max(0, len(valid_comp) - 12)]  # últimos 12 meses (se existir)
default_end = valid_comp[-1]

start_comp, end_comp = st.sidebar.select_slider(
    "Período (competência)",
    options=list(valid_comp),
    value=(default_start, default_end),
    format_func=lambda x: x.strftime("%b %Y"),
)

# ==========================
# FILTROS E SUBSETS
# ==========================
mask_period = (df["Competencia"] >= start_comp) & (df["Competencia"] <= end_comp)
df_period = df.loc[mask_period].copy()

if df_period.empty:
    st.warning("Nenhuma venda no período selecionado.")
    st.stop()

df_rep = df_period[df_period["Representante"] == rep_selected].copy()

# ==========================
# HEADER
# ==========================
st.title("Deep Dive – Representante")

st.subheader(f"Representante: **{rep_selected}**")

st.caption(
    f"Período selecionado: "
    f"{start_comp.strftime('%b %Y')} até {end_comp.strftime('%b %Y')}"
)

st.markdown("---")

# ==========================
# MÉTRICAS PRINCIPAIS
# ==========================
col1, col2, col3, col4, col5 = st.columns(5)

# Total período (do representante)
total_rep = df_rep["Valor"].sum()

# Meses com venda (do representante)
if not df_rep.empty:
    meses_rep = (
        df_rep.groupby([df_rep["Ano"], df_rep["MesNum"]])["Valor"]
        .sum()
        .reset_index(name="ValorMes")
    )
    meses_com_venda = (meses_rep["ValorMes"] > 0).sum()
else:
    meses_com_venda = 0

# Total de meses no período (com base no dataset inteiro)
meses_periodo = (
    df_period.groupby([df_period["Ano"], df_period["MesNum"]])["Valor"]
    .sum()
    .reset_index(name="ValorMes")
)
total_meses_periodo = len(meses_periodo)

# Média mensal (considerando apenas meses com venda do rep)
if meses_com_venda > 0:
    media_mensal = total_rep / meses_com_venda
else:
    media_mensal = 0.0

# Participação do representante no período
total_periodo_geral = df_period["Valor"].sum()
if total_periodo_geral > 0:
    participacao = total_rep / total_periodo_geral
else:
    participacao = 0.0

# Saúde da carteira – pontuação
if STATUS_COL in df_rep.columns:
    # Agrupa por cliente e status (para não contar cliente duplicado no mesmo status)
    clientes_rep = (
        df_rep
        .dropna(subset=[STATUS_COL, "Cliente"])
        .groupby(["Cliente", STATUS_COL], as_index=False)
        .agg({"Valor": "sum"})
    )
    status_counts = clientes_rep.groupby(STATUS_COL)["Cliente"].nunique()
    carteira_score, carteira_label = compute_carteira_score(status_counts)
else:
    carteira_score, carteira_label = 50.0, "Neutra"

# Preenche cards
col1.metric("Total período", format_brl(total_rep))
col2.metric("Média mensal", format_brl(media_mensal))
col3.metric("Meses com venda", f"{meses_com_venda} / {total_meses_periodo}")
col4.metric("Participação", f"{participacao:.1%}")

# Saúde da Carteira – apenas pontuação resumida aqui
with col5:
    st.metric(
        "Saúde da carteira",
        f"{carteira_score:.0f} / 100",
        carteira_label
    )

st.markdown("---")

# ==========================
# EVOLUÇÃO DE VENDAS (LINHA)
# ==========================
st.subheader("Evolução de vendas no período")

if df_rep.empty:
    st.info("Este representante não possui vendas no período selecionado.")
else:
    ts_rep = (
        df_rep
        .groupby("Competencia", as_index=False)["Valor"]
        .sum()
        .sort_values("Competencia")
    )

    chart_ts = (
        alt.Chart(ts_rep)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Competencia:T",
                axis=alt.Axis(title="Competência", format="%b %Y"),
            ),
            y=alt.Y(
                "Valor:Q",
                axis=alt.Axis(title="Faturamento"),
            ),
            tooltip=[
                alt.Tooltip("Competencia:T", title="Competência", format="%b %Y"),
                alt.Tooltip("Valor:Q", title="Faturamento", format=",.2f"),
            ],
        )
        .properties(
            height=260,
        )
    )

    st.altair_chart(chart_ts, use_container_width=True)

st.markdown("---")

# ==========================
# SAÚDE DA CARTEIRA – DETALHES
# ==========================
st.subheader("Saúde da carteira – Detalhes")

if STATUS_COL not in df_rep.columns:
    st.info(
        f"Coluna de status da carteira (`{STATUS_COL}`) não encontrada no dataframe. "
        "Adicione esta coluna no CSV para ver a distribuição de Novos/Perdidos/Crescendo/Caindo/Estáveis."
    )
else:
    # Tabela por cliente e status
    clientes_rep = (
        df_rep
        .dropna(subset=[STATUS_COL, "Cliente"])
        .groupby(["Cliente", STATUS_COL, "Estado", "Cidade"], as_index=False)
        .agg({"Valor": "sum"})
    )

    status_counts = clientes_rep.groupby(STATUS_COL)["Cliente"].nunique().reset_index()
    status_counts = status_counts.rename(
        columns={"Cliente": "QtdClientes", STATUS_COL: "Status"}
    )
    total_clientes = status_counts["QtdClientes"].sum()
    status_counts["%Clientes"] = (
        status_counts["QtdClientes"] / total_clientes if total_clientes > 0 else 0
    )

    col_pie, col_table = st.columns([1, 1.2])

    with col_pie:
        st.caption("Distribuição de clientes por status")
        if total_clientes == 0:
            st.info("Nenhum cliente com status definido para este representante no período.")
        else:
            chart_pie = (
                alt.Chart(status_counts)
                .mark_arc(outerRadius=120)
                .encode(
                    theta=alt.Theta("QtdClientes:Q"),
                    color=alt.Color("Status:N", legend=alt.Legend(title="Status")),
                    tooltip=[
                        alt.Tooltip("Status:N", title="Status"),
                        alt.Tooltip("QtdClientes:Q", title="Clientes"),
                        alt.Tooltip("%Clientes:Q", title="% Clientes", format=".1%"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_pie, use_container_width=True)

    with col_table:
        st.caption("Resumo por status")
        status_counts_display = status_counts.copy()
        status_counts_display["%Clientes"] = status_counts_display["%Clientes"].map(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(
            status_counts_display,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Lista de clientes da carteira")

    # Filtro por status
    status_options = sorted(clientes_rep[STATUS_COL].dropna().unique())
    status_selected = st.multiselect(
        "Filtrar por status",
        options=status_options,
        default=status_options,
    )

    df_clientes_view = clientes_rep.copy()
    if status_selected:
        df_clientes_view = df_clientes_view[df_clientes_view[STATUS_COL].isin(status_selected)]

    df_clientes_view = df_clientes_view.rename(
        columns={
            "Valor": "Faturamento",
            STATUS_COL: "StatusCarteira"
        }
    )

    df_clientes_view["FaturamentoFmt"] = df_clientes_view["Faturamento"].map(format_brl)

    # Ordena por faturamento desc
    df_clientes_view = df_clientes_view.sort_values(
        "Faturamento", ascending=False
    )[
        ["Cliente", "Estado", "Cidade", "StatusCarteira", "FaturamentoFmt"]
    ]

    st.dataframe(
        df_clientes_view,
        hide_index=True,
        use_container_width=True,
    )

# Fim
