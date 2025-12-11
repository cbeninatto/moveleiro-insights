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

# Nome da coluna de status no CSV (ajuste se for diferente)
STATUS_COL = "StatusCarteira"  # ex: Novos / Perdidos / Crescendo / Caindo / Estáveis

# Caminho padrão do CSV dentro do repositório
# (mesmo arquivo que está em performance-moveleiro-v2/data no GitHub)
DEFAULT_CSV_PATH = "data/relatorio_faturamento.csv"

# Se algum dia você quiser ler direto do GitHub Raw em vez do arquivo local,
# pode usar uma URL assim (não é obrigatório, só um exemplo):
# DEFAULT_CSV_PATH = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"


# ==========================
# HELPERS
# ==========================
def format_brl(value: float) -> str:
    if pd.isna(value):
        return "R$ 0,00"
    return "R$ " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def normalize_month_column(df: pd.DataFrame, month_col: str = "Mes") -> pd.DataFrame:
    month_map = {
        "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4,
        "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
        "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
    }

    df = df.copy()

    df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")

    if df[month_col].dtype == "O":
        upper = df[month_col].astype(str).str.strip().str.upper()
        mes_num = upper.map(month_map)
        mask_na = mes_num.isna()
        if mask_na.any():
            mes_num.loc[mask_na] = pd.to_numeric(upper.loc[mask_na], errors="coerce")
        df["MesNum"] = mes_num.astype("Int64")
    else:
        df["MesNum"] = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")

    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce"
    )

    return df


def compute_carteira_score(status_counts: pd.Series) -> tuple[float, str]:
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

    avg = score_total / n_clients  # [-2, 2]
    score_0_100 = (avg + 2) / 4 * 100
    score_0_100 = max(0, min(100, score_0_100))

    if score_0_100 < 40:
        label = "Crítica"
    elif score_0_100 < 60:
        label = "Neutra"
        # note: 40–60 is zona neutra
    else:
        label = "Saudável"

    return score_0_100, label


def _normalize_col_name(col: str) -> str:
    """
    Remove BOM, espaços, coloca em minúsculo e tira alguns acentos
    para facilitar o match: 'Mês  ' -> 'mes'
    """
    s = str(col).strip().replace("\ufeff", "").lower()
    # tirar acentos básicos
    s = (
        s.replace("á", "a").replace("à", "a").replace("ã", "a").replace("â", "a")
         .replace("é", "e").replace("ê", "e")
         .replace("í", "i")
         .replace("ó", "o").replace("õ", "o").replace("ô", "o")
         .replace("ú", "u")
         .replace("ç", "c")
    )
    return s


@st.cache_data(show_spinner=False)
def load_data(source) -> pd.DataFrame:
    """
    Carrega CSV a partir de caminho (str) ou arquivo enviado pelo Streamlit.
    Faz detecção inteligente das colunas Ano/Mês/Valor/Representante/Cliente/Estado/Cidade.
    """
    # Decide se é caminho (string) ou UploadedFile
    if hasattr(source, "read"):  # UploadedFile do st.file_uploader
        file_obj = source
    else:
        file_obj = source  # caminho ou URL

    # Tenta ; depois ,
    read_ok = False
    last_error = None
    for sep in [";", ","]:
        try:
            df = pd.read_csv(file_obj, sep=sep)
            read_ok = True
            break
        except Exception as e:
            last_error = e
            try:
                file_obj.seek(0)
            except Exception:
                pass

    if not read_ok:
        st.error(f"Erro ao ler o CSV: {last_error}")
        st.stop()

    # Mapa de nomes normalizados -> nome original
    col_map = {_normalize_col_name(c): c for c in df.columns}

    # Quais campos lógicos precisamos e quais nomes aceitamos para cada um
    needed = {
        "ano": ["ano", "year"],
        "mes": ["mes", "mesfat", "mes_fat", "competencia"],
        "valor": ["valor", "faturamento", "vl_total", "total"],
        "representante": ["representante", "vendedor", "rep"],
        "cliente": ["cliente", "razaosocial", "nomecliente", "cliente_nome"],
        "estado": ["estado", "uf"],
        "cidade": ["cidade", "municipio"],
    }

    rename_dict = {}
    missing_logical = []

    for logical_name, candidates in needed.items():
        found_original = None
        for cand in candidates:
            if cand in col_map:
                found_original = col_map[cand]
                break
        if found_original is None:
            missing_logical.append(logical_name)
        else:
            # padroniza colunas com inicial maiúscula (Ano, Mes, Valor, etc.)
            rename_dict[found_original] = logical_name.capitalize()

    if missing_logical:
        st.error(
            "Colunas obrigatórias não encontradas no CSV: "
            + ", ".join(missing_logical)
            + "\n\nColunas disponíveis (normalizadas): "
            + ", ".join(sorted(col_map.keys()))
        )
        st.stop()

    df = df.rename(columns=rename_dict)

    # Converte Valor para número (suporta '1.234,56')
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
