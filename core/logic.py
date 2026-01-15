import pandas as pd
from .utils import STATUS_WEIGHTS, STATUS_COL

def get_kpis(df: pd.DataFrame):
    """Calculates top-level KPIs."""
    if df.empty:
        return {"faturamento": 0.0, "volume": 0, "clientes": 0, "cidades": 0}
    
    return {
        "faturamento": df["Valor"].sum(),
        "volume": df["Quantidade"].sum(),
        "clientes": df["Cliente"].nunique(),
        "cidades": df["Cidade"].nunique()
    }

def get_ranking(df: pd.DataFrame):
    """Calculates representative ranking."""
    if df.empty:
        return pd.DataFrame()
    
    ranking = df.groupby("Representante")[["Valor", "Quantidade"]].sum().reset_index()
    ranking.columns = ["Representante", "Faturamento", "Volume"]
    return ranking.sort_values("Faturamento", ascending=False)

def compute_carteira_health(df_filtered: pd.DataFrame, start_date, end_date):
    """Calculates the complex portfolio health logic (Churn/New/Growing)."""
    
    # Identify Time Spans
    s_ts = pd.Timestamp(start_date)
    e_ts = pd.Timestamp(end_date)
    
    months_span = (e_ts.year - s_ts.year) * 12 + (e_ts.month - s_ts.month) + 1
    
    prev_end = s_ts - pd.DateOffset(months=1)
    prev_start = prev_end - pd.DateOffset(months=months_span - 1)

    # Split Current vs Previous Period
    # Note: df_filtered should already contain the data for BOTH periods if possible, 
    # but usually we need to pass the FULL dataframe to this function to find history.
    
    # Simplified logic: We assume df_filtered contains data for the specific rep, 
    # but covering a wider date range than just the "current" selection.
    
    mask_curr = (df_filtered["Competencia"] >= s_ts) & (df_filtered["Competencia"] <= e_ts)
    mask_prev = (df_filtered["Competencia"] >= prev_start) & (df_filtered["Competencia"] <= prev_end)
    
    df_curr = df_filtered.loc[mask_curr]
    df_prev = df_filtered.loc[mask_prev]

    # Aggregate by Client
    curr_agg = df_curr.groupby("Cliente")["Valor"].sum().reset_index().rename(columns={"Valor": "ValorAtual"})
    prev_agg = df_prev.groupby("Cliente")["Valor"].sum().reset_index().rename(columns={"Valor": "ValorAnterior"})

    merged = pd.merge(curr_agg, prev_agg, on="Cliente", how="outer").fillna(0)

    # Classify
    def classify(row):
        va, vp = row["ValorAtual"], row["ValorAnterior"]
        if va > 0 and vp == 0: return "Novos"
        if va == 0 and vp > 0: return "Perdidos"
        if va > 0 and vp > 0:
            ratio = va / vp
            if ratio >= 1.2: return "Crescendo"
            if ratio <= 0.8: return "Caindo"
            return "Estáveis"
        return "N/A"

    merged["Status"] = merged.apply(classify, axis=1)
    
    # Calculate Score
    merged["PesoReceita"] = merged[["ValorAtual", "ValorAnterior"]].max(axis=1)
    total_rev = merged["PesoReceita"].sum()
    
    score_bruto = 0
    if total_rev > 0:
        for status, weight in STATUS_WEIGHTS.items():
            rev_status = merged.loc[merged["Status"] == status, "PesoReceita"].sum()
            score_bruto += weight * (rev_status / total_rev)
            
    # Normalize Score (0-100)
    # Range is roughly -2 to +2. Mapping to 0-100.
    isc = (score_bruto + 2) / 4 * 100
    isc = max(0, min(100, isc))
    
    # Churn Calculation
    base_anterior = merged[merged["ValorAnterior"] > 0]["PesoReceita"].sum()
    lost_rev = merged[merged["Status"] == "Perdidos"]["PesoReceita"].sum()
    churn_rate = (lost_rev / base_anterior * 100) if base_anterior > 0 else 0
    
    # Labels
    if isc < 30: label = "Crítica"
    elif isc < 50: label = "Alerta"
    elif isc < 70: label = "Neutra"
    else: label = "Saudável"
    
    return {
        "score": int(isc),
        "label": label,
        "churn": round(churn_rate, 1),
        "details": merged
    }
