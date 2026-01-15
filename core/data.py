import pandas as pd
import requests
import io
import time
import re
import unicodedata
import streamlit as st

# URLs (from your original file)
GITHUB_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/main/data/relatorio_faturamento.csv"
CITY_GEO_CSV_URL = "https://raw.githubusercontent.com/cbeninatto/performance-moveleiro-v2/main/data/cidades_br_geo.csv"

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Fetches the sales data from GitHub."""
    cb = int(time.time())
    url = f"{GITHUB_CSV_URL}?cb={cb}"
    resp = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))

    # Basic cleanup
    cols_num = ["Valor", "Quantidade", "Ano", "Mes"]
    for c in cols_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    # Create Date Column
    df["MesNum"] = df["Mes"].astype(int)
    df["Ano"] = df["Ano"].astype(int)
    df["Competencia"] = pd.to_datetime(
        dict(year=df["Ano"], month=df["MesNum"], day=1),
        errors="coerce"
    )
    return df

@st.cache_data(ttl=3600)
def load_geo() -> pd.DataFrame:
    """Fetches the geolocation data."""
    cb = int(time.time())
    url = f"{CITY_GEO_CSV_URL}?cb={cb}"
    resp = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=60)
    resp.raise_for_status()
    
    df_geo = pd.read_csv(io.StringIO(resp.text), sep=None, engine="python")
    
    # Standardize columns (Simplified logic)
    cols = {c.lower(): c for c in df_geo.columns}
    lat_c = cols.get("lat") or cols.get("latitude")
    lon_c = cols.get("lon") or cols.get("longitude")
    state_c = cols.get("estado") or cols.get("uf")
    city_c = cols.get("cidade") or cols.get("municipio")

    if not all([lat_c, lon_c, state_c, city_c]):
        return pd.DataFrame()

    df_geo = df_geo[[state_c, city_c, lat_c, lon_c]].copy()
    df_geo.columns = ["Estado", "Cidade", "lat", "lon"]
    
    # Clean Lat/Lon
    for c in ["lat", "lon"]:
        df_geo[c] = df_geo[c].astype(str).str.replace(",", ".").astype(float)

    # Create key for joining
    df_geo["geo_key"] = df_geo["Estado"].str.strip().str.upper() + "|" + df_geo["Cidade"].str.strip().str.upper()
    return df_geo[["geo_key", "lat", "lon"]]
