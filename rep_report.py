import os
import time
import pandas as pd
import requests

INPUT = "data/raw/relatorio_faturamento.csv"
OUT = "data/geo/cidades_latlon.csv"

def read_base():
    df = pd.read_csv(INPUT, sep=None, engine="python", encoding="utf-8")
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    return df

def geocode_city_state(city: str, state: str):
    # Nominatim usage policy: identify your app and throttle
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{city}, {state}, Brasil", "format": "json", "limit": 1}
    headers = {"User-Agent": "moveleiro-insights/1.0 (contact: youremail@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None, None
    return float(data[0]["lat"]), float(data[0]["lon"])

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = read_base()
    cidades = (
        df[["Cidade", "Estado"]]
        .dropna()
        .astype(str)
        .apply(lambda s: s.str.strip())
        .drop_duplicates()
        .sort_values(["Estado", "Cidade"])
        .reset_index(drop=True)
    )

    # resume if file exists
    if os.path.exists(OUT):
        done = pd.read_csv(OUT)
        done["Cidade"] = done["Cidade"].astype(str).str.strip()
        done["Estado"] = done["Estado"].astype(str).str.strip()
        cidades = cidades.merge(done[["Cidade", "Estado"]], on=["Cidade", "Estado"], how="left", indicator=True)
        cidades = cidades[cidades["_merge"] == "left_only"][["Cidade", "Estado"]].reset_index(drop=True)

    rows = []
    for i, row in cidades.iterrows():
        city, state = row["Cidade"], row["Estado"]
        lat, lon = geocode_city_state(city, state)
        rows.append({"Cidade": city, "Estado": state, "lat": lat, "lon": lon})
        print(f"{i+1}/{len(cidades)} {city}-{state}: {lat},{lon}")
        time.sleep(1.1)  # throttle

        # incremental save
        out_df = pd.DataFrame(rows)
        if os.path.exists(OUT):
            prev = pd.read_csv(OUT)
            out_df = pd.concat([prev, out_df], ignore_index=True)
        out_df.drop_duplicates(["Cidade", "Estado"], inplace=True)
        out_df.to_csv(OUT, index=False)

if __name__ == "__main__":
    main()
