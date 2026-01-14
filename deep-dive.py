import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from datetime import date, datetime

# --- Import Core Modules ---
from core.db import db_con
from core.logic import process_portfolio_data
# We assume these functions exist in queries.py. 
# If you haven't added get_rep_ranking yet, I included the SQL logic inline below as a fallback.
from core.queries import (
    get_kpis_data, 
    get_portfolio_health_data, 
    get_rep_ranking 
)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Dashboard V2 | Performance Moveleiro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling for better KPI cards
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SIDEBAR & FILTERS
# ==============================================================================
with st.sidebar:
    st.header("🔍 Filtros de Análise")
    
    # --- Date Filters ---
    col_d1, col_d2 = st.columns(2)
    default_start = date(2023, 1, 1)
    default_end = date(2023, 12, 31)
    
    start_date = col_d1.date_input("Início", default_start)
    end_date = col_d2.date_input("Fim", default_end)
    
    # --- Rep Filter (Dynamic Query) ---
    # Query distinct reps directly from DB for the dropdown
    try:
        reps_df = db_con.execute("SELECT DISTINCT Representante, RepresentanteCodigo FROM vendas ORDER BY Representante").df()
        rep_options = ["Todos"] + reps_df["Representante"].tolist()
    except Exception:
        rep_options = ["Todos"]
        
    selected_rep_name = st.selectbox("Representante", rep_options)
    
    # Map name back to Code (None if Todos)
    selected_rep_code = None
    if selected_rep_name != "Todos":
        code = reps_df.loc[reps_df["Representante"] == selected_rep_name, "RepresentanteCodigo"].values[0]
        # DuckDB expects standard Python integers, not numpy types
        selected_rep_code = int(code) 

    st.markdown("---")
    
    # --- Action Button ---
    # We use a form or a button to prevent recalculating on every calendar click
    if st.button("🔄 Atualizar Dashboard", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# ==============================================================================
# 3. MAIN DASHBOARD LOGIC
# ==============================================================================

st.title(f"📊 Insights de Vendas")
st.markdown(f"**Período:** {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')} | **Visão:** {selected_rep_name}")

if st.session_state.get("run_analysis"):
    
    # Convert dates to string for SQL
    s_date_str = str(start_date)
    e_date_str = str(end_date)

    # ---------------------------------------------------------
    # SECTION A: TOP LEVEL KPIS
    # ---------------------------------------------------------
    try:
        kpis = get_kpis_data(s_date_str, e_date_str, selected_rep_code)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faturamento Total", f"R$ {kpis['faturamento']:,.2f}")
        c2.metric("Volume (un)", f"{kpis['volume']:,.0f}")
        c3.metric("Clientes Ativos", kpis['clientes'])
        c4.metric("Cidades Cobertas", kpis['cidades'])
        
    except Exception as e:
        st.error(f"Erro ao calcular KPIs: {e}")

    # ---------------------------------------------------------
    # SECTION B: RANKING REPRESENTANTES (Conditional)
    # Only shows if "Todos" is selected
    # ---------------------------------------------------------
    if selected_rep_code is None:
        st.markdown("---")
        st.subheader("🏆 Ranking de Representantes")
        
        try:
            df_ranking = get_rep_ranking(s_date_str, e_date_str)
            
            if not df_ranking.empty:
                col_chart, col_table = st.columns([2, 1])
                
                with col_chart:
                    # Altair Horizontal Bar Chart
                    chart_rank = alt.Chart(df_ranking).mark_bar(cornerRadius=3).encode(
                        x=alt.X('Faturamento', title='Faturamento (R$)', axis=alt.Axis(format='~s')),
                        y=alt.Y('Representante', sort='-x', title=None),
                        color=alt.Color('Faturamento', scale=alt.Scale(scheme='blues'), legend=None),
                        tooltip=['Representante', alt.Tooltip('Faturamento', format=',.2f'), 'Volume']
                    ).properties(
                        height=max(300, len(df_ranking) * 25) # Dynamic height
                    )
                    st.altair_chart(chart_rank, use_container_width=True)
                
                with col_table:
                    # Simple clean table
                    st.caption("Top Performers")
                    df_show = df_ranking.copy()
                    df_show['Faturamento'] = df_show['Faturamento'].apply(lambda x: f"R$ {x:,.2f}")
                    st.dataframe(df_show, hide_index=True, use_container_width=True, height=300)
            else:
                st.info("Sem dados de vendas para gerar ranking.")
                
        except Exception as e:
            st.warning(f"Não foi possível gerar o ranking: {e}")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION C: SAÚDE DA CARTEIRA & DISTRIBUIÇÃO
    # ---------------------------------------------------------
    col_health, col_details = st.columns([1, 2])

    with col_health:
        st.subheader("🏥 Saúde da Carteira")
        try:
            # 1. Fetch Raw Data (Current vs Previous) via SQL
            df_health_raw = get_portfolio_health_data(s_date_str, e_date_str, selected_rep_code)
            
            # 2. Process Logic (Python/Pandas)
            health_results = process_portfolio_data(df_health_raw.to_dict(orient="records"))
            
            if health_results:
                score = health_results['score_metrics']['score']
                label = health_results['score_metrics']['label']
                churn = health_results['score_metrics']['churn_rate']
                
                # Big Score Metric
                st.metric("Score de Saúde", f"{score} / 100", delta=label, delta_color="off")
                st.caption(f"Churn Rate (Perda): {churn}%")
                
                # Summary Table by Status
                summary_df = pd.DataFrame(health_results['summary_table'])
                # Reorder based on logical funnel
                order_map = {"Novos": 1, "Crescendo": 2, "Estáveis": 3, "Caindo": 4, "Perdidos": 5}
                summary_df['order'] = summary_df['status'].map(order_map)
                summary_df = summary_df.sort_values('order').drop(columns='order')
                
                st.dataframe(
                    summary_df[['status', 'cliente_id', 'valor_atual']],
                    column_config={
                        "status": "Status",
                        "cliente_id": "Qtd Clientes",
                        "valor_atual": st.column_config.NumberColumn("Receita Atual", format="R$ %.2f")
                    },
                    hide_index=True
                )
            else:
                st.info("Dados insuficientes para cálculo de carteira.")

        except Exception as e:
            st.error(f"Erro no cálculo de saúde: {e}")

    with col_details:
        st.subheader("📍 Análise Geográfica (Preview)")
        # Placeholder for Map - Logic similar to KPIs, simple SQL query
        try:
            # Quick ad-hoc query for map (assuming geo table is ready)
            geo_query = """
                SELECT 
                    v.Estado, v.Cidade, g.lat, g.lon, SUM(v.Valor) as Faturamento
                FROM vendas v
                LEFT JOIN geo g ON v.geo_key = g.geo_key
                WHERE v.Competencia BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                GROUP BY v.Estado, v.Cidade, g.lat, g.lon
                HAVING Faturamento > 0
            """
            params = [s_date_str, e_date_str]
            if selected_rep_code:
                geo_query = geo_query.replace("GROUP BY", f"AND v.RepresentanteCodigo = {selected_rep_code} GROUP BY")
            
            df_map = db_con.execute(geo_query, params).df()
            
            if not df_map.empty and 'lat' in df_map.columns:
                # Simple Plotly Map
                fig_map = px.scatter_mapbox(
                    df_map.dropna(subset=['lat', 'lon']),
                    lat="lat", lon="lon",
                    size="Faturamento",
                    color="Faturamento",
                    hover_name="Cidade",
                    hover_data={"Estado": True, "Faturamento": ":.2f", "lat": False, "lon": False},
                    zoom=3,
                    center={"lat": -15.7, "lon": -47.8},
                    mapbox_style="carto-positron",
                    title="Distribuição de Vendas por Cidade",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Dados geográficos não encontrados ou incompletos.")
                
        except Exception as e:
            st.error(f"Erro ao carregar mapa: {e}")

else:
    # Landing State
    st.info("👈 Selecione os filtros na barra lateral e clique em 'Atualizar Dashboard' para começar.")
