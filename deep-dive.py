import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from datetime import date, datetime

# --- Import Core Modules ---
from core.data import load_data, load_geo
from core.logic import get_kpis, get_ranking, compute_carteira_health
from core.pdf import build_pdf_report, _plotly_to_png_bytes

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
# 2. LOAD DATA (Replaces DB Connection)
# ==============================================================================
try:
    df_all = load_data()
    df_geo = load_geo()
    
    # Pre-process for map
    df_all["geo_key"] = df_all["Estado"].str.strip().str.upper() + "|" + df_all["Cidade"].str.strip().str.upper()
    df_map_base = pd.merge(df_all, df_geo, on="geo_key", how="left")
except Exception as e:
    st.error(f"Erro crítico ao carregar dados: {e}")
    st.stop()

# ==============================================================================
# 3. SIDEBAR & FILTERS
# ==============================================================================
with st.sidebar:
    st.header("🔍 Filtros de Análise")
    
    # --- Date Filters ---
    min_date = df_all["Competencia"].min().date()
    max_date = df_all["Competencia"].max().date()
    
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Início", min_date)
    end_date = col_d2.date_input("Fim", max_date)
    
    # --- Rep Filter ---
    reps = sorted(df_all["Representante"].dropna().unique())
    rep_options = ["Todos"] + list(reps)
    selected_rep_name = st.selectbox("Representante", rep_options)
    
    st.markdown("---")
    
    if st.button("🔄 Atualizar Dashboard", type="primary", use_container_width=True):
        st.session_state.run_analysis = True

# ==============================================================================
# 4. MAIN DASHBOARD LOGIC
# ==============================================================================

st.title(f"📊 Insights de Vendas")
st.markdown(f"**Período:** {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')} | **Visão:** {selected_rep_name}")

if st.session_state.get("run_analysis"):
    
    # Filter Data by Date
    mask_date = (df_all["Competencia"].dt.date >= start_date) & (df_all["Competencia"].dt.date <= end_date)
    df_period = df_all[mask_date].copy()
    
    # Filter by Rep (for Main View)
    if selected_rep_name != "Todos":
        df_view = df_period[df_period["Representante"] == selected_rep_name]
        df_map_view = df_map_base[(df_map_base["Competencia"].dt.date >= start_date) & 
                                  (df_map_base["Competencia"].dt.date <= end_date) & 
                                  (df_map_base["Representante"] == selected_rep_name)]
    else:
        df_view = df_period
        df_map_view = df_map_base[mask_date]

    # ---------------------------------------------------------
    # SECTION A: TOP LEVEL KPIS
    # ---------------------------------------------------------
    kpis = get_kpis(df_view)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faturamento Total", f"R$ {kpis['faturamento']:,.2f}")
    c2.metric("Volume (un)", f"{kpis['volume']:,.0f}")
    c3.metric("Clientes Ativos", kpis['clientes'])
    c4.metric("Cidades Cobertas", kpis['cidades'])

    # ---------------------------------------------------------
    # SECTION B: RANKING (Only if 'Todos')
    # ---------------------------------------------------------
    if selected_rep_name == "Todos":
        st.markdown("---")
        st.subheader("🏆 Ranking de Representantes")
        
        df_ranking = get_ranking(df_view)
        
        if not df_ranking.empty:
            col_chart, col_table = st.columns([2, 1])
            
            with col_chart:
                chart_rank = alt.Chart(df_ranking).mark_bar(cornerRadius=3).encode(
                    x=alt.X('Faturamento', title='Faturamento (R$)', axis=alt.Axis(format='~s')),
                    y=alt.Y('Representante', sort='-x', title=None),
                    color=alt.Color('Faturamento', scale=alt.Scale(scheme='blues'), legend=None),
                    tooltip=['Representante', alt.Tooltip('Faturamento', format=',.2f'), 'Volume']
                ).properties(height=max(300, len(df_ranking) * 25))
                st.altair_chart(chart_rank, use_container_width=True)
            
            with col_table:
                df_show = df_ranking.copy()
                df_show['Faturamento'] = df_show['Faturamento'].apply(lambda x: f"R$ {x:,.2f}")
                st.dataframe(df_show, hide_index=True, use_container_width=True, height=300)

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION C: SAÚDE DA CARTEIRA & DISTRIBUIÇÃO
    # ---------------------------------------------------------
    col_health, col_details = st.columns([1, 2])

    with col_health:
        st.subheader("🏥 Saúde da Carteira")
        
        # We need the FULL dataset (df_all) to calculate history for churn
        # Filter for specific rep if needed
        if selected_rep_name != "Todos":
            df_full_history = df_all[df_all["Representante"] == selected_rep_name]
        else:
            df_full_history = df_all

        health = compute_carteira_health(df_full_history, start_date, end_date)
        
        st.metric("Score de Saúde", f"{health['score']} / 100", delta=health['label'], delta_color="off")
        st.caption(f"Churn Rate (Perda): {health['churn']}%")
        
        # Summary Table
        summary_df = health['details'].groupby("Status").agg(
            qtd_clientes=("Cliente", "count"),
            valor_atual=("ValorAtual", "sum")
        ).reset_index()
        
        order_map = {"Novos": 1, "Crescendo": 2, "Estáveis": 3, "Caindo": 4, "Perdidos": 5}
        summary_df['order'] = summary_df['Status'].map(order_map).fillna(99)
        summary_df = summary_df.sort_values('order').drop(columns='order')
        
        st.dataframe(
            summary_df,
            column_config={
                "Status": "Status",
                "qtd_clientes": "Qtd Clientes",
                "valor_atual": st.column_config.NumberColumn("Receita Atual", format="R$ %.2f")
            },
            hide_index=True
        )

    with col_details:
        st.subheader("📍 Análise Geográfica")
        
        if not df_map_view.empty and 'lat' in df_map_view.columns:
            # Group by City for map
            df_map_agg = df_map_view.groupby(["Cidade", "Estado", "lat", "lon"])["Valor"].sum().reset_index()
            df_map_agg = df_map_agg[df_map_agg["Valor"] > 0]
            
            fig_map = px.scatter_mapbox(
                df_map_agg,
                lat="lat", lon="lon",
                size="Valor",
                color="Valor",
                hover_name="Cidade",
                hover_data={"Estado": True, "Valor": ":.2f", "lat": False, "lon": False},
                zoom=3,
                center={"lat": -15.7, "lon": -47.8},
                mapbox_style="carto-positron",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            # --- PDF Button Integration ---
            if st.button("📄 Gerar Relatório PDF"):
                with st.spinner("Gerando PDF..."):
                    try:
                        # Prepare charts for PDF
                        png_bytes, _ = _plotly_to_png_bytes(fig_map)
                        charts = {"map": png_bytes} # Expand this with other charts
                        
                        pdf_data = build_pdf_report(
                            selected_rep_name, 
                            f"{start_date} - {end_date}",
                            kpis,
                            charts
                        )
                        st.download_button(
                            label="📥 Baixar PDF",
                            data=pdf_data,
                            file_name=f"Relatorio_{selected_rep_name}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Erro no PDF: {e}")
        else:
            st.warning("Sem dados geográficos para o filtro selecionado.")

else:
    st.info("👈 Selecione os filtros na barra lateral e clique em 'Atualizar Dashboard' para começar.")
