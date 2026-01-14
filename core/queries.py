# core/queries.py

def get_rep_ranking(start_date: str, end_date: str):
    """
    Returns a DataFrame with sales per Representative, sorted by Value.
    Used only when 'All' representatives are selected.
    """
    query = """
        SELECT 
            Representante, 
            SUM(Valor) as Faturamento,
            SUM(Quantidade) as Volume
        FROM vendas
        WHERE Competencia BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        GROUP BY Representante
        ORDER BY Faturamento DESC
    """
    
    # DuckDB returns a DF directly
    return db_con.execute(query, [start_date, end_date]).df()
