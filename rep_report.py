import flet as ft
import pandas as pd
import altair as alt
import vl_convert as vlc
from fpdf import FPDF
from fpdf.enums import XPos, YPos, Align
from pathlib import Path
from datetime import datetime
import calendar
import tempfile
import os

# --- CONFIG ---
CSV_PATH = Path(r"C:\Users\Cesar\CB Database\Documents\OPENFIELD\APPS\INSIGHTS\data\raw\relatorio_faturamento.csv")
BASE_DIR = CSV_PATH.parent.parent.parent 

# --- VISUAL CONSTANTS ---
# Colors (R, G, B)
C_PRIMARY   = (59, 130, 246)   # Blue
C_SUCCESS   = (34, 197, 94)    # Green
C_DANGER    = (239, 68, 68)    # Red
C_DARK      = (30, 41, 59)     # Slate 900 (Main Text)
C_MED       = (100, 116, 139)  # Slate 500 (Subtitles/Labels)
C_LIGHT     = (241, 245, 249)  # Slate 100 (Backgrounds)
C_BORDER    = (226, 232, 240)  # Slate 200 (Lines)

# Hex for Altair
HEX_PALETTE = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#ef4444"] 

STATUS_ORDER = ["Novos", "Crescendo", "Estáveis", "Caindo", "Perdidos"]
MONTH_NAMES = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}

# --- HELPERS ---
def with_opacity(opacity: float, color: str) -> str: return f"{opacity},{color}"
def format_brl(v): return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
def format_int(v): return f"{int(v):,}".replace(",", ".")
def format_pct(v): return f"{v:.1%}" if not pd.isna(v) else "-"

def format_brl_compact(value):
    val = float(value)
    if val >= 1_000_000: return f"R$ {val/1_000_000:.1f} mi".replace(".", ",")
    if val >= 1_000: return f"R$ {val/1_000:.1f} k".replace(".", ",")
    return f"R$ {val:,.0f}".replace(",", ".")

# --- DATA LOGIC ---
def load_data():
    if not CSV_PATH.exists(): return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    for c in ["Valor", "Quantidade"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    if "MesNum" in df.columns:
        df["Competencia"] = pd.to_datetime(dict(year=df["Ano"], month=df["MesNum"], day=1), errors="coerce")
    elif "Mes" in df.columns:
        df["Competencia"] = pd.to_datetime(dict(year=df["Ano"], month=df["Mes"], day=1), errors="coerce")
    return df

def get_status_classification(row):
    curr, prev = row["ValorAtual"], row["ValorAnterior"]
    if curr > 0 and prev == 0: return "Novos"
    if curr == 0 and prev > 0: return "Perdidos"
    if curr > 0 and prev > 0:
        ratio = curr / prev
        if ratio >= 1.05: return "Crescendo"
        if ratio <= 0.95: return "Caindo"
    return "Estáveis"

def compute_wallet_health(df_comp):
    if df_comp.empty: return 50, "Neutra"
    df_comp["Peso"] = df_comp[["ValorAtual", "ValorAnterior"]].max(axis=1)
    total = df_comp["Peso"].sum()
    if total == 0: return 50, "Neutra"
    
    weights = {"Novos": 1, "Crescendo": 2, "Estáveis": 1, "Caindo": -1, "Perdidos": -2}
    score_sum = 0
    for st, w in weights.items():
        rev = df_comp.loc[df_comp["Status"]==st, "Peso"].sum()
        score_sum += w * (rev / total)
        
    final = max(0, min(100, (score_sum + 2) / 4 * 100))
    if final < 30: lbl = "Crítica"
    elif final < 50: lbl = "Alerta"
    elif final < 70: lbl = "Neutra"
    else: lbl = "Saudável"
    return int(final), lbl

# --- CHARTING (ALTAIR) ---
def save_chart(chart, w=400, h=300):
    chart = chart.configure_view(strokeWidth=0).configure_axis(
        grid=False, domain=False, labelFontSize=10, titleFontSize=11, labelColor="#64748b"
    ).configure_legend(
        labelFontSize=10, titleFontSize=11, labelColor="#64748b"
    ).properties(
        width=w, height=h, background='white'
    )
    png = vlc.vegalite_to_png(chart.to_json(), scale=2)
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    with open(f.name, "wb") as file: file.write(png)
    return f.name

def make_grouped_bar(df_curr, df_prev, metric_col, color_hex):
    # Prepare Data for Grouped Bar
    df1 = df_curr.groupby("Competencia")[metric_col].sum().reset_index()
    df1["Periodo"] = "Atual"
    df1["Month"] = df1["Competencia"].dt.month
    
    df2 = df_prev.groupby("Competencia")[metric_col].sum().reset_index()
    df2["Periodo"] = "Anterior"
    df2["Month"] = df2["Competencia"].dt.month
    
    combined = pd.concat([df1, df2])
    combined["MonthName"] = combined["Month"].map(MONTH_NAMES)
    
    base = alt.Chart(combined).encode(
        x=alt.X("MonthName:N", sort=list(MONTH_NAMES.values()), axis=alt.Axis(title=None, labelAngle=0)),
        xOffset="Periodo:N"
    )
    
    bar = base.mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y(f"{metric_col}:Q", axis=alt.Axis(title=None, format="~s")),
        color=alt.Color("Periodo:N", scale=alt.Scale(domain=["Atual", "Anterior"], range=[color_hex, "#cbd5e1"]), legend=None),
    )
    
    text = base.mark_text(dy=-5, fontSize=9, color="#1e293b").encode(
        text=alt.Text(f"{metric_col}:Q", format=".2s"),
        y=alt.Y(f"{metric_col}:Q")
    )
    
    return save_chart(bar + text, 220, 150)

def make_modern_donut(df, cat_col, val_col, colors=None):
    df = df[df[val_col] > 0].copy()
    base = alt.Chart(df).encode(theta=alt.Theta(val_col, stack=True))
    
    # Smaller hole (innerRadius 40 vs 60 previously)
    pie = base.mark_arc(innerRadius=40, outerRadius=80).encode(
        color=alt.Color(cat_col, scale=alt.Scale(range=colors) if colors else alt.Undefined, 
                        legend=alt.Legend(orient="right", title=None, symbolType="circle")), 
        order=alt.Order(val_col, sort="descending")
    )
    
    return save_chart(pie, 200, 160)

# --- PDF ENGINE ---
class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_margins(10, 10, 10)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        if self.page_no() > 1:
            self.set_y(8)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(*C_MED)
            self.cell(0, 5, "Insights de Vendas - Detalhamento", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
            self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*C_MED)
        self.cell(0, 10, f'{self.page_no()}', align='R')

    def draw_main_header(self, title, sub1, sub2):
        self.set_y(10)
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(*C_DARK)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_font('Helvetica', '', 10)
        self.set_text_color(*C_MED)
        self.cell(0, 5, sub1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 5, sub2, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.ln(4)
        self.set_draw_color(*C_BORDER)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.ln(4)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*C_DARK)
        self.set_fill_color(*C_LIGHT)
        # Full width bar
        self.cell(190, 8, f"  {title}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def kpi_card(self, label, value, subtext, x, y, w):
        self.set_xy(x, y)
        
        # Uppercase Label
        self.set_font('Helvetica', 'B', 7)
        self.set_text_color(*C_MED)
        self.cell(w, 4, label.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Value
        self.set_xy(x, y+5)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(*C_DARK)
        self.cell(w, 7, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Subtext
        if subtext:
            self.set_xy(x, y+13)
            self.set_font('Helvetica', 'B', 8)
            if "Crítica" in subtext or "-" in subtext: self.set_text_color(*C_DANGER)
            elif "Neutra" in subtext: self.set_text_color(*C_MED)
            else: self.set_text_color(*C_SUCCESS)
            self.cell(w, 4, subtext, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def draw_table(self, x, y, df, col_widths, aligns=None):
        self.set_xy(x, y)
        if not aligns: aligns = ['L'] * len(df.columns)
        
        # Header
        self.set_font('Helvetica', 'B', 7)
        self.set_text_color(*C_MED)
        self.set_fill_color(*C_LIGHT) # Light gray header
        self.set_draw_color(*C_BORDER)
        
        for i, (col, w) in enumerate(zip(df.columns, col_widths)):
            # Header text always left or center depending on logic, mostly left for cleanliness
            self.cell(w, 7, str(col).upper(), border='B', fill=True, align=aligns[i])
        self.ln()
        
        # Rows
        self.set_font('Helvetica', '', 7)
        self.set_text_color(*C_DARK)
        
        fill = False
        for _, row in df.iterrows():
            self.set_x(x)
            # Zebra Striping
            if fill: self.set_fill_color(250, 250, 252); self.set_fill_color(255, 255, 255) 
            
            for i, (val, w) in enumerate(zip(row, col_widths)):
                self.cell(w, 6, str(val), border='B', align=aligns[i])
            
            self.ln()
            if self.get_y() > 275: 
                self.add_page()
                self.set_y(15)

# --- GENERATOR ---
def generate_pdf(rep_name, s_date, e_date, ps_date, pe_date):
    s_date = s_date.replace(day=1)
    e_date = e_date.replace(day=calendar.monthrange(e_date.year, e_date.month)[1])
    ps_date = ps_date.replace(day=1)
    pe_date = pe_date.replace(day=calendar.monthrange(pe_date.year, pe_date.month)[1])

    df = load_data()
    if df.empty: return False, "CSV Vazio"
    
    mask_curr = (df["Competencia"] >= s_date) & (df["Competencia"] <= e_date)
    df_curr = df[mask_curr].copy()
    mask_prev = (df["Competencia"] >= ps_date) & (df["Competencia"] <= pe_date)
    df_prev = df[mask_prev].copy()
    
    if rep_name != "Todos":
        df_curr = df_curr[df_curr["Representante"] == rep_name]
        df_prev = df_prev[df_prev["Representante"] == rep_name]
        
    if df_curr.empty: return False, "Sem dados no período atual."

    total_fat = df_curr["Valor"].sum()
    total_vol = df_curr["Quantidade"].sum()
    
    curr_grp = df_curr.groupby(["Cliente", "Cidade", "Estado"])[["Valor", "Quantidade"]].sum().rename(columns={"Valor": "ValorAtual", "Quantidade": "VolAtual"})
    prev_grp = df_prev.groupby(["Cliente", "Cidade", "Estado"])[["Valor", "Quantidade"]].sum().rename(columns={"Valor": "ValorAnterior", "Quantidade": "VolAnterior"})
    comp = curr_grp.join(prev_grp, how="outer").fillna(0).reset_index()
    
    comp["Status"] = comp.apply(get_status_classification, axis=1)
    score, score_lbl = compute_wallet_health(comp)
    
    comp_sorted = comp.sort_values("ValorAtual", ascending=False)
    comp_sorted["Share"] = comp_sorted["ValorAtual"] / total_fat
    comp_sorted["CumShare"] = comp_sorted["Share"].cumsum()
    n80 = comp_sorted[comp_sorted["CumShare"] <= 0.8].shape[0]
    n80_pct = n80 / len(comp) if len(comp) > 0 else 0

    # --- PDF START ---
    pdf = ReportPDF()
    pdf.add_page()
    
    d_fmt = "%b/%Y"
    pdf.draw_main_header(
        "Relatório de Vendas", 
        f"Representante: {rep_name}", 
        f"Período: {s_date.strftime(d_fmt)} a {e_date.strftime(d_fmt)} (vs {ps_date.strftime(d_fmt)} a {pe_date.strftime(d_fmt)})"
    )

    # 1. KPIs
    y_kpi = pdf.get_y()
    gap, w_kpi = 2, 37
    # Clean layout without boxes, just text blocks
    pdf.kpi_card("Faturamento", format_brl_compact(total_fat), None, 10, y_kpi, w_kpi)
    pdf.kpi_card("Volume", format_int(total_vol), "unidades", 10+w_kpi+gap, y_kpi, w_kpi)
    pdf.kpi_card("Saúde Carteira", f"{score}/100", score_lbl, 10+(w_kpi+gap)*2, y_kpi, w_kpi)
    pdf.kpi_card("N80 (Conc.)", f"{n80}", f"{n80_pct:.0%} base", 10+(w_kpi+gap)*3, y_kpi, w_kpi)
    pdf.kpi_card("Clientes Ativos", f"{len(comp_sorted[comp_sorted['ValorAtual']>0])}", "", 10+(w_kpi+gap)*4, y_kpi, w_kpi)
    
    pdf.ln(22)

    # 2. EVOLUÇÃO
    pdf.section_title("1. Evolução Vendas")
    y_charts = pdf.get_y()
    
    img_fat = make_grouped_bar(df_curr, df_prev, "Valor", "#3b82f6")
    img_vol = make_grouped_bar(df_curr, df_prev, "Quantidade", "#22c55e")
    
    pdf.image(img_fat, x=10, y=y_charts, w=90)
    pdf.image(img_vol, x=105, y=y_charts, w=90)
    
    # Summary Table Below Charts
    pdf.set_y(y_charts + 60)
    
    def get_monthly_sums(d, col): return d.groupby(d["Competencia"].dt.month)[col].sum()
    fat_c, fat_p = get_monthly_sums(df_curr, "Valor"), get_monthly_sums(df_prev, "Valor")
    vol_c, vol_p = get_monthly_sums(df_curr, "Quantidade"), get_monthly_sums(df_prev, "Quantidade")
    
    all_months = sorted(set(fat_c.index) | set(fat_p.index))
    t_data = []
    for m in all_months:
        fc, fp = fat_c.get(m, 0), fat_p.get(m, 0)
        vc, vp = vol_c.get(m, 0), vol_p.get(m, 0)
        t_data.append({
            "Mês": MONTH_NAMES.get(m, str(m)),
            "Fat. Atual": format_brl_compact(fc), "Fat. Ant": format_brl_compact(fp), "Fat %": format_pct((fc-fp)/fp if fp else 0),
            "Vol. Atual": format_int(vc), "Vol. Ant": format_int(vp), "Vol %": format_pct((vc-vp)/vp if vp else 0)
        })
    
    # Table Widths: Total ~190
    widths = [20, 28, 28, 15, 28, 28, 15]
    aligns = ['L', 'R', 'R', 'R', 'R', 'R', 'R']
    pdf.draw_table(10, pdf.get_y(), pd.DataFrame(t_data), widths, aligns)
    pdf.ln(5)

    # 3. CATEGORIAS (Page 2)
    pdf.add_page()
    pdf.section_title("2. Categorias")
    y_cat = pdf.get_y()
    
    cats = df_curr.groupby("Categoria")["Valor"].sum().sort_values(ascending=False).reset_index()
    img_cat = make_modern_donut(cats.head(8), "Categoria", "Valor")
    pdf.image(img_cat, x=10, y=y_cat, w=80)
    
    # Table next to Chart
    cat_disp = cats.head(10).copy()
    cat_disp["Fat."] = cat_disp["Valor"].apply(format_brl)
    cat_disp["Share"] = (cat_disp["Valor"] / total_fat).apply(format_pct)
    # Align table right side
    pdf.draw_table(95, y_cat+10, cat_disp[["Categoria", "Fat.", "Share"]], [50, 30, 15], ['L', 'R', 'R'])
    
    pdf.ln(70)

    # 4. DETALHES CARTEIRA
    pdf.section_title("3. Detalhes da Carteira")
    y_st = pdf.get_y()
    
    st_df = comp.groupby("Status").agg({"ValorAtual": "sum", "VolAtual": "sum", "ValorAnterior": "sum", "VolAnterior": "sum"}).reset_index()
    st_df["Order"] = st_df["Status"].apply(lambda x: STATUS_ORDER.index(x) if x in STATUS_ORDER else 99)
    st_df = st_df.sort_values("Order")
    
    img_st = make_modern_donut(st_df, "Status", "ValorAtual", HEX_PALETTE)
    pdf.image(img_st, x=10, y=y_st, w=80)
    
    # Expanded Status Table
    st_table_data = []
    for _, r in st_df.iterrows():
        v_fat = (r["ValorAtual"] - r["ValorAnterior"])/r["ValorAnterior"] if r["ValorAnterior"] else 0
        v_vol = (r["VolAtual"] - r["VolAnterior"])/r["VolAnterior"] if r["VolAnterior"] else 0
        st_table_data.append({
            "Status": r["Status"],
            "Fat.": format_brl_compact(r["ValorAtual"]),
            "Fat %": format_pct(v_fat),
            "Vol.": format_int(r["VolAtual"]),
            "Vol %": format_pct(v_vol)
        })
    
    pdf.draw_table(95, y_st+10, pd.DataFrame(st_table_data), [25, 25, 15, 20, 15], ['L', 'R', 'R', 'R', 'R'])
    pdf.ln(80)

    # 5. LISTA DETALHADA
    pdf.add_page()
    pdf.section_title("4. Lista Detalhada de Clientes")
    
    comp["VarFat"] = comp["ValorAtual"] - comp["ValorAnterior"]
    comp["PctFat"] = comp.apply(lambda x: x["VarFat"]/x["ValorAnterior"] if x["ValorAnterior"]>0 else 0, axis=1)
    
    for status in STATUS_ORDER:
        sub_df = comp[comp["Status"] == status].sort_values("ValorAtual", ascending=False)
        if sub_df.empty: continue
        
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        
        # Color Header by Status
        c_rgb = tuple(int(HEX_PALETTE[STATUS_ORDER.index(status)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        pdf.set_text_color(*c_rgb)
        pdf.cell(0, 8, f"{status} ({len(sub_df)} clientes)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        disp_rows = []
        for _, row in sub_df.iterrows():
            v_vol = (row["VolAtual"] - row["VolAnterior"])/row["VolAnterior"] if row["VolAnterior"] else 0
            disp_rows.append({
                "Cliente": str(row["Cliente"])[:35],
                "Cidade": str(row["Cidade"])[:18],
                "Fat Atual": format_brl_compact(row["ValorAtual"]),
                "Fat Ant": format_brl_compact(row["ValorAnterior"]),
                "Var %": format_pct(row["PctFat"]),
                "Vol": format_int(row["VolAtual"]),
                "Vol %": format_pct(v_vol)
            })
        
        # Detailed Table Widths (Total ~190)
        # Client(60), City(30), F1(25), F0(25), V%(15), Vol(20), V%(15)
        widths = [60, 30, 25, 25, 15, 20, 15]
        aligns = ['L', 'L', 'R', 'R', 'R', 'R', 'R']
        pdf.draw_table(10, pdf.get_y(), pd.DataFrame(disp_rows), widths, aligns)

    # Cleanup
    for img in [img_fat, img_vol, img_cat, img_st]:
        try: os.remove(img)
        except: pass

    filename = f"Relatorio_{rep_name}_{datetime.now().strftime('%H%M')}.pdf"
    path = BASE_DIR / filename
    pdf.output(path)
    return True, str(path)

# --- UI ---
def main(page: ft.Page):
    page.title, page.bgcolor = "Gerador de Relatórios", "#f8f9fa"
    page.window_width, page.window_height = 550, 800
    page.theme_mode = ft.ThemeMode.LIGHT

    df = load_data()
    if df.empty: page.add(ft.Text("Erro: CSV vazio")); return

    y_opts = [ft.dropdown.Option(str(y)) for y in sorted(df["Ano"].dropna().unique().astype(int))]
    m_opts = [ft.dropdown.Option(text=n, key=str(k)) for k,n in MONTH_NAMES.items()]

    def d_row(txt):
        return ft.Row([
            ft.Text(txt, width=40, weight="bold", color=ft.Colors.BLACK),
            ft.Dropdown(options=m_opts, width=110, dense=True, text_size=12, border_color="grey"),
            ft.Dropdown(options=y_opts, width=80, dense=True, text_size=12, border_color="grey")
        ])

    r1, r2 = d_row("Início"), d_row("Fim")
    r3, r4 = d_row("Início"), d_row("Fim")
    
    cy, py = str(datetime.now().year), str(datetime.now().year-1)
    r1.controls[1].value, r1.controls[2].value = "1", cy
    r2.controls[1].value, r2.controls[2].value = "12", cy
    r3.controls[1].value, r3.controls[2].value = "1", py
    r4.controls[1].value, r4.controls[2].value = "12", py

    dd_rep = ft.Dropdown(label="Representante", options=[ft.dropdown.Option("Todos")], value="Todos", text_size=12, border_color="grey")
    lbl = ft.Text("")

    def gen(e):
        lbl.value, lbl.color = "Gerando...", "blue"
        page.update()
        try:
            d = lambda r, end=False: datetime(int(r.controls[2].value), int(r.controls[1].value), calendar.monthrange(int(r.controls[2].value), int(r.controls[1].value))[1] if end else 1)
            ok, msg = generate_pdf(dd_rep.value, d(r1), d(r2, True), d(r3), d(r4, True))
            lbl.value, lbl.color = (f"Sucesso: {Path(msg).name}", "green") if ok else (msg, "red")
        except Exception as ex: lbl.value, lbl.color = str(ex), "red"
        page.update()

    reps = sorted(df["Representante"].unique())
    dd_rep.options = [ft.dropdown.Option("Todos")] + [ft.dropdown.Option(r) for r in reps]

    page.add(ft.Container(
        content=ft.Column([
            ft.Text("Gerador de Relatórios", size=20, weight="bold", color="black"),
            ft.Divider(),
            ft.Text("Período Atual", size=12, weight="bold", color="black"), r1, r2,
            ft.Divider(),
            ft.Text("Comparativo", size=12, weight="bold", color="black"), r3, r4,
            ft.Divider(),
            dd_rep,
            ft.Container(height=10),
            ft.ElevatedButton("Gerar PDF", on_click=gen, style=ft.ButtonStyle(bgcolor="#22c55e", color="white"), height=45, width=400),
            lbl
        ]), padding=20, bgcolor="white", border_radius=10, shadow=ft.BoxShadow(blur_radius=10, color=with_opacity(0.1, "black"))
    ))

if __name__ == "__main__":
    ft.app(target=main)
