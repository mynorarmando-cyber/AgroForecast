import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="NAVGAR | Sistema de Pronóstico y Planificación Agrícola",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 NAVGAR | Módulo Analítico, Necesidades y Planificación")
st.markdown("Sistema de inteligencia agronómica para el análisis de rendimiento histórico, curvas de producción por vegetal y simulación de siembras.")

# 📁 Selector dinámico de archivos para evitar FileNotFoundError
st.sidebar.header("📁 Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo 'Analisis final.xlsx'", type=["xlsx", "xls"])

if uploaded_file is None:
    st.warning("⚠️ Por favor, sube tu archivo Excel en la barra lateral para iniciar el análisis.")
    st.stop()

# ============================================================
# MOTOR DE PROCESAMIENTO MATEMÁTICO Y ESTADÍSTICO
# ============================================================
@st.cache_data
def process_agricultural_data(file):
    df_raw = pd.read_excel(file, sheet_name=0, header=None)
    
    left = df_raw.iloc[2:, 1:14].copy()
    left.columns = [
        "Finca", "Lote", "Area", "Ciclo", "Codigo", "Vegetal",
        "Referencia", "CantidadV", "DuracionSC", "Kilos",
        "Semana", "Anio", "Mes"
    ]
    
    num_cols = ["Area", "Ciclo", "Codigo", "CantidadV", "DuracionSC", "Kilos", "Semana", "Anio"]
    for c in num_cols:
        left[c] = pd.to_numeric(left[c], errors="coerce")
        
    left["Finca"] = left["Finca"].astype(str).str.strip()
    left["Lote"] = left["Lote"].astype(str).str.strip()
    left["Referencia"] = left["Referencia"].fillna("").astype(str).str.strip()
    left["Referencia"] = left["Referencia"].replace({
        "Extrafino": "Fino", "EXTRAFINO": "Fino", "extrafino": "Fino"
    })
    
    left = left[left["Finca"].ne("nan") & left["Lote"].ne("nan") & left["Ciclo"].notna() & left["Kilos"].notna()].copy()
    
    left["CantidadV"] = left["CantidadV"].fillna(1).clip(lower=1)
    left["AreaEfectiva"] = left["Area"] / left["CantidadV"]
    
    dates = []
    for y, w in zip(left["Anio"], left["Semana"]):
        try:
            dt = pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
        except:
            try:
                dt = pd.to_datetime(f"{int(y)}-01-01") + pd.Timedelta(weeks=int(w)-1)
            except:
                dt = pd.NaT
        dates.append(dt)
    left["FechaSemana"] = dates
    left = left[left["FechaSemana"].notna()].copy()
    
    keys = ["Finca", "Lote", "Ciclo", "Referencia"]
    cycles = left.groupby(keys, dropna=False).agg(
        Area=("AreaEfectiva", "first"),
        TotalKilos=("Kilos", "sum"),
        PrimeraCosecha=("FechaSemana", "min"),
        UltimaCosecha=("FechaSemana", "max"),
        AnioCosecha=("Anio", "max")
    ).reset_index()
    
    cycles["DuracionReal"] = ((cycles["UltimaCosecha"] - cycles["PrimeraCosecha"]).dt.days / 7) + 1
    cycles["DuracionReal"] = pd.to_numeric(cycles["DuracionReal"].round(), errors="coerce")
    cycles["Rendimiento"] = cycles["TotalKilos"] / cycles["Area"].replace(0, np.nan)
    
    left = left.merge(cycles, on=keys, how="left", suffixes=("", "_cycle"))
    relative = ((left["FechaSemana"] - left["PrimeraCosecha"]).dt.days / 7) + 1
    left["SemanaRelativa"] = pd.to_numeric(relative.round(), errors="coerce").astype(int)
    left = left[left["SemanaRelativa"] > 0].copy()
    
    return left, cycles

try:
    data, cycles = process_agricultural_data(uploaded_file)
    st.sidebar.success("¡Datos procesados correctamente!")
except Exception as e:
    st.error(f"Error al procesar el archivo Excel: {e}")
    st.stop()

vegetables = sorted(data["Referencia"].dropna().unique().tolist())

# ============================================================
# INTERFAZ Y PESTAÑAS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📋 Pestaña Necesidades (Curvas)", 
    "🗓️ Simulador Pestaña Plan", 
    "📊 Base de Ciclos Históricos"
])

with tab1:
    st.subheader("📋 Análisis de Comportamiento y Curvas de Producción")
    veg_sel = st.selectbox("Seleccione el Vegetal", vegetables, key="sel_veg_nec")
    
    sub_data = data[data["Referencia"] == veg_sel].copy()
    sub_cycles = cycles[cycles["Referencia"] == veg_sel].copy()
    
    rend_old = sub_cycles[sub_cycles["AnioCosecha"] < 2025]["Rendimiento"].median()
    rend_rec = sub_cycles[sub_cycles["AnioCosecha"] >= 2025]["Rendimiento"].median()
    rend_rec_val = float(rend_rec) * 1.05 if not np.isnan(rend_rec) else float(rend_old) * 1.05
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Rendimiento Histórico (<2025)", f"{rend_old:,.0f} kg/ha" if not np.isnan(rend_old) else "N/D")
    col_m2.metric("Rendimiento Reciente (2025-2026)", f"{rend_rec:,.0f} kg/ha" if not np.isnan(rend_rec) else "N/D")
    col_m3.metric("Rendimiento Recomendado (Plan)", f"{rend_rec_val:,.0f} kg/ha")
    
    def calc_curve(subset):
        if subset.empty:
            return pd.DataFrame()
        w = subset.groupby(["Finca", "Lote", "Ciclo", "Referencia", "SemanaRelativa"], as_index=False).agg(Kilos=("Kilos", "sum"))
        tot = w.groupby(["Finca", "Lote", "Ciclo", "Referencia"])["Kilos"].transform("sum")
        w["Pct"] = w["Kilos"] / tot.replace(0, np.nan)
        c = w.groupby("SemanaRelativa")["Pct"].median().reset_index()
        c.columns = ["Semana", "Porcentaje"]
        t = c["Porcentaje"].sum()
        if t > 0:
            c["Porcentaje"] /= t
        return c

    c_old = calc_curve(sub_data[sub_data["AnioCosecha"] < 2025])
    c_rec = calc_curve(sub_data[sub_data["AnioCosecha"] >= 2025])
    c_all = calc_curve(sub_data)
    
    merged_c = pd.merge(c_old, c_rec, on="Semana", how="outer", suffixes=("_Histórico (<2025)", "_Reciente (2025-2026)")).fillna(0)
    if not c_all.empty:
        merged_c = pd.merge(merged_c, c_all, on="Semana", how="left").rename(columns={"Porcentaje": "_Recomendada"})
        
    st.dataframe(merged_c.style.format("{:.2%}"), use_container_width=True)
    
    df_plot = merged_c.melt(id_vars=["Semana"], var_name="Periodo", value_name="Porcentaje")
    fig = px.bar(df_plot, x="Semana", y="Porcentaje", color="Periodo", barmode="group", title=f"Curva de Cosecha Semanal: {veg_sel}")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🗓️ Simulador de Siembra y Proyección (Pestaña Plan)")
    p_veg = st.selectbox("Seleccione Vegetal para Plan", vegetables, key="sel_veg_plan")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    p_finca = col_p1.text_input("Finca", value="CH")
    p_lote = col_p2.text_input("Lote", value="CH01")
    p_area = col_p3.number_input("Área del Lote (ha)", value=0.6, step=0.1, min_value=0.1)
    
    sub_cyc_p = cycles[cycles["Referencia"] == p_veg]
    def_rend_p = float(sub_cyc_p["Rendimiento"].median() * 1.05) if not sub_cyc_p.empty else 10000.0
    p_rend = st.number_input("Rendimiento Plan (kg/ha)", value=def_rend_p, step=100.0, min_value=100.0)
    
    sub_v_p = data[data["Referencia"] == p_veg]
    curve_p = calc_curve(sub_v_p)
    
    if not curve_p.empty:
        plan_table = curve_p.copy()
        plan_table.rename(columns={"Porcentaje": "Curva_Produccion"}, inplace=True)
        plan_table["Finca"] = p_finca
        plan_table["Lote"] = p_lote
        plan_table["Area"] = p_area
        plan_table["Rendimiento_Plan"] = p_rend
        plan_table["Kilos_Proyectados"] = plan_table["Area"] * plan_table["Curva_Produccion"] * plan_table["Rendimiento_Plan"]
        
        display_df = plan_table[["Semana", "Curva_Produccion", "Rendimiento_Plan", "Kilos_Proyectados"]].copy()
        display_df["Curva_Produccion"] = display_df["Curva_Produccion"].map(lambda x: f"{x:.2%}")
        display_df["Rendimiento_Plan"] = display_df["Rendimiento_Plan"].map(lambda x: f"{x:,.0f} kg/ha")
        display_df["Kilos_Proyectados"] = display_df["Kilos_Proyectados"].map(lambda x: f"{x:,.1f} kg")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        tot_kilos = plan_table["Kilos_Proyectados"].sum()
        st.metric("Producción Total Estimada del Ciclo", f"{tot_kilos:,.1f} kg")
        
        fig_plan = px.bar(plan_table, x="Semana", y="Kilos_Proyectados", title=f"Proyección Semanal: {p_veg} (Finca {p_finca}, Lote {p_lote})")
        st.plotly_chart(fig_plan, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para generar la curva de este vegetal.")

with tab3:
    st.subheader("📊 Resumen General de Ciclos Históricos")
    st.dataframe(cycles, use_container_width=True)
