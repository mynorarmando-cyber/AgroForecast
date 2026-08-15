import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="NAVGAR | Sistema de Pronóstico y Planificación Agrícola",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 NAVGAR | Módulo Analítico, Necesidades y Planificación")
st.markdown("Sistema de inteligencia agronómica para el análisis de rendimiento histórico, curvas de producción por vegetal y simulación de siembras.")

# Barra lateral para carga de archivos
st.sidebar.header("📁 Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo 'Analisis final.xlsx'", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info("👈 Por favor, sube tu archivo Excel **'Analisis final.xlsx'** en la barra lateral para inicializar el sistema.")
    st.stop()

# ============================================================
# MOTOR DE PROCESAMIENTO MATEMÁTICO Y ESTADÍSTICO
# ============================================================
@st.cache_data
def process_agricultural_data(file):
    df_raw = pd.read_excel(file, sheet_name=0, header=None)
    
    # Lectura de la tabla izquierda (Detalle histórico de cosechas)
    left = df_raw.iloc[2:, 1:14].copy()
    left.columns = [
        "Finca", "Lote", "Area", "Ciclo", "Codigo", "Vegetal",
        "Referencia", "CantidadV", "DuracionSC", "Kilos",
        "Semana", "Anio", "Mes"
    ]
    
    # Conversión numérica de columnas clave
    num_cols = ["Area", "Ciclo", "Codigo", "CantidadV", "DuracionSC", "Kilos", "Semana", "Anio"]
    for c in num_cols:
        left[c] = pd.to_numeric(left[c], errors="coerce")
        
    left["Finca"] = left["Finca"].astype(str).str.strip()
    left["Lote"] = left["Lote"].astype(str).str.strip()
    left["Referencia"] = left["Referencia"].fillna("").astype(str).str.strip()
    
    # Normalización de referencias (unificación de Fino / Extrafino)
    left["Referencia"] = left["Referencia"].replace({
        "Extrafino": "Fino", "EXTRAFINO": "Fino", "extrafino": "Fino"
    })
    
    # Filtrar filas vacías o corruptas
    left = left[left["Finca"].ne("nan") & left["Lote"].ne("nan") & left["Ciclo"].notna() & left["Kilos"].notna()].copy()
    
    # Cálculo de Área Efectiva (si hay 2 vegetales en el lote, se divide el área)
    left["CantidadV"] = left["CantidadV"].fillna(1).clip(lower=1)
    left["AreaEfectiva"] = left["Area"] / left["CantidadV"]
    
    # Generación de fechas aproximadas para cálculo de semanas relativas
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
    
    # Agrupación por Ciclo de Producción
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
    
    # Incorporar duración y rendimiento al dataframe detallado
    left = left.merge(cycles, on=keys, how="left", suffixes=("", "_cycle"))
    relative = ((left["FechaSemana"] - left["PrimeraCosecha"]).dt.days / 7) + 1
    left["SemanaRelativa"] = pd.to_numeric(relative.round(), errors="coerce").astype(int)
    left = left[left["SemanaRelativa"] > 0].copy()
    
    return left, cycles

try:
    data, cycles = process_agricultural_data(uploaded_file)
    st.sidebar.success("¡Datos cargados y procesados con éxito!")
except Exception as e:
    st.error(f"Error al procesar el archivo: {e}")
    st.stop()

vegetables = sorted(data["Referencia"].dropna().unique().tolist())

# ============================================================
# PESTAÑAS DE LA APLICACIÓN
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📋 Pestaña Necesidades (Curvas)", 
    "🗓️ Simulador Pestaña Plan", 
    "📊 Base de Ciclos Históricos"
])

# ------------------------------------------------------------
# TAB 1: NECESIDADES Y CURVAS
# ------------------------------------------------------------
with tab1:
    st.subheader("📋 Análisis de Comportamiento y Curvas de Producción")
    st.markdown("Comparativa entre el comportamiento histórico (**<2025**) y reciente (**2025-2026**), reflejando cambios estructurales en la duración de la cosecha y distribución porcentual.")
    
    veg_sel = st.selectbox("Seleccione el Vegetal para Analizar", vegetables, key="sel_veg_nec")
    
    sub_data = data[data["Referencia"] == veg_sel].copy()
    sub_cycles = cycles[cycles["Referencia"] == veg_sel].copy()
    
    # Métricas de rendimiento
    rend_old = sub_cycles[sub_cycles["AnioCosecha"] < 2025]["Rendimiento"].median()
    rend_rec = sub_cycles[sub_cycles["AnioCosecha"] >= 2025]["Rendimiento"].median()
    rend_rec_val = float(rend_rec) * 1.05 if not np.isnan(rend_rec) else float(rend_old) * 1.05
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Rendimiento Histórico (<2025)", f"{rend_old:,.0f} kg/ha" if not np.isnan(rend_old) else "N/D")
    col_m2.metric("Rendimiento Reciente (2025-2026)", f"{rend_rec:,.0f} kg/ha" if not np.isnan(rend_rec) else "N/D")
    col_m3.metric("Rendimiento Recomendado (Plan)", f"{rend_rec_val:,.0f} kg/ha")
    
    # Función para calcular curva porcentual por semana relativa
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
    
    st.markdown("### 📊 Matriz Comparativa de Curvas (%)")
    merged_c = pd.merge(c_old, c_rec, on="Semana", how="outer", suffixes=("_Histórico (<2025)", "_Reciente (2025-2026)")).fillna(0)
    if not c_all.empty:
        merged_c = pd.merge(merged_c, c_all, on="Semana", how="left").rename(columns={"Porcentaje": "_Recomendada"})
        
    st.dataframe(merged_c.style.format("{:.2%}"), use_container_width=True)
    
    # Gráfica de curvas
    df_plot = merged_c.melt(id_vars=["Semana"], var_name="Periodo", value_name="Porcentaje")
    fig = px.bar(df_plot, x="Semana", y="Porcentaje", color="Periodo", barmode="group", title=f"Curva de Cosecha Semanal: {veg_sel}")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 2: SIMULADOR PESTAÑA PLAN
# ------------------------------------------------------------
with tab2:
    st.subheader("🗓️ Simulador de Siembra y Proyección (Pestaña Plan)")
    st.markdown("Fórmula aplicada: **Área del Lote × % Curva de Producción × Rendimiento Plan**")
    
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
        
        fig_plan = px.bar(plan_table, x="Semana", y="Kilos_Proyectados", title=f"Proyección Semanal de Cosecha: {p_veg} (Finca {p_finca}, Lote {p_lote})")
        st.plotly_chart(fig_plan, use_container_width=True)
    else:
        st.warning("No hay suficientes datos registrados para generar la curva de este vegetal.")

# ------------------------------------------------------------
# TAB 3: BASE DE CICLOS
# ------------------------------------------------------------
with tab3:
    st.subheader("📊 Resumen General de Ciclos Históricos Detectados")
    st.markdown("Detalle consolidado por cada ciclo de siembra, área efectiva, kilos totales y duración real calculada.")
    st.dataframe(cycles, use_container_width=True)
