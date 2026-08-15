import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AgroForecast | Modelo Matemático y de Necesidades", layout="wide")

# ============================================================
# CARGA Y PROCESAMIENTO MATEMÁTICO DEL ARCHIVO
# ============================================================
@st.cache_data
def load_and_process_data(file):
    df_raw = pd.read_excel(file, sheet_name=0, header=None)
    
    # Tabla Izquierda (Detalle de Cosechas Semanales)
    left = df_raw.iloc[2:, 1:14].copy()
    left.columns = [
        "Finca", "Lote", "Area", "Ciclo", "Codigo", "Vegetal",
        "Referencia", "CantidadV", "DuracionSC", "Kilos",
        "Semana", "Anio", "Mes"
    ]
    
    for c in ["Area", "Ciclo", "Codigo", "CantidadV", "DuracionSC", "Kilos", "Semana", "Anio"]:
        left[c] = pd.to_numeric(left[c], errors="coerce")
        
    left["Finca"] = left["Finca"].astype(str).str.strip()
    left["Lote"] = left["Lote"].astype(str).str.strip()
    left["Referencia"] = left["Referencia"].fillna("").astype(str).str.strip()
    left["Referencia"] = left["Referencia"].replace({
        "Extrafino": "Fino", "EXTRAFINO": "Fino", "extrafino": "Fino"
    })
    
    left = left[left["Finca"].ne("nan") & left["Lote"].ne("nan") & left["Ciclo"].notna() & left["Kilos"].notna()].copy()
    
    # Cálculo de Fechas y Semanas Relativas por Ciclo
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
        AnioCosecha=("Anio", "max"),
        SemanasObservadas=("FechaSemana", "nunique")
    ).reset_index()
    
    duration = ((cycles["UltimaCosecha"] - cycles["PrimeraCosecha"]).dt.days / 7) + 1
    cycles["DuracionReal"] = pd.to_numeric(duration.round(), errors="coerce")
    cycles["Rendimiento"] = cycles["TotalKilos"] / cycles["Area"].replace(0, np.nan)
    
    left = left.merge(cycles, on=keys, how="left", suffixes=("", "_cycle"))
    
    relative = ((left["FechaSemana"] - left["PrimeraCosecha"]).dt.days / 7) + 1
    left["SemanaRelativa"] = pd.to_numeric(relative.round(), errors="coerce").astype(int)
    left = left[left["SemanaRelativa"] > 0].copy()
    
    return left, cycles

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================
st.title("🌾 AgroForecast | Modelo de Pronóstico y Curvas de Producción")
st.markdown("Modelo matemático para análisis de ciclos, detección de cambios de duración de cosecha (ej. Brócoli) y proyección de la pestaña **Plan**.")

uploaded_file = st.file_uploader("Sube tu archivo 'Analisis final.xlsx'", type=["xlsx", "xls"])
if uploaded_file is None:
    st.info("Por favor sube el archivo de Excel para inicializar el motor analítico.")
    st.stop()

try:
    data, cycles = load_and_process_data(uploaded_file)
except Exception as e:
    st.error(f"Error procesando el archivo: {e}")
    st.stop()

vegetables = sorted(data["Referencia"].dropna().unique().tolist())

tab1, tab2, tab3 = st.tabs(["📋 Reporte Necesidades & Curvas", "🗓️ Simulador Pestaña Plan", "📊 Estadísticas de Ciclos"])

with tab1:
    st.subheader("📋 Análisis de Curvas y Cambio Estructural por Vegetal")
    veg_sel = st.selectbox("Selecciona Vegetal", vegetables, key="veg_nec")
    
    sub_veg = data[data["Referencia"] == veg_sel].copy()
    sub_cyc = cycles[cycles["Referencia"] == veg_sel].copy()
    
    # Métricas clave de rendimiento
    rend_old = sub_cyc[sub_cyc["AnioCosecha"] < 2025]["Rendimiento"].median()
    rend_rec = sub_cyc[sub_cyc["AnioCosecha"] >= 2025]["Rendimiento"].median()
    rend_rec_val = float(rend_rec) * 1.05 if not np.isnan(rend_rec) else float(rend_old) * 1.05
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Rendimiento Histórico (<2025)", f"{rend_old:,.0f} kg/ha" if not np.isnan(rend_old) else "N/D")
    c2.metric("Rendimiento Reciente (2025-2026)", f"{rend_rec:,.0f} kg/ha" if not np.isnan(rend_rec) else "N/D")
    c3.metric("Rendimiento Recomendado (Plan)", f"{rend_rec_val:,.0f} kg/ha")
    
    # Función para calcular curva por periodo
    def get_curve(subset):
        if subset.empty:
            return pd.DataFrame()
        weekly = subset.groupby(["Finca", "Lote", "Ciclo", "Referencia", "SemanaRelativa"], as_index=False).agg(Kilos=("Kilos", "sum"))
        totals = weekly.groupby(["Finca", "Lote", "Ciclo", "Referencia"])["Kilos"].transform("sum")
        weekly["Pct"] = weekly["Kilos"] / totals.replace(0, np.nan)
        curve = weekly.groupby("SemanaRelativa")["Pct"].median().reset_index()
        curve.columns = ["Semana", "Porcentaje"]
        t_sum = curve["Porcentaje"].sum()
        if t_sum > 0:
            curve["Porcentaje"] /= t_sum
        return curve

    curve_old = get_curve(sub_veg[sub_veg["AnioCosecha"] < 2025])
    curve_recent = get_curve(sub_veg[sub_veg["AnioCosecha"] >= 2025])
    curve_all = get_curve(sub_veg)
    
    st.markdown("### 📊 Matriz Comparativa de Curvas de Cosecha")
    merged_curve = pd.merge(curve_old, curve_recent, on="Semana", how="outer", suffixes=("_Histórico", "_Reciente")).fillna(0)
    if not curve_all.empty:
        merged_curve = pd.merge(merged_curve, curve_all, on="Semana", how="left").rename(columns={"Porcentaje": "_Recomendada"})
    
    st.dataframe(merged_curve.style.format("{:.2%}"), use_container_width=True)
    
    # Gráfica
    plot_m = merged_curve.melt(id_vars=["Semana"], var_name="Periodo", value_name="Porcentaje")
    fig = px.bar(plot_m, x="Semana", y="Porcentaje", color="Periodo", barmode="group", title=f"Evolución de Curva de Cosecha: {veg_sel}")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🗓️ Simulador de Siembra y Proyección (Pestaña Plan)")
    st.markdown("Calcula la producción semanal multiplicando: **Área del Lote × % Curva de Producción × Rendimiento Plan**.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    veg_p = col_p1.selectbox("Vegetal para Plan", vegetables, key="veg_p")
    finca_p = col_p2.text_input("Finca", value="CH")
    lote_p = col_p3.text_input("Lote", value="CH01")
    
    col_p4, col_p5 = st.columns(2)
    area_p = col_p4.number_input("Área del Lote (ha)", min_value=0.1, value=0.6, step=0.1)
    
    # Obtener rendimiento recomendado para el vegetal seleccionado
    sub_cyc_p = cycles[cycles["Referencia"] == veg_p]
    def_rend = float(sub_cyc_p["Rendimiento"].median() * 1.05) if not sub_cyc_p.empty else 10000.0
    rend_p = col_p5.number_input("Rendimiento Plan (kg/ha)", min_value=100.0, value=def_rend, step=100.0)
    
    # Obtener curva recomendada o reciente
    sub_v_p = data[data["Referencia"] == veg_p]
    curve_p = get_curve(sub_v_p)
    
    if not curve_p.empty:
        plan_df = curve_p.copy()
        plan_df.rename(columns={"Porcentaje": "Curva_Produccion"}, inplace=True)
        plan_df["Finca"] = finca_p
        plan_df["Lote"] = lote_p
        plan_df["Area"] = area_p
        plan_df["Rendimiento_Plan"] = rend_p
        plan_df["Produccion_Kilos"] = plan_df["Area"] * plan_df["Curva_Produccion"] * plan_df["Rendimiento_Plan"]
        
        disp_plan = plan_df[["Semana", "Curva_Produccion", "Rendimiento_Plan", "Produccion_Kilos"]].copy()
        disp_plan["Curva_Produccion"] = disp_plan["Curva_Produccion"].map(lambda x: f"{x:.1%}")
        disp_plan["Rendimiento_Plan"] = disp_plan["Rendimiento_Plan"].map(lambda x: f"{x:,.0f} kg/ha")
        disp_plan["Produccion_Kilos"] = disp_plan["Produccion_Kilos"].map(lambda x: f"{x:,.1f} kg")
        
        st.dataframe(disp_plan, use_container_width=True, hide_index=True)
        total_k = plan_df["Produccion_Kilos"].sum()
        st.metric("Producción Total Estimada del Ciclo", f"{total_k:,.1f} kg")
        
        fig_plan = px.bar(plan_df, x="Semana", y="Produccion_Kilos", title="Proyección de Kilos por Semana de Cosecha")
        st.plotly_chart(fig_plan, use_container_width=True)
    else:
        st.warning("No hay suficientes datos de curva para este vegetal.")

with tab3:
    st.subheader("📊 Resumen de Ciclos Históricos Registrados")
    st.dataframe(cycles.head(100), use_container_width=True)
