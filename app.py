import io
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AgroForecast | Módulo de Necesidades y Planificación", layout="wide")

# ============================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ============================================================
def normalize_reference(s):
    s = s.fillna("").astype(str).str.strip()
    return s.replace({
        "Extrafino": "Fino",
        "EXTRAFINO": "Fino",
        "extrafino": "Fino",
    })

def read_excel_file(file):
    raw = pd.read_excel(file, sheet_name=0, header=None)
    
    # Tabla 6: B:N (Columnas 1 a 13)
    left = raw.iloc[:, 1:14].copy()
    left.columns = [
        "Finca", "Lote", "Area", "Ciclo", "Codigo", "Vegetal",
        "Referencia", "CantidadV", "DuracionSC", "Kilos",
        "Semana", "Año", "Mes"
    ]
    left["Finca"] = left["Finca"].astype(str).str.strip()
    left["Lote"] = left["Lote"].astype(str).str.strip()
    left["Referencia"] = normalize_reference(left["Referencia"])
    
    cols_num_left = ["Area", "Ciclo", "Codigo", "CantidadV", "DuracionSC", "Kilos", "Semana", "Año", "Mes"]
    for c in cols_num_left:
        left[c] = pd.to_numeric(left[c], errors="coerce")
        
    left = left[left["Finca"].ne("nan") & left["Lote"].ne("nan") & left["Ciclo"].notna() & left["Kilos"].notna()].copy()
    
    # Tabla 10: P:Z (Columnas 15 a 25)
    right = raw.iloc[:, 15:26].copy()
    right.columns = [
        "Finca", "Lote", "Area", "Ciclo", "Vegetal", "Referencia",
        "CantidadV", "DuracionSC", "Total", "Rendimiento", "RendimientoReal"
    ]
    right["Finca"] = right["Finca"].astype(str).str.strip()
    right["Lote"] = right["Lote"].astype(str).str.strip()
    right["Referencia"] = normalize_reference(right["Referencia"])
    
    cols_num_right = ["Area", "Ciclo", "CantidadV", "DuracionSC", "Total", "Rendimiento", "RendimientoReal"]
    for c in cols_num_right:
        right[c] = pd.to_numeric(right[c], errors="coerce")
        
    right = right[right["Finca"].ne("nan") & right["Lote"].ne("nan") & right["Ciclo"].notna()].copy()
    
    return left, right

def prepare_cycles(t6):
    d = t6.copy()
    d["CantidadV"] = d["CantidadV"].fillna(1).clip(lower=1)
    d["AreaEfectiva"] = d["Area"] / d["CantidadV"]
    
    d["SemanaInicio"] = [
        pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u", errors="coerce")
        if pd.notna(y) and pd.notna(w) else pd.NaT
        for y, w in zip(d["Año"], d["Semana"])
    ]
    d = d[d["SemanaInicio"].notna()].copy()
    
    keys = ["Finca", "Lote", "Ciclo", "Referencia"]
    cycles = d.groupby(keys, dropna=False).agg(
        Area=("AreaEfectiva", "first"),
        TotalKilos=("Kilos", "sum"),
        PrimeraCosecha=("SemanaInicio", "min"),
        UltimaCosecha=("SemanaInicio", "max"),
        AñoCosecha=("Año", "max"),
        SemanasObservadas=("SemanaInicio", "nunique")
    ).reset_index()
    
    duration = ((cycles["UltimaCosecha"] - cycles["PrimeraCosecha"]).dt.days / 7) + 1
    cycles["DuracionReal"] = pd.to_numeric(duration.round(), errors="coerce")
    cycles["Rendimiento"] = cycles["TotalKilos"] / cycles["Area"].replace(0, np.nan)
    
    d = d.merge(cycles[keys + ["DuracionReal", "Rendimiento", "AñoCosecha"]], on=keys, how="left")
    
    relative = ((d["SemanaInicio"] - d["PrimeraCosecha"]).dt.days / 7) + 1
    d["SemanaRelativa"] = pd.to_numeric(relative.round(), errors="coerce").astype(int)
    d = d[d["SemanaRelativa"] > 0].copy()
    
    return d, cycles

def get_vegetable_curves(d, cycles, vegetable):
    x = d[d["Referencia"] == vegetable].copy()
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None
        
    cycle_keys = ["Finca", "Lote", "Ciclo", "Referencia"]
    
    def compute_period_curve(subset):
        if subset.empty:
            return pd.DataFrame()
        weekly = subset.groupby(cycle_keys + ["SemanaRelativa"], as_index=False).agg(Kilos=("Kilos", "sum"))
        totals = weekly.groupby(cycle_keys)["Kilos"].transform("sum")
        weekly["Pct"] = weekly["Kilos"] / totals.replace(0, np.nan)
        
        curve = weekly.groupby("SemanaRelativa")["Pct"].median().reset_index()
        curve.columns = ["Semana", "Porcentaje"]
        total_pct = curve["Porcentaje"].sum()
        if total_pct > 0:
            curve["Porcentaje"] /= total_pct
        return curve

    sub_old = x[x["AñoCosecha"] < 2025]
    sub_recent = x[x["AñoCosecha"] >= 2025]
    
    curve_old = compute_period_curve(sub_old)
    curve_recent = compute_period_curve(sub_recent)
    curve_all = compute_period_curve(x)
    
    cy_veg = cycles[cycles["Referencia"] == vegetable]
    rend_old = float(cy_veg.loc[cy_veg["AñoCosecha"] < 2025, "Rendimiento"].median()) if not cy_veg.empty else 0.0
    rend_recent = float(cy_veg.loc[cy_veg["AñoCosecha"] >= 2025, "Rendimiento"].median()) if not cy_veg.empty else 0.0
    rend_rec = float(cy_veg["Rendimiento"].median()) * 1.05 if not cy_veg.empty else 10000.0
    
    return curve_old, curve_recent, curve_all, {"old": rend_old, "recent": rend_recent, "recommended": rend_rec}

# ============================================================
# INTERFAZ DE STREAMLIT
# ============================================================
st.title("🌾 AgroForecast | Módulo de Necesidades y Planificación de Lotes")
st.markdown(
    "Modelo analítico para comparar el comportamiento histórico (<2025) frente al reciente (2025–2026), "
    "analizar la evolución de duración de cosecha y proyectar la pestaña **Plan**."
)

uploaded = st.file_uploader("Sube tu archivo Excel (ej. Analisis final.xlsx o requisitos.xlsx)", type=["xlsx", "xls"])
if uploaded is None:
    st.info("Sube tu archivo Excel para generar la pestaña de necesidades y el plan de siembra.")
    st.stop()

try:
    t6, t10 = read_excel_file(uploaded)
    data, cycles = prepare_cycles(t6)
except Exception as e:
    st.error(f"Error procesando el archivo: {e}")
    st.stop()

vegetables = sorted(data["Referencia"].dropna().unique().tolist())

tab1, tab2, tab3 = st.tabs(["📋 Pestaña Necesidades", "🗓️ Pestaña Plan (Simulador)", "📊 Detalle de Ciclos"])

with tab1:
    st.subheader("📋 Análisis de Necesidades y Curvas por Vegetal")
    veg_sel = st.selectbox("Selecciona Vegetal para analizar", vegetables, key="necesidades_veg")
    
    c_old, c_recent, c_all, yields = get_vegetable_curves(data, cycles, veg_sel)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento <2025 (Histórico)", f"{yields['old']:,.0f} kg/ha" if not np.isnan(yields['old']) else "N/D")
    col2.metric("Rendimiento 2025-2026 (Reciente)", f"{yields['recent']:,.0f} kg/ha" if not np.isnan(yields['recent']) else "N/D")
    col3.metric("Rendimiento Recomendado", f"{yields['recommended']:,.0f} kg/ha")
    
    st.markdown("### 📊 Comparativa de Curvas de Producción (% por Corte / Semana Relativa)")
    
    if not c_old.empty or not c_recent.empty:
        merged = pd.merge(c_old, c_recent, on="Semana", how="outer", suffixes=("_Antiguo", "_Reciente")).fillna(0)
        if not c_all.empty:
            merged = pd.merge(merged, c_all, on="Semana", how="left").rename(columns={"Porcentaje": "_Recomendado"})
        
        st.dataframe(merged.style.format("{:.2%}"), use_container_width=True)
        
        plot_df = merged.melt(id_vars=["Semana"], value_vars=[c for c in merged.columns if c != "Semana"], var_name="Periodo", value_name="Porcentaje")
        fig = px.bar(plot_df, x="Semana", y="Porcentaje", color="Periodo", barmode="group", labels={"Semana": "Semana de Cosecha (Corte)", "Porcentaje": "% de Producción"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay suficientes datos de curvas para este vegetal.")

with tab2:
    st.subheader("🗓️ Simulador de la Pestaña Plan (Programación de Siembra)")
    st.markdown("Calcula automáticamente la producción semanal multiplicando: **Área del Lote × % Curva de Producción × Rendimiento Plan**.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    veg_plan = col_p1.selectbox("Vegetal para Plan", vegetables, key="plan_veg")
    finca_name = col_p2.text_input("Nombre de Finca", value="CH")
    lote_name = col_p3.text_input("Nombre de Lote", value="CH01")
    
    col_p4, col_p5 = st.columns(2)
    area_lote = col_p4.number_input("Área del Lote (ha)", min_value=0.1, value=0.6, step=0.1)
    
    _, _, c_rec, yields_p = get_vegetable_curves(data, cycles, veg_plan)
    default_rend = yields_p["recommended"] if not np.isnan(yields_p["recommended"]) else 10900.0
    
    rend_plan = col_p5.number_input("Rendimiento Plan (kg/ha)", min_value=100.0, value=float(default_rend), step=100.0)
    
    if not c_rec.empty:
        plan_df = c_rec.copy()
        plan_df.rename(columns={"Porcentaje": "Curva_Produccion"}, inplace=True)
        plan_df["Finca"] = finca_name
        plan_df["Lote"] = lote_name
        plan_df["Area"] = area_lote
        plan_df["Rendimiento_Plan"] = rend_plan
        plan_df["Produccion_Kilos"] = plan_df["Area"] * plan_df["Curva_Produccion"] * plan_df["Rendimiento_Plan"]
        
        display_plan = plan_df[["Semana", "Curva_Produccion", "Rendimiento_Plan", "Produccion_Kilos"]].copy()
        display_plan["Curva_Produccion"] = display_plan["Curva_Produccion"].map(lambda x: f"{x:.1%}")
        display_plan["Rendimiento_Plan"] = display_plan["Rendimiento_Plan"].map(lambda x: f"{x:,.0f} kg/ha")
        display_plan["Produccion_Kilos"] = display_plan["Produccion_Kilos"].map(lambda x: f"{x:,.1f} kg")
        
        st.markdown(f"### 📋 Programación para Finca: **{finca_name}** | Lote: **{lote_name}** ({area_lote} ha)")
        st.dataframe(display_plan, use_container_width=True, hide_index=True)
        
        total_kilos_plan = plan_df["Produccion_Kilos"].sum()
        st.metric("Producción Total Estimada del Ciclo", f"{total_kilos_plan:,.1f} kg")
        
        fig_plan = px.bar(plan_df, x="Semana", y="Produccion_Kilos", labels={"Semana": "Semana de Cosecha", "Produccion_Kilos": "Kilos Proyectados"}, title="Proyección de Kilos por Semana de Cosecha")
        st.plotly_chart(fig_plan, use_container_width=True)
    else:
        st.warning("No hay curva disponible para generar el plan.")

with tab3:
    st.subheader("📊 Resumen General de Ciclos Registrados")
    st.dataframe(cycles.head(50), use_container_width=True)
