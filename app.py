# ============================================================
# AGROFORECAST - MÓDULO ADAPTADO A NECESIDADES Y PLAN
# ============================================================
import io
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AgroForecast | Necesidades y Plan", layout="wide")

# ============================================================
# UTILIDADES Y CONFIGURACIÓN
# ============================================================
def normalize_reference(s):
    s = s.fillna("").astype(str).str.strip()
    return s.replace({
        "Extrafino": "Fino",
        "EXTRAFINO": "Fino",
        "extrafino": "Fino",
    })

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_week_date(year, week):
    if pd.isna(year) or pd.isna(week):
        return pd.NaT
    try:
        return pd.to_datetime(f"{int(year)}-W{int(week):02d}-1", format="%G-W%V-%u", errors="coerce")
    except Exception:
        return pd.NaT

# ============================================================
# LECTURA Y ANÁLISIS DE DATOS HISTÓRICOS
# ============================================================
def prepare_model(t6):
    d = t6.copy()
    d["CantidadV"] = pd.to_numeric(d["CantidadV"], errors="coerce").fillna(1).clip(lower=1)
    d["Area"] = pd.to_numeric(d["Area"], errors="coerce")
    d["Kilos"] = pd.to_numeric(d["Kilos"], errors="coerce")
    d["Semana"] = pd.to_numeric(d["Semana"], errors="coerce")
    d["Año"] = pd.to_numeric(d["Año"], errors="coerce")

    d["AreaEfectiva"] = d["Area"] / d["CantidadV"]
    d["SemanaInicio"] = [make_week_date(y, w) for y, w in zip(d["Año"], d["Semana"])]
    d = d[d["SemanaInicio"].notna()].copy()

    keys = ["Finca", "Lote", "Ciclo", "Referencia"]
    cycles = d.groupby(keys, dropna=False).agg(
        Area=("AreaEfectiva", "first"),
        TotalKilos=("Kilos", "sum"),
        PrimeraCosecha=("SemanaInicio", "min"),
        UltimaCosecha=("SemanaInicio", "max"),
        AñoCosecha=("Año", "max"),
    ).reset_index()

    duration = ((cycles["UltimaCosecha"] - cycles["PrimeraCosecha"]).dt.days / 7) + 1
    cycles["DuracionReal"] = pd.to_numeric(duration.round(), errors="coerce")
    cycles["Rendimiento"] = cycles["TotalKilos"] / cycles["Area"].replace(0, np.nan)

    max_year = int(d["Año"].max()) if not d["Año"].empty else 2026
    
    # Clasificación por rangos temporales solicitados
    def get_periodo(y):
        if pd.isna(y): return "Histórico"
        if y >= 2025: return "2025 y 2026"
        return "<25"

    cycles["PeriodoGrupo"] = cycles["AñoCosecha"].apply(get_periodo)
    d = d.merge(cycles[keys + ["DuracionReal", "Rendimiento", "AñoCosecha", "PeriodoGrupo"]], on=keys, how="left")

    relative = ((d["SemanaInicio"] - d["PrimeraCosecha"]).dt.days / 7) + 1
    d["SemanaRelativa"] = pd.to_numeric(relative.round(), errors="coerce")
    d = d[d["SemanaRelativa"].notna()].copy()
    d["SemanaRelativa"] = d["SemanaRelativa"].astype(int)

    return d, cycles

# ============================================================
# CONSTRUCCIÓN DE MATRICES TIPO "NECESIDADES"
# ============================================================
def build_necesidades_matrix(d, cycles, vegetable):
    sub_cycles = cycles[cycles["Referencia"] == vegetable]
    if sub_cycles.empty:
        return None, None

    # Curvas porcentuales por periodo y semana relativa
    sub_d = d[d["Referencia"] == vegetable].copy()
    
    # Calcular totales por ciclo para sacar porcentajes
    cycle_totals = sub_d.groupby(["Finca", "Lote", "Ciclo", "PeriodoGrupo"], dropna=False)["Kilos"].sum().reset_index(name="TotalCiclo")
    sub_d = sub_d.merge(cycle_totals, on=["Finca", "Lote", "Ciclo", "PeriodoGrupo"], how="left")
    sub_d["PctCosecha"] = sub_d["Kilos"] / sub_d["TotalCiclo"].replace(0, np.nan)

    curva_periodo = sub_d.groupby(["PeriodoGrupo", "SemanaRelativa"], dropna=False)["PctCosecha"].mean().reset_index()
    
    # Pivotes para simular los rangos H2:K5 y H9:N12 de la pestaña necesidades
    pivot_curva = curva_periodo.pivot(index="PeriodoGrupo", columns="SemanaRelativa", values="PctCosecha").fillna(0)

    # Rendimientos por semana y periodo (similar a E2:F55 / P2:S55)
    rend_periodo = sub_cycles.groupby("PeriodoGrupo")["Rendimiento"].mean().to_dict()

    return pivot_curva, rend_periodo

# ============================================================
# INTERFAZ DE USUARIO STREAMLIT
# ============================================================
st.title("🌱 AgroForecast — Módulo de Necesidades y Planificación")
st.markdown("Adaptado a la estructura de la pestaña **necesidades** y cálculo de plan por lote.")

with st.sidebar:
    st.header("Carga de Datos")
    uploaded = st.file_uploader("Carga tu archivo Excel histórico", type=["xlsx", "xls"])

if uploaded is None:
    st.info("Por favor, carga un archivo Excel para iniciar el análisis.")
    st.stop()

try:
    raw = pd.read_excel(uploaded, sheet_name=0, header=None)
    left = raw.iloc[:, 1:14].copy()
    left.columns = ["Finca", "Lote", "Area", "Ciclo", "Codigo", "Vegetal", "Referencia", "CantidadV", "DuracionSC", "Kilos", "Semana", "Año", "Mes"]
    left["Referencia"] = normalize_reference(left["Referencia"])
    left = safe_numeric(left, ["Area", "Ciclo", "CantidadV", "Kilos", "Semana", "Año"])
    
    data, cycles = prepare_model(left)
except Exception as e:
    st.error(fError al procesar el archivo: {e}")
    st.stop()

vegetables = sorted(data["Referencia"].dropna().unique().tolist())

tabs = st.tabs(["🧠 Pestaña Necesidades", "🗓️ Pestaña Plan (Simulación)"])

# ============================================================
# TAB 1: NECESIDADES
# ============================================================
with tabs[0]:
    st.subheader("Análisis Estadístico por Cultivo (<25 vs 2025-2026)")
    veg_sel = st.selectbox("Seleccione Vegetal", vegetables, key="nec_veg")

    pivot_curva, rend_dict = build_necesidades_matrix(data, cycles, veg_sel)

    if pivot_curva is not None and not pivot_curva.empty:
        st.markdown("### 📊 Curva Porcentual de Producción por Corte (Semana Relativa)")
        st.dataframe(pivot_curva.style.format("{:.2%}"), use_container_width=True)

        st.markdown("### 💰 Rendimiento Promedio por Periodo (kg/ha)")
        rend_df = pd.DataFrame(list(rend_dict.items()), columns=["Periodo", "Rendimiento Promedio (kg/ha)"])
        st.dataframe(rend_df.style.format({"Rendimiento Promedio (kg/ha)": "{:,.0f}"}), use_container_width=True, hide_index=True)
    else:
        st.warning("No hay suficientes datos para este vegetal.")

# ============================================================
# TAB 2: PLAN
# ============================================================
with tabs[1]:
    st.subheader("Simulación de Planificación de Siembra y Cosecha")
    st.markdown("Cálculo: $\\text{Producción} = \\text{Área del Lote} \\times \\% \\text{ Curva de Producción} \\times \\text{ Rendimiento Plan}$")

    col1, col2, col3 = st.columns(3)
    p_veg = col1.selectbox("Vegetal para Plan", vegetables, key="plan_veg")
    p_area = col2.number_input("Área del Lote (HA)", min_value=0.1, value=1.0, step=0.1)
    p_rend = col3.number_input("Rendimiento Plan (kg/ha)", min_value=100.0, value=11900.0, step=100.0)

    pivot_c, _ = build_necesidades_matrix(data, cycles, p_veg)
    
    if pivot_c is not None and not pivot_c.empty:
        # Tomar por defecto la curva del periodo más reciente o recomendada
        periodo_ref = "2025 y 2026" if "2025 y 2026" in pivot_c.index else pivot_c.index[0]
        curva_vector = pivot_c.loc[periodo_ref]
        
        plan_rows = []
        for semana_rel, pct in curva_vector.items():
            if pct > 0:
                prod = p_area * pct * p_rend
                plan_rows.append({
                    "Semana Cosecha Plan": int(semana_rel),
                    "Curva de Producción": pct,
                    "Rendimiento Plan": p_rend,
                    "Producción Parcial (kg)": prod
                })
        
        plan_df = pd.DataFrame(plan_rows)
        
        st.markdown(f"### 📋 Resultado del Plan para {p_veg} (Usando curva de {periodo_ref})")
        st.dataframe(
            plan_df.style.format({
                "Curva de Producción": "{:.2%}",
                "Rendimiento Plan": "{:,.0f}",
                "Producción Parcial (kg)": "{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        total_prod = plan_df["Producción Parcial (kg)"].sum()
        st.metric("Producción Total Estimada del Lote", f"{total_prod:,.0f} kg")
    else:
        st.warning("Seleccione un vegetal válido para generar el plan.")
