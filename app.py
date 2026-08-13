import io
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AgroForecast | Modelo y Planificador Agrícola", layout="wide")

# ------------------------------------------------------------
# 1. Utilidades y Funciones de Procesamiento de Datos
# ------------------------------------------------------------
def clean_name(x):
    return str(x).strip() if pd.notna(x) else ""

def weighted_quantile(values, quantile, weights=None):
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    v = v[mask]
    if weights is None:
        return float(np.quantile(v, quantile)) if len(v) else np.nan
    w = np.asarray(weights, dtype=float)[mask]
    if len(v) == 0:
        return np.nan
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return float(np.interp(quantile, cum, v))

def recency_weight(year, max_year):
    age = max_year - year
    if age <= 2:
        return 0.50
    if age <= 4:
        return 0.30
    return 0.20

def normalize_reference(s):
    s = s.astype(str).str.strip()
    # Unificación de Fino y Extrafino
    return s.replace({
        "Extrafino": "Fino",
        "EXTRAFINO": "Fino",
        "extrafino": "Fino",
    })

def read_excel_data(file="Analisis final.xlsx"):
    raw = pd.read_excel(file, sheet_name=0, header=None)

    # Tabla 6 (Histórico semanal)
    left = raw.iloc[:, 1:14].copy()
    left.columns = [
        "Finca","Lote","Area","Ciclo","Codigo","Vegetal","Referencia",
        "CantidadV","DuracionSC","Kilos","Semana","Año","Mes"
    ]
    left = left[left["Finca"].notna() & left["Lote"].notna() & left["Ciclo"].notna()]
    left["Referencia"] = normalize_reference(left["Referencia"])
    for c in ["Area","Ciclo","Codigo","CantidadV","DuracionSC","Kilos","Semana","Año"]:
        left[c] = pd.to_numeric(left[c], errors="coerce")
    left = left.dropna(subset=["Kilos","Semana","Año","Area"])

    # Tabla 10 complementaria
    right = raw.iloc[:, 15:26].copy()
    right.columns = [
        "Finca","Lote","Area","Ciclo","Vegetal","Referencia","CantidadV",
        "DuracionSC","Total","Rendimiento","RendimientoReal"
    ]
    right = right[right["Finca"].notna() & right["Lote"].notna() & right["Ciclo"].notna()]
    right["Referencia"] = normalize_reference(right["Referencia"])
    for c in ["Area","Ciclo","CantidadV","DuracionSC","Total","Rendimiento","RendimientoReal"]:
        right[c] = pd.to_numeric(right[c], errors="coerce")

    return left, right

def prepare_model(t6):
    d = t6.copy()
    d["CantidadV"] = d["CantidadV"].fillna(1).clip(lower=1)
    d["AreaEfectiva"] = d["Area"] / d["CantidadV"]
    d["Semana"] = d["Semana"].astype(int)
    d["Año"] = d["Año"].astype(int)

    # Fecha de lunes de la semana ISO
    d["SemanaInicio"] = pd.to_datetime(
        d["Año"].astype(str) + "-W" + d["Semana"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce"
    )

    keys = ["Finca","Lote","Ciclo","Referencia"]
    cycles = (
        d.groupby(keys, dropna=False)
        .agg(
            Area=("AreaEfectiva","first"),
            TotalKilos=("Kilos","sum"),
            PrimeraCosecha=("SemanaInicio","min"),
            UltimaCosecha=("SemanaInicio","max"),
            AñoCosecha=("Año","max"),
            SemanasSC=("SemanaInicio","nunique"),
        )
        .reset_index()
    )
    cycles["DuracionReal"] = (
        ((cycles["UltimaCosecha"] - cycles["PrimeraCosecha"]).dt.days / 7) + 1
    ).round().astype(int)
    cycles["Rendimiento"] = cycles["TotalKilos"] / cycles["Area"].replace(0, np.nan)

    d = d.merge(
        cycles[keys + ["PrimeraCosecha","DuracionReal","Rendimiento"]],
        on=keys, how="left", suffixes=("","_c")
    )
    d["SemanaRelativa"] = (
        ((d["SemanaInicio"] - d["PrimeraCosecha"]).dt.days / 7) + 1
    ).round().astype(int)

    max_year = int(d["Año"].max())
    cycles["PesoRecencia"] = cycles["AñoCosecha"].apply(lambda y: recency_weight(y, max_year))
    cycles["Periodo"] = cycles["AñoCosecha"].apply(lambda y: "2025 y 2026" if y >= 2025 else "<25")

    return d, cycles

def cycle_stats(cycles, vegetable=None):
    x = cycles.copy()
    if vegetable and vegetable != "Todos":
        x = x[x["Referencia"] == vegetable]
    if x.empty:
        return {}
    vals = x["Rendimiento"].dropna().values
    w = x.loc[x["Rendimiento"].notna(), "PesoRecencia"].values
    return {
        "n": len(vals),
        "mean": float(np.average(vals, weights=w)) if len(vals) else np.nan,
        "median": float(weighted_quantile(vals, .50, w)),
        "p25": float(weighted_quantile(vals, .25, w)),
        "p75": float(weighted_quantile(vals, .75, w)),
        "min": float(np.min(vals)) if len(vals) else np.nan,
        "max": float(np.max(vals)) if len(vals) else np.nan,
    }

def build_curve_comparative(d, vegetable):
    """
    Construye las curvas segmentadas por periodo (<25 vs 2025-2026) y la curva recomendada
    tal como se solicitó en los requisitos de análisis temporal.
    """
    x = d[d["Referencia"] == vegetable].copy()
    if x.empty:
        return pd.DataFrame()
    
    x["Periodo"] = x["Año"].apply(lambda y: "2025 y 2026" if y >= 2025 else "<25")
    
    cyc = (
        x.groupby(["Periodo","Finca","Lote","Ciclo","Referencia","SemanaRelativa"], as_index=False)
         .agg(Kilos=("Kilos","sum"))
    )
    total = cyc.groupby(["Periodo","Finca","Lote","Ciclo","Referencia"])["Kilos"].transform("sum")
    cyc["Pct"] = cyc["Kilos"] / total.replace(0, np.nan)

    # Consolidado por periodo y semana relativa
    out_periods = []
    for periodo, gp in cyc.groupby("Periodo"):
        for sw, g in gp.groupby("SemanaRelativa"):
            med = float(g["Pct"].median())
            out_periods.append({"Periodo": periodo, "SemanaRelativa": int(sw), "Porcentaje": med})
    df_p = pd.DataFrame(out_periods)
    
    # Pivoteamos para tener columnas <25 y 2025 y 2026
    pivot = df_p.pivot(index="SemanaRelativa", columns="Periodo", values="Porcentaje").reset_index()
    if "<25" not in pivot.columns:
        pivot["<25"] = 0.0
    if "2025 y 2026" not in pivot.columns:
        pivot["2025 y 2026"] = 0.0
        
    pivot = pivot.fillna(0)
    
    # Curva recomendada (Ponderación 2025-2026 con mayor peso por cambio de duración, ej. brócoli de 4 a 6-7 semanas)
    # Se da un 70% de peso a 2025 y 2026 y 30% a <25
    pivot["Recomendada"] = pivot["2025 y 2026"] * 0.70 + pivot["<25"] * 0.30
    # Normalizar recomendada a suma 1.0
    s_rec = pivot["Recomendada"].sum()
    if s_rec > 0:
        pivot["Recomendada"] = pivot["Recomendada"] / s_rec
        
    return pivot

def seasonality(d, vegetable):
    x = d[d["Referencia"] == vegetable].copy()
    if x.empty:
        return pd.DataFrame(columns=["Semana","FactorEstacional"])
    
    weekly = (
        x.groupby(["Año","Semana","Finca","Lote","Ciclo","Referencia"], as_index=False)
         .agg(Kilos=("Kilos","sum"), Area=("AreaEfectiva","first"))
    )
    weekly["KgHa"] = weekly["Kilos"] / weekly["Area"].replace(0, np.nan)
    max_year = int(x["Año"].max())
    weekly["Peso"] = weekly["Año"].apply(lambda y: recency_weight(y, max_year))
    base = np.average(weekly["KgHa"].dropna(), weights=weekly.loc[weekly["KgHa"].notna(),"Peso"])
    rows = []
    for wk, g in weekly.groupby("Semana"):
        m = np.average(g["KgHa"].dropna(), weights=g.loc[g["KgHa"].notna(),"Peso"]) if g["KgHa"].notna().any() else np.nan
        rows.append((int(wk), m / base if base else 1))
    out = pd.DataFrame(rows, columns=["Semana","FactorEstacional"]).sort_values("Semana")
    out["FactorEstacional"] = out["FactorEstacional"].rolling(5, center=True, min_periods=1).mean()
    return out

def forecast(cycles, d, vegetable, area, first_harvest, scenario):
    stats = cycle_stats(cycles, vegetable)
    if not stats:
        return None, None, None

    if scenario == "Conservador":
        base = stats["p25"]
    elif scenario == "Optimista":
        base = stats["p75"]
    else:
        base = stats["median"]

    curve_df = build_curve_comparative(d, vegetable)
    if curve_df.empty:
        return None, stats, None

    seas = seasonality(d, vegetable)
    
    rows = []
    for _, r in curve_df.iterrows():
        rel = int(r["SemanaRelativa"])
        harvest_date = first_harvest + timedelta(weeks=rel-1)
        iso = harvest_date.isocalendar()
        sw = int(iso.week)
        sf = float(seas.loc[seas["Semana"] == sw, "FactorEstacional"].iloc[0]) if (seas["Semana"] == sw).any() else 1.0
        rows.append({
            "Semana relativa": rel,
            "Fecha": harvest_date,
            "Semana año": sw,
            "Curva recomendada %": r["Recomendada"],
            "Factor estacional": sf,
        })
    out = pd.DataFrame(rows)
    out["Peso ajustado"] = out["Curva recomendada %"] * out["Factor estacional"]
    s_adj = out["Peso ajustado"].sum()
    if s_adj > 0:
        out["Peso ajustado"] = out["Peso ajustado"] / s_adj
        
    out["Rendimiento proyectado kg/ha"] = base
    out["Kilos proyectados"] = area * out["Rendimiento proyectado kg/ha"] * out["Peso ajustado"]

    return out, stats, {"duration": int(curve_df["SemanaRelativa"].max())}

# ------------------------------------------------------------
# 2. Interfaz Gráfica Principal (Streamlit)
# ------------------------------------------------------------
st.title("🌱 AgroForecast — Modelo Gerencial y Planificador Agrícola")
st.caption("Análisis histórico comparativo (<2025 vs. 2025-2026), curvas recomendadas y programación de siembras.")

with st.sidebar:
    st.header("Fuente de Datos")
    uploaded = st.file_uploader("Cargar archivo Excel", type=["xlsx","xls"])
    st.markdown("**Nota:** Si tienes `Analisis final.xlsx` en la misma carpeta, se carga automáticamente.")
    st.divider()
    st.header("Criterios del Modelo")
    st.write("• Fino + Extrafino unificados")
    st.write("• Área efectiva = Área / Cantidad V")
    st.write("• Análisis segmentado: <2025 y 2025-2026")
    st.write("• Curva recomendada ponderada")

source = uploaded if uploaded is not None else "Analisis final.xlsx"

try:
    t6, t10 = read_excel_data(source)
    data, cycles = prepare_model(t6)
except Exception as e:
    st.error("No se encontró el archivo 'Analisis final.xlsx'. Por favor, súbelo usando el botón de la barra lateral.")
    st.stop()

vegetables = sorted(data["Referencia"].dropna().unique().tolist())
st.success(f"Datos cargados con éxito: {len(data):,} registros y {len(cycles):,} ciclos agrícolas analizados.")

tabs = st.tabs([
    "📊 Dashboard Ejecutivo", 
    "📈 Curvas & Comparativa Temporal", 
    "🔮 Pronóstico por Vegetal", 
    "📋 Planificador de Lotes & Rendimiento", 
    "🧪 Trazabilidad & Calidad"
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ciclos Totales", f"{len(cycles):,}")
    c2.metric("Vegetales", f"{len(vegetables)}")
    c3.metric("Años Históricos", f"{int(data.Año.min())}–{int(data.Año.max())}")
    c4.metric("Kilos Históricos", f"{data.Kilos.sum():,.0f}")

    st.subheader("Rendimiento Mediano por Vegetal (kg/ha)")
    ranking = (
        cycles.groupby("Referencia")
        .agg(Rendimiento=("Rendimiento","median"), Ciclos=("Rendimiento","count"))
        .reset_index().sort_values("Rendimiento", ascending=False)
    )
    st.dataframe(ranking.style.format({"Rendimiento": "{:,.0f}"}), use_container_width=True, hide_index=True)

    fig = px.bar(ranking, x="Referencia", y="Rendimiento", hover_data=["Ciclos"],
                 labels={"Rendimiento": "Rendimiento Mediano (kg/ha)", "Referencia": "Vegetal"})
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Análisis de Curvas: Histórico (<2025) vs. Reciente (2025-2026) y Recomendada")
    veg_c = st.selectbox("Seleccione Vegetal para Curva", vegetables, key="curv_veg")
    
    curve_comp = build_curve_comparative(data, veg_c)
    if curve_comp.empty:
        st.warning("No hay suficientes datos para este vegetal.")
    else:
        # Mostrar tabla resumen estilo pestaña necesidades
        st.markdown(f"**Comparativa de distribución porcentual de cosecha para: {veg_c}**")
        display_tbl = curve_comp.copy()
        for col in ["<25", "2025 y 2026", "Recomendada"]:
            if col in display_tbl.columns:
                display_tbl[col] = display_tbl[col].map(lambda x: f"{x:.1%}")
        st.dataframe(display_tbl.rename(columns={"SemanaRelativa": "Semana de Cosecha"}), use_container_width=True, hide_index=True)

        # Gráfico comparativo
        fig_c = px.line(
            curve_comp.melt(id_vars=["SemanaRelativa"], value_vars=["<25", "2025 y 2026", "Recomendada"], 
                            var_name="Periodo / Modelo", value_name="Porcentaje"),
            x="SemanaRelativa", y="Porcentaje", color="Periodo / Modelo", markers=True,
            labels={"SemanaRelativa": "Semana de Cosecha Relativa", "Porcentaje": "Porcentaje de Cosecha"}
        )
        fig_c.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_c, use_container_width=True)

with tabs[2]:
    st.subheader("Motor de Pronóstico y Rendimiento Futuro")
    col1, col2, col3 = st.columns(3)
    vegf = col1.selectbox("Vegetal", vegetables, key="forecastveg")
    area = col2.number_input("Área del Lote (ha)", min_value=0.01, value=1.0, step=0.1)
    first_harvest = col3.date_input("Fecha Estimada de Primera Cosecha", value=date.today())

    scenario = st.radio("Escenario de Rendimiento", ["Conservador (P25)", "Probable (Mediana)", "Optimista (P75)"], horizontal=True)
    sc_map = {"Conservador (P25)": "Conservador", "Probable (Mediana)": "Probable", "Optimista (P75)": "Optimista"}
    
    result, stats, meta = forecast(cycles, data, vegf, area, first_harvest, sc_map[scenario])

    if result is not None:
        total_kilos = result["Kilos proyectados"].sum()
        rend_ha = result["Rendimiento proyectado kg/ha"].iloc[0]
        dur_sem = result["Semana relativa"].max()

        a, b, c = st.columns(3)
        a.metric("Producción Total Estimada", f"{total_kilos:,.0f} kg")
        b.metric("Rendimiento Proyectado", f"{rend_ha:,.0f} kg/ha")
        c.metric("Duración de Cosecha", f"{dur_sem} semanas")

        st.markdown("**Desglose Semanal del Pronóstico**")
        show_res = result.copy()
        show_res["Curva recomendada %"] = show_res["Curva recomendada %"].map(lambda x: f"{x:.1%}")
        show_res["Peso ajustado"] = show_res["Peso ajustado"].map(lambda x: f"{x:.1%}")
        show_res["Kilos proyectados"] = show_res["Kilos proyectados"].map(lambda x: f"{x:,.0f} kg")
        st.dataframe(show_res, use_container_width=True, hide_index=True)

        fig_p = px.bar(result, x="Fecha", y="Kilos proyectados",
                       labels={"Kilos proyectados": "Kilos Proyectados", "Fecha": "Semana de Cosecha"})
        st.plotly_chart(fig_p, use_container_width=True)

        csv_data = result.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar Pronóstico en CSV", csv_data, f"pronostico_{vegf}.csv", "text/csv")
    else:
        st.warning("No hay suficiente histórico para generar el pronóstico de este vegetal.")

with tabs[3]:
    st.subheader("Planificador de Lotes y Programación de Cosecha")
    st.markdown("Simula la programación de siembras multiplicando el **Área del Lote × Porcentaje de la Curva Recomendada × Rendimiento Plan**.")
    
    col_l1, col_l2, col_l3 = st.columns(3)
    finca_sel = col_l1.selectbox("Finca", sorted(data["Finca"].dropna().unique()))
    lotes_disponibles = sorted(data[data["Finca"] == finca_sel]["Lote"].dropna().unique())
    lote_sel = col_l2.selectbox("Lote", lotes_disponibles)
    
    # Obtener área promedio del lote
    lote_data = data[(data["Finca"] == finca_sel) & (data["Lote"] == lote_sel)]
    default_area = float(lote_data["AreaEfectiva"].mean()) if not lote_data.empty else 1.0
    area_lote = col_l3.number_input("Área del Lote (ha)", min_value=0.01, value=default_area, step=0.1)

    veg_plan = st.selectbox("Vegetal a Programar", vegetables, key="plan_veg")
    rend_plan = st.number_input("Rendimiento Plan (kg/ha)", min_value=100.0, value=10900.0, step=500.0)
    siembra_date = st.date_input("Fecha de Siembra / Inicio Cosecha", value=date.today())

    # Generar tabla de plan
    curve_p = build_curve_comparative(data, veg_plan)
    if not curve_p.empty:
        plan_rows = []
        for _, row in curve_p.iterrows():
            sem_rel = int(row["SemanaRelativa"])
            pct = row["Recomendada"]
            sem_date = siembra_date + timedelta(weeks=sem_rel-1)
            kilos_plan = area_lote * pct * rend_plan
            plan_rows.append({
                "Semana de Cosecha": sem_rel,
                "Fecha": sem_date,
                "Curva Recomendada %": pct,
                "Área Lote (ha)": area_lote,
                "Rendimiento Plan (kg/ha)": rend_plan,
                "Kilos Programados": kilos_plan
            })
        df_plan = pd.DataFrame(plan_rows)
        
        # Mostrar resumen superior
        tot_plan_kilos = df_plan["Kilos Programados"].sum()
        a, b = st.columns(2)
        a.metric("Producción Total Programada", f"{tot_plan_kilos:,.0f} kg")
        b.metric("Duración del Ciclo Planificado", f"{len(df_plan)} semanas")

        # Tabla formateada
        show_plan = df_plan.copy()
        show_plan["Curva Recomendada %"] = show_plan["Curva Recomendada %"].map(lambda x: f"{x:.1%}")
        show_plan["Kilos Programados"] = show_plan["Kilos Programados"].map(lambda x: f"{x:,.0f}")
        st.dataframe(show_plan, use_container_width=True, hide_index=True)

        fig_plan = px.bar(df_plan, x="Fecha", y="Kilos Programados",
                          title=f"Programación de Cosecha — Finca: {finca_sel} | Lote: {lote_sel} ({veg_plan})",
                          labels={"Kilos Programados": "Kilos", "Fecha": "Semana"})
        st.plotly_chart(fig_plan, use_container_width=True)
    else:
        st.warning("No hay datos de curva suficientes para este vegetal.")

with tabs[4]:
    st.subheader("Auditoría de Calidad y Trazabilidad de Datos")
    q_kilos = data["Kilos"].isna().sum()
    q_area = data["Area"].isna().sum()
    q_date = data["SemanaInicio"].isna().sum()
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Registros sin Fecha Válida", int(q_date))
    col_b.metric("Registros sin Kilos", int(q_kilos))
    col_c.metric("Registros sin Área", int(q_area))

    st.markdown("**Resumen de Ciclos Analizados**")
    st.dataframe(cycles[["Finca", "Lote", "Ciclo", "Referencia", "Area", "TotalKilos", "Rendimiento", "DuracionReal", "AñoCosecha", "Periodo"]].head(50), use_container_width=True, hide_index=True)
