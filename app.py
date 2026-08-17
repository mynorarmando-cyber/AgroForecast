import streamlit as st
import pandas as pd
from modelo import cargar_tabla6, opciones_disponibles, ejecutar_modelo, generar_reporte_excel

st.set_page_config(page_title="Modelo de Rendimientos", layout="wide")
st.title("🌱 Modelo de rendimientos y curva de cosecha")
st.caption(
    "Sube tu archivo Analisis_final.xlsx (Tabla 6 en la hoja 'Hoja1', columnas B:N). "
    "Se excluyen automáticamente el primer y el último ciclo de cada lote "
    "(captura incompleta / cosecha en curso)."
)

uploaded = st.file_uploader("Archivo Excel con el histórico de cosecha", type=["xlsx"])

if uploaded is None:
    st.info("Sube un archivo Excel para comenzar.")
    st.stop()

t6 = cargar_tabla6(uploaded)
vegetales_disp, fincas_disp = opciones_disponibles(t6)

with st.sidebar:
    st.header("Filtros")
    vegetal_sel = st.selectbox("Vegetal (Referencia)", vegetales_disp)
    # Fino y Extrafino son el mismo vegetal cosechado en distinto punto de madurez
    referencias = [vegetal_sel]
    if vegetal_sel == 'Fino':
        referencias.append('Extrafino')
    elif vegetal_sel == 'Extrafino':
        referencias.append('Fino')

    fincas_sel = st.multiselect("Finca(s)", fincas_disp, default=fincas_disp)

    st.header("Parámetros del modelo")
    maxcorte = st.number_input("Semanas de cosecha a planificar", min_value=1, max_value=10, value=3)
    anio_objetivo = st.number_input(
        "Año objetivo del pronóstico", min_value=2000, max_value=2100,
        value=int(t6['Año'].max()) + 1)
    decay = st.slider(
        "Peso de la recencia (decaimiento por año)", 0.5, 1.0, 0.85, 0.05,
        help="Más bajo = los años recientes pesan mucho más que los antiguos. "
             "1.0 = todos los años pesan igual.")
    excluir_incompletos = st.checkbox(
        "Excluir 1er y último ciclo de cada lote (recomendado)", value=True)

if not fincas_sel:
    st.warning("Selecciona al menos una finca en el panel lateral.")
    st.stop()

try:
    with st.spinner("Calculando modelo estadístico..."):
        resultado = ejecutar_modelo(
            t6, nombre_vegetal=vegetal_sel, referencias_vegetal=referencias,
            fincas=fincas_sel if fincas_sel else None, maxcorte=int(maxcorte),
            anio_objetivo=int(anio_objetivo), decay=decay,
            excluir_incompletos=excluir_incompletos,
        )
except ValueError as e:
    st.error(str(e))
    st.stop()

trend = resultado['trend_stats']
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ciclos analizados", resultado['n_ciclos'])
col2.metric("Ciclos excluidos (incompletos)", resultado['n_filas_excluidas_incompletas'])
col3.metric("Tendencia anual", f"{trend['slope']:,.0f} kg/ha/año",
            help=f"Intervalo de confianza Theil-Sen: [{trend['slope_low']:,.0f}, {trend['slope_high']:,.0f}]")
col4.metric("Significancia", f"p = {trend['p_value']:.1e}",
            help=f"Correlación de rangos (Spearman) = {trend['spearman_rho']:.2f}")

if trend['p_value'] < 0.05:
    st.success(
        f"La tendencia es estadísticamente significativa: el rendimiento "
        f"{'sube' if trend['slope']>0 else 'baja'} de forma consistente año a año, "
        f"no es ruido aleatorio."
    )
else:
    st.info("Con los datos filtrados, la tendencia en el tiempo no es estadísticamente significativa.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Curva de cosecha por año", "📅 Rendimiento real por año",
    "🎯 Recomendación", "🔍 Detalle de ciclos"
])

with tab1:
    st.subheader(f"Curva de cosecha por corte (% del total) — {vegetal_sel}, por año")
    st.caption("Cada columna es un año de inicio de cosecha. La normalización es por año "
               "(cada columna suma 100%). N = número de ciclos que sustentan ese año.")
    curva_tabla = resultado['curva_tabla']
    st.dataframe(curva_tabla.style.format("{:.1%}", na_rep="—"))
    st.bar_chart(curva_tabla.T)
    with st.expander("Tamaño de muestra (N ciclos) por año"):
        st.dataframe(resultado['curva_n'].to_frame('N ciclos'))
    st.markdown("**Curva recomendada** (ponderada por recencia, todos los años):")
    curva_rec_df = resultado['curva_rec'].to_frame('Recomendada')
    curva_rec_df.index.name = 'Corte'
    st.dataframe(curva_rec_df.style.format("{:.1%}"))

with tab2:
    st.subheader(f"Rendimiento real por semana del año — {vegetal_sel}, todos los años")
    st.caption("Cada línea es un año real, del primero al último disponible en los datos filtrados.")
    rend_real = resultado['rend_real_tabla']
    st.line_chart(rend_real)
    with st.expander("Ver tabla (kg/ha) y tamaño de muestra"):
        c1, c2 = st.columns(2)
        c1.write("Rendimiento (kg/ha)")
        c1.dataframe(rend_real.style.format("{:,.0f}", na_rep="—"))
        c2.write("N ciclos por semana/año")
        c2.dataframe(resultado['rend_real_n'])

with tab3:
    st.subheader("Rendimiento recomendado por semana (tendencia + estacionalidad)")
    st.caption(
        f"Tendencia robusta (Theil-Sen) proyectada a {resultado['anio_objetivo']} "
        f"+ estacionalidad por semana ponderada por recencia (decaimiento={decay})."
    )
    comp_df = resultado['rend_real_tabla'].copy()
    comp_df['Recomendado'] = resultado['rend_recomendado']
    st.line_chart(comp_df)
    rec_df = resultado['rend_recomendado'].to_frame('Rendimiento recomendado (kg/ha)')
    rec_df.index.name = 'Semana'
    st.dataframe(rec_df.style.format("{:,.0f}"))

with tab4:
    st.dataframe(resultado['cyc'][
        ['Finca', 'Lote', 'Ciclo', 'SemanaInicio', 'AñoInicio', 'Area', 'Rendimiento', 'resid']
    ].sort_values(['AñoInicio', 'SemanaInicio']))

st.subheader("Descargar reporte completo")
buf = generar_reporte_excel(resultado)
st.download_button(
    f"⬇️ Descargar Necesidades_{vegetal_sel.replace(' ', '_')}.xlsx",
    data=buf, file_name=f"Necesidades_{vegetal_sel.replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

