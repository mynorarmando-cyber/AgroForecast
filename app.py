import streamlit as st
import pandas as pd
from modelo import ejecutar_modelo, generar_reporte_excel

st.set_page_config(page_title="Modelo de Rendimientos", layout="wide")
st.title("🌱 Modelo de rendimientos y curva de cosecha")
st.caption(
    "Sube tu archivo Analisis_final.xlsx (con Tabla 6 en la hoja 'Hoja1', "
    "columnas B:N) para calcular la curva de cosecha y el rendimiento "
    "recomendado por semana."
)

uploaded = st.file_uploader("Archivo Excel con el histórico de cosecha", type=["xlsx"])

with st.sidebar:
    st.header("Parámetros del vegetal")
    nombre_vegetal = st.text_input("Nombre del vegetal", value="Ejote")
    referencias_raw = st.text_input(
        "Referencia(s) en la Tabla 6 (separadas por coma)", value="Fino, Extrafino")
    maxcorte = st.number_input("Semanas de cosecha a planificar", min_value=1, max_value=10, value=3)
    anio_corte_periodo = st.number_input(
        "Año que separa histórico de 'reciente'", min_value=2000, max_value=2100, value=2025)
    anio_objetivo = st.number_input(
        "Año objetivo del pronóstico", min_value=2000, max_value=2100, value=2027)
    peso_reciente = st.slider("Peso del período reciente", 0.0, 1.0, 0.65, 0.05)
    peso_historico = round(1 - peso_reciente, 2)
    st.caption(f"Peso histórico: {peso_historico}")

if uploaded is not None:
    referencias = [r.strip() for r in referencias_raw.split(",") if r.strip()]
    try:
        with st.spinner("Calculando modelo estadístico..."):
            resultado = ejecutar_modelo(
                uploaded, nombre_vegetal=nombre_vegetal, referencias_vegetal=referencias,
                maxcorte=int(maxcorte), anio_corte_periodo=int(anio_corte_periodo),
                anio_objetivo=int(anio_objetivo), peso_reciente=peso_reciente,
                peso_historico=peso_historico,
            )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    trend = resultado['trend_stats']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ciclos analizados", trend['n_ciclos'])
    col2.metric("Ciclos < 2025", resultado['n_old'])
    col3.metric("Ciclos recientes", resultado['n_new'])
    col4.metric("Tendencia anual", f"{trend['slope']:.0f} kg/ha/año",
                help=f"p-value={trend['p_value']:.2e}, r={trend['r']:.2f}")

    if trend['p_value'] < 0.05:
        st.success(
            f"La tendencia es estadísticamente significativa (p={trend['p_value']:.2e}): "
            f"el rendimiento {'sube' if trend['slope']>0 else 'baja'} de forma consistente en el tiempo."
        )
    else:
        st.info("La tendencia en el tiempo no es estadísticamente significativa con estos datos.")

    st.subheader("Curva de cosecha por corte (% del total)")
    curva_df = pd.DataFrame({
        'Corte': list(range(1, int(maxcorte)+1)),
        f'< {anio_corte_periodo}': [resultado['curva']['<25'][c] for c in range(1, int(maxcorte)+1)],
        'Reciente': [resultado['curva']['25-26'][c] for c in range(1, int(maxcorte)+1)],
        'Recomendada': [resultado['curva']['rec'][c] for c in range(1, int(maxcorte)+1)],
    }).set_index('Corte')
    c1, c2 = st.columns([1, 1])
    c1.dataframe(curva_df.style.format("{:.1%}"))
    c2.bar_chart(curva_df)

    st.subheader("Rendimiento por semana del año (kg/ha)")
    semanal_df = pd.DataFrame({
        f'< {anio_corte_periodo}': resultado['semanal']['<25'],
        'Reciente': resultado['semanal']['25-26'],
        'Recomendado': resultado['semanal']['recomendado'],
    })
    st.line_chart(semanal_df)

    with st.expander("Ver detalle de ciclos usados"):
        st.dataframe(resultado['cyc'][
            ['Finca', 'Lote', 'Ciclo', 'SemanaInicio', 'AñoInicio', 'Rendimiento', 'periodo']
        ])

    st.subheader("Descargar reporte")
    buf = generar_reporte_excel(resultado)
    st.download_button(
        "⬇️ Descargar Necesidades_{}.xlsx".format(nombre_vegetal.replace(" ", "_")),
        data=buf, file_name=f"Necesidades_{nombre_vegetal.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Sube un archivo Excel para comenzar.")
