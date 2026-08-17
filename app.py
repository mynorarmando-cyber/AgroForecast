"""
MODELO ESTADISTICO DE RENDIMIENTOS Y CURVA DE COSECHA - EJOTE
================================================================
Genera el reporte "Necesidades_Ejote.xlsx" (curva de cosecha por corte,
rendimiento semanal <25 / 2025-2026 / recomendado) a partir de las
Tabla 6 y Tabla 10 del archivo Analisis_final.xlsx.

COMO USARLO CUANDO TENGAS MAS DATOS:
1. Actualiza Analisis_final.xlsx con las nuevas semanas de cosecha (Tabla 6).
2. Corre: python3 modelo_rendimientos_ejote.py Analisis_final.xlsx
3. Se genera un nuevo Necesidades_Ejote.xlsx con las 4 tablas actualizadas.

Para replicar este mismo modelo en otro vegetal (Brocoli, Grano, etc.),
cambia la lista REFERENCIAS_VEGETAL abajo.
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIGURACION DEL PILOTO (cambia esto para replicar a otro vegetal)
# ---------------------------------------------------------------------------
REFERENCIAS_VEGETAL = ['Fino', 'Extrafino']   # Referencia(s) que forman "Ejote"
NOMBRE_VEGETAL = 'Ejote'
MAXCORTE_PLANIFICACION = 3    # cuantos cortes se usan para la curva operativa
ANIO_CORTE_PERIODO = 2025     # separa "<25" de "reciente"
ANIO_OBJETIVO_PRONOSTICO = 2027  # anio al que se proyecta la tendencia
PESO_RECIENTE = 0.65          # peso del periodo reciente en curva/estacionalidad recomendada
PESO_HISTORICO = 0.35
WINSOR_PCTL = (0.02, 0.98)    # recorte de atipicos para el rendimiento


def circular_smooth(serie_por_semana, window=3):
    """Media movil circular (semana 53 conecta con semana 1)."""
    weeks = np.arange(1, 54)
    arr = serie_por_semana.reindex(weeks).values.astype(float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        idxs = [(i + k) % n for k in range(-window, window + 1)]
        w = arr[idxs]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = w.mean()
    return pd.Series(out, index=weeks)


def cargar_datos(path_excel):
    t6 = pd.read_excel(path_excel, sheet_name='Hoja1', header=1, usecols="B:N", nrows=14084)
    veg = t6[t6['Referencia'].isin(REFERENCIAS_VEGETAL)].copy()
    return veg


def construir_ciclos(veg):
    grp = veg.groupby(['Finca', 'Lote', 'Ciclo', 'Codigo'], as_index=False).agg(
        Kilos=('Kilos', 'sum'), Semana=('Semana', 'first'),
        Año=('Año', 'first'), Area=('Area', 'first'), CantidadV=('Cantidad V', 'first')
    )
    grp = grp.sort_values(['Finca', 'Lote', 'Ciclo', 'Codigo'])
    grp['corte'] = grp.groupby(['Finca', 'Lote', 'Ciclo']).cumcount() + 1
    grp['cycle_total'] = grp.groupby(['Finca', 'Lote', 'Ciclo'])['Kilos'].transform('sum')
    grp['pct'] = grp['Kilos'] / grp['cycle_total']

    cyc = grp[grp.corte == 1][['Finca', 'Lote', 'Ciclo', 'Semana', 'Año', 'Area', 'CantidadV']].rename(
        columns={'Semana': 'SemanaInicio', 'Año': 'AñoInicio'})
    cyc = cyc.merge(grp.groupby(['Finca', 'Lote', 'Ciclo'])['Kilos'].sum().rename('TotalKilos'),
                     on=['Finca', 'Lote', 'Ciclo'])
    cyc['Rendimiento'] = cyc['TotalKilos'] / (cyc['Area'] * cyc['CantidadV'])
    cyc['periodo'] = np.where(cyc['AñoInicio'] < ANIO_CORTE_PERIODO, '<25', '25-26')

    cortes = grp.merge(cyc[['Finca', 'Lote', 'Ciclo', 'periodo']], on=['Finca', 'Lote', 'Ciclo'])
    return cyc, cortes


def curva_de_cosecha(cortes):
    curva = {}
    for periodo in ['<25', '25-26']:
        sub = cortes[(cortes.periodo == periodo) & (cortes.corte <= MAXCORTE_PLANIFICACION)]
        agg = sub.groupby('corte')['pct'].mean().reindex(range(1, MAXCORTE_PLANIFICACION + 1)).fillna(0)
        curva[periodo] = agg / agg.sum()
    rec = curva['<25'] * PESO_HISTORICO + curva['25-26'] * PESO_RECIENTE
    curva['rec'] = rec / rec.sum()
    return curva


def rendimiento_semanal(cyc):
    lo, hi = cyc['Rendimiento'].quantile(list(WINSOR_PCTL))
    cyc = cyc.copy()
    cyc['Rend_w'] = cyc['Rendimiento'].clip(lo, hi)

    slope, intercept, r, p, se = stats.linregress(cyc['AñoInicio'], cyc['Rend_w'])
    cyc['trend_pred'] = intercept + slope * cyc['AñoInicio']
    cyc['resid'] = cyc['Rend_w'] - cyc['trend_pred']

    semanal = {}
    for periodo in ['<25', '25-26']:
        wm = cyc[cyc.periodo == periodo].groupby('SemanaInicio')['Rend_w'].mean()
        semanal[periodo] = circular_smooth(wm).interpolate(limit_direction='both')

    resid_old = cyc[cyc.periodo == '<25'].groupby('SemanaInicio')['resid'].mean()
    resid_new = cyc[cyc.periodo == '25-26'].groupby('SemanaInicio')['resid'].mean()
    seas_old = circular_smooth(resid_old).interpolate(limit_direction='both')
    seas_new = circular_smooth(resid_new).interpolate(limit_direction='both')
    seasonal = PESO_HISTORICO * seas_old + PESO_RECIENTE * seas_new

    trend_target = intercept + slope * ANIO_OBJETIVO_PRONOSTICO
    semanal['recomendado'] = (trend_target + seasonal).clip(lower=0)

    stats_dict = {'slope': slope, 'intercept': intercept, 'p_value': p, 'r': r,
                  'trend_target': trend_target, 'n_ciclos': len(cyc)}
    return semanal, stats_dict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'Analisis_final.xlsx'
    veg = cargar_datos(path)
    cyc, cortes = construir_ciclos(veg)
    curva = curva_de_cosecha(cortes)
    semanal, trend = rendimiento_semanal(cyc)

    print(f"--- {NOMBRE_VEGETAL}: {trend['n_ciclos']} ciclos analizados ---")
    print(f"Tendencia: {trend['slope']:.0f} kg/ha/año (p={trend['p_value']:.2e})")
    print(f"Curva recomendada (cortes 1..{MAXCORTE_PLANIFICACION}):",
          {k: round(v, 3) for k, v in curva['rec'].to_dict().items()})
    print("\nUsa estos objetos (curva, semanal, cyc, cortes) para regenerar")
    print("las tablas del reporte Necesidades_Ejote.xlsx con el mismo formato.")

    cyc.to_csv('ciclos_actualizado.csv', index=False)
    pd.DataFrame(semanal).to_csv('rendimiento_semanal_actualizado.csv')
    print("\nGuardado: ciclos_actualizado.csv, rendimiento_semanal_actualizado.csv")


if __name__ == '__main__':
    main()
