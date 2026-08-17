"""
Lógica del modelo estadístico de rendimientos y curva de cosecha.
Compartida entre la app de Streamlit y el uso por línea de comandos.
"""
import io
import numpy as np
import pandas as pd
from scipy import stats
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter

FONT = 'Calibri'
H_FILL = PatternFill('solid', fgColor='1F4E28')
SUB_FILL = PatternFill('solid', fgColor='D9E8D6')
thin = Side(style='thin', color='BFBFBF')
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)


def circular_smooth(serie_por_semana, window=3):
    """Media móvil circular (semana 53 conecta con semana 1)."""
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


def cargar_datos(file_like, referencias_vegetal):
    """file_like: ruta o objeto tipo archivo (ej. de st.file_uploader)."""
    t6 = pd.read_excel(file_like, sheet_name='Hoja1', header=1, usecols="B:N", nrows=14084)
    veg = t6[t6['Referencia'].isin(referencias_vegetal)].copy()
    if veg.empty:
        raise ValueError(
            f"No se encontraron filas con Referencia en {referencias_vegetal}. "
            f"Referencias disponibles: {sorted(t6['Referencia'].dropna().unique())}"
        )
    return veg


def construir_ciclos(veg, anio_corte_periodo):
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
    cyc['periodo'] = np.where(cyc['AñoInicio'] < anio_corte_periodo, '<25', '25-26')

    cortes = grp.merge(cyc[['Finca', 'Lote', 'Ciclo', 'periodo']], on=['Finca', 'Lote', 'Ciclo'])
    return cyc, cortes


def curva_de_cosecha(cortes, maxcorte, peso_reciente, peso_historico):
    curva = {}
    for periodo in ['<25', '25-26']:
        sub = cortes[(cortes.periodo == periodo) & (cortes.corte <= maxcorte)]
        agg = sub.groupby('corte')['pct'].mean().reindex(range(1, maxcorte + 1)).fillna(0)
        total = agg.sum()
        curva[periodo] = agg / total if total > 0 else agg
    rec = curva['<25'] * peso_historico + curva['25-26'] * peso_reciente
    total = rec.sum()
    curva['rec'] = rec / total if total > 0 else rec
    return curva


def rendimiento_semanal(cyc, anio_objetivo, peso_reciente, peso_historico, winsor_pctl=(0.02, 0.98)):
    lo, hi = cyc['Rendimiento'].quantile(list(winsor_pctl))
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
    seasonal = peso_historico * seas_old + peso_reciente * seas_new

    trend_target = intercept + slope * anio_objetivo
    semanal['recomendado'] = (trend_target + seasonal).clip(lower=0)

    trend_stats = {'slope': slope, 'intercept': intercept, 'p_value': p, 'r': r,
                   'trend_target': trend_target, 'n_ciclos': len(cyc)}
    return semanal, cyc, trend_stats


def ejecutar_modelo(file_like, nombre_vegetal='Ejote', referencias_vegetal=('Fino', 'Extrafino'),
                     maxcorte=3, anio_corte_periodo=2025, anio_objetivo=2027,
                     peso_reciente=0.65, peso_historico=0.35):
    veg = cargar_datos(file_like, list(referencias_vegetal))
    cyc, cortes = construir_ciclos(veg, anio_corte_periodo)
    curva = curva_de_cosecha(cortes, maxcorte, peso_reciente, peso_historico)
    semanal, cyc_full, trend_stats = rendimiento_semanal(
        cyc, anio_objetivo, peso_reciente, peso_historico)
    return {
        'nombre_vegetal': nombre_vegetal, 'curva': curva, 'semanal': semanal,
        'cyc': cyc_full, 'cortes': cortes, 'trend_stats': trend_stats,
        'maxcorte': maxcorte, 'anio_corte_periodo': anio_corte_periodo,
        'anio_objetivo': anio_objetivo, 'n_old': int((cyc.periodo == '<25').sum()),
        'n_new': int((cyc.periodo == '25-26').sum()),
    }


# ---------------------------------------------------------------------------
# Generación del reporte Excel (mismo layout que la pestaña "necesidades")
# ---------------------------------------------------------------------------
def _styled(cell, value, header=False, sub=False, bold=False, numfmt=None):
    cell.value = value
    cell.font = Font(bold=bold or header, color='FFFFFF' if header else '000000', name=FONT)
    if header:
        cell.fill = H_FILL
    elif sub:
        cell.fill = SUB_FILL
    cell.border = BORDER
    if numfmt:
        cell.number_format = numfmt


def generar_reporte_excel(resultado):
    curva = resultado['curva']
    semanal = resultado['semanal']
    maxcorte = resultado['maxcorte']
    nombre = resultado['nombre_vegetal']

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'necesidades'

    _styled(ws['B2'], 'Semana de cosecha plan', header=True)
    _styled(ws['C2'], f'Curva de producción {nombre} plan', header=True)
    for i in range(maxcorte):
        wk = i + 1
        _styled(ws[f'B{3+i}'], wk)
        _styled(ws[f'C{3+i}'], round(float(curva['rec'][wk]), 3), numfmt='0.0%')
    total_row = 3 + maxcorte
    _styled(ws[f'B{total_row}'], 'Total', bold=True)
    ws[f'C{total_row}'] = f'=SUM(C3:C{total_row-1})'
    ws[f'C{total_row}'].number_format = '0.0%'
    ws[f'C{total_row}'].font = Font(bold=True, name=FONT)

    _styled(ws['E2'], 'Semana', header=True)
    _styled(ws['F2'], 'Rendimiento por semana', header=True)
    for wk in range(1, 54):
        r = 2 + wk
        _styled(ws[f'E{r}'], wk)
        _styled(ws[f'F{r}'], round(float(semanal['recomendado'][wk])), numfmt='#,##0')

    _styled(ws['H2'], f'{nombre} Rendimiento', header=True)
    for j in range(maxcorte):
        _styled(ws[f'{get_column_letter(9+j)}2'], f'Corte {j+1}', header=True)
    rows = [('<25', '<25'), ('2025 y 2026', '25-26'), ('Curva recomendada', 'rec')]
    for i, (label, key) in enumerate(rows):
        r = 3 + i
        _styled(ws[f'H{r}'], label, sub=True)
        for j in range(maxcorte):
            wk = j + 1
            _styled(ws[f'{get_column_letter(9+j)}{r}'], round(float(curva[key][wk]), 3), numfmt='0.0%')

    _styled(ws['P2'], 'Semana', header=True)
    _styled(ws['Q2'], 'Rendimiento por semana <25', header=True)
    _styled(ws['R2'], 'Rendimiento por semana 2025 y 2026', header=True)
    _styled(ws['S2'], 'Rendimiento por semana recomendado', header=True)
    for wk in range(1, 54):
        r = 2 + wk
        _styled(ws[f'P{r}'], wk)
        _styled(ws[f'Q{r}'], round(float(semanal['<25'][wk])), numfmt='#,##0')
        _styled(ws[f'R{r}'], round(float(semanal['25-26'][wk])), numfmt='#,##0')
        _styled(ws[f'S{r}'], round(float(semanal['recomendado'][wk])), numfmt='#,##0')

    for col, w in [('B', 14), ('C', 16), ('E', 9), ('F', 14), ('H', 16),
                   ('P', 9), ('Q', 18), ('R', 20), ('S', 20)]:
        ws.column_dimensions[col].width = w
    for j in range(maxcorte):
        ws.column_dimensions[get_column_letter(9+j)].width = 9

    # Hoja de detalle de ciclos (auditoría)
    cyc = resultado['cyc']
    ws2 = wb.create_sheet('Ciclos_Detalle')
    cols = ['Finca', 'Lote', 'Ciclo', 'SemanaInicio', 'AñoInicio', 'Area', 'CantidadV',
            'TotalKilos', 'Rendimiento', 'periodo']
    for i, h in enumerate(cols):
        c = ws2.cell(row=1, column=i+1, value=h)
        c.font = Font(bold=True, color='FFFFFF', name=FONT)
        c.fill = H_FILL
    for i, row in cyc.reset_index(drop=True).iterrows():
        r = i + 2
        for j, col in enumerate(cols):
            v = row[col]
            if col in ('Ciclo', 'SemanaInicio', 'AñoInicio'):
                v = int(v)
            elif col == 'Rendimiento':
                v = round(float(v), 1)
            ws2.cell(row=r, column=j+1, value=v)
    last_row = len(cyc) + 1
    tab = Table(displayName="TablaCiclos", ref=f"A1:J{last_row}")
    tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws2.add_table(tab)
    ws2.freeze_panes = 'A2'
    for col in ['A', 'B', 'J']:
        ws2.column_dimensions[col].width = 12

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf
