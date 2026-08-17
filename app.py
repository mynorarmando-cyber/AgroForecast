
import io
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AgroForecast | Pronóstico agrícola", layout="wide")

# ------------------------------------------------------------
# Utilidades
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

# Umbrales para excluir unicamente atipicos extremos y muy raros.
EXTREME_OUTLIER_MAX_SHARE = 0.05
EXTREME_OUTLIER_IQR = 3.0
EXTREME_OUTLIER_Z = 3.0

def normalize_reference(s):
    s = s.astype(str).str.strip()
    # El usuario indicó que Fino y Extrafino son el mismo vegetal.
    return s.replace({
        "Extrafino": "Fino",
        "EXTRAFINO": "Fino",
        "extrafino": "Fino",
    })

def read_excel(file):
    raw = pd.read_excel(file, sheet_name=0, header=None)

    # Tabla 6: encabezados en la primera fila útil.
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

    # Tabla 10, si está disponible.
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
    # Excel puede contener vacíos, NA, inf o textos en columnas numéricas.
    # Convertimos de forma segura y usamos Int64 (admite NA).
    for col in ["Area", "Ciclo", "Codigo", "CantidadV", "DuracionSC",
                "Kilos", "Semana", "Año", "Mes"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
            d[col] = d[col].replace([np.inf, -np.inf], np.nan)

    d["CantidadV"] = d["CantidadV"].fillna(1).clip(lower=1)
    d["AreaEfectiva"] = d["Area"] / d["CantidadV"]

    # Los registros sin semana o año válido no pueden entrar al cálculo ISO.
    d = d[
        d["Semana"].notna()
        & d["Año"].notna()
        & np.isfinite(d["Semana"])
        & np.isfinite(d["Año"])
    ].copy()

    d["Semana"] = d["Semana"].round().astype("Int64")
    d["Año"] = d["Año"].round().astype("Int64")

    # Fecha de lunes de la semana ISO. Esto permite cruzar años sin errores.
    d["SemanaInicio"] = pd.to_datetime(
        d["Año"].astype(str) + "-W" + d["Semana"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce"
    )

    # Un ciclo + vegetal + finca + lote.
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
    ).round()
    cycles["DuracionReal"] = cycles["DuracionReal"].where(
        cycles["DuracionReal"].notna()
        & np.isfinite(cycles["DuracionReal"])
    ).astype("Int64")
    cycles["Rendimiento"] = cycles["TotalKilos"] / cycles["Area"].replace(0, np.nan)

    d = d.merge(
        cycles[keys + ["PrimeraCosecha","DuracionReal","Rendimiento"]],
        on=keys, how="left", suffixes=("","_c")
    )
    d["SemanaRelativa"] = (
        ((d["SemanaInicio"] - d["PrimeraCosecha"]).dt.days / 7) + 1
    ).round()
    d["SemanaRelativa"] = d["SemanaRelativa"].where(
        d["SemanaRelativa"].notna()
        & np.isfinite(d["SemanaRelativa"])
    ).astype("Int64")

    # Peso de recencia: 50% últimos 2 años, 30% 3-4 años, 20% anteriores.
    valid_years = pd.to_numeric(d["Año"], errors="coerce").dropna()
    if valid_years.empty:
        cycles["PesoRecencia"] = 1.0
    else:
        max_year = int(valid_years.max())
        cycles["PesoRecencia"] = cycles["AñoCosecha"].apply(
            lambda y: recency_weight(y, max_year)
        )

    return d, cycles

def duration_analysis(cycles, vegetable):
    """Analiza duracion real y separa atipicos extremos del modelo."""
    x = cycles[(cycles["Referencia"] == vegetable) & cycles["DuracionReal"].notna() & (cycles["DuracionReal"] > 0)].copy()
    if x.empty:
        return None
    n = len(x)
    q1, q3 = x["DuracionReal"].quantile(.25), x["DuracionReal"].quantile(.75)
    iqr = q3 - q1
    low, high = q1 - EXTREME_OUTLIER_IQR * iqr, q3 + EXTREME_OUTLIER_IQR * iqr
    counts = x["DuracionReal"].value_counts()
    share = x["DuracionReal"].map(counts) / n
    x["AtipicoExtremo"] = (((x["DuracionReal"] < low) | (x["DuracionReal"] > high)) & (share < EXTREME_OUTLIER_MAX_SHARE))
    median = x["DuracionReal"].median()
    mad = np.median(np.abs(x["DuracionReal"] - median))
    if mad > 0:
        rz = (0.6745 * (x["DuracionReal"] - median) / mad).abs()
        x["AtipicoExtremo"] |= ((rz > EXTREME_OUTLIER_Z) & (share < EXTREME_OUTLIER_MAX_SHARE))
    x.loc[share >= 0.10, "AtipicoExtremo"] = False
    model_x = x[~x["AtipicoExtremo"]].copy()
    if len(model_x) < max(3, int(n * .50)):
        model_x = x.copy(); x["AtipicoExtremo"] = False
    valid_years = pd.to_numeric(model_x["AñoCosecha"], errors="coerce").dropna()
    max_year = int(valid_years.max()) if not valid_years.empty else 0
    recent = model_x[model_x["AñoCosecha"] >= max_year - 2].copy()
    m_all, m_recent = model_x["DuracionReal"].mode(), recent["DuracionReal"].mode()
    hist = int(m_all.iloc[0]) if len(m_all) else int(round(model_x["DuracionReal"].median()))
    rec = int(m_recent.iloc[0]) if len(m_recent) else hist
    recommended, reason = hist, "Comportamiento historico dominante"
    if len(recent) >= 3 and rec != hist and (recent["DuracionReal"] == rec).mean() >= .35:
        recommended, reason = rec, "Cambio reciente consistente"
    if len(recent) >= 5:
        rm = int(round(recent["DuracionReal"].median()))
        if rm != hist and (recent["DuracionReal"] == rm).mean() >= .40:
            recommended, reason = rm, "Transicion reciente confirmada"
    confidence = "Alta" if len(model_x) >= 20 and len(recent) >= 5 else ("Media" if len(model_x) >= 8 and len(recent) >= 3 else "Baja")
    return {"n": n, "n_modelo": len(model_x), "recommended": int(recommended), "historical_mode": hist, "recent_mode": rec, "historical_median": float(model_x["DuracionReal"].median()), "recent_median": float(recent["DuracionReal"].median()) if len(recent) else np.nan, "atypical": int(x["AtipicoExtremo"].sum()), "excluded": x[x["AtipicoExtremo"]].copy(), "recent_n": len(recent), "reason": reason, "confidence": confidence}

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

def build_curve(d, vegetable):
    x = d[d["Referencia"] == vegetable].copy()
    if x.empty:
        return pd.DataFrame(columns=["SemanaRelativa","Porcentaje","n"])

    # Excluir de la curva solo ciclos clasificados como atipicos extremos.
    base = x[["Finca","Lote","Ciclo","Referencia","DuracionReal","Año"]].drop_duplicates(["Finca","Lote","Ciclo","Referencia"]).copy()
    base["AñoCosecha"] = base["Año"]
    ds = duration_analysis(base, vegetable)
    if ds is not None and not ds["excluded"].empty:
        keys = ds["excluded"][["Finca","Lote","Ciclo","Referencia"]].drop_duplicates()
        x = x.merge(keys.assign(_ExcluirExtremo=True), on=["Finca","Lote","Ciclo","Referencia"], how="left")
        x = x[x["_ExcluirExtremo"].fillna(False).eq(False)].drop(columns=["_ExcluirExtremo"])

    # Por ciclo: porcentaje de su propia producción total.
    cyc = (
        x.groupby(["Finca","Lote","Ciclo","Referencia","SemanaRelativa"], as_index=False)
         .agg(Kilos=("Kilos","sum"))
    )
    total = cyc.groupby(["Finca","Lote","Ciclo","Referencia"])["Kilos"].transform("sum")
    cyc["Pct"] = cyc["Kilos"] / total.replace(0, np.nan)

    max_year = int(x["Año"].max())
    cyc["Año"] = x.groupby(["Finca","Lote","Ciclo","Referencia"])["Año"].transform("max").values
    cyc["Peso"] = cyc["Año"].apply(lambda y: recency_weight(y, max_year))

    out = []
    for sw, g in cyc.groupby("SemanaRelativa"):
        out.append({
            "SemanaRelativa": int(sw),
            "Porcentaje": weighted_quantile(g["Pct"].values, .50, g["Peso"].values),
            "P25": weighted_quantile(g["Pct"].values, .25, g["Peso"].values),
            "P75": weighted_quantile(g["Pct"].values, .75, g["Peso"].values),
            "n": int(g["Finca"].count())
        })
    out = pd.DataFrame(out).sort_values("SemanaRelativa")
    if not out.empty:
        out["Porcentaje"] = out["Porcentaje"].clip(lower=0)
        out["Porcentaje"] = out["Porcentaje"] / out["Porcentaje"].sum()
    return out

def seasonality(d, vegetable):
    x = d[d["Referencia"] == vegetable].copy()
    if x.empty:
        return pd.DataFrame(columns=["Semana","FactorEstacional"])
    # Producción por ciclo normalizada por ha y comparada contra el nivel medio del vegetal.
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
    # Suavizado para no sobre-reaccionar a pocas observaciones.
    out["FactorEstacional"] = out["FactorEstacional"].rolling(5, center=True, min_periods=1).mean()
    return out

def trend_factor(cycles, vegetable, target_year):
    x = cycles[(cycles["Referencia"] == vegetable) & cycles["Rendimiento"].notna()].copy()
    if len(x) < 4:
        return 1.0
    annual = x.groupby("AñoCosecha").agg(Rendimiento=("Rendimiento","median")).reset_index()
    if len(annual) < 3:
        return 1.0
    slope = np.polyfit(annual["AñoCosecha"], annual["Rendimiento"], 1)[0]
    ref = float(annual["Rendimiento"].median())
    if ref <= 0:
        return 1.0
    f = 1 + (slope / ref) * (target_year - annual["AñoCosecha"].max())
    return float(np.clip(f, 0.80, 1.20))

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

    target_year = first_harvest.year
    tf = trend_factor(cycles, vegetable, target_year)

    curve = build_curve(d, vegetable)
    seas = seasonality(d, vegetable)
    if curve.empty:
        return None, stats, None

    rows = []
    for _, r in curve.iterrows():
        rel = int(r["SemanaRelativa"])
        harvest_date = first_harvest + timedelta(weeks=rel-1)
        iso = harvest_date.isocalendar()
        sw = int(iso.week)
        sf = float(seas.loc[seas["Semana"] == sw, "FactorEstacional"].iloc[0]) if (seas["Semana"] == sw).any() else 1.0
        rows.append({
            "Semana relativa": rel,
            "Fecha": harvest_date,
            "Semana año": sw,
            "Curva base %": r["Porcentaje"],
            "Factor estacional": sf,
        })
    out = pd.DataFrame(rows)
    # La estacionalidad redistribuye el total, pero no cambia arbitrariamente el total.
    out["Peso ajustado"] = out["Curva base %"] * out["Factor estacional"]
    out["Peso ajustado"] = out["Peso ajustado"] / out["Peso ajustado"].sum()
    out["Rendimiento proyectado kg/ha"] = base * tf
    out["Kilos proyectados"] = area * out["Rendimiento proyectado kg/ha"] * out["Peso ajustado"]

    dur_info = duration_analysis(cycles, vegetable)
    duration = int(dur_info["recommended"]) if dur_info is not None else int(curve["SemanaRelativa"].max())
    return out, stats, {"trend_factor": tf, "duration": duration, "atypical_excluded": int(dur_info["atypical"]) if dur_info else 0, "duration_reason": dur_info["reason"] if dur_info else ""}

# ------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------
st.title("🌱 AgroForecast — Rendimiento y Pronóstico Agrícola")
st.caption("Modelo estadístico agrícola basado en ciclos reales, curvas de cosecha, estacionalidad y tendencia.")

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("Carga tu archivo Excel", type=["xlsx","xls"])
    st.markdown("**Estructura esperada:** Tabla 6 en las primeras 13 columnas y Tabla 10 en las siguientes 11.")
    st.divider()
    st.header("Reglas del modelo")
    st.write("• Fino + Extrafino → Fino")
    st.write("• Área efectiva = Área / Cantidad V")
    st.write("• Curva por semana relativa")
    st.write("• Ponderación de recencia: 50% / 30% / 20%")

source = uploaded if uploaded is not None else None

if source is None:
    st.info("Carga el archivo histórico para iniciar el análisis.")
    st.stop()

try:
    t6, t10 = read_excel(source)
    data, cycles = prepare_model(t6)
except Exception as e:
    st.error(f"No se pudo interpretar el Excel: {e}")
    st.stop()

vegetables = sorted(data["Referencia"].dropna().unique().tolist())
st.success(f"Datos cargados: {len(data):,} registros semanales y {len(cycles):,} ciclos.")

tabs = st.tabs(["📊 Dashboard", "🌾 Vegetal", "📈 Curvas", "🔮 Pronóstico", "🧪 Calidad de datos"])

with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Ciclos", f"{len(cycles):,}")
    c2.metric("Vegetales", f"{len(vegetables)}")
    c3.metric("Años", f"{int(data.Año.min())}–{int(data.Año.max())}")
    c4.metric("Kilos históricos", f"{data.Kilos.sum():,.0f}")

    ranking = (
        cycles.groupby("Referencia")
        .agg(Rendimiento=("Rendimiento","median"), Ciclos=("Rendimiento","count"))
        .reset_index().sort_values("Rendimiento", ascending=False)
    )
    st.subheader("Rendimiento mediano por vegetal")
    st.dataframe(ranking.style.format({"Rendimiento":"{:,.0f}"}), use_container_width=True, hide_index=True)

    fig = px.bar(ranking, x="Referencia", y="Rendimiento", hover_data=["Ciclos"],
                 labels={"Rendimiento":"kg/ha", "Referencia":"Vegetal"})
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    veg = st.selectbox("Vegetal", vegetables)
    stats = cycle_stats(cycles, veg)
    curve = build_curve(data, veg)
    seas = seasonality(data, veg)

    a,b,c,d1,e = st.columns(5)
    a.metric("Ciclos", stats["n"])
    b.metric("P25", f'{stats["p25"]:,.0f} kg/ha')
    c.metric("Mediana", f'{stats["median"]:,.0f} kg/ha')
    d1.metric("P75", f'{stats["p75"]:,.0f} kg/ha')
    e.metric("Máximo", f'{stats["max"]:,.0f} kg/ha')

    dur = cycles[cycles["Referencia"]==veg]["DuracionReal"]
    st.write(f"**Duración real:** mediana {dur.median():.0f} semanas | promedio {dur.mean():.1f} semanas | reciente (últimos 2 años) {dur[cycles.loc[cycles['Referencia']==veg,'AñoCosecha'] >= cycles['AñoCosecha'].max()-2].median():.0f} semanas.")

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de rendimiento")
        fig = px.histogram(cycles[cycles["Referencia"]==veg], x="Rendimiento", nbins=30, labels={"Rendimiento":"kg/ha"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Tendencia anual")
        annual = cycles[cycles["Referencia"]==veg].groupby("AñoCosecha")["Rendimiento"].median().reset_index()
        fig = px.line(annual, x="AñoCosecha", y="Rendimiento", markers=True, labels={"Rendimiento":"kg/ha","AñoCosecha":"Año"})
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    veg = st.selectbox("Vegetal para curva", vegetables, key="curveveg")
    curve = build_curve(data, veg)
    st.subheader(f"Curva de producción — {veg}")
    if curve.empty:
        st.warning("No hay suficientes datos.")
    else:
        fig = px.line(curve, x="SemanaRelativa", y="Porcentaje", markers=True,
                      labels={"SemanaRelativa":"Semana relativa de cosecha","Porcentaje":"% del total"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        show = curve.copy()
        show["Porcentaje"] = show["Porcentaje"].map(lambda x: f"{x:.1%}")
        st.dataframe(show.rename(columns={"SemanaRelativa":"Semana","Porcentaje":"Curva recomendada","P25":"P25","P75":"P75"}), use_container_width=True, hide_index=True)

        st.subheader("Estacionalidad por semana del año")
        seas = seasonality(data, veg)
        fig2 = px.line(seas, x="Semana", y="FactorEstacional", markers=True,
                       labels={"FactorEstacional":"Índice estacional"})
        fig2.add_hline(y=1, line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    st.subheader("Motor de pronóstico")
    col1,col2,col3 = st.columns(3)
    vegf = col1.selectbox("Vegetal", vegetables, key="forecastveg")
    area = col2.number_input("Área (ha)", min_value=0.01, value=1.0, step=0.1)
    first_harvest = col3.date_input("Fecha estimada de primera cosecha", value=date.today())

    st.caption("Importante: tu histórico no contiene fecha de siembra. Por eso la primera versión pronostica desde la **fecha de primera cosecha**. Cuando agreguemos Fecha Siembra al histórico, el sistema podrá aprender automáticamente los días siembra→primera cosecha por vegetal.")

    scenario = st.radio("Escenario", ["Conservador","Probable","Optimista"], horizontal=True)
    result, stats, meta = forecast(cycles, data, vegf, area, first_harvest, scenario)

    if result is not None:
        total = result["Kilos proyectados"].sum()
        rendimiento = result["Rendimiento proyectado kg/ha"].iloc[0]
        dur = result["Semana relativa"].max()

        a,b,c = st.columns(3)
        a.metric("Producción total", f"{total:,.0f} kg")
        b.metric("Rendimiento", f"{rendimiento:,.0f} kg/ha")
        c.metric("Duración esperada", f"{dur} semanas")

        st.dataframe(
            result[["Semana relativa","Fecha","Semana año","Peso ajustado","Kilos proyectados"]]
            .rename(columns={"Peso ajustado":"% proyectado"}),
            use_container_width=True, hide_index=True
        )
        fig = px.bar(result, x="Fecha", y="Kilos proyectados",
                     labels={"Kilos proyectados":"kg proyectados","Fecha":"Semana de cosecha"})
        st.plotly_chart(fig, use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar pronóstico CSV", csv, "pronostico_agroforecast.csv", "text/csv")
    else:
        st.warning("No hay suficiente histórico para ese vegetal.")

with tabs[4]:
    st.subheader("Calidad y trazabilidad de datos")
    st.write("El modelo usa la Tabla 6 como fuente principal y recalcula el rendimiento por ciclo.")
    q1 = data["Kilos"].isna().sum()
    q2 = data["Area"].isna().sum()
    q3 = data["SemanaInicio"].isna().sum()
    st.metric("Registros con semana/fecha no interpretable", int(q3))
    st.metric("Registros con kilos faltantes", int(q1))
    st.metric("Registros con área faltante", int(q2))

    st.info(
        "Siguiente mejora recomendada: agregar Fecha de Siembra, Fecha de Primera Cosecha y variedad real. "
        "Con esas tres variables el modelo podrá aprender el tiempo siembra→cosecha y separar mejor efectos de lote, época y variedad."
    )
