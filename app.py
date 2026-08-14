import io
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AgroForecast | Pronóstico agrícola", layout="wide")

# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================
RECENT_YEARS = 2
HISTORICAL_CUTOFF_YEARS = 4
WEIGHT_RECENT = 0.50
WEIGHT_MID = 0.30
WEIGHT_OLD = 0.20


# ============================================================
# UTILIDADES
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


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def weighted_quantile(values, q, weights=None):
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    v = v[mask]
    if len(v) == 0:
        return np.nan

    if weights is None:
        return float(np.quantile(v, q))

    w = np.asarray(weights, dtype=float)[mask]
    valid = np.isfinite(w) & (w > 0)
    v, w = v[valid], w[valid]
    if len(v) == 0:
        return np.nan

    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= w.sum()
    return float(np.interp(q, cumulative, v))


def recency_weight(year, max_year):
    if pd.isna(year):
        return 0.20
    age = int(max_year - year)
    if age <= RECENT_YEARS:
        return WEIGHT_RECENT
    if age <= HISTORICAL_CUTOFF_YEARS:
        return WEIGHT_MID
    return WEIGHT_OLD


def make_week_date(year, week):
    if pd.isna(year) or pd.isna(week):
        return pd.NaT
    try:
        return pd.to_datetime(
            f"{int(year)}-W{int(week):02d}-1",
            format="%G-W%V-%u",
            errors="coerce"
        )
    except Exception:
        return pd.NaT


# ============================================================
# LECTURA DEL EXCEL
# ============================================================
def read_excel(file):
    raw = pd.read_excel(file, sheet_name=0, header=None)

    # Tabla 6: B:N
    left = raw.iloc[:, 1:14].copy()
    left.columns = [
        "Finca", "Lote", "Area", "Ciclo", "Codigo", "Vegetal",
        "Referencia", "CantidadV", "DuracionSC", "Kilos",
        "Semana", "Año", "Mes"
    ]

    left["Finca"] = left["Finca"].astype(str).str.strip()
    left["Lote"] = left["Lote"].astype(str).str.strip()
    left["Referencia"] = normalize_reference(left["Referencia"])

    left = safe_numeric(
        left,
        ["Area", "Ciclo", "Codigo", "CantidadV", "DuracionSC",
         "Kilos", "Semana", "Año", "Mes"]
    )

    left = left[
        left["Finca"].ne("nan")
        & left["Lote"].ne("nan")
        & left["Ciclo"].notna()
        & left["Kilos"].notna()
        & left["Area"].notna()
        & left["Semana"].notna()
        & left["Año"].notna()
    ].copy()

    # Tabla 10: P:Z, si existe.
    right = raw.iloc[:, 15:26].copy()
    right.columns = [
        "Finca", "Lote", "Area", "Ciclo", "Vegetal", "Referencia",
        "CantidadV", "DuracionSC", "Total", "Rendimiento",
        "RendimientoReal"
    ]
    right["Finca"] = right["Finca"].astype(str).str.strip()
    right["Lote"] = right["Lote"].astype(str).str.strip()
    right["Referencia"] = normalize_reference(right["Referencia"])

    right = safe_numeric(
        right,
        ["Area", "Ciclo", "CantidadV", "DuracionSC", "Total",
         "Rendimiento", "RendimientoReal"]
    )

    right = right[
        right["Finca"].ne("nan")
        & right["Lote"].ne("nan")
        & right["Ciclo"].notna()
    ].copy()

    return left, right


# ============================================================
# PREPARACIÓN DE CICLOS
# ============================================================
def prepare_model(t6):
    d = t6.copy()

    d["CantidadV"] = pd.to_numeric(d["CantidadV"], errors="coerce").fillna(1)
    d["CantidadV"] = d["CantidadV"].clip(lower=1)
    d["Area"] = pd.to_numeric(d["Area"], errors="coerce")
    d["Kilos"] = pd.to_numeric(d["Kilos"], errors="coerce")
    d["Semana"] = pd.to_numeric(d["Semana"], errors="coerce")
    d["Año"] = pd.to_numeric(d["Año"], errors="coerce")

    d["AreaEfectiva"] = d["Area"] / d["CantidadV"]
    d["SemanaInicio"] = [
        make_week_date(y, w) for y, w in zip(d["Año"], d["Semana"])
    ]

    d = d[d["SemanaInicio"].notna()].copy()

    keys = ["Finca", "Lote", "Ciclo", "Referencia"]

    cycles = (
        d.groupby(keys, dropna=False)
        .agg(
            Area=("AreaEfectiva", "first"),
            TotalKilos=("Kilos", "sum"),
            PrimeraCosecha=("SemanaInicio", "min"),
            UltimaCosecha=("SemanaInicio", "max"),
            AñoCosecha=("Año", "max"),
            SemanasObservadas=("SemanaInicio", "nunique"),
        )
        .reset_index()
    )

    duration = (
        (cycles["UltimaCosecha"] - cycles["PrimeraCosecha"]).dt.days / 7
    ) + 1

    cycles["DuracionReal"] = pd.to_numeric(
        duration.round(), errors="coerce"
    )
    cycles["DuracionReal"] = cycles["DuracionReal"].where(
        np.isfinite(cycles["DuracionReal"]), np.nan
    )

    cycles["Rendimiento"] = (
        cycles["TotalKilos"] /
        cycles["Area"].replace(0, np.nan)
    )

    # Detección robusta de atípicos por IQR en la duración real por vegetal
    cycles["Atipico"] = False
    for veg, group in cycles.groupby("Referencia"):
        dur = group["DuracionReal"].dropna()
        if len(dur) >= 4:
            q1 = dur.quantile(0.25)
            q3 = dur.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            idx = group[(group["DuracionReal"] < low) | (group["DuracionReal"] > high)].index
            cycles.loc[idx, "Atipico"] = True

    max_year = int(d["Año"].max())
    cycles["PesoRecencia"] = cycles["AñoCosecha"].apply(
        lambda y: recency_weight(y, max_year)
    )

    d = d.merge(
        cycles[
            keys + [
                "PrimeraCosecha",
                "UltimaCosecha",
                "DuracionReal",
                "Rendimiento",
                "AñoCosecha",
                "PesoRecencia",
                "Atipico"
            ]
        ],
        on=keys,
        how="left"
    )

    relative = (
        (d["SemanaInicio"] - d["PrimeraCosecha"]).dt.days / 7
    ) + 1

    d["SemanaRelativa"] = pd.to_numeric(
        relative.round(), errors="coerce"
    )
    d = d[d["SemanaRelativa"].notna()].copy()
    d["SemanaRelativa"] = d["SemanaRelativa"].astype(int)

    return d, cycles


# ============================================================
# ESTADÍSTICA DE RENDIMIENTO (EXCLUYENDO ATÍPICOS)
# ============================================================
def period_label(year, max_year):
    age = max_year - year
    if age <= RECENT_YEARS:
        return "2025-2026 / reciente"
    if age <= HISTORICAL_CUTOFF_YEARS:
        return "3-4 años"
    return "<25 / histórico"


def yield_stats(cycles, vegetable):
    # Excluimos atípicos para limpiar las estadísticas
    x = cycles[
        (cycles["Referencia"] == vegetable)
        & (~cycles["Atipico"])
        & cycles["Rendimiento"].notna()
        & (cycles["Rendimiento"] > 0)
    ].copy()

    if x.empty:
        return None

    max_year = int(cycles["AñoCosecha"].max())
    x["Periodo"] = x["AñoCosecha"].apply(
        lambda y: period_label(y, max_year)
    )

    recent = x[x["AñoCosecha"] >= max_year - RECENT_YEARS]
    old = x[x["AñoCosecha"] < max_year - RECENT_YEARS]

    values = x["Rendimiento"].to_numpy()
    weights = x["PesoRecencia"].to_numpy()

    p25 = weighted_quantile(values, .25, weights)
    p50 = weighted_quantile(values, .50, weights)
    p75 = weighted_quantile(values, .75, weights)
    wmean = weighted_mean(values, weights)

    recent_median = float(recent["Rendimiento"].median()) if not recent.empty else np.nan
    historical_median = float(old["Rendimiento"].median()) if not old.empty else np.nan

    annual = (
        x.groupby("AñoCosecha", as_index=False)["Rendimiento"]
        .median()
        .sort_values("AñoCosecha")
    )

    trend_pct = 0.0
    if len(annual) >= 3 and annual["Rendimiento"].median() > 0:
        slope = np.polyfit(
            annual["AñoCosecha"],
            annual["Rendimiento"],
            1
        )[0]
        trend_pct = float(
            np.clip(
                slope / annual["Rendimiento"].median(),
                -0.20,
                0.20
            )
        )

    recommended = wmean * (1 + trend_pct)
    recommended = float(
        np.clip(
            recommended,
            p25 if np.isfinite(p25) else recommended,
            p75 if np.isfinite(p75) else recommended
        )
    )

    return {
        "n": int(len(x)),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "mean_weighted": float(wmean),
        "recent_median": recent_median,
        "historical_median": historical_median,
        "recommended": recommended,
        "trend_pct": trend_pct,
        "annual": annual,
        "cycles": x,
    }


# ============================================================
# CURVAS (EXCLUYENDO ATÍPICOS)
# ============================================================
def cycle_curve_data(d, vegetable):
    # Excluimos registros pertenecientes a ciclos atípicos
    x = d[(d["Referencia"] == vegetable) & (~d["Atipico"])].copy()
    if x.empty:
        return pd.DataFrame()

    x = x[
        x["SemanaRelativa"].notna()
        & x["Kilos"].notna()
        & (x["Kilos"] >= 0)
    ].copy()

    cycle_keys = ["Finca", "Lote", "Ciclo", "Referencia"]

    weekly = (
        x.groupby(
            cycle_keys + ["SemanaRelativa"],
            as_index=False
        )
        .agg(Kilos=("Kilos", "sum"))
    )

    totals = (
        weekly.groupby(cycle_keys)["Kilos"]
        .transform("sum")
    )
    weekly["Pct"] = weekly["Kilos"] / totals.replace(0, np.nan)

    cycle_year = (
        x.groupby(cycle_keys)["AñoCosecha"]
        .first()
        .reset_index()
    )

    weekly = weekly.merge(
        cycle_year,
        on=cycle_keys,
        how="left"
    )

    max_year = int(x["AñoCosecha"].max())
    weekly["Peso"] = weekly["AñoCosecha"].apply(
        lambda y: recency_weight(y, max_year)
    )

    return weekly


def recommended_curve(d, vegetable):
    weekly = cycle_curve_data(d, vegetable)
    if weekly.empty:
        return pd.DataFrame()

    rows = []
    for week, g in weekly.groupby("SemanaRelativa"):
        rows.append({
            "Semana": int(week),
            "Historico": weighted_quantile(
                g["Pct"], .50, np.ones(len(g))
            ),
            "Reciente": weighted_quantile(
                g.loc[g["AñoCosecha"] >= g["AñoCosecha"].max() - RECENT_YEARS, "Pct"],
                .50
            ) if (g["AñoCosecha"] >= g["AñoCosecha"].max() - RECENT_YEARS).any() else np.nan,
            "P25": weighted_quantile(g["Pct"], .25, g["Peso"]),
            "P50": weighted_quantile(g["Pct"], .50, g["Peso"]),
            "P75": weighted_quantile(g["Pct"], .75, g["Peso"]),
            "N": int(len(g)),
        })

    out = pd.DataFrame(rows).sort_values("Semana")
    out["Recomendado"] = out["P50"].clip(lower=0)

    total = out["Recomendado"].sum()
    if total > 0:
        out["Recomendado"] /= total

    out["P25"] = out["P25"].clip(lower=0)
    out["P75"] = out["P75"].clip(lower=0)

    return out


# ============================================================
# DURACIÓN RECOMENDADA Y ANÁLISIS DE ATÍPICOS
# ============================================================
def duration_analysis(cycles, vegetable):
    x = cycles[
        (cycles["Referencia"] == vegetable)
        & cycles["DuracionReal"].notna()
        & (cycles["DuracionReal"] > 0)
    ].copy()

    if x.empty:
        return None

    max_year = int(cycles["AñoCosecha"].max())
    
    # Subconjunto sin atípicos para calcular la duración esperada limpia
    clean_x = x[~x["Atipico"]].copy()
    recent_clean = clean_x[clean_x["AñoCosecha"] >= max_year - RECENT_YEARS]

    historical_mode = int(clean_x["DuracionReal"].mode().iloc[0]) if not clean_x.empty else int(round(x["DuracionReal"].median()))
    recent_mode = int(recent_clean["DuracionReal"].mode().iloc[0]) if not recent_clean.empty else historical_mode

    recommended = historical_mode
    recent_n = len(recent_clean)
    
    if recent_n >= 3 and recent_mode != historical_mode:
        counts = recent_clean["DuracionReal"].value_counts()
        share = counts.get(recent_mode, 0) / recent_n
        if share >= 0.35:
            recommended = recent_mode
            reason = "Cambio reciente consistente (sin atípicos)"
        else:
            reason = "Comportamiento histórico dominante"
    else:
        reason = "Comportamiento histórico dominante"

    if recent_n >= 5:
        recent_median = int(round(recent_clean["DuracionReal"].median()))
        if recent_median > historical_mode:
            share_long = (recent_clean["DuracionReal"] >= recent_median).mean()
            if share_long >= 0.50:
                recommended = recent_median
                reason = "Transición reciente confirmada"

    confidence = "Alta" if len(clean_x) >= 20 else ("Media" if len(clean_x) >= 8 else "Baja")
    if recent_n < 3:
        confidence = "Baja"

    return {
        "n": len(clean_x),
        "historical_mode": historical_mode,
        "recent_mode": recent_mode,
        "historical_median": float(clean_x["DuracionReal"].median()),
        "recent_median": float(recent_clean["DuracionReal"].median()) if not recent_clean.empty else np.nan,
        "recommended": int(recommended),
        "atypical": int(x["Atipico"].sum()),
        "recent_n": recent_n,
        "reason": reason,
        "confidence": confidence,
        "detail": x,
    }


# ============================================================
# ESTACIONALIDAD (EXCLUYENDO ATÍPICOS)
# ============================================================
def seasonality(d, vegetable):
    # Filtramos también registros atípicos
    x = d[(d["Referencia"] == vegetable) & (~d["Atipico"])].copy()
    if x.empty:
        return pd.DataFrame(columns=["Semana", "FactorEstacional"])

    weekly = (
        x.groupby(
            ["Año", "Semana", "Finca", "Lote", "Ciclo", "Referencia"],
            as_index=False
        )
        .agg(
            Kilos=("Kilos", "sum"),
            Area=("AreaEfectiva", "first")
        )
    )

    weekly["KgHa"] = (
        weekly["Kilos"] /
        weekly["Area"].replace(0, np.nan)
    )

    max_year = int(x["Año"].max())
    weekly["Peso"] = weekly["Año"].apply(
        lambda y: recency_weight(y, max_year)
    )

    valid = weekly["KgHa"].notna() & np.isfinite(weekly["KgHa"])
    base = weighted_mean(
        weekly.loc[valid, "KgHa"],
        weekly.loc[valid, "Peso"]
    )

    if not np.isfinite(base) or base <= 0:
        return pd.DataFrame(columns=["Semana", "FactorEstacional"])

    rows = []
    for week, g in weekly.groupby("Semana"):
        valid_g = g["KgHa"].notna() & np.isfinite(g["KgHa"])
        if not valid_g.any():
            continue
        m = weighted_mean(
            g.loc[valid_g, "KgHa"],
            g.loc[valid_g, "Peso"]
        )
        rows.append({
            "Semana": int(week),
            "FactorEstacional": m / base
        })

    out = pd.DataFrame(rows).sort_values("Semana")
    if out.empty:
        return out

    out["FactorEstacional"] = (
        out["FactorEstacional"]
        .rolling(5, center=True, min_periods=1)
        .mean()
        .clip(0.80, 1.20)
    )
    return out


# ============================================================
# MOTOR DE NECESIDADES
# ============================================================
def build_necesidades(d, cycles):
    rows = []

    for veg in sorted(cycles["Referencia"].dropna().unique()):
        ys = yield_stats(cycles, veg)
        ds = duration_analysis(cycles, veg)
        curve = recommended_curve(d, veg)

        if ys is None:
            continue

        rows.append({
            "Vegetal": veg,
            "Ciclos": ys["n"],
            "Rend. <25 / histórico": ys["historical_median"],
            "Rend. reciente": ys["recent_median"],
            "Rend. P25": ys["p25"],
            "Rend. P50": ys["p50"],
            "Rend. P75": ys["p75"],
            "Rend. recomendado": ys["recommended"],
            "Tendencia %": ys["trend_pct"],
            "Duración histórica": ds["historical_mode"] if ds else np.nan,
            "Duración reciente": ds["recent_mode"] if ds else np.nan,
            "Duración recomendada": ds["recommended"] if ds else np.nan,
            "Confianza duración": ds["confidence"] if ds else "Baja",
            "Ciclos atípicos": ds["atypical"] if ds else np.nan,
            "Motivo duración": ds["reason"] if ds else "",
            "Semanas curva": int(curve["Semana"].max()) if not curve.empty else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# PRONÓSTICO
# ============================================================
def forecast(cycles, d, vegetable, area, first_harvest, scenario):
    ys = yield_stats(cycles, vegetable)
    ds = duration_analysis(cycles, vegetable)
    curve = recommended_curve(d, vegetable)

    if ys is None or ds is None or curve.empty:
        return None, None

    if scenario == "Conservador":
        base = ys["p25"]
    elif scenario == "Optimista":
        base = ys["p75"]
    else:
        base = ys["recommended"]

    curve = curve[curve["Semana"] <= ds["recommended"]].copy()
    if curve.empty:
        return None, None

    curve["Recomendado"] = curve["Recomendado"].clip(lower=0)
    curve["Recomendado"] /= curve["Recomendado"].sum()

    seas = seasonality(d, vegetable)

    rows = []
    for _, r in curve.iterrows():
        rel = int(r["Semana"])
        harvest_date = first_harvest + timedelta(weeks=rel - 1)
        iso = harvest_date.isocalendar()
        week_year = int(iso.week)

        factor = 1.0
        if not seas.empty:
            match = seas[seas["Semana"] == week_year]
            if not match.empty:
                factor = float(match["FactorEstacional"].iloc[0])

        rows.append({
            "Semana relativa": rel,
            "Fecha": harvest_date,
            "Semana año": week_year,
            "Curva recomendada": r["Recomendado"],
            "Factor estacional": factor,
        })

    out = pd.DataFrame(rows)
    out["Peso ajustado"] = (
        out["Curva recomendada"] *
        out["Factor estacional"]
    )
    out["Peso ajustado"] /= out["Peso ajustado"].sum()

    out["Rendimiento plan kg/ha"] = base
    out["Kilos proyectados"] = (
        area *
        out["Rendimiento plan kg/ha"] *
        out["Peso ajustado"]
    )

    meta = {
        "rendimiento": base,
        "duracion": int(ds["recommended"]),
        "confianza": ds["confidence"],
        "motivo_duracion": ds["reason"],
    }

    return out, meta


# ============================================================
# INTERFAZ
# ============================================================
st.title("🌱 AgroForecast — Rendimiento y Pronóstico Agrícola")
st.caption(
    "Motor estadístico con exclusión automática de ciclos atípicos "
    "para rendimiento, duración, curvas y estacionalidad."
)

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader(
        "Carga tu archivo histórico",
        type=["xlsx", "xls"]
    )
    st.markdown(
        "**Tabla 6:** primeras 13 columnas útiles. "
        "**Tabla 10:** siguientes 11 columnas, si existe."
    )

    st.divider()
    st.header("Reglas")
    st.write("• Fino + Extrafino → Fino")
    st.write("• Área efectiva = Área / Cantidad V")
    st.write("• Curva por semana relativa")
    st.write("• Exclusión automática de atípicos (> 1.5 IQR)")

if uploaded is None:
    st.info("Carga el archivo histórico para iniciar el análisis.")
    st.stop()

try:
    t6, t10 = read_excel(uploaded)
    data, cycles = prepare_model(t6)
except Exception as e:
    st.error(f"No se pudo interpretar el Excel: {e}")
    st.stop()

if data.empty or cycles.empty:
    st.error("El Excel no contiene ciclos de cosecha interpretables.")
    st.stop()

vegetables = sorted(
    data["Referencia"].dropna().astype(str).unique().tolist()
)

min_year = int(data["Año"].min())
max_year = int(data["Año"].max())

st.success(
    f"Datos cargados: {len(data):,} registros semanales y "
    f"{len(cycles):,} ciclos."
)

tabs = st.tabs([
    "📊 Dashboard",
    "🌾 Vegetal",
    "📈 Curvas",
    "🧠 Necesidades",
    "🔮 Pronóstico",
    "🗓️ Plan",
    "🧪 Calidad"
])


# ============================================================
# DASHBOARD
# ============================================================
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ciclos", f"{len(cycles):,}")
    c2.metric("Vegetales", f"{len(vegetables)}")
    c3.metric("Años", f"{min_year}–{max_year}")
    c4.metric("Kilos históricos", f"{data.Kilos.sum():,.0f}")

    # Ranking usando ciclos limpios sin atípicos
    clean_cycles_all = cycles[~cycles["Atipico"]]
    ranking = (
        clean_cycles_all.groupby("Referencia")
        .agg(
            Rendimiento=("Rendimiento", "median"),
            Ciclos=("Rendimiento", "count")
        )
        .reset_index()
        .sort_values("Rendimiento", ascending=False)
    )

    st.subheader("Rendimiento mediano por vegetal (Excluyendo atípicos)")
    st.dataframe(
        ranking.style.format({"Rendimiento": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        ranking,
        x="Referencia",
        y="Rendimiento",
        hover_data=["Ciclos"],
        labels={
            "Rendimiento": "kg/ha",
            "Referencia": "Vegetal"
        }
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# VEGETAL
# ============================================================
with tabs[1]:
    veg = st.selectbox("Vegetal", vegetables)
    ys = yield_stats(cycles, veg)
    ds = duration_analysis(cycles, veg)
    curve = recommended_curve(data, veg)

    a, b, c, d1, e = st.columns(5)
    a.metric("Ciclos limpios", ys["n"] if ys else 0)
    b.metric("P25", f'{ys["p25"]:,.0f} kg/ha' if ys else "—")
    c.metric("P50", f'{ys["p50"]:,.0f} kg/ha' if ys else "—")
    d1.metric("P75", f'{ys["p75"]:,.0f} kg/ha' if ys else "—")
    e.metric("Recomendado", f'{ys["recommended"]:,.0f} kg/ha' if ys else "—")

    if ys:
        st.subheader("Rendimiento")
        st.write(
            f"Histórico: **{ys['historical_median']:,.0f} kg/ha** | "
            f"Reciente: **{ys['recent_median']:,.0f} kg/ha** | "
            f"Tendencia: **{ys['trend_pct']:.1%}**"
        )

    st.subheader("Duración real de cosecha")
    if ds:
        a, b, c, d1 = st.columns(4)
        a.metric("Histórica", f"{ds['historical_mode']} semanas")
        b.metric("Reciente", f"{ds['recent_mode']} semanas")
        c.metric("Recomendada", f"{ds['recommended']} semanas")
        d1.metric("Atípicos excluidos", ds["atypical"])
        st.info(
            f"{ds['reason']}. Ciclos limpios analizados: {ds['n']}. "
            f"Confianza: {ds['confidence']}."
        )

    if ys:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tendencia anual")
            fig = px.line(
                ys["annual"],
                x="AñoCosecha",
                y="Rendimiento",
                markers=True,
                labels={"Rendimiento": "kg/ha", "AñoCosecha": "Año"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Distribución de rendimiento (Limpio)")
            fig = px.histogram(
                ys["cycles"],
                x="Rendimiento",
                nbins=30,
                labels={"Rendimiento": "kg/ha"}
            )
            st.plotly_chart(fig, use_container_width=True)

    if ds:
        st.subheader("Listado de ciclos y detección de atípicos")
        detail = ds["detail"][
            [
                "Finca", "Lote", "Ciclo",
                "AñoCosecha", "DuracionReal",
                "Rendimiento", "Atipico"
            ]
        ].copy()
        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# CURVAS
# ============================================================
with tabs[2]:
    veg = st.selectbox("Vegetal para curva", vegetables, key="curveveg")
    curve = recommended_curve(data, veg)

    if curve.empty:
        st.warning("No hay datos suficientes.")
    else:
        chart = curve.melt(
            id_vars=["Semana"],
            value_vars=["P25", "P50", "P75", "Recomendado"],
            var_name="Serie",
            value_name="Porcentaje"
        )

        fig = px.line(
            chart,
            x="Semana",
            y="Porcentaje",
            color="Serie",
            markers=True,
            labels={
                "Semana": "Semana relativa",
                "Porcentaje": "% del total"
            }
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        display = curve.copy()
        for col in ["Historico", "Reciente", "P25", "P50", "P75", "Recomendado"]:
            display[col] = display[col].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Estacionalidad por semana del año (Excluyendo atípicos)")
        seas = seasonality(data, veg)
        if not seas.empty:
            fig2 = px.line(
                seas,
                x="Semana",
                y="FactorEstacional",
                markers=True,
                labels={"FactorEstacional": "Índice estacional"}
            )
            fig2.add_hline(y=1, line_dash="dash")
            st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# NECESIDADES
# ============================================================
with tabs[3]:
    st.subheader("🧠 Motor de recomendación — Necesidades")
    st.caption("Cálculos limpios excluyendo ciclos con duraciones atípicas extremas.")

    necesidades = build_necesidades(data, cycles)

    if necesidades.empty:
        st.warning("No fue posible generar recomendaciones.")
    else:
        display = necesidades.copy()

        for col in [
            "Rend. <25 / histórico",
            "Rend. reciente",
            "Rend. P25",
            "Rend. P50",
            "Rend. P75",
            "Rend. recomendado",
        ]:
            display[col] = display[col].map(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
            )

        display["Tendencia %"] = display["Tendencia %"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Descargar Necesidades CSV",
            necesidades.to_csv(index=False).encode("utf-8-sig"),
            "necesidades_agroforecast.csv",
            "text/csv"
        )


# ============================================================
# PRONÓSTICO
# ============================================================
with tabs[4]:
    st.subheader("🔮 Motor de pronóstico")

    col1, col2, col3 = st.columns(3)
    vegf = col1.selectbox("Vegetal", vegetables, key="forecastveg")
    area = col2.number_input("Área (ha)", min_value=0.01, value=1.0, step=0.1)
    first_harvest = col3.date_input("Fecha estimada de primera cosecha", value=date.today())

    scenario = st.radio(
        "Escenario",
        ["Conservador", "Probable", "Optimista"],
        horizontal=True
    )

    result, meta = forecast(
        cycles,
        data,
        vegf,
        area,
        first_harvest,
        scenario
    )

    if result is not None:
        total = result["Kilos proyectados"].sum()

        a, b, c, d1 = st.columns(4)
        a.metric("Producción total", f"{total:,.0f} kg")
        b.metric("Rendimiento plan", f"{meta['rendimiento']:,.0f} kg/ha")
        c.metric("Duración limpia", f"{meta['duracion']} semanas")
        d1.metric("Confianza", meta["confianza"])

        show = result[
            [
                "Semana relativa",
                "Fecha",
                "Semana año",
                "Curva recomendada",
                "Factor estacional",
                "Peso ajustado",
                "Kilos proyectados"
            ]
        ].copy()

        for col in ["Curva recomendada", "Peso ajustado"]:
            show[col] = show[col].map(lambda x: f"{x:.1%}")

        st.dataframe(show, use_container_width=True, hide_index=True)

        fig = px.bar(
            result,
            x="Fecha",
            y="Kilos proyectados",
            labels={"Kilos proyectados": "kg proyectados", "Fecha": "Semana de cosecha"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Descargar pronóstico CSV",
            result.to_csv(index=False).encode("utf-8-sig"),
            "pronostico_agroforecast.csv",
            "text/csv"
        )
    else:
        st.warning("No hay suficiente histórico limpio para generar un pronóstico.")


# ============================================================
# PLAN
# ============================================================
with tabs[5]:
    st.subheader("🗓️ Plan — rendimiento editable")

    vegp = st.selectbox("Vegetal", vegetables, key="planveg")
    ys = yield_stats(cycles, vegp)
    ds = duration_analysis(cycles, vegp)
    curve = recommended_curve(data, vegp)

    if ys is None or ds is None or curve.empty:
        st.warning("No hay suficiente información para este vegetal.")
    else:
        default_yield = float(round(ys["recommended"], 0))

        plan_yield = st.number_input(
            "Rendimiento plan kg/ha",
            min_value=0.0,
            value=default_yield,
            step=100.0
        )

        plan_curve = curve[curve["Semana"] <= ds["recommended"]].copy()
        plan_curve["Curva plan"] = plan_curve["Recomendado"].clip(lower=0)
        plan_curve["Curva plan"] /= plan_curve["Curva plan"].sum()

        plan_curve["Rendimiento plan kg/ha"] = plan_yield
        plan_curve["Producción kg/ha"] = (
            plan_curve["Rendimiento plan kg/ha"] *
            plan_curve["Curva plan"]
        )

        display = plan_curve[
            ["Semana", "Curva plan", "Rendimiento plan kg/ha", "Producción kg/ha"]
        ].copy()
        display["Curva plan"] = display["Curva plan"].map(lambda x: f"{x:.1%}")

        st.dataframe(display, use_container_width=True, hide_index=True)
        st.metric("Total rendimiento plan", f"{plan_curve['Producción kg/ha'].sum():,.0f} kg/ha")


# ============================================================
# CALIDAD
# ============================================================
with tabs[6]:
    st.subheader("🧪 Calidad y trazabilidad de atípicos")

    total_atipicos = int(cycles["Atipico"].sum())
    st.metric("Total de ciclos detectados como atípicos (excluidos de modelos)", total_atipicos)

    st.write("### Criterio aplicado:")
    st.write("• Se calculó el Rango Intercuartílico (IQR) de la duración real por vegetal.")
    st.write("• Los ciclos con duraciones extremas (como los de 18 semanas) superan el umbral $Q3 + 1.5 \\times IQR$ y se marcan como `Atipico = True`.")
    st.write("• Estos ciclos **ya no participan** en el cálculo de las medianas de rendimiento, las curvas porcentuales por semana relativa, ni en la estacionalidad.")
    st.write("• Sin embargo, se mantienen visibles en la pestaña de cada vegetal para fines de auditoría y trazabilidad.")
