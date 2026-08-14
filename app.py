# ============================================================
# CURVAS (ACOTADAS A LA DURACIÓN RECOMENDADA Y EXCLUYENDO ATÍPICOS)
# ============================================================
def cycle_curve_data(d, vegetable):
    # Excluimos registros pertenecientes a ciclos atípicos
    if "Atipico" in d.columns:
        x = d[(d["Referencia"] == vegetable) & (~d["Atipico"])].copy()
    else:
        x = d[d["Referencia"] == vegetable].copy()
        
    if x.empty:
        return pd.DataFrame()

    # Validar columnas necesarias
    required_cols = ["SemanaRelativa", "Kilos", "Referencia", "AñoCosecha"]
    for col in required_cols:
        if col not in x.columns:
            return pd.DataFrame()

    x = x[
        x["SemanaRelativa"].notna()
        & x["Kilos"].notna()
        & (x["Kilos"] >= 0)
    ].copy()

    cycle_keys = [c for c in ["Finca", "Lote", "Ciclo", "Referencia"] if c in x.columns]

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

    max_year = int(x["AñoCosecha"].max()) if not x["AñoCosecha"].isna().all() else 2026
    
    # Función interna de peso por recencia si no existe global
    def _recency_weight(year):
        if pd.isna(year):
            return 1.0
        if year < 2025:
            return 1.0
        elif year == 2025:
            return 2.0
        else:
            return 3.0

    weekly["Peso"] = weekly["AñoCosecha"].apply(_recency_weight)

    return weekly


def recommended_curve(d, cycles, vegetable):
    weekly = cycle_curve_data(d, vegetable)
    if weekly.empty:
        return pd.DataFrame()

    # Obtener la duración recomendada directamente del modelo o ciclos si está disponible
    max_weeks = 10  # Valor por defecto seguro
    try:
        if cycles is not None and not cycles.empty and "Vegetal" in cycles.columns:
            match_row = cycles[cycles["Vegetal"] == vegetable]
            if not match_row.empty and "Duración recomendada" in match_row.columns:
                val = match_row["Duración recomendada"].iloc[0]
                if pd.notna(val):
                    max_weeks = int(val)
    except Exception:
        pass

    rows = []
    for week, g in weekly.groupby("SemanaRelativa"):
        # Cortar estrictamente a la duración recomendada (ej. 5 o 6 semanas en lugar de 10)
        if week > max_weeks:
            continue
            
        # Cálculo seguro de cuantiles ponderados
        vals = g["Pct"].values
        weights = g["Peso"].values if "Peso" in g.columns else np.ones(len(g))
        
        # Función auxiliar de cuantil ponderado interno
        def _w_quantile(values, q, w=None):
            values = np.array(values)
            if w is None:
                w = np.ones(len(values))
            w = np.array(w)
            mask = ~np.isnan(values) & ~np.isnan(w)
            values = values[mask]
            w = w[mask]
            if len(values) == 0:
                return np.nan
            sort_idx = np.argsort(values)
            values, w = values[sort_idx], w[sort_idx]
            cum_w = np.cumsum(w)
            if cum_w[-1] == 0:
                return np.nan
            return np.interp(q * cum_w[-1], cum_w, values)

        recent_mask = g["AñoCosecha"] >= (g["AñoCosecha"].max() - 2) if "AñoCosecha" in g.columns else np.zeros(len(g), dtype=bool)

        rows.append({
            "Semana": int(week),
            "Historico": _w_quantile(g["Pct"], 0.50, np.ones(len(g))),
            "Reciente": _w_quantile(g.loc[recent_mask, "Pct"], 0.50) if recent_mask.any() else np.nan,
            "P25": _w_quantile(g["Pct"], 0.25, weights),
            "P50": _w_quantile(g["Pct"], 0.50, weights),
            "P75": _w_quantile(g["Pct"], 0.75, weights),
            "N": int(len(g)),
        })

    out = pd.DataFrame(rows).sort_values("Semana")
    if out.empty:
        return out

    out["Recomendado"] = out["P50"].clip(lower=0)

    # Normalizar para que la suma dentro de las semanas recomendadas sea 100%
    total = out["Recomendado"].sum()
    if total > 0:
        out["Recomendado"] /= total

    out["P25"] = out["P25"].clip(lower=0)
    out["P75"] = out["P75"].clip(lower=0)

    return out
