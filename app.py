# ============================================================
# CURVAS (ACOTADAS A LA DURACIÓN RECOMENDADA Y EXCLUYENDO ATÍPICOS)
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


def recommended_curve(d, cycles, vegetable):
    weekly = cycle_curve_data(d, vegetable)
    if weekly.empty:
        return pd.DataFrame()

    # Obtenemos la duración recomendada para este vegetal para acotar la curva
    ds = duration_analysis(cycles, vegetable)
    max_weeks = ds["recommended"] if ds and "recommended" in ds else int(weekly["SemanaRelativa"].max())

    rows = []
    for week, g in weekly.groupby("SemanaRelativa"):
        # Opcional: si una semana relativa supera la duración recomendada del ciclo, la ignoramos o no la graficamos como parte de la curva principal
        if week > max_weeks:
            continue
            
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
    if out.empty:
        return out

    out["Recomendado"] = out["P50"].clip(lower=0)

    total = out["Recomendado"].sum()
    if total > 0:
        out["Recomendado"] /= total

    out["P25"] = out["P25"].clip(lower=0)
    out["P75"] = out["P75"].clip(lower=0)

    return out
