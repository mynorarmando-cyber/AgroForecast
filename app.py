import pandas as pd
import numpy as np

# 1. Carga y procesamiento del archivo de datos históricos
df_raw = pd.read_excel('Analisis final.xlsx', sheet_name=0, header=None)
left = df_raw.iloc[2:, 1:14].copy()
left.columns = [
    "Finca", "Lote", "Area", "Ciclo", "Codigo", "Vegetal",
    "Referencia", "CantidadV", "DuracionSC", "Kilos",
    "Semana", "Anio", "Mes"
]

num_cols = ["Area", "Ciclo", "Codigo", "CantidadV", "DuracionSC", "Kilos", "Semana", "Anio"]
for c in num_cols:
    left[c] = pd.to_numeric(left[c], errors="coerce")

left["Finca"] = left["Finca"].astype(str).str.strip()
left["Lote"] = left["Lote"].astype(str).str.strip()
left["Referencia"] = left["Referencia"].fillna("").astype(str).str.strip()
left["Referencia"] = left["Referencia"].replace({"Extrafino": "Fino", "EXTRAFINO": "Fino", "extrafino": "Fino"})
left = left[left["Finca"].ne("nan") & left["Lote"].ne("nan") & left["Ciclo"].notna() & left["Kilos"].notna()].copy()

left["CantidadV"] = left["CantidadV"].fillna(1).clip(lower=1)
left["AreaEfectiva"] = left["Area"] / left["CantidadV"]

# Cálculo de fechas relativas por ciclo
dates = []
for y, w in zip(left["Anio"], left["Semana"]):
    try:
        dt = pd.to_datetime(f"{int(y)}-W{int(w):02d}-1", format="%G-W%V-%u")
    except:
        try:
            dt = pd.to_datetime(f"{int(y)}-01-01") + pd.Timedelta(weeks=int(w)-1)
        except:
            dt = pd.NaT
    dates.append(dt)
left["FechaSemana"] = dates
left = left[left["FechaSemana"].notna()].copy()

keys = ["Finca", "Lote", "Ciclo", "Referencia"]
cycles = left.groupby(keys, dropna=False).agg(
    Area=("AreaEfectiva", "first"),
    TotalKilos=("Kilos", "sum"),
    PrimeraCosecha=("FechaSemana", "min"),
    UltimaCosecha=("FechaSemana", "max"),
    AnioCosecha=("Anio", "max")
).reset_index()

cycles["DuracionReal"] = ((cycles["UltimaCosecha"] - cycles["PrimeraCosecha"]).dt.days / 7) + 1
cycles["DuracionReal"] = pd.to_numeric(cycles["DuracionReal"].round(), errors="coerce")
cycles["Rendimiento"] = cycles["TotalKilos"] / cycles["Area"].replace(0, np.nan)

left = left.merge(cycles, on=keys, how="left", suffixes=("", "_cycle"))
relative = ((left["FechaSemana"] - left["PrimeraCosecha"]).dt.days / 7) + 1
left["SemanaRelativa"] = pd.to_numeric(relative.round(), errors="coerce").astype(int)
left = left[left["SemanaRelativa"] > 0].copy()

def calc_curve(subset):
    if subset.empty:
        return pd.DataFrame()
    w = subset.groupby(["Finca", "Lote", "Ciclo", "Referencia", "SemanaRelativa"], as_index=False).agg(Kilos=("Kilos", "sum"))
    tot = w.groupby(["Finca", "Lote", "Ciclo", "Referencia"])["Kilos"].transform("sum")
    w["Pct"] = w["Kilos"] / tot.replace(0, np.nan)
    c = w.groupby("SemanaRelativa")["Pct"].median().reset_index()
    c.columns = ["Semana", "Porcentaje"]
    t = c["Porcentaje"].sum()
    if t > 0:
        c["Porcentaje"] /= t
    return c

# 2. Generación del archivo Excel final con ambas pestañas
output_file = "Reporte_Necesidades_Plan_Completo.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # --- PESTAÑA: necesidades ---
    rows_nec = []
    for veg in ["Fino", "Broccoli"]:
        sub_v = left[left["Referencia"] == veg]
        sub_c = cycles[cycles["Referencia"] == veg]
        
        c_old = calc_curve(sub_v[sub_v["AnioCosecha"] < 2025])
        c_rec = calc_curve(sub_v[sub_v["AnioCosecha"] >= 2025])
        c_all = calc_curve(sub_v)
        
        rend_old = sub_c[sub_c["AnioCosecha"] < 2025]["Rendimiento"].median()
        rend_rec = sub_c[sub_c["AnioCosecha"] >= 2025]["Rendimiento"].median()
        rend_rec_val = float(rend_rec) * 1.05 if not np.isnan(rend_rec) else 11000.0
        
        rows_nec.append({
            "Vegetal / Sección": f"--- {veg} ---", "Semana": np.nan, 
            "Curva <25": np.nan, "Curva 2025-2026": np.nan, "Curva Recomendada": np.nan,
            "Rendimiento <25": rend_old, "Rendimiento 2025-2026": rend_rec, "Rendimiento Recomendado": rend_rec_val
        })
        
        max_s = max(len(c_old), len(c_rec), len(c_all), 1)
        for i in range(max_s):
            w_num = i + 1
            p1 = c_old.loc[c_old["Semana"] == w_num, "Porcentaje"].values
            p2 = c_rec.loc[c_rec["Semana"] == w_num, "Porcentaje"].values
            p3 = c_all.loc[c_all["Semana"] == w_num, "Porcentaje"].values
            
            rows_nec.append({
                "Vegetal / Sección": veg,
                "Semana": w_num,
                "Curva <25": p1[0] if len(p1) > 0 else 0,
                "Curva 2025-2026": p2[0] if len(p2) > 0 else 0,
                "Curva Recomendada": p3[0] if len(p3) > 0 else 0,
                "Rendimiento <25": np.nan, "Rendimiento 2025-2026": np.nan, "Rendimiento Recomendado": np.nan
            })
            
    df_nec = pd.DataFrame(rows_nec)
    df_nec.to_excel(writer, sheet_name="necesidades", index=False)
    
    # --- PESTAÑA: Plan ---
    plan_df = pd.DataFrame({
        "Semana": range(1, 53),
        "Curva_Produccion": np.tile([0.35, 0.42, 0.23] + [0]*49, 1)[:52],
        "Rendimiento_Plan": 10900
    })
    plan_df["Area_Lote"] = 0.6
    plan_df["Kilos_Proyectados"] = plan_df["Area_Lote"] * plan_df["Curva_Produccion"] * plan_df["Rendimiento_Plan"]
    plan_df.to_excel(writer, sheet_name="Plan", index=False)

print(f"Archivo generado exitosamente como '{output_file}'")
