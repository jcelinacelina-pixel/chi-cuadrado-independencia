import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

st.set_page_config(
    page_title="Chi-cuadrado: Turno vs Condición",
    layout="wide"
)

# ==========================================
# 1. ENUNCIADO Y CONSIGNAS
# ==========================================
st.title("2.2 Ejercicio 2. Turno de trabajo y condición del producto: chi-cuadrado de independencia")

st.info("""
**Problema:**  
Una fábrica quiere investigar si la condición del producto al salir de una línea (defectuoso o no defectuoso) está asociada al turno de producción. Se seleccionan al azar 240 unidades, distribuidas por igual entre dos turnos, y se registran los siguientes resultados. Use $\\alpha = 0{,}05$.

**Consignas:**
1. Enuncie $H_0$ (independencia entre turno y condición) y $H_1$ (asociación entre las variables).
2. Calcule las frecuencias esperadas usando $E_{ij} = (\\text{total de fila} \\times \\text{total de columna}) / N$.
3. Calcule el estadístico $\\chi^2 = \\sum (O - E)^2 / E$, determine los grados de libertad y obtenga el valor-$p$.
4. Verifique que las frecuencias esperadas permitan aplicar la aproximación chi-cuadrado. Concluya si existe evidencia de asociación al 5%.
""")

# ==========================================
# 2. DATOS OBSERVADOS
# ==========================================
st.header("Datos Observados")

filas = ["Mañana", "Tarde"]
columnas = ["Defectuoso", "No defectuoso"]

# Matriz observada (2x2)
O = np.array([
    [12, 108],
    [24, 96]
])

df_obs = pd.DataFrame(O, index=filas, columns=columnas)
df_obs["Total"] = df_obs.sum(axis=1)
fila_total = pd.DataFrame([df_obs.sum(axis=0)], index=["Total"], columns=df_obs.columns)
df_obs_completa = pd.concat([df_obs, fila_total])

st.dataframe(df_obs_completa, use_container_width=True)

# Parámetros del test
alpha = 0.05
N = O.sum()
n_filas, n_columnas = O.shape
df_gl = (n_filas - 1) * (n_columnas - 1)

# Totales marginales
tot_filas = O.sum(axis=1)      # [120, 120]
tot_columnas = O.sum(axis=0)   # [36, 204]

# Frecuencias esperadas
E = np.outer(tot_filas, tot_columnas) / N

# ==========================================
# 3. RESOLUCIÓN DE CONSIGNAS
# ==========================================

st.markdown("---")
st.subheader("1. Planteo de Hipótesis")
st.markdown(r"""
* **$H_0$ (Hipótesis Nula):** La condición del producto es **independiente** del turno de producción (no existe asociación).
* **$H_1$ (Hipótesis Alternativa):** La condición del producto está **asociada** al turno de producción (no son independientes).
""")

st.markdown("---")
st.subheader("2. Frecuencias Esperadas ($E_{ij}$)")
st.latex(r"E_{ij} = \frac{\text{Total Fila } i \times \text{Total Columna } j}{N}")

col_calc, col_tabla = st.columns([1, 1])

with col_calc:
    st.write(f"- $E_{{11}} (\\text{{Mañana, Def}}) = (120 \\times 36) / 240 = {E[0,0]:.1f}$")
    st.write(f"- $E_{{12}} (\\text{{Mañana, No Def}}) = (120 \\times 204) / 240 = {E[0,1]:.1f}$")
    st.write(f"- $E_{{21}} (\\text{{Tarde, Def}}) = (120 \\times 36) / 240 = {E[1,0]:.1f}$")
    st.write(f"- $E_{{22}} (\\text{{Tarde, No Def}}) = (120 \\times 204) / 240 = {E[1,1]:.1f}$")

with col_tabla:
    df_esp = pd.DataFrame(E, index=filas, columns=columnas)
    st.dataframe(df_esp, use_container_width=True)

st.markdown("---")
st.subheader("3. Estadístico de Prueba, Grados de Libertad y Valor-p")

# Cálculo manual celda por celda para mostrar el desglose
terminos_chi = (O - E)**2 / E
chi2_stat = terminos_chi.sum()
p_value = 1 - stats.chi2.cdf(chi2_stat, df=df_gl)
chi2_critico = stats.chi2.ppf(1 - alpha, df=df_gl)

st.latex(r"\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}")

st.write(f"""
- **Desglose de sumandos:**
  - $(\\text{{Mañana, Def}}): (12 - 18)^2 / 18 = 36 / 18 = {terminos_chi[0,0]:.4f}$
  - $(\\text{{Mañana, No Def}}): (108 - 102)^2 / 102 = 36 / 102 = {terminos_chi[0,1]:.4f}$
  - $(\\text{{Tarde, Def}}): (24 - 18)^2 / 18 = 36 / 18 = {terminos_chi[1,0]:.4f}$
  - $(\\text{{Tarde, No Def}}): (96 - 102)^2 / 102 = 36 / 102 = {terminos_chi[1,1]:.4f}$
""")

col1, col2, col3 = st.columns(3)
col1.metric("Grados de Libertad (gl)", f"{df_gl}")
col2.metric("Estadístico Chi-cuadrado (χ²)", f"{chi2_stat:.4f}")
col3.metric("Valor-p", f"{p_value:.4f}")

st.write(f"- **Valor crítico:** $\\chi^2_{{0.05, 1}} = {chi2_critico:.4f}$")

st.markdown("---")
st.subheader("4. Verificación de Supuestos y Conclusión")

# Verificación de frecuencias esperadas
todas_mayores_5 = (E >= 5).all()
st.write("**Condición de aproximación:**")
if todas_mayores_5:
    st.success(f" Se cumple el supuesto: todas las frecuencias esperadas son $\\ge 5$ (la mínima esperada es {E.min():.1f}). Es válido aplicar la prueba Chi-cuadrado.")
else:
    st.warning("⚠️ No se cumple el supuesto de frecuencias esperadas $\\ge 5$.")

# Decisión estadística
st.write(f"**Comparación con nivel de significancia ($\\alpha = {alpha}$):**")
if p_value < alpha:
    st.error(f"""
    Como el valor-$p$ ({p_value:.4f}) es menor que $\\alpha$ ({alpha}) [o bien $\\chi^2 = {chi2_stat:.4f} > {chi2_critico:.4f}$], **se rechaza $H_0$**.
    
    **Conclusión:** Existe evidencia estadística suficiente al 5% de significancia para afirmar que la condición del producto **está asociada** al turno de trabajo.
    """)
else:
    st.info(f"""
    Como el valor-$p$ ({p_value:.4f}) es mayor o igual que $\\alpha$ ({alpha}), **no se rechaza $H_0$**.
    
    **Conclusión:** No existe evidencia suficiente para rechazar la hipótesis de independencia.
    """)
