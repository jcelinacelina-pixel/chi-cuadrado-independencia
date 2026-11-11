import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Chi-cuadrado Interactivo con Gráficos",
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
# 2. CONTROLES INTERACTIVOS (SIDEBAR)
# ==========================================
st.sidebar.header("Panel de Configuración Interactiva")
st.sidebar.markdown("Modifica los valores para recalcular el ejercicio y actualizar los gráficos:")

alpha = st.sidebar.slider(
    "Nivel de significancia (α)",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.01
)

st.sidebar.subheader("Turno Mañana")
m_def = st.sidebar.number_input("Mañana - Defectuosos", min_value=0, max_value=2000, value=12, step=1)
m_nodef = st.sidebar.number_input("Mañana - No defectuosos", min_value=0, max_value=2000, value=108, step=1)

st.sidebar.subheader("Turno Tarde")
t_def = st.sidebar.number_input("Tarde - Defectuosos", min_value=0, max_value=2000, value=24, step=1)
t_nodef = st.sidebar.number_input("Tarde - No defectuosos", min_value=0, max_value=2000, value=96, step=1)

# Matriz de frecuencias observadas
O = np.array([
    [m_def, m_nodef],
    [t_def, t_nodef]
], dtype=float)

filas = ["Mañana", "Tarde"]
columnas = ["Defectuoso", "No defectuoso"]

# ==========================================
# 3. TABLA DE DATOS OBSERVADOS
# ==========================================
st.header("Datos Observados (Interactivos)")

df_obs = pd.DataFrame(O.astype(int), index=filas, columns=columnas)
df_obs["Total"] = df_obs.sum(axis=1)
fila_total = pd.DataFrame([df_obs.sum(axis=0)], index=["Total"], columns=df_obs.columns)
df_obs_completa = pd.concat([df_obs, fila_total])

st.dataframe(df_obs_completa, use_container_width=True)

N = O.sum()
if N == 0:
    st.error("La suma total de observaciones no puede ser 0. Ingresa valores válidos en la barra lateral.")
    st.stop()

n_filas, n_columnas = O.shape
df_gl = (n_filas - 1) * (n_columnas - 1)
tot_filas = O.sum(axis=1)
tot_columnas = O.sum(axis=0)

if (tot_filas == 0).any() or (tot_columnas == 0).any():
    st.warning("Hay filas o columnas con total 0. Se requieren observaciones en ambas condiciones y turnos.")
    st.stop()

# Frecuencias esperadas
E = np.outer(tot_filas, tot_columnas) / N

# ==========================================
# 4. RESOLUCIÓN DE CONSIGNAS
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
    st.write(f"- $E_{{11}} (\\text{{Mañana, Def}}) = ({tot_filas[0]:.0f} \\times {tot_columnas[0]:.0f}) / {N:.0f} = {E[0,0]:.2f}$")
    st.write(f"- $E_{{12}} (\\text{{Mañana, No Def}}) = ({tot_filas[0]:.0f} \\times {tot_columnas[1]:.0f}) / {N:.0f} = {E[0,1]:.2f}$")
    st.write(f"- $E_{{21}} (\\text{{Tarde, Def}}) = ({tot_filas[1]:.0f} \\times {tot_columnas[0]:.0f}) / {N:.0f} = {E[1,0]:.2f}$")
    st.write(f"- $E_{{22}} (\\text{{Tarde, No Def}}) = ({tot_filas[1]:.0f} \\times {tot_columnas[1]:.0f}) / {N:.0f} = {E[1,1]:.2f}$")

with col_tabla:
    df_esp = pd.DataFrame(np.round(E, 2), index=filas, columns=columnas)
    st.dataframe(df_esp, use_container_width=True)

st.markdown("---")
st.subheader("3. Estadístico de Prueba, Grados de Libertad y Valor-p")

# Cálculos chi-cuadrado
terminos_chi = (O - E)**2 / E
chi2_stat = terminos_chi.sum()
p_value = 1 - stats.chi2.cdf(chi2_stat, df=df_gl)
chi2_critico = stats.chi2.ppf(1 - alpha, df=df_gl)

st.latex(r"\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}")

st.write(f"""
- **Desglose de sumandos:**
  - $(\\text{{Mañana, Def}}): ({O[0,0]:.0f} - {E[0,0]:.2f})^2 / {E[0,0]:.2f} = {terminos_chi[0,0]:.4f}$
  - $(\\text{{Mañana, No Def}}): ({O[0,1]:.0f} - {E[0,1]:.2f})^2 / {E[0,1]:.2f} = {terminos_chi[0,1]:.4f}$
  - $(\\text{{Tarde, Def}}): ({O[1,0]:.0f} - {E[1,0]:.2f})^2 / {E[1,0]:.2f} = {terminos_chi[1,0]:.4f}$
  - $(\\text{{Tarde, No Def}}): ({O[1,1]:.0f} - {E[1,1]:.2f})^2 / {E[1,1]:.2f} = {terminos_chi[1,1]:.4f}$
""")

col1, col2, col3 = st.columns(3)
col1.metric("Grados de Libertad (gl)", f"{df_gl}")
col2.metric("Estadístico Chi-cuadrado (χ²)", f"{chi2_stat:.4f}")
col3.metric("Valor-p", f"{p_value:.4f}")

st.write(f"- **Valor crítico con $\\alpha = {alpha}$:** $\\chi^2_{{{alpha}, {df_gl}}} = {chi2_critico:.4f}$")

st.markdown("---")
st.subheader("4. Verificación de Supuestos y Conclusión")

todas_mayores_5 = (E >= 5).all()
st.write("**Condición de aproximación:**")
if todas_mayores_5:
    st.success(f" Se cumple el supuesto: todas las frecuencias esperadas son $\\ge 5$ (la mínima esperada es {E.min():.2f}). Es válido aplicar la aproximación chi-cuadrado.")
else:
    st.warning(f"⚠️ Atención: hay frecuencias esperadas menores a 5 (la mínima esperada es {E.min():.2f}).")

st.write(f"**Comparación con nivel de significancia ($\\alpha = {alpha}$):**")
if p_value < alpha:
    st.error(f"""
    Como el valor-$p$ ({p_value:.4f}) es menor que $\\alpha$ ({alpha}) [o bien $\\chi^2 = {chi2_stat:.4f} > {chi2_critico:.4f}$], **se rechaza $H_0$**.
    
    **Conclusión:** Existe evidencia estadística suficiente al {(alpha*100):.0f}% de significancia para afirmar que la condición del producto **está asociada** al turno de trabajo.
    """)
else:
    st.info(f"""
    Como el valor-$p$ ({p_value:.4f}) es mayor o igual que $\\alpha$ ({alpha}) [o bien $\\chi^2 = {chi2_stat:.4f} \\le {chi2_critico:.4f}$], **no se rechaza $H_0$**.
    
    **Conclusión:** No existe evidencia estadística suficiente para rechazar la independencia entre el turno y la condición del producto.
    """)

# ==========================================
# 5. VISUALIZACIÓN GRÁFICA
# ==========================================
st.markdown("---")
st.subheader("5. Gráficos Comparativos y Zona de Decisión")

col_g1, col_g2 = st.columns(2)

# Gráfico 1: Comparación de Frecuencias Observadas vs Esperadas
with col_g1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    categorias = ["Mañana - Def", "Mañana - No Def", "Tarde - Def", "Tarde - No Def"]
    valores_O = [O[0,0], O[0,1], O[1,0], O[1,1]]
    valores_E = [E[0,0], E[0,1], E[1,0], E[1,1]]
    
    x = np.arange(len(categorias))
    ancho = 0.35
    
    ax1.bar(x - ancho/2, valores_O, ancho, label="Observado (O)", color="#1f77b4")
    ax1.bar(x + ancho/2, valores_E, ancho, label="Esperado (E)", color="#ff7f0e", alpha=0.85)
    
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Frecuencias Observadas vs. Esperadas")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categorias, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    fig1.tight_layout()
    st.pyplot(fig1)

# Gráfico 2: Distribución Chi-cuadrado y Región de Rechazo
with col_g2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x_max = max(10.0, chi2_stat + 3, chi2_critico + 3)
    x_vals = np.linspace(0.01, x_max, 500)
    y_vals = stats.chi2.pdf(x_vals, df=df_gl)
    
    ax2.plot(x_vals, y_vals, label=f"Densidad χ² (gl={df_gl})", color="black", lw=1.5)
    
    # Zona de rechazo
    x_rechazo = np.linspace(chi2_critico, x_max, 300)
    y_rechazo = stats.chi2.pdf(x_rechazo, df=df_gl)
    ax2.fill_between(x_rechazo, y_rechazo, color="red", alpha=0.3, label=f"Región Crítica (α={alpha})")
    
    # Línea del valor crítico
    ax2.axvline(chi2_critico, color="red", linestyle="--", label=f"χ² Crítico = {chi2_critico:.2f}")
    
    # Línea del estadístico calculado
    ax2.axvline(chi2_stat, color="blue", linestyle="-", lw=2, label=f"χ² Observado = {chi2_stat:.2f}")
    
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, min(1.0, max(y_vals[y_vals < 2]) * 1.2 if len(y_vals[y_vals < 2]) > 0 else 1.0))
    ax2.set_xlabel("Valor de χ²")
    ax2.set_ylabel("Densidad de Probabilidad")
    ax2.set_title("Distribución χ² y Decisión Estadística")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()
    st.pyplot(fig2)
