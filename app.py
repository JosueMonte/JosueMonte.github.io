import streamlit as st
import pandas as pd
import numpy as np

# Título de la aplicación
st.title("Simulador de Estrategias de Inversión")

# Selección de estrategia de inversión
estrategia = st.selectbox("Elige una estrategia de inversión",
                          ["Estrategia de momentum con el mejor activo según Ivy Portfolio",
                           "Estrategia de rebalanceo con los 5 activos según Ivy Portfolio",
                           "Estrategia de momentum según Bold Asset Allocation"])

# Selección de activos
activos = st.multiselect("Selecciona los activos", [
                         "SPY", "EFA", "IYR", "GSG", "AGG"])

# Parámetros para el backtest
parametros = st.slider(
    "Selecciona el parámetro para el backtest", 0.0, 1.0, 0.5)

# Botón para ejecutar el backtest
if st.button("Ejecutar Backtest"):
    # Ejemplo de generación de datos de backtest
    resultados = pd.DataFrame({
        "Fecha": pd.date_range(start="2007-01-01", periods=100, freq="M"),
        "Valor": np.random.rand(100) * parametros
    })
    st.line_chart(resultados.set_index("Fecha"))

# Mostrar resultados del backtest
st.write("Resultados del backtest para la estrategia seleccionada y los activos elegidos.")
