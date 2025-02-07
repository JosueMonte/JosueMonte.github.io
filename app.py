import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Título de la aplicación
st.title("Simulador de Estrategias de Inversión")

# Selección de estrategia de inversión
estrategia = st.selectbox("Elige una estrategia de inversión",
                          ["Estrategia de momentum con el mejor activo según Ivy Portfolio",
                           "Estrategia de momentum con los mejores dos activos según Ivy Portfolio",
                           "Estrategia de momentum con los mejores tres activos según Ivy Portfolio",
                           "Estrategia de rebalanceo con los 5 activos según Ivy Portfolio"])

# Selección de activos
activos = st.multiselect("Selecciona los activos", [
                         "SPY", "EFA", "IYR", "GSG", "AGG"])

# Parámetros para el backtest
parametros = st.slider(
    "Selecciona el rango de fechas para el backtest:",
    datetime(2012, 1, 1), datetime(2024, 12, 31))

# Botón para ejecutar el backtest
if st.button("Ejecutar Backtest"):
    # Ejemplo de generación de datos de backtest
    resultados = pd.DataFrame({
        "Fecha": pd.date_range(start="2007-01-01", periods=100, freq="M"),
        # Ajusta según tus necesidades
        "Valor": np.random.rand(100) * parametros.year
    })
    st.line_chart(resultados.set_index("Fecha"))

# Mostrar resultados del backtest
st.write("Resultados del backtest para la estrategia seleccionada y los activos elegidos.")
