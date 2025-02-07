import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Función para calcular el momentum


def calculate_momentum(prices, lookback_period):
    momentum = prices / prices.rolling(window=lookback_period).mean() - 1
    return momentum

# Función para obtener los datos históricos de los activos


def get_historical_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Función para calcular el Ratio de Sharpe


def sharpe_ratio(returns, risk_free_rate=0.00):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

# Función para calcular el Ratio de Sortino


def sortino_ratio(returns, risk_free_rate=0.00):
    excess_returns = returns - risk_free_rate
    negative_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.std(negative_returns)
    return np.mean(excess_returns) / downside_deviation

# Función para calcular el Máximo Drawdown en términos porcentuales


def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

# Función para encontrar el peor año


def worst_year(data):
    data['year'] = data['date'].dt.year
    annual_returns = data.groupby('year')['return'].sum()
    worst_year = annual_returns.idxmin()
    worst_year_return = annual_returns.min()
    return worst_year, worst_year_return

# Función para encontrar el mejor año


def best_year(data):
    data['year'] = data['date'].dt.year
    annual_returns = data.groupby('year')['return'].sum()
    best_year = annual_returns.idxmax()
    best_year_return = annual_returns.max()
    return best_year, best_year_return


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
parametros = st.slider("Selecciona el rango de fechas para el backtest:", datetime(
    2012, 1, 1), datetime(2024, 12, 31))

# Botón para ejecutar el backtest
if st.button("Ejecutar Backtest"):
    start_date = parametros[0]
    end_date = parametros[1]

    # Fechas de inicio y fin para obtener los datos históricos
    total_period_years = end_date.year - start_date.year - 1
    start_dat = end_date.replace(year=end_date.year - total_period_years)
    start_dat -= timedelta(days=1)
    start_dat = start_dat.strftime('%Y-%m-%d')

    # Obtener los datos históricos de los activos
    historical_data = get_historical_data(activos, start_date, end_date)

    # Calcular los retornos mensuales
    monthly_prices = historical_data.resample('ME').ffill()
    monthly_returns = monthly_prices.pct_change()

    # Definir el periodo de lookback para el momentum
    lookback_period = 10

    # Calcular el momentum para cada activo
    momentum = calculate_momentum(monthly_prices, lookback_period)

    # Inicializar una lista para almacenar las decisiones de inversión mensuales y valores de portafolio
    investment_decisions = []
    portfolio_values_teorico = []
    portfolio_values_real = []
    rendimiento = []
    rendimiento_acumulado = []

    # Inicializar el valor del portafolio teórico (sin comisiones)
    initial_portfolio_value_teorico = 10000
    portfolio_value_teorico = initial_portfolio_value_teorico

    # Costos por comisión y spread compra-venta (faltan los impuestos) - Revisar estructura de costes de IBKR
    commision_buy = 0.0005
    commision_sell = 0.0005
    spread_buy = 0.00025
    spread_sell = 0.00025
    cost_operational = commision_buy + commision_sell + spread_buy + spread_sell

    # Inicializar el valor del portafolio real (con comisiones)
    initial_portfolio_value_real = initial_portfolio_value_teorico * \
        (1 - (commision_buy + spread_buy))
    portfolio_value_real = initial_portfolio_value_real

    # Simular la compra del activo con mejor momentum y rebalanceo mensual
    for date in momentum.index[lookback_period:]:
        # Seleccionar los datos hasta la fecha actual
        current_momentum = momentum.loc[:date].iloc[-1]

        # Encontrar el activo con el mejor momentum y su resultado de momentum
        best_asset = current_momentum.idxmax()
        best_momentum = current_momentum[best_asset]

        # Calcular el precio promedio del mejor activo en los últimos 10 meses
        last_10_months_prices = monthly_prices[best_asset].loc[:date].iloc[-lookback_period:]
        average_price = last_10_months_prices.mean()

        # Calcular el rendimiento del activo seleccionado en el último mes
        asset_return = monthly_returns.loc[date, best_asset]

        # Actualizar el valor del portafolio
        portfolio_value_teorico *= (1 + asset_return)

        # Registrar la decisión de inversión y el valor del portafolio teórico
        investment_decisions.append(
            (date, best_asset, asset_return, portfolio_value_teorico))
        portfolio_values_teorico.append(portfolio_value_teorico)
        rendimiento.append(1 + asset_return)

    # Calcular los rendimientos acumulados
    for i in range(len(rendimiento)):
        if i == 0:
            rendimiento_acumulado.append(rendimiento[i])
        else:
            rendimiento_acumulado.append(
                rendimiento_acumulado[i-1] * rendimiento[i])

    # Inicializar una variable para contar los cambios de activos
    change_count = 0
    # Iterar sobre las decisiones de inversión
    prev_asset = None
    index_change = []
    for decision in investment_decisions:
        current_asset = decision[1]
        if prev_asset is not None and current_asset != prev_asset:
            change_count += 1
            index_change.append(change_count)
        else:
            index_change.append(0)
        prev_asset = current_asset

    # Reemplazar números distintos de cero por 1
    operations = [1 if x != 0 else 0 for x in index_change]

    # Calcular el valor del portafolio real tomando en cuenta los costos operacionales
    for i in range(len(rendimiento)):
        # Aplicar el rendimiento del activo
        portfolio_value_real *= rendimiento[i]
        # Descontar el costo operacional si hay un cambio de activo
        if operations[i] == 1:
            portfolio_value_real *= (1 - cost_operational)
        portfolio_values_real.append(portfolio_value_real)

    # Convertir los valores de portafolio en una serie de pandas
    portfolio_values_series_teorico = pd.Series(
        data=portfolio_values_teorico, index=momentum.index[lookback_period:])
    portfolio_values_series_real = pd.Series(
        data=portfolio_values_real, index=momentum.index[lookback_period:])

    # Calcular el rendimiento acumulado del portafolio
    cumulative_returns_teorico = portfolio_values_series_teorico / \
        initial_portfolio_value_teorico - 1
    cumulative_returns_real = portfolio_values_series_real / \
        initial_portfolio_value_teorico - 1

    # Calcular el rendimiento anualizado del portafolio
    annualized_return_teorico = (
        portfolio_values_series_teorico.iloc[-1] / initial_portfolio_value_teorico) ** (1 / total_period_years) - 1
    annualized_return_real = (
        portfolio_values_series_real.iloc[-1] / initial_portfolio_value_teorico) ** (1 / total_period_years) - 1

    # Calcular la volatilidad del portafolio
    volatility = portfolio_values_series_teorico.pct_change(
    ).std() * np.sqrt(12)  # Anualizar la volatilidad

    # Calcular el Ratio de Sharpe
    sharpe = sharpe_ratio(portfolio_values_series_teorico.pct_change())

    # Calcular el Ratio de Sortino
    sortino = sortino_ratio(portfolio_values_series_teorico.pct_change())

    # Calcular el Máximo Drawdown
    max_dd = calculate_max_drawdown(
        portfolio_values_series_teorico.pct_change())

    # Calcular el peor año
    portfolio_returns = pd.DataFrame(
        {'date': portfolio_values_series_teorico.index, 'return': portfolio_values_series_teorico.pct_change()})
    worst_yr, worst_yr_return = worst_year(portfolio_returns)

    # Calcular el mejor año
    portfolio_returns = pd.DataFrame(
        {'date': portfolio_values_series_teorico.index, 'return': portfolio_values_series_teorico.pct_change()})
    best_yr, best_yr_return = best_year(portfolio_returns)

    # Mostrar las decisiones de inversión
    df_backtesting = pd.DataFrame(data=investment_decisions, columns=[
                                  'Fecha', 'Activo', 'Rendimiento', 'Portfolio_teórico'])
    df_backtesting['Fecha'] = df_backtesting['Fecha'].dt.strftime('%Y-%m-%d')
    df_backtesting['Operaciones'] = operations
    df_backtesting['Rendimiento_acumulado'] = rendimiento_acumulado
    df_backtesting['Portfolio_real'] = portfolio_values_real

    # Datos de la nueva fila
    portfolio_initial = pd.DataFrame({'Fecha': [start_dat], 'Activo': [0], 'Rendimiento': [0], 'Portfolio_teórico': [
                                     initial_portfolio_value_teorico], 'Portfolio_real': [initial_portfolio_value_real], 'Operaciones': [0]})

    # Rendimiento
    return_period_teorico = portfolio_values_series_teorico.iloc[-1] / \
        initial_portfolio_value_teorico - 1
    return_period_real = portfolio_values_series_real.iloc[-1] / \
        initial_portfolio_value_teorico - 1

    # Insertar la nueva fila en la posición 0:
    df_backtesting = pd.concat(
        [portfolio_initial, df_backtesting]).reset_index(drop=True)

    # Calcular la c7o   orrelación entre los activos
    correlation_matrix = monthly_returns.corr()


# Mostrar resultados del backtest
st.write("Resultados del backtest para la estrategia seleccionada y los activos elegidos.")

# Imprimir el total de cambios de activos
print(
    f"Total de operaciones concertadas durante el período del backtest: {change_count} - Promedio de operaciones por año: {(change_count/total_period_years):.2f}")

# Graficar el valor del portafolio en escala logarítmica
plt.figure(figsize=(12, 6))
plt.semilogy(portfolio_values_series_teorico, label='Valor del Portafolio')
plt.title(
    'Valor del Portafolio Basado en Estrategia de Momentum (Escala Logarítmica)')
plt.xlabel('Fecha')
plt.ylabel('Valor del Portafolio (log)')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el rendimiento acumulado del portafolio
plt.figure(figsize=(12, 6))
# Agregar 1 para evitar log(0)
plt.semilogy(cumulative_returns_teorico + 1,
             label='Rendimiento Acumulado del Portafolio')
plt.title('Rendimiento Acumulado del Portafolio Basado en Estrategia de Momentum')
plt.xlabel('Fecha')
plt.ylabel('Rendimiento Acumulado')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar las métricas del portafolio
print(f"Rendimiento final del portafolio teórico: {return_period_teorico:.2%}")
print(f"Rendimiento final del portafolio real: {return_period_real:.2%}")
print(
    f"Rendimiento anualizado del portafolio teórico: {annualized_return_teorico:.2%}")
print(
    f"Rendimiento anualizado del portafolio real: {annualized_return_real:.2%}")
print(
    f"Pérdida porcentual total por coste operacional: {(return_period_real / return_period_teorico - 1):.2%}")
print(
    f"Pérdida porcentual anualizada por coste operacional: {(annualized_return_real / annualized_return_teorico - 1):.2%}")
print(f"Volatilidad anualizada del portafolio: {volatility:.2%}")
print(f"Ratio de Sharpe: {sharpe:.2f}")
print(f"Ratio de Sortino: {sortino:.2f}")
print(f"Máximo Drawdown porcentual: {max_dd:.2%}")
print(f"Peor año: {worst_yr} con un rendimiento de {worst_yr_return:.2%}")
print(f"Mejor año: {best_yr} con un rendimiento de {best_yr_return:.2%}")

# Mostrar la correlación entre los activos
print("Correlación entre los activos:")
print(correlation_matrix)

# Mostrar la correlación entre los activos con un mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Matriz de Correlación entre los Activos')
plt.show()
