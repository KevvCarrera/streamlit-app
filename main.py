import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import sqlalchemy
from dotenv import load_dotenv
import os

# Cargar las variables de entorno
load_dotenv()


# Obtener las credenciales desde las variables de entorno
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Función para ajustar el tamaño de datos reales y predicciones
def ajustar_tamano(y_true, y_pred):
    """
    Ajusta las longitudes de los datos reales (y_true) y predicciones (y_pred)
    para que coincidan antes de calcular métricas.
    """
    n = min(len(y_true), len(y_pred))  # Longitud mínima entre datos reales y predicciones
    return y_true[-n:], y_pred[:n]  # Recorta ambas listas a la longitud mínima


# Título e introducción
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <h1>Pronóstico de Enfermedades</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Esta aplicación utiliza modelos de predicción como ARIMA, Holt-Winters y Redes Neuronales para analizar y pronosticar casos de enfermedades en animales con base en los datos proporcionados."
)

# Agregar el botón para acceder al manual de ayuda
if st.button('Ayuda en línea'):	
    # Redirigir a una nueva pestaña con el enlace
    st.markdown(
        '<a href="https://alexair21.github.io/MicroclinAyuda/" target="_blank">Presiona aquí, para abir la ayuda en línea</a>',
        unsafe_allow_html=True
    )

    
# Configuración de la conexión a la base de datos
engine = sqlalchemy.create_engine(
    f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
)

# Consulta SQL para obtener los datos
query = """
SELECT 
    idEnfermedad,
    anio,
    semana,
    mes,
    fecha,
    enfermedad,
    SUBSTRING_INDEX(localizacion, ',', 1) AS Provincia,
    SUBSTRING_INDEX(SUBSTRING_INDEX(localizacion, ',', 2), ',', -1) AS Distrito,
    SUBSTRING_INDEX(localizacion, ',', -1) AS Departamento,
    nNotificacion,
    CAST(SUBSTRING_INDEX(animalesSuceptibles, ' ', 1) AS UNSIGNED) AS animalesSuceptibles,
    CAST(SUBSTRING_INDEX(nCasos, ' ', 1) AS UNSIGNED) AS nCasos,
    CAST(SUBSTRING_INDEX(nAnimalesMuertos, ' ', 1) AS UNSIGNED) AS nAnimalesMuertos,
    SUBSTRING_INDEX(animalesSuceptibles, ' ', -1) AS Especie,
    CASE 
        WHEN dxLab LIKE 'Negativo%%' THEN 'Negativo'
        WHEN dxLab LIKE 'Positivo%%' THEN 'Positivo'
        WHEN dxLab LIKE 'En Proceso%%' THEN 'En Proceso'
        ELSE NULL
    END AS dxLab
FROM 
    enfermedad
WHERE 
    localizacion LIKE '%%,%%,%%';
"""

# Cargar los datos desde la base de datos con manejo de errores
try:
    df = pd.read_sql_query(query, engine)
    st.success("Datos cargados exitosamente.")
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    df = None

if df is not None:
    # Preprocesamiento de datos
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')
    df['nAnimalesMuertos'] = df['nAnimalesMuertos'].fillna(0)
    
    # Configuración de sidebar
    with st.sidebar:
        especies = df['Especie'].unique()
        especie_seleccionada = st.selectbox("Seleccione una especie", especies)

        # Filtros de fecha separados
        fecha_min = st.date_input("Seleccione la fecha mínima", df['fecha'].min())
        fecha_max = st.date_input("Seleccione la fecha máxima", df['fecha'].max())

        # Filtro por departamento
        departamentos = ['Todos'] + list(df['Departamento'].unique())
        departamento_seleccionado = st.selectbox("Seleccione un departamento (solo para predicción)", departamentos)

    # Filtrar datos por especie y rango de fechas
    df_filtrado = df[(df['Especie'] == especie_seleccionada) & 
                     (df['fecha'] >= pd.to_datetime(fecha_min)) & 
                     (df['fecha'] <= pd.to_datetime(fecha_max))]

    # Filtrar por departamento si no se selecciona "Todos"
    if departamento_seleccionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['Departamento'] == departamento_seleccionado]

    if not df_filtrado.empty:
        # Agregación de datos por fecha
        serie = df_filtrado.groupby('fecha')['nAnimalesMuertos'].sum()

        # Crear un DataFrame para predicciones futuras
        dias_prediccion = 30
        fechas_futuras = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=dias_prediccion)
        
        # Modelos
        st.write(f"### Predicción de mortalidad de animales para la especie {especie_seleccionada}")

        # ARIMA
        try:
            model_arima = ARIMA(serie, order=(5, 1, 0))
            arima_fit = model_arima.fit()
            arima_pred = arima_fit.forecast(steps=dias_prediccion)
        except Exception as e:
            st.warning(f"Error al ajustar ARIMA: {e}")
            arima_pred = [np.nan] * dias_prediccion

        # Holt-Winters
        if len(serie) >= 24:  # Verificar si hay al menos dos ciclos completos
            try:
                hw_model = ExponentialSmoothing(serie, seasonal='add', seasonal_periods=12).fit()
                hw_pred = hw_model.forecast(dias_prediccion)
            except Exception as e:
                st.warning(f"Error al ajustar Holt-Winters: {e}")
                hw_pred = [np.nan] * dias_prediccion
        else:
            st.warning("No hay suficientes datos para ajustar un modelo Holt-Winters con estacionalidad.")
            hw_pred = [np.nan] * dias_prediccion

        # Redes neuronales
        try:
            x = np.arange(len(serie)).reshape(-1, 1)
            y = serie.values
            mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
            mlp.fit(x, y)
            x_pred = np.arange(len(serie), len(serie) + dias_prediccion).reshape(-1, 1)
            nn_pred = mlp.predict(x_pred)
        except Exception as e:
            st.warning(f"Error al ajustar Redes Neuronales: {e}")
            nn_pred = [np.nan] * dias_prediccion
        
        # Graficar los resultados con Plotly (interactivo)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie.index, y=serie, mode='lines', name='Datos reales'))
        fig.add_trace(go.Scatter(x=fechas_futuras, y=arima_pred, mode='lines+markers', name='ARIMA', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=fechas_futuras, y=hw_pred, mode='lines+markers', name='Holt-Winters', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=fechas_futuras, y=nn_pred, mode='lines+markers', name='Redes Neuronales', line=dict(dash='dot')))
        fig.update_layout(title=f"Predicción de mortalidad de animales - {especie_seleccionada}",
                          xaxis_title="Fecha",
                          yaxis_title="Número de muertes",
                          width=1200, height=600)
        st.plotly_chart(fig)

        # Gráfico de evolución por departamento
        st.write("### Evolución por departamento")
        departamentos = df_filtrado['Departamento'].unique()
        departamentos_seleccionados = st.multiselect("Seleccione departamentos", departamentos, default=departamentos)
        
        if departamentos_seleccionados:
            evolucion_departamentos = df_filtrado[df_filtrado['Departamento'].isin(departamentos_seleccionados)]
            evolucion_departamentos = evolucion_departamentos.groupby(['Departamento', 'fecha'])['nCasos'].sum().unstack(0).fillna(0)

            fig2 = go.Figure()
            for departamento in departamentos_seleccionados:
                fig2.add_trace(go.Scatter(x=evolucion_departamentos.index,
                                          y=evolucion_departamentos[departamento],
                                          mode='lines', name=departamento))
            fig2.update_layout(title=f"Evolución de casos por departamento - {especie_seleccionada}",
                               xaxis_title="Fecha",
                               yaxis_title="Número de casos",
                               width=1200, height=600)
            st.plotly_chart(fig2)
        else:
            st.warning("No se seleccionaron departamentos para mostrar.")
    else:
        st.warning("No hay datos para la especie seleccionada o el rango de fechas especificado.")

    # Comparación de modelos
    st.write("### Comparación de modelos")
        
    # Ajustar tamaños antes de calcular métricas
    serie_arima, arima_pred = ajustar_tamano(serie, arima_pred)
    serie_hw, hw_pred = ajustar_tamano(serie, hw_pred)
    serie_nn, nn_pred = ajustar_tamano(serie, nn_pred)
        
    resultados = {
        "Modelo": ["ARIMA", "Holt-Winters", "Redes Neuronales"],
        "MAE": [
            mean_absolute_error(serie_arima, arima_pred) if not np.isnan(arima_pred).all() else np.nan,
            mean_absolute_error(serie_hw, hw_pred) if not np.isnan(hw_pred).all() else np.nan,
            mean_absolute_error(serie_nn, nn_pred) if not np.isnan(nn_pred).all() else np.nan
        ],
        "RMSE": [
            mean_squared_error(serie_arima, arima_pred, squared=False) if not np.isnan(arima_pred).all() else np.nan,
            mean_squared_error(serie_hw, hw_pred, squared=False) if not np.isnan(hw_pred).all() else np.nan,
            mean_squared_error(serie_nn, nn_pred, squared=False) if not np.isnan(nn_pred).all() else np.nan
        ]
    }

    resultados_df = pd.DataFrame(resultados)
    st.dataframe(resultados_df)
        
    st.write("El modelo más óptimo es aquel con menor RMSE.")

