"""
Esta función procesa la información disponible y genera métricas para ser visualizadas.
Es una versión que solo sirve para el Excel actualmente disponible.
Si se implementa después con el servidor hará falta diseñar un funnel propio
"""

# ***************LIBRERÍAS VARIAS***************
import pandas as pd
import openpyxl
import numpy as np
import os

from DataAnalytics.Class_DataSeries import DataSeries
from DataAnalytics.Parametros import dict_CritScore
from Overlord import dictRouter
from Overlord import dictParametros

ruta_perceptron = os.path.dirname(os.path.dirname(__file__)) + dictRouter["ruta_perceptron"]
ruta_resultado = os.path.dirname(os.path.dirname(__file__)) + dictRouter["ruta_ResClus"]

print("arrancando")

# ***************PARÁMETROS Y VALORES***************
feature_cols = dictRouter["feature_cols"]       #importar las columnas que contienen features
col_datos = dictParametros["primera_col_datos"] # Primera columna del dataframe con puntos de medición
col_chassis = dictParametros["col_chassis"]     # columna con el chasis
filas_skip = dictParametros["filas_skip"]       # nro de filas a saltar (las que tienen límites superior e inferior
fil_limS = dictParametros["fil_limS"]           # nr. fil con lim superior de la cota (los parametrizados en perceptron)
fil_limI = dictParametros["fil_limI"]           # nr. fil con lim superior de la cota (los parametrizados en perceptron)

#cargar los datos
df_source = pd.read_excel(ruta_perceptron, engine="openpyxl")

#df que ignora las filas con datos que no son mediciones
df_data = df_source.loc[filas_skip + 1:, :]

#df con las filas saltadas (contienen intervalos y otras infos)
df_skipped = df_source.loc[:filas_skip, :]

# crea una lista de series
lst_series = []
lst_points = []

# esta es la variante manual para ciertas columnas
"""listado_tops_manual = ['D161I[Y]',
                       'CF24P501DA[Z]',
                       'CF24P502DA[Z]',
                       'CF13T526D1A[X]',
                       'L505I[Y]',
                       'CF13T526I1A[X]',
                       'D158I[X]',
                       'Distance Z[Z]',
                       'L455I[Z]',
                       'D120I[X]',
                       'CF24P502IA[Z]',
                       'CF24P503DA[Z]',
                       'CF24P500DA[Z]',
                       'CF24P501IA[Z]',
                       'D158I[Z]',
                       'CF24P503IA[Z]',
                       'L515I3[Z]',
                       'P501D[Z]',
                       'CF7D121IA[X]']
for col_trend in listado_tops_manual:                       
"""

#itera la totalidad de las columnas
for col_trend in df_data.columns[col_datos:]:

    X = df_data.loc[:, col_trend]

    # filtrar las que no son nan
    mask = ~X.isna()
    if mask.sum() > 0:
        # extrae datos para inicializar la serie
        X = df_data.loc[mask, col_trend].to_numpy()
        X = np.expand_dims(X, axis=1)

        # extrae feature vectors, límites y chasis
        features = df_data.loc[mask, feature_cols]
        lim = (df_skipped.loc[fil_limI, col_trend], df_skipped.loc[fil_limS, col_trend])
        chassis = df_data.loc[mask, col_chassis]

        # inicializa la serie
        serie = DataSeries(col_trend, X, features, lim, chassis)

        # calcula datos de la serie
        serie.calculate_k()
        serie.asignar_cluster()
        serie.pval_clusterizado()
        serie.calculate_cs_interval(dict_CritScore)

        lst_series.append(serie)
        lst_points.append(serie.nombre)



# ***************** Preparar resultados para poder ser almacenados y/o procesados ************
# crear dataframe inicialmente vacío
df_headers = ["Punto", "CS_DP", "CS_DP_Logs", "DP_pvals", "DP_weights", "DP_means", "DP_covars"]
df_res = pd.DataFrame(columns=df_headers)

# iterar listados para poder almacenar la info
total_puntos = len(lst_points)
pto = 0

while pto < total_puntos:
    nueva_fila = {
        lst_points[pto]: [lst_points[pto],
                  lst_series[pto].crit_score_val[0],
                  lst_series[pto].crit_score_logs[0],
                  str(lst_series[pto].pvals),
                  str(lst_series[pto].normals["weights"]),
                  str(lst_series[pto].normals["means"]),
                  lst_series[pto].normals["covars"]]}

    df_fila = pd.DataFrame.from_dict(nueva_fila, columns=df_headers, orient='index')
    df_res = pd.concat([df_res, df_fila], ignore_index=True)
    pto = pto + 1

df_res.to_excel(ruta_resultado)