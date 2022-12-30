"""
Esta función controla todas las subfunciones del código:
1. Visualización (lanzar la app de dash)
2. Procesado de los datos (lo que se haría automáticamente en la nube de MBE)

"""


# En este diccionario se guardan las rutas
dictRouter = {
    "ruta_perceptron": "/DataBase/PerceptronOutput.xlsx",    #fichero de la base de datos
    "feature_cols": ['Long', 'Traccion', 'Cond', 'PtaCIzq',
                'PtaTras', 'Techo', 'FLK', 'ECell'],        #features a considerar
    "ruta_ResClus": "/DataBase/ResClus.xlsx"                 #ruta en la que se guardan los resultados del cluster
}



dictParametros = {
    "primera_col_datos": 23,# Primera columna del dataframe con puntos de medición
    "filas_skip": 5,        #filas a saltar (no contienen mediciones, sino límites OK/NOK
    "col_chassis": "JSN",   #nombre de la columna con el chasis
    "fil_limS": 0,          # nr. fil con lim superior de la cota (los parametrizados en perceptron)
    "fil_limI": 1,          # nr. fil con lim superior de la cota (los parametrizados en perceptron)

}

