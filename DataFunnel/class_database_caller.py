
class database_caller:
    """
    Esta clase permite conectarse a la base de datos y retorna las columnas requeridas para graficar evoluciones,
    tendencias, etc.
    """

    def __init__(self, df_source):
        import pandas as pd
        from Overlord import dictRouter
        from Overlord import dictParametros



        # ***************PARÁMETROS Y VALORES***************
        self.feature_cols = dictRouter["feature_cols"]  # importar las columnas que contienen features
        self.col_datos = dictParametros["primera_col_datos"]  # Primera columna del dataframe con puntos de medición
        self.col_chassis = dictParametros["col_chassis"]  # columna con el chasis
        self.filas_skip = dictParametros["filas_skip"]  # nro de filas a saltar (las que tienen límites superior e inferior
        self.fil_limS = dictParametros["fil_limS"]  # nr. fil con lim superior de la cota (los parametrizados en perceptron)
        self.fil_limI = dictParametros["fil_limI"]  # nr. fil con lim superior de la cota (los parametrizados en perceptron)

        # df que ignora las filas con datos que no son mediciones
        self.df_data = df_source.loc[self.filas_skip + 1:, :]

        # df con las filas saltadas (contienen intervalos y otras infos)
        self.df_skipped = df_source.loc[:self.filas_skip, :]


    def generar_serie(self, pto):
        """
        Esta función retorna una serie de datos filtrada para que no contenga espacios en blanco
        :param pto: nombre del punto a retornar
        :return: dataframe para usar en plotly
        """
        import pandas as pd
        self.df_data.loc[:, pto]

        # lista de columnas a mostrar. Escoge la de los features y la del pto a analizar.
        col_list = self.feature_cols[:]
        col_list.append(pto)

        # filtra dataframe para excluir las líneas sin medición en el punto a analizar
        mask_pto = ~(self.df_data.loc[:, pto]).isna()
        df_pto = self.df_data.loc[mask_pto, col_list]

        return df_pto