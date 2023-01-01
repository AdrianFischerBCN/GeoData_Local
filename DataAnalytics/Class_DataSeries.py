# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:05:33 2022

@author: AFICHER
"""


class DataSeries:
    """
    Una dataseries contiene la información de una serie de datos. 
    Contiene las funciones necesarias para generar las estadísticas
    P.e.: detección de clusters, tendencias temporales, etc.
    
    Parameters
        ----------
        name : string
            nombre de la serie de datos.
            
        X : numpy array n*1
            np.array con los valores de X que se deben clusterizar.
            
        features : dataframe
            dataframe con los valores de los features
            
        lim : tuple 2 valores
            tuple con los valores de los dos límites
            
        chassis : dataframe
            dataframe con los números de chassis
            
    """

    def __init__(self, name, X, features, lim, chassis):
        self.X = X
        self.feature_values = features
        self.feature_names = features.columns.to_list()
        self.lims = lim
        self.nombre = name
        self.chassis = chassis

        # parámetros vacíos, pendientes de inicializar con las funciones
        self.k = None
        self.cluster_labels = None
        self.normals = None
        self.pvals = None

        """
        El crit score es como de grave cataloga cada posible desviación
            1. Dobles procesos
            2. Efectos temporales
            
        Tiene dos componentes:
            1. la puntuación en sí (crit_score)
            2. Información adicional sobre el motivo para el crit_score
        
        """
        self.crit_score_val = [0, 0]
        self.crit_score_logs = ["", ""]

    def calculate_k(self, method="BIC_filtered", h="-1"):

        """
        Esta función calcula el número de clusters. Por defecto, lo hace con el Bayesian Information Criterion
        """
        valid_methods = ["BIC_filtered", "BIC_unfiltered", "KernelDensity"]
        if method == valid_methods[0]:
            self.k_BIC(method="filtered")
        elif method == valid_methods[1]:
            self.k_BIC(method="unfiltered")
        elif method == valid_methods[2]:
            self.k_KernelDensity()
        else:
            raise Exception("No se ha reconocido el método de clasificación. Opciones válidas: " + str(valid_methods))

    def k_BIC(self, method="filtered"):
        """
        Esta función calcula el número de clusters a partir del BIC
        BIC: Bayesian Information Criterions

        Parameters
        ----------
        method : STRING, optional
            Método de cálculo. The default is "filtered".
            Unfiltered: utiliza todos los datos
            Filtered: ignora los extremos de la serie (evita efecto outliers)

        Returns
        -------
        None: Guarda el dato en el parámetro k

        """
        import numpy as np

        from sklearn.mixture import GaussianMixture as GMM

        # inicializa lista de bic (bayesian inform. crit.) y valor mínimo
        lowest_bic = np.infty
        bic = []

        # calcula la longitud de X
        lenX = len(self.X)

        # Número total de gausianas a probar. Empíricamente no me ha detectado más que 4. Pongo 6 por seguridad
        n_components_range = range(1, 7)

        if method == "unfiltered":
            X_calculation = self.X

        if method == "filtered":
            # filtra previamente los valores extremos
            q1 = np.quantile(self.X, q=0.25)
            q3 = np.quantile(self.X, q=0.75)
            IQR = q3 - q1
            xmin = q1 - IQR * 1.5
            xmax = q3 + IQR * 1.5

            # Calcula las longitudes tras filtrar
            len_Xmin = len(self.X[self.X > xmin])
            len_Xmax = len(self.X[self.X < xmax])

            if ((lenX - len_Xmin) / lenX < (1 - 0.96) and (lenX - len_Xmax) / lenX < (1 - 0.96)):
                X_calculation = self.X[np.multiply(self.X > xmin, self.X < xmax)]
                X_calculation = np.expand_dims(X_calculation, axis=1)
            else:
                X_calculation = self.X

        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GMM(
                n_components=n_components, covariance_type="full"
            )
            gmm.fit(X_calculation)
            bic.append(gmm.bic(X_calculation))

        # determina el valor de k (uso argmin, por eso le añado +1 ya que empiezo en 1)
        self.k = np.argmin(bic) + 1

    def k_KernelDensity(self, h=-1):
        """
        Esta función genera una clusterización de los valores de X. Para ello se
        aplican tres pasos:
            1. Estimación del número de clusters aplicando KernelDensity
            2. Identificación de los clusters con gaussian mixture model (sklearn)
            3. Asignación de los puntos de X a cada uno de los clusters

        Devuelve el resultado de la clusterización.


        Parameters
        ----------
        X : array
            np.array con los valores de X que se deben clusterizar.

        Returns
        -------
        labels : array
            np.array con los clusters de cada punto X..

        """

        from sklearn.neighbors import KernelDensity
        from scipy.signal import argrelextrema
        import numpy as np
        import math

        # calcula la varianza
        var = np.var(self.X)
        n = len(self.X)

        # máximo/mínimo
        xmin = np.min(self.X)
        xmax = np.max(self.X)

        # si no se indica una h calcula con la MITAD del bandwidth recomendada según wiki

        if h == -1:
            h = np.power(np.power(var, 2.5) * 4 / (3 * n), 0.2)

        # calcula número de bins y redondea al alza
        bins = np.ptp(self.X) / h
        bins = math.ceil(bins)

        # Aplica Kernel Density a los datos
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(self.X)

        # ************* ENCONTRAR MÍNIMOS **************
        s = np.linspace(xmin, xmax)
        e = kde.score_samples(s.reshape(-1, 1))
        mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]

        # determinar el número de clusters
        self.k = len(s[mi]) + 1

    def asignar_cluster(self):
        """
        Esta función asigna cada punto a un cluster utilizando Gaussian Mixture
        Almacena también los parámetros de los clusters
        
        """
        from sklearn.mixture import GaussianMixture

        if self.k is None:
            print(
                "Advertencia: no se había determinado la cantidad de clusters y se ha estimado con el método por defecto")
            self.calculate_k()

        # ************ ASIGNACIÓN MEDIANTE GMM ***************
        gm = GaussianMixture(n_components=self.k, random_state=0).fit(self.X)
        print("centro de los clusters encontrados: ", gm.means_.T)

        self.cluster_labels = gm.predict(self.X)
        self.normals = {
            "weights": gm.weights_,
            "means": gm.means_.squeeze(),
            "covars": gm.covariances_.squeeze()
        }

    def plot_hist(self, mode="return"):
        """
        Esta función permite graficar un histograma de los datos y en función de los parámetros lo hará de distinta manera.
        
        Parameters
        ----------
        mode : string
            indica si se quiere retornar el objeto o graficarlo
            posibles valores:
                    return: devuelve el gráfico
                    notebook: lo reproduce en el 
        
        visualizador : string
            donde se debe mostrar el gráfico. Por defecto lo hará en el navegador que esté configurado
            
        id_name : id del gráfico
        
        """

        assert mode == "return" or mode == "notebook", "El parámetro 'mode' de la función plot_hist debe ser 'return' o 'notebook'."

        # importar las librerías
        import plotly.offline as pyo
        import plotly.graph_objs as go
        import plotly as plt

        data = [go.Histogram(
            x=self.X.squeeze(),
            marker=dict(
                color="blue"
            ),

        )]

        layout = go.Layout(
            title=dict(
                text=self.nombre + " con límites " + str(self.lims),
            )
        )

        fig = go.Figure(data=data, layout=layout)

        if mode == "return":
            return (fig)
        else:
            fig.show(renderer="notebook")

    def plot_heatmap(self, mode="return"):

        import numpy as np
        import plotly.offline as pyo
        import plotly.graph_objs as go
        import plotly as plt

        data = [go.Heatmap(
            x=self.feature_names,
            xgap=3,
            y=[str(x) for x in np.arange(0, self.k)],  # necesario para que grafique eje como categórico y no enteros
            ygap=3,
            z=np.log(self.pvals.T),
            text=self.pvals.T,
            zmin=-60,
            zmax=0,
            colorscale="Viridis"

        )]

        layout = go.Layout(
            title="Pvalores de cada feature y cluster"
        )

        fig_heatmap_pvals = go.Figure(data=data, layout=layout)

        if mode == "return":
            return fig_heatmap_pvals
        else:
            fig_heatmap_pvals.show(renderer="notebook")

    def calculate_cs_interval(self, dict_CritScore):
        """
        Calcula la primera componente de la CritScore, correspondiente a la distancia entre los clusters encontrados
        Si no hay clusters, o si la certeza de los pvalores encontrados es demasiado baja, entonces ignora este parámetro.
        """

        import numpy as np
        import pandas as pd

        if self.k == 1:
            # no se penaliza debido a:
            # no hay clusters a considerar, por lo tanto no penaliza la distancia entre intervalos
            None

        else:
            if self.k is None:
                print(
                    "Advertencia: no se había determinado la cantidad de clusters y se ha estimado con el método por defecto")
                self.calculate_k()

            # calcular top feature
            top_feature = self.feature_names[np.argmin(self.pvals.min(axis=1))]

            if np.isnan(self.lims[1]):
                # si no hay un intervalo definido se toma por valor 4
                rg_interval = dict_CritScore["cs_InterWidth_defaultInterval"]
            else:
                # si hay un valor definido toma el real, el rango de valor admisibles
                rg_interval = self.lims[1] - self.lims[0]

            df_topfeature = pd.DataFrame(
                {"top_feature": self.feature_values.loc[:, top_feature].tolist(),
                 "x": np.squeeze(self.X)})

            means = (df_topfeature.groupby(by="top_feature").mean()).x.to_numpy()
            rg_features = means.max() - means.min()

            cs_InterWidth = rg_features / rg_interval

            # calcula el crit_score por el intervalo
            self.crit_score_val[0] = self.crit_score_val[0] + dict_CritScore["cs_InterWidth_beta"] * np.power(cs_InterWidth,
                                                                                                      dict_CritScore[
                                                                                                          "cs_InterWidth_power"])

            # **************** Generar indicaciones *******************

            # inicializa a cero la lista de indicaciones
            self.crit_score_logs[0] = []

            # itera todos los pvalores obtenidos
            if np.min(self.pvals) < dict_CritScore["cs_InterWidth_minP"]:
                aux_filamin = np.min(self.pvals, axis=1)
                posmin = np.argmin(aux_filamin)
                self.crit_score_logs[0] = self.feature_names[posmin] + " " + "{:2e}".format(np.min(self.pvals))

            else:
                self.crit_score_logs[0] = "No identificado"

    def pval_clusterizado(self):
        """
        Esta función calcula un p-value de los datos encontrados. Para ello utiliza
        un chisquared test y tomará los valores que encuentre en la columna col_X y
        los labels que encuentre en la columna col_variante.
        El test se puede aplicar solamente a uno de los clusteres a la vez

        Si en total hay una única variante no tiene sentido analizar la distrib.
        El pvalor pasa a ser automáticamente 1

        Parameters
        ----------
        
        Returns
        -------
        None.

        """
        import pandas as pd
        import numpy as np
        from scipy.stats import chisquare

        # inicializar dataframe con los labels:
        df = self.feature_values.copy(deep=True)

        # comprueba si hay labels si los hay
        if self.cluster_labels is None:
            print("Advertencia: no se habían calculado los clusters. Calculándolos ahora con Gaussian Mixture")
            self.asignar_cluster()

        col_cluster = "cluster_labels"

        df["cluster_labels"] = self.cluster_labels
        df["valores"] = self.X
        df["Chasis"] = self.chassis

        i = 0
        j = 0
        nr_features = len(self.feature_values.columns)

        # inicializar pvals
        self.pvals = np.ones((nr_features, self.k))

        for col_variante in self.feature_values.columns:

            if len(df.loc[:, col_variante].unique()) == 1:
                # solo hay una variante, se retorna pval = 1
                self.pvals[i, :] = 1
                i = i + 1

            else:

                j = 0

                for cluster_num in df.cluster_labels.unique():
                    # variable auxiliar: número de casos por cluster y variante
                    aux_full = df.groupby([col_variante, col_cluster]).Chasis.count().reset_index().rename(
                        columns={"Chasis": "Casos"})
                    # print(aux_full)

                    # variable auxiliar: todas las combinaciones aunque no haya casos
                    index = pd.MultiIndex.from_product(
                        [df.loc[:, col_variante].unique(), df.loc[:, col_cluster].unique()],
                        names=[col_variante, col_cluster])
                    aux_full_2 = pd.DataFrame(index=index).reset_index()

                    aux_full = aux_full_2.merge(right=aux_full, left_on=[col_variante, col_cluster],
                                                right_on=[col_variante, col_cluster], how="left", ).fillna(0)

                    # tamaño del cluster
                    aux_cluster_size = df.groupby([col_cluster]).Chasis.count().reset_index().rename(
                        columns={"Chasis": "Casos"})

                    # casos por variante
                    aux_casos = df.groupby([col_variante]).Chasis.count().reset_index().rename(
                        columns={"Chasis": "total"})

                    # añade los casos
                    aux_full = aux_full.merge(aux_casos, how="left", left_on=col_variante, right_on=col_variante)

                    # máscara para el cluster
                    cluster_mask = aux_full[col_cluster] == cluster_num

                    # genera listado de datos vistos
                    seen = aux_full.loc[cluster_mask].Casos.to_numpy()

                    # máscara para el cluster
                    cluster_mask = aux_full[col_cluster] == cluster_num

                    # genera listado de datos vistos
                    expected = aux_full.loc[cluster_mask].total.to_numpy()
                    expected = expected * sum(seen) / sum(expected)

                    # print("variante: ",str(col_variante),", cluster: ",str(cluster_num))
                    # print("real: ",seen,"     esperado: ",np.around(expected,0))
                    self.pvals[i, j] = chisquare(f_obs=seen, f_exp=expected, ddof=0)[1]

                    j = j + 1
                i = i + 1
