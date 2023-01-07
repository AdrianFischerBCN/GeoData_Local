#esta función permite añadir un color de fondo a una columna
def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #dde2eb 0%,
                    #dde2eb {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles


def histo_plot(df_pto, pto, col_clusterizar=False):
    """
    :param df_pto: dataframe
    :param pto:
    :param col_clusterizar:
    :return:
    """
    import plotly.graph_objects as go
    import pandas as pd

    print("iniciando histo_plot")

    if col_clusterizar:
        # si no hay valor, por defecto es False y entonces no clusterizar
        # genera la lista de columnas a extraer del dataframe
        col_filtrar = col_clusterizar[:]
        col_filtrar.append(pto)

        # filtra la dataframe
        df_filtrada = df_pto.loc[:, col_filtrar]
        figura = histo_plot_clusterizada(df_filtrada, pto, col_clusterizar)
        return(figura)


    else:
        # si entra aquí es porque se ha especificado col_clusterizar
        figura = histo_plot_plain(df_pto, pto)
        return figura


def histo_plot_clusterizada(df_filtrada, pto, col_clusterizar):
    """
    Esta función se ejecuta desde la principal, hist_plot. Es la variante en la que además de dibujar el histograma
    se debe clusterizar según una o más propiedades

    :param df_filtrada: dataframe con los datos para hacer el clusterizado
    :param pto: string con el nombre de la columna del dataframe que debe ser dibujado
    :param col_clusterizar: string o lista con los nombres de las columnas a clusterizar
    :return: plotly figure. histograma resultante
    """

    import plotly.graph_objects as go
    import pandas as pd

    # si solo hay una columna lo pasa a una lista para que funcione el resto del programa
    if type(col_clusterizar) == "str":
        col_clusterizar = [col_clusterizar]

    # genera la dataframe con la columna extra para hacer el agrupado posterior
    df_comb = df_filtrada.loc[:]
    df_comb["combo"] = df_filtrada.loc[:, col_clusterizar[0]].astype(str)
    if len(col_clusterizar) > 1:
        for comb in col_clusterizar[1:]:
            df_comb["combo"] = df_comb["combo"] + "_" + df_filtrada.loc[:, comb].astype(str)

    # genera el listado de valores
    df_comb = df_comb.groupby('combo')[pto].apply(list).reset_index(name='new')

    fig = go.Figure()

    # añade un histograma por cada set de datos
    for index, row in df_comb.iterrows():
        fig.add_trace(go.Histogram(x=row["new"], name=row["combo"]))

    # Overlay histograms
    fig.update_layout(barmode='overlay')

    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    return fig

def histo_plot_plain(df_filtrada, pto):
    """
    Esta subfunción de histo_plot se activa cuando no hay que clusterizar
    :param df_filtrada:
    :param pto:
    :return:
    """

    import plotly.graph_objects as go
    import pandas as pd

    x = df_filtrada.loc[:,pto].values.tolist()
    data = go.Histogram(x=x, cumulative_enabled=False)
    fig = go.Figure(data=data)
    print("exiting histoplotplain")
    print(type(fig))
    return fig