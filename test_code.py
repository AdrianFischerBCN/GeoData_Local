
import json
import numpy as np
import json
import openpyxl
import os

from DataAnalytics.Class_DataSeries import DataSeries
from DataAnalytics.Parametros import dict_CritScore
from Overlord import dictRouter
from Overlord import dictParametros

# cargar librerías y datos
from dash import Dash, dash_table, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

df_res = pd.read_excel("DataBase/ResClus.xlsx", engine="openpyxl")
df_res.sort_values(by="CS_DP", ascending=False, inplace=True)
df_tabla = df_res.loc[:, ["Punto", "CS_DP"]]
df_tabla["id"] = df_tabla.index #añadir columna de id
df_tabla = df_tabla.round({"CS_DP": 4})

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

# inicializar app
app = Dash(__name__,
           external_stylesheets=[dbc.themes.SLATE])

# convertir en diccinario
dict_data = df_tabla.to_dict("records")

# convertir el diccionario en datable
datatable = dash_table.DataTable(
    data=dict_data,
    columns=[{"name": i, "id": i} for i in df_tabla.columns if i != "id"],
    id="DT_DobleProc_Top",
    page_size=16,  # max filas a mostrar
    row_selectable=False,  # para poder seleccionar la fila, en principio no hace falta
    style_data_conditional=(
            data_bars(df_tabla, 'CS_DP')
    )
)

# layout y visualizar
app.layout = html.Div(
    children=[
        dbc.Row(
            children=html.H1(["Visualizador de dobles procesos"])
        ),
        dbc.Row(
            children=(
                dbc.Col(
                    children=datatable,
                    width=3
                ),
                dbc.Col(
                    children="",
                    width=2
                ),
                dbc.Col(
                    children="",
                    width=6
                )
            )),
        dbc.Row(
            children=[
                html.H2("Detalle del punto"),
                html.P(id="testlabel", children="Prueba")]
        ),
        ]) #cierra Div principal y layout



@app.callback(
    Output(component_id="testlabel", component_property="children"),
    Input(component_id="DT_DobleProc_Top", component_property="active_cell")
)
def update_prueba(active_cell):
    if active_cell is None:
        # si no ha sido seleccionada ninguna celda devolver la primera fila
        return str(df_tabla.iloc[0, 0])
    else:
        # devolver siempre la primera columna, que es la que tiene el nombre del punto
        return str(df_tabla.loc[active_cell["row_id"], df_tabla.columns[0]])


if __name__ == '__main__':
    app.run_server(debug=True)
