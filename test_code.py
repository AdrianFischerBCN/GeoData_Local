

import json
import numpy as np
import json
import openpyxl

import json

import openpyxl
import numpy as np
import json

import os

from DataAnalytics.Class_DataSeries import DataSeries
from DataAnalytics.Parametros import dict_CritScore
from Overlord import dictRouter
from Overlord import dictParametros

import plotly.graph_objects as go

#esta función permite añadir un color de fondo a una columna
from PageComponents.Dash_components_functions import data_bars

# cargar librerías y datos
from dash import Dash, dash_table, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

df_res = pd.read_excel("DataBase/ResClus.xlsx", engine="openpyxl")
df_res.sort_values(by="CS_DP", ascending=False, inplace=True)
df_tabla = df_res.loc[:, ["Punto", "CS_DP"]]
df_tabla["id"] = df_tabla.index #añadir columna de id

df_tabla = df_tabla.round({"CS_DP": 4})



# inicializar app
app = Dash(__name__,
           external_stylesheets=[dbc.themes.SLATE])

# convertir los resultados del Crit_Score de los dobles procesos en diccinario
dict_data = df_tabla.to_dict("records")

# generar el DataTable del CS de dobles procesos
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

# generar una card con información de los puntos que están por encima de 0
mask_CS_DP = df_tabla.CS_DP>0
nr_CS_DP = len(df_tabla.loc[mask_CS_DP])
card_CS_DP = dbc.Card(
    children=[
        html.H5('Dobles Procesos'),
        html.P(nr_CS_DP, style={'color': 'white', 'fontSize': 70, 'font-weight': 'bold'}),
        html.P("Total de dobles procesos significativos encontrados")],
    body=True,
    style={'textAlign': 'center', 'color': 'white'},
    color='DimGrey'
)

# generar un histograma con el CritScore
data = go.Histogram(x=df_tabla.CS_DP, cumulative_enabled=True)
fig_layout = go.Layout(
    title={
        "text": "Histograma CS",
        'x': 0.5,
        'xanchor': 'center',
        "font": {"color": "white"}},
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=250,
    xaxis={
        "showgrid": False},
    yaxis={
        "showgrid": False},
)
CS_DP_histogram = dcc.Graph(figure=go.Figure(data=data, layout=fig_layout))



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
                    children=html.Div([
                        card_CS_DP,
                        html.Br(),
                        CS_DP_histogram]),
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
