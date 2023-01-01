import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__,
                use_pages=True,
                external_stylesheets=[dbc.themes.SLATE],
                assets_folder="assets",
                title="MultiPageApp")

#necesario para añadir al servidor
server = app.server

app.layout = dbc.Row([
    dbc.Col(
        children="",
        width=2
    ),
    dbc.Col(
        children=dash.page_container,
        width=9
    )]

)



if __name__ == "__main__":
    app.run_server(debug=True)