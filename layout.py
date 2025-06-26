from dash import dcc, html
import pages.home as home  # Importiamo la home come pagina predefinita

layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Permette la navigazione
    html.Div(id='page-content', children=home.layout)  # Mostra la home all'avvio
])


