import dash
from dash import html
import dash_bootstrap_components as dbc
from utils.cache_config import DASH_CONFIG, SERVER_CONFIG

# Inizializza l'app Dash con il tema dark di Bootstrap e supporto per le pagine multiple
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    pages_folder="pages",
    **DASH_CONFIG
)

# Pagine da visualizzare nella navbar
PAGES_IN_NAVBAR = ["Home", "Estilo de juego top5", "Estilo de juego", "Scouting"]

# Layout principale dell'app con loading state
app.layout = html.Div([
    # Navbar
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink(
                    page["name"],
                    href=page["path"],
                    active="exact"
                )
            ) for page in dash.page_registry.values() if page["name"] in PAGES_IN_NAVBAR
        ],
        brand="⚽ Aplicación de Análisis y Scouting",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-2"
    ),
    # Contenuto principale con loading state
    dbc.Spinner(
        children=[dash.page_container],
        color="primary",
        type="grow",
        fullscreen=False
    )
], className="bg-dark text-white min-vh-100")

# Configura il server
server = app.server

# Avvia il server con le configurazioni ottimizzate
if __name__ == '__main__':
    app.run_server(**SERVER_CONFIG)
