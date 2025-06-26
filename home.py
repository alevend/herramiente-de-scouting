import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import os
import urllib.parse

dash.register_page(__name__, path="/", name="Home")

# 🔹 Lista delle leghe disponibili
LEAGUES = ["Premier League", "Serie A", "Bundesliga", "La Liga", "Ligue 1"]

ASSETS_PATH = "assets"
DEFAULT_LOGO = "/assets/default_logo.png"

# 🔹 Funzione per ottenere il logo della lega
def get_league_logo(league):
    league_logo_path = f"/assets/{urllib.parse.quote(league)}_logo.png"
    full_path = os.path.join(ASSETS_PATH, f"{league}_logo.png")

    if os.path.exists(full_path):
        return league_logo_path
    else:
        return DEFAULT_LOGO  # Se il logo non esiste, usa un default

# 🔹 Layout della Home Page
layout = dbc.Container(
    [
        html.H1("Ligas", className="text-center my-4", style={"color": "#0082F3"}),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardImg(
                                src=get_league_logo(league),
                                style={"height": "100px", "object-fit": "contain", "backgroundColor": "#1E1E1E",
                                       "padding": "10px", "borderRadius": "10px 10px 0 0"},
                                top=True,
                            ),
                            dbc.CardBody(
                                dbc.Button(
                                    league.upper(),
                                    href=f"/league/2024-25/{urllib.parse.quote(league)}",  # Aggiornato alla stagione 24-25
                                    className="btn-league",
                                    style={"width": "100%", "backgroundColor": "#0A0F26",
                                           "border": "1px solid #0082F3", "fontWeight": "bold", "color": "#FFFFFF"},
                                ),
                                className="text-center",
                                style={"backgroundColor": "#0A0F26"},
                            ),
                        ],
                        className="h-100 shadow-lg custom-card",
                        style={"border": "1px solid #0082F3", "borderRadius": "10px"},
                    ),
                    md=2,
                    className="mb-4",
                )
                for league in LEAGUES
            ],
            justify="center",
        ),
    ],
    fluid=True,
    style={"backgroundColor": "#0A0F26", "color": "#FFFFFF", "padding": "20px"},
)
