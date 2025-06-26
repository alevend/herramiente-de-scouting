import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import os
import urllib.parse
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path_template="/league/<season>/<league_name>", name="League")

ASSETS_PATH = "assets"
DEFAULT_LOGO = "/assets/default_logo.png"

# 🔹 Funzione per ottenere i loghi delle squadre in base alla stagione selezionata
def get_team_logos(league_name, season_selected):
    league_name = urllib.parse.unquote(league_name)
    season_selected = urllib.parse.unquote(season_selected)

    path = os.path.join(ASSETS_PATH, season_selected, league_name)

    if not os.path.exists(path):
        print(f"⚠️ Percorso non trovato: {path}")
        return []

    teams = [team for team in os.listdir(path) if os.path.isdir(os.path.join(path, team))]
    
    logos = []
    for team in teams:
        logo_path = f"/assets/{season_selected}/{league_name}/{team}/logo_{team}.png"
        full_logo_path = os.path.join(ASSETS_PATH, season_selected, league_name, team, f"logo_{team}.png")

        # Se il file non esiste, usa un logo di default
        if not os.path.exists(full_logo_path):
            logo_path = DEFAULT_LOGO

        logos.append({"team": team.replace("_", " "), "logo": logo_path})

    return logos

# 🔹 Layout iniziale con dropdown per selezionare la stagione
layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),  # Per leggere l'URL
        html.H2(id="league-page-title", className="text-center my-4", style={"color": "#0082F3"}),

        # 🔹 Dropdown per selezione stagione
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="league-season-dropdown",
                    options=[
                        {"label": "2024-25", "value": "2024-25"},
                        {"label": "2023-24", "value": "2023-24"},
                        {"label": "2022-23", "value": "2022-23"}
                    ],
                    value="2024-25",  # Stagione predefinita aggiornata
                    clearable=False,
                    className="dropdown-season",
                    style={
                        "backgroundColor": "#0A0F26",
                        "color": "#FFFFFF",
                        "border": "2px solid #0082F3",
                        "borderRadius": "5px",
                        "fontSize": "16px",
                        "padding": "10px",
                        "width": "50%",
                    },
                ),
                width=3,
            ),
            justify="center",
            className="mb-3",
        ),

        dbc.Row(id="league-teams-container"),  # Cambiato l'ID per renderlo unico
    ],
    fluid=True,
    style={"backgroundColor": "#0A0F26", "color": "#FFFFFF", "padding": "20px"},
)

# 🔹 Callback per aggiornare la lista delle squadre
@callback(
    [
        Output("league-page-title", "children"),
        Output("league-teams-container", "children")
    ],
    [
        Input("league-season-dropdown", "value"),
        Input("url", "pathname")
    ]
)
def update_teams(season_selected, pathname):
    """
    Aggiorna la lista delle squadre in base alla stagione selezionata.
    """
    if not pathname:
        raise PreventUpdate

    parts = pathname.strip("/").split("/")
    if len(parts) < 3:
        raise PreventUpdate
    
    league_name = parts[2]  # Estraggo il nome della lega dall'URL
    league_name = urllib.parse.unquote(league_name)  # Decodifica il nome della lega

    logos = get_team_logos(league_name, season_selected)

    if not logos:
        return f"{league_name} Teams - {season_selected}", html.P("⚠️ Nessun logo trovato.", className="text-danger text-center")

    return f"{league_name} Teams - {season_selected}", [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(
                        src=team["logo"],
                        style={
                            "height": "80px",
                            "object-fit": "contain",
                            "backgroundColor": "#1E1E1E",
                            "padding": "10px",
                            "borderRadius": "10px 10px 0 0",
                        },
                        top=True,
                    ),
                    dbc.CardBody(
                        dbc.Button(
                            team["team"],
                            href=f"/team/{urllib.parse.quote(league_name)}/{urllib.parse.quote(team['team'])}",
                            className="btn-team",
                            style={
                                "width": "100%",
                                "backgroundColor": "#0A0F26",
                                "border": "2px solid #0082F3",
                                "fontWeight": "bold",
                                "color": "#FFFFFF",
                                "padding": "10px",
                                "borderRadius": "5px",
                            },
                        ),
                        className="text-center",
                        style={"backgroundColor": "#0A0F26"},
                    ),
                ],
                className="h-100 shadow-sm custom-card",
                style={"border": "2px solid #0082F3", "borderRadius": "10px"},
            ),
            md=2,
            className="mb-4",
        )
        for team in logos
    ]



