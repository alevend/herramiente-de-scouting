import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy import stats
import urllib.parse
from utils.config import POSITION_TRANSLATION, PERFILES_KPI_PESOS, ROLE_PROFILES as ROLE_PROFILES_ES, PROFILE_TO_ROLES

# Ruta de los archivos de datos
DATA_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/datos/fbref/base_datos_tutte_le_leghe_renamed.csv")
ORIGINAL_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/base_datos_tutte_le_leghe_renamed.csv")

# Registro de la página
dash.register_page(__name__, path='/scouting')

# Carica il dataset principale, leggendo TUTTE le colonne
df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False, keep_default_na=False, na_values=[''])

# Carica il dataset originale/completo per unire le colonne mancanti
try:
    ORIGINAL_DF = pd.read_csv(ORIGINAL_FILE, encoding="utf-8", low_memory=False)
    if 'Player_ID' in df.columns and 'Player_ID' in ORIGINAL_DF.columns:
        missing_cols = [col for col in ORIGINAL_DF.columns if col not in df.columns]
        if missing_cols:
            print(f"Scouting: Trovate colonne mancanti. Tentativo di unione per: {missing_cols}")
            df['Player_ID'] = df['Player_ID'].astype(str)
            ORIGINAL_DF['Player_ID'] = ORIGINAL_DF['Player_ID'].astype(str)
            cols_to_merge = ['Player_ID'] + missing_cols
            df = pd.merge(df, ORIGINAL_DF[cols_to_merge], on='Player_ID', how='left')
            print("Scouting: Unione completata con successo.")
except FileNotFoundError:
    print(f"Scouting: Attenzione, file originale non trovato: {ORIGINAL_FILE}")
except Exception as e:
    print(f"Scouting: Errore critico durante l'unione dei file: {e}")

# --- CORREZIONE: Rinomina la colonna del contratto dopo l'unione ---
if 'Contract Until' in df.columns:
    df.rename(columns={'Contract Until': 'Contratto'}, inplace=True)
    print("Colonna 'Contract Until' rinominata in 'Contratto'.")

# Coeficientes de liga
coeff_leghe = {
    "Premier League": 2.0, "La Liga": 2.0, "Serie A": 2.0, "Bundesliga": 2.0,
    "Ligue 1": 2.0, "Primeira Liga": 1.5, "Eredivisie": 1.5,
    "Championship": 1.5, "Belgio": 1.5, "Serie B": 1.0
}

def eta_decimale(valore):
    try:
        anni, giorni = str(valore).split('-')
        return int(anni) + int(giorni) / 365
    except:
        try:
            return float(valore)
        except:
            return None

# Función para convertir el valor de mercado en millones
def converti_valore_mercato(valore):
    try:
        if pd.isna(valore) or valore == '':
            return 0.0
        
        # Convierte el valor en string y elimina todos los posibles símbolos del euro
        valore = str(valore)
        valore = valore.replace('â¬', '').replace('â\x82¬', '').replace('€', '').strip()
        
        # Convierte en número
        if 'm' in valore.lower():
            return float(valore.lower().replace('m', ''))
        elif 'k' in valore.lower():
            return float(valore.lower().replace('k', '')) / 1000
        else:
            return float(valore)
    except:
        return 0.0

# Eseguiamo la pre-elaborazione direttamente sul dataframe 'df'
df['Edad'] = df['Edad'].apply(eta_decimale)
df.dropna(subset=['Edad'], inplace=True)
df['Mín'] = df['Mín'].astype(str).str.replace(',', '').astype(float)
df['Market_Value_M'] = df['Market Value'].apply(converti_valore_mercato)

# Traduce "Jun" en "Junio" una sola vez
if 'Contratto' in df.columns:
    df['Contratto'] = df['Contratto'].fillna('N/A').astype(str).str.replace('Jun', 'Junio', case=False).str.strip()
else:
    df['Contratto'] = 'N/A'

# Aplica los coeficientes de liga de manera más eficiente
df["Coeff Lega"] = 1.0  # valor de default
mask_min = df['Mín'] < 1500
df.loc[mask_min, "Coeff Lega"] = 1.5

# Aplica los coeficientes de las ligas conocidas
for lega, coeff in coeff_leghe.items():
    mask_lega = df['Lega'] == lega
    df.loc[mask_lega, "Coeff Lega"] = coeff

# Renombra la liga para la visualización DESPUÉS de haber aplicado los coeficientes
if 'Lega' in df.columns:
    df['Lega'] = df['Lega'].replace('Belgio', 'Jupiler Pro League')

# Get unique leagues for the filter
leagues = sorted([str(l) for l in df['Lega'].unique() if pd.notna(l)])

def get_player_category(age, score):
    """Determine player category based on age and score."""
    if age is None:
        return "N/A"
    
    # Floor the age to get the integer part
    age_floor = int(age)
    
    if 24 <= age_floor <= 40:
        if score >= 90:
            return "Estrella Mundial"
        elif score >= 82:
            return "Jugador Top"
        elif score >= 75:
            return "Buen Jugador"
        elif score >= 68:
            return "Jugador de Media Tabla"
        elif score >= 63:
            return "Jugador de Baja Tabla"
        else:
            return "Jugador de División 2"
    elif 16 <= age_floor <= 23:
        if score >= 87:
            return "Joven Estrella Mundial"
        elif score >= 80:
            return "Futura Estrella"
        elif score >= 74:
            return "Joven con Calidad"
        elif score >= 65:
            return "Buen Joven"
        else:
            return "Joven"
    else:
        return "N/A"

# Elimina las funciones y listas no más necesarias
min_eta, max_eta = (int(df['Edad'].min()), int(df['Edad'].max())) if not df.empty else (15, 40)

# --- MAPPA DI TRADUZIONE SPAGNOLO -> ITALIANO ---
# Necessaria perché i nomi dei profili nei dati (colonne Score_*, Category_*) sono in italiano.
ROLE_PROFILES_IT = {
    "Goalkeeper": ["Portero"],
    "Centre-Back": ["Centrale - Marcatura", "Centrale - Deep Distributor", "Recuperatore 20-80"],
    "Left-Back": ["Sentinel Fullback 20-80", "Quinto ATA 70-30", "Arnold 80-20"],
    "Right-Back": ["Sentinel Fullback 20-80", "Quinto ATA 70-30", "Arnold 80-20"],
    "Defensive Midfield": ["Difensivo", "Play", "Box-to-Box"],
    "Central Midfield": ["Difensivo", "Play", "Box-to-Box"],
    "Midfielder": ["Difensivo", "Play", "Box-to-Box"],
    "Attacking Midfield": ["Dieci", "Box-to-Box", "Interior 90-10"],
    "Left Midfield": ["Quinto ATA 70-30", "Arnold 80-20", "1vs1"],
    "Right Midfield": ["Quinto ATA 70-30", "Arnold 80-20", "1vs1"],
    "Left Winger": ["1vs1", "Interior 90-10"],
    "Right Winger": ["1vs1", "Interior 90-10"],
    "Second Striker": ["Dieci", "Mobile Forward", "Target", "1vs1", "Interior 90-10"],
    "Centre-Forward": ["Mobile Forward", "Target", "Lethal Striker"]
}
SPANISH_TO_ITALIAN_PROFILES = {}
for role, spanish_profiles in ROLE_PROFILES_ES.items():
    if role in ROLE_PROFILES_IT:
        italian_profiles = ROLE_PROFILES_IT[role]
        for i, spanish_profile in enumerate(spanish_profiles):
            if i < len(italian_profiles):
                SPANISH_TO_ITALIAN_PROFILES[spanish_profile] = italian_profiles[i]
# --- FINE MAPPA ---

# Layout de la página
layout = dbc.Container([
    html.H1("Aplicación de Análisis y Scouting", className="text-center my-4", style={"color": "#0082F3"}),
    
    dbc.Row([
        dbc.Col([
            html.H4("Filtros", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    # Selección de perfil
                    html.Label("Perfil KPI"),
                    dcc.Dropdown(
                        id='profile-dropdown',
                        options=[{"label": p, "value": p} for p in PERFILES_KPI_PESOS.keys()],
                        value=list(PERFILES_KPI_PESOS.keys())[0],
                        clearable=False,
                        style={
                            'color': 'black',
                            'background-color': 'white'
                        }
                    ),
                    
                    # Checklist para las posiciones
                    html.Label("Posiciones", className="mt-3"),
                    dbc.Checklist(
                        id='position-checklist',
                        options=[],
                        value=[],
                        inline=True,
                        style={'color': 'white', 'margin-left': '10px'}
                    ),
                    
                    # Filtro de ligas
                    html.Label("Ligas", className="mt-3"),
                    dbc.Checklist(
                        id='league-checklist',
                        options=[{'label': lega, 'value': lega} for lega in leagues],
                        value=leagues,
                        style={'color': 'white', 'margin-left': '10px', 'max-height': '150px', 'overflow-y': 'auto', 'border': '1px solid #444', 'padding': '10px', 'border-radius': '5px'}
                    ),
                    
                    # Filtro Agente Libre
                    html.Label("Agente Libre", className="mt-3"),
                    dbc.Checklist(
                        id='agente-libre-checklist',
                        options=[{'label': '', 'value': 'AGENTE_LIBRE'}],
                        value=[],
                        inline=True,
                        style={'color': 'white', 'margin-left': '10px'}
                    ),
                    
                    # Filtro minutos
                    html.Label("Rango de minutos jugados", className="mt-3"),
                    dcc.RangeSlider(
                        id='min-minutes-slider',
                        min=0,
                        max=int(df["Mín"].max()),
                        step=100,
                        value=[1000, int(df["Mín"].max())],
                        marks={i: str(i) for i in range(0, int(df["Mín"].max()) + 1, 500)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    # Filtro edad
                    html.Label("Rango de Edad", className="mt-3"),
                    dcc.RangeSlider(
                        id='age-range-slider',
                        min=min_eta,
                        max=max_eta,
                        step=1,
                        value=[min_eta, max_eta],
                        marks={i: str(i) for i in range(min_eta, max_eta + 1, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),

                    # Filtro valor de mercado
                    html.Label("Valor de mercado (M€)", className="mt-3"),
                    dcc.RangeSlider(
                        id='market-value-slider',
                        min=0,
                        max=250,
                        step=5,
                        value=[0, 250],
                        marks={i: f"{i}M€" for i in range(0, 251, 50)}
                    ),
                ])
            ], className="mb-4")
        ], width=3),
        
        # Tabla de jugadores
        dbc.Col([
            html.H4("Jugadores", className="mb-3"),
            dash_table.DataTable(
                id='players-table',
                page_size=20,
                style_header={
                    'backgroundColor': '#1a1a1a',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'backgroundColor': '#0A0F26',
                    'color': 'white',
                    'textAlign': 'left'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#141824'
                    },
                    {
                        'if': {'column_id': 'Perfil'},
                        'color': '#0082F3',
                        'cursor': 'pointer',
                        'textDecoration': 'none'
                    }
                ],
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Perfil'},
                        'width': '100px'
                    }
                ],
                markdown_options={"html": True},
                sort_action='native',
                filter_action='native'
            )
        ], width=9)
    ])
], fluid=True, style={"backgroundColor": "#0A0F26", "color": "#FFFFFF", "padding": "20px"})

@callback(
    [Output('position-checklist', 'options'),
     Output('position-checklist', 'value')],
    [Input('profile-dropdown', 'value')]
)
def update_position_checklist(selected_profile):
    if not selected_profile:
        return [], []
    
    valid_positions = PROFILE_TO_ROLES.get(selected_profile, [])
    
    options = [{'label': POSITION_TRANSLATION.get(pos, pos), 'value': pos} for pos in valid_positions]
    
    # Selecciona todas las posiciones por default
    values = valid_positions
    
    return options, values

@callback(
    [Output("players-table", "data"),
     Output("players-table", "columns")],
    [Input("profile-dropdown", "value"),
     Input("position-checklist", "value"),
     Input("league-checklist", "value"),
     Input("min-minutes-slider", "value"),
     Input("age-range-slider", "value"),
     Input("market-value-slider", "value"),
     Input("agente-libre-checklist", "value")]
)
def update_table(profilo, posizioni_valide, selected_leagues, minuti_range, age_range, market_value_range, agente_libre_status):
    if not profilo or not posizioni_valide or not selected_leagues:
        return [], []

    # Aplica los filtros base
    min_minuti, max_minuti = minuti_range
    mask = (
        df["Position"].isin(posizioni_valide) &
        df["Lega"].isin(selected_leagues) &
        df["Mín"].notna() &
        (df["Mín"] >= min_minuti) &
        (df["Mín"] <= max_minuti)
    )

    # Aplica el filtro del market value si se especifica
    if market_value_range:
        min_value, max_value = market_value_range
        mask = mask & (df["Market_Value_M"] >= min_value) & (df["Market_Value_M"] <= max_value)
    
    dff = df[mask].copy()

    if age_range:
        dff = dff[(dff["Edad"] >= age_range[0]) & (dff["Edad"] <= age_range[1])]

    # Aplica filtro agente libre
    if agente_libre_status:
        dff = dff[dff['Contratto'] == 'Junio 30, 2025']
    
    # Logica per visualizzare lo score corretto
    if profilo:
        # Traduce il profilo da spagnolo a italiano per trovare la colonna giusta
        italian_profile = SPANISH_TO_ITALIAN_PROFILES.get(profilo)
        if italian_profile:
            score_col = f'Score_{italian_profile}'
            category_col = f'Category_{italian_profile}'
            
            # Se le colonne esistono, le usa per 'Rating' e 'Categoría'
            if score_col in dff.columns and category_col in dff.columns:
                dff['Rating'] = dff[score_col].round(1)
                dff['Categoría'] = dff[category_col]
            else:
                dff['Rating'] = 0
                dff['Categoría'] = 'N/A'
        else:
            dff['Rating'] = 0
            dff['Categoría'] = 'N/A'
    else:
        # Se nessun profilo è selezionato, le colonne rimangono vuote
        dff['Rating'] = 0
        dff['Categoría'] = 'N/A'

    # Add player category based on age and score
    dff["Categoria"] = dff.apply(
        lambda row: get_player_category(row["Edad"], row["Rating"]), 
        axis=1
    )

    dff = dff.sort_values(by="Rating", ascending=False)
    
    # Añade la columna Perfil con el link
    dff['Perfil'] = dff['Name'].apply(
        lambda x: f'[Ver Perfil](/player/{urllib.parse.quote(x)})'
    )
    
    # Ricalcula Market_Value_M del Market Value original
    dff['Market_Value_M'] = dff['Market Value'].apply(converti_valore_mercato)
    
    # Formatta el Market Value usando Market_Value_M
    dff['Market Value'] = dff['Market_Value_M'].apply(
        lambda x: f"€{x:.1f}M" if pd.notna(x) else "N/A"
    )

    # La traducción del contratto se ha movido al inicio de carga
    
    # Traduce las posiciones
    dff['Position'] = dff['Position'].map(POSITION_TRANSLATION).fillna(dff['Position'])
    
    # Columnas para mostrar y su traducción
    col_mapping = {
        "Name": "Nombre", "Edad": "Edad", "Height": "Altura", "Foot": "Pie",
        "Position": "Posición", "Squadra": "Equipo", "Market Value": "Valor de Mercado", "Contratto": "Fin de Contrato",
        "Mín": "Min", "Rating": "Rating", "Categoria": "Categoría", "Perfil": "Perfil"
    }
    
    df_display = dff.rename(columns=col_mapping)
    
    # Formatta los minutos sin coma
    if 'Min' in df_display.columns:
        df_display['Min'] = df_display['Min'].round(0).astype(int)

    colonne_mostrate = [
        "Nombre", "Edad", "Altura", "Pie", "Posición", "Equipo", "Valor de Mercado", "Fin de Contrato", "Min", "Rating", "Categoría", "Perfil"
    ]
    
    data = df_display[colonne_mostrate].round(1).to_dict("records")
    columns = [{"name": col, "id": col, "presentation": "markdown"} if col == "Perfil" else {"name": col, "id": col} for col in colonne_mostrate]
    
    return data, columns 