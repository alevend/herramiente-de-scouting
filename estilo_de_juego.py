import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import urllib.parse
from utils.team_utils import get_canonical_team_name

# Register the page
dash.register_page(__name__, path='/estilo-de-juego')

# Lista delle leghe disponibili
AVAILABLE_LEAGUES = {
    'Serie A': {'wyscout': 'Serie A'},
    'Premier League': {'wyscout': 'Premier League'},
    'La Liga': {'wyscout': 'La Liga'},
    'Bundesliga': {'wyscout': 'Bundesliga'},
    'Ligue 1': {'wyscout': 'Ligue 1'},
    'Serie B': {'wyscout': 'Serie B'},
    'Primeira Liga': {'wyscout': 'Primeira Liga'},
    'Eredivisie': {'wyscout': 'Eredivise'},
    'Championship': {'wyscout': 'Championship'},
    'Bundesliga 2': {'wyscout': 'Bundesliga 2'},
    'Ligue 2': {'wyscout': 'Ligue 2'},
    'Super Lig': {'wyscout': 'Super Lig'},
    'Jupiler Pro League': {'wyscout': 'Jupiler Pro League'}
}

# Mapping dei nomi delle squadre per gestire le differenze tra nomi cartelle e nomi file
TEAM_NAME_MAPPING = {
    'Premier League': {
        'Nott\'ham Forest': 'Nottingham Forest',
        'Manchester Utd': 'Manchester United',
        'Newcastle Utd': 'Newcastle United',
        'Wolves': 'Wolverhampton Wanderers'
    },
    'La Liga': {
        'Atlético Madrid': 'Atlético Madrid',
        'Celta Vigo': 'Celta de Vigo',
        'Leganés': 'Leganés'
    }
}

def process_team_stats(file_path, league=None):
    """Processa i dati statistici di una squadra da un file Excel."""
    try:
        # Mappa per far corrispondere il nome della lega dal dropdown a quello nel file Excel
        league_to_competition = {
            'Serie A': 'Italy. Serie A',
            'Premier League': 'England. Premier League',
            'La Liga': 'Spain. La Liga',
            'Bundesliga': 'Germany. Bundesliga',
            'Ligue 1': 'France. Ligue 1',
            'Serie B': 'Italy. Serie B',
            'Primeira Liga': 'Portugal. Primeira Liga',
            'Eredivisie': 'Netherlands. Eredivisie',
            'Championship': 'England. Championship',
            'Bundesliga 2': 'Germany. 2. Bundesliga',
            'Ligue 2': 'France. Ligue 2',
            'Super Lig': 'Turkey. Süper Lig',
            'Jupiler Pro League': 'Belgium. Jupiler Pro League'
        }
        competition_name = league_to_competition.get(league)

        if "2023-2024" in file_path:
            team_folder_name = os.path.basename(file_path)
            canonical_name = get_canonical_team_name(team_folder_name.replace("_", " "))
            
            indices_file = os.path.join(file_path, "Indices.xlsx")
            general_file = os.path.join(file_path, "General.xlsx")
            
            df_indices = pd.read_excel(indices_file)
            df_general = pd.read_excel(general_file)
            
            ppda = df_indices.iloc[0, 3]
            passes_per_possession = df_indices.iloc[0, 2]
            recoveries = df_general.iloc[:, 22].mean()
            
            return canonical_name, ppda, passes_per_possession, recoveries
            
        else:
            df = pd.read_excel(file_path)
            team_name_from_file = os.path.basename(file_path).split('.')[0].replace("Team Stats ", "")
            
            # Applica la mappatura per trovare il nome corretto da cercare nella colonna 'Equipo'
            league_specific_mapping = TEAM_NAME_MAPPING.get(league, {})
            equipo_name_to_search_in_col = league_specific_mapping.get(team_name_from_file, team_name_from_file)
            
            # Il nome canonico da ritornare è quello del file, per coerenza nel grafico
            canonical_name_to_return = get_canonical_team_name(team_name_from_file)
            # Il nome da cercare è quello mappato
            canonical_name_to_search = get_canonical_team_name(equipo_name_to_search_in_col)

            # Assicurati che i nomi delle colonne siano stringhe e senza spazi extra
            df.columns = [str(c).strip() for c in df.columns]

            # Filtra per squadra usando il nome mappato
            team_rows = df[df.iloc[:, 4].apply(lambda x: get_canonical_team_name(str(x)) == canonical_name_to_search)]
            
            # Filtra ulteriormente per competizione se il nome è disponibile
            if competition_name and not team_rows.empty:
                team_rows = team_rows[team_rows.iloc[:, 2] == competition_name]

            if team_rows.empty:
                return None, None, None, None
            
            # Calcola la media dei KPI solo sulle righe filtrate
            ppda_values = team_rows.iloc[:, 108].dropna()
            passes_values = team_rows.iloc[:, 104].dropna()
            recoveries_values = team_rows.iloc[:, 22].dropna()
            
            if ppda_values.empty or passes_values.empty or recoveries_values.empty:
                return None, None, None, None
            
            ppda = ppda_values.mean()
            passes_per_possession = passes_values.mean()
            recoveries = recoveries_values.mean()
            
            return canonical_name_to_return, ppda, passes_per_possession, recoveries
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None

# Layout
layout = dbc.Container([
    html.H1("Estilo de Juego", className="text-center my-4", style={"color": "#0082F3"}),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='league-dropdown',
                options=[{'label': league, 'value': league} for league in AVAILABLE_LEAGUES.keys()],
                value='Serie A',
                style={
                    'backgroundColor': '#0A0F26',
                    'color': 'black',
                    'marginBottom': '20px'
                }
            )
        ], width=6, className="mx-auto")
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='season-dropdown',
                options=[
                    {'label': '2024-25', 'value': '2024-2025'},
                    {'label': '2023-24', 'value': '2023-2024'}
                ],
                value='2024-2025',
                style={
                    'backgroundColor': '#0A0F26',
                    'color': 'black',
                    'marginBottom': '20px'
                }
            )
        ], width=6, className="mx-auto", id='season-dropdown-col', style={'display': 'none'})
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='playing-style-scatter',
                config={'displayModeBar': False},
                style={'height': '600px'}
            )
        ], width=9),
        dbc.Col([
            html.Div(id='quadrant-legend', className="mt-4")
        ], width=3)
    ])
], fluid=True, style={"backgroundColor": "#0A0F26", "color": "#FFFFFF", "padding": "20px"})

@callback(
    Output('season-dropdown-col', 'style'),
    Input('league-dropdown', 'value')
)
def toggle_season_dropdown(selected_league):
    if selected_league in ['Serie A', 'Serie B']:
        return {'display': 'block'}
    return {'display': 'none'}

@callback(
    [Output('playing-style-scatter', 'figure'),
     Output('quadrant-legend', 'children')],
    [Input('league-dropdown', 'value'),
     Input('season-dropdown', 'value')]
)
def update_scatter(selected_league, selected_season):
    """Aggiorna il grafico scatter e la legenda."""
    # Costruisci il percorso ai file delle statistiche
    if selected_season == "2023-2024":
        # Per la stagione 2023-2024, il percorso è diverso
        stats_path = f"datos/wyscout/{selected_season}/{selected_season}/{AVAILABLE_LEAGUES[selected_league]['wyscout'].upper()}_{selected_season}"
        # Raccogli i dati di tutte le squadre
        team_data = []
        # Cerca le cartelle delle squadre invece dei file Excel
        for team_folder in glob.glob(os.path.join(stats_path, "*/")):
            team_name, ppda, passes, recoveries = process_team_stats(team_folder.rstrip("/"), selected_league)
            if team_name and ppda is not None and passes is not None and recoveries is not None:
                team_data.append({
                    'team': team_name,
                    'ppda': ppda,
                    'passes': passes,
                    'recoveries': recoveries
                })
    else:
        # Per la stagione 2024-2025, usa il percorso originale
        stats_path = f"datos/wyscout/{selected_season}/{AVAILABLE_LEAGUES[selected_league]['wyscout']}"
        # Raccogli i dati di tutte le squadre
        team_data = []
        for team_file in glob.glob(os.path.join(stats_path, "*.xlsx")):
            team_name, ppda, passes, recoveries = process_team_stats(team_file, selected_league)
            if team_name and ppda is not None and passes is not None and recoveries is not None:
                team_data.append({
                    'team': team_name,
                    'ppda': ppda,
                    'passes': passes,
                    'recoveries': recoveries
                })
    
    # Se non ci sono dati, ritorna un grafico vuoto
    if not team_data:
        return go.Figure(), []
    
    # Crea il DataFrame
    df = pd.DataFrame(team_data)
    
    # Calcola le medie
    mean_ppda = df['ppda'].mean()
    mean_passes = df['passes'].mean()
    
    # Crea il grafico scatter
    fig = go.Figure()
    
    # Aggiungi linee tratteggiate per le medie
    fig.add_hline(y=mean_passes, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_vline(x=mean_ppda, line_dash="dash", line_color="white", opacity=0.5)
    
    # Calcola i range degli assi
    x_range = max(df['ppda']) - min(df['ppda'])
    y_range = max(df['passes']) - min(df['passes'])
    
    # Normalizza i valori di recuperi per la scala di colori (0-1)
    min_recoveries = df['recoveries'].min()
    max_recoveries = df['recoveries'].max()
    
    # Aggiungi i punti e le etichette con scala di colori
    fig.add_trace(go.Scatter(
        x=df['ppda'],
        y=df['passes'],
        mode='markers+text',
        text=[f"{team}." for team in df['team']],
        textposition="top center",
        marker=dict(
            size=12,
            color=df['recoveries'],
            colorscale=[
                [0, 'red'],
                [0.5, 'yellow'],
                [1, 'green']
            ],
            showscale=True,
            colorbar=dict(
                title="Balones Recuperados Altos",
                titleside="right",
                thickness=15,
                len=0.5,
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            line=dict(color='white', width=1)
        ),
        textfont=dict(color='white'),
        hovertemplate='<b>%{text}</b><br>' +
                     'PPDA: %{x:.2f}<br>' +
                     'Passes per Possession: %{y:.2f}<br>' +
                     'Balones Recuperados Altos: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Aggiungi le etichette dei quadranti
    quadrant_annotations = [
        dict(
            x=min(df['ppda']) + x_range*0.15,
            y=max(df['passes']) - y_range*0.15,
            text="<b>Combinativo con presión alta</b>",
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color='white', size=14),
            bgcolor='rgba(10, 15, 38, 0.8)',
            borderpad=8
        ),
        dict(
            x=min(df['ppda']) + x_range*0.15,
            y=min(df['passes']) + y_range*0.15,
            text="<b>Directo con presión alta</b>",
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(color='white', size=14),
            bgcolor='rgba(10, 15, 38, 0.8)',
            borderpad=8
        ),
        dict(
            x=max(df['ppda']) - x_range*0.15,
            y=min(df['passes']) + y_range*0.15,
            text="<b>Directo con presión baja</b>",
            showarrow=False,
            xanchor='right',
            yanchor='top',
            font=dict(color='white', size=14),
            bgcolor='rgba(10, 15, 38, 0.8)',
            borderpad=8
        ),
        dict(
            x=max(df['ppda']) - x_range*0.15,
            y=max(df['passes']) - y_range*0.15,
            text="<b>Combinativo con presión baja</b>",
            showarrow=False,
            xanchor='right',
            yanchor='bottom',
            font=dict(color='white', size=14),
            bgcolor='rgba(10, 15, 38, 0.8)',
            borderpad=8
        )
    ]
    
    # Aggiorna il layout
    fig.update_layout(
        plot_bgcolor='#0A0F26',
        paper_bgcolor='#0A0F26',
        font=dict(color='white'),
        xaxis=dict(
            title='PPDA',
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            autorange="reversed"
        ),
        yaxis=dict(
            title='Passes per Possession',
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a'
        ),
        showlegend=False,
        height=600,
        width=800,
        margin=dict(l=50, r=50, t=50, b=50),
        annotations=quadrant_annotations
    )
    
    # Aggiorna i range degli assi
    fig.update_xaxes(range=[max(df['ppda']) + x_range*0.1, min(df['ppda']) - x_range*0.1])
    fig.update_yaxes(range=[min(df['passes']) - y_range*0.1, max(df['passes']) + y_range*0.1])
    
    # Aggiorna i range degli assi
    fig.update_xaxes(range=[max(df['ppda']) + x_range*0.1, min(df['ppda']) - x_range*0.1])
    fig.update_yaxes(range=[min(df['passes']) - y_range*0.1, max(df['passes']) + y_range*0.1])

    # Definisci la soglia per la zona mista
    ppda_threshold = x_range * 0.1
    passes_threshold = y_range * 0.1
    
    # Classifica le squadre nei quadranti
    def classify_style(row):
        if (abs(row['ppda'] - mean_ppda) < ppda_threshold and 
            abs(row['passes'] - mean_passes) < passes_threshold):
            return 'Misto'
        elif row['ppda'] < mean_ppda and row['passes'] > mean_passes:
            return 'Combinativo con presión alta'
        elif row['ppda'] < mean_ppda and row['passes'] < mean_passes:
            return 'Directo con presión alta'
        elif row['ppda'] > mean_ppda and row['passes'] > mean_passes:
            return 'Combinativo con presión baja'
        else:
            return 'Directo con presión baja'
    
    df['quadrant'] = df.apply(classify_style, axis=1)
    
    # Crea la legenda raggruppata per quadrante
    quadrant_teams = {
        'Misto': [],
        'Combinativo con presión alta': [],
        'Directo con presión alta': [],
        'Combinativo con presión baja': [],
        'Directo con presión baja': []
    }
    
    for _, row in df.iterrows():
        quadrant_teams[row['quadrant']].append(row['team'])
    
    # Aggiungi il rettangolo semi-trasparente per la zona mista
    fig.add_shape(
        type="rect",
        x0=mean_ppda - ppda_threshold,
        y0=mean_passes - passes_threshold,
        x1=mean_ppda + ppda_threshold,
        y1=mean_passes + passes_threshold,
        line=dict(color="white", width=1, dash="dot"),
        fillcolor="rgba(255, 255, 255, 0.1)",
        layer="below"
    )
    
    # Crea il contenuto della legenda
    legend_content = []
    
    # Aggiungi spiegazione della scala di colori
    legend_content.extend([
        html.H5("Escala de Colores", 
               style={
                   'color': '#0082F3',
                   'marginTop': '20px',
                   'fontSize': '16px',
                   'fontWeight': 'bold'
               }),
        html.Div([
            html.Span("🟢 Verde: ", style={'color': '#00FF00', 'fontWeight': 'bold'}),
            html.Span("Alto recupero de balones", style={'color': 'white'}),
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("🟡 Amarillo: ", style={'color': '#FFFF00', 'fontWeight': 'bold'}),
            html.Span("Medio recupero de balones", style={'color': 'white'}),
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("🔴 Rojo: ", style={'color': '#FF0000', 'fontWeight': 'bold'}),
            html.Span("Bajo recupero de balones", style={'color': 'white'}),
        ], style={'marginBottom': '15px'})
    ])
    
    for quadrant, teams in quadrant_teams.items():
        if teams:
            legend_content.extend([
                html.H5(quadrant, 
                       style={
                           'color': '#0082F3',
                           'marginTop': '20px',
                           'fontSize': '16px',
                           'fontWeight': 'bold'
                       }),
                html.Div([
                    html.Span([
                        dcc.Link(
                            team,
                            href=f"/team/{urllib.parse.quote(selected_league)}/{urllib.parse.quote(get_canonical_team_name(team))}",
                            style={
                                'color': 'white',
                                'textDecoration': 'none',
                                'cursor': 'pointer'
                            }
                        ),
                        ", " if i < len(teams) - 1 else ""
                    ]) for i, team in enumerate(sorted(teams))
                ], style={
                    'color': 'white',
                    'fontSize': '14px',
                    'marginBottom': '15px'
                })
            ])
    
    return fig, legend_content