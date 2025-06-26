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
dash.register_page(__name__, path='/estilo-de-juego-top5')

# Costanti
WYSCOUT_DATA_PATH = "datos/wyscout"

# Lista delle top 5 leghe
TOP_5_LEAGUES = {
    'Serie A': {'wyscout': 'Serie A'},
    'Premier League': {'wyscout': 'Premier League'},
    'La Liga': {'wyscout': 'La Liga'},
    'Bundesliga': {'wyscout': 'Bundesliga'},
    'Ligue 1': {'wyscout': 'Ligue 1'}
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

def process_team_stats(file_path, league):
    # Riutilizziamo la stessa funzione dal file estilo_de_juego.py
    df = pd.read_excel(file_path)
    
    team_name = os.path.basename(file_path).split('.')[0].replace("Team Stats ", "")
    
    # Gestisci i casi speciali delle squadre con nomi diversi
    if team_name == "Inter":
        team_rows = df[df.iloc[:, 4] == "Internazionale"]
    elif team_name == "Manchester Utd":
        team_rows = df[df.iloc[:, 4] == "Manchester United"]
    elif team_name == "Newcastle Utd":
        team_rows = df[df.iloc[:, 4] == "Newcastle United"]
    elif team_name == "Nott'ham Forest":
        team_rows = df[df.iloc[:, 4] == "Nottingham Forest"]
    elif team_name == "Tottenham":
        team_rows = df[df.iloc[:, 4].str.contains("Tottenham", case=False, na=False)]
    elif team_name == "West Ham":
        team_rows = df[df.iloc[:, 4].str.contains("West Ham", case=False, na=False)]
    elif team_name == "Wolves":
        team_rows = df[df.iloc[:, 4].str.contains("Wolverhampton", case=False, na=False)]
    else:
        team_rows = df[df.iloc[:, 4] == team_name]

    if team_rows.empty:
        team_rows = df[df.iloc[:, 4].str.contains(team_name, case=False, na=False)]
    
    # PPDA è nella colonna 108 (indice 108)
    ppda_values = team_rows.iloc[:, 108].dropna()
    # Promedio pases è nella colonna 104 (indice 104)
    passes_values = team_rows.iloc[:, 104].dropna()
    
    if ppda_values.empty or passes_values.empty:
        raise ValueError(f"Valori mancanti per {team_name}")
    
    # Calcola le medie
    ppda = ppda_values.mean()
    passes_per_possession = passes_values.mean()
    
    return ppda, passes_per_possession, team_name

# Layout
layout = dbc.Container([
    html.H1("Estilo de Juego - Top 5 Ligas", className="text-center my-4", style={"color": "#0082F3"}),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='season-dropdown-top5',
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
        ], width=6, className="mx-auto")
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='playing-style-scatter-top5',
                config={'displayModeBar': False},
                style={'height': '800px'}
            )
        ], width=9),
        dbc.Col([
            html.Div(id='quadrant-legend-top5', className="mt-4")
        ], width=3)
    ])
], fluid=True, style={"backgroundColor": "#0A0F26", "color": "#FFFFFF", "padding": "20px"})

@callback(
    [Output('playing-style-scatter-top5', 'figure'),
     Output('quadrant-legend-top5', 'children')],
    [Input('season-dropdown-top5', 'value')]
)
def update_scatter_top5(selected_season):
    team_data = []

    # Mappa per far corrispondere il nome della lega dal dropdown a quello nel file Excel
    league_to_competition = {
        'Serie A': 'Italy. Serie A',
        'Premier League': 'England. Premier League',
        'La Liga': 'Spain. La Liga',
        'Bundesliga': 'Germany. Bundesliga',
        'Ligue 1': 'France. Ligue 1'
    }

    for league, league_info in TOP_5_LEAGUES.items():
        wyscout_league_name = league_info['wyscout']
        competition_name = league_to_competition.get(league)
        
        if selected_season == "2023-2024":
            base_path = os.path.join(WYSCOUT_DATA_PATH, selected_season, selected_season, f"{wyscout_league_name.upper()}_{selected_season}")
            if not os.path.exists(base_path):
                base_path = os.path.join(WYSCOUT_DATA_PATH, selected_season, wyscout_league_name)

            if not os.path.exists(base_path): continue

            for team_folder in os.listdir(base_path):
                folder_path = os.path.join(base_path, team_folder)
                if os.path.isdir(folder_path):
                    try:
                        indices_file = os.path.join(folder_path, "Indices.xlsx")
                        general_file = os.path.join(folder_path, "General.xlsx")
                        
                        if os.path.exists(indices_file) and os.path.exists(general_file):
                            df_indices = pd.read_excel(indices_file)
                            df_general = pd.read_excel(general_file)
                            
                            ppda = pd.to_numeric(df_indices.iloc[0, 3], errors='coerce')
                            passes_per_possession = pd.to_numeric(df_indices.iloc[0, 2], errors='coerce')
                            recoveries = pd.to_numeric(df_general.iloc[:, 22], errors='coerce').mean()
                            
                            canonical_name = get_canonical_team_name(team_folder.replace("_", " "))
                            
                            team_data.append({
                                'team': canonical_name, 'league': league,
                                'ppda': ppda, 'passes': passes_per_possession, 'recoveries': recoveries
                            })
                    except Exception as e:
                        print(f"Error processing {folder_path} for 2023-24: {e}")
                        continue
        else:
            # Logica aggiornata per 2024-2025
            stats_path = os.path.join(WYSCOUT_DATA_PATH, selected_season, wyscout_league_name)
            if not os.path.exists(stats_path): continue

            for team_file in glob.glob(os.path.join(stats_path, "*.xlsx")):
                try:
                    df = pd.read_excel(team_file)
                    team_name_from_file = os.path.basename(team_file).split('.')[0].replace("Team Stats ", "")
                    
                    # Applica la mappatura per trovare il nome corretto da cercare nella colonna 'Equipo'
                    league_specific_mapping = TEAM_NAME_MAPPING.get(league, {})
                    equipo_name_to_search = league_specific_mapping.get(team_name_from_file, team_name_from_file)
                    
                    canonical_name_to_return = get_canonical_team_name(team_name_from_file)
                    canonical_name_to_search = get_canonical_team_name(equipo_name_to_search)

                    df.columns = [str(c).strip() for c in df.columns]

                    # Filtra per squadra usando il nome mappato
                    team_rows = df[df.iloc[:, 4].apply(lambda x: get_canonical_team_name(str(x)) == canonical_name_to_search)]
                    
                    # Filtra ulteriormente per competizione
                    if competition_name and not team_rows.empty:
                        team_rows = team_rows[team_rows.iloc[:, 2] == competition_name]
                    
                    if not team_rows.empty:
                        ppda = pd.to_numeric(team_rows.iloc[:, 108], errors='coerce').mean()
                        passes = pd.to_numeric(team_rows.iloc[:, 104], errors='coerce').mean()
                        recoveries = pd.to_numeric(team_rows.iloc[:, 22], errors='coerce').mean()
                        
                        team_data.append({
                            'team': canonical_name_to_return, 'league': league,
                            'ppda': ppda, 'passes': passes, 'recoveries': recoveries
                        })
                except Exception as e:
                    print(f"Error processing {team_file} for 2024-25: {e}")
                    continue

    if not team_data:
        return go.Figure(layout={'plot_bgcolor': '#0A0F26', 'paper_bgcolor': '#0A0F26'}), [html.P("No hay datos disponibles para la temporada seleccionada.", style={'color': 'white'})]
        
    df = pd.DataFrame(team_data).dropna(subset=['ppda', 'passes', 'recoveries'])
    
    if df.empty:
        return go.Figure(layout={'plot_bgcolor': '#0A0F26', 'paper_bgcolor': '#0A0F26'}), [html.P("Los datos procesados están vacíos o no contienen valores válidos.", style={'color': 'white'})]

    mean_ppda = df['ppda'].mean()
    mean_passes = df['passes'].mean()
    
    fig = go.Figure()
    
    fig.add_hline(y=mean_passes, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_vline(x=mean_ppda, line_dash="dash", line_color="white", opacity=0.5)

    fig.add_trace(go.Scatter(
        x=df['ppda'],
        y=df['passes'],
        mode='markers+text',
        text=[f"{row['team']}" for _, row in df.iterrows()],
        textposition="top center",
        marker=dict(
            size=12,
            color=df['recoveries'],
            colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
            showscale=True,
            colorbar=dict(
                title="Recuperos Altos",
                titleside="right",
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{text}</b><br>Liga: %{customdata}<br>PPDA: %{x:.2f}<br>Pases por Posesión: %{y:.2f}<br>Recuperos Altos: %{marker.color:.2f}<extra></extra>',
        customdata=df['league'],
        textfont=dict(color='white', size=10)
    ))
    
    x_range = df['ppda'].max() - df['ppda'].min()
    y_range = df['passes'].max() - df['passes'].min()
    
    quadrant_annotations = [
        dict(x=df['ppda'].min() + x_range*0.15, y=df['passes'].max() - y_range*0.15, text="<b>Combinativo con presión alta</b>", showarrow=False, xanchor='left', yanchor='bottom', font=dict(color='white', size=14), bgcolor='rgba(10, 15, 38, 0.8)', borderpad=8),
        dict(x=df['ppda'].min() + x_range*0.15, y=df['passes'].min() + y_range*0.15, text="<b>Directo con presión alta</b>", showarrow=False, xanchor='left', yanchor='top', font=dict(color='white', size=14), bgcolor='rgba(10, 15, 38, 0.8)', borderpad=8),
        dict(x=df['ppda'].max() - x_range*0.15, y=df['passes'].min() + y_range*0.15, text="<b>Directo con presión baja</b>", showarrow=False, xanchor='right', yanchor='top', font=dict(color='white', size=14), bgcolor='rgba(10, 15, 38, 0.8)', borderpad=8),
        dict(x=df['ppda'].max() - x_range*0.15, y=df['passes'].max() - y_range*0.15, text="<b>Combinativo con presión baja</b>", showarrow=False, xanchor='right', yanchor='bottom', font=dict(color='white', size=14), bgcolor='rgba(10, 15, 38, 0.8)', borderpad=8)
    ]
    
    fig.update_layout(
        plot_bgcolor='#0A0F26', paper_bgcolor='#0A0F26', font=dict(color='white'),
        xaxis=dict(title='PPDA', gridcolor='#1a1a1a', zerolinecolor='#1a1a1a', autorange="reversed"),
        yaxis=dict(title='Pases por Posesión', gridcolor='#1a1a1a', zerolinecolor='#1a1a1a'),
        showlegend=False, height=800, width=1200, margin=dict(l=50, r=50, t=50, b=50),
        annotations=quadrant_annotations
    )
    
    fig.update_xaxes(range=[df['ppda'].max() + x_range*0.1, df['ppda'].min() - x_range*0.1])
    fig.update_yaxes(range=[df['passes'].min() - y_range*0.1, df['passes'].max() + y_range*0.1])
    
    ppda_threshold = x_range * 0.1
    passes_threshold = y_range * 0.1
    
    def classify_style(row):
        if (abs(row['ppda'] - mean_ppda) < ppda_threshold and abs(row['passes'] - mean_passes) < passes_threshold): return 'Mixto'
        elif row['ppda'] < mean_ppda and row['passes'] > mean_passes: return 'Combinativo con presión alta'
        elif row['ppda'] < mean_ppda and row['passes'] < mean_passes: return 'Directo con presión alta'
        elif row['ppda'] > mean_ppda and row['passes'] > mean_passes: return 'Combinativo con presión baja'
        else: return 'Directo con presión baja'
    
    df['quadrant'] = df.apply(classify_style, axis=1)
    
    quadrant_teams = {q: [] for q in ['Mixto', 'Combinativo con presión alta', 'Directo con presión alta', 'Combinativo con presión baja', 'Directo con presión baja']}
    for _, row in df.iterrows():
        quadrant_teams[row['quadrant']].append(row)
    
    fig.add_shape(type="rect", x0=mean_ppda - ppda_threshold, y0=mean_passes - passes_threshold, x1=mean_ppda + ppda_threshold, y1=mean_passes + passes_threshold, line=dict(color="white", width=1, dash="dot"), fillcolor="rgba(255, 255, 255, 0.1)", layer="below")
    
    legend_content = [
        html.H5("Escala de Colores", style={'color': '#0082F3', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': 'bold'}),
        html.Div([html.Span("🟢 Verde: ", style={'color': '#00FF00', 'fontWeight': 'bold'}), "Alto recupero de balones"], style={'color': 'white'}),
        html.Div([html.Span("🟡 Amarillo: ", style={'color': '#FFFF00', 'fontWeight': 'bold'}), "Medio recupero de balones"], style={'color': 'white'}),
        html.Div([html.Span("🔴 Rojo: ", style={'color': '#FF0000', 'fontWeight': 'bold'}), "Bajo recupero de balones"], style={'color': 'white', 'marginBottom': '15px'})
    ]
    
    for quadrant, teams_in_quadrant in quadrant_teams.items():
        if teams_in_quadrant:
            legend_content.extend([
                html.H5(quadrant, style={'color': '#0082F3', 'marginTop': '20px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                html.Div([
                    html.Span([
                        dcc.Link(
                            team_row['team'],
                            href=f"/team/{urllib.parse.quote(team_row['league'])}/{urllib.parse.quote(team_row['team'])}",
                            style={'color': 'white', 'textDecoration': 'none', 'cursor': 'pointer'}
                        ),
                        f" ({team_row['league'][0]})",
                        ", " if i < len(teams_in_quadrant) - 1 else ""
                    ]) for i, team_row in enumerate(sorted(teams_in_quadrant, key=lambda x: x['team']))
                ], style={'color': 'white', 'fontSize': '12px', 'marginBottom': '15px'})
            ])
    
    return fig, legend_content 