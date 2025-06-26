import dash
from dash import html, dcc, callback, Input, Output, State, ALL, ctx, callback_context
import dash_bootstrap_components as dbc
import os
import urllib.parse
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from utils.team_utils import get_team_from_pathname, get_league_from_pathname, get_team_logo, TEAM_LEAGUE_MAPPING, TEAM_NAME_MAPPING, get_canonical_team_name
from dash.exceptions import PreventUpdate

# Registrazione della pagina
dash.register_page(__name__, path_template="/team/<league>/<team_name>")

# Percorsi
ASSETS_FOLDER = os.path.normpath("assets/2024-25")
FBREF_FOLDER = os.path.normpath("datos/fbref")
WYSCOUT_FOLDER = os.path.normpath("datos/wyscout/2024-2025")

def get_team_age_stats(team_name, league):
    """Calcola le statistiche dell'età della squadra"""
    try:
        # Lista di possibili nomi della cartella
        possible_folder_names = [
            team_name,  # Nome originale
            team_name.replace(" ", "_"),  # Nome con underscore
        ]
        
        # Aggiungi il nome mappato
        mapped_name = TEAM_NAME_MAPPING.get(team_name, "")
        if isinstance(mapped_name, list):
            possible_folder_names.extend(mapped_name)  # Se è una lista, estendi
        elif mapped_name:
            possible_folder_names.append(mapped_name)  # Se è una stringa, aggiungi
        
        # Se il nome è già nella forma mappata, aggiungi anche il nome originale
        if team_name in TEAM_NAME_MAPPING.values():
            original_name = {v: k for k, v in TEAM_NAME_MAPPING.items() if not isinstance(v, list)}.get(team_name)
            if original_name:
                possible_folder_names.extend([
                    original_name,
                    original_name.replace(" ", "_")
                ])
        
        # Rimuovi nomi vuoti e duplicati
        possible_folder_names = list(set(filter(None, possible_folder_names)))
        
        # Prova tutti i possibili percorsi
        df = None
        used_path = None
        for folder_name in possible_folder_names:
            try:
                file_path = os.path.join(FBREF_FOLDER, league, folder_name, "estandar_unificato.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    used_path = file_path
                    break
            except Exception:
                continue
        
        if df is None:
            print(f"Nessun file trovato per {team_name} ({league}). Percorsi tentati:")
            for folder_name in possible_folder_names:
                print(f"- {os.path.join(FBREF_FOLDER, league, folder_name, 'estandar_unificato.csv')}")
            return None
            
        # Filtra i giocatori con nome non nullo
        df = df[df['Name'].notna()]
        
        # Estrai solo gli anni dall'età (prima del trattino)
        df['Edad_anni'] = df['Edad'].apply(lambda x: int(str(x).split('-')[0]))
        
        # Calcola statistiche età usando solo gli anni
        ages = df['Edad_anni']
        avg_age = round(ages.mean(), 1)
        min_age = int(ages.min())
        max_age = int(ages.max())
        
        return {
            'average': avg_age,
            'min': min_age,
            'max': max_age
        }
    except Exception as e:
        print(f"Errore nel calcolo dell'età media per {team_name} ({league}): {e}")
        return None

def create_age_indicator(age_stats):
    """Crea l'indicatore dell'età media"""
    if not age_stats:
        return go.Figure()
    
    fig = go.Figure()
    
    # Aggiungi il valore principale
    fig.add_trace(go.Indicator(
        mode="number",
        value=age_stats['average'],
        number={'font': {'size': 32, 'color': 'white'}},
        title={
            'text': "PROMEDIO EDAD",
            'font': {'size': 12, 'color': 'white'},
            'align': 'center'
        },
        domain={'x': [0, 1], 'y': [0.6, 1]}
    ))
    
    # Crea una scala orizzontale per il range di età
    fig.add_trace(go.Scatter(
        x=[age_stats['min'], age_stats['max']],
        y=[0, 0],
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
        showlegend=False
    ))
    
    # Aggiungi il marker per l'età media
    fig.add_trace(go.Scatter(
        x=[age_stats['average']],
        y=[0],
        mode='markers',
        marker=dict(
            color='white',
            size=8,
            symbol='circle'
        ),
        showlegend=False
    ))
    
    # Aggiungi le etichette min e max
    fig.add_annotation(
        x=age_stats['min'],
        y=0,
        text=str(age_stats['min']),
        showarrow=False,
        yshift=-15,
        font=dict(color='white', size=10)
    )
    fig.add_annotation(
        x=age_stats['max'],
        y=0,
        text=str(age_stats['max']),
        showarrow=False,
        yshift=-15,
        font=dict(color='white', size=10)
    )
    
    # Aggiorna il layout
    fig.update_layout(
        paper_bgcolor="#0A0F26",
        plot_bgcolor="#0A0F26",
        margin=dict(t=40, b=20, l=20, r=20),
        height=150,
        width=150,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[
                age_stats['min'] - 1,
                age_stats['max'] + 1
            ]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[-0.5, 0.5]
        )
    )
    
    return fig

def create_results_pie(team_name, league):
    """Crea il grafico a torta per vittorie, pareggi e sconfitte"""
    try:
        # Usa la mappa di corrispondenza se disponibile
        folder_name = TEAM_NAME_MAPPING.get(team_name, team_name.replace(" ", "_"))
        
        # Leggi il file di classificazione
        file_path = os.path.join(FBREF_FOLDER, league, "clasificacion.csv")
        df = pd.read_csv(file_path)
        
        # Trova i dati della squadra
        team_stats = df[df['Equipo'].str.strip() == folder_name]
        if team_stats.empty:
            # Prova con il nome originale
            team_stats = df[df['Equipo'].str.strip() == team_name]
            if team_stats.empty:
                print(f"No data found for team {team_name}")
                return go.Figure()
        
        team_stats = team_stats.iloc[0]
        
        # Crea il grafico a torta
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            values=[team_stats['PG'], team_stats['PE'], team_stats['PP']],
            labels=['Ganados', 'Empates', 'Derrotas'],
            hole=0.4,
            marker=dict(colors=['#2ecc71', '#f1c40f', '#e74c3c']),
            textinfo='value',
            textfont=dict(size=14, color='white'),
            hoverinfo='label+value',
            showlegend=True,
            textposition='inside'
        ))
        
        # Aggiorna il layout
        fig.update_layout(
            showlegend=True,
            legend=dict(
                x=1.2,
                y=0.5,
                xanchor='left',
                yanchor='middle',
                orientation='v',
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor="#0A0F26",
            plot_bgcolor="#0A0F26",
            font=dict(color="white"),
            margin=dict(t=10, b=10, l=10, r=80),  # Aumentato il margine destro per la legenda
            height=150,
            width=200  # Aumentato per accomodare la legenda
        )
        
        return fig
    except Exception as e:
        print(f"Errore nella creazione del grafico: {e}")
        return go.Figure()

def create_goals_chart(team_name, league):
    """Crea il grafico a torta per i gol fatti e subiti"""
    try:
        # Leggi il file di classificazione
        file_path = os.path.join(FBREF_FOLDER, league, "clasificacion.csv")
        df = pd.read_csv(file_path)
        
        # Trova i dati della squadra
        team_stats = df[df['Equipo'] == team_name].iloc[0]
        
        # Crea il grafico a torta
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            values=[team_stats['GF'], team_stats['GC']],
            labels=['Goles a Favor', 'Goles en Contra'],
            hole=0.7,
            direction='clockwise',
            rotation=90,
            showlegend=True,
            textinfo='value',
            textfont=dict(size=14, color='white'),
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            textposition='inside'
        ))
        
        # Aggiungi la differenza reti al centro
        fig.add_annotation(
            text=f"DG<br>{team_stats['DG']:+d}",
            x=0.5, y=0.5,
            font=dict(size=16, color='white'),
            showarrow=False
        )
        
        # Aggiorna il layout
        fig.update_layout(
            showlegend=True,
            legend=dict(
                x=1.2,
                y=0.5,
                xanchor='left',
                yanchor='middle',
                orientation='v',
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor="#0A0F26",
            plot_bgcolor="#0A0F26",
            font=dict(color="white"),
            margin=dict(t=10, b=10, l=10, r=80),  # Aumentato il margine destro per la legenda
            height=150,
            width=200  # Aumentato per accomodare la legenda
        )
        
        return fig
    except Exception as e:
        print(f"Errore nella creazione del grafico: {e}")
        return go.Figure()

def get_buildup_stats(team_name, league):
    """Recupera e calcola le statistiche di build-up per una squadra"""
    try:
        # Leggi il file di build-up per la lega
        file_path = os.path.join(WYSCOUT_FOLDER, league, f"build-up_{league}.csv")
        if not os.path.exists(file_path):
            print(f"File build-up non trovato per {league}: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        
        # KPI da utilizzare
        kpi_list = [
            'Promedio pases por posesión',
            'Lanzamiento largo %',
            'Balones perdidos bajos',
            'OPPDA',
            'Posesión de balón %'
        ]
        
        # Ottieni il nome standardizzato dalla mappa
        standardized_name = TEAM_NAME_MAPPING.get(team_name, team_name)
        
        # Cerca prima con il nome standardizzato
        team_data = df[df['Equipo'].str.strip() == standardized_name]
        
        # Se non trova, prova con il nome originale
        if team_data.empty:
            team_data = df[df['Equipo'].str.strip() == team_name]
        
        # Se ancora non trova, prova con tutte le chiavi che mappano allo stesso nome standardizzato
        if team_data.empty:
            possible_names = [k for k, v in TEAM_NAME_MAPPING.items() if v == standardized_name]
            for name in possible_names:
                temp_data = df[df['Equipo'].str.strip() == name]
                if not temp_data.empty:
                    team_data = temp_data
                    break
        
        if team_data.empty:
            print(f"Dati build-up non trovati per {team_name} ({standardized_name}) in {league}")
            return None
            
        # Calcola statistiche per ogni KPI
        stats = {}
        for col in kpi_list:
            try:
                value = team_data[col].iloc[0]
                
                # Se il valore è NaN o vuoto, usa 0
                if pd.isna(value) or value == '':
                    stats[col] = {
                        'scaled_value': 0,
                        'normalized': 0,
                        'raw_value': 0,
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                    continue
                
                # Calcola min e max escludendo i valori NaN
                min_val = df[col].dropna().min()
                max_val = df[col].dropna().max()
                
                # Calcola il rank percentile (sempre dal più grande al più piccolo)
                percentile_rank = df[col].rank(pct=True)
                team_percentile = percentile_rank[team_data.index[0]]
                
                stats[col] = {
                    'scaled_value': team_percentile,
                    'normalized': team_percentile / 100,
                    'raw_value': value,
                    'min': min_val,
                    'max': max_val
                }
            except Exception as e:
                print(f"Errore nel calcolo delle statistiche per {col} ({team_name}): {e}")
                stats[col] = {
                    'scaled_value': 0,
                    'normalized': 0,
                    'raw_value': 0,
                    'min': 0,
                    'max': 0
                }
            
        return stats
    except Exception as e:
        print(f"Errore nel calcolo delle statistiche di build-up per {team_name} ({league}): {e}")
        return None

def create_buildup_radar(team_name, league, comparison_team=None):
    """Crea il radar chart per le statistiche di build-up"""
    stats = get_buildup_stats(team_name, league)
    comparison_stats = get_buildup_stats(comparison_team, league) if comparison_team else None
    
    if not stats:
        return go.Figure()
    
    # Mappa per la traduzione e formattazione delle etichette
    label_mapping = {
        'Promedio pases por posesión': ('Promedio pases por posesión', ''),
        'Lanzamiento largo %': ('Lanzamiento largo %', ''),
        'Balones perdidos bajos': ('Balones perdidos bajos', ''),
        'OPPDA': ('OPPDA', ''),
        'Posesión de balón %': ('Posesión de balón %', '')
    }
    
    # Ordine specifico delle metriche per il radar (in senso orario partendo dall'alto)
    ordered_metrics = [
        'OPPDA',                      # In alto
        'Balones perdidos bajos',     # In alto a destra
        'Lanzamiento largo %',        # A destra
        'Promedio pases por posesión', # In basso
        'Posesión de balón %'         # A sinistra
    ]
    
    # Prepara i dati per il radar chart
    categories = ordered_metrics
    values = []
    raw_values = []
    
    # Leggi il file per ottenere i valori min/max
    file_path = os.path.join(WYSCOUT_FOLDER, league, f"build-up_{league}.csv")
    df = pd.read_csv(file_path)
    
    # Calcola i valori per la squadra principale
    for category in categories:
        value = df[df['Equipo'].str.contains(team_name)][category].iloc[0]
        min_val = df[category].min()
        max_val = df[category].max()
        
        # Normalizza sempre dal più grande al più piccolo
        if pd.isna(value):
            normalized = 0.1  # Valore minimo per NaN
            scaled = 10  # Valore minimo scalato
        else:
            normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            # Assicura un valore minimo visibile
            normalized = max(normalized, 0.1)
            scaled = normalized * 100
            
        values.append(scaled)
        raw_values.append(value)
    
    # Calcola i valori per la squadra di confronto
    comparison_values = []
    comparison_raw_values = []
    if comparison_team:
        for category in categories:
            value = df[df['Equipo'].str.contains(comparison_team)][category].iloc[0]
            min_val = df[category].min()
            max_val = df[category].max()
            
            if pd.isna(value):
                normalized = 0.1
                scaled = 10
            else:
                normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                normalized = max(normalized, 0.1)
                scaled = normalized * 100
            
            comparison_values.append(scaled)
            comparison_raw_values.append(value)
    
    # Calcola gli angoli per i settori
    n_metrics = len(categories)
    sector_angle = 360 / n_metrics
    
    fig = go.Figure()
    
    # Aggiungi la circonferenza esterna
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatterpolar(
        r=[100]*100,
        theta=np.degrees(theta),
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Disegna il radar della squadra principale
    angles = [(i * sector_angle) % 360 for i in range(len(categories))]
    
    # Aggiungi i settori della squadra principale
    for i, (angle, value, raw_value) in enumerate(zip(angles, values, raw_values)):
        opacity = max(value/100, 0.2)
        color = f'rgba(0, 149, 255, {opacity})'
        
        # Crea i punti per formare il settore
        theta_sector = np.linspace(angle - sector_angle*0.4, angle + sector_angle*0.4, 50)
        r_sector = np.array([value] * 50)
        
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([[0], r_sector, [0]]),
            theta=np.concatenate([[angle], theta_sector, [angle]]),
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=0),
            name=f"{team_name} - {categories[i]}",
            customdata=[[raw_value]]*52,
            hovertemplate=f"{team_name}<br>{categories[i]}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
        ))
    
    # Se c'è una squadra di confronto, disegna i suoi settori sopra
    if comparison_team and comparison_values:
        for i, (angle, value, raw_value) in enumerate(zip(angles, comparison_values, comparison_raw_values)):
            # Crea i punti per formare il settore
            theta_sector = np.linspace(angle - sector_angle*0.4, angle + sector_angle*0.4, 50)
            r_sector = np.array([value] * 50)
            
            fig.add_trace(go.Scatterpolar(
                r=np.concatenate([[0], r_sector, [0]]),
                theta=np.concatenate([[angle], theta_sector, [angle]]),
                mode='lines',
                fill='toself',
                fillcolor='rgba(255, 69, 0, 0.6)',
                line=dict(color='rgba(255, 69, 0, 0.6)', width=0),
                name=f"{comparison_team} - {categories[i]}",
                customdata=[[raw_value]]*52,
                hovertemplate=f"{comparison_team}<br>{categories[i]}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
            ))
    
    # Prepara le etichette e le posizioni angolari
    tick_vals = []
    tick_text = []
    for i, category in enumerate(categories):
        angle = (i * sector_angle) % 360
        main_text, sub_text = label_mapping[category]
        tick_vals.append(angle)
        tick_text.append(main_text if not sub_text else main_text)
    
    # Aggiorna il layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=True,
                gridcolor="rgba(255, 255, 255, 0.1)",
                tickfont=dict(color='white', size=8),
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['', '', '', '', '']
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(color='white', size=8),
                gridcolor="rgba(255, 255, 255, 0.1)",
                rotation=90,
                direction='clockwise'
            ),
            bgcolor='#0A0F26'
        ),
        showlegend=False,
        paper_bgcolor='#0A0F26',
        plot_bgcolor='#0A0F26',
        margin=dict(t=50, b=50, l=50, r=50),
        height=400,
        width=400,
        title=dict(
            text='MÉTRICAS DE CONSTRUCCIÓN<br><span style="font-size: 10px; color: #666">(Valores porcentuales)</span>',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=14, color='white')
        )
    )
    
    return fig

def create_kpi_table(team_name, league):
    """Crea una tabella con le statistiche dettagliate per ogni KPI"""
    try:
        # Leggi il file di build-up per la lega
        file_path = os.path.join(WYSCOUT_FOLDER, league, f"build-up_{league}.csv")
        df = pd.read_csv(file_path)
        
        # Rimuovi eventuali suffissi dai nomi delle squadre (es. "-2")
        df['Equipo'] = df['Equipo'].str.replace(r'-\d+$', '', regex=True)
        
        # Ordine desiderato delle metriche
        metrics_order = [
            'Promedio pases por posesión',
            'Lanzamiento largo %',
            'Balones perdidos bajos',
            'OPPDA',
            'Posesión de balón %'
        ]
        
        # Crea la tabella
        table_header = [
            html.Thead(html.Tr([
                html.Th("KPI", style={'width': '25%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Ranking", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Valor", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("", style={'width': '35%', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Mín / Máx", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Promedio Liga", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'})
            ], style={'backgroundColor': '#1a1a2e'}))
        ]
        
        rows = []
        for metric in metrics_order:
            # Crea una copia del dataframe con solo le colonne necessarie
            df_metric = df[['Equipo', metric]].copy()
            
            # Rimuovi le righe con valori NaN per questo metric
            df_metric = df_metric.dropna()
            
            # Aggiungi una colonna di ranking (sempre dal più grande al più piccolo)
            df_metric['rank'] = df_metric[metric].rank(ascending=False, method='min')
            
            # Ottieni i dati della squadra
            team_data = df_metric[df_metric['Equipo'] == team_name]
            
            if team_data.empty:
                continue
                
            value = team_data[metric].iloc[0]
            ranking = int(team_data['rank'].iloc[0])
            
            # Calcola min, max e media (escludendo i NaN)
            metric_min = df[metric].min()
            metric_max = df[metric].max()
            metric_avg = df[metric].mean()
            
            # Calcola il valore normalizzato per la barra (0-1)
            if pd.isna(value):
                normalized = 0.1  # Valore minimo per NaN
                color = 'rgba(128, 128, 128, 0.5)'  # Grigio per NaN
            else:
                # Normalizza sempre dal più grande al più piccolo
                normalized = (value - metric_min) / (metric_max - metric_min) if metric_max != metric_min else 0.5
                # Assicura un valore minimo visibile
                normalized = max(normalized, 0.1)
                
                # Usa lo stesso colore azzurro con opacità variabile
                opacity = max(normalized, 0.2)  # Minimo 0.2 di opacità
                color = f'rgba(0, 149, 255, {opacity})'  # Azzurro
            
            # Crea la barra colorata con un minimo del 10%
            bar_style = {
                'backgroundColor': color,
                'width': f'{max(normalized * 100, 10)}%',
                'height': '20px',
                'borderRadius': '4px',
                'transition': 'width 0.3s ease'
            }
            
            # Formatta i valori numerici
            value_str = f"{value:.2f}" if not pd.isna(value) else "N/A"
            minmax_str = f"{metric_min:.2f} / {metric_max:.2f}"
            avg_str = f"{metric_avg:.2f}"
            
            # Crea la riga della tabella
            row = html.Tr([
                html.Td(metric, style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"#{ranking}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(value_str, style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(html.Div(style=bar_style), style={'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(minmax_str, style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(avg_str, style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'})
            ], style={'backgroundColor': '#0A0F26'})
            rows.append(row)
        
        table_body = [html.Tbody(rows)]
        
        return html.Div([
            html.H3("MÉTRICAS DE CONSTRUCCIÓN", 
                   style={
                       'textAlign': 'center',
                       'marginBottom': '20px',
                       'color': 'white',
                       'fontSize': '24px',
                       'fontWeight': 'bold'
                   }),
            dbc.Table(
                table_header + table_body,
                bordered=False,
                dark=True,
                hover=True,
                responsive=True,
                striped=False,
                style={
                    'backgroundColor': '#0A0F26',
                    'color': 'white',
                    'width': '100%',
                    'borderCollapse': 'separate',
                    'borderSpacing': '0 1px'
                }
            )
        ], style={
            'backgroundColor': '#0A0F26',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        })
        
    except Exception as e:
        print(f"Errore nella creazione della tabella KPI: {e}")
        return html.Div("Error creating KPI table")

def calculate_similarity(team_name, league):
    """Calcola la similarità tra una squadra e tutte le altre confrontando KPI per KPI e quadrante di stile di gioco"""
    try:
        # Leggi il file di build-up per la lega
        file_path = os.path.join(WYSCOUT_FOLDER, league, f"build-up_{league}.csv")
        df = pd.read_csv(file_path)
        
        # Rimuovi eventuali suffissi dai nomi delle squadre
        df['Equipo'] = df['Equipo'].str.replace(r'-\d+$', '', regex=True)
        
        # Lista delle metriche
        metrics = [
            'Promedio pases por posesión',
            'Lanzamiento largo %',
            'Balones perdidos bajos',
            'OPPDA',
            'Posesión de balón %'
        ]
        
        # Ottieni i dati della squadra target
        target_team_data = df[df['Equipo'] == team_name]
        if target_team_data.empty:
            print(f"No data found for team {team_name}")
            return []
        
        # Calcola le medie per determinare i quadranti
        mean_ppda = df['OPPDA'].mean()
        mean_passes = df['Promedio pases por posesión'].mean()
        
        # Determina il quadrante della squadra target
        target_quadrant = ''
        if target_team_data['OPPDA'].iloc[0] < mean_ppda:
            if target_team_data['Promedio pases por posesión'].iloc[0] > mean_passes:
                target_quadrant = 'Combinativo con blocco alto'
            else:
                target_quadrant = 'Diretto con blocco alto'
        else:
            if target_team_data['Promedio pases por posesión'].iloc[0] > mean_passes:
                target_quadrant = 'Combinativo con blocco basso'
            else:
                target_quadrant = 'Diretto con blocco basso'
        
        # Calcola i percentili per ogni metrica (sempre dal più grande al più piccolo)
        percentiles = {}
        for metric in metrics:
            percentiles[metric] = df[metric].rank(pct=True)
        
        # Calcola la similarità per tutte le squadre
        similarities = []
        for _, row in df.iterrows():
            if row['Equipo'] != team_name:
                # Determina il quadrante della squadra corrente
                current_quadrant = ''
                if row['OPPDA'] < mean_ppda:
                    if row['Promedio pases por posesión'] > mean_passes:
                        current_quadrant = 'Combinativo con blocco alto'
                    else:
                        current_quadrant = 'Diretto con blocco alto'
                else:
                    if row['Promedio pases por posesión'] > mean_passes:
                        current_quadrant = 'Combinativo con blocco basso'
                    else:
                        current_quadrant = 'Diretto con blocco basso'
                
                # Calcola la similarità basata sui percentili
                similarity = 0
                for metric in metrics:
                    target_percentile = percentiles[metric][target_team_data.index[0]]
                    current_percentile = percentiles[metric][row.name]
                    # La similarità è il complemento della differenza tra i percentili
                    similarity += 1 - abs(target_percentile - current_percentile)
                
                # Normalizza la similarità totale
                similarity = similarity / len(metrics)
                
                # Aggiungi bonus per squadre nello stesso quadrante
                if current_quadrant == target_quadrant:
                    similarity += 0.2  # Bonus del 20% per squadre nello stesso quadrante
                    similarity = min(similarity, 1.0)  # Assicura che non superi 1.0
                
                similarities.append({
                    'team': row['Equipo'],
                    'similarity': similarity,
                    'quadrant': current_quadrant
                })
        
        # Ordina per similarità e prendi i primi 4
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        similar_teams = similarities[:4]
        
        return similar_teams
        
    except Exception as e:
        print(f"Errore nel calcolo della similarità: {str(e)}")
        return []

def create_similar_teams_card(team_name, league, card_id_prefix="buildup"):
    """Crea una card con le quattro squadre più simili"""
    similar_teams = calculate_similarity(team_name, league)
    
    if not similar_teams:
        return html.Div("No similar teams found")
    
    return html.Div([
        html.H4("EQUIPOS CON ESTILO DE CONSTRUCCIÓN SIMILAR", 
               style={
                   'textAlign': 'center',
                   'color': 'white',
                   'marginBottom': '10px',
                   'fontSize': '16px',
                   'fontWeight': 'bold'
               }),
        html.Div([
            html.Div([
                html.Button([
                    html.Img(
                        src=get_team_logo(league, team['team']),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'objectFit': 'contain',
                            'marginBottom': '3px'
                        }
                    ) if get_team_logo(league, team['team']) else html.Div(
                        team['team'][:3].upper(),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'backgroundColor': '#1a1a2e',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'borderRadius': '50%',
                            'color': 'white',
                            'fontSize': '10px',
                            'marginBottom': '3px'
                        }
                    ),
                    html.Div(team['team'], style={
                        'color': 'white',
                        'fontSize': '11px',
                        'textAlign': 'center',
                        'marginBottom': '2px'
                    })
                ],
                id={'type': f'{card_id_prefix}-similar-team-btn', 'index': team['team']},
                n_clicks=0,
                style={
                    'background': 'none',
                    'border': 'none',
                    'cursor': 'pointer',
                    'padding': '8px',
                    'backgroundColor': '#1a1a2e',
                    'borderRadius': '8px',
                    'margin': '3px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
                    'width': '80px',
                    'border': '1px solid rgba(255,255,255,0.1)',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                })
            for team in similar_teams
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(2, 1fr)',
                'gap': '5px',
                'justifyContent': 'center'
            })
        ])
    ], style={
        'backgroundColor': '#0A0F26',
        'padding': '15px',
        'borderRadius': '8px',
        'marginRight': '15px',
        'width': '180px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.2)'
    })

def create_offensive_radar(team_name, league, comparison_team=None):
    """Crea il radar chart per le metriche offensive"""
    try:
        # Leggi il file delle metriche offensive
        file_path = os.path.join(WYSCOUT_FOLDER, league, f"file_1_3_{league}.csv")
        df = pd.read_csv(file_path)
        
        # Trova i dati della squadra principale
        team_data = df[df['Equipo'] == team_name].iloc[0]
        
        # Trova i dati della squadra di confronto se presente
        comparison_data = None
        if comparison_team:
            comparison_data = df[df['Equipo'] == comparison_team].iloc[0]
        
        # Metriche da visualizzare
        metrics = {
            'contrataques': 'Contraataques',
            'pases cruzados Z3': 'Centros Z3',
            'pases en profundidad': 'Pases en Profundidad',
            'pases Z3': 'Pases Z3',
            'entradas al area': 'Entradas al Área',
            'cambios de juego': 'Cambios de Juego',
            'carries Z3': 'Conducciones Z3'
        }
        
        # Calcola i valori normalizzati per entrambe le squadre
        values = []
        raw_values = []
        comparison_values = []
        comparison_raw_values = []
        
        for metric_key in metrics.keys():
            # Valori per la squadra principale
            raw_value = team_data[metric_key]
            raw_values.append(raw_value)
            
            # Calcola il valore normalizzato (0-100)
            min_val = df[metric_key].min()
            max_val = df[metric_key].max()
            if max_val == min_val:
                values.append(50)
                if comparison_data is not None:
                    comparison_values.append(50)
                    comparison_raw_values.append(comparison_data[metric_key])
            else:
                val = ((raw_value - min_val) / (max_val - min_val)) * 100
                values.append(val)
                if comparison_data is not None:
                    comp_raw = comparison_data[metric_key]
                    comparison_raw_values.append(comp_raw)
                    comp_val = ((comp_raw - min_val) / (max_val - min_val)) * 100
                    comparison_values.append(comp_val)
        
        fig = go.Figure()
        
        # Aggiungi la circonferenza esterna
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatterpolar(
            r=[100]*100,
            theta=np.degrees(theta),
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Calcola l'angolo per ogni metrica
        n_metrics = len(metrics)
        sector_angle = 360 / n_metrics
        angles = [(i * sector_angle) % 360 for i in range(n_metrics)]
        
        # Disegna i settori della squadra principale
        for i, (metric_key, metric_name, value, raw_value) in enumerate(zip(metrics.keys(), metrics.values(), values, raw_values)):
            angle = angles[i]
            opacity = max(value/100, 0.2)
            color = f'rgba(0, 149, 255, {opacity})'
            
            # Crea i punti per formare il settore
            theta_sector = np.linspace(angle - sector_angle*0.4, angle + sector_angle*0.4, 50)
            r_sector = np.array([value] * 50)
            
            fig.add_trace(go.Scatterpolar(
                r=np.concatenate([[0], r_sector, [0]]),
                theta=np.concatenate([[angle], theta_sector, [angle]]),
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(color=color, width=0),
                name=f"{team_name} - {metric_name}",
                showlegend=False,
                customdata=[[raw_value]]*52,
                hovertemplate=f"{team_name}<br>{metric_name}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
            ))
        
        # Se c'è una squadra di confronto, disegna i suoi settori
        if comparison_data is not None:
            for i, (metric_key, metric_name, value, raw_value) in enumerate(zip(metrics.keys(), metrics.values(), comparison_values, comparison_raw_values)):
                angle = angles[i]
                
                # Crea i punti per formare il settore
                theta_sector = np.linspace(angle - sector_angle*0.4, angle + sector_angle*0.4, 50)
                r_sector = np.array([value] * 50)
                
                fig.add_trace(go.Scatterpolar(
                    r=np.concatenate([[0], r_sector, [0]]),
                    theta=np.concatenate([[angle], theta_sector, [angle]]),
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(255, 69, 0, 0.6)',
                    line=dict(color='rgba(255, 69, 0, 0.6)', width=0),
                    name=f"{comparison_team} - {metric_name}",
                    showlegend=False,
                    customdata=[[raw_value]]*52,
                    hovertemplate=f"{comparison_team}<br>{metric_name}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
                ))
        
        # Aggiorna il layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showline=False,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    tickfont=dict(color='white', size=8),
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['', '', '', '', '']
                ),
                angularaxis=dict(
                    tickmode='array',
                    ticktext=list(metrics.values()),
                    tickvals=angles,
                    tickfont=dict(color='white', size=8),
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    rotation=90,
                    direction='clockwise'
                ),
                bgcolor='#0A0F26'
            ),
            showlegend=False,
            paper_bgcolor='#0A0F26',
            plot_bgcolor='#0A0F26',
            margin=dict(t=50, b=50, l=50, r=50),
            height=400,
            width=400,
            title=dict(
                text='MÉTRICAS OFENSIVAS<br><span style="font-size: 10px; color: #666">(Valores porcentuales)</span>',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=14, color='white')
            )
        )
        
        return fig
    except Exception as e:
        print(f"Errore nella creazione del radar offensivo: {e}")
        return go.Figure()

def create_offensive_table(team_name, league):
    """Crea una tabella con le statistiche dettagliate per le metriche offensive"""
    try:
        # Leggi il file delle metriche offensive
        file_path = os.path.join(WYSCOUT_FOLDER, league, f"file_1_3_{league}.csv")
        df = pd.read_csv(file_path)
        
        # Metriche da visualizzare con traduzione
        metrics = {
            'contrataques': 'Contraataques',
            'pases cruzados Z3': 'Centros Z3',
            'pases en profundidad': 'Pases en Profundidad',
            'pases Z3': 'Pases Z3',
            'entradas al area': 'Entradas al Área',
            'cambios de juego': 'Cambios de Juego',
            'carries Z3': 'Conducciones Z3'
        }
        
        # Crea la tabella
        table_header = [
            html.Thead(html.Tr([
                html.Th("KPI", style={'width': '25%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Ranking", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Valor", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("", style={'width': '35%', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Mín / Máx", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Promedio Liga", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'})
            ], style={'backgroundColor': '#1a1a2e'}))
        ]
        
        rows = []
        for metric, metric_name in metrics.items():
            # Calcola ranking
            df['rank'] = df[metric].rank(ascending=False, method='min')
            team_stats = df[df['Equipo'] == team_name].iloc[0]
            ranking = int(team_stats['rank'])
            
            # Calcola statistiche
            value = team_stats[metric]
            min_val = df[metric].min()
            max_val = df[metric].max()
            avg_val = df[metric].mean()
            
            # Normalizza per la barra colorata
            if max_val == min_val:
                normalized = 0.5
            else:
                normalized = (value - min_val) / (max_val - min_val)
            
            # Colore basato sul valore normalizzato
            opacity = max(normalized, 0.2)
            color = f'rgba(0, 149, 255, {opacity})'
            
            # Crea la barra colorata
            bar_width = normalized * 100
            bar_style = {
                'backgroundColor': color,
                'width': f'{max(bar_width, 10)}%',
                'height': '20px',
                'borderRadius': '4px'
            }
            
            # Aggiungi la riga alla tabella
            row = html.Tr([
                html.Td(metric_name, style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"#{ranking}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{value:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(html.Div(style=bar_style), style={'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{min_val:.2f} / {max_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{avg_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'})
            ], style={'backgroundColor': '#0A0F26'})
            rows.append(row)
        
        return html.Div([
            html.H3("MÉTRICAS OFENSIVAS", 
                   style={
                       'textAlign': 'center',
                       'marginBottom': '20px',
                       'color': 'white',
                       'fontSize': '24px',
                       'fontWeight': 'bold'
                   }),
            dbc.Table(
                table_header + [html.Tbody(rows)],
                bordered=False,
                dark=True,
                hover=True,
                responsive=True,
                striped=False,
                style={
                    'backgroundColor': '#0A0F26',
                    'color': 'white',
                    'width': '100%',
                    'borderCollapse': 'separate',
                    'borderSpacing': '0 1px'
                }
            )
        ], style={
            'backgroundColor': '#0A0F26',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        })
    except Exception as e:
        print(f"Errore nella creazione della tabella offensiva: {e}")
        return html.Div("Error creating offensive metrics table")

def calculate_offensive_similarity(team_name, league):
    """Calcola la similarità offensiva tra una squadra e tutte le altre"""
    try:
        # Leggi il file delle metriche offensive
        file_path = os.path.join(WYSCOUT_FOLDER, league, f"file_1_3_{league}.csv")
        df = pd.read_csv(file_path)
        
        # Metriche da considerare per la similarità
        metrics = [
            'contrataques',
            'pases cruzados Z3',
            'pases en profundidad',
            'pases Z3',
            'entradas al area',
            'cambios de juego',
            'carries Z3'
        ]
        
        # Ottieni i dati della squadra target usando startswith per essere più flessibili
        target_team = df[df['Equipo'].str.startswith(team_name)]
        if target_team.empty:
            print(f"No data found for team {team_name}")
            return []
            
        target_team_data = target_team.iloc[0]
        similarities = []
        
        # Per ogni squadra nel campionato (esclusa la squadra target)
        for idx, row in df[~df['Equipo'].str.startswith(team_name)].iterrows():
            try:
                metric_similarities = []
                
                # Calcola la similarità per ogni metrica
                for metric in metrics:
                    # Ottieni i valori min e max del campionato per questa metrica
                    metric_min = df[metric].min()
                    metric_max = df[metric].max()
                    
                    # Se min e max sono uguali, passa alla prossima metrica
                    if metric_max == metric_min:
                        continue
                    
                    # Normalizza i valori delle due squadre (0-100%)
                    target_normalized = (target_team_data[metric] - metric_min) / (metric_max - metric_min)
                    other_normalized = (row[metric] - metric_min) / (metric_max - metric_min)
                    
                    # Calcola la similarità come complemento della differenza percentuale
                    similarity = 1 - abs(target_normalized - other_normalized)
                    metric_similarities.append(similarity)
                
                # Calcola la similarità media per questa squadra
                if metric_similarities:
                    avg_similarity = sum(metric_similarities) / len(metric_similarities)
                    similarities.append({
                        'team': row['Equipo'],
                        'similarity': avg_similarity
                    })
            except Exception as e:
                print(f"Error calculating similarity for {row['Equipo']}: {e}")
                continue
        
        # Ordina per similarità e prendi le prime 4
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:4]
        
    except Exception as e:
        print(f"Error in calculate_offensive_similarity: {e}")
        return []

def create_offensive_similar_teams_card(team_name, league):
    """Crea una card con le quattro squadre più simili per stile offensivo"""
    similar_teams = calculate_offensive_similarity(team_name, league)
    
    if not similar_teams:
        return html.Div("No similar teams found")
    
    return html.Div([
        html.H4("EQUIPOS CON ESTILO OFENSIVO SIMILAR", 
               style={
                   'textAlign': 'center',
                   'color': 'white',
                   'marginBottom': '10px',
                   'fontSize': '16px',
                   'fontWeight': 'bold'
               }),
        html.Div([
            html.Div([
                html.Button([
                    html.Img(
                        src=get_team_logo(league, team['team']),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'objectFit': 'contain',
                            'marginBottom': '3px'
                        }
                    ) if get_team_logo(league, team['team']) else html.Div(
                        team['team'][:3].upper(),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'backgroundColor': '#1a1a2e',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'borderRadius': '50%',
                            'color': 'white',
                            'fontSize': '10px',
                            'marginBottom': '3px'
                        }
                    ),
                    html.Div(team['team'], style={
                        'color': 'white',
                        'fontSize': '11px',
                        'textAlign': 'center',
                        'marginBottom': '2px'
                    })
                ],
                id={'type': 'offensive-similar-team-btn', 'index': team['team']},
                n_clicks=0,
                style={
                    'background': 'none',
                    'border': 'none',
                    'cursor': 'pointer',
                    'padding': '8px',
                    'backgroundColor': '#1a1a2e',
                    'borderRadius': '8px',
                    'margin': '3px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
                    'width': '80px',
                    'border': '1px solid rgba(255,255,255,0.1)',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                })
            for team in similar_teams
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(2, 1fr)',
                'gap': '5px',
                'justifyContent': 'center'
            })
        ])
    ], style={
        'backgroundColor': '#0A0F26',
        'padding': '15px',
        'borderRadius': '8px',
        'marginRight': '15px',
        'width': '180px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.2)'
    })

def create_atalanta_comparison_card():
    """Crea una card fissa per il confronto con l'Atalanta"""
    return html.Div([
        html.H4("COMPARE WITH", 
               style={
                   'textAlign': 'center',
                   'color': 'white',
                   'marginBottom': '10px',
                   'fontSize': '16px',
                   'fontWeight': 'bold'
               }),
        html.Div([
            html.Button([
                html.Img(
                    src=get_team_logo("Serie A", "Atalanta"),
                    style={
                        'width': '45px',
                        'height': '45px',
                        'objectFit': 'contain',
                        'marginBottom': '3px'
                    }
                ) if get_team_logo("Serie A", "Atalanta") else html.Div(
                    "ATA",
                    style={
                        'width': '45px',
                        'height': '45px',
                        'backgroundColor': '#1a1a2e',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'borderRadius': '50%',
                        'color': 'white',
                        'fontSize': '12px',
                        'marginBottom': '3px'
                    }
                ),
                html.Div("Atalanta", style={
                    'color': 'white',
                    'fontSize': '12px',
                    'textAlign': 'center',
                    'marginBottom': '2px'
                })
            ],
            id='atalanta-comparison-btn',
            n_clicks=0,
            style={
                'background': 'none',
                'border': 'none',
                'cursor': 'pointer',
                'padding': '10px',
                'backgroundColor': '#1a1a2e',
                'borderRadius': '8px',
                'margin': '3px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
                'width': '100px',
                'border': '1px solid rgba(255,255,255,0.1)',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center'
            })
        ], style={
            'display': 'flex',
            'justifyContent': 'center'
        })
    ], style={
        'backgroundColor': '#0A0F26',
        'padding': '15px',
        'borderRadius': '8px',
        'marginRight': '15px',
        'width': '180px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.2)',
        'marginBottom': '20px'  # Aggiungi spazio sotto la card
    })

def get_most_used_formation(team_name, league):
    league = league.replace("_", " ")
    try:
        folder_name = TEAM_NAME_MAPPING.get(team_name, team_name)
        possible_folder_names = [
            team_name,
            team_name.replace(" ", "_"),
            folder_name if isinstance(folder_name, str) else None,
            folder_name.replace(" ", "_") if isinstance(folder_name, str) else None
        ]
        all_mapped_values = []
        for v in TEAM_NAME_MAPPING.values():
            if isinstance(v, list):
                all_mapped_values.extend(v)
            else:
                all_mapped_values.append(v)
        if team_name in all_mapped_values:
            original_name = None
            for k, v in TEAM_NAME_MAPPING.items():
                if (isinstance(v, list) and team_name in v) or v == team_name:
                    original_name = k
                    break
            if original_name:
                possible_folder_names.extend([
                    original_name,
                    original_name.replace(" ", "_")
                ])
        possible_folder_names = list(set(filter(None, possible_folder_names)))
        df = None
        for folder_name in possible_folder_names:
            try:
                file_path = os.path.join(WYSCOUT_FOLDER, league, f"Team Stats {folder_name}.xlsx")
                if os.path.exists(file_path):
                    df = pd.read_excel(file_path)
                    break
            except Exception as e:
                continue
        if df is None:
            return "4-3-3"
        df = df.iloc[2:].reset_index(drop=True)
        df_team = df.iloc[::2].reset_index(drop=True)
        formations = df_team['Seleccionar esquema'].str.extract(r'(\d-\d-\d(?:-\d)*)')[0].value_counts()
        if formations.empty:
            return "4-3-3"
        most_used = formations.index[0]
        return most_used
    except Exception as e:
        return "4-3-3"

def create_formation_pitch(team_name, league):
    """Crea il campo da calcio con la formazione della squadra"""
    formation = get_most_used_formation(team_name, league)
    
    # Definisci le coordinate per ogni formazione (campo orizzontale)
    formation_coords = {
        "4-3-3": {
            "x": [10,  # Portiere
                 25, 25, 25, 25,  # Difensori
                 50, 50, 50,  # Centrocampisti
                 75, 75, 75],  # Attaccanti
            "y": [50,  # Portiere
                 20, 40, 60, 80,  # Difensori
                 30, 50, 70,  # Centrocampisti
                 20, 50, 80]  # Attaccanti
        },
        "4-4-2": {
            "x": [10,  # Portiere
                 25, 25, 25, 25,  # Difensori
                 50, 50, 50, 50,  # Centrocampisti
                 75, 75],  # Attaccanti
            "y": [50,  # Portiere
                 20, 40, 60, 80,  # Difensori
                 20, 40, 60, 80,  # Centrocampisti
                 35, 65]  # Attaccanti
        },
        "3-5-2": {
            "x": [10,  # Portiere
                 25, 25, 25,  # Difensori
                 50, 50, 50, 50, 50,  # Centrocampisti
                 75, 75],  # Attaccanti
            "y": [50,  # Portiere
                 30, 50, 70,  # Difensori
                 20, 35, 50, 65, 80,  # Centrocampisti
                 35, 65]  # Attaccanti
        },
        "3-4-3": {
            "x": [10,  # Portiere
                 25, 25, 25,  # Difensori
                 50, 50, 50, 50,  # Centrocampisti
                 75, 75, 75],  # Attaccanti
            "y": [50,  # Portiere
                 30, 50, 70,  # Difensori
                 20, 40, 60, 80,  # Centrocampisti
                 20, 50, 80]  # Attaccanti
        },
        "3-4-2-1": {
            "x": [10,  # Portiere
                 25, 25, 25,  # Difensori
                 50, 50, 50, 50,  # Centrocampisti
                 65, 65,  # Trequartisti
                 75],  # Attaccante
            "y": [50,  # Portiere
                 30, 50, 70,  # Difensori
                 20, 40, 60, 80,  # Centrocampisti
                 35, 65,  # Trequartisti
                 50]  # Attaccante
        },
        "4-2-3-1": {
            "x": [10,  # Portiere
                 25, 25, 25, 25,  # Difensori
                 45, 45,  # Centrocampisti centrali
                 65, 65, 65,  # Trequartisti
                 75],  # Attaccante
            "y": [50,  # Portiere
                 20, 40, 60, 80,  # Difensori
                 35, 65,  # Centrocampisti centrali
                 20, 50, 80,  # Trequartisti
                 50]  # Attaccante
        },
        "3-4-1-2": {
            "x": [10,  # Portiere
                 25, 25, 25,  # Difensori
                 50, 50, 50, 50,  # Centrocampisti
                 65,  # Trequartista
                 75, 75],  # Attaccanti
            "y": [50,  # Portiere
                 30, 50, 70,  # Difensori
                 20, 40, 60, 80,  # Centrocampisti
                 50,  # Trequartista
                 35, 65]  # Attaccanti
        },
        "4-3-1-2": {
            "x": [10,  # Portiere
                 25, 25, 25, 25,  # Difensori
                 50, 50, 50,  # Centrocampisti
                 65,  # Trequartista
                 75, 75],  # Attaccanti
            "y": [50,  # Portiere
                 20, 40, 60, 80,  # Difensori
                 30, 50, 70,  # Centrocampisti
                 50,  # Trequartista
                 35, 65]  # Attaccanti
        },
        "4-3-2-1": {
            "x": [10,  # Portiere
                 25, 25, 25, 25,  # Difensori
                 50, 50, 50,  # Centrocampisti
                 65, 65,  # Trequartisti
                 75],  # Attaccante
            "y": [50,  # Portiere
                 20, 40, 60, 80,  # Difensori
                 30, 50, 70,  # Centrocampisti
                 35, 65,  # Trequartisti
                 50]  # Attaccante
        },
        "4-4-1-1": {
            "x": [10,  # Portiere
                 25, 25, 25, 25,  # Difensori
                 50, 50, 50, 50,  # Centrocampisti
                 65,  # Trequartista
                 75],  # Attaccante
            "y": [50,  # Portiere
                 20, 40, 60, 80,  # Difensori
                 20, 40, 60, 80,  # Centrocampisti
                 50,  # Trequartista
                 50]  # Attaccante
        }
    }
    
    # Usa le coordinate della formazione specificata o 4-3-3 come fallback
    coords = formation_coords.get(formation, formation_coords["4-3-3"])
    
    # Crea il campo da calcio
    fig = go.Figure()
    
    # Aggiungi il rettangolo del campo (orizzontale)
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=100, y1=100,
        line=dict(color="white", width=2),
        fillcolor="#0A0F26"
    )
    
    # Aggiungi l'area di rigore (orizzontale)
    fig.add_shape(
        type="rect",
        x0=0, y0=30,
        x1=20, y1=70,
        line=dict(color="white", width=1),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Aggiungi la linea di metà campo
    fig.add_shape(
        type="line",
        x0=50, y0=0,
        x1=50, y1=100,
        line=dict(color="white", width=1)
    )
    
    # Aggiungi il cerchio di centrocampo
    fig.add_shape(
        type="circle",
        x0=45, y0=45,
        x1=55, y1=55,
        line=dict(color="white", width=1),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Aggiungi i punti dei giocatori
    fig.add_trace(go.Scatter(
        x=coords["x"],
        y=coords["y"],
        mode="markers+text",
        marker=dict(
            size=15,
            color="white",
            symbol="circle"
        ),
        text=[str(i) for i in range(1, len(coords["x"])+1)],  # Numeri dei giocatori
        textposition="middle center",
        textfont=dict(
            size=10,
            color="#0A0F26"
        ),
        hoverinfo="none"
    ))
    
    # Aggiorna il layout
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#0A0F26",
        paper_bgcolor="#0A0F26",
        margin=dict(l=0, r=0, t=30, b=0),
        height=150,
        width=250,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[-5, 105]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[-5, 105],
            scaleanchor="x",
            scaleratio=1
        ),
        title=dict(
            text=formation,
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
            font=dict(size=12, color="white")
        )
    )
    
    return fig

def create_formation_diagram(formation):
    """Crea un diagramma della formazione con punti sul campo usando Plotly"""
    # Parsing della formazione (es. "3-5-2" -> [3, 5, 2])
    players = [int(n) for n in formation.split("-")]
    
    # Crea una nuova figura con sfondo trasparente
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    # Dimensioni del campo
    field_length = 100
    field_width = 68
    
    # Disegna il rettangolo del campo con linee più spesse
    rect = plt.Rectangle((0, 0), field_length, field_width, 
                        facecolor='none', edgecolor='white', alpha=0.9,
                        linewidth=2.5)  # Aumentato spessore e opacità
    ax.add_patch(rect)
    
    # Disegna la linea di metà campo
    plt.axvline(x=field_length/2, color='white', alpha=0.8, linewidth=1.5)
    
    # Disegna il cerchio di centrocampo
    circle = plt.Circle((field_length/2, field_width/2), 8, 
                       facecolor='none', edgecolor='white', alpha=0.8,
                       linewidth=1.5)
    ax.add_patch(circle)
    
    # Disegna le aree di rigore
    # Area di sinistra
    penalty_box_left = plt.Rectangle((0, field_width/4), 16, field_width/2, 
                                   facecolor='none', edgecolor='white', alpha=0.8,
                                   linewidth=1.5)
    ax.add_patch(penalty_box_left)
    
    # Area di destra
    penalty_box_right = plt.Rectangle((field_length-16, field_width/4), 16, field_width/2, 
                                    facecolor='none', edgecolor='white', alpha=0.8,
                                    linewidth=1.5)
    ax.add_patch(penalty_box_right)
    
    # Calcola le posizioni dei giocatori
    x_positions = []
    y_positions = []
    
    # Portiere
    x_positions.append(8)
    y_positions.append(field_width/2)
    
    # Posizioni per ogni linea di giocatori dalla difesa all'attacco
    x_spacing = (field_length - 20) / (len(players) + 1)
    for i, num_players in enumerate(players):
        x = 15 + (i + 1) * x_spacing
        
        if num_players == 1:
            y_positions.append(field_width/2)
            x_positions.append(x)
        else:
            y_spacing = field_width / (num_players + 1)
            for j in range(num_players):
                y_positions.append(y_spacing * (j + 1))
                x_positions.append(x)
    
    # Disegna i punti dei giocatori
    plt.scatter(x_positions, y_positions, color='white', s=120, alpha=1)
    
    # Configura gli assi
    ax.set_xlim(-5, field_length + 5)
    ax.set_ylim(-5, field_width + 5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Rimuovi i bordi
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Converti il plot in immagine base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', 
                transparent=True, dpi=120)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Codifica l'immagine in base64
    graphic = base64.b64encode(image_png).decode('utf-8')
    
    # Crea una figura Plotly che mostra l'immagine
    fig = go.Figure()
    
    fig.add_layout_image(
        dict(
            source='data:image/png;base64,' + graphic,
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="stretch",
            layer="below"
        )
    )
    
    # Aggiorna il layout
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=150,
        width=250,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[0, 1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            range=[0, 1]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_defensive_stats(team_name, league):
    """Get defensive stats for a team"""
    try:
        df = pd.read_csv(f"datos/wyscout/2024-2025/{league}/defensivas.csv")
        team_stats = df[df['Equipo'] == team_name].iloc[0]
        return team_stats
    except Exception as e:
        print(f"Error getting defensive stats: {e}")
        return None

def create_defensive_table(team_name, league):
    """Create a table with defensive KPIs"""
    try:
        df = pd.read_csv(f"datos/wyscout/2024-2025/{league}/defensivas.csv")
        team_stats = df[df['Equipo'] == team_name].iloc[0]
        
        # Define metrics and their display names
        metrics = {
            'Balones recuperados altos': 'Balones Recuperados Altos',
            'PPDA': 'PPDA',
            'Faltas': 'Faltas',
            'Tiros en contra': 'Tiros en Contra',
            '%Pos AV': 'Posesión Adversario %',
            'Pases x Pos AV': 'Pases por Posesión Adversario',
            'Pases Z3 AV': 'Pases Zona 3 Adversario',
            'TAP AV': 'Toque Area Penalti Adversario',  # Nella tabella
            'Contras AV': 'Contraataques Adversario',
            'Pases Largos AV': 'Pases Largos Adversario'
        }
        
        # Create table header
        table_header = [
            html.Thead(html.Tr([
                html.Th("KPI", style={'width': '25%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Ranking", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Valor", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("", style={'width': '35%', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Mín / Máx", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Promedio Liga", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'})
            ], style={'backgroundColor': '#1a1a2e'}))
        ]
        
        # Create table rows
        rows = []
        for metric, display_name in metrics.items():
            # Calculate ranking
            rank_series = df[metric].rank(ascending=False, method='min')
            ranking = int(rank_series[team_stats.name])
            
            # Calculate statistics
            value = team_stats[metric]
            min_val = df[metric].min()
            max_val = df[metric].max()
            avg_val = df[metric].mean()
            
            # Normalize for colored bar
            if pd.isna(value):
                normalized = 0
            elif max_val == min_val:
                normalized = 0.5
            else:
                normalized = (value - min_val) / (max_val - min_val)
            
            # Color based on normalized value
            opacity = max(normalized, 0.2)
            color = f'rgba(0, 149, 255, {opacity})'
            
            # Create colored bar
            bar_width = normalized * 100
            bar_style = {
                'backgroundColor': color,
                'width': f'{max(bar_width, 10)}%',
                'height': '20px',
                'borderRadius': '4px'
            }
            
            # Add row to table
            row = html.Tr([
                html.Td(display_name, style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"#{ranking}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{value:.2f}" if not pd.isna(value) else "N/A", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(html.Div(style=bar_style), style={'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{min_val:.2f} / {max_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{avg_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'})
            ], style={'backgroundColor': '#0A0F26'})
            rows.append(row)
        
        return html.Div([
            html.H3("MÉTRICAS DEFENSIVAS", 
                   style={
                       'textAlign': 'center',
                       'marginBottom': '20px',
                       'color': 'white',
                       'fontSize': '24px',
                       'fontWeight': 'bold'
                   }),
            dbc.Table(
                table_header + [html.Tbody(rows)],
                bordered=False,
                dark=True,
                hover=True,
                responsive=True,
                striped=False,
                style={
                    'backgroundColor': '#0A0F26',
                    'color': 'white',
                    'width': '100%',
                    'borderCollapse': 'separate',
                    'borderSpacing': '0 1px'
                }
            )
        ], style={
            'backgroundColor': '#0A0F26',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        })
    except Exception as e:
        print(f"Error creating defensive table: {e}")
        return html.Div("Error loading defensive stats")

def get_set_piece_stats(team_name, league):
    """Get set piece (balon parado) statistics for a team"""
    try:
        # Read the set piece stats file
        file_path = os.path.join(WYSCOUT_FOLDER, league, "Balon parado.csv")  # Nome corretto del file
        df = pd.read_csv(file_path)
        
        # Find team data
        team_data = df[df['Equipo'] == team_name].iloc[0]
        
        return {
            'J Balon Parados': team_data['J Balon Parados'],
            '% Remate BL': team_data['% Remate BL'],
            'Corners': team_data['Corners'],
            '% Remate C': team_data['% Remate C'],
            'J Balon Parados avv': team_data['J Balon Parados avv'],
            '% Remate BL avv': team_data['% Remate BL avv'],
            'Corners avv': team_data['Corners avv'],
            '% Remate C avv': team_data['% Remate C avv']
        }
    except Exception as e:
        print(f"Error getting set piece stats for {team_name}: {e}")
        return None

def create_defensive_radar(team_name, league, comparison_team=None):
    """Create a radar chart with defensive metrics"""
    try:
        df = pd.read_csv(f"datos/wyscout/2024-2025/{league}/defensivas.csv")
        
        # Define metrics and their display names
        metrics = {
            'Balones recuperados altos': 'Balones Recuperados Altos',
            'PPDA': 'PPDA',
            'Faltas': 'Faltas',
            'Tiros en contra': 'Tiros en Contra',
            '%Pos AV': 'Posesión Adversario %',
            'Pases x Pos AV': 'Pases por Posesión Adversario',
            'Pases Z3 AV': 'Pases Zona 3 Adversario',
            'TAP AV': 'Toque Area Penalti Adversario',  # Nel radar
            'Contras AV': 'Contraataques Adversario',
            'Pases Largos AV': 'Pases Largos Adversario'
        }
        
        categories = list(metrics.keys())
        angles = [(i * 360 / len(categories)) % 360 for i in range(len(categories))]
        
        # Calculate values for main team
        values = []
        raw_values = []
        for category in categories:
            value = df[df['Equipo'] == team_name][category].iloc[0]
            
            if pd.isna(value):
                normalized = 0.1  # Minimum value for NaN
                scaled = 10
            else:
                # Calculate normalized value (0-100)
                min_val = df[category].min()
                max_val = df[category].max()
                if max_val == min_val:
                    scaled = 50
                else:
                    scaled = ((value - min_val) / (max_val - min_val)) * 100
                    # Ensure minimum visibility
                    scaled = max(scaled, 10)
            
            values.append(scaled)
            raw_values.append(value if not pd.isna(value) else 0)
        
        # Create figure
        fig = go.Figure()
        
        # Add grid lines
        for i in range(20, 120, 20):
            fig.add_trace(go.Scatterpolar(
                r=[i]*100,
                theta=np.linspace(0, 360, 100),
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add sectors for main team
        for i, (angle, value, raw_value) in enumerate(zip(angles, values, raw_values)):
            opacity = max(value/100, 0.2)
            color = f'rgba(0, 149, 255, {opacity})'
            
            # Create points to form sector
            theta_sector = np.linspace(angle - 360/len(categories)*0.4, angle + 360/len(categories)*0.4, 50)
            r_sector = np.array([value] * 50)
            
            fig.add_trace(go.Scatterpolar(
                r=np.concatenate([[0], r_sector, [0]]),
                theta=np.concatenate([[angle], theta_sector, [angle]]),
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(color=color, width=0),
                name=f"{team_name} - {metrics[categories[i]]}",
                customdata=[[raw_value]]*52,
                hovertemplate=f"{team_name}<br>{metrics[categories[i]]}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
            ))
        
        # Add comparison team if provided
        if comparison_team:
            comp_values = []
            comp_raw_values = []
            for category in categories:
                value = df[df['Equipo'] == comparison_team][category].iloc[0]
                
                if pd.isna(value):
                    normalized = 0.1
                    scaled = 10
                else:
                    # Calculate normalized value (0-100)
                    min_val = df[category].min()
                    max_val = df[category].max()
                    if max_val == min_val:
                        scaled = 50
                    else:
                        scaled = ((value - min_val) / (max_val - min_val)) * 100
                        # Ensure minimum visibility
                        scaled = max(scaled, 10)
                
                comp_values.append(scaled)
                comp_raw_values.append(value if not pd.isna(value) else 0)
            
            for i, (angle, value, raw_value) in enumerate(zip(angles, comp_values, comp_raw_values)):
                theta_sector = np.linspace(angle - 360/len(categories)*0.4, angle + 360/len(categories)*0.4, 50)
                r_sector = np.array([value] * 50)
                
                fig.add_trace(go.Scatterpolar(
                    r=np.concatenate([[0], r_sector, [0]]),
                    theta=np.concatenate([[angle], theta_sector, [angle]]),
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(255, 69, 0, 0.6)',
                    line=dict(color='rgba(255, 69, 0, 0.6)', width=0),
                    name=f"{comparison_team} - {metrics[categories[i]]}",
                    customdata=[[raw_value]]*52,
                    hovertemplate=f"{comparison_team}<br>{metrics[categories[i]]}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showline=False,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    tickfont=dict(color='white', size=8),
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['', '', '', '', '']
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=angles,
                    ticktext=[metrics[cat] for cat in categories],
                    tickfont=dict(color='white', size=8),
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    rotation=90,
                    direction='clockwise'
                ),
                bgcolor='#0A0F26'
            ),
            showlegend=False,
            paper_bgcolor='#0A0F26',
            plot_bgcolor='#0A0F26',
            margin=dict(t=50, b=50, l=50, r=50),
            height=400,
            width=400,
            title=dict(
                text='MÉTRICAS DEFENSIVAS<br><span style="font-size: 10px; color: #666">(Valores porcentuales)</span>',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=14, color='white')
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error creating defensive radar: {e}")
        return go.Figure()

def create_set_piece_radar(team_name, league, comparison_team=None):
    """Create a radar chart with set piece metrics"""
    try:
        df = pd.read_csv(f"datos/wyscout/2024-2025/{league}/Balon parado.csv")
        
        # Define metrics with exact Spanish names
        metrics = {
            'J Balon Parados': 'Jugadas a Balón Parado',
            '% Remate BL': '% Remate Balón Parado',
            'Corners': 'Córners',
            '% Remate C': '% Remate Córner',
            'J Balon Parados avv': 'Jugadas a Balón Parado Adversario',
            '% Remate BL avv': '% Remate Balón Parado Adversario',
            'Corners avv': 'Córners Adversario',
            '% Remate C avv': '% Remate Córner Adversario'
        }
        
        categories = list(metrics.keys())
        angles = [(i * 360 / len(categories)) % 360 for i in range(len(categories))]
        
        # Calculate values for main team
        values = []
        raw_values = []
        for category in categories:
            value = df[df['Equipo'] == team_name][category].iloc[0]
            
            if pd.isna(value):
                normalized = 0.1  # Minimum value for NaN
                scaled = 10
            else:
                # Calculate normalized value (0-100)
                min_val = df[category].min()
                max_val = df[category].max()
                if max_val == min_val:
                    scaled = 50
                else:
                    scaled = ((value - min_val) / (max_val - min_val)) * 100
                    # Ensure minimum visibility
                    scaled = max(scaled, 10)
            
            values.append(scaled)
            raw_values.append(value if not pd.isna(value) else 0)
        
        # Create figure
        fig = go.Figure()
        
        # Add grid lines
        for i in range(20, 120, 20):
            fig.add_trace(go.Scatterpolar(
                r=[i]*100,
                theta=np.linspace(0, 360, 100),
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add sectors for main team
        for i, (angle, value, raw_value) in enumerate(zip(angles, values, raw_values)):
            opacity = max(value/100, 0.2)
            color = f'rgba(0, 149, 255, {opacity})'
            
            # Create points to form sector
            theta_sector = np.linspace(angle - 360/len(categories)*0.4, angle + 360/len(categories)*0.4, 50)
            r_sector = np.array([value] * 50)
            
            fig.add_trace(go.Scatterpolar(
                r=np.concatenate([[0], r_sector, [0]]),
                theta=np.concatenate([[angle], theta_sector, [angle]]),
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(color=color, width=0),
                name=f"{team_name} - {metrics[categories[i]]}",
                customdata=[[raw_value]]*52,
                hovertemplate=f"{team_name}<br>{metrics[categories[i]]}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
            ))
        
        # Add comparison team if provided
        if comparison_team:
            comp_values = []
            comp_raw_values = []
            for category in categories:
                value = df[df['Equipo'] == comparison_team][category].iloc[0]
                
                if pd.isna(value):
                    normalized = 0.1
                    scaled = 10
                else:
                    # Calculate normalized value (0-100)
                    min_val = df[category].min()
                    max_val = df[category].max()
                    if max_val == min_val:
                        scaled = 50
                    else:
                        scaled = ((value - min_val) / (max_val - min_val)) * 100
                        # Ensure minimum visibility
                        scaled = max(scaled, 10)
                
                comp_values.append(scaled)
                comp_raw_values.append(value if not pd.isna(value) else 0)
            
            for i, (angle, value, raw_value) in enumerate(zip(angles, comp_values, comp_raw_values)):
                theta_sector = np.linspace(angle - 360/len(categories)*0.4, angle + 360/len(categories)*0.4, 50)
                r_sector = np.array([value] * 50)
                
                fig.add_trace(go.Scatterpolar(
                    r=np.concatenate([[0], r_sector, [0]]),
                    theta=np.concatenate([[angle], theta_sector, [angle]]),
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(255, 69, 0, 0.6)',
                    line=dict(color='rgba(255, 69, 0, 0.6)', width=0),
                    name=f"{comparison_team} - {metrics[categories[i]]}",
                    customdata=[[raw_value]]*52,
                    hovertemplate=f"{comparison_team}<br>{metrics[categories[i]]}<br>Value: %{{customdata[0]:.2f}}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showline=False,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    tickfont=dict(color='white', size=8),
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['', '', '', '', '']
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=angles,
                    ticktext=[metrics[cat] for cat in categories],
                    tickfont=dict(color='white', size=8),
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    rotation=90,
                    direction='clockwise'
                ),
                bgcolor='#0A0F26'
            ),
            showlegend=False,
            paper_bgcolor='#0A0F26',
            plot_bgcolor='#0A0F26',
            margin=dict(t=50, b=50, l=50, r=50),
            height=400,
            width=400,
            title=dict(
                text='MÉTRICAS DE BALÓN PARADO<br><span style="font-size: 10px; color: #666">(Valores porcentuales)</span>',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=14, color='white')
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error creating set piece radar: {e}")
        return go.Figure()

def create_set_piece_table(team_name, league):
    """Create a table with set piece KPIs"""
    try:
        df = pd.read_csv(f"datos/wyscout/2024-2025/{league}/Balon parado.csv")
        team_stats = df[df['Equipo'] == team_name].iloc[0]
        
        # Define metrics with exact Spanish names
        metrics = {
            'J Balon Parados': 'J Balon Parados',
            '% Remate BL': '% Remate BL',
            'Corners': 'Corners',
            '% Remate C': '% Remate C',
            'J Balon Parados avv': 'J Balon Parados avv',
            '% Remate BL avv': '% Remate BL avv',
            'Corners avv': 'Corners avv',
            '% Remate C avv': '% Remate C avv'
        }
        
        # Create table header
        table_header = [
            html.Thead(html.Tr([
                html.Th("KPI", style={'width': '25%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Ranking", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Valor", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("", style={'width': '35%', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Mín / Máx", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Promedio Liga", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'})
            ], style={'backgroundColor': '#1a1a2e'}))
        ]
        
        # Create table rows
        rows = []
        for metric, display_name in metrics.items():
            # Calculate ranking
            rank_series = df[metric].rank(ascending=False, method='min')
            ranking = int(rank_series[team_stats.name])
            
            # Calculate statistics
            value = team_stats[metric]
            min_val = df[metric].min()
            max_val = df[metric].max()
            avg_val = df[metric].mean()
            
            # Normalize for colored bar
            if pd.isna(value):
                normalized = 0
            elif max_val == min_val:
                normalized = 0.5
            else:
                normalized = (value - min_val) / (max_val - min_val)
            
            # Color based on normalized value
            opacity = max(normalized, 0.2)
            color = f'rgba(0, 149, 255, {opacity})'
            
            # Create colored bar
            bar_width = normalized * 100
            bar_style = {
                'backgroundColor': color,
                'width': f'{max(bar_width, 10)}%',
                'height': '20px',
                'borderRadius': '4px'
            }
            
            # Add row to table
            row = html.Tr([
                html.Td(display_name, style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"#{ranking}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{value:.2f}" if not pd.isna(value) else "N/A", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(html.Div(style=bar_style), style={'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{min_val:.2f} / {max_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                html.Td(f"{avg_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'})
            ], style={'backgroundColor': '#0A0F26'})
            rows.append(row)
        
        return html.Div([
            html.H3("MÉTRICAS DE BALÓN PARADO", 
                   style={
                       'textAlign': 'center',
                       'marginBottom': '20px',
                       'color': 'white',
                       'fontSize': '24px',
                       'fontWeight': 'bold'
                   }),
            dbc.Table(
                table_header + [html.Tbody(rows)],
                bordered=False,
                dark=True,
                hover=True,
                responsive=True,
                striped=False,
                style={
                    'backgroundColor': '#0A0F26',
                    'color': 'white',
                    'width': '100%',
                    'borderCollapse': 'separate',
                    'borderSpacing': '0 1px'
                }
            )
        ], style={
            'backgroundColor': '#0A0F26',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        })
    except Exception as e:
        print(f"Error creating set piece table: {e}")
        return html.Div("Error loading set piece stats")

def get_country_flag(league):
    """Restituisce l'URL della bandiera del paese della lega"""
    country_flags = {
        "Serie A": "/assets/2024-25/Serie A/bandera_italia",
        "Premier League": "/assets/2024-25/Premier League/bandera_Inglaterra",
        "La Liga": "/assets/2024-25/La Liga/bandera_España",
        "Bundesliga": "/assets/2024-25/Bundesliga/bandera_Alemania",
        "Ligue 1": "/assets/2024-25/Ligue 1/bandera_Francia"
    }
    return country_flags.get(league)

def calculate_defensive_similarity(team_name, league):
    """Calcola la similarità difensiva tra una squadra e tutte le altre"""
    try:
        # Leggi il file delle metriche difensive
        file_path = os.path.join(WYSCOUT_FOLDER, league, "defensivas.csv")
        df = pd.read_csv(file_path)
        
        # Metriche da considerare per la similarità difensiva
        metrics = [
            'Balones recuperados altos',
            'PPDA',
            'Faltas',
            'Tiros en contra',
            '%Pos AV',
            'Pases x Pos AV',
            'Pases Z3 AV',
            'TAP AV',
            'Contras AV',
            'Pases Largos AV'
        ]
        
        # Ottieni i dati della squadra target
        target_team = df[df['Equipo'].str.startswith(team_name)]
        if target_team.empty:
            print(f"No data found for team {team_name}")
            return []
            
        target_team_data = target_team.iloc[0]
        similarities = []
        
        # Per ogni squadra nel campionato (esclusa la squadra target)
        for idx, row in df[~df['Equipo'].str.startswith(team_name)].iterrows():
            try:
                metric_similarities = []
                
                # Calcola la similarità per ogni metrica
                for metric in metrics:
                    # Ottieni i valori min e max del campionato per questa metrica
                    metric_min = df[metric].min()
                    metric_max = df[metric].max()
                    
                    # Se min e max sono uguali, passa alla prossima metrica
                    if metric_max == metric_min:
                        continue
                    
                    # Normalizza i valori delle due squadre (0-100%)
                    target_normalized = (target_team_data[metric] - metric_min) / (metric_max - metric_min)
                    other_normalized = (row[metric] - metric_min) / (metric_max - metric_min)
                    
                    # Calcola la similarità come complemento della differenza percentuale
                    similarity = 1 - abs(target_normalized - other_normalized)
                    metric_similarities.append(similarity)
                
                # Calcola la similarità media per questa squadra
                if metric_similarities:
                    avg_similarity = sum(metric_similarities) / len(metric_similarities)
                    similarities.append({
                        'team': row['Equipo'],
                        'similarity': avg_similarity
                    })
            except Exception as e:
                print(f"Error calculating defensive similarity for {row['Equipo']}: {e}")
                continue
        
        # Ordina per similarità e prendi le prime 4
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:4]
        
    except Exception as e:
        print(f"Error in calculate_defensive_similarity: {e}")
        return []

def calculate_set_piece_similarity(team_name, league):
    """Calcola la similarità nei calci piazzati tra una squadra e tutte le altre"""
    try:
        # Leggi il file delle metriche dei calci piazzati
        file_path = os.path.join(WYSCOUT_FOLDER, league, "Balon parado.csv")
        df = pd.read_csv(file_path)
        
        # Metriche da considerare per la similarità nei calci piazzati
        metrics = [
            'J Balon Parados',
            '% Remate BL',
            'Corners',
            '% Remate C',
            'J Balon Parados avv',
            '% Remate BL avv',
            'Corners avv',
            '% Remate C avv'
        ]
        
        # Ottieni i dati della squadra target
        target_team = df[df['Equipo'].str.startswith(team_name)]
        if target_team.empty:
            print(f"No data found for team {team_name}")
            return []
            
        target_team_data = target_team.iloc[0]
        similarities = []
        
        # Per ogni squadra nel campionato (esclusa la squadra target)
        for idx, row in df[~df['Equipo'].str.startswith(team_name)].iterrows():
            try:
                metric_similarities = []
                
                # Calcola la similarità per ogni metrica
                for metric in metrics:
                    # Ottieni i valori min e max del campionato per questa metrica
                    metric_min = df[metric].min()
                    metric_max = df[metric].max()
                    
                    # Se min e max sono uguali, passa alla prossima metrica
                    if metric_max == metric_min:
                        continue
                    
                    # Normalizza i valori delle due squadre (0-100%)
                    target_normalized = (target_team_data[metric] - metric_min) / (metric_max - metric_min)
                    other_normalized = (row[metric] - metric_min) / (metric_max - metric_min)
                    
                    # Calcola la similarità come complemento della differenza percentuale
                    similarity = 1 - abs(target_normalized - other_normalized)
                    metric_similarities.append(similarity)
                
                # Calcola la similarità media per questa squadra
                if metric_similarities:
                    avg_similarity = sum(metric_similarities) / len(metric_similarities)
                    similarities.append({
                        'team': row['Equipo'],
                        'similarity': avg_similarity
                    })
            except Exception as e:
                print(f"Error calculating set piece similarity for {row['Equipo']}: {e}")
                continue
        
        # Ordina per similarità e prendi le prime 4
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:4]
        
    except Exception as e:
        print(f"Error in calculate_set_piece_similarity: {e}")
        return []

def create_defensive_similar_teams_card(team_name, league):
    """Crea una card con le quattro squadre più simili per stile difensivo (stile uniforme)"""
    similar_teams = calculate_defensive_similarity(team_name, league)
    if not similar_teams:
        return html.Div("No similar teams found")
    return html.Div([
        html.H4("EQUIPOS CON ESTILO DEFENSIVO SIMILAR", 
               style={
                   'textAlign': 'center',
                   'color': 'white',
                   'marginBottom': '10px',
                   'fontSize': '16px',
                   'fontWeight': 'bold'
               }),
        html.Div([
            html.Div([
                html.Button([
                    html.Img(
                        src=get_team_logo(league, team['team']),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'objectFit': 'contain',
                            'marginBottom': '3px'
                        }
                    ) if get_team_logo(league, team['team']) else html.Div(
                        team['team'][:3].upper(),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'backgroundColor': '#1a1a2e',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'borderRadius': '50%',
                            'color': 'white',
                            'fontSize': '10px',
                            'marginBottom': '3px'
                        }
                    ),
                    html.Div(team['team'], style={
                        'color': 'white',
                        'fontSize': '11px',
                        'textAlign': 'center',
                        'marginBottom': '2px'
                    })
                ],
                id={'type': 'defensive-similar-team-btn', 'index': team['team']},
                n_clicks=0,
                style={
                    'background': 'none',
                    'border': 'none',
                    'cursor': 'pointer',
                    'padding': '8px',
                    'backgroundColor': '#1a1a2e',
                    'borderRadius': '8px',
                    'margin': '3px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
                    'width': '80px',
                    'border': '1px solid rgba(255,255,255,0.1)',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                })
            for team in similar_teams
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(2, 1fr)',
                'gap': '5px',
                'justifyContent': 'center'
            })
        ])
    ], style={
        'backgroundColor': '#0A0F26',
        'padding': '15px',
        'borderRadius': '8px',
        'marginRight': '15px',
        'width': '180px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.2)'
    })

def create_set_piece_similar_teams_card(team_name, league):
    """Crea una card con le quattro squadre più simili per stile nei calci piazzati (stile uniforme)"""
    similar_teams = calculate_set_piece_similarity(team_name, league)
    if not similar_teams:
        return html.Div("No similar teams found")
    return html.Div([
        html.H4("EQUIPOS CON ESTILO DE BALÓN PARADO SIMILAR", 
               style={
                   'textAlign': 'center',
                   'color': 'white',
                   'marginBottom': '10px',
                   'fontSize': '16px',
                   'fontWeight': 'bold'
               }),
        html.Div([
            html.Div([
                html.Button([
                    html.Img(
                        src=get_team_logo(league, team['team']),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'objectFit': 'contain',
                            'marginBottom': '3px'
                        }
                    ) if get_team_logo(league, team['team']) else html.Div(
                        team['team'][:3].upper(),
                        style={
                            'width': '35px',
                            'height': '35px',
                            'backgroundColor': '#1a1a2e',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'borderRadius': '50%',
                            'color': 'white',
                            'fontSize': '10px',
                            'marginBottom': '3px'
                        }
                    ),
                    html.Div(team['team'], style={
                        'color': 'white',
                        'fontSize': '11px',
                        'textAlign': 'center',
                        'marginBottom': '2px'
                    })
                ],
                id={'type': 'set-piece-similar-team-btn', 'index': team['team']},
                n_clicks=0,
                style={
                    'background': 'none',
                    'border': 'none',
                    'cursor': 'pointer',
                    'padding': '8px',
                    'backgroundColor': '#1a1a2e',
                    'borderRadius': '8px',
                    'margin': '3px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.3)',
                    'width': '80px',
                    'border': '1px solid rgba(255,255,255,0.1)',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                })
            for team in similar_teams
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(2, 1fr)',
                'gap': '5px',
                'justifyContent': 'center'
            })
        ])
    ], style={
        'backgroundColor': '#0A0F26',
        'padding': '15px',
        'borderRadius': '8px',
        'marginRight': '15px',
        'width': '180px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.2)'
    })

# Layout della pagina
layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    html.Div(id="team-header", className="mt-2")
], fluid=True, style={"backgroundColor": "#0A0F26", "color": "#FFFFFF", "padding": "10px"})

@callback(
    Output("team-header", "children"),
    Input("url", "pathname")
)
def update_team_page(pathname):
    if not pathname:
        return None
    
    parts = pathname.strip("/").split("/")
    if len(parts) < 3:
        return None

    league = urllib.parse.unquote(parts[1])
    team_name = urllib.parse.unquote(parts[2])
    canonical_name = get_canonical_team_name(team_name)
    # (debug rimosso)
    expected_league = TEAM_LEAGUE_MAPPING.get(canonical_name)
    if not expected_league or expected_league != league:
        return None
    
    logo_src = get_team_logo(league, canonical_name)
    flag_src = get_country_flag(league)
    
    # Create all figures and tables
    results_fig = create_results_pie(canonical_name, league)
    goals_fig = create_goals_chart(canonical_name, league)
    age_stats = get_team_age_stats(canonical_name, league)
    age_fig = create_age_indicator(age_stats)
    formation = get_most_used_formation(canonical_name, league)
    
    buildup_fig = create_buildup_radar(canonical_name, league)
    offensive_fig = create_offensive_radar(canonical_name, league)
    defensive_fig = create_defensive_radar(canonical_name, league)
    set_piece_fig = create_set_piece_radar(canonical_name, league)
    
    buildup_table = create_kpi_table(canonical_name, league)
    offensive_table = create_offensive_table(canonical_name, league)
    defensive_table = create_defensive_table(canonical_name, league)
    set_piece_table = create_set_piece_table(canonical_name, league)
    
    return html.Div([
        # Prima riga - Header con statistiche base
    html.Div([
        # Logo e bandiera
        html.Div([
            html.Div([
                    html.Img(src=logo_src, style={"width": "100px", "height": "100px", "objectFit": "contain"}) if logo_src else None,
            ]),
            html.Div([
                    html.Img(src=flag_src, style={"width": "30px", "height": "20px", "objectFit": "contain", "marginLeft": "35px"}) if flag_src else None,
            ])
            ], style={"display": "flex", "flexDirection": "row", "alignItems": "flex-start", "marginRight": "50px"}),
        
        # Grafico risultati
        html.Div([
                dcc.Graph(figure=results_fig, config={'displayModeBar': False})
        ], style={"display": "inline-block", "marginRight": "50px"}),
        
        # Grafico gol
        html.Div([
                dcc.Graph(figure=goals_fig, config={'displayModeBar': False})
        ], style={"display": "inline-block", "marginRight": "50px"}),
        
        # Indicatore età media
        html.Div([
                dcc.Graph(figure=age_fig, config={'displayModeBar': False})
        ], style={"display": "inline-block", "marginRight": "50px"}),
        
        # Formazione
        html.Div([
            html.Div([
                    html.Div(formation, style={"color": "white", "fontSize": "24px", "fontWeight": "bold", "textAlign": "center", "marginBottom": "10px"}),
                html.Div([
                        dcc.Graph(figure=create_formation_diagram(formation), config={'displayModeBar': False}),
                        dbc.Button("Plantilla y Perfiles", id='plantilla-button', color='primary', className='ms-3',
                        href=f"/team/{league}/{canonical_name}/plantilla",
                                 style={'height': 'fit-content', 'whiteSpace': 'nowrap', 'fontSize': '14px'})
                    ], style={"display": "flex", "flexDirection": "row", "alignItems": "center", "justifyContent": "flex-start", "width": "100%"})
                ], style={"backgroundColor": "#0A0F26", "padding": "15px", "borderRadius": "8px", "width": "fit-content"})
            ], style={"display": "inline-block", "marginLeft": "0px", "backgroundColor": "#0A0F26"})
        ], style={"display": "flex", "flexDirection": "row", "alignItems": "center", "padding": "20px",
                  "backgroundColor": "#0A0F26", "marginBottom": "20px", "width": "100%", "justifyContent": "flex-start"}),
        
        # Seconda riga - Radar charts 2x2 con cards di similarità
        html.Div([
            # Bottone Atalanta comparison
        html.Div([
                dbc.Button(
                    "Comparar con Atalanta",
                    id="atalanta-comparison-btn",
                    color="primary",
                    className="mb-4",
                    n_clicks=0,
                    style={'width': 'fit-content', 'margin': '0 auto', 'display': 'block'}
                )
            ]),
            
            # Prima riga di radar
            dbc.Row([
                # Build-up Radar + Similar Card
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=buildup_fig, id="buildup-radar"), width=8),
                        dbc.Col(create_similar_teams_card(canonical_name, league, "buildup"), width=4)
                    ])
                ], width=6),
                
                # Offensive Radar + Similar Card
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=offensive_fig, id="offensive-radar"), width=8),
                        dbc.Col(create_offensive_similar_teams_card(canonical_name, league), width=4)
                    ])
                ], width=6)
            ], className="mb-4"),
        
            # Seconda riga di radar
            dbc.Row([
                # Defensive Radar + Similar Card
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=defensive_fig, id="defensive-radar"), width=8),
                        dbc.Col(create_defensive_similar_teams_card(canonical_name, league), width=4)
                    ])
                ], width=6),
        
                # Set Piece Radar + Similar Card
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=set_piece_fig, id="set-piece-radar"), width=8),
                        dbc.Col(create_set_piece_similar_teams_card(canonical_name, league), width=4)
                    ])
                ], width=6)
            ])
        ], style={"backgroundColor": "#0A0F26", "padding": "20px", "marginBottom": "20px"}),
        
        # Terza riga - Tabelle in colonna
        html.Div([
            # Build-up Table
            html.Div(buildup_table, className="mb-4"),
            
            # Offensive Table
            html.Div(offensive_table, className="mb-4"),
            
            # Defensive Table
            html.Div(defensive_table, className="mb-4"),
            
            # Set Piece Table
            html.Div(set_piece_table)
        ], style={"backgroundColor": "#0A0F26", "padding": "20px"})
        
    ], style={"backgroundColor": "#0A0F26", "padding": "20px"})

@callback(
    [Output("buildup-radar", "figure"),
     Output("offensive-radar", "figure"),
     Output("defensive-radar", "figure"),
     Output("set-piece-radar", "figure")],
    [Input({"type": "buildup-similar-team-btn", "index": ALL}, "n_clicks"),
     Input({"type": "offensive-similar-team-btn", "index": ALL}, "n_clicks"),
     Input({"type": "defensive-similar-team-btn", "index": ALL}, "n_clicks"),
     Input({"type": "set-piece-similar-team-btn", "index": ALL}, "n_clicks"),
     Input("atalanta-comparison-btn", "n_clicks")],
    [State("url", "pathname")]
)
def update_radar_comparisons(buildup_clicks, offensive_clicks, defensive_clicks, set_piece_clicks, atalanta_clicks, pathname):
    if not pathname:
        raise PreventUpdate
    
    parts = pathname.strip("/").split("/")
    if len(parts) < 3:
        raise PreventUpdate

    league = urllib.parse.unquote(parts[1])
    team_name = urllib.parse.unquote(parts[2])
    canonical_name = get_canonical_team_name(team_name)
    
    # Get the triggered input
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id']
    
    # Handle Atalanta comparison
    if trigger_id == "atalanta-comparison-btn.n_clicks":
        return (
            create_buildup_radar(canonical_name, league, "Atalanta"),
            create_offensive_radar(canonical_name, league, "Atalanta"),
            create_defensive_radar(canonical_name, league, "Atalanta"),
            create_set_piece_radar(canonical_name, league, "Atalanta")
        )
    
    # Handle similar team clicks
    if "buildup-similar-team-btn" in trigger_id:
        idx = json.loads(trigger_id.split(".")[0])["index"]
        similar_teams = calculate_similarity(canonical_name, league)
        if idx < len(similar_teams):
            comparison_team = similar_teams[idx]["team"]
            return (
                create_buildup_radar(canonical_name, league, comparison_team),
                create_offensive_radar(canonical_name, league),
                create_defensive_radar(canonical_name, league),
                create_set_piece_radar(canonical_name, league)
            )
    
    elif "offensive-similar-team-btn" in trigger_id:
        idx = json.loads(trigger_id.split(".")[0])["index"]
        similar_teams = calculate_offensive_similarity(canonical_name, league)
        if idx < len(similar_teams):
            comparison_team = similar_teams[idx]["team"]
            return (
                create_buildup_radar(canonical_name, league),
                create_offensive_radar(canonical_name, league, comparison_team),
                create_defensive_radar(canonical_name, league),
                create_set_piece_radar(canonical_name, league)
            )
    
    elif "defensive-similar-team-btn" in trigger_id:
        idx = json.loads(trigger_id.split(".")[0])["index"]
        similar_teams = calculate_defensive_similarity(canonical_name, league)
        team_obj = next((team for team in similar_teams if team['team'] == idx), None)
        if team_obj:
            comparison_team = team_obj["team"]
            return (
                create_buildup_radar(canonical_name, league),
                create_offensive_radar(canonical_name, league),
                create_defensive_radar(canonical_name, league, comparison_team),
                create_set_piece_radar(canonical_name, league)
            )
    
    elif "set-piece-similar-team-btn" in trigger_id:
        idx = json.loads(trigger_id.split(".")[0])["index"]
        similar_teams = calculate_set_piece_similarity(canonical_name, league)
        team_obj = next((team for team in similar_teams if team['team'] == idx), None)
        if team_obj:
            comparison_team = team_obj["team"]
            return (
                create_buildup_radar(canonical_name, league),
                create_offensive_radar(canonical_name, league),
                create_defensive_radar(canonical_name, league),
                create_set_piece_radar(canonical_name, league, comparison_team)
            )
    
    # Default case: return all radars without comparison
    return (
        create_buildup_radar(canonical_name, league),
        create_offensive_radar(canonical_name, league),
        create_defensive_radar(canonical_name, league),
        create_set_piece_radar(canonical_name, league)
    )