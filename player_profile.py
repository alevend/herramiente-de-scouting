import dash
from dash import html, dcc, callback, Input, Output, State, ClientsideFunction, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import os
from urllib.parse import unquote, quote
import numpy as np
from functools import lru_cache
from scipy.spatial.distance import euclidean, cosine
import unicodedata
import re

# Importa le configurazioni centralizzate IN SPAGNOLO. QUESTA è l'unica fonte.
from utils.config import POSITION_TRANSLATION, PERFILES_KPI_PESOS, PROFILE_TO_ROLES, ROLE_PROFILES as ROLE_PROFILES_ES
from utils.data_utils import find_similar_players, normalize_kpi_with_coefficients, load_main_data
from utils.config import PERFILES_KPI_PESOS # Import separato per forzare l'aggiornamento

# Definisco i percorsi degli asset qui, dato che non sono in config.py
default_logo = "/assets/default_logo.png"
sin_foto = "/assets/Sin_foto.png"

# --- SOLUZIONE DEFINITIVA ---
# Mappa per tradurre i profili SPAGNOLO -> ITALIANO per l'accesso ai dati.
# I dati nel CSV usano i nomi in italiano, quindi questa traduzione è FONDAMENTALE.
ROLE_PROFILES_IT = {
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
# --- FINE SOLUZIONE ---

# Register the page with correct path template
dash.register_page(__name__, path_template="/player/<player_name>")

def convert_to_numeric(value):
    """Convert string values to numeric, handling various formats."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Remove any commas and convert to float
        return float(str(value).replace(',', ''))
    except (ValueError, TypeError):
        return 0

def get_player_category(age, score):
    """Determine player category based on age and score."""
    if age is None:
        return "N/A"
    
    # Floor the age to get the integer part
    try:
        age_floor = int(float(age))
    except (ValueError, TypeError):
        return "N/A"
    
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

# Lista delle categorie possibili
CATEGORIE = [
    "Estrella Mundial",
    "Jugador Top",
    "Buen Jugador",
    "Jugador de Media Tabla",
    "Jugador de Baja Tabla",
    "Jugador de División 2",
    "Joven Estrella Mundial",
    "Futura Estrella",
    "Joven con Calidad",
    "Buen Joven",
    "Joven"
]

# Coefficienti lega
coeff_leghe = {
    "Premier League": 2.0, "La Liga": 2.0, "Serie A": 2.0, "Bundesliga": 2.0,
    "Ligue 1": 2.0, "Primeira Liga": 1.5, "Eredivisie": 1.5,
    "Championship": 1.5, "Belgio": 1.5, "Serie B": 1.0
}

# Percorso del file dati
DATA_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/datos/fbref/base_datos_tutte_le_leghe_renamed.csv")
ORIGINAL_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/base_datos_tutte_le_leghe_renamed.csv")

def normalize_player_name(name):
    """
    Normalizza il nome del giocatore gestendo caratteri speciali e accenti.
    """
    try:
        # Se il nome è bytes, decodifica come UTF-8
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        # Prova a decodificare se sembra essere codificato in modo errato
        try:
            # Prova a decodificare se sembra essere codificato in latin1 ma interpretato come utf-8
            name = name.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        # Normalizza i caratteri Unicode
        name = unicodedata.normalize('NFKD', name)
        # Ricomponi i caratteri
        name = unicodedata.normalize('NFC', name)
        
        # Rimuovi caratteri non stampabili e spazi multipli
        name = re.sub(r'\s+', ' ', name.strip())
        
        return name
    except Exception:
        return name

# Lista delle colonne che devono rimanere stringhe
STRING_COLUMNS = ['Squadra', 'Lega', 'Height', 'Position', 'Foot', 'Name', 'Nationality', 'Signed', 'Contract Until']

# Converti le colonne numeriche in modo efficiente
def convert_numeric_columns(df):
    # Lista delle colonne da NON convertire (categorie e stringhe)
    non_numeric_cols = [col for col in df.columns if col.startswith('Category_')] + STRING_COLUMNS
    
    # Converti solo le colonne non categoriche e non stringhe
    numeric_columns = [col for col in df.select_dtypes(include=['object']).columns 
                      if col not in non_numeric_cols]
    
    for col in numeric_columns:
        try:
            if df[col].str.contains(r'[0-9]', na=False).any():
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), 
                                      errors='coerce').astype('float32')
        except (AttributeError, TypeError):
            continue
    return df

# Carica il dataset principale
GLOBAL_DF = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False, keep_default_na=False, na_values=[''])

# Carica il dataset originale/completo per unire le colonne mancanti
try:
    ORIGINAL_DF = pd.read_csv(ORIGINAL_FILE, encoding="utf-8", low_memory=False)

    if 'Player_ID' in GLOBAL_DF.columns and 'Player_ID' in ORIGINAL_DF.columns:
        
        # Identifica le colonne presenti nel file originale ma non in quello principale
        missing_cols = [col for col in ORIGINAL_DF.columns if col not in GLOBAL_DF.columns]
        
        if missing_cols:
            print(f"Trovate colonne mancanti nel file principale. Tentativo di unione per: {missing_cols}")
            
            # Assicura che la chiave di join sia dello stesso tipo (stringa) per evitare errori
            GLOBAL_DF['Player_ID'] = GLOBAL_DF['Player_ID'].astype(str)
            ORIGINAL_DF['Player_ID'] = ORIGINAL_DF['Player_ID'].astype(str)
            
            # Seleziona solo le colonne mancanti + la chiave di join dal file originale
            cols_to_merge = ['Player_ID'] + missing_cols
            
            # Esegui il merge a sinistra per aggiungere le colonne a GLOBAL_DF
            GLOBAL_DF = pd.merge(
                GLOBAL_DF,
                ORIGINAL_DF[cols_to_merge],
                on='Player_ID',
                how='left'
            )
            print(f"Unione completata con successo.")
        else:
            print("Nessuna colonna aggiuntiva da unire.")
    else:
        print("Avviso: 'Player_ID' non trovato in uno o entrambi i file. Impossibile unire le colonne.")

except FileNotFoundError:
    print(f"Attenzione: File originale non trovato in {ORIGINAL_FILE}. Verranno usati solo i dati del file principale.")
except Exception as e:
    print(f"Errore critico durante l'unione dei file di dati: {e}")

# Normalizza i nomi dei giocatori DOPO aver unito i dati, per coerenza
GLOBAL_DF['Name'] = GLOBAL_DF['Name'].apply(normalize_player_name)

# Converti le colonne numeriche, preservando le stringhe
GLOBAL_DF = convert_numeric_columns(GLOBAL_DF)

def create_profile_selector(profiles, player_data):
    """Crea le card selezionabili per i profili, mostrando solo il nome in spagnolo."""
    cols = []
    for profile in profiles:
        # Traduce il profilo in italiano per accedere ai dati, ma non lo visualizza.
        italian_profile = SPANISH_TO_ITALIAN_PROFILES.get(profile, "")
        
        card_content = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # Mostra solo il titolo in spagnolo.
                    html.H4(profile, 
                           className="card-title",
                           style={
                               'fontSize': '1.2rem',
                               'marginBottom': '1rem',
                               'color': 'white',
                               'textAlign': 'center'
                           }),
                    html.Div([
                        # Score (usa il nome italiano per accedere ai dati)
                        html.H2(
                            f"{player_data.get(f'Score_{italian_profile}', 20):.1f}",
                            style={
                                'fontSize': '2.5rem',
                                'fontWeight': 'bold',
                                'marginBottom': '1rem',
                                'color': '#0082F3',
                                'textAlign': 'center'
                            }
                        ),
                        # Categoria (usa il nome italiano per accedere ai dati)
                        html.Div(
                            player_data.get(f'Category_{italian_profile}', 'N/A'),
                            style={
                                'backgroundColor': get_category_color(
                                    player_data.get(f'Category_{italian_profile}', 'N/A')
                                ),
                                'color': '#000000',
                                'padding': '0.5rem',
                                'borderRadius': '4px',
                                'display': 'inline-block',
                                'marginBottom': '1rem',
                                'fontSize': '0.9rem',
                                'fontWeight': 'bold',
                                'width': '80%',
                                'textAlign': 'center'
                            }
                        )
                    ], style={
                        'textAlign': 'center',
                        'marginBottom': '1rem',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'gap': '0.5rem'
                    }),
                    # Bottone Visualizza
                    dbc.Button(
                        "Ver",
                        id={'type': 'profile-select-btn', 'profile': profile},
                        color="primary",
                        className="mt-2",
                        style={'width': '100%'}
                    )
                ])
            ], 
            className="h-100",
            style={
                'cursor': 'pointer',
                'backgroundColor': '#0A0F26',
                'color': 'white',
                'border': '1px solid rgba(255, 255, 255, 0.2)'
            })
        ], width=12 // len(profiles) if len(profiles) <= 4 else 3)
        cols.append(card_content)
        
    return html.Div([dbc.Row(cols, className="mb-4")])

def create_base_info_card(player_data):
    """Crea la card con le informazioni base del giocatore."""
    try:
        # Formatta l'età come numero intero
        age = int(float(player_data['Edad']))
        
        # Gestisci l'altezza
        height = player_data['Height']
        if pd.notna(height):
            try:
                # Rimuovi 'm' e converti la virgola in punto
                height_str = str(height).lower().replace('m', '').strip().replace(',', '.')
                height_val = float(height_str)
                height = f"{height_val:.2f}m"
            except (ValueError, TypeError):
                height = "N/A"
        else:
            height = "N/A"
        
        # Gestisci squadra e campionato
        squadra = player_data['Squadra']
        if pd.notna(squadra) and str(squadra).lower() not in ['nan', 'none', '']:
            squadra = str(squadra).strip()
        else:
            squadra = "N/A"
        
        lega = player_data['Lega']
        if pd.notna(lega) and str(lega).lower() not in ['nan', 'none', '']:
            lega = str(lega).strip()
        else:
            lega = "N/A"
        
        # Formatta il valore di mercato (in milioni)
        market_value = player_data['Market_Value_M']
        if pd.notna(market_value):
            market_value = f"€{market_value:.1f}M"
        else:
            market_value = "N/A"

        # Gestisci i nuovi campi
        nationality = str(player_data.get('Nationality', 'N/A'))
        signed = str(player_data.get('Signed', 'N/A'))
        
        # Traduci la posizione
        position = player_data.get('Position', 'N/A')
        translated_position = POSITION_TRANSLATION.get(position, position)

        return dbc.Card([
            dbc.CardHeader("Información Básica"),
            dbc.CardBody([
                html.P([html.Strong("Edad: "), f"{age}"]),
                html.P([html.Strong("Posición: "), translated_position]),
                html.P([html.Strong("Equipo: "), squadra]),
                html.P([html.Strong("Liga: "), lega]),
                html.P([html.Strong("Nacionalidad: "), nationality]),
                html.P([html.Strong("Fecha de Fichaje: "), signed]),
                html.P([html.Strong("Pie: "), player_data['Foot']]),
                html.P([html.Strong("Altura: "), height]),
                html.P([html.Strong("Valor de Mercado: "), market_value]),
            ])
        ], className="mb-4")
    except Exception as e:
        print(f"Errore nella creazione della card info base: {str(e)}")
        return dbc.Card([
            dbc.CardHeader("Información Básica"),
            dbc.CardBody([
                html.P("Error al cargar la información")
            ])
        ], className="mb-4")

def create_stats_card(player_data):
    """Crea la card con le statistiche principali."""
    try:
        # Converti i minuti in float, gestendo la virgola come separatore delle migliaia
        minutes = float(str(player_data['Mín']).replace(',', ''))
        
        # Ottieni le statistiche
        pj = player_data.get('PJ', 'N/A')
        ta = player_data.get('TA', 0)
        tr = player_data.get('TR', 0)
        goals = player_data.get('Gls.', 0)
        assists = player_data.get('Ass', 0)
        
        return dbc.Card([
            dbc.CardHeader("Estadísticas Principales"),
            dbc.CardBody([
                html.P([html.Strong("Minutos jugados: "), f"{int(minutes):,}"]),
                html.P([html.Strong("Partidos jugados: "), f"{pj}"]),
                html.P([html.Strong("Goles: "), f"{goals}"]),
                html.P([html.Strong("Asistencias: "), f"{assists}"]),
                html.P([html.Strong("Tarjetas: "), f"🟨 {ta} 🟥 {tr}"]),
            ])
        ], className="mb-4")
    except Exception as e:
        print(f"Errore nella creazione della card statistiche: {str(e)}")
        return dbc.Card([
            dbc.CardHeader("Estadísticas Principales"),
            dbc.CardBody([
                html.P("Error al cargar las estadísticas")
            ])
        ], className="mb-4")

def calculate_similarity_scores(player_data, profile, all_players_df):
    """Calculate similarity scores for a player in a specific profile."""
    try:
        # Get KPIs for the profile. 'profile' is already in Spanish,
        # which is the correct key for PERFILES_KPI_PESOS.
        profile_kpis = PERFILES_KPI_PESOS[profile]
        all_kpis = {**profile_kpis["con_palla"], **profile_kpis["senza_palla"]}
        kpi_list = list(all_kpis.keys())
        
        # Get player's position
        player_position = player_data['Position']
        
        # Filter players by position and minimum minutes (1500)
        min_minutes = convert_to_numeric(1500)  # Converti il valore minimo di minuti
        position_mask = (all_players_df['Position'] == player_position) & (all_players_df['Mín'].apply(convert_to_numeric) >= min_minutes)
        comparable_players = all_players_df[position_mask].copy()
        
        # Skip if no comparable players
        if len(comparable_players) < 2:
            return []
        
        # Create feature vectors for all players
        feature_vectors = []
        player_names = []
        
        for idx, row in comparable_players.iterrows():
            vector = []
            valid_vector = True
            
            for kpi in kpi_list:
                col = next((c for c in row.index if kpi in c), None)
                if col:
                    try:
                        val = convert_to_numeric(row[col])
                        if pd.isna(val) or np.isinf(val):
                            valid_vector = False
                            break
                        vector.append(val)
                    except:
                        valid_vector = False
                        break
                else:
                    valid_vector = False
                    break
            
            if valid_vector and len(vector) == len(kpi_list):
                feature_vectors.append(vector)
                player_names.append(row['Name'])
        
        if not feature_vectors or len(feature_vectors) < 2:
            return []
        
        # Convert to numpy array and normalize
        feature_vectors = np.array(feature_vectors)
        
        # Avoid division by zero in normalization
        min_vals = feature_vectors.min(axis=0)
        max_vals = feature_vectors.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        feature_vectors_normalized = (feature_vectors - min_vals) / range_vals
        
        # Get the index of the target player
        try:
            target_idx = player_names.index(player_data['Name'])
        except ValueError:
            return []
            
        target_vector = feature_vectors_normalized[target_idx]
        
        # Calculate similarities for all players
        similarities = []
        for i, vector in enumerate(feature_vectors_normalized):
            if i != target_idx:  # Skip comparing player with themselves
                try:
                    # Calculate Euclidean similarity (convert distance to similarity)
                    eucl_dist = euclidean(target_vector, vector)
                    eucl_sim = 1 / (1 + eucl_dist)
                    
                    # Calculate Cosine similarity
                    # Add small epsilon to avoid zero division
                    vector_with_eps = vector + 1e-10
                    target_with_eps = target_vector + 1e-10
                    cos_sim = 1 - cosine(target_with_eps, vector_with_eps)
                    
                    # Ensure similarities are valid numbers
                    if not np.isnan(eucl_sim) and not np.isnan(cos_sim):
                        # Combine similarities with weights
                        combined_sim = (0.15 * eucl_sim) + (0.85 * cos_sim)
                        
                        # Get current team from the original dataframe
                        player_row = comparable_players[comparable_players['Name'] == player_names[i]].iloc[0]
                        
                        similarities.append({
                            'name': player_names[i],
                            'similarity': combined_sim,
                            'team': player_row['Squadra'],
                            'league': player_row['Lega']
                        })
                except Exception as e:
                    print(f"Errore nel calcolo della similarità per {player_names[i]}: {str(e)}")
                    continue
        
        # Sort by similarity and get top 8
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:8]
        
    except Exception as e:
        print(f"Errore generale nel calcolo delle similarità: {str(e)}")
        return []

def convert_minutes(min_str):
    """Converte correttamente i minuti gestendo la virgola come separatore delle migliaia."""
    try:
        if isinstance(min_str, (int, float)):
            return float(min_str)
        return float(str(min_str).replace(',', ''))
    except:
        return 0.0

def normalize_kpi_with_coefficients(value, kpi, player_info, all_players_df, position):
    """Normalizza un KPI considerando i coefficienti delle leghe e i minuti giocati."""
    try:
        # Filtra i giocatori per posizione e minuti minimi (1500)
        min_minutes = 1500
        # Converti i minuti in modo sicuro
        minutes = all_players_df['Mín'].apply(convert_minutes)
        position_mask = (all_players_df['Position'] == position) & (minutes >= min_minutes)
        comparable_players = all_players_df[position_mask]
        
        if len(comparable_players) == 0:
            return 50  # Valore di default se non ci sono giocatori comparabili
        
        # Trova la colonna corrispondente al KPI
        col = next((c for c in comparable_players.columns if kpi in c), None)
        if not col:
            return 50
        
        # Converti i valori in float e applica i coefficienti delle leghe
        values = pd.to_numeric(comparable_players[col].astype(str).str.replace(',', ''), errors='coerce')
        coeff_values = values * comparable_players['Coeff Lega']
        
        # Gestisci i valori NaN
        coeff_values = coeff_values.fillna(coeff_values.mean())
        
        # Calcola min e max dei valori con coefficienti
        min_val = coeff_values.min()
        max_val = coeff_values.max()
        
        if max_val == min_val:
            return 50
        
        # Applica il coefficiente al valore del giocatore
        player_value = pd.to_numeric(str(player_info[col]).replace(',', ''), errors='coerce')
        if pd.isna(player_value):
            return 1
            
        player_league_coeff = player_info['Coeff Lega']
        value_with_coeff = player_value * player_league_coeff
        
        # Normalizza da 1 a 99 per mostrare meglio le differenze relative
        normalized = ((value_with_coeff - min_val) * 98 / (max_val - min_val) + 1).round(1)
        
        # Assicurati che il valore sia tra 1 e 99
        return min(max(normalized, 1), 99)
    except Exception as e:
        print(f"Errore nella normalizzazione del KPI {kpi}: {str(e)}")
        return 50

def create_radar_chart(player_info, profile, comparison_player=None):
    """Crea il radar chart con settori sovrapposti per il confronto e una legenda pulita."""
    try:
        # Usa il profilo in spagnolo per trovare i kpi corretti
        profile_kpis = PERFILES_KPI_PESOS[profile]
        all_kpis = {**profile_kpis.get("con_palla", {}), **profile_kpis.get("senza_palla", {})}
        
        def get_kpi_values(p_info):
            values = {}
            for kpi in all_kpis.keys():
                col = next((c for c in p_info.index if kpi in c), None)
                if col:
                    try:
                        normalized_val = normalize_kpi_with_coefficients(p_info[col], kpi, p_info, GLOBAL_DF, p_info['Position'])
                        values[kpi] = normalized_val
                    except Exception: values[kpi] = 1
                else: values[kpi] = 1
            return values

        kpi_values = get_kpi_values(player_info)
        fig = go.Figure()
        
        kpis = list(kpi_values.keys())
        n_metrics = len(kpis)
        sector_angle = 360 / n_metrics
        angles = [(i * sector_angle) for i in range(n_metrics)]
        
        # Disegna i settori per il giocatore principale
        main_values = list(kpi_values.values())
        main_player_name = player_info['Name']
        for i, (kpi, value) in enumerate(zip(kpis, main_values)):
            angle = angles[i]
            opacity = max(value/100, 0.4)
            color = f'rgba(0, 130, 243, {opacity})'
            theta_sector = np.linspace(angle - sector_angle*0.4, angle + sector_angle*0.4, 20)
            r_sector = np.array([value] * 20)
            fig.add_trace(go.Scatterpolar(
                r=np.concatenate([[0], r_sector, [0]]),
                theta=np.concatenate([[angle], theta_sector, [angle]]),
                mode='lines', fill='toself', fillcolor=color, line=dict(width=0),
                name=main_player_name,
                legendgroup=main_player_name,
                showlegend=(i == 0),
                hovertemplate=f"{kpi}: {value:.1f}<extra></extra>"
            ))

        # Disegna i settori SOVRAPPOSTI per il giocatore di confronto
        if comparison_player is not None:
            comp_kpi_values = get_kpi_values(comparison_player)
            comp_values = [comp_kpi_values.get(k, 1) for k in kpis]
            comp_player_name = comparison_player['Name']
            for i, (kpi, value) in enumerate(zip(kpis, comp_values)):
                angle = angles[i]
                color = 'rgba(255, 87, 51, 0.6)'
                theta_sector = np.linspace(angle - sector_angle*0.4, angle + sector_angle*0.4, 20)
                r_sector = np.array([value] * 20)
                fig.add_trace(go.Scatterpolar(
                    r=np.concatenate([[0], r_sector, [0]]),
                    theta=np.concatenate([[angle], theta_sector, [angle]]),
                    mode='lines',
                    fill='toself',
                    fillcolor=color,
                    line=dict(width=0),
                    name=comp_player_name,
                    legendgroup=comp_player_name,
                    showlegend=(i == 0),
                    hovertemplate=f"{kpi}: {value:.1f}<extra></extra>"
                ))

        # Layout del grafico
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 100]),
                angularaxis=dict(
                    tickmode='array', tickvals=angles, ticktext=[f'<b>{k}</b>' for k in kpis],
                    tickfont=dict(color='white', size=10), gridcolor="rgba(255, 255, 255, 0.1)"
                ),
                bgcolor='#0A0F26'
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color='white')),
            paper_bgcolor='#0A0F26', plot_bgcolor='#0A0F26',
            title=dict(text=f"Radar: {profile}", y=0.95, x=0.5, xanchor='center', yanchor='top', font=dict(color='white'))
        )
        return fig
    except Exception as e:
        print(f"Errore nella creazione del radar chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return go.Figure()

def create_similar_players_card(similar_players, profile, player_name, selected_player=None):
    """Crea la card con i giocatori simili, senza percentuale."""
    if not similar_players:
        return dbc.Card(dbc.CardBody("Nessun giocatore simile trovato."))
        
    return dbc.Card([
        dbc.CardHeader("Jugadores Similares"),
        dbc.CardBody([
            html.Div([
                html.P([
                    html.Strong(f"{i+1}. "),
                    html.Button(
                        player['name'],
                        id={'type': 'compare-player-btn', 'profile': profile, 'player': player['name'], 'index': i},
                        className="btn btn-link p-0",
                        style={
                            'border': 'none', 'background': 'none', 'cursor': 'pointer',
                            'color': '#0082F3' if player['name'] == selected_player else 'white',
                            'textDecoration': 'none'
                        }
                    ),
                    # Rimuoviamo la percentuale di similarità dalla visualizzazione
                    f" ({player['team']} - {player['league']})"
                ]) for i, player in enumerate(similar_players)
            ])
        ])
    ], className="mb-3")

def get_category_color(category):
    """Restituisce il colore appropriato per ogni categoria."""
    colors = {
        "Estrella Mundial": "#FFD700",      # Oro
        "Jugador Top": "#40E0D0",           # Turchese
        "Buen Jugador": "#98FB98",          # Verde chiaro
        "Jugador de Media Tabla": "#87CEEB", # Azzurro
        "Jugador de Baja Tabla": "#B0C4DE",  # Blu chiaro
        "Jugador de División 2": "#DEB887",  # Beige
        "Joven Estrella Mundial": "#FFD700", # Oro
        "Futura Estrella": "#40E0D0",       # Turchese
        "Joven con Calidad": "#98FB98",     # Verde chiaro
        "Buen Joven": "#87CEEB",            # Azzurro
        "Joven": "#B0C4DE",                 # Blu chiaro
        "N/A": "#808080"                     # Grigio
    }
    return colors.get(category, "#808080")

def create_kpi_table(player_name, profile):
    """Crea la tabella KPI con ranking condizionale per campionati Top 5 o altri."""
    try:
        # Definisco le leghe Top 5
        TOP_5_LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
        
        # Ottengo le informazioni del giocatore
        player_info = GLOBAL_DF[GLOBAL_DF['Name'] == player_name].iloc[0]
        player_position = player_info['Position']
        player_league = player_info['Lega']
        translated_position = POSITION_TRANSLATION.get(player_position, player_position)

        # Determino se il giocatore è in una lega Top 5
        is_top_5_player = player_league in TOP_5_LEAGUES

        # Filtro i giocatori in base alla nuova logica di ranking
        if is_top_5_player:
            league_mask = GLOBAL_DF['Lega'].isin(TOP_5_LEAGUES)
            ranking_group_text = "las ligas Top 5"
        else:
            league_mask = ~GLOBAL_DF['Lega'].isin(TOP_5_LEAGUES)
            ranking_group_text = "otras ligas"
        
        filtered_df = GLOBAL_DF[
            (GLOBAL_DF['Mín'].apply(convert_to_numeric) >= 1500) & 
            (GLOBAL_DF['Position'] == player_position) &
            (league_mask)
        ].copy()

        if filtered_df.empty:
            return html.Div("No hay suficientes datos para generar el ranking.")
        
        # Ottieni i KPI dal profilo selezionato
        profile_kpis = PERFILES_KPI_PESOS[profile]
        all_kpis = {**profile_kpis["con_palla"], **profile_kpis["senza_palla"]}
        kpi_list = list(all_kpis.keys())
        
        # Lista delle statistiche che hanno versione per 90 minuti
        stats_per_90 = [
            'Tkl', 'TklG', '3.º def.', '3.º cent.', '3.º ataq.', 'Bloqueos', 'PasesB', 'Int',
            'Desp.', 'Fls', 'Recup.', 'PrgC', 'PrgP', 'PrgR', 'Dist. tot.', 'Dist. prg.',
            'PC', 'P1/3', 'CrAP', 'TAtaq. pen.', 'Succ', 'Camb.', 'PPA', 'C1/3', 'PL',
            'Att', 'TAP', 'T3.º ataq.', 'Toques'
        ]
        
        # Crea la tabella
        table_header = [
            html.Thead(html.Tr([
                html.Th("KPI", style={'width': '25%', 'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Ranking", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Value", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("", style={'width': '35%', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Min / Max", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'}),
                html.Th("Position Average", style={'width': '10%', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#1a1a2e'})
            ], style={'backgroundColor': '#1a1a2e'}))
        ]
        
        rows = []
        for kpi in kpi_list:
            # Determina se usare la versione per 90 minuti
            base_kpi = kpi.replace(" per 90s", "")  # Rimuovi il suffisso se presente
            use_per_90 = base_kpi in stats_per_90
            
            # Cerca la colonna corrispondente
            if use_per_90:
                col = next((c for c in filtered_df.columns if f"{base_kpi} per 90s" in c), None)
                if not col:  # Se non trova la versione per 90, usa quella normale
                    col = next((c for c in filtered_df.columns if base_kpi in c), None)
            else:
                col = next((c for c in filtered_df.columns if kpi in c), None)
                
            if col:
                try:
                    # Converti i valori in numeri
                    values = filtered_df[col].astype(str).str.replace(',', '').astype(float)
                    
                    # Calcola ranking
                    filtered_df['rank'] = values.rank(ascending=False, method='min')
                    player_stats = filtered_df[filtered_df['Name'] == player_name].iloc[0]
                    ranking = int(player_stats['rank'])
                    total_players = len(filtered_df)  # Numero totale di giocatori confrontabili
                    
                    # Calcola statistiche
                    value = float(str(player_stats[col]).replace(',', ''))
                    min_val = values.min()
                    max_val = values.max()
                    avg_val = values.mean()
                    
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
                    
                    # Mostra il nome della statistica con indicazione se è per 90 minuti
                    display_name = f"{base_kpi} (per 90')" if use_per_90 else kpi
                    
                    # Aggiungi la riga alla tabella con solo la posizione nel ranking
                    row = html.Tr([
                        html.Td(display_name, style={'textAlign': 'left', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                        html.Td(f"#{ranking}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                        html.Td(f"{value:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                        html.Td(html.Div(style=bar_style), style={'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                        html.Td(f"{min_val:.2f} / {max_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                        html.Td(f"{avg_val:.2f}", style={'textAlign': 'center', 'padding': '10px', 'borderBottom': '1px solid rgba(255,255,255,0.1)'})
                    ], style={'backgroundColor': '#0A0F26'})
                    rows.append(row)
                    
                except Exception as e:
                    print(f"Errore per KPI {kpi}: {str(e)}")
                    continue
        
        return html.Div([
            html.H3(f"Estadísticas y Ranking", 
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
            ),
            # Legenda sotto la tabella (tradotta e con logica aggiornata)
            html.Div(
                f"* Ranking basado en {total_players} jugadores con más de 1500 minutos en la posición de '{translated_position}' en {ranking_group_text}",
                style={
                    'color': 'rgba(255, 255, 255, 0.7)',
                    'fontSize': '12px',
                    'marginTop': '10px',
                    'textAlign': 'left',
                    'fontStyle': 'italic'
                }
            )
        ], style={
            'backgroundColor': '#0A0F26',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
        })
        
    except Exception as e:
        print(f"Errore nella creazione della tabella KPI: {str(e)}")
        return html.Div("Errore nel caricamento della tabella KPI")

def create_radar_container():
    """Crea il contenitore per il radar e la barra di ricerca, con testo in spagnolo."""
    return html.Div([
        # Barra di ricerca (sempre presente)
        html.Div([
            html.H4("Comparar con otros jugadores", className="mb-3", style={"color": "#0082F3"}),
            html.P("Selecciona un perfil para ver jugadores comparables",
                   style={"fontSize": "0.9em", "color": "rgba(255, 255, 255, 0.7)"}),
            dcc.Dropdown(
                id='player-search',
                options=[],
                placeholder="Buscar un jugador...",
                style={
                    'backgroundColor': '#1a1a2e',
                    'color': 'white',
                    'border': '1px solid rgba(255, 255, 255, 0.2)',
                }
            ),
        ], className="mb-4"),
        
        # Container per il radar e i giocatori simili
        html.Div(id='radar-and-similar-container')
    ])

def calculate_kpi_scores(df):
    """Calcola i punteggi per ogni profilo e li aggiunge al DataFrame."""
    # Uso i profili italiani per il calcolo
    for profile_name, kpis in PERFILES_KPI_PESOS.items():
        # ... (logica di calcolo invariata)
        # ... (usa role_profiles_it al suo interno)
        compatible_roles = [role for role, profiles in ROLE_PROFILES_ES.items() if profile_name in profiles]
        # ...
    return df

def layout(player_name=None):
    """
    Layout function for the player profile page.
    player_name: URL parameter containing the encoded player name
    """
    if player_name is None:
        return html.Div("Giocatore non specificato", style={"color": "white", "padding": "20px"})
    
    try:
        # Decode the URL-encoded player name
        decoded_name = unquote(player_name)
        normalized_name = normalize_player_name(decoded_name)
        
        # Find the player in the dataset
        player_data = GLOBAL_DF[GLOBAL_DF['Name'] == normalized_name]
        if len(player_data) == 0:
            return html.Div(f"Giocatore '{decoded_name}' non trovato", style={"color": "white", "padding": "20px"})
        
        player_data = player_data.iloc[0]
        
        # Debug: stampa le categorie per questo giocatore
        print(f"\nCategorie per {normalized_name}:")
        for col in GLOBAL_DF.columns:
            if 'Category_' in col:
                print(f"{col}: {player_data[col]}")
        
        player_position = player_data['Position']
        possible_profiles = ROLE_PROFILES_ES.get(player_position, [])
        
        return dbc.Container([
            # Store per i dati del giocatore
            dcc.Store(id='player-data-store', data={
                'name': normalized_name,
                'position': player_position,
                'profiles': possible_profiles
            }),
            
            # Store per il profilo selezionato
            dcc.Store(id='selected-profile-store'),
            
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(decoded_name, className="text-center my-4", style={"color": "#0082F3"}),
                ], width=12)
            ]),
            
            # Info Cards in una riga
            dbc.Row([
                dbc.Col([create_base_info_card(player_data)], width=6),
                dbc.Col([create_stats_card(player_data)], width=6),
            ], className="mb-4"),
            
            # Profile selector cards with scores
            create_profile_selector(possible_profiles, player_data),
            
            # Container for radar, search, and similar players
            create_radar_container(),
            
            # Back button
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "← Torna allo Scouting",
                        href="/scouting",
                        color="primary",
                        className="mt-3"
                    )
                ], width=12)
            ])
        ], fluid=True, style={"backgroundColor": "#0A0F26", "color": "#FFFFFF", "padding": "20px"})
    
    except Exception as e:
        print(f"Errore nel caricamento del profilo: {str(e)}")
        return html.Div([
            html.H1("Errore nel caricamento del profilo", style={"color": "red"}),
            html.P(str(e)),
            dbc.Button(
                "← Torna allo Scouting",
                href="/scouting",
                color="primary",
                className="mt-3"
            )
        ], style={"color": "white", "padding": "20px"})

@callback(
    Output('selected-profile-store', 'data'),
    [Input({'type': 'profile-select-btn', 'profile': ALL}, 'n_clicks')],
    [State({'type': 'profile-select-btn', 'profile': ALL}, 'id')]
)
def update_selected_profile(n_clicks, btn_ids):
    if not any(n_clicks):
        return None
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    profile = eval(button_id)['profile']
    return profile

@callback(
    Output('radar-and-similar-container', 'children'),
    [Input('player-data-store', 'data'),
     Input('selected-profile-store', 'data'),
     Input({'type': 'compare-player-btn', 'profile': ALL, 'player': ALL, 'index': ALL}, 'n_clicks'),
     Input('player-search', 'value')],
    [State({'type': 'compare-player-btn', 'profile': ALL, 'player': ALL, 'index': ALL}, 'id')]
)
def update_radar_and_similar(player_data, selected_profile, n_clicks, search_value, btn_ids):
    if not player_data or not selected_profile:
        return []
    
    ctx = dash.callback_context
    triggered = ctx.triggered[0] if ctx.triggered else None
    
    player_name = player_data['name']
    player_info = GLOBAL_DF[GLOBAL_DF['Name'] == player_name].iloc[0]
    
    # Determina se è stato selezionato un giocatore dalla ricerca
    comparison_player_name = None
    if triggered and triggered['prop_id'] == 'player-search.value' and search_value:
        comparison_player_name = search_value
    # Altrimenti controlla se è stato cliccato un bottone di confronto
    elif triggered and 'prop_id' in triggered and '.n_clicks' in triggered['prop_id']:
        clicked_id = eval(triggered['prop_id'].split('.')[0])
        clicked_player = clicked_id['player']
        comparison_player_name = clicked_player
    
    try:
        # Se un giocatore è stato selezionato per il confronto, crea il grafico con entrambi
        if comparison_player_name and comparison_player_name != player_name:
            comparison_player_data = GLOBAL_DF[GLOBAL_DF['Name'] == comparison_player_name]
            if not comparison_player_data.empty:
                # CORREZIONE: Passa la Series (.iloc[0]) e non il DataFrame
                radar_chart_fig = create_radar_chart(player_info, selected_profile, comparison_player_data.iloc[0])
        else:
            # Altrimenti, crea il grafico solo per il giocatore principale
            radar_chart_fig = create_radar_chart(player_info, selected_profile)
        
        # Calcola i giocatori simili
        similar_players = calculate_similarity_scores(player_info, selected_profile, GLOBAL_DF)
        
        return [
            # Layout con radar a sinistra e giocatori simili a destra
            dbc.Row([
                # Radar chart (6 colonne)
                dbc.Col([
                    dcc.Graph(figure=radar_chart_fig, config={'displayModeBar': False})
                ], width=6),
                
                # Giocatori simili (6 colonne)
                dbc.Col([
                    create_similar_players_card(
                        similar_players, 
                        selected_profile, 
                        player_name,
                        selected_player=comparison_player_name
                    )
                ], width=6)
            ]),
            
            # Tabella KPI con titolo tradotto
            dbc.Row([
                dbc.Col([
                    html.H3("Estadísticas y Ranking", className="mb-3", style={"color": "#0082F3"}),
                    create_kpi_table(player_name, selected_profile)
                ], width=12)
            ])
        ]
        
    except Exception as e:
        print(f"Errore nel creare il radar per {selected_profile}: {str(e)}")
        return html.Div(f"Errore nel caricamento del profilo {selected_profile}")

@callback(
    Output('player-search', 'style'),
    [Input('player-search', 'value')]
)
def update_dropdown_style(value):
    """Aggiorna lo stile del dropdown quando è selezionato un valore."""
    base_style = {
        'backgroundColor': '#1a1a2e',
        'color': 'white',
        'border': '1px solid rgba(255, 255, 255, 0.2)',
    }
    if value:
        base_style.update({
            'backgroundColor': '#0A0F26',
            'border': '1px solid #0082F3',
        })
    return base_style

@callback(
    Output('player-search', 'options'),
    [Input('selected-profile-store', 'data')]
)
def update_search_options(selected_profile):
    """Aggiorna le opzioni della barra di ricerca in base al profilo selezionato."""
    if not selected_profile:
        return []
    
    # Trova tutti i ruoli che possono utilizzare questo profilo
    compatible_roles = [role for role, profiles in ROLE_PROFILES_ES.items() if selected_profile in profiles]
    
    if not compatible_roles:
        return []
    
    # Filtra i giocatori per ruoli compatibili e minuti minimi (1500)
    players_df = GLOBAL_DF[
        (GLOBAL_DF['Position'].isin(compatible_roles)) & 
        (GLOBAL_DF['Mín'].apply(convert_to_numeric) >= 1500)
    ]
    
    # Crea le opzioni con nome, squadra e lega
    options = [
        {'label': f"{row['Name']} ({row['Squadra']} - {row['Lega']})", 'value': row['Name']}
        for _, row in players_df.iterrows()
    ]
    
    return sorted(options, key=lambda x: x['label'])

@callback(
    [Output('player-profile-content', 'children'),
     Output('store-selected-player', 'data')],
    [Input('url', 'pathname'),
     Input('profile-selector', 'value'),
     Input('similar-players-table', 'active_cell'),
     State('store-selected-player', 'data'),
     State('similar-players-table', 'data')]
)
def update_player_profile(pathname, selected_profile, active_cell, stored_data, table_data):
    print(f"\n\n--- DEBUG DEFINITIVO: Il callback update_player_profile è stato chiamato! ---")
    print(f"INPUT - pathname: {pathname}")
    print(f"INPUT - selected_profile: {selected_profile}")
    print(f"--------------------------------------------------------------------------\n")

    player_name = unquote(pathname.split('/')[-1])

    if not player_name or player_name == "player":
        return [html.Div("Seleziona un giocatore per visualizzare il profilo.")], dash.no_update
        
    try:
        player_info = GLOBAL_DF[GLOBAL_DF['Name'] == player_name].iloc[0]
    except IndexError:
        print(f"\n--- DEBUG (player_profile.py): ERRORE LETALE ---")
        print(f"Non sono riuscito a trovare il giocatore '{player_name}' nel DataFrame principale.")
        print(f"Controllare se il nome è presente e se la normalizzazione è corretta.\n")
        return [html.Div(f"Giocatore '{player_name}' non trovato nel database.")], dash.no_update

    profile_kpis = PERFILES_KPI_PESOS[selected_profile]
    all_kpis = {**profile_kpis["con_palla"], **profile_kpis["senza_palla"]}
    
    print("\n--- DEBUG (player_profile.py): Sto per chiamare find_similar_players ---")
    similar_players_df = calculate_similarity_scores(player_info, selected_profile, GLOBAL_DF)
    
    comparison_player_name = None
    if active_cell and table_data and active_cell['row'] < len(table_data):
        comparison_player_name = table_data[active_cell['row']]['Name']

    try:
        fig = create_radar_chart(player_info, selected_profile, comparison_player_name)
        
        similar_players_card = create_similar_players_card(
            similar_players_df, 
            selected_profile, 
            player_name, 
            selected_player=comparison_player_name
        )
        
        player_card = create_base_info_card(player_info)
        kpi_table = create_kpi_table(player_name, selected_profile)
        
        layout = dbc.Container([
            dbc.Row([
                dbc.Col(player_card, width=12, md=4),
                dbc.Col(dcc.Graph(figure=fig, config={'displayModeBar': False}), width=12, md=8)
            ]),
            dbc.Row([
                dbc.Col(similar_players_card, width=12, md=4),
                dbc.Col(kpi_table, width=12, md=8)
            ])
        ], fluid=True)

        new_store_data = {'player': player_name, 'profile': selected_profile}
        return [layout], new_store_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [html.Div(f"Si è verificato un errore nell'aggiornamento: {e}")], dash.no_update