import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import os
import urllib.parse
import unicodedata

# Registra la pagina
dash.register_page(__name__, path_template="/team/<league>/<team>/plantilla")

# Percorsi dei file
DATA_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/datos/fbref/base_datos_tutte_le_leghe_renamed.csv")
SCOUTING_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/datos/fbref/scouting_scores.csv")

# Dizionario per tradurre gli stili di gioco dall'italiano allo spagnolo
STYLE_TRANSLATIONS = {
    'Combinativo con blocco alto': 'Combinativo con presión alta',
    'Diretto con blocco alto': 'Directo con presión alta',
    'Combinativo con blocco basso': 'Combinativo con presión baja',
    'Diretto con blocco basso': 'Directo con presión baja',
    'Misto': 'Mixto'
}

# Dizionario per tradurre i profili dall'italiano allo spagnolo
PROFILE_TRANSLATIONS = {
    'Centrale - Marcatura': 'Marcador',
    'Centrale - Deep Distributor': 'Central de salida',
    'Recuperatore 20-80': 'Recuperador rapido',
    'Sentinel Fullback 20-80': 'Lateral defensivo',
    'Quinto ATA 70-30': 'Carrilero de recorrido',
    'Arnold 80-20': 'Lateral tecnico y de salida',
    'Difensivo': 'Pivote defensivo',
    'Play': 'Play',
    'Box-to-Box': 'Box to box',
    'Dieci': 'Diez',
    'Interior 90-10': 'Extremo interior',
    '1vs1': 'Extremo de 1vs1',
    'Mobile Forward': 'Delantero movil',
    'Target': 'Delantero Referencia',
    'Lethal Striker': 'Delantero de area'
}

# Dizionario per tradurre le posizioni dall'inglese allo spagnolo
POSITION_TRANSLATIONS = {
    'Goalkeeper': 'Portero',
    'Centre-Back': 'Defensa Central',
    'Left-Back': 'Lateral Izquierdo',
    'Right-Back': 'Lateral Derecho',
    'Defensive Midfield': 'Pivote defensivo',
    'Central Midfield': 'Mediocentro',
    'Midfielder': 'Centrocampista',
    'Attacking Midfield': 'Centrocampista ofensivo',
    'Left Midfield': 'Carrilero Izquierdo',
    'Right Midfield': 'Carrilero Derecho',
    'Left Winger': 'Extremo Izquierdo',
    'Right Winger': 'Extremo Derecho',
    'Second Striker': 'Segundo Delantero',
    'Centre-Forward': 'Delantero'
}

def translate_profile(profile):
    """Traduce il profilo in spagnolo"""
    if pd.isna(profile):
        return "Perfil no determinado"
    
    profile_str = str(profile).strip()
    return PROFILE_TRANSLATIONS.get(profile_str, profile_str)

def translate_position(position):
    """Traduce la posizione in spagnolo"""
    if pd.isna(position):
        return "Posición no determinada"
    
    position_str = str(position).strip()
    return POSITION_TRANSLATIONS.get(position_str, position_str)

def get_team_style(team_name):
    """Legge lo stile di gioco della squadra e lo traduce in spagnolo"""
    try:
        # Mappa delle leghe
        league_mapping = {
            'Serie A': 'Serie A',
            'Premier League': 'Premier League',
            'La Liga': 'La Liga',
            'Bundesliga': 'Bundesliga',
            'Ligue 1': 'Ligue 1'
        }
        
        # Cerca nelle cartelle delle leghe
        for league_name, wyscout_name in league_mapping.items():
            file_path = os.path.join("datos/wyscout/2024-2025", wyscout_name, f"build-up_{wyscout_name}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Equipo'] = df['Equipo'].str.replace(r'-\d+$', '', regex=True)
                
                team_row = df[df['Equipo'] == team_name]
                if not team_row.empty and 'Estilo' in df.columns:
                    style = team_row['Estilo'].iloc[0]
                    # Traduci lo stile dall'italiano allo spagnolo
                    return STYLE_TRANSLATIONS.get(style, "Estilo no determinado")
                    
        return "Estilo no determinado"
            
    except Exception as e:
        print(f"Error al recuperar el estilo: {e}")
        return "Estilo no determinado"

def get_profiles_for_position(position):
    """Restituisce i profili disponibili per una posizione"""
    position = str(position).strip()
    
    profiles = {
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
    
    # Cerca una corrispondenza esatta
    if position in profiles:
        return [translate_profile(p) for p in profiles[position]]
        
    # Se non troviamo una corrispondenza esatta, proviamo a trovare una corrispondenza parziale
    for key, value in profiles.items():
        if key.lower() in position.lower():
            return [translate_profile(p) for p in value]
            
    return []

def clean_player_name(name):
    """Pulisce il nome del giocatore da caratteri speciali"""
    try:
        name = str(name).replace('-', ' ')
        
        special_chars = {
            'š': 's', 'č': 'c', 'ć': 'c', 'đ': 'd', 'ž': 'z',
            'Š': 'S', 'Č': 'C', 'Ć': 'C', 'Đ': 'D', 'Ž': 'Z',
            'ı': 'i', 'ğ': 'g', 'ş': 's', 'ö': 'o', 'ü': 'u',
            'ñ': 'n', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o',
            'ú': 'u', 'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O',
            'Ú': 'U', 'Å': 'a', 'å': 'a'
        }
        
        normalized = unicodedata.normalize('NFKD', name)
        for special, normal in special_chars.items():
            normalized = normalized.replace(special, normal)
            
        cleaned = ''.join(c for c in normalized if c.isascii() or c.isspace())
        return ' '.join(cleaned.split())
    except Exception as e:
        print(f"Errore nella pulizia del nome: {e}")
        return str(name)

def get_player_scores(player_data):
    """Recupera i punteggi dei profili di un giocatore"""
    try:
        scores = {}
        for col in player_data.index:
            if col.startswith('Score_'):
                profile = col[6:]  # Rimuove 'Score_'
                score = player_data[col]
                if pd.notna(score):
                    scores[profile] = round(float(score), 1)
        return scores
            
    except Exception as e:
        print(f"Errore nel recupero dei punteggi: {e}")
        return {}

def get_best_profile_for_player(player_name):
    """Recupera il miglior profilo per un giocatore dal file degli score"""
    try:
        if not os.path.exists(SCOUTING_FILE):
            return "N/A"
            
        scores_df = pd.read_csv(SCOUTING_FILE, encoding='utf-8')
        row = scores_df[scores_df['Name'] == player_name]
        if row.empty:
            return "N/A"
        row = row.iloc[0]
        best_profile = ""
        best_score = 0
        for col in row.index:
            if col.startswith('Score_'):
                profile = col[6:]
                score = row[col]
                if pd.notna(score) and float(score) > best_score:
                    best_score = float(score)
                    best_profile = profile
        # Traduci il profilo prima di restituirlo
        return translate_profile(best_profile) if best_profile else "N/A"
    except Exception as e:
        print(f"Error al obtener mejor perfil: {e}")
        return "N/A"

def get_squad_data(team_name):
    """Recupera i dati della rosa della squadra"""
    try:
        if not os.path.exists(DATA_FILE):
            return pd.DataFrame()
            
        # Leggi il file
        try:
            df = pd.read_csv(DATA_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_FILE, encoding='latin1')
            
        # Filtra per squadra
        squad_df = df[df['Squadra'] == team_name].copy()
        if squad_df.empty:
            return pd.DataFrame()
            
        # Seleziona e rinomina le colonne
        columns = {
            'Name': 'Name',
            'Position': 'Position',
            'Edad': 'Edad',
            'Mín': 'Min',
            'Gls.': 'Gls',
            'Ass': 'Ass',
            'Contract Until': 'Contract Until'
        }
        
        # Assicurati che tutte le colonne richieste esistano
        available_columns = [col for col in columns.keys() if col in df.columns]
        squad_df = squad_df[available_columns].rename(columns={col: columns[col] for col in available_columns})
        
        # Converti i dati numerici
        squad_df['Min'] = pd.to_numeric(squad_df['Min'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        squad_df['Gls'] = pd.to_numeric(squad_df['Gls'], errors='coerce').fillna(0)
        squad_df['Ass'] = pd.to_numeric(squad_df['Ass'], errors='coerce').fillna(0)
        
        # Rimuovi duplicati
        squad_df = squad_df.dropna(subset=['Name']).drop_duplicates(subset=['Name'])
        
        # Aggiungi il gruppo di ruolo
        def get_role_group(pos):
            pos = str(pos).upper()
            if 'GOALKEEPER' in pos or 'GK' in pos:
                return 'GK'
            elif 'BACK' in pos or 'DEFENDER' in pos:
                return 'DF'
            elif 'MIDFIELD' in pos:
                return 'MF'
            elif any(x in pos for x in ['FORWARD', 'STRIKER', 'WINGER']):
                return 'FW'
            return 'Other'
            
        squad_df['RoleGroup'] = squad_df['Position'].apply(get_role_group)
        
        # Ordina per ruolo e minuti
        role_order = {'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3, 'Other': 4}
        squad_df['RoleOrder'] = squad_df['RoleGroup'].map(role_order)
        squad_df = squad_df.sort_values(['RoleOrder', 'Min'], ascending=[True, False])
        
        return squad_df
        
    except Exception as e:
        print(f"Errore nel recupero della rosa: {e}")
        return pd.DataFrame()

def get_age_from_string(age_str):
    """Extrae los años de una cadena en formato numérico"""
    try:
        if pd.isna(age_str):
            return "N/A"
        
        # Converte il valore in float e poi prende solo la parte intera
        age_float = float(age_str)
        return str(int(age_float))
    except Exception as e:
        print(f"Error extracting age from {age_str}: {e}")
        return "N/A"

def create_squad_table(df):
    """Crea la tabla de la plantilla"""
    if df is None or df.empty:
        return html.Div("Datos de la plantilla no disponibles", 
                        style={"color": "white", "textAlign": "center"})
        
    try:
        # Print column names and a sample of Edad values
        print("DataFrame columns:", df.columns.tolist())  # Debug print
        print("\nSample of Edad values:")  # Debug print
        if 'Edad' in df.columns:
            print(df['Edad'].head())  # Debug print
        
        # Group players by role
        role_groups = {
            'GK': 'Porteros',
            'DF': 'Defensas',
            'MF': 'Centrocampistas',
            'FW': 'Delanteros'
        }
        
        # Create tables for each role group
        tables = []
        for role_code, role_name in role_groups.items():
            role_df = df[df['RoleGroup'] == role_code]
            if not role_df.empty:
                # Add role group header
                tables.append(html.H3(role_name, style={"color": "white", "marginTop": "20px", "marginBottom": "10px"}))
                
                # Header della tabella
                header = html.Thead([
                    html.Tr([
                        html.Th("Nombre", style={"width": "20%", "textAlign": "left", "fontSize": "12px"}),
                        html.Th("Posición", style={"width": "15%", "textAlign": "center", "fontSize": "12px"}),
                        html.Th("Edad", style={"width": "8%", "textAlign": "center", "fontSize": "12px"}),
                        html.Th("Min", style={"width": "8%", "textAlign": "center", "fontSize": "12px"}),
                        html.Th("Goles", style={"width": "8%", "textAlign": "center", "fontSize": "12px"}),
                        html.Th("Asistencias", style={"width": "8%", "textAlign": "center", "fontSize": "12px"}),
                        html.Th("Fin de Contrato", style={"width": "12%", "textAlign": "center", "fontSize": "12px"}),
                        html.Th("Mejor Perfil", style={"width": "19%", "textAlign": "center", "fontSize": "12px"}),
                        html.Th("", style={"width": "10%", "textAlign": "center", "fontSize": "12px"})
                    ], style={"backgroundColor": "#1a1a2e"})
                ])
                
                rows = []
                for _, player in role_df.iterrows():
                    # Dati del giocatore
                    player_name = str(player['Name']) if pd.notna(player['Name']) else ""
                    position = translate_position(str(player['Position'])) if pd.notna(player['Position']) else ""
                    edad = get_age_from_string(player.get('Edad'))
                    contract_until = str(player['Contract Until']) if pd.notna(player.get('Contract Until')) else "N/A"
                    
                    # Miglior Profilo (usando il file degli score)
                    best_profile = get_best_profile_for_player(player_name)
                    
                    # Riga del giocatore
                    rows.append(html.Tr([
                        # Nome
                        html.Td(
                            player_name,
                            style={"textAlign": "left", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Posizione
                        html.Td(
                            position,
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Età
                        html.Td(
                            edad,
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Minuti
                        html.Td(
                            str(int(player['Min'])) if pd.notna(player['Min']) else "0",
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Gol
                        html.Td(
                            str(int(player['Gls'])) if pd.notna(player['Gls']) else "0",
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Assist
                        html.Td(
                            str(int(player['Ass'])) if pd.notna(player['Ass']) else "0",
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Scadenza Contratto
                        html.Td(
                            contract_until,
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Miglior Profilo
                        html.Td(
                            best_profile if best_profile else "N/A",
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        ),
                        # Link al profilo
                        html.Td(
                            dcc.Link(
                                "👤",
                                href=f"/player/{urllib.parse.quote(player_name)}",
                                style={"fontSize": "16px"}
                            ),
                            style={"textAlign": "center", "fontSize": "12px", "padding": "4px"}
                        )
                    ]))
                
                table = dbc.Table(
                    [header, html.Tbody(rows)],
                    bordered=False,
                    dark=True,
                    hover=True,
                    responsive=True,
                    style={"marginBottom": "20px"}
                )
                tables.append(table)
        
        return html.Div(tables)
        
    except Exception as e:
        print(f"Errore nella creazione della tabella: {e}")
        return html.Div("Errore nella creazione della tabella", 
                        style={"color": "white", "textAlign": "center"})

def layout(league=None, team=None, **kwargs):
    """Layout della pagina"""
    if not league or not team:
        return html.Div("Parametri mancanti", style={"color": "white"})
        
    try:
        # Decodifica parametri
        league = urllib.parse.unquote(league)
        team = urllib.parse.unquote(team)
        
        # Recupera dati
        squad_df = get_squad_data(team)
        team_style = get_team_style(team)
        
        # Layout
        return dbc.Container([
            # Titolo
            html.H1([
                f"{team} - ",
                html.Span(team_style, style={"color": "#0082F3"})
            ], className="text-center my-4", style={"color": "white"}),
            
            # Rosa
            dbc.Row([
                dbc.Col([
                    html.H2("Plantilla y Perfiles", className="mb-4", style={"color": "white"}),
                    create_squad_table(squad_df)
                ])
            ])
        ], fluid=True, style={
            "backgroundColor": "#0A0F26",
            "minHeight": "100vh",
            "padding": "20px"
        })
        
    except Exception as e:
        print(f"Errore nel layout: {e}")
        return html.Div("Errore nel caricamento della pagina", 
                       style={"color": "white"}) 