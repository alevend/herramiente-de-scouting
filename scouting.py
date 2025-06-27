import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy import stats
import urllib.parse

# Percorso dei file dati
DATA_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/datos/fbref/base_datos_tutte_le_leghe_renamed.csv")
ORIGINAL_FILE = os.path.normpath("C:/Users/Gianluigi/Desktop/dash_scouting_app/base_datos_tutte_le_leghe_renamed.csv")

# Registrazione della pagina
dash.register_page(__name__, path='/scouting')

# Coefficienti lega
coeff_leghe = {
    "Premier League": 2.0, "La Liga": 2.0, "Serie A": 2.0, "Bundesliga": 2.0,
    "Ligue 1": 2.0, "Primeira Liga": 1.5, "Eredivisie": 1.5,
    "Championship": 1.5, "Belgio": 1.5, "Serie B": 1.0
}

# Profili KPI
profili_kpi_pesi = {
    "Centrale - Marcatura": {
        "con_palla": {"% Cmp.2": 5, "% Cmp.3": 5},
        "senza_palla": {"TklG": 15, "3.º cent.": 15, "Recup.": 10, "Fls": 5, "DTkl%": 15, "% de ganados": 20}
    },
    "Centrale - Deep Distributor": {
        "con_palla": {"3.º cent.": 10, "Dist. prg.": 15, "P1/3": 15, "% Cmp.1": 15, "% Cmp.2": 10, "% Cmp.3": 5},
        "senza_palla": {"3.º cent.": 10, "Int": 10, "DTkl%": 5, "% de ganados": 5}
    },
    "Recuperatore 20-80": {
        "con_palla": {"% Cmp.2": 10, "% Cmp.3": 10},
        "senza_palla": {"Int": 25, "Bloqueos": 20, "Desp.": 10, "Recup.": 15, "DTkl%": 5, "% de ganados": 5}
    },
    "Sentinel Fullback 20-80": {
        "con_palla": {"% Cmp": 5, "Camb.": 15},
        "senza_palla": {"Tkl": 15, "Recup.": 15, "Fls": 5, "DTkl%": 15, "% de ganados": 15, "PasesB": 15}
    },
    "Quinto ATA 70-30": {
        "con_palla": {"PrgC": 5, "Succ": 5, "CrAP": 25, "PrgR": 10, "xAG": 10, "TAtaq. pen.": 10, "Att.2": 5},
        "senza_palla": {"3.º cent.": 10, "Int": 10, "TklG": 5, "% de ganados": 5}
    },
    "Arnold 80-20": {
        "con_palla": {"3.º ataq.": 5, "PC": 15, "Exitosa%": 15, "Camb.": 20, "PrgP": 15, "% Cmp": 10},
        "senza_palla": {"3.º cent.": 5, "Int": 5, "TklG": 5, "% de ganados": 5}
    },
    "Difensivo": {
        "con_palla": {"% Cmp": 10, "% Cmp.1": 10},
        "senza_palla": {"Int": 15, "Recup.": 25, "Fls": 5, "TklG": 10, "PasesB": 10, "DTkl%": 10, "% de ganados": 5}
    },
    "Play": {
        "con_palla": {"Toques": 25, "PrgP": 15, "P1/3": 10, "% Cmp.1": 10, "% Cmp.3": 10},
        "senza_palla": {"Tkl": 15, "Tkl+Int": 15}
    },
    "Box-to-Box": {
        "con_palla": {"PrgC": 30, "Dist. prg.": 20, "Succ": 5, "npxG+xAG":10, "PrgR": 5},
        "senza_palla": {"Tkl": 10, "Tkl+Int": 10, "% de ganados": 5, "Recup.": 5}
    },
    "Dieci": {
        "con_palla": {"P1/3": 5, "xAG": 10, "PPA": 5, "SCA90": 20, "C1/3": 7.5, "PrgP": 5, "xG": 5, "DistD": 15, "T/90": 5, "PL": 5, "Att": 5, "Exitosa%": 2.5},
        "senza_palla": {"3.º ataq": 5, "Recup.": 5}
    },
    "1vs1": {
        "con_palla": {"xAG": 10, "CrAP": 10, "SCA90": 10, "PL": 5, "xG": 5, "Att": 10, "Exitosa%": 30},
        "senza_palla": {"Tkl+Int": 5, "FR": 10, "Recup.": 5}
    },
    "Interior 90-10": {
        "con_palla": {"xAG": 25, "SCA90": 20, "TAP": 15, "PPA": 15, "T/90": 5, "xG": 10, "Att": 5},
        "senza_palla": {"3.º ataq": 5, "Recup.": 5}
    },
    "Mobile Forward": {
        "con_palla": {"xG": 10, "xAG": 10, "PrgR": 15, "SCA90": 15, "T/90": 10, "T3.º ataq.": 5, "C1/3": 15, "Gls.": 5},
        "senza_palla": {"Recup.": 5, "FR": 10}
    },
    "Target": {
        "con_palla": {"PrgR": 20, "npxG": 10, "Ass": 20, "T/90": 10, "TAtaq. pen.": 10, "% de TT": 10, "Gls.": 5},
        "senza_palla": {"% de ganados": 10, "FR": 5}
    },
    "Lethal Striker": {
        "con_palla": {"npxG": 10, "T/90": 10, "G/TalArc": 10, "TAtaq. pen.": 20, "% de TT": 10, "Gls.": 20},
        "senza_palla": {"% de ganados": 20}
    }
}

# Mapping dei ruoli con i profili possibili
ROLE_PROFILES = {
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

def eta_decimale(valore):
    try:
        anni, giorni = str(valore).split('-')
        return int(anni) + int(giorni) / 365
    except:
        try:
            return float(valore)
        except:
            return None

# Lista delle colonne necessarie
COLONNE_NECESSARIE = [
    'Name', 'Position', 'Height', 'Foot', 'Market Value', 'Edad', 'Mín', 'Lega', 'Squadra', 'link',
    # Colonne per i profili
    'TklG', '3.º def.', '3.º cent.', '3.º ataq.', 'Recup.', 'Fls', 'DTkl%', '% de ganados',
    'Dist. tot.', 'Dist. prg.', 'P1/3', '% Cmp', '% Cmp.1', '% Cmp.2', '% Cmp.3',
    'Int', 'Bloqueos', 'Desp.', 'Tkl', 'PasesB',
    'PrgC', 'Succ', 'Camb.', 'PrgR', 'xAG', 'TAtaq. pen.',
    'PC', 'Exitosa%', 'CrAP', 'PrgP', 'Gls.', 'npxG', 'T/90', 'G/TalArc', 'Ass', 'Att.2'
]

# Funzione per convertire il valore di mercato in milioni
def converti_valore_mercato(valore):
    try:
        if pd.isna(valore) or valore == '':
            return 0.0
        
        # Converte il valore in stringa e rimuove tutti i possibili simboli dell'euro
        valore = str(valore)
        valore = valore.replace('â¬', '').replace('â\x82¬', '').replace('€', '').strip()
        
        # Converte in numero
        if 'm' in valore.lower():
            return float(valore.lower().replace('m', ''))
        elif 'k' in valore.lower():
            return float(valore.lower().replace('k', '')) / 1000
        else:
            return float(valore)
    except:
        return 0.0

# Funzione per caricare i dati da entrambi i file
def load_data():
    try:
        # Prima prova a leggere le colonne disponibili nel file principale
        available_cols = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
        cols_to_load = [col for col in available_cols if col in COLONNE_NECESSARIE]
        
        # Carica il file principale con le colonne disponibili
        df_main = pd.read_csv(
            DATA_FILE,
            usecols=cols_to_load,
            low_memory=False
        )
        
        # Controlla quali colonne mancano
        missing_cols = [col for col in COLONNE_NECESSARIE if col not in df_main.columns]
        
        if missing_cols:
            try:
                # Verifica quali colonne sono disponibili nel file originale
                available_cols_orig = pd.read_csv(ORIGINAL_FILE, nrows=0).columns.tolist()
                cols_to_load = ['Name'] + [col for col in missing_cols if col in available_cols_orig]
                
                if len(cols_to_load) > 1:  # se ci sono colonne da caricare oltre a 'Name'
                    # Carica solo le colonne mancanti dal file originale
                    df_original = pd.read_csv(
                        ORIGINAL_FILE,
                        usecols=cols_to_load,
                        low_memory=False
                    )
                    
                    # Merge solo delle colonne mancanti
                    df = pd.merge(df_main, df_original, on='Name', how='left')
                else:
                    df = df_main
            except Exception as e:
                print(f"Attenzione nel caricamento del file originale: {e}")
                df = df_main
        else:
            df = df_main
            
    except Exception as e:
        print(f"Errore nel caricamento del file principale: {e}")
        # Prova a caricare il file originale
        try:
            # Verifica quali colonne sono disponibili
            available_cols = pd.read_csv(ORIGINAL_FILE, nrows=0).columns.tolist()
            cols_to_load = [col for col in available_cols if col in COLONNE_NECESSARIE]
            
            df = pd.read_csv(
                ORIGINAL_FILE,
                usecols=cols_to_load,
                low_memory=False
            )
        except Exception as e:
            print(f"Errore nel caricamento del file originale: {e}")
            raise Exception("Nessun file dati trovato")
    
    # Aggiungi colonne mancanti con valori di default
    for col in COLONNE_NECESSARIE:
        if col not in df.columns:
            print(f"Aggiunta colonna mancante con valori di default: {col}")
            if col in ['Edad', 'Mín']:
                df[col] = 0
            elif col in ['Market Value']:
                df[col] = '€0M'
            elif col == 'link':
                df[col] = df['Name'].apply(lambda x: f'/player/{urllib.parse.quote(x)}')
            else:
                df[col] = ''
    
    # Converti i tipi di dati per ottimizzare la memoria
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    
    return df

# Carica i dati
try:
    df = load_data()
except Exception as e:
    print(f"Errore nel caricamento dei dati: {e}")
    df = pd.DataFrame(columns=COLONNE_NECESSARIE)  # DataFrame vuoto ma con le colonne corrette

# Parsing dei dati
df['Edad'] = df['Edad'].apply(eta_decimale)
df['Mín'] = df['Mín'].astype(str).str.replace(',', '').astype(float)
df['Market_Value_M'] = df['Market Value'].apply(converti_valore_mercato)

# Calcola il coefficiente lega in modo più efficiente
df["Coeff Lega"] = 1.0  # valore di default
mask_min = df['Mín'] < 1500
df.loc[mask_min, "Coeff Lega"] = 1.5

# Applica i coefficienti delle leghe conosciute
for lega, coeff in coeff_leghe.items():
    mask_lega = df['Lega'] == lega
    df.loc[mask_lega, "Coeff Lega"] = coeff

# Calcolo degli score per ogni profilo una volta sola
for profilo in profili_kpi_pesi.keys():
    # Determina le posizioni valide per il profilo
    if profilo in ["Centrale - Marcatura", "Centrale - Deep Distributor", "Recuperatore 20-80"]:
        posizioni_valide = ["Centre-Back"]
    elif profilo == "Sentinel Fullback 20-80":
        posizioni_valide = ["Left-Back", "Right-Back"]
    elif profilo in ["Quinto ATA 70-30", "Arnold 80-20"]:
        posizioni_valide = ["Left-Back", "Right-Back", "Left Midfield", "Right Midfield"]
    elif profilo in ["Difensivo", "Play", "Box-to-Box"]:
        posizioni_valide = ["Defensive Midfield", "Central Midfield", "Midfielder"]
        if profilo == "Box-to-Box":
            posizioni_valide.append("Attacking Midfield")
    elif profilo == "Dieci":
        posizioni_valide = ["Attacking Midfield", "Second Striker"]
    elif profilo == "1vs1":
        posizioni_valide = ["Right Winger", "Right Midfield", "Left Winger", "Left Midfield", "Second Striker", "Attacking Midfield"]
    elif profilo == "Interior 90-10":
        posizioni_valide = ["Right Winger", "Left Winger", "Second Striker", "Attacking Midfield"]
    elif profilo in ["Mobile Forward", "Target"]:
        posizioni_valide = ["Centre-Forward", "Second Striker"]
    elif profilo == "Lethal Striker":
        posizioni_valide = ["Centre-Forward"]
    else:
        posizioni_valide = ["Left-Back", "Right-Back", "Left Midfield", "Right Midfield"]
    
    # Filtra per posizione
    df_pos = df[df["Position"].isin(posizioni_valide)].copy()
    
    # Calcola lo score per questo profilo
    score_series = pd.Series(0.0, index=df_pos.index)
    for sezione, kpi_pesi in profili_kpi_pesi[profilo].items():
        for kpi, peso in kpi_pesi.items():
            colonna = next((c for c in df_pos.columns if kpi in c), None)
            if colonna:
                valori = pd.to_numeric(df_pos[colonna], errors='coerce') * df_pos['Coeff Lega']
                minimo, massimo = valori.min(), valori.max()
                if massimo > minimo:
                    # Normalizzazione da 20 a 99 invece che da 1 a 99
                    valori_norm = ((valori - minimo) * 79 / (massimo - minimo) + 20).round(2)
                else:
                    valori_norm = pd.Series(20, index=df_pos.index)
                score_series += valori_norm.fillna(20) * peso
    
    # Normalizza a 99, mantenendo 20 come minimo
    massimo_score = score_series.max()
    if massimo_score > 20:
        # Riscala mantenendo 20 come minimo
        score_series = (((score_series - 20) * (79)) / (massimo_score - 20) + 20).round(1)
    else:
        score_series = pd.Series(20, index=df_pos.index)
    
    # Salva lo score nel DataFrame principale
    df.loc[df_pos.index, f'Score_{profilo}'] = score_series

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

# Layout della pagina
layout = dbc.Container([
    html.H1("Scouting Dashboard", className="text-center my-4", style={"color": "#0082F3"}),
    
    dbc.Row([
        dbc.Col([
            html.H4("Filtri", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    # Selezione profilo
                    html.Label("Profilo KPI"),
                    dcc.Dropdown(
                        id='profile-dropdown',
                        options=[{"label": p, "value": p} for p in profili_kpi_pesi.keys()],
                        value=list(profili_kpi_pesi.keys())[0],
                        clearable=False,
                        style={
                            'color': 'black',
                            'background-color': 'white'
                        }
                    ),
                    
                    # Filtro categoria
                    html.Label("Categoria", className="mt-3"),
                    dcc.Dropdown(
                        id='category-dropdown',
                        options=[{"label": cat, "value": cat} for cat in CATEGORIE],
                        value=None,
                        clearable=True,
                        style={
                            'color': 'black',
                            'background-color': 'white'
                        }
                    ),
                    
                    # Filtro minuti
                    html.Label("Range minuti giocati", className="mt-3"),
                    dcc.RangeSlider(
                        id='min-minutes-slider',
                        min=0,
                        max=int(df["Mín"].max()),
                        step=100,
                        value=[1000, int(df["Mín"].max())],
                        marks={i: str(i) for i in range(0, int(df["Mín"].max()) + 1, 500)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    # Filtro età
                    html.Label("Età massima", className="mt-3"),
                    dcc.Input(
                        id='max-age-input',
                        type="number",
                        value=25,
                        min=15,
                        max=40,
                        step=1
                    ),

                    # Filtro valore di mercato
                    html.Label("Valore di mercato (M€)", className="mt-3"),
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
        
        # Tabella giocatori
        dbc.Col([
            html.H4("Giocatori", className="mb-3"),
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
                        'if': {'column_id': 'Profilo'},
                        'color': '#0082F3',
                        'cursor': 'pointer',
                        'textDecoration': 'none'
                    }
                ],
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Profilo'},
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
    [Output("players-table", "data"),
     Output("players-table", "columns")],
    [Input("profile-dropdown", "value"),
     Input("min-minutes-slider", "value"),
     Input("max-age-input", "value"),
     Input("market-value-slider", "value"),
     Input("category-dropdown", "value")]
)
def update_table(profilo, minuti_range, max_eta, market_value_range, selected_category):
    # Determina le posizioni valide per il profilo
    if profilo in ["Centrale - Marcatura", "Centrale - Deep Distributor", "Recuperatore 20-80"]:
        posizioni_valide = ["Centre-Back"]
    elif profilo == "Sentinel Fullback 20-80":
        posizioni_valide = ["Left-Back", "Right-Back"]
    elif profilo in ["Quinto ATA 70-30", "Arnold 80-20"]:
        posizioni_valide = ["Left-Back", "Right-Back", "Left Midfield", "Right Midfield"]
    elif profilo in ["Difensivo", "Play", "Box-to-Box"]:
        posizioni_valide = ["Defensive Midfield", "Central Midfield", "Midfielder"]
        if profilo == "Box-to-Box":
            posizioni_valide.append("Attacking Midfield")
    elif profilo == "Dieci":
        posizioni_valide = ["Attacking Midfield", "Second Striker"]
    elif profilo == "1vs1":
        posizioni_valide = ["Right Winger", "Right Midfield", "Left Winger", "Left Midfield", "Second Striker", "Attacking Midfield"]
    elif profilo == "Interior 90-10":
        posizioni_valide = ["Right Winger", "Left Winger", "Second Striker", "Attacking Midfield"]
    elif profilo in ["Mobile Forward", "Target"]:
        posizioni_valide = ["Centre-Forward", "Second Striker"]
    elif profilo == "Lethal Striker":
        posizioni_valide = ["Centre-Forward"]
    else:
        posizioni_valide = ["Left-Back", "Right-Back", "Left Midfield", "Right Midfield"]

    # Applica i filtri base
    min_minuti, max_minuti = minuti_range
    mask = (
        df["Position"].isin(posizioni_valide) &
        df["Mín"].notna() &
        (df["Mín"] >= min_minuti) &
        (df["Mín"] <= max_minuti)
    )

    # Applica il filtro del market value se specificato
    if market_value_range:
        min_value, max_value = market_value_range
        mask = mask & (df["Market_Value_M"] >= min_value) & (df["Market_Value_M"] <= max_value)
    
    df_filtrato = df[mask].copy()
    if max_eta:
        df_filtrato = df_filtrato[df_filtrato["Edad"] <= max_eta]
    
    # Usa lo score pre-calcolato
    df_filtrato["Score"] = df_filtrato[f'Score_{profilo}']
    
    # Add player category based on age and score
    df_filtrato["Categoria"] = df_filtrato.apply(
        lambda row: get_player_category(row["Edad"], row["Score"]), 
        axis=1
    )
    
    # Applica il filtro per categoria se selezionata
    if selected_category:
        df_filtrato = df_filtrato[df_filtrato["Categoria"] == selected_category]
    
    df_filtrato = df_filtrato.sort_values(by="Score", ascending=False)
    
    # Aggiungi la colonna Profilo con il link
    df_filtrato['Profilo'] = df_filtrato['Name'].apply(
        lambda x: f'[Visualizza](/player/{urllib.parse.quote(x)})'
    )
    
    # Ricalcola Market_Value_M dal Market Value originale
    df_filtrato['Market_Value_M'] = df_filtrato['Market Value'].apply(converti_valore_mercato)
    
    # Formatta il Market Value usando Market_Value_M
    df_filtrato['Market Value'] = df_filtrato['Market_Value_M'].apply(
        lambda x: f"€{x:.1f}M" if pd.notna(x) else "N/A"
    )
    
    colonne_mostrate = ["Name", "Edad", "Height", "Foot", "Position", "Squadra", "Market Value", "Mín", "Score", "Categoria", "Profilo"]
    
    # Formatta i minuti senza virgola
    df_filtrato['Mín'] = df_filtrato['Mín'].round(0).astype(int)
    
    data = df_filtrato[colonne_mostrate].round(1).to_dict("records")
    columns = [{"name": col, "id": col} for col in colonne_mostrate]
    
    # Configura la colonna Profilo per renderizzare i link markdown
    columns[-1]["presentation"] = "markdown"
    
    return data, columns 