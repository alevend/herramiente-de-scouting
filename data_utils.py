import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, cosine
import unicodedata
from unidecode import unidecode
from utils.position_mapping import get_role_group
from utils.config import PERFILES_KPI_PESOS
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

MIN_MINUTES = 500

def get_team_age_stats(team_name, league):
    """Ottiene le statistiche sull'età della squadra"""
    try:
        # Costruisci il percorso del file Excel
        file_path = os.path.join("datos", "wyscout", "2024-2025", league, f"Team Stats {team_name}.xlsx")
        
        # Leggi il file Excel
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            # Calcola l'età media
            mean_age = df['Age'].mean()
            # Calcola l'età minima e massima
            min_age = df['Age'].min()
            max_age = df['Age'].max()
            return {
                'mean': mean_age,
                'min': min_age,
                'max': max_age
            }
    except Exception as e:
        print(f"Error getting age stats for {team_name}: {str(e)}")
    
    # Valori di default se qualcosa va storto
    return {
        'mean': 25,
        'min': 18,
        'max': 35
    }

def get_most_used_formation(team_name, league):
    """Ottiene la formazione più utilizzata dalla squadra"""
    try:
        # Costruisci il percorso del file CSV delle formazioni
        file_path = os.path.join("datos", "wyscout", "2024-2025", league, "formazioni.csv")
        
        # Leggi il file CSV
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Filtra per la squadra
            team_formations = df[df['Equipo'] == team_name]
            if not team_formations.empty:
                # Prendi la formazione più utilizzata
                return team_formations.iloc[0]['Formacion']
    except Exception as e:
        print(f"Error getting formation for {team_name}: {str(e)}")
    
    # Formazione di default se qualcosa va storto
    return "4-3-3"

def convert_to_numeric(value):
    """Converte un valore in numerico, gestendo stringhe e virgole."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(',', ''))
        except (ValueError, AttributeError):
            return np.nan
    return np.nan

def normalize_player_name(name):
    if isinstance(name, str):
        # Rimuove gli accenti e altri segni diacritici
        name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
        # Sostituisce caratteri speciali specifici
        name = name.replace('ø', 'o').replace('Ø', 'O')
        # Converte in minuscolo e rimuove spazi extra
        return unidecode(name).lower().strip()
    return name

def normalize_kpi_with_coefficients(value, kpi, player_info, df, position):
    """
    Normalizza un singolo KPI per un giocatore tenendo conto dei coefficienti di lega
    e dei percentili specifici per ruolo.
    """
    try:
        role_group = get_role_group(position)
        
        # Filtra il DataFrame per il gruppo di ruolo corretto
        df_role = df[df['Position'].apply(get_role_group) == role_group]
        
        # Cerca la colonna corretta che contiene il KPI
        kpi_col = next((c for c in df_role.columns if kpi in c), None)
        if not kpi_col: return 1 # Valore di default se il kpi non è trovato

        # Calcola il valore del giocatore con il coefficiente di lega
        league_coeff = df.loc[player_info.name, 'Coeff Lega']
        player_value_adj = value * league_coeff

        # Calcola i percentili sul DataFrame filtrato per ruolo
        percentile = stats.percentileofscore(df_role[kpi_col].dropna(), player_value_adj)
        
        # Normalizza il percentile in una scala da 1 a 100
        return max(1, percentile)
    
    except Exception as e:
        # In caso di errore, ritorna un valore di default
        return 1

def calculate_similarity_scores(player_data, profile, all_players_df):
    """
    Calcola i punteggi di similarità per tutti i giocatori candidati.
    """
    try:
        profile_kpis = PERFILES_KPI_PESOS.get(profile, {})
        all_kpis = {**profile_kpis.get("con_palla", {}), **profile_kpis.get("senza_palla", {})}
        kpi_list = list(all_kpis.keys())
        
        if not kpi_list:
            return []

        player_position = player_data['Position']
        
        # Abbasso la soglia dei minuti per trovare più giocatori
        min_minutes = 900 
        
        all_players_df['Min_numeric'] = all_players_df['Mín'].apply(convert_to_numeric)
        
        position_mask = (all_players_df['Position'] == player_position) & (all_players_df['Min_numeric'] >= min_minutes)
        comparable_players = all_players_df[position_mask].copy()
        
        if len(comparable_players) < 2:
            return []
        
        feature_vectors = []
        player_names = []
        
        # Aggiungi il giocatore di riferimento per la normalizzazione se non presente
        if player_data['Name'] not in comparable_players['Name'].values:
            comparable_players = pd.concat([pd.DataFrame([player_data]), comparable_players], ignore_index=True)

        for _, row in comparable_players.iterrows():
            vector = []
            valid_vector = True
            for kpi in kpi_list:
                col = next((c for c in row.index if kpi in c), None)
                if col and pd.notna(row[col]):
                    val = convert_to_numeric(row[col])
                    if pd.isna(val) or np.isinf(val):
                        valid_vector = False; break
                    vector.append(val)
                else:
                    valid_vector = False; break
            
            if valid_vector:
                feature_vectors.append(vector)
                player_names.append(row['Name'])
        
        if len(feature_vectors) < 2:
            return []
            
        feature_vectors = np.array(feature_vectors)
        min_vals, max_vals = feature_vectors.min(axis=0), feature_vectors.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        normalized_vectors = (feature_vectors - min_vals) / range_vals
        
        try:
            target_idx = player_names.index(player_data['Name'])
            target_vector = normalized_vectors[target_idx]
        except ValueError:
            return []
            
        similarities = []
        for i, vector in enumerate(normalized_vectors):
            if player_names[i] == player_data['Name']: continue
                
            eucl_sim = 1 / (1 + euclidean(target_vector, vector))
            cos_sim = 1 - cosine(target_vector + 1e-9, vector + 1e-9)
            
            if not np.isnan(eucl_sim) and not np.isnan(cos_sim):
                combined_sim = (0.15 * eucl_sim) + (0.85 * cos_sim)
                player_row = comparable_players[comparable_players['Name'] == player_names[i]].iloc[0]
                similarities.append({
                    'name': player_names[i],
                    'similarity': combined_sim,
                    'team': player_row['Squadra'],
                    'league': player_row['Lega']
                })
                
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:5]
        
    except Exception as e:
        print(f"Errore generale nel calcolo delle similarità: {str(e)}")
        import traceback
        traceback.print_exc()
        return [] 

def load_main_data():
    """Carica e prepara il dataframe principale."""
    # ... (logica invariata)
    return df

def load_coefficients():
    """Carica i coefficienti di lega."""
    # ... (logica invariata)
    return coeff_leghe

def find_similar_players(player_name, profile, league):
    """
    Trova i 5 giocatori più simili in base al profilo.
    """
    GLOBAL_DF = load_main_data() # Carica i dati qui
    COEFFICIENTS = load_coefficients() # Carica i dati qui
    
    try:
        player_info = GLOBAL_DF[GLOBAL_DF['Name'] == player_name].iloc[0]
        player_position = player_info['Position']
        
        # Filtra il DataFrame per trovare candidati
        candidates = GLOBAL_DF[
            (GLOBAL_DF['Name'] != player_name) &
            (GLOBAL_DF['Position'] == player_position) &
            (GLOBAL_DF['Mín'] >= MIN_MINUTES)
        ].copy()
        
        if candidates.empty:
            return []

        # ... (resto della funzione invariato)
        
    except (IndexError, KeyError):
        # Se il giocatore non viene trovato, restituisce una lista vuota
        return []
    except Exception as e:
        import traceback
        print(f"Errore inatteso in find_similar_players: {e}")
        traceback.print_exc()
        return [] 