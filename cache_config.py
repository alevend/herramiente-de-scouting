from functools import lru_cache
import pandas as pd
import os

"""
Configurazioni per il caching e il server Dash
"""

# Configurazioni per l'app Dash
DASH_CONFIG = {
    'suppress_callback_exceptions': True,
    'update_title': None
}

# Configurazioni per il server
SERVER_CONFIG = {
    'debug': True,
    'host': '0.0.0.0',
    'port': 8050
}

# Percorsi base
WYSCOUT_FOLDER = os.path.join("datos", "wyscout", "2024-2025")
FBREF_FOLDER = os.path.join("datos", "fbref")

@lru_cache(maxsize=None)
def load_wyscout_data(league="Serie A", season="2024-2025"):
    """
    Carica e memorizza nella cache i dati di Wyscout.
    """
    try:
        base_path = os.path.join(WYSCOUT_FOLDER, season, league)
        files = {
            "General.xlsx": None,
            "Indices.xlsx": None,
            "Organizacion.xlsx": None,
            "Acciones_ofensivas.xlsx": None
        }
        
        dfs = {}
        for file_name in files:
            file_path = os.path.join(base_path, file_name)
            try:
                df = pd.read_excel(file_path)
                # Ottimizza i tipi di dati
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = df[col].astype('float32')
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = df[col].astype('int32')
                dfs[file_name] = df
            except:
                dfs[file_name] = pd.DataFrame()
        
        return (dfs.get("General.xlsx", pd.DataFrame()),
                dfs.get("Indices.xlsx", pd.DataFrame()),
                dfs.get("Organizacion.xlsx", pd.DataFrame()),
                dfs.get("Acciones_ofensivas.xlsx", pd.DataFrame()))
                
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@lru_cache(maxsize=None)
def load_fbref_data(league="Serie A", season="2024-2025"):
    """
    Carica e memorizza nella cache i dati di FBRef.
    """
    try:
        file_path = os.path.join(FBREF_FOLDER, season, league, "porteria_avanzada.csv")
        df = pd.read_csv(file_path)
        
        # Ottimizza i tipi di dati
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
            
        return df
    except:
        return pd.DataFrame()

def check_specific_data_files(league="Serie A", season="2024-2025"):
    """
    Verifica l'esistenza dei file necessari solo per una specifica lega e stagione.
    """
    missing_files = []
    
    # Verifica file Wyscout
    wyscout_path = os.path.join(WYSCOUT_FOLDER, season, league)
    if not os.path.exists(wyscout_path):
        missing_files.append(f"Directory Wyscout: {wyscout_path}")
    else:
        required_files = ["General.xlsx", "Indices.xlsx", "Organizacion.xlsx", "Acciones_ofensivas.xlsx"]
        for file in required_files:
            if not os.path.exists(os.path.join(wyscout_path, file)):
                missing_files.append(f"File Wyscout: {os.path.join(wyscout_path, file)}")
                
    # Verifica file FBRef
    fbref_file = os.path.join(FBREF_FOLDER, season, league, "porteria_avanzada.csv")
    if not os.path.exists(fbref_file):
        missing_files.append(f"File FBRef: {fbref_file}")
    
    if missing_files:
        print("⚠️ File mancanti:")
        for file in missing_files:
            print(f"  - {file}")
    
    return len(missing_files) == 0 