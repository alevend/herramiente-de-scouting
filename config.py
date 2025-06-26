import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "datos")
WYSCOUT_FOLDER = os.path.join(DATA_DIR, "wyscout")

# League specific paths
def get_league_data_path(league, season="2024-2025"):
    return os.path.join(WYSCOUT_FOLDER, season, league)

# Asset paths
def get_team_assets_path(league, team_name):
    return os.path.join(ASSETS_DIR, league, team_name)

def get_flag_path(country):
    return os.path.join(ASSETS_DIR, "flags", f"{country}.png")

# League configurations
LEAGUE_CONFIGS = {
    "Serie A": {
        "country": "italy",
        "flag": "/assets/flags/italy.png",
    },
    "Premier League": {
        "country": "england",
        "flag": "/assets/flags/england.png",
    },
    "La Liga": {
        "country": "spain",
        "flag": "/assets/flags/spain.png",
    },
    "Bundesliga": {
        "country": "germany",
        "flag": "/assets/flags/germany.png",
    },
    "Ligue 1": {
        "country": "france",
        "flag": "/assets/flags/france.png",
    }
}

# Team name mappings
TEAM_NAME_MAPPING = {
    "Atalanta": "Atalanta",
    "Inter": "Inter",
    "Milan": "Milan",
    "Juventus": "Juventus",
    "Roma": "Roma",
    "Lazio": "Lazio",
    "Napoli": "Napoli",
    "Fiorentina": "Fiorentina"
    # Add more teams as needed
}

# League mappings
TEAM_LEAGUE_MAPPING = {
    "Atalanta": "Serie A",
    "Inter": "Serie A",
    "Milan": "Serie A",
    "Juventus": "Serie A",
    "Roma": "Serie A",
    "Lazio": "Serie A",
    "Napoli": "Serie A",
    "Fiorentina": "Serie A"
    # Add more teams as needed
}

# Translations
POSITION_TRANSLATION = {
    "Goalkeeper": "Portero",
    "Centre-Back": "Defensa Central",
    "Left-Back": "Lateral Izquierdo",
    "Right-Back": "Lateral Derecho",
    "Defensive Midfield": "Pivote defensivo",
    "Central Midfield": "Mediocentro",
    "Midfielder": "Centrocampista",
    "Attacking Midfield": "Centrocampista ofensivo",
    "Left Midfield": "Carrilero Izquierdo",
    "Right Midfield": "Carrilero Derecho",
    "Left Winger": "Extremo Izquierdo",
    "Right Winger": "Extremo Derecho",
    "Second Striker": "Segundo Delantero",
    "Centre-Forward": "Delantero"
}

# KPI Profiles (Spanish)
PERFILES_KPI_PESOS = {
    "Marcador": {
        "con_palla": {"% Cmp.2": 5, "% Cmp.3": 5},
        "senza_palla": {"TklG": 15, "3.º cent.": 15, "Recup.": 10, "Fls": 5, "DTkl%": 15, "% de ganados": 20}
    },
    "Central de salida": {
        "con_palla": {"% Cmp.2": 10, "Dist. prg.": 15, "P1/3": 15, "% Cmp.1": 15, "% Cmp.3": 5},
        "senza_palla": {"3.º cent.": 10, "Int": 10, "DTkl%": 5, "% de ganados": 5}
    },
    "Recuperador rapido": {
        "con_palla": {"% Cmp.2": 10, "% Cmp.3": 10},
        "senza_palla": {"Int": 25, "Bloqueos": 20, "Desp.": 10, "Recup.": 15, "DTkl%": 5, "% de ganados": 5}
    },
    "Lateral defensivo": {
        "con_palla": {"% Cmp": 5, "Camb.": 15},
        "senza_palla": {"Tkl": 15, "Recup.": 15, "Fls": 5, "DTkl%": 15, "% de ganados": 15, "PasesB": 15}
    },
    "Carrilero de recorrido": {
        "con_palla": {"PrgC": 5, "Succ": 5, "CrAP": 25, "PrgR": 10, "xAG": 10, "TAtaq. pen.": 10, "Att.2": 5},
        "senza_palla": {"3.º cent.": 10, "Int": 10, "TklG": 5, "% de ganados": 5}
    },
    "Lateral tecnico y de salida": {
        "con_palla": {"3.º ataq.": 5, "PC": 15, "Exitosa%": 15, "Camb.": 20, "PrgP": 15, "% Cmp": 10},
        "senza_palla": {"3.º cent.": 5, "Int": 5, "TklG": 5, "% de ganados": 5}
    },
    "Pivote defensivo": {
        "con_palla": {"% Cmp": 10, "% Cmp.1": 10},
        "senza_palla": {"Int": 15, "Recup.": 25, "Fls": 5, "TklG": 10, "PasesB": 10, "DTkl%": 10, "% de ganados": 5}
    },
    "Play": {
        "con_palla": {"Toques": 25, "PrgP": 15, "P1/3": 10, "% Cmp.1": 10, "% Cmp.3": 10},
        "senza_palla": {"Tkl": 15, "Tkl+Int": 15}
    },
    "Box to box": {
        "con_palla": {"PrgC": 30, "Dist. prg.": 20, "Succ": 5, "npxG+xAG":10, "PrgR": 5},
        "senza_palla": {"Tkl": 10, "Tkl+Int": 10, "% de ganados": 5, "Recup.": 5}
    },
    "Diez": {
        "con_palla": {"P1/3": 5, "xAG": 10, "PPA": 5, "SCA90": 20, "C1/3": 7.5, "PrgP": 5, "xG": 5, "Dist": 15, "T/90": 5, "PL": 5, "Att": 5, "Exitosa%": 2.5},
        "senza_palla": {"3.º ataq": 5, "Recup.": 5}
    },
    "Extremo de 1vs1": {
        "con_palla": {"xAG": 10, "CrAP": 10, "SCA90": 10, "PL": 5, "xG": 5, "Att": 10, "Exitosa%": 30},
        "senza_palla": {"Tkl+Int": 5, "FR": 10, "Recup.": 5}
    },
    "Extremo interior": {
        "con_palla": {"xAG": 25, "SCA90": 20, "TAP": 15, "PPA": 15, "T/90": 5, "xG": 10, "Att": 5},
        "senza_palla": {"3.º ataq": 5, "Recup.": 5}
    },
    "Delantero movil": {
        "con_palla": {"xG": 10, "xAG": 10, "PrgR": 15, "SCA90": 15, "T/90": 10, "T3.º ataq.": 5, "C1/3": 15, "Gls.": 5},
        "senza_palla": {"Recup.": 5, "FR": 10}
    },
    "Delantero referencia": {
        "con_palla": {"PrgR": 20, "npxG": 10, "Ass": 20, "T/90": 10, "TAtaq. pen.": 10, "% de TT": 10, "Gls.": 5},
        "senza_palla": {"% de ganados": 10, "FR": 5}
    },
    "Delantero de area": {
        "con_palla": {"npxG": 10, "T/90": 10, "G/TalArc": 10, "TAtaq. pen.": 20, "% de TT": 10, "Gls.": 20},
        "senza_palla": {"% de ganados": 20}
    },
    "Portero": {}
}

ROLE_PROFILES = {
    "Goalkeeper": ["Portero"],
    "Centre-Back": ["Marcador", "Central de salida", "Recuperador rapido"],
    "Left-Back": ["Lateral defensivo", "Carrilero de recorrido", "Lateral tecnico y de salida"],
    "Right-Back": ["Lateral defensivo", "Carrilero de recorrido", "Lateral tecnico y de salida"],
    "Defensive Midfield": ["Pivote defensivo", "Play", "Box to box"],
    "Central Midfield": ["Pivote defensivo", "Play", "Box to box"],
    "Midfielder": ["Pivote defensivo", "Play", "Box to box"],
    "Attacking Midfield": ["Diez", "Box to box", "Extremo interior"],
    "Left Midfield": ["Carrilero de recorrido", "Lateral tecnico y de salida", "Extremo de 1vs1"],
    "Right Midfield": ["Carrilero de recorrido", "Lateral tecnico y de salida", "Extremo de 1vs1"],
    "Left Winger": ["Extremo de 1vs1", "Extremo interior"],
    "Right Winger": ["Extremo de 1vs1", "Extremo interior"],
    "Second Striker": ["Diez", "Delantero movil", "Delantero referencia", "Extremo de 1vs1", "Extremo interior"],
    "Centre-Forward": ["Delantero movil", "Delantero referencia", "Delantero de area"]
}

# Invert ROLE_PROFILES to map profiles to roles
PROFILE_TO_ROLES = {}
for role, profiles in ROLE_PROFILES.items():
    for profile in profiles:
        if profile not in PROFILE_TO_ROLES:
            PROFILE_TO_ROLES[profile] = []
        PROFILE_TO_ROLES[profile].append(role) 