import os
import re
import urllib.parse

# Mappa delle squadre alle rispettive leghe
TEAM_LEAGUE_MAPPING = {
    # Serie A
    'Atalanta': 'Serie A',
    'Bologna': 'Serie A',
    'Cagliari': 'Serie A',
    'Como': 'Serie A',
    'Empoli': 'Serie A',
    'Fiorentina': 'Serie A',
    'Genoa': 'Serie A',
    'Hellas Verona': 'Serie A',
    'Inter': 'Serie A',
    'Juventus': 'Serie A',
    'Lazio': 'Serie A',
    'Lecce': 'Serie A',
    'Milan': 'Serie A',
    'Monza': 'Serie A',
    'Napoli': 'Serie A',
    'Roma': 'Serie A',
    'Salernitana': 'Serie A',
    'Sassuolo': 'Serie A',
    'Torino': 'Serie A',
    'Udinese': 'Serie A',
    # La Liga
    'Athletic Club': 'La Liga',
    'Atlético Madrid': 'La Liga',
    'Barcelona': 'La Liga',
    'Cádiz': 'La Liga',
    'Celta Vigo': 'La Liga',
    'Getafe': 'La Liga',
    'Girona': 'La Liga',
    'Granada': 'La Liga',
    'Las Palmas': 'La Liga',
    'Mallorca': 'La Liga',
    'Osasuna': 'La Liga',
    'Rayo Vallecano': 'La Liga',
    'Real Betis': 'La Liga',
    'Real Madrid': 'La Liga',
    'Real Sociedad': 'La Liga',
    'Sevilla': 'La Liga',
    'Valencia': 'La Liga',
    'Villarreal': 'La Liga',
    # Premier League
    'Arsenal': 'Premier League',
    'Aston Villa': 'Premier League',
    'Bournemouth': 'Premier League',
    'Brentford': 'Premier League',
    'Brighton': 'Premier League',
    'Burnley': 'Premier League',
    'Chelsea': 'Premier League',
    'Crystal Palace': 'Premier League',
    'Everton': 'Premier League',
    'Fulham': 'Premier League',
    'Liverpool': 'Premier League',
    'Luton': 'Premier League',
    'Manchester City': 'Premier League',
    'Manchester United': 'Premier League',
    'Newcastle United': 'Premier League',
    'Nottingham Forest': 'Premier League',
    'Sheffield United': 'Premier League',
    'Tottenham': 'Premier League',
    'West Ham': 'Premier League',
    'Wolves': 'Premier League',
}

# Mapping per le variazioni dei nomi delle squadre
TEAM_NAME_MAPPING = {
    # Serie A
    'Atalanta': 'Atalanta',
    'Bologna': 'Bologna',
    'Cagliari': 'Cagliari',
    'Como': 'Como',
    'Empoli': 'Empoli',
    'Fiorentina': 'Fiorentina',
    'Genoa': 'Genoa',
    'Hellas Verona': 'Hellas Verona',
    'Inter': ['Inter', 'Internazionale', 'Inter Milan', 'Inter_Milan'],
    'Juventus': 'Juventus',
    'Lazio': 'Lazio',
    'Lecce': 'Lecce',
    'Milan': 'Milan',
    'Monza': 'Monza',
    'Napoli': 'Napoli',
    'Roma': 'Roma',
    'Torino': 'Torino',
    'Udinese': 'Udinese',
    # La Liga
    'Athletic Club': ['Athletic Club', 'Athletic_Club', 'Athletic'],
    'Atlético Madrid': ['Atlético Madrid', 'Atletico Madrid', 'Atletico', 'Atletico_Madrid'],
    'Barcelona': ['Barcelona', 'Barca'],
    'Cádiz': ['Cádiz', 'Cadiz'],
    'Celta Vigo': ['Celta Vigo', 'Celta', 'Celta_Vigo'],
    'Getafe': 'Getafe',
    'Girona': 'Girona',
    'Granada': 'Granada',
    'Las Palmas': ['Las Palmas', 'Las_Palmas'],
    'Mallorca': 'Mallorca',
    'Osasuna': 'Osasuna',
    'Rayo Vallecano': ['Rayo Vallecano', 'Rayo', 'Rayo_Vallecano'],
    'Real Betis': ['Real Betis', 'Betis', 'Real_Betis'],
    'Real Madrid': ['Real Madrid', 'Madrid', 'Real_Madrid'],
    'Real Sociedad': ['Real Sociedad', 'La Real', 'Real_Sociedad'],
    'Sevilla': 'Sevilla',
    'Valencia': 'Valencia',
    'Villarreal': 'Villarreal',
    # Premier League
    'Arsenal': 'Arsenal',
    'Aston Villa': ['Aston Villa', 'Villa', 'Aston_Villa'],
    'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
    'Brighton': ['Brighton', 'Brighton & Hove Albion', 'Brighton_Hove_Albion'],
    'Burnley': 'Burnley',
    'Chelsea': 'Chelsea',
    'Crystal Palace': ['Crystal Palace', 'Palace', 'Crystal_Palace'],
    'Everton': 'Everton',
    'Fulham': 'Fulham',
    'Liverpool': 'Liverpool',
    'Luton': ['Luton', 'Luton Town', 'Luton_Town'],
    'Manchester City': ['Manchester City', 'Man City', 'City', 'Manchester_City'],
    'Manchester United': ['Manchester United', 'Man United', 'Man Utd', 'Manchester_United'],
    'Newcastle United': ['Newcastle United', 'Newcastle', 'Newcastle_United'],
    'Nottingham Forest': ['Nottingham Forest', 'Forest', 'Nottingham_Forest'],
    'Sheffield United': ['Sheffield United', 'Sheffield Utd', 'Sheffield_United'],
    'Tottenham': ['Tottenham', 'Tottenham Hotspur', 'Spurs'],
    'West Ham': ['West Ham', 'West Ham United', 'West_Ham'],
    'Wolves': ['Wolves', 'Wolverhampton Wanderers', 'Wolverhampton_Wanderers']
}

def get_canonical_team_name(team_name):
    """
    Ottiene il nome canonico (standard) di una squadra, gestendo vari formati del nome.
    Restituisce il nome canonico se trovato, altrimenti restituisce il nome di input.
    """
    # Prima controlla le corrispondenze dirette
    if team_name in TEAM_NAME_MAPPING:
        return team_name
    
    # Poi controlla i nomi alternativi
    for canonical_name, variations in TEAM_NAME_MAPPING.items():
        if isinstance(variations, list):
            if team_name in variations:
                return canonical_name
        elif variations == team_name:
            return canonical_name
    
    return team_name

def get_team_from_pathname(pathname):
    """Estrae il nome della squadra dal pathname"""
    if not pathname:
        return None
    
    # Il pathname è nella forma /team/league/team_name
    parts = pathname.strip('/').split('/')
    if len(parts) >= 3 and parts[0] == 'team':
        team_name = urllib.parse.unquote(parts[2])  # Decodifica l'URL
        return team_name.replace('_', ' ')
    return None

def get_league_from_pathname(pathname):
    """Estrae il nome della lega dal pathname"""
    if not pathname:
        return None
    
    # Il pathname è nella forma /team/league/team_name
    parts = pathname.strip('/').split('/')
    if len(parts) >= 3 and parts[0] == 'team':
        league_name = urllib.parse.unquote(parts[1])  # Decodifica l'URL
        return league_name.replace('_', ' ')
    return None

def get_team_logo(league, team_name):
    try:
        # Normalizza il nome della squadra
        normalized_name = team_name.replace(" ", "_")
        
        # Possibili nomi dei file logo
        possible_logo_names = [
            f"logo_{normalized_name}.png",
            f"{normalized_name}_logo.png",
            "logo.png",
            f"{normalized_name}.png"
        ]
        
        # Possibili nomi delle cartelle
        possible_folders = [
            normalized_name,
            team_name,
            team_name.replace(" ", "_"),
            team_name.replace(" ", "")
        ]
        
        # Aggiungi variazioni dal mapping
        mapped_name = TEAM_NAME_MAPPING.get(team_name, "")
        if isinstance(mapped_name, list):
            for name in mapped_name:
                possible_folders.extend([
                    name,
                    name.replace(" ", "_"),
                    name.replace(" ", "")
                ])
        elif mapped_name:
            possible_folders.extend([
                mapped_name,
                mapped_name.replace(" ", "_"),
                mapped_name.replace(" ", "")
            ])
        
        # Rimuovi duplicati
        possible_folders = list(set(possible_folders))
        
        tried_paths = []
        for folder in possible_folders:
            base_path = os.path.join('assets', '2024-25', league, folder)
            if os.path.exists(base_path):
                for logo_name in possible_logo_names:
                    logo_path = os.path.join(base_path, logo_name)
                    tried_paths.append(os.path.abspath(logo_path))
                    if os.path.exists(logo_path):
                        return f"/assets/2024-25/{league}/{folder}/{logo_name}"
        
        # Se non trova nulla, prova con il percorso di default
        default_path = f"/assets/2024-25/default_logo.png"
        if os.path.exists(os.path.join('assets', '2024-25', 'default_logo.png')):
            return default_path
            
        print(f"Logo non trovato per {team_name} in {league}. Percorsi tentati: {tried_paths}")
        return default_path
    except Exception as e:
        print(f"Errore nel recupero del logo per {team_name}: {e}")
        return "/assets/2024-25/default_logo.png"

def get_country_flag(league):
    """Recupera il percorso della bandiera del paese per una data lega"""
    league_country_mapping = {
        'Serie A': 'italia',
        'Premier League': 'inglaterra',
        'La Liga': 'espana',
        'Bundesliga': 'alemania',
        'Ligue 1': 'francia'
    }
    
    country = league_country_mapping.get(league)
    if not country:
        return None
        
    flag_path = f"/assets/2024-25/{league}/bandera_{country}"
    return flag_path if os.path.exists(os.path.join('assets', '2024-25', league, f'bandera_{country}')) else None 