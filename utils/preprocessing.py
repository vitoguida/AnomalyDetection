import pandas as pd
#questo file serve per estrarre le righe in cui l'accesso è avvenuto da un utente e non da un computer internamente
# Definizione delle colonne
colonne = [
    'time',
    'source user@domain',
    'destination user@domain',
    'source computer',
    'destination computer',
    'authentication type',
    'logon type',
    'authentication orientation',
    'success/failure'
]

# Carica il dataset dal file senza intestazione
df = pd.read_csv('../data/processed/xdays/day9.txt', names=colonne)

# Filtra le righe in cui 'source user@domain' inizia con 'U'
df_utenti = df[df['source user@domain'].str.match(r'^U')]


df_utenti.to_csv('../data/processed/day9Filtrato.csv', index=False)
