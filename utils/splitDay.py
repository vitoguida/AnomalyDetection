import os
#uesto file serve per dividere il dataset originario in 58 dataset giornalieri
INPUT_FILE = '../data/raw/auth.txt'
OUTPUT_DIR = '../data/processed'
SECONDS_PER_DAY = 86400
TOTAL_DAYS = 58

# Assicurati che la cartella di output esista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Apri tutti i file di output in anticipo per velocità
output_files = {
    day: open(os.path.join(OUTPUT_DIR, f'day{day+1}.txt'), 'w', encoding='utf-8')
    for day in range(TOTAL_DAYS)
}

with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
    for line_num, line in enumerate(infile, 1):
        try:
            time_str = line.split(',', 1)[0]
            day_index = int(int(time_str) // SECONDS_PER_DAY)

            if 0 <= day_index < TOTAL_DAYS:
                output_files[day_index].write(line)
            else:
                # Se il giorno è fuori dal range atteso
                pass
        except Exception as e:
            # Log degli errori se necessario
            pass

        # (Opzionale) stampa un progresso ogni 10 milioni di righe
        if line_num % 10_000_000 == 0:
            print(f'Processed {line_num} lines...')

# Chiudi tutti i file
for f in output_files.values():
    f.close()
