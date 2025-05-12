import torch
import torch.nn as nn
import os

# 1. Carica vocabolario
def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return word2idx

# 2. Carica righe tokenizzate, separando timestamp e token
def load_tokenized_data_with_timestamps(token_file, word2idx):
    data = []
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            timestamp = parts[0]
            tokens = parts[1:]
            indices = [word2idx[token] for token in tokens if token in word2idx]
            data.append((timestamp, indices))
    return data

# 3. Crea l'embedding layer
def create_embedding_layer(vocab_size, embedding_dim, device='cpu'):
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    return embedding.to(device)

# 4. Embedding + salvataggio batch da 10.000 righe (timestamp incluso)
def convert_tokens_to_embeddings_batch_save(data_with_timestamps, embedding_layer, output_path, batch_size=10000, device='cpu'):
    os.makedirs(output_path, exist_ok=True)
    batch = []
    batch_count = 0

    for i, (timestamp, indices) in enumerate(data_with_timestamps):
        if not indices:
            continue
        tensor = torch.tensor(indices, dtype=torch.long).to(device)

        with torch.no_grad():
            embedded = embedding_layer(tensor).cpu()

        batch.append((timestamp, embedded))

        if len(batch) == batch_size:
            batch_path = os.path.join(output_path, f'embedding_batch_{batch_count}.pt')
            torch.save(batch, batch_path)
            print(f"Salvato batch {batch_count} con {len(batch)} righe in: {batch_path}")
            batch_count += 1
            batch = []

    # salva l'ultimo batch se presente
    if batch:
        batch_path = os.path.join(output_path, f'embedding_batch_{batch_count}.pt')
        torch.save(batch, batch_path)
        print(f"Salvato batch finale {batch_count} con {len(batch)} righe in: {batch_path}")


# ===== MAIN =====
if __name__ == "__main__":
    vocab_path = '../data/processed/vocabulary.txt'
    token_path = '../data/processed/tokenized_log.txt'
    output_dir = '../data/processed/embeddings'

    embedding_dim = 128
    batch_size = 1000000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Usando il dispositivo: {device}")

    word2idx = load_vocab(vocab_path)
    data_with_timestamps = load_tokenized_data_with_timestamps(token_path, word2idx)
    embedding_layer = create_embedding_layer(vocab_size=len(word2idx), embedding_dim=embedding_dim, device=device)

    convert_tokens_to_embeddings_batch_save(
        data_with_timestamps,
        embedding_layer,
        output_path=output_dir,
        batch_size=batch_size,
        device=device
    )

    print("Tutti i batch sono stati salvati con timestamp.")

"""
import os
import torch

def merge_embedding_batches(input_dir, output_file):
    all_data = []

    # Ordina i file per batch index (assumendo formato: embedding_batch_0.pt, ...)
    files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    for f in files:
        path = os.path.join(input_dir, f)
        batch = torch.load(path)
        all_data.extend(batch)  # ogni elemento è una tupla: (timestamp, embedding)
        print(f"Caricato: {f} ({len(batch)} righe)")

    torch.save(all_data, output_file)
    print(f"\nSalvato tutto in: {output_file} ({len(all_data)} righe totali)")

# ===== ESEMPIO USO =====
if __name__ == "__main__":
    input_dir = "embeddings"
    output_file = "all_embeddings.pt"
    merge_embedding_batches(input_dir, output_file)
"""


