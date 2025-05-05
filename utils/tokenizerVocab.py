import csv
from collections import Counter


def create_word_level_vocabulary_from_files(log_file_paths, frequency_threshold=40):
    """
    Creates a word-level vocabulary from multiple CSV log files, excluding the first column.

    Args:
        log_file_paths (list): A list of paths to the CSV log files.
        frequency_threshold (int): The minimum frequency for a word to be
                                   included in the vocabulary.

    Returns:
        tuple: A tuple containing:
            - vocabulary (set): A set of unique words in the vocabulary.
            - word_counts (Counter): A Counter object containing the frequency
                                     of each word across all files.
    """
    all_words = []
    word_counts = Counter()  # Initialize an empty Counter
    for log_file_path in log_file_paths:
        with open(log_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Skip the first column (timestamp)
                for field in row[1:]:
                    if "@" in field:
                        username, domain = field.split("@", 1)
                        all_words.extend([username, domain])
                    else:
                        all_words.append(field)
        word_counts.update(Counter(all_words))  # update the word counts for each file.

    vocabulary = {word for word, count in word_counts.items() if count >= frequency_threshold}
    vocabulary.add("<OOV>")
    return vocabulary, word_counts


def tokenize_word_level_from_files(log_file_paths, vocabulary):
    """
    Tokenizes multiple CSV log files into a sequence of word-level tokens,
    excluding the first column from tokenization and replacing out-of-vocabulary words with "<OOV>".

    Args:
        log_file_paths (list): A list of paths to the CSV log files.
        vocabulary (set): The set of words in the vocabulary.

    Returns:
        list: A list of lists, where each inner list represents a row
              from the CSV file, with the first element as the timestamp and the rest tokenized.
              The list contains data from all the files
    """
    tokenized_rows = []
    for log_file_path in log_file_paths:
        with open(log_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                timestamp = row[0]
                tokenized_row = [timestamp]
                for field in row[1:]:
                    if "@" in field:
                        username, domain = field.split("@", 1)
                        tokenized_row.extend([
                            username if username in vocabulary else "<OOV>",
                            domain if domain in vocabulary else "<OOV>"
                        ])
                    else:
                        tokenized_row.append(field if field in vocabulary else "<OOV>")
                tokenized_rows.append(tokenized_row)
    return tokenized_rows


def save_vocabulary(vocabulary, output_file_path="../data/processed/vocabulary.txt"):
    """Saves the vocabulary to a text file, one word per line."""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for word in sorted(list(vocabulary)):
            f.write(f"{word}\n")
    print(f"Vocabulary saved to: {output_file_path}")


def save_tokenized_data(tokenized_data, output_file_path="../data/processed/tokenized_log.txt"):
    """Saves the tokenized data to a text file, with each row's tokens joined by spaces."""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for row in tokenized_data:
            f.write(" ".join(row) + "\n")
    print(f"Tokenized data saved to: {output_file_path}")


# Example Usage (assuming you have two files named 'your_log_file1.csv' and 'your_log_file2.csv'):
if __name__ == "__main__":
    log_files = ["../data/processed/day8Filtrato.csv", "../data/processed/day9Filtrato.csv"]  # Replace with the actual paths to your files

    try:
        vocabulary, word_counts = create_word_level_vocabulary_from_files(log_files)
        print("Vocabulary:", vocabulary)
        print("\nWord Counts:", word_counts)

        # Save the vocabulary
        save_vocabulary(vocabulary)

        tokenized_log = tokenize_word_level_from_files(log_files, vocabulary)
        print("\nTokenized Log (Word Level):")
        for row in tokenized_log:
            print(row)

        # Save the tokenized data
        save_tokenized_data(tokenized_log)

    except FileNotFoundError as e:
        print(f"Error: One or more files not found: {e}")