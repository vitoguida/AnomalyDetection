"""
#quato file serve per estrarre gli accessi associati a un utente specifico
input_file_path = '../data/raw/auth.txt'  # Replace with the actual path to your text file

output_file_path = '../data/processed/user213.txt'  # Replace with the desired path for the output file
count = 0
try:
    with open(input_file_path, 'r') as infile:
        with open(output_file_path, 'w') as outfile:

            for line in infile:
                if count > 300:
                    break
                elif "U213" in line:
                    count = count + 1
                    outfile.write(line)
    print(f"Successfully extracted lines containing 'U3005' from '{input_file_path}' to '{output_file_path}'")
except FileNotFoundError:
    print(f"Error: The file '{input_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")"""

