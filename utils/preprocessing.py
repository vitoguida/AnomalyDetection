
import os
import csv
import pandas as pd

SECONDS_PER_DAY = 86400

def sec2day(seconds):
    """Seconds to number of whole days."""
    day = int(seconds) // SECONDS_PER_DAY
    return day

def split_by_day(log_filename, out_dir, keep_days=None):
    """Split a raw LANL log file into separate days based on the timestamp.

    Also filters out non-user activity and splits the source/destination
    user/domain.
    """
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    current_day = -1

    def get_filename(day):
        return os.path.join(out_dir, f"{day}.csv")

    out_file = None
    with open(log_filename, encoding="utf8") as f:
        for line in f:
            split_line = line.strip().split(",", maxsplit=2)
            sec = split_line[0]
            day = sec2day(sec)
            user = split_line[1]

            if day > max(keep_days):
                break

            if not (day in keep_days and user.startswith("U")):
                continue

            if day > current_day:
                current_day = day
                print(f"Processing day {current_day}...")
                try:
                    out_file.close()
                except AttributeError:
                    pass

                out_file = open(get_filename(current_day), "w", encoding="utf8")
            elif day < current_day:
                raise RuntimeError
            else:
                pass

            out_file.write(line)
    if not out_file is None:
        out_file.close()

"""def process_logfiles_for_training(auth_file, red_file, days_to_keep, output_dir, sample_output_dir, test_output_dir):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        split_by_day(auth_file, tmpdir, keep_days=days_to_keep)

        for day in days_to_keep:
            infile = os.path.join(tmpdir, f"{day}.csv")
            outfile = os.path.join(output_dir, f"{day}.csv")
            add_redteam_to_log(day, infile, outfile, red_file)
            # Generate sample and test files by copying the outfile (full data)
            sample_outfile = os.path.join(sample_output_dir, f"{day}.csv")
            generate_subset(outfile, sample_outfile, 10000)
            test_outfile = os.path.join(test_output_dir, f"{day}.csv")
            generate_subset(outfile, test_outfile, 200)

        # Copy the first 10000 lines of redteam.txt and auth.txt to the test directory
        red_outfile = os.path.join(test_output_dir, "redteam.txt")
        with open(red_file, "r", encoding="utf8") as in_file, open(red_outfile, "w", encoding="utf8") as out_file:
            for i, line in enumerate(in_file):
                out_file.write(line)
                if i >= 10000:
                    break

        auth_outfile = os.path.join(test_output_dir, "auth_head.txt")
        with open(auth_file, "r", encoding="utf8") as in_file, open(auth_outfile, "w", encoding="utf8") as out_file:
            for i, line in enumerate(in_file):
                out_file.write(line)
                if i >= 10000:
                    break

        # Copy the first 100 lines of the raw day 8 data to the test directory
        if 8 in days_to_keep:
            raw_day_8 = os.path.join(tmpdir, "8.csv")
            raw_day_8_outfile = os.path.join(test_output_dir, "raw_8_head.csv")
            with open(raw_day_8, "r", encoding="utf8") as in_file, open(
                    raw_day_8_outfile, "w", encoding="utf8"
            ) as out_file:
                for i, line in enumerate(in_file):
                    out_file.write(line)
                    if i >= 100:
                        break"""


def add_redteam_to_log(input_file, output_file, red_team):
    log_columns = ['time', 'source user@domain', 'destination user@domain', 'source computer', 'destination computer',
                   'authentication type', 'logon type', 'authentication orientation', 'success/failure']
    red_columns = ['time', 'source user@domain', 'source computer', 'destination computer']
    red_team = pd.read_csv(red_team,header=None)
    red_team.columns = red_columns

    logs = pd.read_csv(input_file, header=None)
    logs.columns = log_columns

    # Define shared columns
    shared_columns = ['time', 'source user@domain', 'source computer', 'destination computer']

    # Create keys for comparison
    red_team['match_key'] = red_team[shared_columns].astype(str).agg('|'.join, axis=1)
    logs['match_key'] = logs[shared_columns].astype(str).agg('|'.join, axis=1)

    # Mark red team entries in logs with 1 (match) and 0 (no match)
    logs['is_redteam'] = logs['match_key'].isin(set(red_team['match_key'])).astype(int)


    # Remove the helper column
    logs.drop(columns='match_key', inplace=True)

    # Save to CSV
    logs.to_csv(output_file, index=False)


def split_user_domain_fields(infile_path, outfile_path):
    with open(infile_path, "r", encoding="utf8", newline='') as infile, \
            open(outfile_path, "w", encoding="utf8", newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Splitta i campi 1 e 2 (secondo e terzo, indice 1 e 2)
            user1, domain1 = row[1].split("@") if "@" in row[1] else (row[1], "")
            user2, domain2 = row[2].split("@") if "@" in row[2] else (row[2], "")

            # Costruisce la nuova riga
            new_row = [row[0], user1, domain1, user2, domain2] + row[3:]
            writer.writerow(new_row)

import pandas as pd



# Esempio di utilizzo:
# filter_sequences("input_file.csv", seq_len=10, output_csv="output_file.csv")

def generateCSVTest(input_csv, seq_len, output_csv):
    with open(input_csv,"r", encoding="utf8", newline='') as infile:
        reader = list(csv.reader(infile))  # convertiamo in lista per indicizzazione

    selected_indices = set()

    for i, row in enumerate(reader):
        if row and row[-1] == '1':
            start = max(0, i - seq_len)
            for j in range(start, i + 1):  # include anche la riga stessa con '1'
                selected_indices.add(j)

    filtered_rows = [reader[i] for i in sorted(selected_indices)]

    with open(output_csv,"w", encoding="utf8", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(filtered_rows)




if __name__ == '__main__':
    #INPUT_FILE = '../data/raw/auth.txt'
    #OUTPUT_DIR = '../data/processed/xdays'
    #split_by_day(INPUT_FILE,OUTPUT_DIR, keep_days=[6,7,8])
    #add_redteam_to_log("../data/processed/xdays/8.csv","../data/processed/8withred.csv","../data/raw/redteam.txt")
    #split_user_domain_fields("../data/processed/7withred.csv","../data/processed/7withredSplit.csv")
    generateCSVTest("../data/processed/8withredSplit.csv",10,"../data/processed/8Test.csv")




