import os
import csv
import pandas as pd

# Number of seconds in a day
SECONDS_PER_DAY = 86400


def sec2day(seconds):
    """
    Convert a timestamp in seconds to the corresponding whole day.

    Args:
        seconds (int or str): Timestamp in seconds.

    Returns:
        int: Whole day count since epoch.
    """
    day = int(seconds) // SECONDS_PER_DAY
    return day


def split_by_day(log_filename, out_dir, keep_days=None):
    """
    Split a large authentication log file into daily CSV files for specific days.

    Args:
        log_filename (str): Path to the raw log file.
        out_dir (str): Directory where the split files will be saved.
        keep_days (list of int): List of days to retain and process.

    Notes:
        - Only lines with users starting with "U" and whose day is in `keep_days` are processed.
        - Lines are grouped and written into one file per day.
        - Stops processing once the day exceeds the maximum of `keep_days`.
    """

    # Create output directory if it doesn't exist
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    current_day = -1  # Tracks the currently processed day

    def get_filename(day):
        # Helper function to generate output file path for a given day
        return os.path.join(out_dir, f"{day}.csv")

    out_file = None
    with open(log_filename, encoding="utf8") as f:
        for line in f:
            # Parse line: expected format "timestamp,user,..."
            split_line = line.strip().split(",", maxsplit=2)
            sec = split_line[0]
            day = sec2day(sec)
            user = split_line[1]

            # Skip further processing if day exceeds max of keep_days
            if day > max(keep_days):
                break

            # Skip if day not in keep_days or user does not start with 'U'
            if not (day in keep_days and user.startswith("U")):
                continue

            # Handle file transition when the day changes
            if day > current_day:
                current_day = day
                print(f"Processing day {current_day}...")

                # Close previous day's file if open
                try:
                    out_file.close()
                except AttributeError:
                    pass

                # Open a new file for the new day
                out_file = open(get_filename(current_day), "w", encoding="utf8")

            elif day < current_day:
                # This shouldn't happen if log is sorted by time
                raise RuntimeError

            # Write the line to the current day's file
            out_file.write(line)

    # Close the last open file, if any
    if out_file is not None:
        out_file.close()


def add_redteam_to_log(input_file, output_file, red_team):
    """
    Add a binary flag to log entries indicating if the entry matches known red team activity.

    Args:
        input_file (str): Path to input daily log file (CSV).
        output_file (str): Path to output file with red team annotation.
        red_team (str): Path to red team activity file (CSV with no header).

    Process:
        - Creates a unique match key for each row in both datasets using selected columns.
        - Compares and flags log rows that match red team activity.
        - Appends a new column 'is_redteam' (1 if match found, 0 otherwise).
    """

    # Define expected columns in logs and red team file
    log_columns = ['time', 'source user@domain', 'destination user@domain', 'source computer',
                   'destination computer', 'authentication type', 'logon type',
                   'authentication orientation', 'success/failure']
    red_columns = ['time', 'source user@domain', 'source computer', 'destination computer']

    # Load red team activity data and assign column names
    red_team = pd.read_csv(red_team, header=None)
    red_team.columns = red_columns

    # Load authentication logs
    logs = pd.read_csv(input_file, header=None)
    logs.columns = log_columns

    # Define columns to match entries
    shared_columns = ['time', 'source user@domain', 'source computer', 'destination computer']

    # Create composite keys for comparison
    red_team['match_key'] = red_team[shared_columns].astype(str).agg('|'.join, axis=1)
    logs['match_key'] = logs[shared_columns].astype(str).agg('|'.join, axis=1)

    # Mark matches in logs with 1 (red team activity), 0 otherwise
    logs['is_redteam'] = logs['match_key'].isin(set(red_team['match_key'])).astype(int)

    # Drop helper column used for comparison
    logs.drop(columns='match_key', inplace=True)

    # Save annotated logs to CSV
    logs.to_csv(output_file, index=False)


def split_user_domain_fields(infile_path, outfile_path):
    """
    Splits 'user@domain' fields into separate 'user' and 'domain' columns for both source and destination.

    Args:
        infile_path (str): Path to input CSV file.
        outfile_path (str): Path to output CSV file.

    Note:
        Operates on columns at index 1 and 2: 'source user@domain' and 'destination user@domain'.
    """

    with open(infile_path, "r", encoding="utf8", newline='') as infile, \
            open(outfile_path, "w", encoding="utf8", newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Split source and destination user@domain into two fields each
            user1, domain1 = row[1].split("@") if "@" in row[1] else (row[1], "")
            user2, domain2 = row[2].split("@") if "@" in row[2] else (row[2], "")

            # Construct new row with separated fields
            new_row = [row[0], user1, domain1, user2, domain2] + row[3:]
            writer.writerow(new_row)


def generateCSVTest(input_csv, seq_len, output_csv):
    """
    Generate a CSV test dataset by extracting sequences around red team events.

    Args:
        input_csv (str): Input CSV file with an 'is_redteam' column as the last column.
        seq_len (int): Number of lines before a red team event to include.
        output_csv (str): Output CSV file path.

    Process:
        - Finds rows marked as red team activity (last column == '1').
        - For each such row, extracts the previous `seq_len` rows and the row itself.
        - Eliminates duplicates and writes the subset to output.
    """

    # Read entire CSV into memory for index-based access
    with open(input_csv, "r", encoding="utf8", newline='') as infile:
        reader = list(csv.reader(infile))

    selected_indices = set()  # Store unique indices to keep

    # Identify red team event rows and backtrack `seq_len` rows before each
    for i, row in enumerate(reader):
        if row and row[-1] == '1':
            start = max(0, i - seq_len)
            for j in range(start, i + 1):
                selected_indices.add(j)

    # Extract the relevant rows and write to output
    filtered_rows = [reader[i] for i in sorted(selected_indices)]

    with open(output_csv, "w", encoding="utf8", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(filtered_rows)



if __name__ == '__main__':
    #path to the raw file
    INPUT_FILE = '../data/raw/auth.txt'
    #path to the directory where the single days will be saved
    OUTPUT_DIR = '../data/processed/xdays'

    #selecting the days to keep, day 7 for training/val and day 8 for test
    split_by_day(INPUT_FILE,OUTPUT_DIR, keep_days=[7,8])
    #splitting the user@domain field for the day 7
    split_user_domain_fields("../data/processed/7withred.csv", "../data/processed/7withredSplit.csv")

    #create a new file (day 8) with labelled logs
    add_redteam_to_log("../data/processed/xdays/8.csv","../data/processed/8withred.csv","../data/raw/redteam.txt")
    # splitting the user@domain field for the day 8
    split_user_domain_fields("../data/processed/8withred.csv", "../data/processed/8withredSplit.csv")
    #Generate a CSV test dataset by extracting sequences around red team events
    generateCSVTest("../data/processed/8withredSplit.csv",10,"../data/processed/8Test.csv")




