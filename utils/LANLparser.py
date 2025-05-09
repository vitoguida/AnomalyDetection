import csv
import json
import os
import sys
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict

from tqdm import tqdm

class LANLReader:
    """Reader class for parsing LANL log data."""

    def __init__(self, file_pointer, normalized=True, has_red=False) -> None:
        self.field_names = [
            "time",
            "src_user",
            "src_domain",
            "dst_user",
            "dst_domain",
            "src_pc",
            "dst_pc",
            "auth_type",
            "logon_type",
            "auth_orient",
            "success",
        ]
        self._csv_field_names = [
            "time",
            "src_user@src_domain",
            "dst_user@dst_domain",
            "src_pc",
            "dst_pc",
            "auth_type",
            "logon_type",
            "auth_orient",
            "success",
        ]
        self.normalized = normalized
        self.has_red = has_red
        self.file_pointer = file_pointer

        if self.has_red:
            self._csv_field_names.append("is_red")
            self.field_names.append("is_red")

        if normalized:
            self._csv_field_names = self.field_names

    def __iter__(self):
        reader = csv.DictReader(self.file_pointer, fieldnames=self._csv_field_names)
        for row in reader:

            if row[self.field_names[-1]] is None or None in row:
                raise RuntimeError("The number of fields in the data does not match the settings provided.")

            data = row

            if not self.normalized:
                for u, d in zip(["src_user", "dst_user"], ["src_domain", "dst_domain"]):
                    data[u], data[d] = data[f"{u}@{d}"].split("@")
                    del data[f"{u}@{d}"]

            data["dst_user"] = data["dst_user"].replace("$", "")

            yield data