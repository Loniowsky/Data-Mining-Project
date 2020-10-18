import pandas as pd


class DataLoader:

    @staticmethod
    def from_file(file_path: str):
        return pd.read_csv(file_path)
