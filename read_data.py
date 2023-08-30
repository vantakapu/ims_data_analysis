import os
import re
import pandas as pd

class DataProcessor:
    
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.data = {}
        self.dataframe = None
    
    def read_data_from_files(self, progress_callback=None, total_folders=0):
        folder_count = 0
        for subdir, dirs, files in os.walk(self.rootdir):
            for dir in dirs:
                folder_path = os.path.join(subdir, dir)
                folder_data = []
                for file in os.listdir(folder_path):
                    if file.endswith(".txt"):
                        filepath = os.path.join(folder_path, file)
                        with open(filepath, "r") as f:
                            raw_df = pd.read_csv(f, delimiter='\t', skiprows=5, header=None, names=["data"])
                            folder_data.append(raw_df)
                if folder_data:
                    self.data[dir] = pd.concat(folder_data, axis=1)
                
                # Update the progress
                folder_count += 1
                if progress_callback:
                    progress_callback(folder_count, total_folders)
    
    def clean_data(self):
        df = pd.concat(self.data.values(), axis=1, keys=self.data.keys())
        rows_to_ignore = list(range(1, 9))
        df = df.drop(rows_to_ignore)
        first_row_str = df.iloc[0].to_string(index=False)
        rip_values = re.findall(r'RIP=(\d+)', first_row_str)
        df.iloc[0] = rip_values
        retention_row_str = df.iloc[2].to_string(index=False)
        retention_values = re.findall(r'Retention Time: (\d+)', retention_row_str)
        df.iloc[2] = retention_values
        df = df.drop([df.index[3], df.index[1]])
        df = df.replace(',', '.', regex=True)
        df = df.reset_index(drop=True)
        self.dataframe = df
    
    def process_data(self, progress_callback=None, total_folders=0):
        self.read_data_from_files(progress_callback, total_folders)
        self.clean_data()
        return self.dataframe
    
    def process_all(self):
        return self.process_data()
