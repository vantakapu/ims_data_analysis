import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import read_data as rd

from preprocess_steps import DataFrameProcessor
from scipy.signal import correlate
from data_reduction import align_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram
from preprocess_steps import find_peaks
    
    
    
    
    

# Access the data from read_data file
rootdir = "C:\\Users\\ravindhar.vantakapu\\Desktop\\Data_testing\\Series\\Series 5_Pat 15 VK + Pat 21 + 22_28.04.23"
object = rd.DataProcessor(rootdir)
raw_df2 = object.process_data()
raw_df2





    
find_peaks = DataFrameProcessor.process_all(raw_df2)
find_peaks






class ClusterProcessor():
    
    @staticmethod
    def cluster_analysis(df, max_distance):
        
        # Save the first three columns
        first_three_cols = df.iloc[:, :3].copy()
            
        # Drop the first three columns and transpose the remaining DataFrame
        df = df.drop(df.columns[:3], axis=1)
        transposed_df = df.transpose()

        # Convert the transposed DataFrame to numeric values and fill any NaN values with 0
        transposed_df = transposed_df.apply(pd.to_numeric, errors='coerce')
        transposed_df = transposed_df.fillna(0)

        # Perform hierarchical clustering with complete linkage and Euclidean distance
        linkage_matrix = linkage(transposed_df, method='complete', metric='euclidean')

        # Assign samples to clusters based on the distance threshold
        labels = fcluster(linkage_matrix, max_distance, criterion='distance')

        # Assign the cluster labels as column names
        df.columns = labels
        df = df.iloc[:, :].apply(pd.to_numeric, errors='coerce')

            
        combined_df = df.groupby(df.columns, axis=1).mean()
        combined_df = combined_df.fillna(0)

        # Add the first three columns back to the DataFrame
        combined_df = pd.concat([first_three_cols, combined_df], axis=1)
        
        combined_df.index.name = 'File'
            
        return combined_df
    
    
    
    @staticmethod
    def process_and_save_clustered_data(df):
    
        # Group by the "File" and "group" columns, then calculate the mean of non-zero values for the other columns
        result_df = df.groupby(['File', 'group']).apply(lambda x: x[x != 0].iloc[:, 3:].mean(numeric_only=True))
        
        result_df.sort_values(by='group', inplace=True)
            
        # Reset the index to make "File" and "group" regular columns
        result_df.reset_index(inplace=True)

        # Set "File" as the index and name it
        result_df.set_index('File', inplace=True)
        result_df.index.name = 'File'

        # Drop rows with NaN values
        result_df.dropna(inplace=True)
        result_df.reset_index(inplace=True)
    
        return result_df
    
    @staticmethod
    def process_all(df, max_distance):
        clustered_df = ClusterProcessor.cluster_analysis(df, max_distance)
        result = ClusterProcessor.process_and_save_clustered_data(clustered_df)
        
        return result
    
    
    
if __name__ == '__main__': 
    cluster_result = ClusterProcessor.process_all(find_peaks, max_distance=3)
    cluster_result
    cluster_result.to_excel('final_cluster_3.xlsx', index=False)
    










