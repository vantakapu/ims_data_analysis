import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import argrelextrema



import read_data as rd
# import data_reduction as dt




rootdir = "C:\\Users\\ravindhar.vantakapu\\Desktop\\Trigger Sample 230324112124"
processor = rd.DataProcessor(rootdir)
raw_df = processor.process_data()
# raw_df2.to_csv('raw.csv')
raw_df.info()
raw_df
raw_df2 = raw_df.copy()
        
        
        
        
        
        
        
        
            
class DataFrameProcessor:

    @staticmethod
    def process_dataframe(df1):
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.droplevel(1)
        
        raw_df_trans = df1.transpose()
        raw_df_trans['group'] = raw_df_trans.index.str.extract('_group_(\d+)', expand=False)
        raw_df_trans = raw_df_trans.dropna(subset=['group'])
        raw_df_trans['group'] = raw_df_trans['group'].astype(int)
        cols = ['group']  + [col for col in raw_df_trans.columns if col != 'group']
        raw_df_trans = raw_df_trans[cols]
        raw_df_trans.index = raw_df_trans.index.to_series().str.split('_group_', expand=True)[0].squeeze()
        final_df = raw_df_trans.rename(columns={0: 'RIP', 1: 'retention_time'})
        final_df['retention_time'] = pd.to_numeric(final_df['retention_time'])
        final_df = final_df[final_df['retention_time'] <= 240]
        return final_df
    
    @staticmethod
    def plot_heatmap_by_group(df):
        
        step = 10
        df = df.apply(pd.to_numeric)
        group_indices = df.index.unique()

        for group_index in group_indices:
            group_df = df.loc[group_index].sort_values(by='retention_time')
            group_data = group_df.iloc[:, 3:]

            # Plotting the heatmap for the current group
            plt.figure(figsize=(10, 6))
            sns.heatmap(group_data, cmap="YlGnBu", cbar=False)  # Use 'jet' colormap

            # Set the labels for x-axis and y-axis
            plt.xlabel('Spectras')
            plt.ylabel('Retention Time')
            
            # Set the y-axis tick labels as the 'retention_time' values
            y_ticks = range(1, len(group_df) + 1, step)
            plt.yticks(range(len(group_df))[::step], y_ticks)
            
            
            # Set the title for the heatmap
            plt.title(f'Heatmap of Intensity for {group_index}')
            plt.legend()

            # Display the heatmap
            plt.show()

    

    @staticmethod
    def baseline_correction(df):
        df = df.apply(pd.to_numeric)
        pure_noise_rows = df[df['retention_time'].between(1, 20)].index
        min_intensity = df.loc[pure_noise_rows, :].iloc[:, 3:].min()
        corrected_df = df.copy()
        corrected_df.iloc[:, 3:] = corrected_df.iloc[:, 3:] - min_intensity
        return corrected_df

    @staticmethod
    def smoothing(df):
        window_size = 3
        for col in df.columns[3:]:
            df[col] = df[col].rolling(window_size, center=True).mean()
        return df

    @staticmethod
    def find_peaks(dataframe, neighborhood_size, threshold):
        peak_indices = dataframe.iloc[:, 3:].apply(lambda col: argrelextrema(col.values, np.greater, order=neighborhood_size)[0])
        peaks_df = pd.DataFrame(columns=dataframe.columns, index=dataframe.index)
        peaks_df.iloc[:, :3] = dataframe.iloc[:, :3]
        for col in dataframe.columns[3:]:
            peaks_df[col].iloc[peak_indices[col]] = np.where(dataframe[col].iloc[peak_indices[col]] > threshold, dataframe[col].iloc[peak_indices[col]], 0)
        peaks_df.fillna(0, inplace=True)
        peaks_df.index.name = dataframe.index.name
        return peaks_df
    
    
    
    
    @staticmethod
    def right_side_data(df, skip_columns=50):
        df = df.apply(pd.to_numeric)
        group = df['group']
        retention_time = df['retention_time']
        rip = df['RIP']
        df = df.iloc[:, 3:]
        column_sums = df.sum()
        max_sum_index = column_sums.idxmax()
        thick_line_index = df.columns.get_loc(max_sum_index)
        first_column_index = max(thick_line_index - 100, 0) + skip_columns
        new_df = df.iloc[:, first_column_index:]
        new_df.insert(0, 'group', group)
        new_df.insert(1, 'RIP', rip)
        new_df.insert(2, 'retention_time', retention_time)
        return new_df
    
    
    @staticmethod
    def align_peaks(peaks_df):
        # Extracting the first three columns to preserve them
        first_three_columns = peaks_df.iloc[:, :3]

        # Extracting relevant columns for peak analysis (excluding the first three columns)
        peaks_data_relevant = peaks_df.iloc[:, 3:]

        # Selecting the first sample
        first_sample_index = peaks_data_relevant.index[100]
        first_sample_data = peaks_data_relevant.loc[first_sample_index]

        # Calculating the average reference peak
        average_reference_peak = first_sample_data.iloc[800:1200].mean()

        # Function to find the shift required to align the peaks
        def find_shift(reference, signal):
            cross_corr = correlate(reference, signal, mode='full')
            shift = cross_corr.argmax() - (len(signal) - 1)
            return shift

        # Calculating the shift for each sample
        shifts = {}
        for idx, row in peaks_data_relevant.iterrows():
            sample_signal = row[800:2000]
            shift = find_shift(average_reference_peak, sample_signal)
            shifts[idx] = shift

        # Function to apply the shift to the entire spectrum
        def apply_shift(signal, shift):
            return np.roll(signal, shift)

        # Applying the shifts to align the peaks
        aligned_peaks_data = peaks_data_relevant.copy()
        drift_time_columns = aligned_peaks_data.columns
        for idx, shift in shifts.items():
            aligned_peaks_data.loc[idx, drift_time_columns] = apply_shift(aligned_peaks_data.loc[idx, drift_time_columns], shift)

        # Concatenating the first three columns and the aligned data
        aligned_peaks_data = pd.concat([first_three_columns, aligned_peaks_data], axis=1)
        
        return aligned_peaks_data
    
    @staticmethod
    def process_all(raw_df1):
        processed_df = DataFrameProcessor.process_dataframe(raw_df1)
        right_side_df = DataFrameProcessor.right_side_data(processed_df)
        heatmap = DataFrameProcessor.plot_heatmap_by_group(right_side_df)
        baseline_df = DataFrameProcessor.baseline_correction(right_side_df)
        smoothed_df = DataFrameProcessor.smoothing(baseline_df)

        # Setting the neighborhood_size and calculating threshold
        neighborhood_size = 8
        row_std = smoothed_df[1500].std()
        threshold = 3 * row_std

        peaks_df = DataFrameProcessor.find_peaks(smoothed_df, neighborhood_size, threshold)
        aligned_df = DataFrameProcessor.align_peaks(peaks_df)
        
        return heatmap, aligned_df
    







#if __name__ == '__main__':
    
find_peaks = DataFrameProcessor.process_all(raw_df2)
find_peaks
