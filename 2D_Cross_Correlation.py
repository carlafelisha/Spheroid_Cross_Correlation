import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def drop(file, threshold): 
    # Data cleaning: Remove rows with null 'TrackID'
    file.drop(file[file['TrackID'].isnull()].index, inplace=True)
    cells = file['TrackID'].unique()
    
    # Drop rows with false signals
    for cell in cells:
        cell_file = file[file['TrackID'] == cell]
        if len(cell_file) < threshold:
            file.drop(file[file['TrackID'] == cell].index, inplace=True)
    
    # Calculate center positions
    x_c, y_c = file['Position X'].mean(), file['Position Y'].mean()
    time = file['Time'].unique()
    
    for t in time:
        time_file = file[file['Time'] == t].copy()
        
        # Calculate 2D distance to center
        time_file.loc[:, '2D dis to C'] = np.sqrt((time_file['Position X'] - x_c) ** 2 + (time_file['Position Y'] - y_c) ** 2)
        
        # Calculate mean and standard deviation of the distances
        mean_dis = time_file['2D dis to C'].mean()
        std_dis = time_file['2D dis to C'].std()
        
        # Identify and exclude cells that are too far away from the center of the spheroid
        far_cells = time_file[time_file['2D dis to C'] > mean_dis + 3 * std_dis]
        ids = far_cells['TrackID'].unique()
        
        for id in ids:
            file.drop(file[(file['Time'] == t) & (file['TrackID'] == id)].index, inplace=True)
        
    return file

def stretched_exp(r, g, a, Lcorr):
        return (1 - a) * np.exp(-(r / Lcorr)**g) + a

def Lcorr_fitting(r, Cvv):
    ## Perform curve fitting
    params, covariance = curve_fit(stretched_exp, r, Cvv)
    std_dev_params = np.sqrt(np.diag(covariance))

    # Extracting individual parameters
    g = params[0]
    a = params[1]
    Lcorr = params[2]

    # Standard deviations
    std_dev_g = std_dev_params[0]
    std_dev_a = std_dev_params[1]
    std_dev_Lcorr = std_dev_params[2]
    return g, a, Lcorr, std_dev_g, std_dev_a, std_dev_Lcorr

def velocity_spat(file):
    cells = file['TrackID'].unique()
    file_velocity = []
    
    for cell in cells:
        cell_file = file[file['TrackID'] == cell].sort_values(by='Time').copy()
        cell_file = cell_file.drop_duplicates(subset="Time")
        
        # Calculate time difference
        cell_file.loc[:, 'dt'] = cell_file['Time'].diff()
        
        # Calculate velocity components
        cell_file.loc[:, 'vx'] = cell_file['Position X'].diff() / cell_file['dt']
        cell_file.loc[:, 'vy'] = cell_file['Position Y'].diff() / cell_file['dt']
        cell_file.loc[:, 'vz'] = cell_file['Position Z'].diff() / cell_file['dt']
        
        file_velocity.append(cell_file.dropna())
    
    file_v = pd.concat(file_velocity, ignore_index=True)
    time = sorted(file_v['Time'].unique())
    file_norm = []
    
    for t in time:
        time_file = file_v[file_v['Time'] == t].copy()
        
        # Relative Velocity (subtract mean velocity)
        vcx, vcy, vcz = time_file['vx'].mean(), time_file['vy'].mean(), time_file['vz'].mean()
        time_file.loc[:, 'vx'] -= vcx
        time_file.loc[:, 'vy'] -= vcy

        # Normalize velocity components in the xz plane
        mag_xz = np.sqrt(time_file['vx'] ** 2 + time_file['vz'] ** 2)
        time_file['vx_xz'] = time_file['vx'] / mag_xz
        time_file['vz_xz'] = time_file['vz'] / mag_xz

        # Normalize velocity components in the yz plane
        mag_yz = np.sqrt(time_file['vy'] ** 2 + time_file['vz'] ** 2)
        time_file['vy_yz'] = time_file['vy'] / mag_yz
        time_file['vz_yz'] = time_file['vz'] / mag_yz
        
        file_norm.append(time_file.dropna(subset=['vx_xz', 'vz_xz', 'vy_yz', 'vz_yz']))
    file_normalized = pd.concat(file_norm, ignore_index=True)

    # Velocity Filtering =====================
    # file_v = file_v[abs(file_v['vx']) > 0.2]
    # file_v = file_v[abs(file_v['vy']) > 0.2]
    # file_v = file_v[abs(file_v['vz']) > 0.2]
    #  =======================================
    
    return file_normalized

def correlation_spat(file, t):
    # Filter the file for the specified time point
    file_t = file[file['Time'] == t]
    pair_file_xz = pd.DataFrame()
    pair_file_yz = pd.DataFrame()

    dr_xz = []
    dr_yz = []
    inner_xz = []
    inner_yz = []

    # Iterate through pairs of rows to calculate distances and inner products
    for ind_i, row_i in file_t.iterrows():
        for ind_j, row_j in file_t.iterrows():
            if ind_i <= ind_j:
                distance_xz = np.sqrt((row_i['Position X'] - row_j['Position X']) ** 2 + (row_i['Position Z'] - row_j['Position Z']) ** 2)
                dr_xz.append(distance_xz)
                inner_product_xz = (row_i['vx_xz'] * row_j['vx_xz'] + row_i['vz_xz'] * row_j['vz_xz'])
                inner_xz.append(inner_product_xz)

                distance_yz = np.sqrt((row_i['Position Y'] - row_j['Position Y']) ** 2 + (row_i['Position Z'] - row_j['Position Z']) ** 2)
                dr_yz.append(distance_yz)
                inner_product_yz = (row_i['vy_yz'] * row_j['vy_yz'] + row_i['vz_yz'] * row_j['vz_yz'])
                inner_yz.append(inner_product_yz)
                
    # Create DataFrames with distances and inner products in XZ and YZ planes
    pair_file_xz['dr'] = dr_xz
    pair_file_xz['inner'] = inner_xz
    pair_file_yz['dr'] = dr_yz
    pair_file_yz['inner'] = inner_yz

    # Group inner product values into bins based on distance between neighbors in XZ and YZ planes
    pair_file_xz['bin'] = pd.cut(pair_file_xz['dr'], bins=np.linspace(pair_file_xz['dr'].min(), max(pair_file_xz['dr'].max(), pair_file_yz['dr'].max()), 100), include_lowest=True, duplicates='drop')
    pair_file_yz['bin'] = pd.cut(pair_file_yz['dr'], bins=np.linspace(pair_file_yz['dr'].min(), max(pair_file_xz['dr'].max(), pair_file_yz['dr'].max()), 100), include_lowest=True, duplicates='drop')

    # Calculate the average correlation within each bin in XZ and YZ planes
    bins_xz = sorted(pair_file_xz['bin'].unique())
    bins_yz = sorted(pair_file_yz['bin'].unique())
    corr_file_xz = pd.DataFrame()
    corr_file_yz = pd.DataFrame()
    dist_xz = []
    corr_xz = []
    dist_yz = []
    corr_yz = []

    for bin_xz in bins_xz:
        bin_file_xz = pair_file_xz[pair_file_xz['bin'] == bin_xz]
        dist_xz.append(bin_file_xz['dr'].mean())
        corr_xz.append(bin_file_xz['inner'].mean())

    for bin_yz in bins_yz:
        bin_file_yz = pair_file_yz[pair_file_yz['bin'] == bin_yz]
        dist_yz.append(bin_file_yz['dr'].mean())
        corr_yz.append(bin_file_yz['inner'].mean())

    # Create the final DataFrames with average distance and correlation values in XZ and YZ planes
    corr_file_xz['Distance'] = dist_xz
    corr_file_xz['Correlation'] = corr_xz
    corr_file_xz['Cumulative_Avg_Correlation'] = corr_file_xz['Correlation'].expanding().mean()
    corr_file_xz['Label'] = np.zeros(len(corr_file_xz))

    corr_file_yz['Distance'] = dist_yz
    corr_file_yz['Correlation'] = corr_yz
    corr_file_yz['Cumulative_Avg_Correlation'] = corr_file_yz['Correlation'].expanding().mean()
    corr_file_yz['Label'] = np.ones(len(corr_file_yz))

    # Pooling XZ and YZ data 
    both = pd.concat([corr_file_xz, corr_file_yz])
    bin_width = 1 #0.01
    bins = pd.interval_range(start=0, end=both['Distance'].max(), freq=bin_width, closed='left')
    both['distance_bin'] = pd.cut(both['Distance'], bins)
    bins_both = both['distance_bin'].cat.categories
    
    dist_both = []
    corr_both = []

    for bin in bins_both:
        bin_data = both[both['distance_bin'] == bin]
        if not bin_data.empty:
            data_xz = bin_data[bin_data['Label'] == 0]
            data_yz = bin_data[bin_data['Label'] == 1]
            total_size = len(data_xz) + len(data_yz)
            if total_size > 0:
                weight_xz = len(data_xz) / total_size
                weight_yz = len(data_yz) / total_size
                weighted_corr = (data_xz['Correlation'].mean() * weight_xz) + (data_yz['Correlation'].mean() * weight_yz)
            else:
                weighted_corr = 0

            corr_both.append(weighted_corr)
            dist_both.append(bin_data['Distance'].mean())  # Calculate mean distance for the bin

    both_file = pd.DataFrame({'Distance': dist_both, 'Correlation': corr_both})
    both_file['Cumulative_Avg_Correlation'] = both_file['Correlation'].expanding().mean()
    
    return both_file, corr_file_xz, corr_file_yz
    
def plot_velocity_corr(file, t, name, fitting=0):
    normalized_velocity_data = velocity_spat(file)
    _, corr_file_xz, corr_file_yz = correlation_spat(normalized_velocity_data, t)

    plt.figure(figsize=(8, 6))
    plt.ylim([-1, 1])
    plt.scatter(corr_file_xz['Distance'], corr_file_xz['Cumulative_Avg_Correlation'], color='blue', label='XZ Plane')
    plt.scatter(corr_file_yz['Distance'], corr_file_yz['Cumulative_Avg_Correlation'], color='red', label='YZ Plane')
    plt.xlabel('Distance between each cell pair (\u03bcm)')
    plt.ylabel('Correlation')
    plt.title(f'{name} at timeframe = {t}')
    plt.legend()
    if fitting:
        g, a, Lcorr, std_dev_g, std_dev_a, std_dev_Lcorr = Lcorr_fitting(corr_file_xz['Distance'], corr_file_xz['Cumulative_Avg_Correlation'])
        plt.plot(corr_file_xz['Distance'], stretched_exp(corr_file_xz['Distance'], g, a, Lcorr), label='Fitted curve XZ')
        print('XZ Fitting Parameters:')
        print(f'γ: {g}, {std_dev_g}')
        print(f'a: {a}, {std_dev_a}')
        print(f'Lcorr: {Lcorr}, {std_dev_Lcorr}')

        g, a, Lcorr, std_dev_g, std_dev_a, std_dev_Lcorr = Lcorr_fitting(corr_file_yz['Distance'], corr_file_yz['Cumulative_Avg_Correlation'])
        plt.plot(corr_file_yz['Distance'], stretched_exp(corr_file_yz['Distance'], g, a, Lcorr), label='Fitted curve YZ')
        print('\nYZ Fitting Parameters:')
        print(f'γ: {g}, {std_dev_g}')
        print(f'a: {a}, {std_dev_a}')
        print(f'Lcorr: {Lcorr}, {std_dev_Lcorr}')
    plt.show()

def plot_pooled_corr(file, t, name, fitting=0):
    normalized_velocity_data = velocity_spat(file)
    both_file, _, _ = correlation_spat(normalized_velocity_data, t)

    plt.figure(figsize=(10, 6))
    plt.ylim([-0.2,1])
    plt.plot(both_file['Distance'], both_file['Cumulative_Avg_Correlation'], color='b', marker='o', linestyle='None')
    plt.xlabel('Distance')
    plt.ylabel('Cumulative Average Correlation')
    plt.title(f'Pooled Correlation for {name} at timeframe = {t}')
    plt.grid(True)

    if fitting:
        try: 
            g, a, Lcorr, std_dev_g, std_dev_a, std_dev_Lcorr = Lcorr_fitting(both_file['Distance'], both_file['Cumulative_Avg_Correlation'])
            plt.plot(both_file['Distance'], stretched_exp(both_file['Distance'], g, a, Lcorr), label='Fitted curve')
            plt.text(0.6, -0.4, f'Lcorr = {Lcorr}\nstd_Lcorr = {std_dev_Lcorr}', ha='center', fontsize=10, wrap=True)
            plt.legend()

            save_dir = os.path.join(name)
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            save_path = os.path.join(save_dir, f'{t}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            return Lcorr, std_dev_Lcorr
        
        except Exception as e:
            print(f"Error during fitting: {e}")
            save_dir = os.path.join(name)
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            save_path = os.path.join(save_dir, f'{t}_nofit.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            return None, None
