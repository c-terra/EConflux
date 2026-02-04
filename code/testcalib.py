#HIIIIIIIIII CHRISSSSSSSSSSSS



import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gmean, hmean
from scipy.interpolate import griddata


class DataMapper:
    def __init__(self, num_neighbors=5, max_distance_xy=5.0, max_distance_z=0.5, ckdMethod='num', njobs=1):
        self.num_neighbors = num_neighbors
        self.max_distance_xy = max_distance_xy
        self.max_distance_z = max_distance_z
        self.ckdMethod = ckdMethod   # 'num' for original implementation, 'dist' to average all points within the max_distance, 'auto' to automatically determine the max_distance
        self.njobs = njobs

    def mapXYZ(self, mesh_df, dataframes, dfnames):
        mesh_df = mesh_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z'])
        result_df = mesh_df.copy()
        distance_data = []
        
        maxDist = np.sqrt((self.max_distance_xy**2) + self.max_distance_z**2)
        
        for idx, df in enumerate(dataframes):
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z'])
            if all(col in df.columns for col in ['x', 'y', 'Z']):
                
                tree = cKDTree(df[['x', 'y', 'Z']].values)
                
                if self.ckdMethod == 'num':
                    if self.num_neighbors > 1:
                        distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, distance_upper_bound=maxDist, workers=self.njobs)
                        
                    else:
                        distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, workers=self.njobs)

                                        
                    for i, (dists, idxs) in enumerate(zip(distances, indices)):
                        if self.num_neighbors > 1:
                            for dist in dists:
                                distance_data.append({'x': mesh_df.iloc[i]['x'], 'y': mesh_df.iloc[i]['y'], 'Z': mesh_df.iloc[i]['Z'], 'distance': dist})
                        else:
                            distance_data.append({'x': mesh_df.iloc[i]['x'], 'y': mesh_df.iloc[i]['y'], 'Z': mesh_df.iloc[i]['Z'], 'distance': dists})

                            
                elif self.ckdMethod == 'dist':
                    
                    indices = tree.query_ball_point(mesh_df[['x', 'y', 'Z']].values, maxDist, workers=self.njobs)
                    
                    for row_idx, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'Z': mesh_df.loc[row_idx, 'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2) +
                                                                      ((mesh_df.loc[row_idx, 'Z'] - df.loc[index, 'Z'])**2)
                                                                      )})
                    
                elif self.ckdMethod == 'auto':
                    
                    distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, workers=self.njobs)
                    minDist = []
                    for dists in distances:
                        minDist.append(min(dists))
                    distThresh = max(minDist)
                    
                    indices = tree.query_ball_point(mesh_df[['x', 'y', 'Z']].values, distThresh, workers=self.njobs)
                    
                    # idxarr = np.array(idxs)
                    # points = mesh_df[idxarr]

                    
                    
                    for row_idx, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'Z': mesh_df.loc[row_idx,'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2) +
                                                                      ((mesh_df.loc[row_idx, 'Z'] - df.loc[index, 'Z'])**2)
                                                                      )})
                
                if self.num_neighbors > 1:
                
                    for col in df.columns:
                        if col not in ['x', 'y', 'Z']:
                            arithmetic_means, geometric_means, harmonic_means = [], [], []
                                
                            for row in indices:
                                if len(df) not in row:
                                    valid_values = df.loc[row, col].values
                                else:
                                    valid_values = []
        
                                if len(valid_values) > 0:
                                    arithmetic_means.append(valid_values.mean())
                                    geometric_means.append(gmean(valid_values) if np.all(valid_values > 0) else np.nan)
                                    harmonic_means.append(hmean(valid_values) if np.all(valid_values > 0) else np.nan)
                                else:
                                    arithmetic_means.append(np.nan)
                                    geometric_means.append(np.nan)
                                    harmonic_means.append(np.nan)  
                        
                            result_df[f'{dfnames[idx]}_{col}_arith'] = arithmetic_means
                            if 'inph' in col:
                                result_df[f'{dfnames[idx]}_{col}_geom'] = arithmetic_means
                                result_df[f'{dfnames[idx]}_{col}_harm'] = arithmetic_means
                            else:
                                result_df[f'{dfnames[idx]}_{col}_geom'] = geometric_means
                                result_df[f'{dfnames[idx]}_{col}_harm'] = harmonic_means
        
                else:
                
                    for col in df.columns:
                        if col not in ['x', 'y', 'Z']:
                            nearest_value = []
                            
                            for i in indices:
                                nearest_value.append(df.loc[i, col])
                                
                            result_df[f'{dfnames[idx]}_{col}'] = nearest_value


        return result_df, distance_data

    def mapXY(self, mesh_df, dataframes, dfnames):
        mesh_df = mesh_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
        result_df = mesh_df.copy()
        distance_data = []
        
        for idx, df in enumerate(dataframes):
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
            if all(col in df.columns for col in ['x', 'y']):
                
                tree = cKDTree(df[['x', 'y']].values)
                
                if self.ckdMethod == 'num':
                    if self.num_neighbors > 1:
                        distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, distance_upper_bound=self.max_distance_xy, workers=self.njobs)
                        
                    else:
                        distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, workers=self.njobs)

                                        
                    for i, (dists, idxs) in enumerate(zip(distances, indices)):
                        if self.num_neighbors > 1:
                            for dist in dists:
                                distance_data.append({'x': mesh_df.iloc[i]['x'], 'y': mesh_df.iloc[i]['y'], 'distance': dist})
                        else:
                            distance_data.append({'x': mesh_df.iloc[i]['x'], 'y': mesh_df.iloc[i]['y'], 'distance': dists})

                            
                elif self.ckdMethod == 'dist':
                    
                    indices = tree.query_ball_point(mesh_df[['x', 'y']].values, self.max_distance_xy, workers=self.njobs)
                    
                    for row_idx, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'Z': mesh_df.loc[row_idx, 'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2)
                                                                      )})
                    
                elif self.ckdMethod == 'auto':
                    
                    distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, workers=self.njobs)
                    minDist = []
                    for dists in distances:
                        minDist.append(min(dists))
                    distThresh = max(minDist)
                    
                    indices = tree.query_ball_point(mesh_df[['x', 'y']].values, distThresh, workers=self.njobs)

                    
                    for row_idx, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2)
                                                                      )})
                
                if self.num_neighbors > 1:
                
                    for col in df.columns:
                        if col not in ['x', 'y']:
                            arithmetic_means, geometric_means, harmonic_means = [], [], []
                                
                            for row in indices:
                                if len(df) not in row:
                                    valid_values = df.loc[row, col].values
                                else:
                                    valid_values = []
        
                                if len(valid_values) > 0:
                                    arithmetic_means.append(valid_values.mean())
                                    geometric_means.append(gmean(valid_values) if np.all(valid_values > 0) else np.nan)
                                    harmonic_means.append(hmean(valid_values) if np.all(valid_values > 0) else np.nan)
                                else:
                                    arithmetic_means.append(np.nan)
                                    geometric_means.append(np.nan)
                                    harmonic_means.append(np.nan)  
                        
                            if dfnames[idx] == '':
                                result_df[f'{col}_arith'] = arithmetic_means
                                if 'inph' in col:
                                    result_df[f'{col}_geom'] = arithmetic_means
                                    result_df[f'{col}_harm'] = arithmetic_means
                                else:
                                    result_df[f'{col}_geom'] = geometric_means
                                    result_df[f'{col}_harm'] = harmonic_means
                            else:
                                result_df[f'{dfnames[idx]}_{col}_arith'] = arithmetic_means
                                if 'inph' in col:
                                    result_df[f'{dfnames[idx]}_{col}_geom'] = arithmetic_means
                                    result_df[f'{dfnames[idx]}_{col}_harm'] = arithmetic_means
                                else:
                                    result_df[f'{dfnames[idx]}_{col}_geom'] = geometric_means
                                    result_df[f'{dfnames[idx]}_{col}_harm'] = harmonic_means
        
                else:
                
                    for col in df.columns:
                        if col not in ['x', 'y']:
                            nearest_value = []
                            
                            for i in indices:
                                nearest_value.append(df.loc[i, col])
                                
                            if dfnames[idx] == '':
                                result_df[f'{col}'] = nearest_value
                            else:
                                result_df[f'{dfnames[idx]}_{col}'] = nearest_value
                            
        return result_df, distance_data
    
    def mapXY_test(self, mesh_df, dataframes, dfnames):
        mesh_df = mesh_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
        result_df = mesh_df.copy()
        mesh_points = mesh_df[['x', 'y']].values
        distance_data = []
    
        for idx, df in enumerate(dataframes):
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
            if not all(col in df.columns for col in ['x', 'y']):
                continue
    
            df_points = df[['x', 'y']].values
            tree = cKDTree(df_points)
    
            distances, indices = tree.query(mesh_points, k=self.num_neighbors)
            
            # Ensure 2D shape for single neighbor case
            if self.num_neighbors == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]
            
            within_max_distance = distances <= self.max_distance_xy
            
            # Track distances
            mesh_x = mesh_points[:, 0]
            mesh_y = mesh_points[:, 1]
            for i in range(len(mesh_points)):
                for dist in distances[i][within_max_distance[i]]:
                    distance_data.append({'x': mesh_x[i], 'y': mesh_y[i], 'distance': dist})
    
            # Compute means for each column
            for col in df.columns:
                if col in ['x', 'y']:
                    continue
    
                df_col_values = df[col].values
                arth_means = np.full(len(mesh_points), np.nan)
                geom_means = np.full(len(mesh_points), np.nan)
                harm_means = np.full(len(mesh_points), np.nan)
    
                for i in range(len(mesh_points)):
                    valid_idx = indices[i][within_max_distance[i]]
                    if valid_idx.size == 0:
                        continue
                    values = df_col_values[valid_idx]
                    if values.size > 0:
                        arth_means[i] = np.mean(values)
                        if np.all(values > 0):
                            geom_means[i] = gmean(values)
                            harm_means[i] = hmean(values)
    
                result_df[f'{col}_arth'] = arth_means
                result_df[f'{col}_geom'] = geom_means
                result_df[f'{col}_harm'] = harm_means
    
        return result_df, distance_data

    
    def create_mean_dfs(self, df):
        arth_columns = [col for col in df.columns if '_arith' in col]
        geom_columns = [col for col in df.columns if '_geom' in col]
        harm_columns = [col for col in df.columns if '_harm' in col]
        other_columns = [col for col in df.columns if col not in arth_columns + geom_columns + harm_columns]
        
        df_arth = df[other_columns + arth_columns].rename(columns={col: col.replace('_arith', '') for col in arth_columns})
        df_geom = df[other_columns + geom_columns].rename(columns={col: col.replace('_geom', '') for col in geom_columns})
        df_harm = df[other_columns + harm_columns].rename(columns={col: col.replace('_harm', '') for col in harm_columns})
        
        return df_arth, df_geom, df_harm

    def plot_distance_histogram(self, distance_data, xyAnn = (0.7,0.8)):
        distM = pd.DataFrame(distance_data)
        
        mse = np.sqrt(np.mean(distM['distance']**2))
        mean_distance = np.mean(distM['distance'])
        std_distance = np.std(distM['distance'])
        
        plt.hist(distM['distance'], bins=50, edgecolor='black')
        plt.xlabel('Displacement (m)')
        plt.ylabel('Frequency')
        
        plt.annotate(
            f'N = {len(distM)}\nMSE: {mse:.2f}\nMean: {mean_distance:.2f}\nStd: {std_distance:.2f}',
            xy = xyAnn, 
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
        
        plt.show()

    def filter_and_rename(self, eca_arth, minCut=5, maxCut=5, MaxECa=50, minECa=0, keepInphase=False):
        # Compute xMin and xMax for the 'X' column specifically
        xMin = eca_arth['X'].min() + minCut
        xMax = eca_arth['X'].max() - maxCut
        if 'HCP0.5' in eca_arth:
            eca_arth = eca_arth[
                (eca_arth['X'] > xMin) & (eca_arth['X'] < xMax) &
                (eca_arth['HCP1.0'] < MaxECa) &
                (eca_arth['PRP1.1'] < MaxECa) &
                (eca_arth['HCP0.5'] < MaxECa) &
                (eca_arth['PRP0.6'] < MaxECa)]
            subs = [
                'X', 'HCP0.5', 'PRP0.6', 
                'HCP1.0', 'PRP1.1']
            
            eca_arth_subs = eca_arth[subs]
    
            # Rename columns
            eca_df = eca_arth_subs.rename(columns={
                'HCP0.5': 'HCP0.5f9000h.15',
                'PRP0.6': 'PRP0.6f9000h.15',
                'HCP1.0': 'HCP1.0f9000h.15', 
                'PRP1.1': 'PRP1.1f9000h.15'})
        else:
            # Filter the DataFrame with corrected conditions
            eca_arth = eca_arth[
                (eca_arth['X'] > xMin) & (eca_arth['X'] < xMax) &
                (eca_arth['HCP1.0'] < MaxECa) &
                (eca_arth['PRP1.1'] < MaxECa) &
                (eca_arth['HCP2.0'] < MaxECa) &
                (eca_arth['PRP2.1'] < MaxECa) &
                (eca_arth['HCP4.0'] < MaxECa) &
                (eca_arth['PRP4.1'] < MaxECa) &
                (eca_arth['HCP1.0'] > minECa) &
                (eca_arth['PRP1.1'] > minECa) &
                (eca_arth['HCP2.0'] > minECa) &
                (eca_arth['PRP2.1'] > minECa) &
                (eca_arth['HCP4.0'] > minECa) &
                (eca_arth['PRP4.1'] > minECa)
                ]
            # eca_arth['X'] = np.round(eca_arth['X'], 2)
    
        # Select subset of columns
        if keepInphase:
            subs = [
                'X', 'HCP1.0', 'HCP1.0_inph',
                'PRP1.1', 'PRP1.1_inph',
                'PRP2.1', 'PRP2.1_inph',
                'HCP2.0', 'HCP2.0_inph',
                'PRP4.1', 'PRP4.1_inph',
                'HCP4.0', 'HCP4.0_inph'
                ]
        else:
            subs = [
                'X', 'PRP1.1', 'PRP2.1',
                'HCP1.0', 'PRP4.1',
                'HCP2.0', 'HCP4.0', 
                ]
        
        eca_arth_subs = eca_arth[subs]
        
        # Rename columns
        eca_df = eca_arth_subs
        
        return eca_df
    
    def process_and_bin_data(self, eca_df, final_df, path, filename_ec, filename_eca,
                             nbins=300, xDis=0.5, zLayers=20, interpMethod='nearest',
                             useGeomspace=True):

        import numpy as np
        import pandas as pd
    
        # Extract values from eca_df and final_df
        eca = eca_df.values
        ec = np.array(final_df[['X', 'Z', 'Conductivity(mS/m)']])
        print('Initial length ECa', len(eca))
        print('Initial length EC', len(ec))
        print(np.min(ec[:, 2]))
    
        minX = np.round(np.min(eca[:, 0]), 1)
        maxX = np.round(np.max(eca[:, 0]), 1)
    
        # Crop the data based on the location of EMI calibration data
        ec = ec[np.where((ec[:, 0] >= minX) & (ec[:, 0] <= maxX)), :][0]
        print('After crop length ECa', len(eca))
        print('After crop length EC', len(ec))
    
        # Prepare the x, z, and conductivity data
        x = ec[:, 0]
        z = ec[:, 1]
        cond = ec[:, 2]
        xi = np.arange(np.min(x), np.max(x), xDis)
    
        # Choose spacing method for z
        z_min, z_max = np.min(z), np.max(z)
        if useGeomspace:
            if z_min >= 0 or z_max >= 0:
                raise ValueError("For geomspace, Z values must be negative (depths).")
            zi = -np.geomspace(abs(z_min), abs(z_max), zLayers)
        else:
            zi = np.linspace(z_min, z_max, zLayers)
    
        xi, zi = np.meshgrid(xi, zi)
        condi = griddata((x, z), cond, (xi, zi), method=interpMethod)
        x = np.unique(xi)
        z = np.unique(zi)
        condxz = np.array(np.meshgrid(x, z)).T.reshape(-1, 2)
        cond = condi.T.flatten()
        ec = np.concatenate((condxz, cond[:, None]), axis=1)
        ec = ec[~np.isnan(ec[:, 2]), :]
        ec = np.round(ec, 2)
        print('After meshing length ECa', len(eca))
        print('After meshing length EC', len(ec))
    
        midDepths = -np.unique(ec[:, 1])
        bins = np.round(np.linspace(minX, maxX, nbins + 1), 1)
        binID = np.digitize(ec[:, 0], bins)
    
        ertEC = np.empty((nbins, len(midDepths)))
        midDepthsr = midDepths[::-1]
    
        for i in range(nbins):
            for j in range(len(midDepths)):
                idepth = np.isclose(ec[:, 1], -midDepthsr[j], atol=1e-3)
                ibins = binID == (i + 1)
                valid_points = ec[idepth & ibins, 2]
                ertEC[i, j] = np.mean(valid_points) if valid_points.size > 0 else np.nan
    
        df_ec = pd.DataFrame(np.round(ertEC, 3), columns=[f'd{col:.3f}' for col in midDepthsr])
    
        # Binning for eca
        binID = np.digitize(eca[:, 0], bins)
        eca2 = np.zeros((nbins, eca.shape[1] - 1))
    
        for i in range(nbins):
            ie = binID == (i + 1)
            if np.sum(ie) > 0:
                eca2[i, :] = np.mean(eca[ie, 1:], axis=0)
    
        headers = eca_df.columns[1:].tolist()
        df_eca = pd.DataFrame(np.round(eca2, 1), columns=headers)
    
        rows_to_drop = df_eca.eq(0).any(axis=1)
        df_eca = df_eca[~rows_to_drop].reset_index(drop=True)
        df_ec = df_ec[~rows_to_drop].reset_index(drop=True)
    
        df_ec.to_csv(os.path.join(path, filename_ec), index=False)
        df_eca.to_csv(os.path.join(path, filename_eca), index=False)
    
        print('After binning length ECa', len(df_eca))
        print('After binning length EC', len(df_ec))
    
        return df_eca, df_ec
