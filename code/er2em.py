# import libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gmean, hmean
from scipy.interpolate import griddata

# define DataMapper used for mapping ERI data to EMI data
class DataMapper:
    '''
        Initializes the mapper with spatial search constraints.

        NB: Column headers require the following format, horizontal co-planar (HCP) or 
        perpendicular (PRP), followed by the coil spacing (i.e., 1.0 or 1.1), and _inph 
        represents the associated in-phase column. 
        e.g. ['HCP1.0', 'HCP1.0_inph', 'PRP1.1', 'PRP1.1_inph']
        
        Parameters:
        - num_neighbors: Number of nearby points to consider for averaging.
        - max_distance_xy: Horizontal search radius (meters).
        - max_distance_z: Vertical search radius (meters).
        - ckdMethod: 'num' (fixed neighbors), 'dist' (radius-based),
                     'autoNum' (determine distance threshold and use number of k-nearest neighbors), 
                     or 'autoDist' (determine distance threshold and use all nearest neighbors within that theshold).
        - njobs: Number of CPU threads to use for parallel processing in scipy.ckdTree. 
                 Default is 1. Setting to -1 will use all available threads.
    
        collocate one or more source datasets (dataframes)
           onto a target mesh (mesh_df) using nearest neighbors / radius search, and optionally compute
           arithmetic/geometric/harmonic averages + distance diagnostics
           
    '''

    def __init__(self, num_neighbors=5, max_distance_xy=5.0, max_distance_z=0.5, ckdMethod='num', njobs=1): # standard kwargs for the DataMapper class
        self.num_neighbors = num_neighbors # number of neighbors spatially closest to the data
        self.max_distance_xy = max_distance_xy # maximum distance to be examined on the x-y plane
        self.max_distance_z = max_distance_z # maximum depth/elevation to be examined on the z axis
        self.ckdMethod = ckdMethod   # 'num' for original implementation, 'dist' to average all points within the max_distance, 'auto' to automatically determine the max_distance
        self.njobs = njobs   # Number of threads to use for parallel processing. Default is 1. Setting to -1 uses all available CPU threads.
    
    # defining mesh dataframe with x, y, Z geometry for mapping, collocation, and binning of data
    def mapXYZ(self, mesh_df, dataframes, dfnames):
        # Clean the target mesh data by removing infinite/NaN values and ensuring coordinates exist
        mesh_df = mesh_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z']) # remove inf and/or NaN values from mesh
        result_df = mesh_df.copy() # copy mesh geometry to save informed values on to
        distance_data = [] # empty list for distances between data

        # Using a single 3D search radius combining horizontal and vertical tolerances
        # Calculate the 3D Euclidean distance limit (hypotenuse of XY and Z limits)
        maxDist = np.sqrt((self.max_distance_xy**2) + self.max_distance_z**2) # define max search distance when mapping

        # Iterate through each source dataframe (e.g., different sensor datasets)
        for idx, df in enumerate(dataframes):
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z']) # remove inf and/or NaN values from dataframe
            if all(col in df.columns for col in ['x', 'y', 'Z']): # step to mapping with clean data
                
                tree = cKDTree(df[['x', 'y', 'Z']].values) # map using cKDTree from scipy.spatial 

                #Query by specific number of neighbors ('num')

                """With distance_upper_bound, cKDTree uses a sentinel for "no neighbor
                   distance = inf and index = len(df). Downstream logic must ignore those invalid neighbors"""
                
                if self.ckdMethod == 'num': # determine if number of neighbors is defined
                    if self.num_neighbors > 1: 
                        distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, distance_upper_bound=maxDist, workers=self.njobs)
                        
                    else:
                        # Find the single nearest Neighbor (k=1) 
                        distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, workers=self.njobs)

                    """Records mesh-to-source distances for QC. For k>1, stores multiple distances
                     per mesh point (one per neighbor); for k=1 stores a single distance per mesh point """
                    
                    for i, (dists, idxs) in enumerate(zip(distances, indices)):
                        if self.num_neighbors != 1:
                            remidxs = []
                            if len(df) not in idxs:
                                for j in range(len(idxs)):
                                    if abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[i, 'Z']) > self.max_distance_z:   # Remove points outside of max_distance_z
                                        remidxs.append(j)
                            idxs = [val for k, val in enumerate(idxs) if k not in remidxs]
                            dists = [val for l, val in enumerate(dists) if l not in remidxs]
                            for m in range(len(idxs)):
                                distance_data.append({'x': mesh_df.loc[idxs[m], 'x'],
                                                      'y': mesh_df.loc[idxs[m], 'y'],
                                                      'Z': mesh_df.loc[idxs[m], 'Z'],
                                                      'distance': dists[m]})
                        else:
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'Z': mesh_df.loc[i, 'Z'],
                                                  'distance': dists})

                # Query all points within a fixed distance ('dist')            
                elif self.ckdMethod == 'dist':
                   
                    # Returns a list of indices for ALL points within the radius (variable number of neighbors per point)
                    indices = tree.query_ball_point(mesh_df[['x', 'y', 'Z']].values, maxDist, workers=self.njobs)

                    # Calculate distances manually since query_ball_point only returns indices
                    for row_idx, idxs in enumerate(indices):
                        remidxs = []
                        if len(df) not in idxs:
                            for j in range(len(idxs)):
                                if abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[row_idx, 'Z']) > self.max_distance_z:  # Remove points outside of max_distance_z
                                    remidxs.append(j)
                        idxs = [val for i, val in enumerate(idxs) if i not in remidxs]
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'Z': mesh_df.loc[row_idx, 'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2) +
                                                                      ((mesh_df.loc[row_idx, 'Z'] - df.loc[index, 'Z'])**2)
                                                                      )})
                # Automatically determine a threshold based on nearest-neighbor distances and applies the 'num' ckdMethod.  
                elif self.ckdMethod == 'autoNum':

                    # First, query k-neighbors to find how far apart points typically are
                    distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, workers=self.njobs)
                    minDist = []
                    if self.num_neighbors > 1:
                        for dists in distances:
                            minDist.append(min(dists))
                    else:
                        minDist.append(distances)
                    distThresh = max(minDist) # Set threshold to cover all points

                    # Re-query using the calculated dynamic threshold
                    indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, distance_upper_bound=distThresh, workers=self.njobs)

                    # update indexes based on distance appendix
                    for row_idx, idxs in enumerate(indices):
                        remidxs = []
                        if len(df) not in idxs:
                            for j in range(len(idxs)):
                                if abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[row_idx, 'Z']) > self.max_distance_z:  # Remove points outside of max_distance_z
                                    remidxs.append(j)
                        idxs = [val for i, val in enumerate(idxs) if i not in remidxs]
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'Z': mesh_df.loc[row_idx, 'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2) +
                                                                      ((mesh_df.loc[row_idx, 'Z'] - df.loc[index, 'Z'])**2)
                                                                      )}) # define distance as the hypotenuse between points
                
                # Automatically determine a threshold based on nearest-neighbor distances and applies the 'dist' ckdMethod.  
                elif self.ckdMethod == 'autoDist':
                    
                    distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, workers=self.njobs)
                    minDist = []
                    if self.num_neighbors > 1:
                        for dists in distances:
                            minDist.append(min(dists))
                    else:
                        minDist.append(distances)
                    distThresh = max(minDist)
                    
                    # Re-query using the calculated dynamic threshold
                    indices = tree.query_ball_point(mesh_df[['x', 'y', 'Z']].values, distThresh, workers=self.njobs)
                    
                    # update indexes based on distance appendix
                    for row_idx, idxs in enumerate(indices):
                        remidxs = []
                        if len(df) not in idxs:
                            for j in range(len(idxs)):
                                if abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[row_idx, 'Z']) > self.max_distance_z:   # Remove points outside of max_distance_z
                                    remidxs.append(j)
                        idxs = [val for i, val in enumerate(idxs) if i not in remidxs]
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'Z': mesh_df.loc[row_idx, 'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2) +
                                                                      ((mesh_df.loc[row_idx, 'Z'] - df.loc[index, 'Z'])**2)
                                                                      )})
                            
                # Statistical aggregation
                # Calculate means (Arithmetic, Geometric, Harmonic) based on the neighbors found above
                # arithmetic mean works for any real values
                # geometric/harmonic means requires strictly positive values (undefined for <=0)
                # in-phase (“inph”) can be negative, so intentionally skips geom/harm for those columns
                
                if self.num_neighbors > 1: # apply different mean calcuations if number of neighbors is called
                
                    for col in df.columns:
                        if col not in ['x', 'y', 'Z']:
                            arithmetic_means, geometric_means, harmonic_means = [], [], [] # empty arrays for the arithmetic, geometric, and harmonic means
                                
                            for row in indices:
                                # Ensure we don't include out-of-bounds indices from cKDTree
                                if len(df) not in row:
                                    valid_values = df.loc[row, col].values # define values within bounds for means
                                else:
                                    valid_values = [] # if no values, not valid, empty array
        
                                if len(valid_values) > 0: # calculate means where possible based on the distance away and number of neighbors
                                    arithmetic_means.append(valid_values.mean()) #arithmetic mean for valid values
                                    # Geometric/Harmonic means require positive values
                                    geometric_means.append(gmean(valid_values) if np.all(valid_values > 0) else np.nan) # geometric mean for valid values
                                    harmonic_means.append(hmean(valid_values) if np.all(valid_values > 0) else np.nan) # harmonic mean for valid values
                                else:
                                    # If no neighbors found, append NaN
                                    arithmetic_means.append(np.nan) # if not valid, NaN arithmetic mean
                                    geometric_means.append(np.nan) # if not valid, NaN geometric mean
                                    harmonic_means.append(np.nan)  # if not valid, NaN harmonic mean

                            # Store results in the result dataframe
                            # Handle cases where dfname is empty 
                            if dfnames[idx] == '':
                                result_df[f'{col}_arith'] = arithmetic_means
                                if 'inph' in col:
                                    result_df[f'{col}_geom'] = arithmetic_means
                                    result_df[f'{col}_harm'] = arithmetic_means
                                else:
                                    result_df[f'{col}_geom'] = geometric_means
                                    result_df[f'{col}_harm'] = harmonic_means
                            else:
                                result_df[f'{dfnames[idx]}_{col}_arith'] = arithmetic_means # append arithmetic means to results df
                                # 'inph' (In-phase) data can be negative, so we skip Geom/Harm means for it
                                if 'inph' in col:
                                    result_df[f'{dfnames[idx]}_{col}_geom'] = arithmetic_means # append in-phase arithmetic means only
                                    result_df[f'{dfnames[idx]}_{col}_harm'] = arithmetic_means
                                else:
                                    result_df[f'{dfnames[idx]}_{col}_geom'] = geometric_means # append geometric means to results df
                                    result_df[f'{dfnames[idx]}_{col}_harm'] = harmonic_means # append harmonic means to results df
        
                else:
                    # Simple 1-to-1 nearest value mapping (no averaging needed) 
                    for col in df.columns:
                        if col not in ['x', 'y', 'Z']:
                            nearest_value = [] # when no number of neighbors defined, just map to closest point
                            
                            for i in indices:
                                nearest_value.append(df.loc[i, col]) # define nearest values when number of neighbors not defined
                                
                            # Append nearest values when num_neighbors = 1
                            if dfnames[idx] == '':
                                result_df[f'{col}'] = nearest_value
                            else:
                                result_df[f'{dfnames[idx]}_{col}'] = nearest_value

        return result_df, distance_data # call complete results df and distance data
        
    """ 2D collocation (surface-only): ignores vertical separation and uses only XY proximity """
    def mapXY(self, mesh_df, dataframes, dfnames):
        '''
        Maps dataframes onto a 2D mesh (X, Y).
        Similar to mapXYZ but ignores the vertical (Z) dimension.
        '''
        mesh_df = mesh_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
        result_df = mesh_df.copy()
        distance_data = []
        
        for idx, df in enumerate(dataframes):
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])
            if all(col in df.columns for col in ['x', 'y']):
                
                # KDTree built on X and Y only 
                tree = cKDTree(df[['x', 'y']].values)
                
                if self.ckdMethod == 'num':
                    if self.num_neighbors > 1:
                        distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, distance_upper_bound=self.max_distance_xy, workers=self.njobs)
                        
                    else:
                        distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, workers=self.njobs)

                    for i, (dists, idxs) in enumerate(zip(distances, indices)):
                        if self.num_neighbors != 1:
                            for dist in dists:
                                distance_data.append({'x': mesh_df.loc[[i], 'x'],
                                                      'y': mesh_df.loc[[i], 'y'],
                                                      'Z': mesh_df.loc[[i], 'Z'],
                                                      'distance': dist})
                        else:
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'Z': mesh_df.loc[i, 'Z'],
                                                  'distance': dists})

                            
                elif self.ckdMethod == 'dist':
                    
                    indices = tree.query_ball_point(mesh_df[['x', 'y']].values, self.max_distance_xy, workers=self.njobs)
                    
                    for row_idx, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2)
                                                                      )})
                    
                elif self.ckdMethod == 'autoNum':
                    
                    distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, workers=self.njobs)
                    minDist = []
                    if self.num_neighbors > 1:
                        for dists in distances:
                            minDist.append(min(dists))
                    else:
                        minDist.append(distances)
                    distThresh = max(minDist)
                    
                    indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, distance_upper_bound=distThresh, workers=self.njobs)
                    
                    for row_idx, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2)
                                                                      )})
                            
                elif self.ckdMethod == 'autoDist':
                    
                    distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, workers=self.njobs)
                    minDist = []
                    if self.num_neighbors > 1:
                        for dists in distances:
                            minDist.append(min(dists))
                    else:
                        minDist.append(distances)
                    distThresh = max(minDist)
                    
                    indices = tree.query_ball_point(mesh_df[['x', 'y']].values, distThresh, workers=self.njobs)
                    
                    for row_idx, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[row_idx, 'x'],
                                                  'y': mesh_df.loc[row_idx, 'y'],
                                                  'distance': np.sqrt(((mesh_df.loc[row_idx, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[row_idx, 'y'] - df.loc[index, 'y'])**2)
                                                                      )})
        
                # Statistical Aggregation (2D) 
                if (self.num_neighbors > 1) | (self.ckdMethod != 'num'):
                
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

                            # Handle cases where dfname is empty 
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
        '''
        Experimental high-speed 2D mapping method utilizing vectorization for distance tracking.
        '''
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

            # Vectorized query: Find neighbors for ALL mesh points at once
            distances, indices = tree.query(mesh_points, k=self.num_neighbors)
            
            # Ensure 2D shape for single neighbor case
            if self.num_neighbors == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]

            # create boolean mask for valid neighbors 
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
        '''
        Splits a multi-stat dataframe into three distinct dataframes for 
        Arithmetic, Geometric, and Harmonic averages respectively.
        '''
        arth_columns = [col for col in df.columns if '_arith' in col]
        geom_columns = [col for col in df.columns if '_geom' in col]
        harm_columns = [col for col in df.columns if '_harm' in col]
        # Identify columns that are NOT statistical 
        other_columns = [col for col in df.columns if col not in arth_columns + geom_columns + harm_columns]

        # Create new DataFrames and strip suffic from column names
        df_arth = df[other_columns + arth_columns].rename(columns={col: col.replace('_arith', '') for col in arth_columns})
        df_geom = df[other_columns + geom_columns].rename(columns={col: col.replace('_geom', '') for col in geom_columns})
        df_harm = df[other_columns + harm_columns].rename(columns={col: col.replace('_harm', '') for col in harm_columns})
        
        return df_arth, df_geom, df_harm

    def plot_distance_histogram(self, distance_data, xyAnn = (0.7,0.8)):
        '''
        Generates a histogram of the spatial displacement between the mesh and raw data.
        Provides Mean Squared Error (MSE) and standard deviation stats.
        '''
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

    def filter_and_rename(self, eca, minCut=5, maxCut=5, maxECa=50, minECa=0, keepInphase=False):
        '''
        Filters out outliers and crops the horizontal extent of the ECa data.
        Standardizes column names for specific sensor configurations (HCP/PRP).

        NB: Column headers require the following format, horizontal co-planar (HCP) or perpendicular (PRP), followed by the coil spacing (i.e., 1.0 or 1.1)
        e.g. ['HCP1.0', 'HCP1.0_inph', 'PRP1.1', 'PRP1.1_inph']
        '''
        # Compute xMin and xMax for the 'X' column specifically
        xMin = eca['X'].min() + minCut
        xMax = eca['X'].max() - maxCut
        
        ecainphcols = eca.filter(regex='HCP|PRP|VCP').columns.tolist()  # Find columns containing coil orientations (presumably should just by quadrature and in-phase measurements)
                        
        ecacols = [col for col in ecainphcols if '_inph' not in col]  # Isolate quadrature columns.
        
        ### Will mess around with this in a later version to more robsutly isolate coil orientation and spacing
        # if any('_' in col for col in ecacols):     
        #     filtcols = [s.replace('_', '', 1) for s in ecainphcols]
        #     eca.rename(columns=dict(zip(ecainphcols, filtcols)), inplace=True)
        #     ecainphcols = eca.filter(regex='HCP|PRP|VCP').columns.tolist()
        #     ecacols = [col for col in ecainphcols if '_inph' not in col]
            
        # Filter the DataFrame with corrected conditions
        startLen = len(eca)
        eca = eca.loc[(eca['X'] > xMin) & (eca['X'] < xMax)]
        xFiltLen = len(eca)
        for col in ecacols:
            eca = eca.loc[(eca[col] > minECa) & (eca[col] < maxECa)]
        ecaFiltLen = len(eca)
    
        # Print how many measurements are filtered using X and ECa filters.
        print('Started with ' + str(startLen) + ' matched elements and measurements. \n' +
              str(startLen - xFiltLen) + '(' + str((startLen - xFiltLen)/startLen) + ' %) matched points removed from spatial filtering. \n' +
              str(xFiltLen - ecaFiltLen) + '(' + str((xFiltLen - ecaFiltLen)/ecaFiltLen) + ' %) matched points removed from ECa filtering.')
    
        # Select subset of columns
        subs = ['X']
        if keepInphase:
            for col in ecainphcols:
                subs.append(col)
        else:
            for col in ecacols:
                subs.append(col)
        
        eca_subs = eca[subs]
        
        # Isolate necessary columns
        eca_df = eca_subs
        
        return eca_df
    
    def process_and_bin_data(self, eca_df, final_df, path, filename_ec, filename_eca,
                             nbins=300, xDis=0.5, zLayers=20, interpMethod='nearest',
                             useGeomspace=True):
        '''
        Creates a structured 2D grid from unorganized survey data. 
        - Crops EC data to ECa bounds.
        - Interpolates conductivity onto a mesh.
        - Bins the results horizontally (X) for final profile generation.
        '''
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

        # Perform 2D grid interpolation                         
        xi, zi = np.meshgrid(xi, zi)
        condi = griddata((x, z), cond, (xi, zi), method=interpMethod)
                                 
       #Flatten Results back into coordinate list
        x = np.unique(xi)
        z = np.unique(zi)
        condxz = np.array(np.meshgrid(x, z)).T.reshape(-1, 2)
        cond = condi.T.flatten()
        ec = np.concatenate((condxz, cond[:, None]), axis=1)
        ec = ec[~np.isnan(ec[:, 2]), :]
        ec = np.round(ec, 2)
        print('After meshing length ECa', len(eca))
        print('After meshing length EC', len(ec))

        # Binning the Inverted Conductivity (EC)                          
        midDepths = -np.unique(ec[:, 1])
        bins = np.round(np.linspace(minX, maxX, nbins + 1), 1)
        binID = np.digitize(ec[:, 0], bins)
    
        ertEC = np.empty((nbins, len(midDepths)))
        midDepthsr = midDepths[::-1]

        # iterate bins and depths to calculate average conductivity per cell                          
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

        # Drop Empty Bins                         
        rows_to_drop = df_eca.eq(0).any(axis=1)
        df_eca = df_eca[~rows_to_drop].reset_index(drop=True)
        df_ec = df_ec[~rows_to_drop].reset_index(drop=True)

        # Export Processed Data                         
        df_ec.to_csv(os.path.join(path, filename_ec), index=False)
        df_eca.to_csv(os.path.join(path, filename_eca), index=False)
    
        print('After binning length ECa', len(df_eca))
        print('After binning length EC', len(df_ec))
    
        return df_eca, df_ec
