# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import gmean, hmean
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

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

    def __init__(self, num_neighbors=5, max_distance_xy=5.0, max_distance_z=0.5, ckdMethod='num', njobs=1,
                 weighted=False, weight_power=2, weight_eps=1e-12): # standard kwargs for the DataMapper class
        valid_methods = {'num', 'dist', 'autoNum', 'autoDist'}
        if int(num_neighbors) < 1:
            raise ValueError("num_neighbors must be at least 1.")
        if float(max_distance_xy) <= 0:
            raise ValueError("max_distance_xy must be greater than 0.")
        if float(max_distance_z) < 0:
            raise ValueError("max_distance_z cannot be negative.")
        if ckdMethod not in valid_methods:
            raise ValueError(
                "ckdMethod must be one of " + ", ".join(sorted(valid_methods)) + "."
            )
        if float(weight_power) <= 0:
            raise ValueError("weight_power must be greater than 0.")
        if float(weight_eps) <= 0:
            raise ValueError("weight_eps must be greater than 0.")
        self.num_neighbors = num_neighbors # number of neighbors spatially closest to the data
        self.weighted = weighted          # if True, weight neighbor means by inverse distance (IDW); False = unweighted (original behavior)
        self.weight_power = weight_power  # IDW power p: weight = 1 / (distance**p + weight_eps); higher p => more local
        self.weight_eps = weight_eps      # floor so a coincident neighbor (distance 0) gets a finite, dominant weight
        self.max_distance_xy = max_distance_xy # maximum distance to be examined on the x-y plane
        self.max_distance_z = max_distance_z # maximum depth/elevation to be examined on the z axis
        self.ckdMethod = ckdMethod   # 'num' for original implementation, 'dist' to average all points within the max_distance, 'auto' to automatically determine the max_distance
        self.njobs = njobs   # Number of threads to use for parallel processing. Default is 1. Setting to -1 uses all available CPU threads.
    
    # defining mesh dataframe with x, y, Z geometry for mapping, collocation, and binning of data
    def _agg_means(self, values, coords=None, mesh_pt=None):
        """Return (arithmetic, geometric, harmonic) means over `values`.

        When self.weighted is True and `coords` (the neighbors' coordinates) and
        `mesh_pt` (the target mesh point's coordinates) are given, each neighbor
        is weighted by inverse distance, w_i = 1 / (d_i**weight_power +
        weight_eps), so closer source points contribute more (a coincident point
        dominates). With self.weighted False -- or coords/mesh_pt omitted -- the
        weights are uniform and the results equal the plain arithmetic,
        geometric, and harmonic means. Geometric/harmonic means require strictly
        positive values and return NaN otherwise (e.g. in-phase data).
        """
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.nan, np.nan, np.nan
        if self.weighted and coords is not None and mesh_pt is not None:
            d = np.sqrt(np.sum((np.asarray(coords, dtype=float) - np.asarray(mesh_pt, dtype=float)) ** 2, axis=1))
            w = 1.0 / (np.power(d, self.weight_power) + self.weight_eps)
        else:
            w = np.ones(values.size)
        arith = np.average(values, weights=w)
        if np.all(values > 0):
            geom = gmean(values, weights=w)
            harm = hmean(values, weights=w)
        else:
            geom = np.nan
            harm = np.nan
        return arith, geom, harm

    def mapXYZ(self, mesh_df, dataframes, dfnames, topo_correct='auto',
              topo_tol=None, surf_k=12, topo_verbose=True):
        # Clean the target mesh data by removing infinite/NaN values and ensuring coordinates exist
        mesh_df = mesh_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z']) # remove inf and/or NaN values from mesh
        mesh_df.reset_index(drop=True, inplace=True)
        result_df = mesh_df.copy() # copy mesh geometry to save informed values on to
        distance_data = [] # empty list for distances between data

        # Using a single 3D search radius combining horizontal and vertical tolerances
        # Calculate the 3D Euclidean distance limit (hypotenuse of XY and Z limits)
        maxDist = np.sqrt((self.max_distance_xy**2) + self.max_distance_z**2) # define max search distance when mapping

        # Iterate through each source dataframe (e.g., different sensor datasets)
        for idx, df in enumerate(dataframes):
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z']) # remove inf and/or NaN values from dataframe
            df.reset_index(drop=True, inplace=True)
            if all(col in df.columns for col in ['x', 'y', 'Z']): # step to mapping with clean data
                
                # --- optional topographic alignment: drape source onto mesh surface ---
                if topo_correct and len(df) and len(mesh_df):
                    off = self._topo_offset(mesh_df, df, surf_k=surf_k)
                    max_abs = float(np.nanmax(np.abs(off))) if len(off) else 0.0
                    med_abs = float(np.median(np.abs(off))) if len(off) else 0.0  # robust to surface-estimate outliers
                    self.topo_offset_summary = {
                        'median_offset_m': float(np.median(off)) if len(off) else 0.0,
                        'median_abs_offset_m': med_abs,
                        'std_offset_m': float(np.std(off)) if len(off) else 0.0,
                        'max_abs_offset_m': max_abs,
                        'datum_like': bool(len(off) and np.std(off) < 0.5),
                    }
                    tol = topo_tol if topo_tol is not None else max(0.25, 2.0 * self.max_distance_z)
                    # Trigger on a ROBUST measure (median |offset|), NOT the max: a few noisy
                    # surface-estimate outliers on a dense, already-aligned source (e.g. ERI -> ERI)
                    # must not fire a spurious drape. A real datum/topographic mismatch offsets
                    # the bulk of points, so its median |offset| exceeds tol.
                    apply_corr = (topo_correct is True) or (topo_correct == 'auto' and med_abs > tol)
                    if apply_corr:
                        df = df.copy()
                        df['Z'] = df['Z'].values + off          # shift source into the mesh's vertical frame
                        if topo_verbose:
                            kind = 'datum shift' if self.topo_offset_summary['datum_like'] else 'topographic drape'
                            print("[mapXYZ] {} applied to '{}': median |offset| {:.2f} m (signed {:.2f} m, max {:.2f} m)".format(
                                  kind, dfnames[idx] if dfnames[idx] else 'source',
                                  med_abs, self.topo_offset_summary['median_offset_m'], max_abs))
                    elif topo_verbose and topo_correct == 'auto':
                        print("[mapXYZ] no significant topographic mismatch (median |offset| "
                              "{:.2f} m <= {:.2f} m tol; max {:.2f} m)".format(med_abs, tol, max_abs))

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
                            for j in range(len(idxs)):
                                if dists[j] == np.inf:
                                    remidxs.append(j)
                                elif abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[i, 'Z']) > self.max_distance_z:   # Remove points outside of max_distance_z
                                    remidxs.append(j)
                            idxs = [val for k, val in enumerate(idxs) if k not in remidxs]
                            dists = [val for k, val in enumerate(dists) if k not in remidxs]
                            for m in range(len(idxs)):
                                if dists[m] == np.inf:
                                    continue
                                distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                      'y': mesh_df.loc[i, 'y'],
                                                      'Z': mesh_df.loc[i, 'Z'],
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
                    for i, idxs in enumerate(indices):
                        remidxs = []
                        for j in range(len(idxs)):
                            if idxs[j] == len(df):
                                remidxs.append(j)
                            elif abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[i, 'Z']) > self.max_distance_z:  # Remove points outside of max_distance_z
                                remidxs.append(j)
                        idxs = [val for i, val in enumerate(idxs) if i not in remidxs]
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'Z': mesh_df.loc[i, 'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[i, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[i, 'y'] - df.loc[index, 'y'])**2) +
                                                                      ((mesh_df.loc[i, 'Z'] - df.loc[index, 'Z'])**2)
                                                                      )})
                            
                # Automatically determine distance threshold such that each mesh element is matched with num_neighbors amount of points
                # and applies the 'num' ckdMethod.  
                elif self.ckdMethod == 'autoNum':

                    # First, query k-neighbors to find how far apart points typically are
                    distances, _ = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, workers=self.njobs)
                    distThresh = float(np.nanmax(distances))

                    # Re-query using the calculated dynamic threshold
                    distances, indices = tree.query(mesh_df[['x', 'y', 'Z']].values, k=self.num_neighbors, distance_upper_bound=distThresh, workers=self.njobs)

                    # update indexes based on distance appendix
                    for i, (dists, idxs) in enumerate(zip(distances, indices)):
                        remidxs = []
                        for j in range(len(idxs)):
                            if dists[j] == np.inf:
                                remidxs.append(j)
                            elif abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[i, 'Z']) > self.max_distance_z:  # Remove points outside of max_distance_z
                                remidxs.append(j)
                        idxs = [val for k, val in enumerate(idxs) if k not in remidxs]
                        dists = [val for k, val in enumerate(dists) if k not in remidxs]
                        for m in range(len(idxs)):
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'Z': mesh_df.loc[i, 'Z'],
                                                  'distance': dists[m]}) # define distance as the hypotenuse between points
                
                # Automatically determine distance threshold such that each mesh element is matched with at least 1 point and applies the 'dist' ckdMethod.  
                elif self.ckdMethod == 'autoDist':
                    
                    distances, _ = tree.query(mesh_df[['x', 'y', 'Z']].values, k=1, workers=self.njobs)
                    distThresh = float(np.nanmax(distances))
                    
                    # Re-query using the calculated dynamic threshold
                    indices = tree.query_ball_point(mesh_df[['x', 'y', 'Z']].values, distThresh, workers=self.njobs)
                    
                    # update indexes based on distance appendix
                    for i, idxs in enumerate(indices):
                        remidxs = []
                        if len(df) not in idxs:
                            for j in range(len(idxs)):
                                if abs(df.loc[idxs[j], 'Z'] - mesh_df.loc[i, 'Z']) > self.max_distance_z:   # Remove points outside of max_distance_z
                                    remidxs.append(j)
                        idxs = [val for i, val in enumerate(idxs) if i not in remidxs]
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'Z': mesh_df.loc[i, 'Z'],
                                                  'distance': np.sqrt(((mesh_df.loc[i, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[i, 'y'] - df.loc[index, 'y'])**2) +
                                                                      ((mesh_df.loc[i, 'Z'] - df.loc[index, 'Z'])**2)
                                                                      )})
                            
                # Statistical aggregation
                # Calculate means (Arithmetic, Geometric, Harmonic) based on the neighbors found above
                # arithmetic mean works for any real values
                # geometric/harmonic means requires strictly positive values (undefined for <=0)
                # in-phase (“inph”) can be negative, so intentionally skips geom/harm for those columns
                
                if self.num_neighbors > 1: # apply different mean calcuations if number of neighbors is called
                
                    mesh_pts = mesh_df[['x', 'y', 'Z']].values  # positional, aligned with the query order of `indices`
                    for col in df.columns:
                        if col not in ['x', 'y', 'Z']:
                            arithmetic_means, geometric_means, harmonic_means = [], [], [] # empty arrays for the arithmetic, geometric, and harmonic means
                                
                            for i, row in enumerate(indices):
                                # Ensure we don't include out-of-bounds indices from cKDTree
                                if len(df) not in row:
                                    valid_values = df.loc[row, col].values # define values within bounds for means
                                    # inverse-distance weighting (when self.weighted) uses neighbor coords vs this mesh point
                                    coords = df.loc[row, ['x', 'y', 'Z']].values if self.weighted else None
                                    a_mean, g_mean, h_mean = self._agg_means(valid_values, coords, mesh_pts[i] if self.weighted else None)
                                else:
                                    # If a neighbor slot is the cKDTree sentinel, treat the cell as unmatched
                                    a_mean, g_mean, h_mean = np.nan, np.nan, np.nan

                                arithmetic_means.append(a_mean) # arithmetic mean (distance-weighted when self.weighted)
                                geometric_means.append(g_mean)  # geometric mean; NaN where values are non-positive
                                harmonic_means.append(h_mean)   # harmonic mean; NaN where values are non-positive

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
        
    
    def _topo_offset(self, mesh_df, df, surf_k=12):
        """Per-source-point vertical offset that drapes the source's surface onto
        the mesh (e.g. ERI) surface at the same horizontal position. Returns an
        array aligned to df's rows; adding it to df['Z'] puts both datasets in a
        common vertical reference. mesh_df and df must already be free of NaN
        x/y/Z. The mesh surface near a point is the max Z of its `surf_k` nearest
        mesh nodes; the source surface is the top of that point's sounding
        (max Z among points sharing its x,y)."""
        mz = mesh_df['Z'].values
        mtree = cKDTree(mesh_df[['x', 'y']].values)
        km = int(min(surf_k, len(mz)))
        _, mi = mtree.query(df[['x', 'y']].values, k=km)
        mi = np.asarray(mi).reshape(len(df), -1)
        surf_mesh = mz[mi].max(axis=1)
        sz = df['Z'].values
        stree = cKDTree(df[['x', 'y']].values)
        ks = int(min(surf_k, len(sz)))
        _, si = stree.query(df[['x', 'y']].values, k=ks)
        si = np.asarray(si).reshape(len(df), -1)
        surf_src = sz[si].max(axis=1)
        return surf_mesh - surf_src

    def detect_topo_offset(self, mesh_df, source_df, surf_k=12):
        """Quantify any vertical/topographic mismatch between a source dataset
        (e.g. EMI models) and a mesh that follows topography (e.g. an ERI
        survey). For each source point, the mesh surface elevation at its
        horizontal position is compared with the top of that point's sounding.

        Returns (offsets, summary). `summary` holds median/mean/std/max |offset|
        in metres and `datum_like` (True when the offset is ~constant -- a datum
        shift -- rather than spatially varying topography). A max |offset| near
        zero means the two already share a vertical reference and mapXYZ needs no
        correction; a large value (depth-vs-elevation, or a datum offset) means
        the raw Z values don't overlap and matching would fail without it."""
        msh = mesh_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z'])
        src = source_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y', 'Z'])
        if len(msh) == 0 or len(src) == 0:
            return np.array([]), {'note': 'empty input', 'max_abs_offset_m': 0.0}
        off = self._topo_offset(msh, src, surf_k=surf_k)
        mx = float(np.nanmax(np.abs(off)))
        return off, {
            'median_offset_m': float(np.median(off)),
            'median_abs_offset_m': float(np.median(np.abs(off))),
            'mean_offset_m': float(np.mean(off)),
            'std_offset_m': float(np.std(off)),
            'max_abs_offset_m': mx,
            'datum_like': bool(np.std(off) < 0.5),
        }

    def _aggregate_xy_flat(self, values, source_index, target_index,
                           distances, n_targets, means):
        """Vectorized neighbor aggregation used by :meth:`mapXY`.

        ``source_index`` and ``target_index`` describe a flattened sparse
        target-to-source neighbor graph.  Aggregation with ``np.bincount``
        avoids one Python/SciPy function call per target cell.
        """
        values = np.asarray(values, dtype=float)
        source_index = np.asarray(source_index, dtype=int)
        target_index = np.asarray(target_index, dtype=int)
        distances = np.asarray(distances, dtype=float)
        out = {name: np.full(n_targets, np.nan, dtype=float) for name in means}
        if source_index.size == 0:
            return out

        neighbor_values = values[source_index]
        finite = np.isfinite(neighbor_values) & np.isfinite(distances)
        if not finite.any():
            return out

        target_valid = target_index[finite]
        values_valid = neighbor_values[finite]
        distances_valid = distances[finite]
        if self.weighted:
            weights = 1.0 / (
                np.power(distances_valid, self.weight_power) + self.weight_eps
            )
        else:
            weights = np.ones(values_valid.size, dtype=float)

        weight_sum = np.bincount(
            target_valid, weights=weights, minlength=n_targets
        )
        if "arith" in means:
            weighted_sum = np.bincount(
                target_valid, weights=weights * values_valid,
                minlength=n_targets,
            )
            np.divide(
                weighted_sum, weight_sum, out=out["arith"],
                where=weight_sum > 0,
            )

        if "geom" in means or "harm" in means:
            positive = values_valid > 0
            nonpositive_count = np.bincount(
                target_valid[~positive], minlength=n_targets
            )
            target_positive = target_valid[positive]
            values_positive = values_valid[positive]
            weights_positive = weights[positive]
            positive_weight_sum = np.bincount(
                target_positive, weights=weights_positive,
                minlength=n_targets,
            )
            valid_positive_target = (
                (positive_weight_sum > 0) & (nonpositive_count == 0)
            )

            if "geom" in means:
                log_sum = np.bincount(
                    target_positive,
                    weights=weights_positive * np.log(values_positive),
                    minlength=n_targets,
                )
                log_mean = np.full(n_targets, np.nan, dtype=float)
                np.divide(
                    log_sum, positive_weight_sum, out=log_mean,
                    where=valid_positive_target,
                )
                out["geom"][valid_positive_target] = np.exp(
                    log_mean[valid_positive_target]
                )

            if "harm" in means:
                inverse_sum = np.bincount(
                    target_positive,
                    weights=weights_positive / values_positive,
                    minlength=n_targets,
                )
                np.divide(
                    positive_weight_sum, inverse_sum, out=out["harm"],
                    where=valid_positive_target & (inverse_sum > 0),
                )
        return out

    """ 2D collocation (surface-only): ignores vertical separation and uses only XY proximity """
    def mapXY(self, mesh_df, dataframes, dfnames,
              means=("arith", "geom", "harm"), return_distances=True,
              deduplicate_xy=False):
        """Map one or more source dataframes to target rows using XY proximity.

        The neighbor search and statistical aggregation are vectorized.  The
        default output remains compatible with older calls.  For faster,
        lower-memory workflows, request only the needed means and optionally
        suppress the distance-record list.

        Parameters
        ----------
        mesh_df : pandas.DataFrame
            Target rows containing ``x`` and ``y``.
        dataframes : sequence of pandas.DataFrame
            Source tables containing ``x`` and ``y`` plus mapped variables.
        dfnames : sequence of str
            One optional output-column prefix per source table.
        means : str or sequence of {"arith", "geom", "harm"}
            Neighbor statistics to calculate. Defaults to all three.
        return_distances : bool, default True
            Return target-to-source distance records for diagnostics. False
            returns an empty list and avoids allocating those records.
        deduplicate_xy : bool, default False
            Query each unique target XY coordinate once and broadcast mapped
            values back to repeated target rows. This is exact for mapXY
            because Z is not part of the search.

        Returns
        -------
        result_df : pandas.DataFrame
            Original finite-coordinate target rows plus mapped columns.
        distance_data : list of dict
            Accepted links with target ``x``, ``y`` and ``distance``; empty
            when ``return_distances=False``. With ``deduplicate_xy=True``, each
            unique horizontal target contributes links only once.
        """
        if len(dataframes) != len(dfnames):
            raise ValueError("dataframes and dfnames must have the same length.")
        if isinstance(means, str):
            means = (means,)
        means = tuple(dict.fromkeys(means))
        valid_means = {"arith", "geom", "harm"}
        invalid_means = sorted(set(means).difference(valid_means))
        if not means or invalid_means:
            raise ValueError(
                "means must contain one or more of 'arith', 'geom', and 'harm'."
            )

        mesh_df = (mesh_df.replace([np.inf, -np.inf], np.nan)
                          .dropna(subset=["x", "y"]).reset_index(drop=True))
        if mesh_df.empty:
            raise ValueError("mesh_df contains no finite x/y coordinates.")
        result_df = mesh_df.copy()
        mesh_xy = mesh_df[["x", "y"]].to_numpy(dtype=float)
        if deduplicate_xy:
            query_xy, inverse = np.unique(
                mesh_xy, axis=0, return_inverse=True
            )
        else:
            query_xy = mesh_xy
            inverse = np.arange(len(mesh_df), dtype=int)
        n_query = len(query_xy)
        distance_data = []

        for frame_number, frame in enumerate(dataframes):
            df = (frame.replace([np.inf, -np.inf], np.nan)
                       .dropna(subset=["x", "y"]).reset_index(drop=True))
            if df.empty:
                raise ValueError(
                    f"Source dataframe {frame_number} contains no finite x/y coordinates."
                )
            source_xy = df[["x", "y"]].to_numpy(dtype=float)
            tree = cKDTree(source_xy)
            n_source = len(df)
            k = max(int(self.num_neighbors), 1)

            if self.ckdMethod == "num":
                distances, indices = tree.query(
                    query_xy, k=k,
                    distance_upper_bound=self.max_distance_xy,
                    workers=self.njobs,
                )
                if k == 1:
                    distances = distances[:, None]
                    indices = indices[:, None]
                rectangular = True
            elif self.ckdMethod == "autoNum":
                # A global threshold large enough to retain the requested
                # nearest neighbors is equivalent to this single unconstrained
                # query, avoiding the legacy duplicate KD-tree query.
                distances, indices = tree.query(
                    query_xy, k=k, workers=self.njobs
                )
                if k == 1:
                    distances = distances[:, None]
                    indices = indices[:, None]
                rectangular = True
            else:
                radius = self.max_distance_xy
                if self.ckdMethod == "autoDist":
                    nearest, _ = tree.query(query_xy, k=1, workers=self.njobs)
                    radius = float(np.nanmax(nearest))
                neighbor_rows = tree.query_ball_point(
                    query_xy, radius, workers=self.njobs
                )
                counts = np.fromiter(
                    (len(row) for row in neighbor_rows), dtype=int,
                    count=n_query,
                )
                if counts.sum():
                    source_index = np.concatenate([
                        np.asarray(row, dtype=int)
                        for row in neighbor_rows if len(row)
                    ])
                    target_index = np.repeat(np.arange(n_query), counts)
                    delta = source_xy[source_index] - query_xy[target_index]
                    distance_flat = np.sqrt(np.sum(delta * delta, axis=1))
                else:
                    source_index = np.empty(0, dtype=int)
                    target_index = np.empty(0, dtype=int)
                    distance_flat = np.empty(0, dtype=float)
                rectangular = False

            if rectangular:
                valid = (indices != n_source) & np.isfinite(distances)
                target_matrix = np.broadcast_to(
                    np.arange(n_query)[:, None], indices.shape
                )
                source_index = indices[valid].astype(int, copy=False)
                target_index = target_matrix[valid].astype(int, copy=False)
                distance_flat = distances[valid].astype(float, copy=False)

            if return_distances and source_index.size:
                target_coords = query_xy[target_index]
                records = pd.DataFrame({
                    "x": target_coords[:, 0],
                    "y": target_coords[:, 1],
                    "distance": distance_flat,
                })
                distance_data.extend(records.to_dict(orient="records"))

            prefix = dfnames[frame_number]
            single_nearest = self.ckdMethod == "num" and k == 1
            for col in df.columns:
                if col in ("x", "y"):
                    continue
                numeric = pd.to_numeric(df[col], errors="coerce").to_numpy(
                    dtype=float
                )
                requested = set(means)
                if single_nearest:
                    requested.add("arith")
                if "inph" in col:
                    # Geometric/harmonic means are undefined for signed
                    # in-phase values. Preserve the historical behavior by
                    # placing the arithmetic value under requested suffixes.
                    requested.add("arith")
                aggregated = self._aggregate_xy_flat(
                    numeric, source_index, target_index, distance_flat,
                    n_query, requested,
                )
                base = f"{prefix}_{col}" if prefix else col
                if single_nearest:
                    result_df[base] = aggregated["arith"][inverse]
                    continue
                for mean_name in means:
                    values = (aggregated["arith"]
                              if "inph" in col else aggregated[mean_name])
                    result_df[f"{base}_{mean_name}"] = values[inverse]

        return result_df, distance_data

    def _mapXY_legacy(self, mesh_df, dataframes, dfnames):
        '''
        Maps dataframes onto a 2D mesh (X, Y).
        Similar to mapXYZ but ignores the vertical (Z) dimension.
        '''
        if len(dataframes) != len(dfnames):
            raise ValueError("dataframes and dfnames must have the same length.")

        mesh_df = (mesh_df.replace([np.inf, -np.inf], np.nan)
                          .dropna(subset=['x', 'y']).reset_index(drop=True))
        if mesh_df.empty:
            raise ValueError("mesh_df contains no finite x/y coordinates.")
        result_df = mesh_df.copy()
        distance_data = []
        
        for idx, df in enumerate(dataframes):
            df = (df.replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=['x', 'y']).reset_index(drop=True))
            if df.empty:
                raise ValueError(f"Source dataframe {idx} contains no finite x/y coordinates.")
            if all(col in df.columns for col in ['x', 'y']):
                
                # KDTree built on X and Y only 
                tree = cKDTree(df[['x', 'y']].values)
                
                if self.ckdMethod == 'num':
                    if self.num_neighbors > 1:
                        distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, distance_upper_bound=self.max_distance_xy, workers=self.njobs)
                        
                    else:
                        distances, indices = tree.query(
                            mesh_df[['x', 'y']].values,
                            k=1,
                            distance_upper_bound=self.max_distance_xy,
                            workers=self.njobs,
                        )

                    if self.num_neighbors == 1:
                        distances = distances[:, None]
                        indices = indices[:, None]

                    for i, (dists, idxs) in enumerate(zip(distances, indices)):
                        valid = (idxs != len(df)) & np.isfinite(dists)
                        for distance in np.asarray(dists)[valid]:
                            # x/y identify the TARGET mesh row. The previous
                            # implementation incorrectly indexed mesh_df with a
                            # SOURCE-row index, producing wrong coordinates and
                            # sometimes a KeyError.
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'distance': float(distance)})

                # Query all points within a fixed distance ('dist')            
                elif self.ckdMethod == 'dist':
                   
                    # Returns a list of indices for ALL points within the radius (variable number of neighbors per point)
                    indices = tree.query_ball_point(mesh_df[['x', 'y']].values, self.max_distance_xy, workers=self.njobs)

                    # Calculate distances manually since query_ball_point only returns indices
                    for i, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'distance': np.sqrt(((mesh_df.loc[i, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[i, 'y'] - df.loc[index, 'y'])**2)
                                                                      )})
                            
                # Automatically determine distance threshold such that each mesh element is matched with num_neighbors amount of points
                # and applies the 'num' ckdMethod.  
                elif self.ckdMethod == 'autoNum':

                    # First, query k-neighbors to find how far apart points typically are
                    distances, _ = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, workers=self.njobs)
                    distThresh = float(np.nanmax(distances))

                    # Re-query using the calculated dynamic threshold
                    distances, indices = tree.query(mesh_df[['x', 'y']].values, k=self.num_neighbors, distance_upper_bound=distThresh, workers=self.njobs)

                    if self.num_neighbors == 1:
                        distances = distances[:, None]
                        indices = indices[:, None]

                    # update indexes based on distance appendix
                    for i, (dists, idxs) in enumerate(zip(distances, indices)):
                        remidxs = []
                        for j in range(len(idxs)):
                            if dists[j] == np.inf:
                                remidxs.append(j)
                        idxs = [val for k, val in enumerate(idxs) if k not in remidxs]
                        dists = [val for k, val in enumerate(dists) if k not in remidxs]
                        for m in range(len(idxs)):
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'distance': dists[m]}) # define distance as the hypotenuse between points
                
                # Automatically determine distance threshold such that each mesh element is matched with at least 1 point and applies the 'dist' ckdMethod.  
                elif self.ckdMethod == 'autoDist':
                    
                    distances, _ = tree.query(mesh_df[['x', 'y']].values, k=1, workers=self.njobs)
                    distThresh = float(np.nanmax(distances))
                    
                    # Re-query using the calculated dynamic threshold
                    indices = tree.query_ball_point(mesh_df[['x', 'y']].values, distThresh, workers=self.njobs)
                    
                    # update indexes based on distance appendix
                    for i, idxs in enumerate(indices):
                        for index in idxs:
                            distance_data.append({'x': mesh_df.loc[i, 'x'],
                                                  'y': mesh_df.loc[i, 'y'],
                                                  'distance': np.sqrt(((mesh_df.loc[i, 'x'] - df.loc[index, 'x'])**2) + 
                                                                      ((mesh_df.loc[i, 'y'] - df.loc[index, 'y'])**2)
                                                                      )})
        
                # Statistical Aggregation (2D) 
                if (self.num_neighbors > 1) or (self.ckdMethod != 'num'):
                
                    mesh_pts = mesh_df[['x', 'y']].values  # positional, aligned with the query order of `indices`
                    for col in df.columns:
                        if col not in ['x', 'y']:
                            arithmetic_means, geometric_means, harmonic_means = [], [], []
                                
                            numeric = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
                            for i, row in enumerate(indices):
                                row = np.asarray(row, dtype=int)
                                row = row[row != len(df)]
                                if row.size:
                                    finite = np.isfinite(numeric[row])
                                    row = row[finite]
                                valid_values = numeric[row]
                                coords = df.loc[row, ['x', 'y']].values if (self.weighted and row.size) else None
                                a_mean, g_mean, h_mean = self._agg_means(
                                    valid_values,
                                    coords,
                                    mesh_pts[i] if self.weighted else None,
                                )

                                arithmetic_means.append(a_mean)
                                geometric_means.append(g_mean)
                                harmonic_means.append(h_mean)  

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
                            
                            for i in indices[:, 0]:
                                nearest_value.append(df.loc[i, col] if i != len(df) else np.nan)
                                
                            if dfnames[idx] == '':
                                result_df[f'{col}'] = nearest_value
                            else:
                                result_df[f'{dfnames[idx]}_{col}'] = nearest_value
                                
        return result_df, distance_data

    def create_mean_dfs(self, df):
        '''
        Splits a multi-stat dataframe into three distinct dataframes for
        Arithmetic, Geometric, and Harmonic averages respectively.

        Statistical suffixes are matched only at the end of a column name.
        This avoids accidentally classifying a non-statistical column that
        merely contains text such as "_geom" in the middle of its name.
        '''
        arth_columns = [col for col in df.columns if str(col).endswith('_arith')]
        geom_columns = [col for col in df.columns if str(col).endswith('_geom')]
        harm_columns = [col for col in df.columns if str(col).endswith('_harm')]

        statistical_columns = set(arth_columns + geom_columns + harm_columns)
        other_columns = [col for col in df.columns if col not in statistical_columns]

        def _build(stat_columns, suffix):
            rename = {
                col: str(col)[:-len(suffix)]
                for col in stat_columns
            }
            return df[other_columns + stat_columns].rename(columns=rename).copy()

        df_arth = _build(arth_columns, '_arith')
        df_geom = _build(geom_columns, '_geom')
        df_harm = _build(harm_columns, '_harm')

        return df_arth, df_geom, df_harm

    @staticmethod
    def _ax_scale(ax, ref=3.8):
        """Scale factor (~1 for a ref-inch panel) so titles, labels, markers and
        colorbars shrink to fit whatever axis they are drawn into -- e.g. a
        crowded 2x2. Falls back to 1.0 if the axis size can't be determined."""
        try:
            fig = ax.figure
            pos = ax.get_position()
            w_in = pos.width * fig.get_size_inches()[0]
            h_in = pos.height * fig.get_size_inches()[1]
            s = min(w_in, h_in) / ref
        except Exception:
            s = 1.0
        return float(np.clip(s, 0.4, 1.5))

    @staticmethod
    def _units_label(base, units, log=False, dimensionless=False):
        """Append physical units to a colorbar label. log=True wraps them as
        log10(units); dimensionless=True (e.g. a ratio) appends nothing."""
        if not units or dimensionless:
            return base
        if log:
            return base + r" ($\log_{10}$(" + str(units) + "))"
        return base + " (" + str(units) + ")"

    def plot_distance_histogram(self, distance_data, xyAnn=(0.97, 0.97),
                                savefig=False, ax=None, figdir=None):
        '''
        Histogram of the mesh-to-source spatial displacement, annotated with the
        RMS displacement, mean and standard deviation. Pass `ax` to draw into an
        existing panel; fonts/markers scale to the panel size so it stays
        legible in a crowded subplot grid.
        '''
        distM = pd.DataFrame(distance_data)
        if distM.empty or 'distance' not in distM:
            raise ValueError(
                "distance_data is empty; no source points were found inside the search distance."
            )
        rms = np.sqrt(np.mean(distM['distance'] ** 2))   # RMS displacement (not MSE)
        mean_distance = np.mean(distM['distance'])
        std_distance = np.std(distM['distance'])

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6, 4))
        s = self._ax_scale(ax)

        ax.hist(distM['distance'], bins=50, edgecolor='black',
                linewidth=max(0.2, 0.6 * s))
        ax.set_xlabel('Displacement (m)', fontsize=14 * s, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14 * s, fontweight='bold')
        ax.set_title('Spatial Displacement Histogram', fontsize=16 * s, fontweight='bold')
        ax.tick_params(labelsize=12 * s)

        ha = 'right' if xyAnn[0] > 0.5 else 'left'
        va = 'top' if xyAnn[1] > 0.5 else 'bottom'
        ax.annotate(
            f'N = {len(distM)}\nRMS: {rms:.2f}\nMean: {mean_distance:.2f}\nStd: {std_distance:.2f}',
            xy=xyAnn, xycoords='axes fraction', ha=ha, va=va, fontsize=12 * s,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      linewidth=0.5 * s))

        if standalone:
            fig.tight_layout()
        if savefig and figdir:
            ax.figure.savefig(figdir + '_histogram.png', dpi=300)
        return ax

    def filter_and_rename(self, eca, minCut=5, maxCut=5, maxECa=50, minECa=0, keepInphase=False):
        '''
        Filters out outliers and crops the horizontal extent of the ECa data.
        Standardizes the calibration input to X plus HCP/PRP/VCP coil columns.

        Coil matching is case-insensitive. In-phase fields are identified by
        the ``_inph`` token and are excluded unless ``keepInphase=True``.
        '''
        if 'X' not in eca.columns:
            raise KeyError(
                "eca must contain the along-profile coordinate column 'X'. "
                f"Available columns: {eca.columns.tolist()}"
            )
        if minCut < 0 or maxCut < 0:
            raise ValueError("minCut and maxCut cannot be negative.")
        if minECa >= maxECa:
            raise ValueError("minECa must be smaller than maxECa.")

        eca = eca.copy()
        eca['X'] = pd.to_numeric(eca['X'], errors='coerce')
        eca = eca.replace([np.inf, -np.inf], np.nan).dropna(subset=['X'])
        if eca.empty:
            raise ValueError("eca contains no finite X coordinates.")

        def _is_coil(col):
            upper = str(col).upper()
            return any(tag in upper for tag in ('HCP', 'PRP', 'VCP'))

        ecainphcols = [col for col in eca.columns if _is_coil(col)]
        ecacols = [
            col for col in ecainphcols
            if '_INPH' not in str(col).upper()
        ]

        if not ecacols:
            raise ValueError(
                "No ECa coil columns were found. Expected names containing "
                "HCP, PRP, or VCP.\n"
                f"Available columns: {eca.columns.tolist()}"
            )

        # Convert selected measurements before spatial/value filtering. This
        # prevents string/object columns from failing numeric comparisons.
        for col in ecainphcols:
            eca[col] = pd.to_numeric(eca[col], errors='coerce')

        finite_x = eca['X'].to_numpy(dtype=float)
        xMin = float(np.nanmin(finite_x)) + float(minCut)
        xMax = float(np.nanmax(finite_x)) - float(maxCut)
        if not np.isfinite(xMin) or not np.isfinite(xMax) or xMin >= xMax:
            raise ValueError(
                "The X cuts remove the full profile; reduce minCut/maxCut or check X."
            )

        startLen = len(eca)
        eca = eca.loc[(eca['X'] >= xMin) & (eca['X'] <= xMax)].copy()
        xFiltLen = len(eca)

        # Retain rows for which every calibration ECa coil is finite and lies
        # inside the requested strict range.
        for col in ecacols:
            eca = eca.loc[
                np.isfinite(eca[col]) &
                (eca[col] > minECa) &
                (eca[col] < maxECa)
            ].copy()
        ecaFiltLen = len(eca)

        spatial_pct = 100.0 * (startLen - xFiltLen) / startLen if startLen else np.nan
        eca_pct = 100.0 * (xFiltLen - ecaFiltLen) / xFiltLen if xFiltLen else np.nan
        print(
            f"Started with {startLen} matched elements and measurements.\n"
            f"{startLen - xFiltLen} ({spatial_pct:.2f}%) matched points removed "
            "by spatial filtering.\n"
            f"{xFiltLen - ecaFiltLen} ({eca_pct:.2f}%) matched points removed "
            "by ECa filtering."
        )

        if eca.empty:
            raise ValueError(
                "No matched rows remain after the X and ECa filters. "
                "Inspect the ECa ranges and the X cut settings."
            )

        selected = ecainphcols if keepInphase else ecacols
        return eca[['X'] + selected].copy()

    def process_and_bin_data(self, eca_df, final_df, path, filename_ec, filename_eca,
                             nbins=300, xDis=0.5, zLayers=20, interpMethod='nearest',
                             useGeomspace=True, maxDpth=None, min_thick=None, d0=0.0,
                             smoothEC=False, smoothMethod='mean',
                             smoothWindow=5, savgolPolyorder=2):
        '''
        Creates a structured 2D grid from unorganized survey data. 
        - Crops EC data to ECa bounds.
        - Interpolates conductivity onto a mesh.
        - Bins the results horizontally (X) for final profile generation.
        - Optionally smooths each final ERT conductivity profile across depth
          columns (within rows) before export.

        Parameters
        ----------
        maxDpth : float or None, default None
            Maximum depth override. When set, z_min is overridden to -maxDpth.
        min_thick : float or None, default None
            Thinnest (top) layer thickness when generating geometrically growing
            depth layers.
        d0 : float, default 0.0
            Top elevation/depth baseline for geometric layer generation.
        smoothEC : bool, default False
            If True, smooth df_ec across depth columns for each row. The ECa
            dataframe is not smoothed.
        smoothMethod : {'mean', 'median', 'savgol'}, default 'mean'
            'mean'   : centered rolling arithmetic mean across depth layers.
            'median' : centered rolling median across depth layers.
            'savgol' : Savitzky-Golay polynomial smoothing across depth layers.
                       Standard Savitzky-Golay assumes equally spaced samples,
                       so this method requires useGeomspace=False.
        smoothWindow : int, default 5
            Odd number of adjacent depth columns in the centered smoothing
            window. Must be >= 3 when smoothEC=True.
        savgolPolyorder : int, default 2
            Polynomial order for Savitzky-Golay smoothing. Must be smaller
            than smoothWindow.
        '''
        if int(nbins) < 2:
            raise ValueError("nbins must be at least 2.")
        if float(xDis) <= 0:
            raise ValueError("xDis must be greater than 0.")
        if int(zLayers) < 2:
            raise ValueError("zLayers must be at least 2.")
        if interpMethod not in {'nearest', 'linear', 'cubic'}:
            raise ValueError("interpMethod must be 'nearest', 'linear', or 'cubic'.")

        valid_smooth_methods = {'mean', 'median', 'savgol'}
        if smoothEC:
            if smoothMethod not in valid_smooth_methods:
                raise ValueError(
                    "smoothMethod must be 'mean', 'median', or 'savgol'."
                )
            if int(smoothWindow) != smoothWindow or int(smoothWindow) < 3:
                raise ValueError("smoothWindow must be an integer >= 3.")
            smoothWindow = int(smoothWindow)
            if smoothWindow % 2 == 0:
                raise ValueError(
                    "smoothWindow must be odd (for example 3, 5, or 7) so the "
                    "window is centered on a depth layer."
                )
            if smoothMethod == 'savgol':
                if useGeomspace:
                    raise ValueError(
                        "smoothMethod='savgol' requires useGeomspace=False because "
                        "standard Savitzky-Golay filtering assumes equally spaced "
                        "depth samples. Use 'mean'/'median' with geometric depth "
                        "spacing, or set useGeomspace=False."
                    )
                if int(savgolPolyorder) != savgolPolyorder or int(savgolPolyorder) < 0:
                    raise ValueError("savgolPolyorder must be a non-negative integer.")
                savgolPolyorder = int(savgolPolyorder)
                if savgolPolyorder >= smoothWindow:
                    raise ValueError(
                        "savgolPolyorder must be smaller than smoothWindow."
                    )

        required_final = ['X', 'Z', 'Conductivity(mS/m)']
        missing_final = [c for c in required_final if c not in final_df.columns]
        if missing_final:
            raise KeyError(f"final_df is missing required columns: {missing_final}")
        if 'X' not in eca_df.columns or eca_df.shape[1] < 2:
            raise KeyError("eca_df must contain 'X' plus at least one ECa coil column.")

        # Extract finite numeric values from eca_df and final_df.
        eca_clean = eca_df.apply(pd.to_numeric, errors='coerce')
        final_clean = final_df[required_final].apply(pd.to_numeric, errors='coerce')
        eca_clean = eca_clean.replace([np.inf, -np.inf], np.nan).dropna()
        final_clean = final_clean.replace([np.inf, -np.inf], np.nan).dropna()
        if eca_clean.empty or final_clean.empty:
            raise ValueError("No finite ECa or ERT conductivity rows are available for binning.")

        eca = eca_clean.to_numpy(dtype=float)
        ec = final_clean.to_numpy(dtype=float)
        print('Initial length ECa', len(eca))
        print('Initial length EC', len(ec))
        print(np.min(ec[:, 2]))
    
        minX = np.round(np.min(eca[:, 0]), 2)
        maxX = np.round(np.max(eca[:, 0]), 2)
    
        # Crop the ERT model to the horizontal support of the calibration ECa.
        ec = ec[(ec[:, 0] >= minX) & (ec[:, 0] <= maxX), :]
        if ec.size == 0:
            raise ValueError(
                "No ERT conductivity cells overlap the filtered ECa X range "
                f"({minX:.3f} to {maxX:.3f} m)."
            )
        print('After crop length ECa', len(eca))
        print('After crop length EC', len(ec))
    
        # Prepare the x, z, and conductivity data
        x = ec[:, 0]
        z = ec[:, 1]
        cond = ec[:, 2]
        xi = np.arange(np.min(x), np.max(x), xDis)
        if xi.size == 0 or not np.isclose(xi[-1], np.max(x)):
            xi = np.append(xi, np.max(x))
    
        # Choose spacing method for z
        z_min, z_max = np.min(z), np.max(z)
        if maxDpth is not None:
            z_min = -abs(float(maxDpth))

        if useGeomspace:
            if min_thick is not None:
                def _geom_growth_depths(max_depth, n_layers, first_thk=0.25, top=0.0, include_surface=False):
                    """n_layers layers from `top` to exactly max_depth, thicknesses growing
                    geometrically. Guarantees: thinnest (top) layer == first_thk, monotonic
                    increase, deepest interface == max_depth. Returns (interfaces, thicknesses)."""
                    span = max_depth - top
                    n = int(n_layers)
                    if first_thk * n >= span:                       # too many thick layers -> uniform
                        thk = np.full(n, span / n)
                    else:
                        lo, hi = 1.0 + 1e-9, 10.0                    # bisection for growth ratio r
                        f = lambda r: first_thk * (r**n - 1) / (r - 1) - span
                        for _ in range(200):
                            mid = 0.5 * (lo + hi)
                            hi, lo = (mid, lo) if f(mid) > 0 else (hi, mid)
                        thk = first_thk * (0.5 * (lo + hi)) ** np.arange(n)
                    bottoms = top + np.cumsum(thk)
                    interfaces = np.concatenate([[top], bottoms]) if include_surface else bottoms
                    return interfaces, thk

                zi, thckns = _geom_growth_depths(-z_min, zLayers, first_thk=min_thick, top=d0, include_surface=False)
                zi *= -1
                zi = zi[::-1]
            elif z_min >= 0 or z_max >= 0:
                raise ValueError("For geomspace, Z values must be negative (depths).")
            else:
                zi = -np.geomspace(abs(z_min), abs(z_max), zLayers)
        else:
            zi = np.linspace(z_min, z_max, zLayers)

        # Perform 2D grid interpolation                         
        xi, zi = np.meshgrid(xi, zi)
        condi = griddata((x, z), cond, (xi, zi), method=interpMethod)

        # Linear/cubic griddata returns NaN outside the convex hull.  A single
        # missing layer makes EMagPy's forward response NaN for that calibration
        # row, so fill only those gaps with nearest-neighbour estimates.
        if interpMethod != 'nearest' and np.isnan(condi).any():
            gap = np.isnan(condi)
            nearest = griddata((x, z), cond, (xi, zi), method='nearest')
            condi = np.where(np.isfinite(condi), condi, nearest)

            # Report extrapolation distance for auditability
            dgap, _ = cKDTree(np.c_[x, z]).query(np.c_[xi[gap], zi[gap]])
            print(f'   filled {gap.sum()} of {gap.size} grid cells '
                  f'({100 * gap.mean():.1f}%) outside the ERT convex hull by nearest '
                  f'neighbour; max extrapolation {dgap.max():.2f} m')
            if dgap.max() > self.max_distance_xy:      
                print(f'   ! WARNING: fill reaches {dgap.max():.1f} m - exceeds the '
                      'EMI footprint; consider trimming instead')
                                 
        # Flatten Results back into coordinate list
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
        bins = np.linspace(minX, maxX, nbins + 1)
        binID = np.clip(np.digitize(ec[:, 0], bins) - 1, 0, nbins - 1)
    
        ertEC = np.empty((nbins, len(midDepths)))
        midDepthsr = midDepths[::-1]

        # iterate bins and depths to calculate average conductivity per cell                          
        for i in range(nbins):
            for j in range(len(midDepths)):
                idepth = np.isclose(ec[:, 1], -midDepthsr[j], atol=1e-3)
                ibins = binID == i
                valid_points = ec[idepth & ibins, 2]
                ertEC[i, j] = np.mean(valid_points) if valid_points.size > 0 else np.nan
    
        df_ec = pd.DataFrame(np.round(ertEC, 2), columns=[f'd{col:.2f}' for col in midDepthsr])
    
        # Binning for eca
        binID = np.clip(np.digitize(eca[:, 0], bins) - 1, 0, nbins - 1)
        eca2 = np.full((nbins, eca.shape[1] - 1), np.nan)
    
        for i in range(nbins):
            ie = binID == i
            if np.sum(ie) > 0:
                eca2[i, :] = np.nanmean(eca[ie, 1:], axis=0)
    
        headers = eca_df.columns[1:].tolist()
        df_eca = pd.DataFrame(np.round(eca2, 2), columns=headers)

        # EMagPy requires row-for-row calibration pairs and a complete EC
        # profile.  Remove any bin that is incomplete in either file.
        valid_rows = (np.isfinite(df_eca.to_numpy(dtype=float)).all(axis=1) &
                      np.isfinite(df_ec.to_numpy(dtype=float)).all(axis=1))
        n_dropped = int((~valid_rows).sum())
        df_eca = df_eca.loc[valid_rows].reset_index(drop=True)
        df_ec = df_ec.loc[valid_rows].reset_index(drop=True)
        if len(df_eca) < 2:
            raise ValueError(
                "Fewer than two complete co-located calibration rows remain. "
                "Reduce nbins, use a larger xDis, or inspect spatial overlap."
            )
        if (df_ec.to_numpy(dtype=float) <= 0).any():
            raise ValueError(
                "Interpolated ERT conductivity contains zero or negative values. "
                "This can occur from higher-order interpolation overshoot; use "
                "interpMethod='linear' or 'nearest', or inspect the ERT model."
            )

        # Optional smoothing across DEPTH columns within each row.
        # This changes the vertical conductivity profile supplied to the EMI
        # forward model, while leaving the measured/binned ECa data untouched.
        if smoothEC:
            if smoothWindow > df_ec.shape[1]:
                raise ValueError(
                    f"smoothWindow={smoothWindow} exceeds the number of depth "
                    f"columns ({df_ec.shape[1]}). Reduce smoothWindow or increase zLayers."
                )

            if smoothMethod == 'mean':
                # Transpose so pandas rolls down each original row's depth profile.
                # min_periods=1 retains the shallowest/deepest layers by using the
                # available truncated centered window at the profile edges.
                df_ec = (df_ec.T
                               .rolling(window=smoothWindow, center=True, min_periods=1)
                               .mean()
                               .T)

            elif smoothMethod == 'median':
                df_ec = (df_ec.T
                               .rolling(window=smoothWindow, center=True, min_periods=1)
                               .median()
                               .T)

            elif smoothMethod == 'savgol':
                smoothed = savgol_filter(
                    df_ec.to_numpy(dtype=float),
                    window_length=smoothWindow,
                    polyorder=savgolPolyorder,
                    axis=1,
                    mode='interp'
                )
                df_ec = pd.DataFrame(smoothed, columns=df_ec.columns, index=df_ec.index)

            # Keep the historical EC export precision.
            df_ec = df_ec.round(3)

            # Mean/median preserve positivity when inputs are positive, but
            # polynomial smoothing can overshoot. Do not pass a non-physical
            # conductivity profile silently to the forward model.
            if (~np.isfinite(df_ec.to_numpy(dtype=float))).any():
                raise ValueError(
                    "Depth smoothing produced non-finite ERT conductivity values."
                )
            if (df_ec.to_numpy(dtype=float) <= 0).any():
                raise ValueError(
                    "Depth smoothing produced zero or negative ERT conductivity. "
                    "Reduce smoothWindow/savgolPolyorder or use mean/median smoothing."
                )

            print(
                f"Smoothed ERT conductivity across depth columns: "
                f"method={smoothMethod}, window={smoothWindow}" +
                (f", polyorder={savgolPolyorder}" if smoothMethod == 'savgol' else '')
            )

        # Export Processed Data                         
        os.makedirs(path, exist_ok=True)
        df_ec.to_csv(os.path.join(path, filename_ec), index=False)
        df_eca.to_csv(os.path.join(path, filename_eca), index=False)
    
        print('After binning length ECa', len(df_eca))
        print('After binning length EC', len(df_ec))
        print('Incomplete calibration bins removed', n_dropped)
    
        return df_eca, df_ec

    # ==========================================================
    # ---- INTERPOLATION-CONFIDENCE DIAGNOSTICS ----------------
    # support_field / support_by_depth / dispersion_field /
    # loocv_interpolation / plot_confidence_section
    # ==========================================================

    def _query_neighbors(self, mesh_df, source_df, depth_col="Z"):
        """Return cleaned mesh, cleaned source, and the (n_mesh, k) index array
        with the cKDTree sentinel preserved. Mirrors mapXYZ's 'num' path."""
        mesh = (mesh_df.replace([np.inf, -np.inf], np.nan)
                       .dropna(subset=["x", "y", depth_col]).reset_index(drop=True))
        src = (source_df.replace([np.inf, -np.inf], np.nan)
                        .dropna(subset=["x", "y", depth_col]).reset_index(drop=True))

        max_dist = np.sqrt(self.max_distance_xy ** 2 + self.max_distance_z ** 2)
        tree = cKDTree(src[["x", "y", depth_col]].values)
        k = max(int(self.num_neighbors), 1)

        dist, idx = tree.query(mesh[["x", "y", depth_col]].values, k=k,
                               distance_upper_bound=max_dist, workers=self.njobs)
        if k == 1:                       # force 2D for uniform downstream handling
            dist, idx = dist[:, None], idx[:, None]
        return mesh, src, dist, idx

    @staticmethod
    def _keep_valid(neigh_row, mesh_z_i, src_z, n_src, max_dz):
        """Drop the cKDTree sentinel and enforce the anisotropic Z tolerance,
        exactly as mapXYZ does when it removes points beyond max_distance_z."""
        j = neigh_row[neigh_row != n_src]
        if j.size:
            j = j[np.abs(src_z[j] - mesh_z_i) <= max_dz]
        return j

    # ================================================================== #
    # 1) SPATIAL SUPPORT  (distance field + n_eff + fill fraction)
    # ================================================================== #
    def support_field(self, mesh_df, source_df, depth_col="Z", keep_cols=None,
                      group_col=None):
        """One row per mesh element describing how well the source data support
        the interpolated value there.

        Columns: x, y, Z, nn_dist (3D), nn_dist_xy, nn_dist_z, n_eff, supported.
        A cell with supported == False received no real neighbor and is handled
        by fill_missing (mean / value / griddata), i.e. it is extrapolated, not
        interpolated. Sets self.fill_fraction = share of such cells.
        """
        mesh, src, _, idx = self._query_neighbors(mesh_df, source_df, depth_col)
        mesh_xyz = mesh[["x", "y", depth_col]].values
        src_xyz = src[["x", "y", depth_col]].values
        src_z = src_xyz[:, 2]
        n_src = len(src)
        src_grp = (src[group_col].values
                   if (group_col is not None and group_col in src.columns) else None)
        nsnd = []

        rows = []
        for i in range(len(mesh)):
            j = self._keep_valid(idx[i], mesh_xyz[i, 2], src_z, n_src,
                                 self.max_distance_z)                
            n_eff = int(j.size)                                       
            if n_eff:
                d3 = np.sqrt(((src_xyz[j] - mesh_xyz[i]) ** 2).sum(axis=1))
                m = d3.argmin()
                nn_dist = float(d3[m])
                nn_xy = float(np.hypot(src_xyz[j[m], 0] - mesh_xyz[i, 0],
                                       src_xyz[j[m], 1] - mesh_xyz[i, 1]))
                nn_z = float(abs(src_xyz[j[m], 2] - mesh_xyz[i, 2]))
            else:
                nn_dist = nn_xy = nn_z = np.nan
            if src_grp is not None:
                nsnd.append(int(np.unique(src_grp[j]).size) if n_eff else 0)
            rows.append((mesh_xyz[i, 0], mesh_xyz[i, 1], mesh_xyz[i, 2],
                         nn_dist, nn_xy, nn_z, n_eff, n_eff > 0))

        sup = pd.DataFrame(rows, columns=["x", "y", depth_col, "nn_dist",
                                          "nn_dist_xy", "nn_dist_z", "n_eff",
                                          "supported"])
        if src_grp is not None:
            sup["n_soundings"] = nsnd
        if keep_cols:
            for _c in keep_cols:
                if _c in mesh.columns:
                    sup[_c] = mesh[_c].values
        self.unsupported_fraction = float(1.0 - sup["supported"].mean())
        # Backward-compatible alias retained for older notebooks.
        self.fill_fraction = self.unsupported_fraction
        return sup

    def support_field_xy(self, mesh_df, source_df, depth_col="Z",
                         keep_cols=None, group_col=None):
        """Describe horizontal support for :meth:`mapXY` at every target row.

        Unlike ``support_field``, this method deliberately ignores vertical
        separation.  It is therefore the matching diagnostic for ``mapXY``.

        Parameters
        ----------
        mesh_df : pandas.DataFrame
            Target rows containing ``x`` and ``y``. ``depth_col`` is copied
            when present so the result can be plotted as an X-depth section.
        source_df : pandas.DataFrame
            Source observations containing ``x`` and ``y``.
        depth_col : str, default "Z"
            Optional target depth/elevation column to retain.
        keep_cols : sequence of str or None
            Additional target columns to copy to the result (for example X).
        group_col : str or None
            Optional source identifier.  When supplied, ``n_soundings`` is the
            number of unique groups among the accepted neighbors.

        Returns
        -------
        pandas.DataFrame
            Target coordinates, nearest horizontal distance ``nn_dist``,
            effective neighbor count ``n_eff``, and ``supported`` flag.
        """
        mesh = (mesh_df.replace([np.inf, -np.inf], np.nan)
                       .dropna(subset=["x", "y"]).reset_index(drop=True))
        src = (source_df.replace([np.inf, -np.inf], np.nan)
                        .dropna(subset=["x", "y"]).reset_index(drop=True))
        if mesh.empty or src.empty:
            raise ValueError("mesh_df and source_df need finite x/y coordinates.")

        tree = cKDTree(src[["x", "y"]].values)
        k = max(int(self.num_neighbors), 1)
        dist, idx = tree.query(
            mesh[["x", "y"]].values,
            k=k,
            distance_upper_bound=self.max_distance_xy,
            workers=self.njobs,
        )
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]

        groups = (src[group_col].to_numpy()
                  if group_col is not None and group_col in src.columns else None)
        rows = []
        for i in range(len(mesh)):
            valid = (idx[i] != len(src)) & np.isfinite(dist[i])
            d = dist[i][valid]
            j = idx[i][valid]
            n_eff = int(j.size)
            row = {
                "x": mesh.loc[i, "x"],
                "y": mesh.loc[i, "y"],
                "nn_dist": float(np.min(d)) if n_eff else np.nan,
                # Keep the same distance-column schema as support_field().
                # For an XY-only search, total distance equals horizontal
                # distance and vertical distance is not evaluated.
                "nn_dist_xy": float(np.min(d)) if n_eff else np.nan,
                "nn_dist_z": np.nan,
                "n_eff": n_eff,
                "supported": bool(n_eff),
            }
            if depth_col in mesh.columns:
                row[depth_col] = mesh.loc[i, depth_col]
            if groups is not None:
                row["n_soundings"] = int(np.unique(groups[j]).size) if n_eff else 0
            rows.append(row)

        sup = pd.DataFrame(rows)
        if keep_cols:
            for col in keep_cols:
                if col in mesh.columns and col not in sup.columns:
                    sup[col] = mesh[col].values
        self.unsupported_fraction = float(1.0 - sup["supported"].mean())
        self.fill_fraction = self.unsupported_fraction
        return sup

    def support_by_depth(self, support_df, depth_col="Z", nbins=12):
        """Depth-resolved roll-up of support_field(): the panel that explains
        where (with depth) interpolation confidence collapses, and would flag
        a non-informative pairing before any inversion is run."""
        s = support_df.copy()
        required = {depth_col, "supported", "nn_dist", "n_eff"}
        missing = sorted(required.difference(s.columns))
        if missing:
            raise KeyError(f"support_df is missing required columns: {missing}")
        s["depth_bin"] = pd.cut(s[depth_col], bins=nbins)
        aggregations = {
            "n_cells": ("supported", "size"),
            "frac_supported": ("supported", "mean"),
            "median_nn_dist": ("nn_dist", "median"),
            "mean_n_eff": ("n_eff", "mean"),
        }
        # XYZ support provides both components. XY support now exposes the
        # same schema, but these guards also accept older support tables.
        if "nn_dist_xy" in s.columns:
            aggregations["median_nn_dist_xy"] = ("nn_dist_xy", "median")
        if "nn_dist_z" in s.columns:
            aggregations["median_nn_dist_z"] = ("nn_dist_z", "median")
        return (s.groupby("depth_bin", observed=True)
                 .agg(**aggregations)
                 .reset_index())

    # ================================================================== #
    # 2) LOCAL DISPERSION  (log std + geometric CV + 3-means divergence)
    # ================================================================== #
    def dispersion_field(self, mesh_df, source_df, value_cols=None,
                         depth_col="Z", skip_substr="inph", keep_cols=None,
                         group_col=None):
        """Per-cell spread of the neighbor set, for each mapped variable.

        For each value column it adds:
          {col}_logstd  : std of log10(neighbors)  (ratio-space spread)
          {col}_gcv     : geometric coefficient of variation
          {col}_meandiv : (arith - harm) / geom, the divergence among the three
                          means mapXYZ already computes (0 if neighbors equal;
                          grows with skew / heterogeneity).
        Columns containing `skip_substr` (e.g. in-phase, which can be negative)
        are skipped for the log/geom-based measures. Meaningful only for
        num_neighbors > 1.
        """
        mesh, src, _, idx = self._query_neighbors(mesh_df, source_df, depth_col)
        if value_cols is None:
            value_cols = [c for c in src.columns if c not in ("x", "y", depth_col)]
        src_z = src[depth_col].values
        n_src = len(src)
        mesh_z = mesh[depth_col].values
        src_grp = (src[group_col].values
                   if (group_col is not None and group_col in src.columns) else None)

        res = mesh[["x", "y", depth_col]].copy()
        for col in value_cols:
            if skip_substr and skip_substr in col:
                continue
            vals = src[col].values
            logstd = np.full(len(mesh), np.nan)
            gcv = np.full(len(mesh), np.nan)
            meandiv = np.full(len(mesh), np.nan)
            between = np.full(len(mesh), np.nan)   # lateral spread across soundings
            within = np.full(len(mesh), np.nan)    # vertical spread within a sounding
            for i in range(len(mesh)):
                j = self._keep_valid(idx[i], mesh_z[i], src_z, n_src,
                                     self.max_distance_z)
                vr = vals[j]
                gr = src_grp[j] if src_grp is not None else None
                fin = np.isfinite(vr)
                vr = vr[fin]
                if gr is not None:
                    gr = gr[fin]
                posm = vr > 0                       # log / geom need positives
                vp = vr[posm]
                gp = gr[posm] if gr is not None else None
                if vp.size >= 2:
                    s_ln = np.log(vp).std(ddof=1)
                    logstd[i] = np.log10(vp).std(ddof=1)
                    gcv[i] = np.sqrt(np.expm1(s_ln ** 2))
                    a, g, h = vp.mean(), gmean(vp), hmean(vp)
                    meandiv[i] = (a - h) / g if g > 0 else np.nan
                    if gp is not None:
                        lv = np.log10(vp)
                        means, varis = [], []
                        for gid in np.unique(gp):
                            sel = gp == gid
                            means.append(lv[sel].mean())
                            if sel.sum() >= 2:
                                varis.append(lv[sel].var(ddof=1))
                        means = np.asarray(means)
                        between[i] = float(means.std(ddof=1)) if means.size >= 2 else 0.0
                        within[i] = float(np.sqrt(np.mean(varis))) if varis else 0.0
            res[f"{col}_logstd"] = logstd
            res[f"{col}_gcv"] = gcv
            res[f"{col}_meandiv"] = meandiv
            if src_grp is not None:
                res[f"{col}_between"] = between
                res[f"{col}_within"] = within
        if keep_cols:
            for _c in keep_cols:
                if _c in mesh.columns:
                    res[_c] = mesh[_c].values
        return res

    # ================================================================== #
    # 3) LEAVE-ONE-OUT CROSS-VALIDATION of the interpolation
    # ================================================================== #
    def loocv_interpolation(self, source_df, value_col, mean="arith",
                            depth_col="Z", keep_cols=None, group_col=None):
        """Cross-validate the interpolation in log10 space.

        group_col=None -> leave-one-POINT-out: each point is held out and
        predicted from its k nearest other points. group_col set (e.g. a
        sounding / borehole id) -> leave-one-SOUNDING-out: every point of a
        group is withheld together and predicted only from OTHER groups. Use
        the grouped form whenever the source points are not mutually
        independent -- e.g. a per-sounding 1D inversion, where points sharing a
        column come from one inverted model and point-wise LOOCV would be
        over-optimistic (a held-out point is "predicted" mostly by its own
        sounding).

        Returns
        -------
        residuals : DataFrame [x, y, Z, observed, predicted, resid_log10
                    (+ group_col and keep_cols when supplied)]
        summary   : dict (RMSE_log, MAE_log, bias_log, median_ratio, counts,
                    cv_mode; n_groups and per_group_RMSE_log when grouped)
        """
        subset = ["x", "y", depth_col, value_col]
        if group_col is not None:
            subset = subset + [group_col]
        src = (source_df.replace([np.inf, -np.inf], np.nan)
                        .dropna(subset=subset).reset_index(drop=True))
        src = src[src[value_col] > 0].reset_index(drop=True)   # log needs > 0
        xyz = src[["x", "y", depth_col]].values
        v = src[value_col].values
        n = len(src)
        k = max(int(self.num_neighbors), 1)
        max_dist = np.sqrt(self.max_distance_xy ** 2 + self.max_distance_z ** 2)
        reducer = {"arith": np.mean, "geom": gmean, "harm": hmean}[mean]

        def _predict(j, xyz_in, v_in, z0):
            if j.size:
                j = j[np.abs(xyz_in[j, 2] - z0) <= self.max_distance_z]
            if not j.size:
                return np.nan
            vals = v_in[j]
            return reducer(vals) if (mean == "arith" or np.all(vals > 0)) else np.mean(vals)

        pred = np.full(n, np.nan)
        if group_col is None:
            tree = cKDTree(xyz)
            _, idx = tree.query(xyz, k=k + 1, distance_upper_bound=max_dist,
                                workers=self.njobs)        # k+1 so self can be dropped
            if k + 1 == 1:
                idx = idx[:, None]
            for i in range(n):
                j = idx[i]
                j = j[(j != i) & (j != n)]                 # drop self + sentinel
                pred[i] = _predict(j, xyz, v, xyz[i, 2])
        else:
            groups = src[group_col].values
            for g in np.unique(groups):                    # leave one sounding out
                hold = np.where(groups == g)[0]
                keep = groups != g
                if not keep.any():
                    continue
                xyz_in, v_in = xyz[keep], v[keep]
                tree_g = cKDTree(xyz_in)
                _, ii = tree_g.query(xyz[hold], k=k,
                                     distance_upper_bound=max_dist, workers=self.njobs)
                if k == 1:
                    ii = ii[:, None]
                nin = len(xyz_in)
                for row, hi in enumerate(hold):
                    j = ii[row]
                    j = j[j != nin]                        # drop sentinel
                    pred[hi] = _predict(j, xyz_in, v_in, xyz[hi, 2])

        ok = np.isfinite(pred)
        resid = np.log10(pred[ok]) - np.log10(v[ok])
        residuals = pd.DataFrame({
            "x": xyz[ok, 0], "y": xyz[ok, 1], depth_col: xyz[ok, 2],
            "observed": v[ok], "predicted": pred[ok], "resid_log10": resid})
        if group_col is not None:
            residuals[group_col] = src[group_col].values[ok]
        if keep_cols:
            for _c in keep_cols:
                if _c in src.columns and _c not in residuals.columns:
                    residuals[_c] = src[_c].values[ok]

        if resid.size:
            rmse = float(np.sqrt(np.mean(resid ** 2))); mae = float(np.mean(np.abs(resid)))
            bias = float(np.mean(resid)); medr = float(10 ** np.median(resid))
        else:
            rmse = mae = bias = medr = float("nan")
        summary = {
            "n_used": int(ok.sum()),
            "n_unpredicted": int((~ok).sum()),
            "RMSE_log": rmse, "MAE_log": mae,
            "bias_log": bias,                  # +ve => interpolation over-predicts
            "median_ratio": medr,
            "cv_mode": "leave-one-sounding-out" if group_col is not None else "leave-one-point-out",
        }
        if group_col is not None and resid.size:
            grp = residuals.groupby(group_col)["resid_log10"]
            summary["n_groups"] = int(grp.ngroups)
            summary["per_group_RMSE_log"] = grp.apply(
                lambda s: float(np.sqrt(np.mean(s.values ** 2)))).to_dict()
        return residuals, summary

    # ================================================================== #
    # shared plotting helper (x-Z section) -- callable from VSE
    # ================================================================== #
    @staticmethod
    def plot_confidence_section(df, value, depth_col="Z", x_col="x",
                                title=None, cmap="viridis", log_color=False,
                                center=None, value_space=None, clabel=None,
                                clip=None, units=None, log_units=False,
                                xlim=None, figname=None, ax=None, plotDepth=False,
                                fontKws=None, **scatter_kw):
        """Scatter any per-cell confidence field on a length-vs-depth section.

        Pass `ax` to draw into an existing panel: no figure-level tight_layout
        or show is triggered, and the title, labels, ticks, marker size and
        colorbar all scale to the panel size so the section stays legible in a
        crowded 2x2.

        center : float or None
            If given (e.g. 0.0 for signed LOOCV residuals), use a diverging
            TwoSlopeNorm centred on this value so that `center` renders white.
        xlim : tuple or None
            Shared x-axis limits (e.g. the mesh X range) so multiple sections
            line up exactly.
        """
        from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(8, 4))
        s = DataMapper._ax_scale(ax)

        c = np.asarray(df[value].values, dtype=float)
        ctr = center
        if value_space == "ratio":          # show a log10 residual in linear (ratio) form
            c = np.power(10.0, c)
            if ctr is not None:
                ctr = 10.0 ** ctr

        extend = "neither"
        if ctr is not None:
            lim = np.nanpercentile(np.abs(c - ctr), 98)
            lim = lim if (np.isfinite(lim) and lim > 0) else (abs(ctr) or 1.0)
            vmin, vmax = ctr - lim, ctr + lim
            if value_space == "ratio":
                vmin = max(vmin, 1e-9)      # ratios are strictly positive
            norm = TwoSlopeNorm(vcenter=ctr, vmin=vmin, vmax=vmax)
            extend = "both"
            if cmap == "viridis":
                cmap = "coolwarm"
        elif log_color:
            pos = c[c > 0]
            norm = LogNorm(vmin=np.nanmin(pos), vmax=np.nanmax(c)) if pos.size else None
        elif clip is not None:
            lo, hi = np.nanpercentile(c, clip)   # robust limits so outliers
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):   # don't flatten the panel
                lo, hi = np.nanmin(c), np.nanmax(c)
            norm = Normalize(vmin=lo, vmax=hi)
            extend = "both"
        else:
            norm = None

        depth = np.asarray(df[depth_col].values, dtype=float)
        if np.max(depth) < 0:
            depth = -depth
            depthOrElev = 'Depth'
        elif (np.max(depth) > 0) and plotDepth:
            depth -= (np.nanmax(depth) + 0.01)
            depth *= -1
            depthOrElev = 'Depth'
        else:
            depthOrElev = 'Elevation'

        sc = ax.scatter(df[x_col], depth, c=c, cmap=cmap, norm=norm,
                        s=scatter_kw.pop("s", max(2.0, 12 * s ** 2)), **scatter_kw)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, extend=extend)
        cb.set_label(DataMapper._units_label(clabel or value, units,
                                             log=(log_units or value_space == "log10"),
                                             dimensionless=(value_space == "ratio")),
                     fontsize=14 * s, fontweight='bold')
        cb.ax.tick_params(labelsize=12 * s)
        ax.set_xlabel("Length (m)", fontsize=14 * s, fontweight='bold')
        ax.set_ylabel(depthOrElev + " (m)", fontsize=14 * s, fontweight='bold')
        ax.tick_params(labelsize=12 * s)
        if depthOrElev == 'Depth':
            ax.invert_yaxis()
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(title or value, fontsize=16 * s, fontweight='bold')
        if standalone:
            ax.figure.tight_layout()
        if figname and standalone:
            ax.figure.savefig(figname, dpi=300)
        return ax

    @staticmethod
    def doi_filter(source_df, doi_col, depth_col="Z", depth_positive_down=False):
        """Drop source points deeper than their depth-of-investigation (DOI).

        Each 1D EMI sounding is only constrained above its DOI; deeper layers
        are unconstrained model space and should not seed or validate the
        colocation. `doi_col` holds the DOI as a positive depth magnitude (m).
        With depth_positive_down=False (default; Z increases upward, negative
        down) a point is kept when Z >= -DOI. Rows with non-finite DOI are kept.
        """
        df = source_df.copy()
        doi = np.abs(df[doi_col].values.astype(float))
        z = df[depth_col].values.astype(float)
        if depth_positive_down:
            keep = ~np.isfinite(doi) | (z <= doi)
        else:
            keep = ~np.isfinite(doi) | (z >= -doi)
        return df[keep].reset_index(drop=True)
    
    def build_comparison_grid(self, grid, models, agg="geom",
                              coord_cols=("x", "y", "Z"), keep_cols=("X",),
                              max_dist=None, require_all=True):
        """Aggregate several models onto ONE common comparison grid (fine -> coarse)
        so each cell holds a single representative value per model, for fair and
        approximately independent cell-by-cell statistics.

        Unlike mapXYZ (nearest source -> mesh, which REPLICATES a coarse source
        across many fine cells), this assigns every source cell to its nearest
        comparison-grid cell and AVERAGES within the cell -- the correct
        direction when the comparison grid is at (or below) each model's native
        resolution. Build the grid at the limiting (coarsest, usually EMI)
        resolution so every model is smoothed equally.

        Parameters
        ----------
        grid : DataFrame
            Comparison-cell centroids; must contain `coord_cols` (+ any
            `keep_cols`, e.g. along-line 'X').
        models : dict
            {output_name: {"df": DataFrame, "value": value_col,
                           "area": area_col or None, "mask": flag_col or None}}
            Rows where `mask` == 1 (e.g. a DOI flag) are dropped before
            aggregation.
        agg : {"geom","arith","median"}
            Within-cell aggregation. "geom" (default) and "arith" honour `area`
            weights; "median" is unweighted. EC is skewed/log -> "geom".
        max_dist : float or None
            Drop source cells farther than this (Euclidean in coord space) from
            every grid centroid. None keeps all.
        require_all : bool
            Keep only cells with a valid value from EVERY model (common support).

        Returns
        -------
        DataFrame: coord_cols + present keep_cols + one column per model +
        "<name>_n" (source-cell count aggregated into that cell, per model).
        Write it to CSV and feed straight to EConfluxStats.
        """
        coord_cols = list(coord_cols)
        g = grid.replace([np.inf, -np.inf], np.nan).dropna(subset=coord_cols).reset_index(drop=True)
        out = g[coord_cols + [c for c in keep_cols if c in g.columns]].copy()
        ncells = len(out)
        tree = cKDTree(g[coord_cols].values)

        for name, spec in models.items():
            vcol = spec["value"]; acol = spec.get("area"); mcol = spec.get("mask")
            sub = spec["df"].copy()
            if mcol is not None and mcol in sub.columns:
                sub = sub[sub[mcol] != 1]
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=coord_cols + [vcol])
            sub = sub[sub[vcol] > 0]
            agg_val = np.full(ncells, np.nan); cnt = np.zeros(ncells, dtype=int)
            if len(sub):
                dist, gi = tree.query(sub[coord_cols].values)
                if max_dist is not None:
                    ok = dist <= max_dist
                    sub, gi = sub[ok], gi[ok]
                v = sub[vcol].values.astype(float)
                w = (np.abs(sub[acol].values.astype(float))
                     if (acol is not None and acol in sub.columns) else np.ones(len(v)))
                np.add.at(cnt, gi, 1)
                if agg == "median":
                    med = pd.DataFrame({"g": gi, "v": v}).groupby("g")["v"].median()
                    agg_val[med.index.values] = med.values
                else:
                    wsum = np.zeros(ncells); np.add.at(wsum, gi, w)
                    nz = wsum > 0
                    if agg == "geom":
                        ls = np.zeros(ncells); np.add.at(ls, gi, w * np.log(v))
                        agg_val[nz] = np.exp(ls[nz] / wsum[nz])
                    else:  # arith
                        vs = np.zeros(ncells); np.add.at(vs, gi, w * v)
                        agg_val[nz] = vs[nz] / wsum[nz]
            out[name] = agg_val
            out[name + "_n"] = cnt

        if require_all:
            out = out.dropna(subset=list(models.keys())).reset_index(drop=True)
        return out