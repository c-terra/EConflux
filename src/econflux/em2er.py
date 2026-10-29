"""em2er.py

Self-contained mapper to seed ResIPy k.mesh.df['res0'] from EMI data.
Includes the helper `local_grid_to_utm_along_line` so no external geometry
utilities are required.

Exports:
- local & UTM grid (either via 'local_grid_to_utm_along_line' or 'direct_local_grid_to_um_along_line')
- Informed ERI mesh (via em2er_map)
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from math import ceil, floor
from econflux.er2em import DataMapper
import matplotlib.tri as tri

def local_grid_to_utm_along_line(mesh_df: pd.DataFrame,
                                 utm_pair1: Tuple[float, float],
                                 utm_pair2: Tuple[float, float]) -> pd.DataFrame:
    """
    Convert local mesh (X, Y) coordinates into UTM (easting, northing)
    along a line defined by two UTM points.

    Parameters
    ----------
    mesh_df : DataFrame with at least ['X', 'Y']
    utm_pair1, utm_pair2 : tuples (easting, northing)

    Returns
    -------
    DataFrame with columns ['easting', 'northing']
    """
    x1, y1 = utm_pair1
    x2, y2 = utm_pair2

    dx = float(x2 - x1)
    dy = float(y2 - y1)
    L = float(np.hypot(dx, dy))
    if L == 0:
        raise ValueError("utm_pair1 and utm_pair2 cannot be the same point.")

    ux, uy = dx / L, dy / L  # unit vector along the line

    # Rotation + translation (note: +X along line; +Y to the left)
    easting = x1 + mesh_df["X"] * ux - mesh_df["Y"] * uy
    northing = y1 + mesh_df["X"] * uy + mesh_df["Y"] * ux

    return pd.DataFrame({"easting": easting, "northing": northing})


def direct_local_grid_to_utm_along_line(local_grid_df, coord_list, polyDeg=3, numSamples=1000, lineLength=95, correctCoords=False, checkLine=False, saveElec=False, elecPath=None):
    
    if (max(coord_list['x']) - min(coord_list['x'])) > (max(coord_list['y']) - min(coord_list['y'])):
        func = P.fit(coord_list['x'], coord_list['y'], polyDeg)
        xx, yy = func.linspace(len(coord_list))
        sampX, sampY = func.linspace(numSamples)
        if checkLine:
            plt.plot(coord_list['x'], coord_list['y'], 'o', label='Original Points')
            plt.plot(xx, yy, lw=2, label='Interpolated Line')
            plt.gca().set_aspect('equal')
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
            plt.show()
            plt.close()
    else:
        func = P.fit(coord_list['y'], coord_list['x'], polyDeg)
        yy, xx = func.linspace(len(coord_list))
        sampY, sampX = func.linspace(numSamples)
        if checkLine:
            plt.plot(coord_list['y'], coord_list['x'], 'o', label='Original Points')
            plt.plot(yy, xx, lw=2, label='Interpolated Line')
            plt.gca().set_aspect('equal')
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1.6))
            plt.show()
            plt.close()
    
    if correctCoords:
        coord_list['x'], coord_list['y'] = xx, yy
    
    linePos = np.zeros(numSamples)
    linePos = np.sqrt((sampX**2)+(sampY**2))
    if linePos[0] > linePos[-1]:
        relLinePos = abs((linePos - max(linePos)) / (max(linePos)-min(linePos)))
    else:
        relLinePos = (linePos - min(linePos)) / (max(linePos)-min(linePos))
    relLinePos *= lineLength
    
    elecDf = pd.DataFrame(np.transpose([relLinePos, np.zeros(numSamples)]), columns=['x', 'z'])
    
    tree = cKDTree(elecDf[['x', 'z']])
    dist, idx = tree.query(local_grid_df[['X', 'Z']])

    for i in range(0, len(local_grid_df)):
        local_grid_df.loc[i, ['x', 'y']] = [sampX[idx[i]], sampY[idx[i]]]

    if saveElec:
        coord_list.to_csv(elecPath + 'corrElecCoords.csv', index=False)
        
    if correctCoords:
        return local_grid_df, coord_list
    else:
        return local_grid_df

class em2er_map:
    """
    Map EMI inversion values onto a mesh for ERI inversion.

    Parameters
    ----------
    mesh : pd.DataFrame or path to .csv
        Dataframe or .csv to file with mesh information. 
        Must have 'X', 'Y', and 'Z' columns and a column for starting resistivity.
    resCol : str
        Name of column in 'mesh' that contains starting resistivity values
    elecUTM : pd.DataFrame
        Copy of electrodes from inversion project (k.elec) with UTM 'x', 'y', and 'z'
    emi : str
        Path to the EMI .csv (X, Y, Depth, Linear Resistivity columns present)
    linearOrPoly : str
        Define whether to use a linear or polynomial interpolation to convert UTM to local coordintes
    alphaomega : Tuple[int, int]
        Positional indexes into elecUTM to define the line direction.
        Defaults to the first and last index of elecUTM but should be the start and end of your survey line (start elec index, end elec index)
        Only used for local grid to UTM transformation and if linearOrPoly == 'linear'.
    elecSpacing : float
        Electrode spacing used in survey. Used to determine survey length. Default is 1.
    polydeg : int
        Polynomial degree to use for electrode interpolation (if linearOrInterp != 'linear'). Default is 3.
    mapvars : dict
        Passed to testcalib.DataMapper(num_neighbors=2, max_distance_xy=5, max_distance_z=0.3) for ease of updating this function down the line.
    emires_col : str
        Column from the mapped dataframe to use for res0 merge
        (default: "_Linear Resistivity_arithmetic_mean", other options available in the DataMapper function).
    keep_cols : Optional[Tuple[str, str, str]]
        Coordinate column names (default: ('X','Y','Z')), optional.
    meshobj : Optional[resipy.project]
        ResIPy Project object. Will be used in place of 'mesh' parameter and will return informed mesh already applied to Project.
        Your ResIPy inversion project mesh and electrodes must already be defined
    """

    def __init__(
        self,
        mesh: pd.DataFrame or str,
        resCol: str,
        elecUTM: pd.DataFrame,
        emi: pd.DataFrame or str,
        linearOrPoly: str,
        alphaomega: Tuple[int, int] = [0, 1],
        elecSpacing: float = 1,
        polydeg: int = 3,
        mapvars: Optional[Dict[str, Any]] = dict(num_neighbors=5, max_distance_xy=5, max_distance_z=0.3),
        emires_col: Optional[str] = "_Linear Resistivity_arith",
        keep_cols: Optional[Tuple[str, str, str]] = ("X", "Y", "Z"),
        meshobj: Optional = None,
        units: Optional = 'm'
    ):
            
        self.mesh = mesh
        self.resCol = resCol
        self.elecUTM = elecUTM
        self.emi = emi
        self.linearOrPoly = linearOrPoly
        self.alphaomega = alphaomega
        self.elecSpacing = elecSpacing
        self.polydeg = polydeg
        self.mapvars = mapvars
        self.emires_col = emires_col
        self.keep_cols = keep_cols
        self.meshobj = meshobj
        self.units = units

        if type(mesh) == str:
            self.mesh = pd.read_csv(mesh)
        elif self.meshobj is not None:
            mesh = self.meshobj.mesh.df
        else:
            self.mesh = mesh

        self.survLen = (len(elecUTM)-1) * self.elecSpacing

        #populated during run()
        self.utm_pair1: Optional[Tuple[float, float]] = None
        self.utm_pair2: Optional[Tuple[float, float]] = None
        self.meshUTM: Optional[pd.DataFrame] = None
        self.emi_inv: Optional[pd.DataFrame] = None
        self.emi2mesh: Optional[pd.DataFrame] = None
        self.dist = None

    def prep_mesh_utm(self): #convert electrode positions to UTM from just electrode numbers
        if self.linearOrPoly == 'linear':
            i1, i2 = self.alphaomega
            self.utm_pair1 = (self.elecUTM['x'].iloc[i1], self.elecUTM['y'].iloc[i1])
            self.utm_pair2 = (self.elecUTM['x'].iloc[i2], self.elecUTM['y'].iloc[i2])
    
            #convert local grid points to UTM coordinates along the line
            xyData_UTM = local_grid_to_utm_along_line(self.mesh.copy(), self.utm_pair1, self.utm_pair2)
            
            meshUTM = pd.concat([self.mesh.copy(), xyData_UTM], axis=1)
            meshUTM = meshUTM.rename(columns={'easting': 'x', 'northing': 'y'})

        else:
            meshUTM = direct_local_grid_to_utm_along_line(self.mesh, self.elecUTM, self.polydeg, lineLength=self.survLen, checkLine=True)
        #standardize coordinate naming and de-duplicate
        meshUTM = meshUTM.loc[:, ~meshUTM.columns.duplicated()]

        #remove unnecessary columns
        meshUTM = meshUTM[['X', 'Y', 'Z', 'x', 'y']]
        self.meshUTM = meshUTM

    def load_emi(self): #load EMI csv
        if type(self.emi) == str:
            emi_inv = pd.read_csv(self.emi)
        else:
            emi_inv = self.emi.copy()
        #match column names expected by mapping function
        emi_inv = emi_inv.rename(columns={'X': 'x', 'Y': 'y', 'Depth': 'Z'})
        self.emi_inv = emi_inv

    def map_to_mesh(
        self,
        correctTopo: str or bool = 'auto',
        topoTol: Optional[float] = None,
        surfacePts: int = 12,
        maxDpth: Optional[float] = None,
        elev_or_depth: Optional[str] = None,
        zBySounding: bool = False,
        zTol: float = 1.0
    ):
        """
        Map EMI inversion models onto the ERI mesh using distance-weighted
        spatial interpolation. Optionally applies topographically corrected
        mapping and filters EMI points by a maximum depth cutoff.

        correctTopo : {"auto", True, False}
            Controls topography correction during spatial mapping. "auto"
            automatically detects whether topographic surface correction is
            required.
        topoTol : Optional[float]
            Elevation tolerance threshold for topographic surface matching.
            If provided, also acts as the tolerance for coordinate similarity
            checks unless overridden by `zTol`.
        surfacePts : int
            Number of near-surface points used to construct the topography surface.
            Default is 12.
        maxDpth : Optional[float]
            Maximum depth cutoff value. EMI data points below this depth are
            excluded prior to spatial mapping onto the mesh.
        elev_or_depth : {"elev", "depth", None}
            Specifies whether the Z-coordinates in the EMI models and ERI mesh
            are represented as elevations ("elev") or depths ("depth"). Required
            when `maxDpth` is defined or when verifying coordinate representation.
        zBySounding : bool
            If True and `elev_or_depth` is "elev", shifts each EMI sounding
            independently from an elevation to a depth relative to its local
            surface elevation. If False, all soundings are shifted at once relative
            to the overall maximum elevation in the EMI dataset. Default is False.
        zTol : float
            Tolerance threshold for surface Z-coordinate similarity checks.
            When `elev_or_depth` is "elev", checks similarity between maximum Z
            values (`emi_z_max` and `mesh_z_max`). When `elev_or_depth` is "depth",
            checks similarity between minimum Z values (`emi_z_min` and `mesh_z_min`).
            Default is 1.0.
        """
        if self.meshUTM is None or self.emi_inv is None:
            raise RuntimeError("Run prep_mesh_utm() and load_emi() before map_to_mesh().")

        # Validate coordinate representation inputs
        if maxDpth is not None and elev_or_depth is None:
            raise ValueError("elev_or_depth must be specified ('elev' or 'depth') when maxDpth is provided.")

        # Ensure representation consistency between EMI models and ERI mesh
        if elev_or_depth is not None:
            if elev_or_depth not in ('elev', 'depth'):
                raise ValueError(f"Unknown elev_or_depth setting: '{elev_or_depth}'. Expected 'elev' or 'depth'.")

            tol = topoTol if topoTol is not None else zTol

            if elev_or_depth == 'elev':
                # Check similarity between maximum Z values (surface elevations)
                emi_z_max = float(self.emi_inv['Z'].max())
                mesh_z_max = float(self.meshUTM['Z'].max())
                diff = abs(emi_z_max - mesh_z_max)

                if diff > tol:
                    raise ValueError(
                        f"Representation mismatch: elev_or_depth='elev', but maximum Z values differ by {diff:.2f} "
                        f"(EMI max Z: {emi_z_max:.2f}, mesh max Z: {mesh_z_max:.2f}), exceeding tolerance ({tol}). "
                        f"Verify that both datasets are represented as elevations. If representation is correct, increase zTol."
                    )
            elif elev_or_depth == 'depth':
                # Check similarity between minimum Z values (surface level baseline)
                emi_z_min = float(self.emi_inv['Z'].min())
                mesh_z_min = float(self.meshUTM['Z'].min())
                diff = abs(emi_z_min - mesh_z_min)

                if diff > tol:
                    raise ValueError(
                        f"Representation mismatch: elev_or_depth='depth', but minimum Z values differ by {diff:.2f} "
                        f"(EMI min Z: {emi_z_min:.2f}, mesh min Z: {mesh_z_min:.2f}), exceeding tolerance ({tol}). "
                        f"Verify that both datasets are represented as depths. If representation is correct, increase zTol."
                    )

        # Apply depth cutoff filtering if maxDpth is defined
        if maxDpth is not None:
            if elev_or_depth == 'elev':
                if zBySounding:
                    # Shift each EMI sounding independently based on its local surface elevation
                    group_cols = [c for c in ['x', 'y'] if c in self.emi_inv.columns]
                    if group_cols:
                        z_surface = self.emi_inv.groupby(group_cols)['Z'].transform('max')
                    else:
                        z_surface = self.emi_inv['Z'].max()
                    depth = z_surface - self.emi_inv['Z']
                else:
                    # Shift all soundings at once relative to maximum overall elevation
                    z_surface = self.emi_inv['Z'].max()
                    depth = z_surface - self.emi_inv['Z']
            else:  # elev_or_depth == 'depth'
                depth = self.emi_inv['Z'].abs()

            # Filter EMI points exceeding the maximum depth threshold
            self.emi_inv = self.emi_inv[depth <= maxDpth].copy()

        # Execute spatial mapping from EMI models to ERI mesh
        mapper = DataMapper(**self.mapvars)
        self.mapper = mapper
        emi2mesh, dist = mapper.mapXYZ(
            self.meshUTM, [self.emi_inv], [''],
            topo_correct=correctTopo, topo_tol=topoTol, surf_k=surfacePts
        )
        self.emi2mesh = emi2mesh.copy()
        self.dist = dist

    def fill_missing(self, fillstrat='mean', res0fill=None, interpMeth='nearest',
                     interpBounds=None, area_col=None, region_col=None,
                     foreground_region=None, background='nearest',
                     check=False, plotParams={'vmin': None, 'vmax': None, 'cmap': 'viridis'}):
        """
        Fill mesh elements that received no mapped EMI value, producing a
        complete res0 starting model. Works on structured or unstructured
        (triangular) meshes -- everything operates on element centroids.

        fillstrat : {"mean","value","interp","none"}
            "mean"   fills each data column with its mean (area-weighted if
                     `area_col` is given).
            "value"  fills with `res0fill`.
            "interp" interpolates at the empty element centroids with
                     scipy.griddata(method=`interpMeth`); any cell still empty
                     (outside the data hull for 'linear'/'cubic', in a
                     background region, or outside `interpBounds`) is then
                     filled according to `background`.
            "none"   leaves NaNs.
        res0fill : Optional[float]
            Value for fillstrat="value".
        interpMeth : Optional[str]
            Method passed to scipy.griddata: 'nearest', 'linear', or 'cubic'.
        interpBounds : Optional[[(xmin,xmax),(zmin,zmax)]]
            If given, only cells inside these bounds are interpolated and the
            rest are handled by `background`. If None, defaults to interpolating
            10% on either side of the ERI survey and down to 1/3 the survey length.
        area_col : Optional[str]
            Per-element area/volume column used to area-weight the 'mean' fill
            and the 'mean'/'median' backgrounds. For ResIPy this is 'Area' (2D)
            or 'Volume' (3D), present only after calling k.mesh.cellArea().
            prep_mesh_utm() keeps only coordinates, so the column is re-attached
            here from self.mesh by centroid. Default None (unweighted, OFF).
        region_col : Optional[str]
            Region/marker column enabling region-aware filling: only foreground
            cells are interpolated, while other (background / padding) cells get
            `background`. For ResIPy this is 'region'. Default None (OFF -- every
            empty cell is interpolated).
        foreground_region : Optional[value or list]
            Marker value(s) of the foreground. If None and `region_col` is set,
            auto-detected as the region holding the most mapped values.
        background : float | {"nearest","mean","median"}
            How to fill cells not assigned by interpolation. A float sets a
            constant resistivity; 'nearest' takes the nearest mapped value;
            'mean'/'median' use a homogeneous summary of the mapped values
            (area-weighted if `area_col`). Default 'nearest'.
        check : Optional[bool]
            If True, scatter-plot the filled mesh centroids for inspection.
        plotParams : Optional[dict]
            'vmin','vmax','cmap' for the check plot.
        """
        if self.emi2mesh is None:
            raise ValueError('EMI models have not been mapped to ERI mesh. Run map_to_mesh() function first.')
        df = self.emi2mesh.copy()

        # prep_mesh_utm() trims the mesh to coordinates, so re-attach any
        # requested area/region column from self.mesh by element centroid.
        _added = []
        for extra in (area_col, region_col):
            if extra and extra not in df.columns:
                src = getattr(self, 'mesh', None)
                if (src is not None and extra in src.columns
                        and all(c in src.columns for c in ['X', 'Y', 'Z'])):
                    df = df.merge(
                        src[['X', 'Y', 'Z', extra]].drop_duplicates(['X', 'Y', 'Z']),
                        on=['X', 'Y', 'Z'], how='left')
                    _added.append(extra)
                else:
                    print(f"fill_missing: column '{extra}' not found in mesh; "
                          f"that option is disabled.")

        cols_to_exclude = set(self.keep_cols)
        for extra in (area_col, region_col):
            if extra:
                cols_to_exclude.add(extra)
        data_cols = [c for c in df.columns if c not in cols_to_exclude]

        def _aw(kn):                       # area weights for a boolean mask
            if area_col and area_col in df.columns:
                return np.abs(df.loc[kn, area_col].values.astype(float))
            return np.ones(int(kn.sum()))

        if fillstrat == "mean":            # homogeneous (optionally area-weighted) fill
            for col in data_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    kn = df[col].notna()
                    if kn.any():
                        df[col] = df[col].fillna(
                            self._wmean(df.loc[kn, col].values, _aw(kn)))

        elif fillstrat == "value":         # user-defined constant
            df[data_cols] = df[data_cols].fillna(res0fill)

        elif fillstrat == 'interp':        # direct-centroid interpolation + background tier
            # foreground selection (default: all cells)
            if region_col is not None and region_col in df.columns:
                if foreground_region is None:
                    kn_all = df[self.emires_col].notna()
                    fg = ([df.loc[kn_all, region_col].value_counts().idxmax()]
                          if kn_all.any() else list(df[region_col].dropna().unique()))
                else:
                    fg = (list(foreground_region)
                          if isinstance(foreground_region, (list, tuple, set))
                          else [foreground_region])
                in_fg = df[region_col].isin(fg)
            else:
                in_fg = pd.Series(True, index=df.index)

            in_bounds = pd.Series(True, index=df.index)
            if interpBounds is None:
                x0, x1, z0, z1 = 0 - (self.survLen*0.1), self.survLen*1.1, -(self.survLen/3), self.emi2mesh['Z'].max()
                in_bounds = ((df['X'] >= x0) & (df['X'] <= x1)
                             & (df['Z'] >= min(z0, z1)) & (df['Z'] <= max(z0, z1)))
            else:
                (x0, x1), (z0, z1) = interpBounds
                in_bounds = ((df['X'] >= x0) & (df['X'] <= x1)
                             & (df['Z'] >= min(z0, z1)) & (df['Z'] <= max(z0, z1)))

            for col in data_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                kn = df[col].notna()
                if not kn.any():
                    continue
                known_xz = df.loc[kn, ['X', 'Z']].values
                known_v = df.loc[kn, col].values.astype(float)

                # (1) interpolate directly at the empty foreground centroids
                tgt = (~kn) & in_fg & in_bounds
                if tgt.any():
                    df.loc[tgt, col] = griddata(known_xz, known_v,
                                                df.loc[tgt, ['X', 'Z']].values,
                                                method=interpMeth)

                # (2) fill anything still empty (out-of-hull / background / out-of-bounds)
                still = df[col].isna()
                if still.any():
                    if isinstance(background, (int, float)) and not isinstance(background, bool):
                        df.loc[still, col] = float(background)
                    elif background == 'nearest':
                        df.loc[still, col] = griddata(known_xz, known_v,
                                                      df.loc[still, ['X', 'Z']].values,
                                                      method='nearest')
                    else:                  # 'mean' / 'median' homogeneous background
                        df.loc[still, col] = self._summary(known_v, _aw(kn), background)

            if check:
                self._plot_interp_fill(df, plotParams, xlim=[x0, x1], zlim=[z0, z1])

        elif fillstrat == "none":
            pass
        else:
            raise ValueError(f"Unknown fillstrat: {fillstrat}")

        if _added:
            df = df.drop(columns=[c for c in _added if c in df.columns])
        self.emi2mesh = df

    # --- fill helpers ------------------------------------------------------
    @staticmethod
    def _wmean(vals, w):
        vals = np.asarray(vals, dtype=float); w = np.asarray(w, dtype=float)
        if vals.size == 0:
            return np.nan
        if not (np.isfinite(w).all() and w.sum() > 0):
            w = np.ones_like(vals)
        return float(np.sum(w * vals) / np.sum(w))

    @staticmethod
    def _wmedian(vals, w):
        vals = np.asarray(vals, dtype=float); w = np.asarray(w, dtype=float)
        if vals.size == 0:
            return np.nan
        if not (np.isfinite(w).all() and w.sum() > 0):
            w = np.ones_like(vals)
        o = np.argsort(vals); v, wv = vals[o], w[o]
        return float(v[np.searchsorted(np.cumsum(wv), 0.5 * wv.sum())])

    @classmethod
    def _summary(cls, vals, w, how):
        return cls._wmean(vals, w) if how == "mean" else cls._wmedian(vals, w)

    def _plot_interp_fill(self, df, plotParams, xlim=None, zlim=None):
        from matplotlib import colors
        col = self.emires_col
        pos = df.loc[df[col] > 0, col]
        vmin = plotParams.get('vmin'); vmax = plotParams.get('vmax')
        if vmin is None:
            vmin = float(np.nanmin(pos)) if len(pos) else 1.0
        if vmax is None:
            vmax = float(np.nanmax(df[col]))
        fig, ax = plt.subplots(layout='constrained')
        sc = ax.scatter(df['X'], df['Z'], c=df[col],
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                        cmap=plotParams.get('cmap', 'viridis'), s=8)
        cbar = plt.colorbar(sc, ax=ax, location='bottom', shrink=0.75,
                            aspect=25, extend='both')
        cbar.set_label(col)
        ax.set_title('Interpolated / filled mesh')
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Depth (m)')
        plt.show()

    def merge_into_mesh(self):
        if self.emi2mesh is None:
            return

        #take coordinates and corresponding EMI values to be mapped
        needed_cols = list(self.keep_cols) + [self.emires_col]
        missing = [c for c in needed_cols if c not in self.emi2mesh.columns]
        if missing:
            raise KeyError(f"Expected mapped columns missing: {missing}. "
                           f"Available: {list(self.emi2mesh.columns)}")
	
	#merge EMI values onto the ERI mesh
        merged = pd.merge(
            self.mesh,
            self.emi2mesh[needed_cols],
            on=list(self.keep_cols),
            how='left'
        )

        #overwrite all res0 values previously in the ERI mesh
        if self.resCol not in merged.columns:
            merged[self.resCol] = pd.NA
        merged[self.resCol] = merged[self.emires_col].combine_first(merged[self.resCol])

        merged = merged.drop(columns=[self.emires_col])
        if self.meshobj is not None:
            self.meshobj.mesh.df = merged
        else:
            self.meshobj = merged

    def run(self) -> Dict[str, Any]:
        """
        Execute the package and update k.mesh.df in place with default settings.
        """
        self.prep_mesh_utm()
        self.load_emi()
        self.map_to_mesh()
        self.fill_missing()
        self.merge_into_mesh()

        if self.meshobj is not None:
            return {
                'utm_pair1': self.utm_pair1,
                'utm_pair2': self.utm_pair2,
                'meshUTM': self.meshUTM,
                'emi_inv': self.emi_inv,
                'emi2mesh': self.emi2mesh,
                'dist': self.dist
            }
        else:
            return {
                'utm_pair1': self.utm_pair1,
                'utm_pair2': self.utm_pair2,
                'meshUTM': self.meshUTM,
                'emi_inv': self.emi_inv,
                'emi2mesh': self.emi2mesh,
                'dist': self.dist,
                'informedMesh': self.meshobj
            }


    def confidence(self, value_col, depth_bins=12, loocv_mean="arith",
                   x_col="X", group_col=None, doi_col=None, line_coord=None,
                   resid_space="ratio", plot=False, xlim=None, ax=None,
                   savefig=False, figpath=None, plotDepth=False):
        """Interpolation-confidence diagnostics for the EMI->mesh mapping.

        Must be called after map_to_mesh()/run() so self.mapper, self.meshUTM
        and self.emi_inv are populated. `value_col` is a column in the EMI data
        being mapped (self.emi_inv) -- a raw channel / linear-resistivity
        column, NOT a *_arith output.

        Section plots use the local along-line coordinate `x_col` (default 'X',
        carried from the ERT mesh). The EMI source points have no native local
        coordinate, so for the LOOCV panel each source point is assigned the
        `x_col` of its nearest mesh node, putting all three panels on one
        consistent axis. Falls back to UTM 'x' if `x_col` is absent from the
        mesh. Returns (and stores on self.confidence_results) a dict with keys:
        support, support_by_depth, dispersion, loocv_residuals, loocv_summary,
        fill_fraction.
        """
        if getattr(self, "mapper", None) is None:
            raise RuntimeError("Run map_to_mesh() (or run()) before confidence().")

        mapper, mesh, source = self.mapper, self.meshUTM, self.emi_inv
        if doi_col is not None and doi_col in source.columns:
            source = mapper.doi_filter(source, doi_col, depth_col="Z")
        if line_coord is not None:
            # Put BOTH mesh and source on a global along-line coordinate (e.g.
            # the fitted-line projection used for the final 2-D sections),
            # instead of the per-survey ResIPy mesh 'X' (which is a local,
            # laterally padded frame). The source keeps its OWN coordinate -- the
            # whole survey line -- rather than borrowing a single mesh's X, so
            # the residual panel spans the full line, not one mesh's footprint.
            mesh = mesh.copy();   mesh[x_col]   = np.asarray(line_coord(mesh), dtype=float)
            source = source.copy(); source[x_col] = np.asarray(line_coord(source), dtype=float)
            use_local = True
            keep = [x_col]
            source_x = source
        else:
            use_local = x_col in mesh.columns
            keep = [x_col] if use_local else None

            # give EMI source points a local along-line coordinate by borrowing the
            # nearest mesh node's x_col, so the residual panel shares the axis
            source_x = source
            if use_local:
                mtree = cKDTree(mesh[["x", "y"]].values)
                _, mi = mtree.query(source[["x", "y"]].values, k=1)
                source_x = source.copy()
                source_x[x_col] = mesh[x_col].values[mi]

        sup = mapper.support_field(mesh, source, keep_cols=keep, group_col=group_col)
        depth_summary = mapper.support_by_depth(sup, nbins=depth_bins)
        disp = mapper.dispersion_field(mesh, source, value_cols=[value_col],
                                       keep_cols=keep, group_col=group_col)
        resid, loocv = mapper.loocv_interpolation(source_x, value_col,
                                                  mean=loocv_mean, keep_cols=keep,
                                                  group_col=group_col)

        results = {
            "support": sup,
            "support_by_depth": depth_summary,
            "dispersion": disp,
            "loocv_residuals": resid,
            "loocv_summary": loocv,
            "fill_fraction": float(getattr(mapper, "fill_fraction", np.nan)),
        }
        self.confidence_results = results

        if plot:
            xc = x_col if use_local else "x"
            if xlim is None:
                if line_coord is not None:
                    allx = np.concatenate([np.asarray(mesh[x_col].values, float),
                                           np.asarray(source[x_col].values, float)])
                    xlim = (float(np.nanmin(allx)), float(np.nanmax(allx)))
                elif use_local:
                    xlim = (float(mesh[x_col].min()), float(mesh[x_col].max()))
                else:
                    xlim = None
            gcv = value_col + "_gcv"
            rlabel = "Predicted / Observed" if resid_space == "ratio" else r"residual $\log_{10}$"
            rtitle = "LOOCV Residuals (Pred/Obs)" if resid_space == "ratio" else "LOOCV residuals"    
            
            if figpath is not None:
                nnfigname = figpath + 'nnDistance.png'
                gcvfigname = figpath + 'gcv.png'
                loocvfigname = figpath + 'loocv.png'
            else:
                nnfigname = None
                gcvfigname = None
                loocvfigname = None
            
            if ax is not None:
                if len(ax) == 3:
                    mapper.plot_confidence_section(sup[sup["supported"]], 'nn_dist',
                                                   x_col=xc, xlim=xlim,
                                                   title="Spatial support (NN distance)", ax=ax[0], clabel=f'Nearest Neighbor Distance ({self.units})',
                                                   figname=nnfigname, plotDepth=plotDepth)
                    if gcv in disp.columns and np.isfinite(disp[gcv]).any():
                        mapper.plot_confidence_section(disp, gcv, x_col=xc, xlim=xlim,
                                                       title="Local dispersion (" + gcv + ")", clabel=f'{value_col} Geometric Coeff.\nof Variation',
                                                       ax=ax[1], figname=gcvfigname, plotDepth=plotDepth)
        
                    mapper.plot_confidence_section(resid, "resid_log10", x_col=xc,
                                                   xlim=xlim, center=0.0,
                                                   value_space=resid_space, clabel=rlabel,
                                                   title=rtitle, ax=ax[2], figname=loocvfigname,
                                                   plotDepth=plotDepth)
            else:
                mapper.plot_confidence_section(sup[sup["supported"]], "nn_dist",
                                               x_col=xc, xlim=xlim, clabel=f'Nearest Neighbor Distance ({self.units})',
                                               title="Spatial support (NN distance)", figname=nnfigname,
                                               plotDepth=plotDepth)
                if gcv in disp.columns and np.isfinite(disp[gcv]).any():
                    mapper.plot_confidence_section(disp, gcv, x_col=xc, xlim=xlim, clabel=f'{value_col} Geometric Coeff.\nof Variation',
                                                   title="Local dispersion (" + gcv + ")", figname=gcvfigname,
                                                   plotDepth=plotDepth)

        print("EM2ER interpolation-confidence summary")
        print("  fill fraction: {:.1%}".format(results["fill_fraction"]))
        print("  LOOCV log10 RMSE/MAE/bias: {:.3f} / {:.3f} / {:+.3f}".format(
            loocv["RMSE_log"], loocv["MAE_log"], loocv["bias_log"]))
        print("  linear factors  RMSE/MAE/bias: x{:.2f} / x{:.2f} / x{:.2f}".format(
            10 ** loocv["RMSE_log"], 10 ** loocv["MAE_log"], 10 ** loocv["bias_log"]))
        return results


__all__ = ["local_grid_to_utm_along_line", "direct_local_grid_to_utm_along_line", "em2er_map"]