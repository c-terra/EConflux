"""em2er.py

Self-contained mapper to seed ResIPy k.mesh.df['res0'] from EMI data.
Includes the helper `local_grid_to_utm_along_line` so no external geometry
utilities are required.

Exports:
- local_grid_to_utm_along_line
- em2er_map
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

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
        
    return local_grid_df

class em2er_map:
    """
    Map EMI inversion values onto a ResIPy mesh and use them to set k.mesh.df['res0'] for ERI inversion.

    Parameters
    ----------
    k : resipy.Project
        Your ResIPy inversion project mesh and electrodes must already be defined
    elecUTM : pd.DataFrame
        Copy of electrodes from inversion project (k.elec) with UTM 'x', 'y', and 'z'
    emicsv : str
        Path to the Rutgers-style EMI .csv (X, Y, Depth, Linear Resistivity columns present)
    linearOrPoly : str
        Define whether to use a linear or polynomial interpolation to convert UTM to local coordintes
    alphaomega : Tuple[int, int]
        Positional indexes into elecUTM to define the line direction (defaults to (0, 12) but should be the start and end of your survey line (min elec num, max elec num)).
    polydeg : int
        Polynomial degree to use for electrode interpolation (if linearOrInterp != 'linear')
    mapvars : dict
        Passed to testcalib.DataMapper(num_neighbors=2, max_distance_xy=5, max_distance_z=0.3) for ease of updating this function down the line.
    fillstrat : {"mean","value","none"}
        How to fill NaNs after mapping. "mean" fills each column with its column mean.
        "value" uses `res0fill`. "none" leaves NaNs.
    res0fill : Optional[float] #fill with a particular value
        Value for fillstrat="value".
    emires_col : str
        Column from the mapped dataframe to use for res0 merge
        (default: "_Linear Resistivity_arithmetic_mean", other options available in the DataMapper function).
    keep_cols : Optional[Tuple[str, str, str]]
        Coordinate column names (default: ('X','Y','Z')), optional.
    """

    def __init__(
        self,
        k,
        elecUTM: pd.DataFrame,
        emicsv: str,
        linearOrPoly: str,
        alphaomega: Tuple[int, int] = (0, 12),
        polydeg: int = 3,
        mapvars: Optional[Dict[str, Any]] = None,
        fillstrat: str = "mean",
        res0fill: Optional[float] = None,
        emires_col: str = "_Linear Resistivity_arithmetic_mean",
        keep_cols: Tuple[str, str, str] = ("X", "Y", "Z")
    ):
        self.k = k
        self.elecUTM = elecUTM
        self.emicsv = emicsv
        self.linearOrPoly = linearOrPoly
        self.alphaomega = alphaomega
        self.polydeg = polydeg
        self.mapvars = mapvars or dict(num_neighbors=5, max_distance_xy=5, max_distance_z=0.3)
        self.fillstrat = fillstrat
        self.res0fill = res0fill
        self.emires_col = emires_col
        self.keep_cols = keep_cols

        #populated during run()
        self.utm_pair1: Optional[Tuple[float, float]] = None
        self.utm_pair2: Optional[Tuple[float, float]] = None
        self.meshUTM: Optional[pd.DataFrame] = None
        self.emi_inv: Optional[pd.DataFrame] = None
        self.emi2mesh: Optional[pd.DataFrame] = None
        self.dist = None

    def _prep_mesh_utm(self): #convert electrode positions to UTM from just electrode numbers
        if self.linearOrPoly == 'linear':
            i1, i2 = self.alphaomega
            self.utm_pair1 = (self.elecUTM['x'].iloc[i1], self.elecUTM['y'].iloc[i1])
            self.utm_pair2 = (self.elecUTM['x'].iloc[i2], self.elecUTM['y'].iloc[i2])
    
            #convert local grid points to UTM coordinates along the line
            xyData_UTM = local_grid_to_utm_along_line(self.k.mesh.df.copy(), self.utm_pair1, self.utm_pair2)
            
            meshUTM = pd.concat([self.k.mesh.df.copy(), xyData_UTM], axis=1)
            meshUTM = meshUTM.rename(columns={'easting': 'x', 'northing': 'y'})

        else:
            meshUTM = direct_local_grid_to_utm_along_line(self.k.mesh.df, self.elecUTM, self.polydeg, checkLine=True)
        #standardize coordinate naming and de-duplicate
        meshUTM = meshUTM.loc[:, ~meshUTM.columns.duplicated()]

        #remove unnecessary columns
        meshUTM = meshUTM[['X', 'Y', 'Z', 'x', 'y']]
        self.meshUTM = meshUTM

    def _load_emi(self): #load EMI csv, Rutgers-style
        emi_inv = pd.read_csv(self.emicsv)
        #match column names expected by mapping function
        emi_inv = emi_inv.rename(columns={'X': 'x', 'Y': 'y', 'Depth': 'Z'})
        self.emi_inv = emi_inv

    def _map_to_mesh(self):
        #import testcalib function, needs to be available outside this script
        from er2em import DataMapper

	#map EMI data to ERI mesh using select parameters
        mapper = DataMapper(**self.mapvars) 
        emi2mesh, dist = mapper.mapXYZ(self.meshUTM, [self.emi_inv], [''])
        self.emi2mesh = emi2mesh.copy()
        self.dist = dist

    def _fill_missing(self):
        if self.emi2mesh is None:
            return
        df = self.emi2mesh.copy()

        #fill res0 data columns with arithmetic mean values if not mapped
        cols_to_exclude = set(self.keep_cols)
        data_cols = [c for c in df.columns if c not in cols_to_exclude]

        if self.fillstrat == "mean":
            for col in data_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
        elif self.fillstrat == "value":
            df[data_cols] = df[data_cols].fillna(self.res0fill)
        elif self.fillstrat == "none":
            pass
        else:
            raise ValueError(f"Unknown fillstrat: {self.fillstrat}")

        self.emi2mesh = df

    def _merge_into_mesh(self):
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
            self.k.mesh.df,
            self.emi2mesh[needed_cols],
            on=list(self.keep_cols),
            how='left'
        )

        #overwrite all res0 values previously in the ERI mesh
        if 'res0' not in merged.columns:
            merged['res0'] = pd.NA
        merged['res0'] = merged[self.emires_col].combine_first(merged['res0'])

        merged = merged.drop(columns=[self.emires_col])
        self.k.mesh.df = merged

    def run(self) -> Dict[str, Any]:
        """
        Execute the package and update k.mesh.df in place.
        """
        self._prep_mesh_utm()
        self._load_emi()
        self._map_to_mesh()
        self._fill_missing()
        self._merge_into_mesh()

        return {
            'utm_pair1': self.utm_pair1,
            'utm_pair2': self.utm_pair2,
            'meshUTM': self.meshUTM,
            'emi_inv': self.emi_inv,
            'emi2mesh': self.emi2mesh,
            'dist': self.dist
        }

__all__ = ["local_grid_to_utm_along_line", "em2er_map"]
