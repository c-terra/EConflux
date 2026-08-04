import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hydrostats as hs
import hydroeval as he
from scipy.stats import spearmanr, gmean
from sklearn.metrics import r2_score
import matplotlib.ticker
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, Colormap, Normalize
import pysymlog as psl
psl.register_mpl()

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'

def is_divergent(cmap) -> bool:
    """Returns True if the colormap is divergent, otherwise False."""
    # List of official Matplotlib divergent colormap base names
    diverging_names = {
        'piyg', 'prgn', 'brbg', 'puor', 'rdgy', 
        'rdylbu', 'rdylgn', 'rdbu', 'coolwarm', 
        'bwr', 'seismic', 'twilight', 'twilight_shifted'
    }

    # Extract string name if a Colormap object is passed
    if isinstance(cmap, Colormap):
        cmap_name = cmap.name.lower()
    elif isinstance(cmap, str):
        cmap_name = cmap.lower()
    else:
        raise TypeError("Input must be a string or a matplotlib Colormap object.")
    
    # Strip the '_r' suffix if the colormap is reversed
    if cmap_name.endswith('_r'):
        cmap_name = cmap_name[:-2]
                
    is_div = (cmap_name in diverging_names) | ('div' in cmap_name)
        
    return is_div

class EConfluxStats:
    '''
    EConflux Statistics and Visualization class for
    comparing ER and EM datasets (raw and informed),
    generating diagnostic plots, and computing performance
    metrics.

    Inputs are assumed to be positive-valued and may span multiple orders
    of magnitude.
    '''

    def __init__(self, filepath, informingCol, rawCol, informedCol, drop_flag_col=None, 
                 informingMethod=None, informedMethod=None, units='mS/m', ticks=None, **fontkws):
        
        '''
        Load ER and EM datasets, apply basic cleaning, and configure plotting defaults.

        Notes for users:
        * NaNs are always dropped from the three core columns (informingCol, rawCol, informedCol).
        * Optional flag columns can be used to exclude rows.
        * Values are filtered to be positive (i.e., > 0) to support log-based analyses.
        * Units are used for labeling only; no unit conversion is performed.
        
        NB: fontkws: A dictionary to set font properties for all figures created. Possible options include:
            fontfamily, fontweight, titlesize, titleweight, axislabelsize, axislabelweight, axisticksize, cbartitlesize, cbartitleweight, cbarticklabelsize
            
            See https://matplotlib.org/stable/api/font_manager_api.html#matplotlib.font_manager.FontProperties for font property options
        '''
        
        # Loading dataset
        self.data = pd.read_csv(filepath)

        # Dropping flagged rows
        self.data.sort_values(by=["Z", "x", "y"], inplace=True, ascending=False)
        if type(drop_flag_col) is list:
            self.data = self.data.dropna(subset=[informingCol, rawCol, informedCol], ignore_index=True)

            for col in drop_flag_col:
                if col in self.data.columns:
                    self.data = self.data.drop(self.data.index[self.data[col] == 1])
        elif type(drop_flag_col) == str:
            self.data = self.data.dropna(subset=[informingCol, rawCol, informedCol], ignore_index=True)

            if drop_flag_col in self.data.columns:
                self.data = self.data.drop(self.data.index[self.data[drop_flag_col] == 1])

        else:
            self.data = self.data.dropna(subset=[informingCol, rawCol, informedCol], ignore_index=True)

        self.data = self.data.loc[(self.data[informingCol] > 0) & (self.data[rawCol] > 0) & (self.data[informedCol] > 0)]
        self.data.reset_index(drop=True, inplace=True)

        # Defining core variables
        self.X = self.data[informingCol]  # Inversion result from reference method
        self.y = self.data[rawCol]      # Inversion result from raw uninformed data
        self.Y = self.data[informedCol]   # Inversion result of informed data (incorporating the reference method)

        # Defining log-transformed variables
        self.Xlog = np.log10(self.X)
        self.ylog = np.log10(self.y)
        self.Ylog = np.log10(self.Y)

        self.informingMethod = informingMethod
        self.informedMethod = informedMethod
        # remember source column names (used by ground_truth_check, etc.)
        self.informingCol = informingCol
        self.rawCol = rawCol
        self.informedCol = informedCol

        self.units = units
        
        self.ticks = ticks
        self.ticktypes = list(type(item) for item in ticks)
        
        # Setting default font properties
        fontkwDict = {'fontfamily': 'sans-serif',
                      'fontweight': 'bold',
                      'titlefontsize': 14,
                      'titlefontweight': 'bold',
                      'axislabelsize': 12,
                      'axislabelweight': 'bold',
                      'axisticksize': 9,
                      'cbartitlesize': 10,
                      'cbartitleweight': 'bold',
                      'cbarticklabelsize': 8}
        
        for key in list(fontkwDict.keys()):
            if key in list(fontkws.keys()):
                fontkwDict[key] = fontkws[key]
        
        plt.rcParams['font.family'] = fontkwDict['fontfamily']
        plt.rcParams['font.weight'] = fontkwDict['fontweight']
        plt.rcParams['xtick.labelsize'] = fontkwDict['axisticksize']
        plt.rcParams['ytick.labelsize'] = fontkwDict['axisticksize']

        # title font
        self.titleFontKws = {
            'fontsize': fontkwDict['titlefontsize'],
            'fontweight': fontkwDict['titlefontweight'],
            }
        # axis label font
        self.labelFontKws = {
            'fontsize': fontkwDict['axislabelsize'],
            'fontweight': fontkwDict['axislabelweight'],
            }
        
        # axis tick label font
        self.axisTickFontKws = {
            'labelsize': fontkwDict['axisticksize']
            }
        
        # colorbar title font
        self.cbarTitleKws = {
            'fontsize': fontkwDict['cbartitlesize'],
            'fontweight': fontkwDict['cbartitleweight']}
        # colorbar tick font
        self.cbarTickKws = {
            'labelsize': fontkwDict['cbarticklabelsize'],
            }


    # ==============================================================
    # ---------------------- METRICS METHODS -----------------------
    # ==============================================================

    @staticmethod
    def KGEnp(sim, obs, decomp=False):
        '''
        Non-parametric Kling–Gupta Efficiency, KGEnp (Pool et al., 2018).

        Evaluates model performance using:
        * rank structure (Spearman correlation),
        * relative variability (distribution shape),
        * mean bias.

        Designed to be robust to outliers and non-normal data,
        which are common in environmental time series.
        
        sim: numpy.array
            Simulated values
        obs: numpy.array
            Observed values
        decomp: Optional[bool]
            If True, will return alpha, beta, and r components of non-parametric KGE.
        '''
        
        sim, obs = np.asarray(sim, float), np.asarray(obs, float)
        mask = np.isfinite(sim) & np.isfinite(obs)
        sim, obs = sim[mask], obs[mask]
        if sim.size == 0:
            raise ValueError("No valid data")

        mean_sim, mean_obs = np.mean(sim), np.mean(obs)
        fdc_sim = np.sort(sim / (mean_sim * len(sim)))
        fdc_obs = np.sort(obs / (mean_obs * len(obs)))

        alpha = 1 - 0.5 * np.sum(np.abs(fdc_sim - fdc_obs))
        beta = mean_sim / mean_obs
        r = spearmanr(sim, obs).correlation
        kge = 1 - np.sqrt((alpha - 1) ** 2 + (beta - 1) ** 2 + (r - 1) ** 2)

        if decomp:
            return alpha, beta, r, kge
        else:
            return kge
        
    def metrics(self, logOrlin='log', decompKGE=False):
        '''
        Return dataframe of evaluation metrics for raw and informed datasets
        
        logOrlin: Optional[str]
            Whether to use the log-transformed or linear values for statistical metrics. Default is 'log'.
        decompKGE: Optional[bool]
            If True, will include separate components of non-parametric KGE in metrics list.
        '''
        
        if logOrlin == 'log':
            infSim, rawSim, obs = self.Ylog, self.ylog, self.Xlog
            self.log = True
        else:
            infSim, rawSim, obs = self.Y, self.y, self.X
            self.log = False
            
        if decompKGE:
            alphaInf, betaInf, rInf, kgeInf = self.KGEnp(infSim, obs, decomp=True)
            alphaRaw, betaRaw, rRaw, kgeRaw = self.KGEnp(rawSim, obs, decomp=True)
            metrics = {
                "KGE_np": [kgeInf, kgeRaw], # non-parametric Kling–Gupta Efficiency (Pool et al., 2018)
                "KGE_alpha": [alphaInf, alphaRaw], # alpha parameter for non-parametric KGE. Represents bias between simulated and observed standard deviation
                "KGE_beta": [betaInf, betaRaw], # beta parameter for non-parametric KGE. Represents bias between simulated and observed means
                "KGE_r": [rInf, rRaw], # r parameter for non-parametric KGE. Represents Spearman's correlation coefficient. Should equal Spearman r calculated later
                "KGE_2009": [hs.kge_2009(infSim, obs), hs.kge_2009(rawSim, obs)],  # Kling–Gupta Efficiency (Gupta et al., 2009)
                "KGE_2012": [hs.kge_2012(infSim, obs), hs.kge_2012(rawSim, obs)], # Kling–Gupta Efficiency (Kling et al., 2012)
                "NSE": [hs.nse(infSim, obs), hs.nse(rawSim, obs)], # Nash–Sutcliffe Efficiency
                "R²": [hs.r_squared(infSim, obs), hs.r_squared(rawSim, obs)], # Coefficient of determination
                "Pearson r": [hs.pearson_r(infSim, obs), hs.pearson_r(rawSim, obs)], # Pearson correlation coefficient
                "Spearman r": [hs.spearman_r(infSim, obs), hs.spearman_r(rawSim, obs)], # Spearman's correlation coefficient
                "ME": [hs.me(infSim, obs), hs.me(rawSim, obs)], # Mean error
                "MAE": [hs.mae(infSim, obs), hs.mae(rawSim, obs)], # Mean absolute error
                "RMSE": [hs.rmse(infSim, obs), hs.rmse(rawSim, obs)], # Root mean square error
                "NRMSE_range": [hs.nrmse_range(infSim, obs), hs.nrmse_range(rawSim, obs)]# Normalized root mean square error
            }
        else:
            metrics = {
                "KGE_np": [self.KGEnp(infSim, obs), self.KGEnp(rawSim, obs)],   
                "KGE_2009": [hs.kge_2009(infSim, obs), hs.kge_2009(rawSim, obs)],  
                "KGE_2012": [hs.kge_2012(infSim, obs), hs.kge_2012(rawSim, obs)],   
                "NSE": [hs.nse(infSim, obs), hs.nse(rawSim, obs)],   
                "R²": [hs.r_squared(infSim, obs), hs.r_squared(rawSim, obs)],   
                "Pearson r": [hs.pearson_r(infSim, obs), hs.pearson_r(rawSim, obs)],   
                "Spearman r": [hs.spearman_r(infSim, obs), hs.spearman_r(rawSim, obs)],   
                "ME": [hs.me(infSim, obs), hs.me(rawSim, obs)],   
                "MAE": [hs.mae(infSim, obs), hs.mae(rawSim, obs)],   
                "RMSE": [hs.rmse(infSim, obs), hs.rmse(rawSim, obs)],   
                "NRMSE_range": [hs.nrmse_range(infSim, obs), hs.nrmse_range(rawSim, obs)]   
            }
        if (self.informingMethod is not None) & (self.informedMethod is not None):
            return pd.DataFrame(metrics, index=[f"Informed ({self.informedMethod} vs {self.informingMethod})",
                                                f"Raw ({self.informedMethod} vs {self.informingMethod})"]).T.round(2)
        else:
            return pd.DataFrame(metrics, index=["Informed (Y. vs X)", "Raw (y vs X)"]).T.round(2)

    
    # # ---------- Power-law fitting ----------
    @staticmethod
    def _powerlaw_fit(x, y):
        """Fit a power-law y = a * x^b using log–log regression."""
        
        X_log = np.log10(x)
        Y_log = np.log10(y)

        coeffs = np.polyfit(X_log, Y_log, 1)
        p = np.poly1d(coeffs)

        a = 10**coeffs[1]
        b = coeffs[0]
        r2 = r2_score(Y_log.values, X_log.values)

        return a, b, r2, p

    # ==============================================================
    # -------------------- VISUALIZATION METHODS -------------------
    # ==============================================================
    
    def scatter_plots(self, log=True, grayscale=False, figname=None):
        """
        Scatter plots comparing ER vs EM (raw and informed).
        
        log: Optional[bool]
            If True, will plot the log transformed values
        grayscale: Optional[bool]
            If True, will plot in grayscale
        figname: Optional[str]
            Name to use for saving the plot to a .png
        """
        # Fit power-law models (raw + informed)
        if log:
            a_nc, b_nc, r2_nc, _ = self._powerlaw_fit(self.X, self.y)
            a_cal, b_cal, r2_cal, _ = self._powerlaw_fit(self.X, self.Y)   
        else:
            coeffs_nc = np.polyfit(self.X, self.y, 1)
            a_nc, b_nc, r2_nc = coeffs_nc[1], coeffs_nc[0], r2_score(self.y.values, self.X.values)
            
            coeffs_c = np.polyfit(self.X, self.Y, 1)
            a_cal, b_cal, r2_cal = coeffs_c[1], coeffs_c[0], r2_score(self.Y.values, self.X.values)
    
        fig, ax = plt.subplots(constrained_layout=True)
    
        if (self.informingMethod is not None) & (self.informedMethod is not None):
            if grayscale:
                ax.scatter(self.X, self.y, marker='o', c='gray', s=20, edgecolor='k', label=f"{self.informedMethod}" + r"$_{raw}$", alpha=1)
                ax.scatter(self.X, self.Y, marker='o', c='white', s=20, edgecolor='k', label=f"{self.informedMethod}" + r"$_{informed}$", alpha=1)
            else:
                ax.scatter(self.X, self.y, marker='o', c='dodgerblue', s=8, label=f"{self.informedMethod}" + r"$_{raw}$", alpha=0.5)
                ax.scatter(self.X, self.Y, marker='o', c='goldenrod', s=8, label=f"{self.informedMethod}" + r"$_{informed}$", alpha=0.5)

        else:
            ax.scatter(self.X, self.y, c='dodgerblue', label="EC$_{raw}$", alpha=0.6)
            ax.scatter(self.X, self.Y, c='goldenrod', label="EC$_{informed}$", alpha=0.6)
    
        if log:
            # Power-law fits (log10-transformed for consistency)
            x_fit = np.logspace(np.log10(self.X.min()), np.log10(self.X.max()), 200)
            y_fit_nc = a_nc * x_fit ** b_nc
            y_fit_cal = a_cal * x_fit ** b_cal
        else:
            x_fit = np.linspace(self.X.min(), self.X.max(), 200)
            y_fit_nc = a_nc + (x_fit * b_nc)
            y_fit_cal = a_cal + (x_fit * b_cal)
    
        if grayscale:
            ax.plot(x_fit, y_fit_nc, c="dimgray", lw=4, alpha=0.8, path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
                       label=f"Raw Fit: a={a_nc:.2f}, b={b_nc:.2f}, R$^2$={r2_nc:.3f}")
            ax.plot(x_fit, y_fit_cal, c="gainsboro", lw=4, alpha=1, path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
                       label=f"Informed Fit: a={a_cal:.2f}, b={b_cal:.2f}, R$^2$={r2_cal:.3f}")
        else:
            ax.plot(x_fit, y_fit_nc, c="darkblue", lw=4, alpha=0.8,
                       label=f"Raw Fit: a={a_nc:.2f}, b={b_nc:.2f}, R$^2$={r2_nc:.3f}")
            ax.plot(x_fit, y_fit_cal, c="maroon", lw=4, alpha=0.8,
                       label=f"Informed Fit: a={a_cal:.2f}, b={b_cal:.2f}, R$^2$={r2_cal:.3f}")


        ax.axline((0, 0), slope=1, color="k", linestyle="--", lw=3)

        ax.set_title("Raw–Informed Scatter Comparison with Power-Law Fits", **self.titleFontKws)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        if (self.ticks is not None) & (self.informingMethod is not None) & (self.informedMethod is not None):
            ax.set_xlabel(f"{self.informingMethod} EC ({self.units})", **self.labelFontKws) 
            ax.set_ylabel(f"{self.informedMethod} EC ({self.units})", **self.labelFontKws)
            ax.set_xlim(self.ticks[0], self.ticks[-1])
            ax.set_ylim(self.ticks[0], self.ticks[-1])
            ax.set_xticks(self.ticks)
            ax.set_yticks(self.ticks)
        elif (self.ticks is None) & (self.informingMethod is not None) & (self.informedMethod is not None):
            ax.set_xlabel(f"{self.informingMethod} EC ({self.units})", **self.labelFontKws) 
            ax.set_ylabel(f"{self.informedMethod} EC ({self.units})", **self.labelFontKws)
        else:
            ax.set_xlabel(f"Dataset 1 EC ({self.units})", **self.labelFontKws) 
            ax.set_xlabel(f"Dataset 2 EC ({self.units})", **self.labelFontKws) 
            
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(float(x))))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, pos: str(float(y))))
        
        if float not in self.ticktypes:
            formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)


        leg = ax.legend(fontsize=self.axisTickFontKws.get('fontsize', 8), loc='best', framealpha=0.75)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
        
        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.savefig('scatter.png', dpi=300)
        plt.show()
    
        return {
            "Raw Fit": {"a": a_nc, "b": b_nc, "R$^2$": r2_nc},
            "Informed Fit": {"a": a_cal, "b": b_cal, "R²": r2_cal}
        }

    def kde_histograms(self, nbins=100, ylims=None, figname=None):
       '''
       Compare EC distributions using log-spaced histograms.

       Plots show how raw and informed datasets differ from the reference with 
       respect to distribution shape, with annotations of summary error metrics.
       
       nbins: Optional[int]
           Number of bins to use for histogram. Default is 100.
       figname: Optional[str]
           Name to use for saving the plot to a .png
       ''' 
       
       fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

       # Raw
       if (self.informingMethod is not None) & (self.informedMethod is not None):
           ax[0].hist(self.X, bins=np.geomspace(self.X.min(), self.X.max(), nbins), density=True, alpha=0.7, color='dodgerblue', label=self.informingMethod)
           ax[0].hist(self.y, bins=np.geomspace(self.y.min(), self.y.max(), nbins), density=True, alpha=0.7, color='peru', label=self.informedMethod)
       else:
           ax[0].hist(self.X, bins=np.geomspace(self.X.min(), self.X.max(), nbins), density=True, alpha=0.7, color='dodgerblue', label='Dataset 1')
           ax[0].hist(self.y, bins=np.geomspace(self.y.min(), self.y.max(), nbins), density=True, alpha=0.7, color='peru', label='Dataset 2')
       ax[0].legend()
       ax[0].set_xlabel("Electrical Conductivity (mS/m)", **self.labelFontKws)
       ax[0].set_xscale('log')
       if self.ticks is not None:
           ax[0].set_xlim(self.ticks[0], self.ticks[1])
           ax[0].set_xticks(self.ticks)
           
       ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
       ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(float(x))))
       
       if self.log:
           ax[0].text(0.05, 0.95,
                      f"RMSE: {hs.rmse(self.ylog, self.Xlog):.2f}\n"
                      f"PBIAS: {he.evaluator(he.pbias, self.ylog, self.Xlog)[0]:.2f}\n"
                      f"KGE_np: {self.KGEnp(self.ylog, self.Xlog):.2f}",
                      transform=ax[0].transAxes, va="top", fontsize=9)
       else:
           ax[0].text(0.05, 0.95,
                      f"RMSE: {hs.rmse(self.y, self.X):.2f}\n"
                      f"PBIAS: {he.evaluator(he.pbias, self.y, self.X)[0]:.2f}\n"
                      f"KGE_np: {self.KGEnp(self.y, self.X):.2f}",
                      transform=ax[0].transAxes, va="top", fontsize=9)
        
       ax[0].set_title(f"KDE Histogram (Raw {self.informedMethod} vs. {self.informingMethod}) ({nbins} bins)", **self.titleFontKws)
       ax[0].set_ylabel('Probability Density', **self.labelFontKws)
    
       # Informed
       if (self.informingMethod is not None) & (self.informedMethod is not None):
           ax[1].hist(self.X, bins=np.geomspace(self.X.min(), self.X.max(), nbins), density=True, alpha=0.7, color='dodgerblue', label=self.informingMethod)
           ax[1].hist(self.Y, bins=np.geomspace(self.Y.min(), self.Y.max(), nbins), density=True, alpha=0.7, color='peru', label=self.informedMethod)
       else:
           ax[1].hist(self.X, bins=np.geomspace(self.X.min(), self.X.max(), nbins), density=True, alpha=0.7, color='dodgerblue', label='Dataset 1')
           ax[1].hist(self.Y, bins=np.geomspace(self.Y.min(), self.Y.max(), nbins), density=True, alpha=0.7, color='peru', label='Dataset 2')
       ax[1].legend()
       ax[1].set_xlabel(f"Electrical Conductivity ({self.units})", **self.labelFontKws)
       ax[1].set_xscale('log')
       if self.ticks is not None:
           ax[1].set_xlim(self.ticks[0], self.ticks[1])
           ax[1].set_xticks(self.ticks)
           
       ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
       ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(float(x))))
       
       if self.log:
           ax[1].text(0.05, 0.95,
                      f"RMSE: {hs.rmse(self.Ylog, self.Xlog):.2f}\n"
                      f"PBIAS: {he.evaluator(he.pbias, self.Ylog, self.Xlog)[0]:.2f}\n"
                      f"KGE_np: {self.KGEnp(self.Ylog, self.Xlog):.2f}",
                      transform=ax[1].transAxes, va="top", fontsize=9)
       else:
           ax[1].text(0.05, 0.95,
                      f"RMSE: {hs.rmse(self.Y, self.X):.2f}\n"
                      f"PBIAS: {he.evaluator(he.pbias, self.Y, self.X)[0]:.2f}\n"
                      f"KGE_np: {self.KGEnp(self.Y, self.X):.2f}",
                      transform=ax[1].transAxes, va="top", fontsize=9)
        
       ax[1].set_title(f"KDE Histogram (Informed {self.informedMethod} vs. {self.informingMethod}) ({nbins} bins)", **self.titleFontKws)
    
       if ylims is not None:
           ax[0].set_ylim(ylims[0], ylims[1])
    
       plt.tight_layout()
       if figname is not None:
           plt.savefig(figname, dpi=300)
       else:
           plt.savefig('kde hist.png', dpi=300)
       plt.show()
        
    def qq_plot(self, sim, obs, xlabel='Dataset 1', ylabel='Dataset 2', title="QQ Plot", figname=None):
        '''
        Quantile-quantile plot (Q-Q) for comparing distributions.

        This diagnostic tool compares distributional similarity rather than pointwise
        agreement; deviations from the 1:1 line indicate skewness, tail differences, or outliers.
        
        sim: numpy.array
            Simulated values
        obs: numpy.array
            Observed values
        xlabel: Optional[str]
            Label for x-axis of Q-Q plot
        ylabel: Optional[str]
            Label for y-axis of Q-Q plot
        figname: Optional[str]
            Name to use for saving the plot to a .png
        '''
        d1, d2 = np.sort(sim[~np.isnan(sim)]), np.sort(obs[~np.isnan(obs)])
        q = np.linspace(0, 1, min(len(d1), len(d2)))
        q1, q2 = np.quantile(d1, q), np.quantile(d2, q)

        plt.figure(figsize=(7, 5))
        plt.plot(q1, q2, "o", alpha=0.5)
        plt.plot([q1[0], q1[-1]], [q1[0], q1[-1]], "r--")
        if (self.informingMethod is not None) & (self.informedMethod is not None):
            xlabel = self.informingMethod
            ylabel = self.informedMethod
            
        plt.xlabel(xlabel, **self.labelFontKws)
        plt.ylabel(ylabel, **self.labelFontKws)
        plt.title(title, **self.titleFontKws)
        plt.tight_layout()
        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.savefig('qq.png', dpi=300)
        plt.show()

    def bland_altman(self, sim, obs, 
                     nonparam=False, title=None, ax=None, marker='+', markersize=None,
                     ymin=None, ymax=None, xmin=None, xmax=None, savefig=False, figname=None, xticks=None, logList=None,
                     legendParams={'plot': True, 'loc': 'best'},
                     linekwargs={'color': 'orangered', 'linestyle': ['-', '--'], 'label': ['Mean Difference', 'Upper Agreement Limit', 'Lower Agreement Limit']}, **kwargs):
        '''
        Bland–Altman plot.
        Plots the difference (sim - obs) versus their mean to assess systematic
        bias and limits of agreement across the measurement range.
        Returns the plot and shows summary statistics.
            
        sim: numpy.array
            Simulated values
        obs: numpy.array
            Observed values
        nonparam: Optional[bool]
            If True, will plot the non-parametric Bland-Altman plots using the median of the differences and the 2.5th and 97.5th quantiles to determine bounds (see Chen and Kao 2021, or Gerke 2020)
        title: Optional[str]
            Title for Bland-Altman plot
        ax: Optional[matplotlib.Axes]
            Matplotlib Axes object to use for plotting
        marker: Optional[str]
            Marker type to use ofr Bland-Altman points. Default is '+'. See https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers for more options
        markersize: Optional[float]
            Size (in points) for Bland-Altman markers
        ymin: Optional[float]
            Minimum value to use for the y-axis (difference between sim and obs values)
        ymax: Optional[float]
            Maximum value to use for the y-axis (difference between sim and obs values)
        xmin: Optional[float]
            Minimum value to use for the x-axis (Mean of sim and obs values)
        xmax: Optional[float]
            Maximum value to use for the x-axis (Mean of sim and obs values)     
        savefig: Optional[bool]
            If True, will save Bland-Altman plot to current directory or path specified in figname
        figname: Optional[str]
            Name (or path and name) to use for saving the plot to a .png
        xticks: Optional[list]
            List of numbers to set for x-axis ticks
        logList: A list where the first entry determines whether to use a log-scale for the x-axis (0 or False for linear, 1 or True for log)
                and the second entry determines whether to use a log-scale for the y-axis. (e.g. [1, 0] means that the x-axis is a log-scale and the y-axis is linear)
        legendParams: Optional[dict]
            Dictionary to define whether to plot the legend ('plot') and what location to pass to ax.legend(loc='') ('loc')
        linekwargs: Optional[dict]
            Dictionary to define colors, linestyles, and labels for each line plotted on Bland-Altman plot
        '''
        diff = sim - obs
        mean = (sim + obs) / 2   # Mean difference represents bias

        sdev_diff = np.std(diff)   # defining the standard-deviation of the difference

        mainlinestyle = linekwargs['linestyle'][0]
        loalinestyle = linekwargs['linestyle'][1]
        del linekwargs['linestyle']
        midlinelabel = linekwargs['label'][0]
        upperlinelabel = linekwargs['label'][1]
        lowerlinelabel = linekwargs['label'][2]
        del linekwargs['label']

        if nonparam:
            mid = np.median(diff)
            # 97.5th and 2.5th quantiles define the limits of agreement
            upper = np.quantile(diff, 0.975)
            lower = np.quantile(diff, 0.025)
        else:
            sdev_diff = np.std(diff)
            mid = np.mean(diff)
            # ±1.96σ defines the 95% limits of agreement
            upper = mid + (1.96 * sdev_diff)  # the upper confidence limit of agreement
            lower = mid - (1.96 * sdev_diff)  # the lower confidence limit of agreement
        
        if logList is not None:
            if type(logList[0]) != bool:
                xLog = bool(logList[0])
            else:
                xLog = logList[0]
            if type(logList[1]) != bool:
                yLog = bool(logList[1])
            else:
                yLog = logList[1]
        else:
            xLog = None
            yLog = None

        # ---- Plot ----
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(mean, diff, marker=marker, s=markersize, **kwargs)

        ax.axhline(mid, linestyle=mainlinestyle, label=midlinelabel, **linekwargs)
        ax.axhline(upper, linestyle=loalinestyle, label=upperlinelabel, **linekwargs)
        ax.axhline(lower, linestyle=loalinestyle, label=lowerlinelabel, **linekwargs)
        if legendParams['plot']:
            leg = ax.legend(loc=legendParams['loc'], fontsize=self.axisTickFontKws.get('labelsize', 6), framealpha=0.75)
            for lh in leg.legend_handles:
                lh.set_alpha(1)

        if ax.get_xlabel() == '':
            ax.set_xlabel("Mean EC " + f"({self.units})", **self.labelFontKws)
        if ax.get_ylabel() == '':
            ax.set_ylabel("EC Difference " + f"({self.informedMethod} − {self.informingMethod}) ({self.units})", **self.labelFontKws)
        if xLog:
            ax.set_xscale('log')
        if yLog:
            ax.set_yscale('symmetriclog', shift=1)   # Use 'symmetriclog' (from pysymlog library) for the y-axis so a log-distribution for negative values can be shown
        ax.set_ylim(ymin, ymax)
        if (xmin is not None) & (xmax is not None):
            ax.set_xlim(xmin, xmax)
        
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(float(x))))
        ax.set_title(title, **self.titleFontKws)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, pos: str(float(y))))

        if float not in self.ticktypes:
            formatter = matplotlib.ticker.StrMethodFormatter("{x:1g}")
            plt.gca().xaxis.set_major_formatter(formatter)
        if float not in [type(t) for t in ax.get_yticks()]:
            formatter = matplotlib.ticker.StrMethodFormatter("{x:1g}")
            plt.gca().yaxis.set_major_formatter(formatter)

        if savefig:
            if figname is not None:
                plt.savefig(figname, dpi=300)
            else:
                plt.savefig('bland_altman.png', dpi=300)
        
        if nonparam:
            return print('\nMedian Difference: ' + str(round(mid, 3)) + ' ' + self.units + '\n' + 
                         'Upper Limit of Agreement: ' + str(round(upper, 3)) + ' ' + self.units + '\n' +
                         'Lower Limit of Agreement: ' + str(round(lower, 3)) + ' ' + self.units)

        else:    
            return print('\nMean Difference: ' + str(round(mid, 3)) + ' ' + self.units + '\n' + 
                         'Upper Limit of Agreement: ' + str(round(upper, 3)) + ' ' + self.units + '\n' + 
                         'Lower Limit of Agreement: ' + str(round(lower, 3)) + ' ' + self.units)

    # ==========================================================
    # ---- INTERPOLATION-CONFIDENCE REPORT ---------------------
    # ==========================================================

    def confidence_report(self, mapper, mesh_df, source_df, value_col,
                          mesh_x_col="X", source_x_col="x", depth_col="Z",
                          depth_bins=12, loocv_mean="arith",
                          cmap_support="viridis", cmap_disp="magma",
                          resid_space="ratio", figname=None, xlim=None):
        """Run support, dispersion, and LOOCV diagnostics for one mapping and
        render a four-panel confidence report.

        Parameters
        ----------
        mapper : DataMapper
            A DataMapper configured with the SAME num_neighbors / max_distance_xy /
            max_distance_z used to produce the mapping, so the diagnostics describe
            the neighbor sets that actually generated the mapped values.
        mesh_df : DataFrame
            Target mesh with 'x', 'y', depth_col (and ideally a local length column
            named by mesh_x_col, e.g. 'X', for the section x-axis).
        source_df : DataFrame
            Source data being interpolated (e.g. raw EMI soundings) with 'x', 'y',
            depth_col and value_col.
        value_col : str
            Column in source_df to diagnose (raw channel, not a *_arith column).
        mesh_x_col, source_x_col : str
            Horizontal axis used for the mesh-based panels and the LOOCV (source)
            panel respectively. mesh_x_col falls back to 'x' if absent.
        depth_bins : int
            Number of depth layers for the support-vs-depth profile.
        loocv_mean : {'arith','geom','harm'}
            Averaging rule cross-validated; match what you used when mapping.

        Returns
        -------
        dict with keys: support, support_by_depth, dispersion, loocv_residuals,
        loocv_summary, fill_fraction.
        """
        if mesh_x_col not in mesh_df.columns:
            mesh_x_col = "x"

        # Clean the mesh ONCE so the diagnostic outputs align row-for-row with it,
        # which lets us attach the local length column for plotting.
        mesh_c = (mesh_df.replace([np.inf, -np.inf], np.nan)
                         .dropna(subset=["x", "y", depth_col]).reset_index(drop=True))

        # --- the three diagnostics (all on the mapper) ---
        sup = mapper.support_field(mesh_c, source_df, depth_col=depth_col)
        depth_summary = mapper.support_by_depth(sup, depth_col=depth_col, nbins=depth_bins)
        disp = mapper.dispersion_field(mesh_c, source_df, value_cols=[value_col],
                                       depth_col=depth_col)
        # give EMI source points a local along-line coordinate by borrowing the
        # nearest mesh node's mesh_x_col, so the residual panel shares the axis
        source_x = source_df
        if mesh_x_col not in source_df.columns:
            from scipy.spatial import cKDTree
            mtree = cKDTree(mesh_c[["x", "y"]].values)
            _, mi = mtree.query(source_df[["x", "y"]].values, k=1)
            source_x = source_df.copy()
            source_x[mesh_x_col] = mesh_c[mesh_x_col].values[mi]
        resid, loocv = mapper.loocv_interpolation(source_x, value_col,
                                                  mean=loocv_mean, depth_col=depth_col,
                                                  keep_cols=[mesh_x_col])
        fill_fraction = float(getattr(mapper, "fill_fraction", np.nan))

        # attach horizontal coordinate to mesh-based fields (row-aligned with mesh_c)
        sup[mesh_x_col] = mesh_c[mesh_x_col].values
        disp[mesh_x_col] = mesh_c[mesh_x_col].values

        gcv_col = f"{value_col}_gcv"
        have_disp = (gcv_col in disp.columns) and bool(np.isfinite(disp[gcv_col]).any())

        # ----------------------------- figure -----------------------------
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax_sup, ax_disp = axes[0, 0], axes[0, 1]
        ax_res, ax_prof = axes[1, 0], axes[1, 1]

        # (a) spatial support: nearest-neighbor distance over supported cells
        s = sup[sup["supported"]]
        sc = ax_sup.scatter(s[mesh_x_col], s[depth_col], c=s["nn_dist"],
                            cmap=cmap_support, s=14)
        fig.colorbar(sc, ax=ax_sup, shrink=0.85).set_label("NN distance (m)")
        ax_sup.set_title(f"Spatial support (fill fraction = {fill_fraction:.1%})",
                         **self.titleFontKws)

        # (b) local dispersion: geometric CV (robust color limits so a few
        #     extreme-dispersion cells don't flatten the whole panel)
        if have_disp:
            d = disp[np.isfinite(disp[gcv_col])]
            gv = d[gcv_col].values
            lo, hi = np.nanpercentile(gv, (2, 98))
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                lo, hi = np.nanmin(gv), np.nanmax(gv)
            sc = ax_disp.scatter(d[mesh_x_col], d[depth_col], c=gv,
                                 cmap=cmap_disp, s=14, vmin=lo, vmax=hi)
            fig.colorbar(sc, ax=ax_disp, shrink=0.85, extend="both").set_label("geometric CV")
            ax_disp.set_title(f"Local dispersion of {value_col}", **self.titleFontKws)
        else:
            ax_disp.text(0.5, 0.5, "dispersion needs\nnum_neighbors > 1",
                         ha="center", va="center", transform=ax_disp.transAxes)
            ax_disp.set_title("Local dispersion", **self.titleFontKws)

        # (c) LOOCV residuals -- linear (ratio) space by default
        r = resid["resid_log10"].values
        if resid_space == "ratio":
            r = np.power(10.0, r); rc = 1.0; rlabel = "pred / obs"
        else:
            rc = 0.0; rlabel = r"residual $\log_{10}$"
        lim = np.nanpercentile(np.abs(r - rc), 98) if r.size else 1.0
        lim = lim if (np.isfinite(lim) and lim > 0) else 1.0
        vmin, vmax = rc - lim, rc + lim
        if resid_space == "ratio":
            vmin = max(vmin, 1e-9)
        sc = ax_res.scatter(resid[mesh_x_col], resid[depth_col], c=r,
                            cmap="coolwarm",
                            norm=TwoSlopeNorm(vcenter=rc, vmin=vmin, vmax=vmax), s=14)
        fig.colorbar(sc, ax=ax_res, shrink=0.85).set_label(rlabel)
        ax_res.set_title("LOOCV interpolation residuals", **self.titleFontKws)

        for ax, xcol in ((ax_sup, mesh_x_col), (ax_disp, mesh_x_col),
                         (ax_res, mesh_x_col)):
            lbl = "Length (m)" if xcol in ("X", "x") else xcol
            ax.set_xlabel(lbl, **self.labelFontKws)
            ax.set_ylabel("Depth (m)", **self.labelFontKws)
        if xlim is None:
            xlim = (float(mesh_c[mesh_x_col].min()), float(mesh_c[mesh_x_col].max()))
        for ax in (ax_sup, ax_disp, ax_res):
            ax.set_xlim(*xlim)

        # (d) support-vs-depth profile (the panel that flags non-informative pairings)
        mids = depth_summary["depth_bin"].apply(lambda iv: iv.mid).astype(float)
        l1, = ax_prof.plot(depth_summary["frac_supported"], mids, "o-",
                           color="steelblue", label="supported fraction")
        ax_prof.set_xlabel("Supported fraction", color="steelblue", **self.labelFontKws)
        ax_prof.set_ylabel("Depth (m)", **self.labelFontKws)
        ax_prof.tick_params(axis="x", colors="steelblue")
        ax_prof.set_xlim(-0.02, 1.02)
        ax2 = ax_prof.twiny()
        l2, = ax2.plot(depth_summary["median_nn_dist"], mids, "s--",
                       color="darkorange", label="median NN dist")
        ax2.set_xlabel("Median NN distance (m)", color="darkorange", **self.labelFontKws)
        ax2.tick_params(axis="x", colors="darkorange")
        ax_prof.legend(handles=[l1, l2], loc="lower right", fontsize=8)
        ax_prof.set_title("Support vs depth", **self.titleFontKws)

        # bottom-margin summary line
        summary_line = (
            f"LOOCV ({loocv_mean}):  RMSE={loocv['RMSE_log']:.3f} (x{10 ** loocv['RMSE_log']:.2f})   "
            f"MAE={loocv['MAE_log']:.3f} (x{10 ** loocv['MAE_log']:.2f})   "
            f"bias={loocv['bias_log']:+.3f} (x{10 ** loocv['bias_log']:.2f})   "
            f"median ratio={loocv['median_ratio']:.3f}    |    "
            f"n used={loocv['n_used']} (unpred. {loocv['n_unpredicted']})    |    "
            f"fill fraction={fill_fraction:.1%}")

        fig.suptitle(f"Interpolation confidence report — {value_col}",
                     fontsize=self.titleFontKws.get("fontsize", 14) + 2,
                     fontweight="bold")
        fig.text(0.5, 0.015, summary_line, ha="center", fontsize=10, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(figname or "confidence_report.png", dpi=300)
        plt.show()

        print("Interpolation confidence summary")
        print(f"  fill fraction (cells extrapolated, not interpolated): {fill_fraction:.1%}")
        print(f"  LOOCV log10 RMSE / MAE / bias: {loocv['RMSE_log']:.3f} / "
              f"{loocv['MAE_log']:.3f} / {loocv['bias_log']:+.3f}")
        if have_disp:
            print(f"  median geometric CV of mapped {value_col}: "
                  f"{np.nanmedian(disp[gcv_col]):.3f}")

        return {
            "support": sup,
            "support_by_depth": depth_summary,
            "dispersion": disp,
            "loocv_residuals": resid,
            "loocv_summary": loocv,
            "fill_fraction": fill_fraction,
        }

    # ==========================================================
    # ---- GROUND-TRUTH VALIDATION -----------------------------
    # ==========================================================
    def ground_truth_check(self, gt_df, value_col, model_df=None, model_col=None,
                           coord_cols=("x", "y", "Z"), k=8, radius=None,
                           agg="geom", weight="distance", weight_power=1.0,
                           weight_col=None, dev_space="ratio", plot=True,
                           plot_x_col="X", cmap="viridis", devCmap='coolwarm',
                           units=None, figname=None, cbticks=None, diffticks=None):
        """Compare point ground-truth measurements (groundwater / surface water
        / soil) against the informed inversion result.

        For each ground-truth point the surrounding mesh elements are gathered
        -- the k nearest (k=...) or all within `radius` -- their model values
        summarised (`agg` = 'geom' | 'arith' | 'median'), and the deviation
        from the ground-truth value reported in linear and log10 space. EC is
        assumed positive and roughly log-distributed, so 'geom' is the default
        summary and deviations are reported mainly in log10.

        gt_df values must be in the SAME quantity/units as `model_col` -- no
        conversion is performed (e.g. compare conductivity with conductivity).
        Defaults compare against the informed column (self.informedCol) of
        self.data. Returns one row per ground-truth point and, if plot=True,
        draws the inversion section with the points overlaid plus a signed
        deviation map. Also stored on self.ground_truth_results.
        """
        from scipy.spatial import cKDTree
        model_df = self.data if model_df is None else model_df
        model_col = self.informedCol if model_col is None else model_col
        coord_cols = list(coord_cols)
        njobs = int(getattr(self, "njobs", 1))

        md = (model_df.replace([np.inf, -np.inf], np.nan)
                      .dropna(subset=coord_cols + [model_col]).reset_index(drop=True))
        md = md[md[model_col] > 0].reset_index(drop=True)
        gt = (gt_df.replace([np.inf, -np.inf], np.nan)
                   .dropna(subset=coord_cols + [value_col]).reset_index(drop=True))
        gt = gt[gt[value_col] > 0].reset_index(drop=True)

        mco = md[coord_cols].values
        tree = cKDTree(mco)
        mval = md[model_col].values
        wcol = (md[weight_col].values
                if (weight_col is not None and weight_col in md.columns) else None)

        def _wagg(vals, w):
            w = np.asarray(w, dtype=float)
            if (w.sum() <= 0) or not np.all(np.isfinite(w)):
                w = np.ones_like(vals)
            if agg == "geom":
                return float(np.exp(np.sum(w * np.log(vals)) / np.sum(w)))
            if agg == "arith":
                return float(np.sum(w * vals) / np.sum(w))
            return float(np.median(vals))           # median: unweighted

        rows = []
        for _, gpt in gt.iterrows():
            p = gpt[coord_cols].values.astype(float)
            if radius is not None:
                idx = np.asarray(tree.query_ball_point(p, r=radius, workers=njobs), dtype=int)
                dd = np.sqrt(((mco[idx] - p) ** 2).sum(axis=1)) if idx.size else np.array([])
            else:
                dd, idx = tree.query(p, k=min(k, len(md)))
                idx = np.atleast_1d(idx); dd = np.atleast_1d(dd)
            vals = mval[idx]
            m = np.isfinite(vals) & (vals > 0)
            vals = vals[m]; dd = dd[m]; sidx = np.atleast_1d(idx)[m]
            obs = float(gpt[value_col])
            row = {c: float(gpt[c]) for c in coord_cols}
            if vals.size:
                if weight == "distance":
                    w = 1.0 / (dd ** weight_power + 1e-9)
                else:
                    w = np.ones_like(vals)
                if wcol is not None:
                    w = w * wcol[sidx]
                msum = _wagg(vals, w)
                dev = float(np.log10(msum) - np.log10(obs))
                row.update(observed=obs, model=msum, n_elements=int(vals.size),
                           local_logstd=float(np.log10(vals).std(ddof=1)) if vals.size > 1 else 0.0,
                           dev_linear=msum - obs, ratio=msum / obs,
                           dev_log10=dev, abs_dev_log10=abs(dev))
            else:
                row.update(observed=obs, model=np.nan, n_elements=0,
                           local_logstd=np.nan, dev_linear=np.nan, ratio=np.nan,
                           dev_log10=np.nan, abs_dev_log10=np.nan)
            rows.append(row)

        result = pd.DataFrame(rows)
        self.ground_truth_results = result
        d = result["dev_log10"].dropna().values
        print("Ground-truth validation vs", model_col)
        print("  points compared: {} / {}".format(d.size, len(result)))
        if d.size:
            rmse, mae, bias = (float(np.sqrt(np.mean(d ** 2))),
                               float(np.mean(np.abs(d))), float(np.mean(d)))
            print("  log10 RMSE / MAE / bias: {:.3f} / {:.3f} / {:+.3f}".format(rmse, mae, bias))
            print("  linear factors RMSE / MAE / bias: x{:.2f} / x{:.2f} / x{:.2f}".format(
                10 ** rmse, 10 ** mae, 10 ** bias))
            print("  median model/observed ratio: {:.3f}".format(float(10 ** np.median(d))))

        if plot:
            if any(word in model_col for word in ['inf', 'informed', 'Inf', 'Informed']):
                self._plot_ground_truth(md, mval, model_col, result, coord_cols,
                                        plot_x_col, cmap, devCmap, figname, dev_space=dev_space, rawOrInf='inf', units=units, cbticks=cbticks, diffticks=diffticks)
            else:
                self._plot_ground_truth(md, mval, model_col, result, coord_cols,
                                        plot_x_col, cmap, devCmap, figname, dev_space=dev_space, rawOrInf='raw', units=units, cbticks=cbticks, diffticks=diffticks)
        return result

    def _plot_ground_truth(self, md, mval, model_col, result, coord_cols,
                           plot_x_col, cmap, devCmap, figname, dev_space="ratio",
                           rawOrInf='inf', units=None, cbticks=None, diffticks=None):
        
        """Section overlay + signed-deviation map for ground_truth_check()."""
        u = units if units is not None else getattr(self, "units", None)
        if 'S/m' in u:
            metric = 'Conductivity'
        elif 'Ohm.m' in u:
            metric = 'Resistivity'
        else:
            metric = None
        from matplotlib.colors import LogNorm, TwoSlopeNorm
        from scipy.spatial import cKDTree
        depth_col = coord_cols[-1]
        lat_cols = list(coord_cols[:-1])

        if plot_x_col in md.columns:
            mx = md[plot_x_col].values
            if plot_x_col in result.columns:
                gx = result[plot_x_col].values
            else:                                   # project gt onto local X
                tr = cKDTree(md[lat_cols].values)
                _, mi = tr.query(result[lat_cols].values, k=1)
                gx = md[plot_x_col].values[mi]
        else:
            mx = md[lat_cols[0]].values
            gx = result[lat_cols[0]].values
        gz = result[depth_col].values

        if max(gz) < 0:
            md[depth_col] *= -1
            gz *= -1
            depthOrElev = 'Depth'
        else:
            depthOrElev = 'Elevation'

        if cbticks is not None:
            norm = LogNorm(vmin=cbticks[0], vmax=cbticks[-1])
        else:
            pos = mval[mval > 0]
            norm = LogNorm(vmin=np.nanmin(pos), vmax=np.nanmax(mval))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))

        a1.scatter(mx, md[depth_col].values, c=mval, cmap=cmap, norm=norm,
                   s=10, alpha=0.5)
        sc = a1.scatter(gx, gz, c=result["observed"].values, cmap=cmap, norm=norm,
                        s=140, edgecolor="k", linewidth=1.3)
        cb = fig.colorbar(sc, ax=a1, shrink=0.85)
        if u in model_col:
            if metric is not None:
                label = metric
            else:
                label = model_col.split('(')[0]
            units = model_col.split('(')[1]
            cb.set_label(label.title() + ' (' + units, **self.cbarTitleKws)
        else:
            if metric is not None:
                cb.set_label(metric.title() + (f" ({u})" if u else ""), **self.cbarTitleKws)
            else:
                cb.set_label(model_col.title() + (f" ({u})" if u else ""), **self.cbarTitleKws)
        if cbticks is not None:
            cb.set_ticks(cbticks)
            cb.set_ticklabels([str(t) for t in cbticks])
        cb.ax.tick_params(**self.cbarTickKws)
        if rawOrInf == 'raw':
            plotType = 'raw'
        else:
            plotType = 'informed'
        a1.set_title(f"Ground truth over {plotType} inversion".title(), **self.titleFontKws)
        a1.set_xlabel("Length (m)", **self.labelFontKws)
        a1.set_ylabel(depthOrElev + " (m)", **self.labelFontKws)
        if depthOrElev == 'Depth':
            a1.invert_yaxis()

        if dev_space == "ratio":
            d = result["ratio"].values; dc = 1.0; dlabel = "model / obs"
        elif dev_space == "difference":
            d = result["dev_linear"].values; dc = 0.0
            dlabel = f"model $-$ obs ({u})" if u else "model $-$ obs"
        else:
            d = result["dev_log10"].values; dc = 0.0; dlabel = r"model$-$obs ($\log_{10}$)"
        if diffticks is not None:
            vmin, vmax = diffticks[0], diffticks[-1]
        else:
            lim = np.nanpercentile(np.abs(d - dc), 98) if np.isfinite(d).any() else 1.0
            lim = lim if (np.isfinite(lim) and lim > 0) else (abs(dc) or 1.0)
            vmin, vmax = dc - lim, dc + lim
        if dev_space == "ratio":
            vmin = max(vmin, 1e-9)
        a2.scatter(mx, md[depth_col].values, c="0.85", s=8)
        if is_divergent(devCmap):
            sc2 = a2.scatter(gx, gz, c=d, cmap=devCmap,
                             norm=TwoSlopeNorm(vcenter=dc, vmin=vmin, vmax=vmax),
                             s=140, edgecolor="k", linewidth=1.3)
        else:
            sc2 = a2.scatter(gx, gz, c=d, cmap=devCmap,
                             norm=Normalize(vmin=vmin, vmax=vmax),
                             s=140, edgecolor="k", linewidth=1.3)
        cb2 = fig.colorbar(sc2, ax=a2, shrink=0.85)
        cb2.set_label(dlabel, **self.cbarTitleKws)
        if diffticks is not None:
            cb2.set_ticks(diffticks)
            cb2.set_ticklabels([str(t) for t in diffticks])
        cb2.ax.tick_params(**self.cbarTickKws)
        a2.set_title("Model$-$ground-truth deviation".title(), **self.titleFontKws)
        a2.set_xlabel("Length (m)", **self.labelFontKws)
        a2.set_ylabel(depthOrElev + " (m)", **self.labelFontKws)
        if depthOrElev == 'Depth':
            a2.invert_yaxis()
        fig.tight_layout()
        if figname:
            plt.savefig(figname, dpi=300)
        plt.show()
