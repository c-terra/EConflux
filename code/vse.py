# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:14:07 2025

@author: Chris
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydrostats as hs
import hydroeval as he
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib as mpl
import matplotlib.cm as cm
import math
import matplotlib.ticker
import matplotlib.colors as clr
import pysymlog as psl
psl.register_mpl()

plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'arial'

class EConfluxStats:
    """
    EConflux Statistics and Visualization Class
    For comparing ER vs EM datasets (raw and informed),
    generating diagnostic plots, and computing hydrological metrics.
    """

    def __init__(self, filepath, informingCol, rawCol, informedCol, drop_flag_col=None, 
                 informingMethod=None, informedMethod=None, units='mS/m', ticks=None, **fontkws):
        
        """
        **fontkws: A dictionary to set font properties for all figures created. Possible options include:
            fontfamily, fontweight, titlesize, titleweight, axislabelsize, axislabelweight, axisticksize, cbartitlesize, cbartitleweight, cbarticklabelsize
            
            See https://matplotlib.org/stable/api/font_manager_api.html#matplotlib.font_manager.FontProperties for font property options
        
        """
        
        # Load dataset
        self.data = pd.read_csv(filepath)

        # Drop flagged rows
        self.data.sort_values(by=["Z", "x", "y"], inplace=True, ascending=False)
        if type(drop_flag_col) is list:
            self.data = self.data.dropna(subset=[informingCol, rawCol, informedCol], ignore_index=True)

            for col in drop_flag_col:
                if col in self.data.columns:
                    self.data = self.data.drop(self.data.index[self.data[col] == 1])
        else:
            self.data = self.data.dropna(subset=[informingCol, rawCol, informedCol], ignore_index=True)

            if drop_flag_col in self.data.columns:
                self.data = self.data.drop(self.data.index[self.data[drop_flag_col] == 1])

        if drop_flag_col == 'nan':
            self.data = self.data.dropna(subset=[informingCol, rawCol, informedCol], ignore_index=True)

        # Core variables
        self.X = self.data[informingCol].loc[self.data[informingCol] > 0]  # Inversion Result from Informing Method
        self.y = self.data[rawCol].loc[self.data[rawCol] > 0]        # Raw Inversion Result
        self.Y = self.data[informedCol].loc[self.data[informedCol] > 0]    # Informed Inversion Result

        # Log-transformed variables
        self.Xlog = np.log10(self.X)
        self.ylog = np.log10(self.y)
        self.Ylog = np.log10(self.Y)

        self.informingMethod = informingMethod
        self.informedMethod = informedMethod

        self.units = units
        
        self.ticks = ticks

        # Set default font properties
        
        
        
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


        self.titleFontKws = {
            'fontsize': fontkwDict['titlefontsize'],
            'fontweight': fontkwDict['titlefontweight'],
            }
        
        self.labelFontKws = {
            'fontsize': fontkwDict['axislabelsize'],
            'fontweight': fontkwDict['axislabelweight'],
            }


        self.cbarTitleKws = {
            'fontsize': fontkwDict['cbartitlesize'],
            'fontweight': fontkwDict['cbartitleweight']}

        self.cbarTickKws = {
            'fontsize': fontkwDict['cbarticklabelsize'],
            }


    # ==============================================================
    # ---------------------- METRICS METHODS -----------------------
    # ==============================================================

    @staticmethod
    def KGEnp(sim, obs):
        """Non-parametric Kling–Gupta Efficiency."""
        sim, obs = np.asarray(sim, float), np.asarray(obs, float)
        # mask = np.isfinite(sim) & np.isfinite(obs)
        # sim, obs = sim[mask], obs[mask]
        if sim.size == 0:
            raise ValueError("No valid data")

        mean_sim, mean_obs = np.mean(sim), np.mean(obs)
        fdc_sim = np.sort(sim / (mean_sim * len(sim)))
        fdc_obs = np.sort(obs / (mean_obs * len(obs)))

        alpha = 1 - 0.5 * np.sum(np.abs(fdc_sim - fdc_obs))
        beta = mean_sim / mean_obs
        r = spearmanr(sim, obs).correlation

        return 1 - np.sqrt((alpha - 1) ** 2 + (beta - 1) ** 2 + (r - 1) ** 2)

    def metrics(self, logOrlin='log'):
        """Return dataframe of evaluation metrics for raw and informed datasets."""
        
        if logOrlin == 'log':
            infSim, rawSim, obs = self.Ylog, self.ylog, self.Xlog
        else:
            infSim, rawSim, obs = self.Y, self.y, self.X
        
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

        reg = LinearRegression().fit(X_log.values.reshape(-1, 1), Y_log.values.reshape(-1, 1))
        b = reg.coef_[0]
        log_a = reg.intercept_
        a = 10 ** log_a
        r2 = r2_score(Y_log.values.reshape(-1, 1), reg.predict(X_log.values.reshape(-1, 1)))

        return a, b, r2, reg

    # ==============================================================
    # -------------------- VISUALIZATION METHODS -------------------
    # ==============================================================
    
    def scatter_plots(self, figname=None):
        """
        Scatter plots comparing ER vs EM (raw and informed).
        Left panel: log10 scatter of observed values.
        Right panel: log10 scatter with power-law fits.
        """
        # Fit power-law models (raw + informed)
        a_nc, b_nc, r2_nc, _ = self._powerlaw_fit(self.X, self.y)
        a_cal, b_cal, r2_cal, _ = self._powerlaw_fit(self.X, self.Y)
    
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
        # -----------------------------------------------------------------
        # Left: log10 scatter (linearized)
        if (self.informingMethod is not None) & (self.informedMethod is not None):
            ax[0].scatter(self.X, self.y, c='dodgerblue', label=f"{self.informedMethod}" + r"$_{raw}$", alpha=0.6)
        else:
            ax[0].scatter(self.X, self.y, c='dodgerblue', label="EM$_{raw}$", alpha=0.6)
        ax[0].axline((0, 0), slope=1, color="k", linestyle="--", alpha=0.5)

        ax[0].legend(fontsize=8)
        ax[0].set_title("Raw–Informed Scatter Comparison", self.titleFontKws)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        
        if (self.ticks is not None) & (self.informingMethod is not None) & (self.informedMethod is not None):
            ax[0].set_xlabel(f"{self.informingMethod} EC ({self.units})", **self.labelFontKws) 
            ax[0].set_ylabel(f"{self.informedMethod} EC ({self.units})", **self.labelFontKws)
            ax[0].set_xlim(self.ticks[0], self.ticks[-1])
            ax[0].set_ylim(self.ticks[0], self.ticks[-1])
            ax[0].set_xticks(self.ticks)
            ax[0].set_yticks(self.ticks)
        elif (self.ticks is None) & (self.informingMethod is not None) & (self.informedMethod is not None):
            ax[0].set_xlabel(f"{self.informingMethod} EC ({self.units})", **self.labelFontKws) 
            ax[0].set_ylabel(f"{self.informedMethod} EC ({self.units})", **self.labelFontKws)
        else:
            ax[0].set_xlabel(f"Dataset 1 EC ({self.units})", **self.labelFontKws) 
            ax[0].set_xlabel(f"Dataset 2 EC ({self.units})", **self.labelFontKws) 
    
        ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(float(x))))
        ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, pos: str(float(y))))
        

    
        # -----------------------------------------------------------------
        # Right: log10 scatter with power-law fits (same style as ax[0])
        if (self.informingMethod is not None) & (self.informedMethod is not None):
            ax[1].scatter(self.X, self.y, c='dodgerblue', label=f"{self.informedMethod}" + r"$_{raw}$", alpha=0.6)
            ax[1].scatter(self.X, self.Y, c='goldenrod', label=f"{self.informedMethod}" + r"$_{informed}$", alpha=0.6)
        else:
            ax[1].scatter(self.X, self.y, c='dodgerblue', label="EM$_{raw}$", alpha=0.6)
            ax[1].scatter(self.X, self.Y, c='goldenrod', label="EM$_{informed}$", alpha=0.6)
        ax[1].axline((0, 0), slope=1, color="k", linestyle="--", alpha=0.5)
    
        # Power-law fits (log10-transformed for consistency)
        x_fit = np.logspace(np.log10(self.X.min()), np.log10(self.X.max()), 200)
        y_fit_nc = a_nc * x_fit ** b_nc
        y_fit_cal = a_cal * x_fit ** b_cal
    
        ax[1].plot(x_fit, y_fit_nc, c="royalblue", lw=2, alpha=0.8,
                   label=f"Raw Fit: a={a_nc[0]:.2f}, b={b_nc[0]:.2f}, R$^2$={r2_nc:.2f}")
        ax[1].plot(x_fit, y_fit_cal, c="chocolate", lw=2, alpha=0.8,
                   label=f"Informed Fit: a={a_cal[0]:.2f}, b={b_cal[0]:.2f}, R$^2$={r2_cal:.2f}")

        ax[1].legend(fontsize=9, frameon=False)
        ax[1].set_title("Raw–Informed Scatter Comparison with Power-Law Fits", **self.titleFontKws)
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        
        if (self.ticks is not None) & (self.informingMethod is not None) & (self.informedMethod is not None):
            ax[1].set_xlabel(f"{self.informingMethod} EC ({self.units})", **self.labelFontKws) 
            ax[1].set_ylabel(f"{self.informedMethod} EC ({self.units})", **self.labelFontKws)
            ax[1].set_xlim(self.ticks[0], self.ticks[-1])
            ax[1].set_ylim(self.ticks[0], self.ticks[-1])
            ax[1].set_xticks(self.ticks)
            ax[1].set_yticks(self.ticks)
        elif (self.ticks is None) & (self.informingMethod is not None) & (self.informedMethod is not None):
            ax[1].set_xlabel(f"{self.informingMethod} EC ({self.units})", **self.labelFontKws) 
            ax[1].set_ylabel(f"{self.informedMethod} EC ({self.units})", **self.labelFontKws)
        else:
            ax[1].set_xlabel(f"Dataset 1 EC ({self.units})", **self.labelFontKws) 
            ax[1].set_xlabel(f"Dataset 2 EC ({self.units})", **self.labelFontKws) 
            
        ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(float(x))))
        ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, pos: str(float(y))))
        
    
        plt.tight_layout()
        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.savefig('scatter.png', dpi=300)
        plt.show()
    
        return {
            "Raw Fit": {"a": a_nc, "b": b_nc, "R$^2$": r2_nc},
            "Informed Fit": {"a": a_cal, "b": b_cal, "R²": r2_cal}
        }
    
        return {
            "Raw Fit": {"a": a_nc, "b": b_nc, "R$^2$": r2_nc},
            "Informed Fit": {"a": a_cal, "b": b_cal, "R²": r2_cal}
        }

    def kde_histograms(self, nbins=100, figname=None):
        """Kernel density and histogram plots with metrics annotated."""
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
        
        ax[1].text(0.05, 0.95,
                   f"RMSE: {hs.rmse(self.Y, self.X):.2f}\n"
                   f"PBIAS: {he.evaluator(he.pbias, self.Y, self.X)[0]:.2f}\n"
                   f"KGE_np: {self.KGEnp(self.Y, self.X):.2f}",
                   transform=ax[1].transAxes, va="top", fontsize=9)
        
        ax[1].set_title(f"KDE Histogram (Informed {self.informedMethod} vs. {self.informingMethod}) ({nbins} bins)", **self.titleFontKws)


        plt.tight_layout()
        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.savefig('kde hist.png', dpi=300)
        plt.show()
        
    def qq_plot(self, sim, obs, xlabel='Dataset 1', ylabel='Dataset 2', title="QQ Plot", figname=None):
        """Quantile-Quantile plot."""
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
                     title=None, ax=None, ymin=None, ymax=None, xmin=None, xmax=None, figname=None, 
                     cbar=False, dcolor=False, dcmap='cividis', dbnds=None, xticks=None, logList=None, legendParams={'plot': True, 'loc': 'best'}):
        """
        Bland–Altman plot with 95% confidence limits.
        Returns summary statistics and shows plot.
                
        logList: A list where the first entry determines whether to use a log-scale for the x-axis (0 or False for linear, 1 or True for log)
                and the second entry determines whether to use a log-scale for the y-axis. (e.g. [1, 0] means that the x-axis is a log-scale and the y-axis is linear)     
                
        """
        diff = sim - obs
        mean = (sim + obs) / 2

        sdev_diff = np.std(diff)
        mean_diff = np.mean(diff)
        UCL = mean_diff + (1.96 * sdev_diff)
        LCL = mean_diff - (1.96 * sdev_diff)

        ratio_lower, ratio_upper = np.exp(LCL), np.exp(UCL)
        
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
        if dcolor:
            if dbnds is not None:
                if plt.get_cmap(dcmap).N == 256:
                    dcmap = plt.get_cmap(dcmap, len(dbnds))
                norm = mpl.colors.BoundaryNorm(dbnds, ncolors=len(dbnds))
            else:
                norm = mpl.colors.Normalize(vmin=0, vmax=abs(self.data['Z'].min()))
                        
            cmmapable = cm.ScalarMappable(norm, dcmap)
            
            # ticks = np.linspace(0, abs(self.data['Z'].min()).round(1), 5)
            
            # formattedTicks = np.array([f"{x:.1f}" for x in ticks])
            ax.scatter(mean, diff, c=abs(self.data['Z']), cmap=dcmap, norm=norm, alpha=0.3, marker="+")
            if cbar:
                cb = plt.colorbar(cmmapable, ax=ax, location='right', orientation='vertical', boundaries=dbnds)
                cb.set_label('Depth (m)', **self.cbarTitleKws)
                cb.ax.tick_params(axis='y', labelsize=self.cbarTickKws['fontsize'])
        else:
            ax.scatter(mean, diff, color='k', alpha=0.2, marker="+")
        
        ax.axhline(mean_diff, color="orangered", linestyle="-", label="Mean diff")
        ax.axhline(UCL, color="orangered", linestyle="--", label="Upper 95% limit")
        ax.axhline(LCL, color="orangered", linestyle="--", label="Lower 95% limit")
        if legendParams['plot']:
            ax.legend(loc=legendParams['loc'])

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
        # plt.tight_layout()
        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.savefig('bland_altman.png', dpi=300)
        # plt.show()
        

        return {
            "Mean difference": round(mean_diff, 3),
            "Std. deviation": round(sdev_diff, 3),
            "Upper 95% limit": round(UCL, 3),
            "Lower 95% limit": round(LCL, 3),
            "Ratio range": f"{ratio_lower:.2f}× – {ratio_upper:.2f}× ER"
        }