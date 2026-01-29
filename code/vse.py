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
from sklearn.metrics import r2_score
import matplotlib as mpl
import matplotlib.cm as cm
import math
import matplotlib.ticker
import matplotlib.colors as clr
import matplotlib.patheffects as pe
import pysymlog as psl
psl.register_mpl()

plt.rcParams['font.size'] = 14
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


        print('Max Depth = ' + str(abs(min(self.data['Z']))))
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
        self.ticktypes = list(type(item) for item in ticks)
        
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
    def KGEnp(sim, obs, decomp=False):
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
        kge = 1 - np.sqrt((alpha - 1) ** 2 + (beta - 1) ** 2 + (r - 1) ** 2)

        if decomp:
            return alpha, beta, r, kge
        else:
            return kge

    def metrics(self, logOrlin='log', decompKGE=False):
        """Return dataframe of evaluation metrics for raw and informed datasets."""
        
        if logOrlin == 'log':
            infSim, rawSim, obs = self.Ylog, self.ylog, self.Xlog
        else:
            infSim, rawSim, obs = self.Y, self.y, self.X
        
        alphaInf, betaInf, rInf, kgeInf = self.KGEnp(infSim, obs, decomp=True)
        alphaRaw, betaRaw, rRaw, kgeRaw = self.KGEnp(rawSim, obs, decomp=True)
        
        if decompKGE:
            metrics = {
                "KGE_np": [kgeInf, kgeRaw],
                "KGE_alpha": [alphaInf, alphaRaw],
                "KGE_beta": [betaInf, betaRaw],
                "KGE_r": [rInf, rRaw],
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
        else:
            metrics = {
                "KGE_np": [kgeInf, kgeRaw],
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
        pred_y = p(X_log)

        a = 10**coeffs[1]
        b = coeffs[0]
        r2 = r2_score(X_log.values.reshape(-1, 1), pred_y.reshape(-1, 1))

        return a, b, r2, p

    # ==============================================================
    # -------------------- VISUALIZATION METHODS -------------------
    # ==============================================================
    
    def scatter_plots(self, log=True, grayscale=False, figname=None):
        """
        Scatter plots comparing ER vs EM (raw and informed).
        Left panel: log10 scatter of observed values.
        Right panel: log10 scatter with power-law fits.
        """
        # Fit power-law models (raw + informed)
        if log:
            a_nc, b_nc, r2_nc, _ = self._powerlaw_fit(self.X, self.y)
            a_cal, b_cal, r2_cal, _ = self._powerlaw_fit(self.X, self.Y)            
    
        fig, ax = plt.subplots(layout='constrained')
    
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
    
        # Power-law fits (log10-transformed for consistency)
        x_fit = np.logspace(np.log10(self.X.min()), np.log10(self.X.max()), 200)
        y_fit_nc = a_nc * x_fit ** b_nc
        y_fit_cal = a_cal * x_fit ** b_cal
    
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

        leg = ax.legend(fontsize=11, frameon=False)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
            
        ax.set_title("Raw–Informed Scatter Comparison with Power-Law Fits", **self.titleFontKws)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(float(x))))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, pos: str(float(y))))
        
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
        
        if float not in self.ticktypes:
            formatter = matplotlib.ticker.StrMethodFormatter("{x:d}")
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)
        
        ax.tick_params(which='minor', labelbottom=False, labeltop=False, labelleft=False, labelright=False) 
        
        # plt.tight_layout()
        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.savefig('scatter.png', dpi=300)
        plt.show()
    
        return {
            "Raw Fit": {"a": a_nc, "b": b_nc, "R$^2$": r2_nc},
            "Informed Fit": {"a": a_cal, "b": b_cal, "R²": r2_cal}
        }

    def kde_histograms(self, nbins=100, figname=None):
        """Kernel density and histogram plots with metrics annotated."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')

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
        
        titlekws = self.titleFontKws
        if self.titleFontKws['fontsize'] > 17:
            titlekws['fontsize'] = 17
            
        ax[0].set_title(f"KDE Histogram (Raw {self.informedMethod} vs. {self.informingMethod}) ({nbins} bins)", **titlekws)
        ax[0].set_ylabel('Probability Density', **self.labelFontKws)
        
        ax[0].tick_params(which='minor', labelbottom=False, labeltop=False, labelleft=False, labelright=False) 

        if float not in self.ticktypes:
            formatter = matplotlib.ticker.StrMethodFormatter("{x:d}")
            plt.gca().xaxis.set_major_formatter(formatter)

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
        
        ax[1].set_title(f"KDE Histogram (Informed {self.informedMethod} vs. {self.informingMethod}) ({nbins} bins)", **titlekws)

        ax[1].tick_params(which='minor', labelbottom=False, labeltop=False, labelleft=False, labelright=False) 

        if float not in self.ticktypes:
            formatter = matplotlib.ticker.StrMethodFormatter("{x:d}")
            plt.gca().xaxis.set_major_formatter(formatter)

        # plt.tight_layout()
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
                     nonparam=False, title=None, ax=None, marker='+', markersize=None, ymin=None, ymax=None, xmin=None, xmax=None, figname=None, 
                     cbar=False, dcolor=False, dcmap='cividis', dbnds=None, xticks=None, logList=None,
                     legendParams={'plot': True, 'loc': 'best'},
                     linekwargs={'color': 'orangered', 'linestyle': ['-', '--'], 'label': ['Mean Difference', 'Upper Agreement Limit', 'Lower Agreement Limit']}, **kwargs):
        """
        Bland–Altman plot with 95% confidence limits.
        Returns summary statistics and shows plot.
                
        logList: A list where the first entry determines whether to use a log-scale for the x-axis (0 or False for linear, 1 or True for log)
                and the second entry determines whether to use a log-scale for the y-axis. (e.g. [1, 0] means that the x-axis is a log-scale and the y-axis is linear)     
                
        """
        diff = sim - obs
        mean = (sim + obs) / 2

        
        mainlinestyle = linekwargs['linestyle'][0]
        loalinestyle = linekwargs['linestyle'][1]
        del linekwargs['linestyle']
        
        midlinelabel = linekwargs['label'][0]
        upperlinelabel = linekwargs['label'][1]
        lowerlinelabel = linekwargs['label'][2]
        del linekwargs['label']

        if nonparam:
            mid = np.median(diff)
            upper = np.quantile(diff, 0.975)
            lower = np.quantile(diff, 0.025)
        else:
            sdev_diff = np.std(diff)
            mid = np.mean(diff)
            upper = mid + (1.96 * sdev_diff)
            lower = mid - (1.96 * sdev_diff)

        # ratio_lower, ratio_upper = np.exp(lower), np.exp(upper)  # Double-check need for this
        
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
            ax.scatter(mean, diff, c=abs(self.data['Z']), marker=marker, s=markersize, cmap=dcmap, norm=norm, **kwargs)
            if cbar:
                cb = plt.colorbar(cmmapable, ax=ax, location='right', orientation='vertical', boundaries=dbnds)
                cb.set_label('Depth (m)', **self.cbarTitleKws)
                cb.ax.tick_params(axis='y', labelsize=self.cbarTickKws['fontsize'])
        else:
            ax.scatter(mean, diff, marker=marker, s=markersize, **kwargs)

        ax.axhline(mid, linestyle=mainlinestyle, label=midlinelabel, **linekwargs)
        ax.axhline(upper, linestyle=loalinestyle, label=upperlinelabel, **linekwargs)
        ax.axhline(lower, linestyle=loalinestyle, label=lowerlinelabel, **linekwargs)
        if legendParams['plot']:
            leg = ax.legend(loc=legendParams['loc'], fontsize=plt.rcParams['font.size']-3)
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

        if figname is not None:
            plt.savefig(figname, dpi=300)
        else:
            plt.savefig('bland_altman.png', dpi=300)
        # plt.show()
        
        if nonparam:
            return print('\nMedian Difference: ' + str(round(mid, 3)) + ' ' + self.units + '\n' + 
                         'Upper Limit of Agreement: ' + str(round(upper, 3)) + ' ' + self.units + '\n' +
                         'Lower Limit of Agreement: ' + str(round(lower, 3)) + ' ' + self.units)

        else:    
            return print('\nMean Difference: ' + str(round(mid, 3)) + ' ' + self.units + '\n' + 
                         'Upper Limit of Agreement: ' + str(round(upper, 3)) + ' ' + self.units + '\n' + 
                         'Lower Limit of Agreement: ' + str(round(lower, 3)) + ' ' + self.units)
