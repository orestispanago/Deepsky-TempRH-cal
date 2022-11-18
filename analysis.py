import pandas as pd
import glob
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_matrix_lower(df):
    """ Plots lower triangle of scatter matrix """

    def corrfunc(x, y, **kwargs):
        """ Calculates Pearson's R and annotates axis
        Use on seaborn scatter matrix"""
        r, _ = stats.pearsonr(x, y)
        r2 = r ** 2
        ax = plt.gca()
        ax.annotate("$r^2$ = {:.2f}".format(r2),
                    xy=(.1, .9), xycoords=ax.transAxes, fontsize='x-small')

    def slope_intercept(x, y, **kwargs):
        """ Calculates slope + intercept and annotates axis
        Use on seaborn scatter matrix"""
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax = plt.gca()
        ax.annotate("y={0:.1f}x+{1:.1f}".format(slope, intercept),
                    xy=(.1, .9), xycoords=ax.transAxes, fontsize='x-small')

    grid = sns.PairGrid(data=df, vars=list(df), height=1)
    for m, n in zip(*np.triu_indices_from(grid.axes, k=0)):
        grid.axes[m, n].set_visible(False)
    grid = grid.map_lower(plt.scatter, s=0.2)
    grid.map_lower(corrfunc)
    grid.set(alpha=1)
    grid.fig.suptitle('Air Temperature (°C)')
    
    
plt.rcParams["axes.labelsize"] = 14


csv_files = sorted(glob.glob("raw/*.csv"))

df_list = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file, parse_dates=True, index_col='time')
    # df = df.resample('1min').mean()
    df = df[df['Tmean']<45]
    df = df[df['Tstdev']<0.4]
    df['Tmean'].plot(title=csv_file)
    plt.show()
    # df['Tstdev'].plot(title=csv_file)
    # plt.show()
    df_list.append(df)

df_all = pd.concat(df_list)
# df_all = df_all.resample('1min').mean()

pivoted = df_all.pivot(columns='stationID', values='Tmean')
pivoted = pivoted.resample('1min').mean().dropna()

scatter_matrix_lower(pivoted)
