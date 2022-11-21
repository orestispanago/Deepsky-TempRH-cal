import pandas as pd
import glob
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 42


def read_csv_files(csv_files):
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, parse_dates=True, index_col='time')
        df_list.append(df)
    df_all = pd.concat(df_list)
    return df_all


def scatter_matrix_lower(df, title='Air Temperature (°C)'):
    """ Plots lower triangle of scatter matrix """

    def corrfunc(x, y, **kwargs):
        """ Calculates Pearson's R and annotates axis
        Use on seaborn scatter matrix"""
        r, _ = stats.pearsonr(x, y)
        r2 = r ** 2
        ax = plt.gca()
        ax.annotate("$r^2$ = {:.2f}".format(r2),
                    xy=(.1, .9), xycoords=ax.transAxes, fontsize=SMALL_SIZE)

    def slope_intercept(x, y, **kwargs):
        """ Calculates slope + intercept and annotates axis
        Use on seaborn scatter matrix"""
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        ax = plt.gca()
        ax.annotate("y={0:.1f}x+{1:.1f}".format(slope, intercept),
                    xy=(.1, .9), xycoords=ax.transAxes, fontsize=SMALL_SIZE)
    plt.rcParams["axes.labelsize"] = MEDIUM_SIZE
    grid = sns.PairGrid(data=df, vars=list(df), height=1)
    for m, n in zip(*np.triu_indices_from(grid.axes, k=0)):
        grid.axes[m, n].set_visible(False)
    grid.fig.set_size_inches(10,10)
    grid = grid.map_lower(plt.scatter, s=0.2)
    grid.map_lower(corrfunc)
    grid.set(alpha=1)
    grid.fig.suptitle(title, y=0.85, fontsize = MEDIUM_SIZE)
    plt.show()


def filter_data(df):
    df = df[df['Tmean']<45]
    df = df[df['Tstdev']<0.4]
    df = df[df['RHstdev']<4]
    return df

csv_files = sorted(glob.glob("raw/*.csv"))

df_all = read_csv_files(csv_files)
df_all = filter_data(df_all)
pivoted = df_all.pivot(columns='stationID', values='Tmean')
pivoted = pivoted.resample('1min').mean().dropna()
pivoted.columns = [f"Station {i}" for i in list(pivoted)]

scatter_matrix_lower(pivoted)


# pivoted = df_all.pivot(columns='stationID', values='RHmean')
# pivoted = pivoted.resample('1min').mean().dropna()
# pivoted.columns = [f"station{i}" for i in list(pivoted)]
# scatter_matrix_lower(pivoted, title="RH (%)")


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
for station_id, df in df_all.groupby('stationID'):
    df = df.resample('1min').mean()
    df["Tmean"].plot()
    # plt.scatter(df["Tmean"], df["RHmean"], marker='.')
    plt.title(f"Station {station_id}")
    plt.show()