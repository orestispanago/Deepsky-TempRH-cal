import pandas as pd
import glob
import numpy as np
from math import log10, floor
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

SMALL_SIZE = 8
MEDIUM_SIZE = 14
LARGE_SIZE = 20
BIGGER_SIZE = 42
     
logger = logging.getLogger()

logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)


file_handler = logging.FileHandler('report.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

def read_csv_files(csv_files):
    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, parse_dates=True, index_col='time')
        df_list.append(df)
    df_all = pd.concat(df_list)
    return df_all


def scatter_matrix_lower(df, title='Air Temperature (°C)', pic_name=""):
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
        ax.annotate(f"y={slope:-.2f}x{intercept:+.2f}",
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
    plt.savefig(pic_name)
    plt.show()


def filter_data(df):
    df = df[df['Tmean']<45]
    df = df[df['Tstdev']<0.4]
    df = df[df['RHstdev']<4]
    return df


def iqr(df, xcol, ycol):
    resids = df[ycol] - df[xcol]
    q75 = np.percentile(resids, 75)
    q25 = np.percentile(resids, 25)
    iqr = q75 - q25  # InterQuantileRange
    good = df.loc[((resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr)))]
    bad = df.loc[~((resids > (q25 - 1.5 * iqr)) & (resids < (q75 + 1.5 * iqr)))]
    return good, bad


def plot_scatter_outliers(good, bad, xcol='T', ycol='Tmean', 
                          title="Air Temperature (°C)",
                          folder=''):
    x = good[xcol]
    y = good[ycol]
    reg_results = stats.linregress(x,y)
    logger.info(f"{'-' * 50}")
    logger.info(f"Station: {station_id}")
    logger.info(title)
    count=len(good)+len(bad)
    logger.info(f'Count: {count}')
    logger.info(f'Non-outliers: {len(good)}')
    logger.info(f'Outliers: {len(bad)}')
    logger.info(f'Outlier percentage: {len(bad)/count*100:.1f} %')
    formatted_equation = format_equation(reg_results)
    slope = reg_results.slope
    intercept = reg_results.intercept
    plt.figure(figsize=(10,10))
    plt.scatter(good[xcol], good[ycol], s=7, label='Non-outliers')
    plt.scatter(bad[xcol], bad[ycol], s=7, label='Outliers')
    plt.plot(x, intercept + slope*x, 'r', label=formatted_equation)
    plt.legend(fontsize=LARGE_SIZE, markerscale=3.)
    plt.title(title, fontsize=LARGE_SIZE)
    plt.xlabel('Meteo', fontsize=LARGE_SIZE)
    plt.ylabel(f'Station {station_id}', fontsize=LARGE_SIZE)
    plt.xticks(fontsize=MEDIUM_SIZE)
    plt.yticks(fontsize=MEDIUM_SIZE)
    plt.savefig(f'{folder}/{station_id}.png')
    plt.show()
    return reg_results

def compare_all(df_list, ref_col='T',test_col="Tmean", 
                title="Air Temperature (°C)",
                pic_name=""):
    df = pd.concat(df_list)
    pivoted = df.pivot(columns='stationID', values=test_col)
    pivoted = pivoted.resample('1min').mean().dropna()
    pivoted.columns = [f"Station {i:.0f}" for i in list(pivoted)]
    pivoted_all = pd.concat([pivoted, meteo[ref_col]], axis=1).dropna()
    pivoted_all.rename(columns={ref_col: 'Meteo'}, inplace=True)
    scatter_matrix_lower(pivoted_all, title=title, pic_name=pic_name)



def round_to_1(x):
    """ Rounds float to first significant digit """
    return round(x, -int(floor(log10(abs(x)))))

def num_w_error(number, err):
    err = round_to_1(err)
    err_decimals = f"{err}"[::-1].find('.')
    number = round(number, err_decimals)
    return number, err

def format_equation(reg, with_errors=False):
    slope, slope_err = num_w_error(reg.slope, reg.stderr)
    intercept, intercept_err = num_w_error(reg.intercept, reg.intercept_stderr)
    line = f"y=({slope}+-{slope_err})x+({intercept}+-{intercept_err}), r2={reg.rvalue**2:.2f}"
    logger.info(line)
    if not with_errors:
        line = f"y={slope:-}x{intercept:+}"
        return line
    return line

csv_files = sorted(glob.glob("raw/*.csv"))

df_all = read_csv_files(csv_files[:-1])
# df_all = filter_data(df_all)

meteo = pd.read_csv('Meteo_1min_2021_qc.csv', parse_dates=True, index_col='Time')
meteo = meteo.resample('1min').mean()

good_T = []
good_RH = []
for station_id, df in df_all.groupby('stationID'):
    df = df.resample('1min').mean()
    merged = pd.concat([df, meteo], axis=1).dropna()
    good_t, bad_t = iqr(merged, xcol='T', ycol='Tmean')
    good_T.append(good_t)
    good_rh, bad_rh = iqr(merged, xcol='phi', ycol='RHmean')
    good_RH.append(good_rh)
    plot_scatter_outliers(good_t, bad_t, folder='plots/temp')
    reg_results = plot_scatter_outliers(good_rh, bad_rh, xcol='phi', ycol='RHmean', 
                          title='Relative Humidity (%)',
                          folder='plots/rh')




# compare_all(good_T, pic_name='plots/temp.png')
# compare_all(good_RH, ref_col='phi', test_col='RHmean',
#             title = 'Relative Humidity (%)',
#             pic_name='plots/rh.png')



    