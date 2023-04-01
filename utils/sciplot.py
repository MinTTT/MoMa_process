import matplotlib as mpl
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def ggplot():
    mpl.style.use('ggplot')
    mpl.rcParams['lines.linewidth'] = 2.6  # lines 对象的粗细，plot等图中的线条粗细
    mpl.rcParams['font.size'] = 16  # 全局字体大小
    mpl.rcParams['axes.linewidth'] = 3  # 子图边框粗细
    mpl.rcParams['figure.subplot.wspace'] = 0.4  # 子图之间宽度，<1
    mpl.rcParams['figure.subplot.hspace'] = 0.4  # 子图之间高度，<1
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['lines.markersize'] = 13.0  # marker size


def whitegrid():
    mpl.style.use('seaborn-whitegrid')
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['figure.subplot.wspace'] = 0.4  # 子图之间宽度，<1
    mpl.rcParams['figure.subplot.hspace'] = 0.4  # 子图之间高度，<1


    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.linewidth'] = 6  # 子图边框粗细

    mpl.rcParams['font.family'] = 'Arial'  # 全局字体大小
    mpl.rcParams['font.size'] = 35  # 全局字体大小
    mpl.rcParams['font.weight'] = 'bold'

    mpl.rcParams['lines.linewidth'] = 10  # lines 对象的粗细，plot等图中的线条粗细
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['lines.markersize'] = 16.0  # marker size
    mpl.rcParams['xtick.major.size'] = 18  # xtick major length
    mpl.rcParams['xtick.major.width'] = 5
    mpl.rcParams['xtick.minor.size'] = mpl.rcParams['xtick.major.size'] * 2 / 4  # xtick minor length
    mpl.rcParams['xtick.minor.width'] = 5

    mpl.rcParams['xtick.direction'] = 'out'  # xtick length
    mpl.rcParams['ytick.major.size'] = 18  # ytick length
    mpl.rcParams['ytick.minor.size'] = mpl.rcParams['ytick.major.size'] * 2 / 4  # xtick minor length
    mpl.rcParams['ytick.direction'] = 'out'  # ytick length
    mpl.rcParams['ytick.minor.width'] = 5
    mpl.rcParams['ytick.major.width'] = 5
    mpl.rcParams['figure.figsize'] = (8, 8)
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.grid'] = False


def twoaxiserrorbar(x, y, hues, data):
    class_set = [list(set(data[cl])) for cl in hues]
    class_shape = [len(cls) for cls in class_set]
    class_comb = [np.arange(cls2) for cls2 in class_shape]
    class_x, class_y = np.meshgrid(class_comb[0], class_comb[1])
    _class = [[class_set[0][i], class_set[1][j]] for i, j in zip(class_x.flatten(), class_y.flatten())]
    data_x = np.zeros((len(_class), 2))
    data_y = np.zeros((len(_class), 2))
    for ind, cal_co in enumerate(_class):
        rdata_x = data[x].loc[(data[hues[0]] == cal_co[0]) & (data[hues[1]] == cal_co[1])]
        rdata_y = data[y].loc[(data[hues[0]] == cal_co[0]) & (data[hues[1]] == cal_co[1])]
        data_x[ind, :] = [rdata_x.mean(), np.std(rdata_x)]
        data_y[ind, :] = [rdata_y.mean(), np.std(rdata_y)]
    plt.gca()
    df = pd.DataFrame({x: data_x[:, 0], 'x_error': data_x[:, 1], y: data_y[:, 0], 'y_error': data_y[:, 1],
                       hues[0]: np.array(_class)[:, 0], hues[1]: np.array(_class)[:, 1]})
    sns.lineplot(x=x, y=y, data=df, hue=hues[0], legend=False)
    sns.scatterplot(x=x, y=y, data=df, hue=hues[0])
    plt.errorbar(data_x[:, 0], data_y[:, 0], xerr=data_x[:, 1], yerr=data_y[:, 1], lw=0,
                 elinewidth=3, capsize=3, ecolor='k')
    return None


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s = r'%s{\times}%s' % (significand, exponent)
        else:
            s = r'%s%s' % (significand, exponent)
        return "${}$".format(s)


def aspect_ratio(ratio, ax=None):
    if ax is None:
        ax = plt.gca()
    scale_x = ax.get_xscale()
    xmin, xmax = ax.get_xlim()

    scale_y = ax.get_yscale()
    ymin, ymax = ax.get_ylim()
    if scale_x == 'log':
        xmin, xmax = np.log((xmin, xmax))
    if scale_y == 'log':
        ymin, ymax = np.log((ymin, ymax))
    ax.set_aspect(ratio * np.ptp([xmin, xmax])/np.ptp([ymin, ymax]))
    return None