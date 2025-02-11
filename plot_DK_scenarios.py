import covasim as cv
import pandas as pd
import sciris as sc
import pylab as pl
import numpy as np
from matplotlib import ticker
import datetime as dt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Paths and filenames
figsfolder = 'figs'
resfolder = 'results'
scenarios = ['lift07', 'lift21', 'phased']
labels = ['Reopen Feb 8', 'Reopen Feb 23', 'Phased reopening from Feb 23']
T = sc.tic()

# Define plotting functions
#%% Helper functions

def format_ax(ax, sim, key=None):
    @ticker.FuncFormatter
    def date_formatter(x, pos):
        return (sim['start_day'] + dt.timedelta(days=int(x))).strftime('%b\n%y')
    ax.xaxis.set_major_formatter(date_formatter)
    if key != 'r_eff':
        sc.commaticks()
    pl.xlim([0, sim['n_days']])
    pl.axvspan(lockdown1[0], lockdown1[1], color='steelblue', alpha=0.2, lw=0)
    pl.axvspan(lockdown2[0], lockdown2[1], color='steelblue', alpha=0.2, lw=0)

    return

def plotter(key, sims, ax, label='', ylabel='', low_q=0.05, high_q=0.95, subsample=2):

    which = key.split('_')[1]
    try:
        color = cv.get_colors()[which]
    except:
        color = [0.5,0.5,0.5]

    ys = []
    for s in sims:
        ys.append(s.results[key].values)
    yarr = np.array(ys)

    best = pl.median(yarr, axis=0)
    low  = pl.quantile(yarr, q=low_q, axis=0)
    high = pl.quantile(yarr, q=high_q, axis=0)


    tvec = np.arange(len(best))
#    tempsim = cv.Sim(datafile='../UK_Covid_cases_january03.xlsx')
#    sim = sims[0]
#    if key in tempsim.data:
#        data_t = np.array((tempsim.data.index-sim['start_day'])/np.timedelta64(1,'D'))
#        inds = np.arange(0, len(data_t), subsample)
#        data = tempsim.data[key][inds]
#        pl.plot(data_t[inds], data, 'd', c=color, markersize=10, alpha=0.5, label='Data')

    fill_label = None
    end = None
    start = 2 if key == 'r_eff' else 0
    pl.fill_between(tvec[start:end], low[start:end], high[start:end], facecolor=color, alpha=0.2, label=fill_label)
    pl.plot(tvec[start:end], best[start:end], c=color, label=label, lw=4, alpha=1.0)

    sc.setylim()

    datemarks = pl.array([sim.day('2020-03-01'),sim.day('2020-05-01'),
                          sim.day('2020-07-01'),sim.day('2020-09-01'),
                          sim.day('2020-11-01'),sim.day('2021-01-01'),
                          sim.day('2021-03-01'),sim.day('2021-05-01')])
    ax.set_xticks(datemarks)
    pl.ylabel(ylabel)

    return


# Fonts and sizes
font_size = 36
font_family = 'Libertinus Sans'
pl.rcParams['font.size'] = font_size
pl.rcParams['font.family'] = font_family

# Plot locations
# Subplot sizes
xgapl = 0.1
xgapm = 0.02
xgapr = 0.03
ygapb = 0.075
ygapm = 0.02
ygapt = 0.05
nrows = 3
ncols = 3
dx = (1 - (ncols - 1) * xgapm - xgapl - xgapr) / ncols
dy = (1 - (nrows - 1) * ygapm - ygapb - ygapt) / nrows
nplots = nrows * ncols
ax = {}



pl.figure(figsize=(24, 16))
# Import files
filepaths = [f'{resfolder}/denmark_scen_{scen}.obj' for scen in scenarios]
sims = sc.odict()
msims = sc.odict()

for scen in scenarios:
    filepath = f'{resfolder}/denmark_scen_{scen}.obj'
    msims[scen] = sc.loadobj(filepath)
    sims[scen] = msims[scen].sims
    msims[scen].reduce()

sim = sims[0][0] # Extract a sim to refer to
lockdown1 = [sim.day('2020-03-13'),sim.day('2020-04-15')]
lockdown2 = [sim.day('2020-12-09'),sim.day('2021-02-07')]

# Extract weekly infection data
w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 = cv.date('2021-02-07'), cv.date('2021-02-14'), cv.date('2021-02-21'), cv.date('2021-02-28'), \
                                 cv.date('2021-03-07'), cv.date('2021-03-14'), cv.date('2021-03-21'), cv.date('2021-03-28'), \
                                 cv.date('2021-04-04'), cv.date('2021-04-11'), cv.date('2021-04-18'), cv.date('2021-04-25')
wd = [sim.day(w0), sim.day(w1), sim.day(w2), sim.day(w3), sim.day(w4), sim.day(w5), sim.day(w6), sim.day(w7), sim.day(w8), sim.day(w9), sim.day(w10), sim.day(w11)]
inf_med = []
inf_low = []
inf_high = []
for scen in scenarios:
    inf_med.append(msims[scen].results['new_infections'].values[wd])
    inf_low.append(msims[scen].results['new_infections'].low[wd])
    inf_high.append(msims[scen].results['new_infections'].high[wd])

epsx = 0.003
llpad = 0.01

for nc in range(ncols):
    pl.figtext(xgapl + (dx + xgapm) * nc + epsx, ygapb + dy * nrows + ygapm * (nrows - 1) + llpad, labels[nc],
           fontsize=36, fontweight='bold', bbox={'edgecolor': 'none', 'facecolor': 'white', 'alpha': 0.5, 'pad': 4})

for pn in range(nplots):
    ax[pn] = pl.axes([xgapl + (dx + xgapm) * (pn % ncols), ygapb + (ygapm + dy) * (pn // ncols), dx, dy])
    print([xgapl + (dx + xgapm) * (pn % ncols), ygapb + (ygapm + dy) * (pn // ncols)])
    print(list(sims.keys())[pn % ncols])
    format_ax(ax[pn], sim)

    if (pn%ncols) != 0:
        ax[pn].set_yticklabels([])
    else:
        ax[pn].set_ylabel('New infections')

    if pn in range(ncols):
        plotter('r_eff', sims[pn % ncols], ax[pn])
        ax[pn].set_ylim(0, 3.5)
        ax[pn].axhline(y=1, color='red', linestyle='--')
        if (pn%ncols) == 0:
            ax[pn].set_ylabel('R')
    elif pn in range(ncols,ncols*2):
        plotter('cum_deaths', sims[pn % ncols], ax[pn])
        ax[pn].set_ylim(0, 2_500)
        if (pn%ncols) == 0:
            ax[pn].set_ylabel('Total deaths')
    else:
        plotter('new_infections', sims[pn % ncols], ax[pn])
        ax[pn].set_ylim(0, 12_000)
        if (pn%ncols) == 0:
            ax[pn].set_ylabel('New infections')

    if pn not in range(ncols):
        ax[pn].set_xticklabels([])

cv.savefig(f'{figsfolder}/fig_DK_scens.png', dpi=100)


################################################################################
# ## Fig 3
################################################################################
pl.figure(figsize=(24, 12))
#font_size = 24
#pl.rcParams['font.size'] = font_size

# Subplot sizes
xgapl = 0.07
xgapm = 0.1
xgapr = 0.01
ygapb = 0.11
ygapm = 0.1
ygapt = 0.02
nrows = 1
ncols = 2
dx1 = (1-(ncols-1)*xgapm-xgapl-xgapr)*0.6
dx2 = (1-(ncols-1)*xgapm-xgapl-xgapr)*0.4
dy = (1-(nrows-1)*ygapm-ygapb-ygapt)/nrows
nplots = nrows*ncols

colors = pl.cm.GnBu(np.array([0.3,0.65,1.]))

# Fig 3A. box plot chart
box_ax = pl.axes([xgapl, ygapb, dx1, dy])
x = np.arange(12)

for sn in range(3):
    box_ax.errorbar(x+0.1*sn-0.3, inf_med[sn], yerr=[inf_low[sn], inf_high[sn]], fmt='o', color=colors[sn], label=labels[sn], ecolor=colors[sn], ms=20, elinewidth=3, capsize=0)

box_ax.set_xticks(x-0.15)
#box_ax.set_xticklabels(labels)

@ticker.FuncFormatter
def date_formatter(x, pos):
    return (cv.date('2021-02-09') + dt.timedelta(days=x*7)).strftime('%b\n%d')

box_ax.xaxis.set_major_formatter(date_formatter)
pl.ylabel('Estimated daily infections')
sc.boxoff(ax=box_ax)
sc.commaticks()
box_ax.legend(frameon=False)


# B. Cumulative total infections
width = 0.8  # the width of the bars
x = [0,1,2]
data = np.array([msims[sn].results['cum_infections'].values[-1]-msims[sn].results['cum_infections'].values[sim.day('2021-02-07')] for sn in scenarios])
bar_ax = pl.axes([xgapl+xgapm+dx1, ygapb, dx2, dy])
for sn,scen in enumerate(scenarios):
    bar_ax.bar(x[sn], data[sn]/1e3, width, color=colors[sn], alpha=1.0)

scenarios = ['lift07', 'lift21', 'phased']
labels = ['Reopen Feb 8', 'Reopen Feb 23', 'Phased reopening from Feb 23']

bar_ax.set_xticklabels(['', 'Reopen\nFeb 8', 'Reopen\nFeb 23', 'Phased\nFeb 23'])
sc.boxoff()
sc.commaticks()
bar_ax.set_ylabel('Total estimated infections\nFeb 8 - April 1 (000s)')

cv.savefig(f'{figsfolder}/fig_bars.png', dpi=100)




sc.toc(T)