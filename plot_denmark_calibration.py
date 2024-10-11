import covasim as cv
import pandas as pd
import sciris as sc
import pylab as pl
import numpy as np
from matplotlib import ticker
import datetime as dt
import matplotlib.patches as patches

# Filepaths
figsfolder = 'figs'
simsfilepath = 'results/denmark_sim.obj'
calibration_end = '2021-01-03'
addhist=True
T = sc.tic()

# Import files
simsfile = sc.loadobj(simsfilepath)

# Define plotting functions
#%% Helper functions

def format_ax(ax, sim, key=None):
    @ticker.FuncFormatter
    def date_formatter(x, pos):
        return (sim['start_day'] + dt.timedelta(days=x)).strftime('%b')
    ax.xaxis.set_major_formatter(date_formatter)
    pl.xlim([0, sim.day(calibration_end)+20])
    sc.boxoff()
    return

def plotter(key, sims, ax, ys=None, calib=False, label='', ylabel='', low_q=0.025, high_q=0.975, smooth=False, flabel=True, startday=None, subsample=2, chooseseed=None):

    which = key.split('_')[1]
    try:
        color = cv.get_colors()[which]
    except:
        color = [0.5,0.5,0.5]
    if which == 'diagnoses':
        color = [0.03137255, 0.37401   , 0.63813918, 1.        ]
    elif which == '':
        color = [0.82400815, 0.        , 0.        , 1.        ]

    if ys is None:
        ys = []
        for s in sims:
            ys.append(s.results[key].values)

    yarr = np.array(ys)
    if chooseseed is not None:
        best = sims[chooseseed].results[key].values
    else:
        best = pl.median(yarr, axis=0)
    low  = pl.quantile(yarr, q=low_q, axis=0)
    high = pl.quantile(yarr, q=high_q, axis=0)

    sim = sims[0] # For having a sim to refer to

    start = 0
    if startday is not None:
        start = sim.day(startday)
    end = sim.day(calibration_end)

    tvec = np.arange(len(best))
    if key in sim.data:
        data_t = np.array((sim.data.index-sim['start_day'])/np.timedelta64(1,'D'))
        inds = np.arange(start, len(data_t), subsample)
        pl.plot(data_t[inds], sim.data[key][inds], 'd', c=color, markersize=15, alpha=0.75, label='Data')

    if flabel:
        if which == 'infections':
            fill_label = '95% projected interval'
        else:
            fill_label = '95% projected interval'
    else:
        fill_label = None
    pl.fill_between(tvec[start:end], low[start:end], high[start:end], facecolor=color, alpha=0.2, label=fill_label)
    pl.plot(tvec[start:end], best[start:end], c=color, label=label, lw=4, alpha=1.0)

    sc.setylim()

    xmin,xmax = ax.get_xlim()
    if calib:
        ax.set_xticks(pl.arange(xmin+2, xmax, 28))
    else:
        ax.set_xticks(pl.arange(xmin+2, xmax, 28))

    pl.ylabel(ylabel)
    #datemarks = pl.array([sim.day('2020-07-01'), sim.day('2020-08-01'), sim.day('2020-09-01'), sim.day('2020-10-01')]) * 1.
    #ax.set_xticks(datemarks)

    return



# Fonts and sizes
font_size = 36
font_family = 'Libertinus Sans'
pl.rcParams['font.size'] = font_size
pl.rcParams['font.family'] = font_family
pl.figure(figsize=(24,16))

# Extract a sim to refer to
sims = simsfile.sims
sim = sims[0]

# Plot locations
ygapb = 0.05
ygapm = 0.05
ygapt = 0.01
xgapl = 0.065
xgapm = 0.05
xgapr = 0.02
remainingy = 1-(ygapb+ygapm+ygapt)
remainingx = 1-(xgapl+xgapm+xgapr)
dy = remainingy/2
dx1 = 0.5
dx2 = 1-dx1-(xgapl+xgapm+xgapr)
ax = {}

# a: Cumulative diagnoses and age histograms
ax[0] = pl.axes([xgapl, ygapb+ygapm+dy, dx1, dy])
format_ax(ax[0], sim)
plotter('cum_diagnoses', sims, ax[0], calib=True, label='Model', ylabel='Cumulative diagnoses')
#plot_intervs(sim)

# Add histogram
if addhist:
    agehists = []

    for s,sim in enumerate(sims):
        agehist = sim['analyzers'][0]
        if s == 0:
            age_data = agehist.data
        agehists.append(agehist.hists[-1])
    raw_x = age_data['age'].values
    raw_pos = age_data['cum_diagnoses'].values

    x = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"] #["0-29", "30-54", "55+"]
    pos = raw_pos #[raw_pos[0:3].sum(), raw_pos[6:11].sum(), raw_pos[11:].sum()]

    # From the model
    mposlist = []
    for hists in agehists:
        mposlist.append(hists['diagnosed'])
    mposarr = np.array(mposlist)
    low_q = 0.1
    high_q = 0.9
    raw_mpbest = pl.mean(mposarr, axis=0)
    raw_mplow  = pl.quantile(mposarr, q=low_q, axis=0)
    raw_mphigh = pl.quantile(mposarr, q=high_q, axis=0)

    mpbest = raw_mpbest
    mplow = raw_mplow
    mphigh = raw_mphigh

#    mpbest = [raw_mpbest[0:6].sum(), raw_mpbest[6:11].sum(), raw_mpbest[11:].sum()]
#    mplow = [raw_mplow[0:6].sum(), raw_mplow[6:11].sum(), raw_mplow[11:].sum()]
#    mphigh = [raw_mphigh[0:6].sum(), raw_mphigh[6:11].sum(), raw_mphigh[11:].sum()]

    # Plotting
    w = 0.4
    off = .8
    #bins = x.tolist()

    ax1s = pl.axes([xgapl+0.2*dx1, ygapb+ygapm+1.5*dy, 0.25, 0.15])
    # ax = pl.subplot(4,2,7)
    c1 = [0.3,0.3,0.6]
    c2 = [0.6,0.7,0.9]
    X = np.arange(len(x))
    XX = X+w-off
    pl.bar(X, pos, width=w, label='Data', facecolor=c1)
    pl.bar(XX, mpbest, width=w, label='Model', facecolor=c2)
    #pl.bar(x-off,pos, width=w, label='Data', facecolor=c1)
    #pl.bar(xx, mpbest, width=w, label='Model', facecolor=c2)
    for i,ix in enumerate(XX):
        pl.plot([ix,ix], [mplow[i], mphigh[i]], c='k')
    ax1s.set_xticks((X+XX)/2)
    ax1s.set_xticklabels(x)
    pl.xlabel('Age')
    pl.ylabel('Cases')
    sc.boxoff(ax1s)
    pl.legend(frameon=False, bbox_to_anchor=(0.7,1.1))


# b. New diagnoses since Sep 15
ax[1] = pl.axes([xgapl+xgapm+dx1, ygapb+ygapm+dy, dx2, dy])
format_ax(ax[1], sim)
plotter('new_diagnoses', sims, ax[1], startday='2020-09-15', calib=True, label='Diagnoses\n(modeled)', ylabel='Cumulative diagnoses', flabel=False)
pl.legend(loc='upper left', frameon=False)
#pl.ylim([0, 10e3])

# c. cumulative and active infections
ax[2] = pl.axes([xgapl, ygapb, dx1, dy])
format_ax(ax[2], sim)
plotter('cum_infections', sims, ax[2], calib=True, label='Cumulative infections\n(modeled)', ylabel='', flabel=False)
plotter('n_infectious', sims, ax[2], calib=True, label='Active infections\n(modeled)', ylabel='Estimated infections', flabel=False)
pl.legend(loc='upper left', frameon=False)

# d. cumulative deaths
ax[3] = pl.axes([xgapl+xgapm+dx1, ygapb, dx2, dy])
format_ax(ax[3], sim)
plotter('cum_deaths', sims, ax[3], calib=True, label='Deaths\n(modeled)', ylabel='Cumulative deaths', flabel=False)
pl.legend(loc='upper left', frameon=False)
#pl.ylim([0, 10e3])

cv.savefig(f'{figsfolder}/fig1_calibration.png', dpi=100)

sc.toc(T)