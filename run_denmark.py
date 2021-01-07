'''
Denmark scenarios for evaluating testing and tracing
'''

import sciris as sc
import covasim as cv
import pylab as pl
import numpy as np
import matplotlib as mplt

########################################################################
# Settings and initialisation
########################################################################
# Check version
cv.check_version('2.0.0')
cv.git_info('covasim_version.json')

# Saving and plotting settings
do_plot = 1
do_save = 1
save_sim = 0
do_show = 0
verbose = 1
seed    = 1
n_runs = 200
to_plot = sc.objdict({
    'Cumulative diagnoses': ['cum_diagnoses'],
    'Cumulative infections': ['cum_infections'],
    'New diagnoses': ['new_diagnoses'],
    'New infections': ['new_infections'],
    'Cumulative admissions': ['cum_severe'],
    'Cumulative deaths': ['cum_deaths'],
})


# Define what to run
runoptions = ['quickfit', # Does a quick preliminary calibration. Quick to run, ~30s
              'fullfit',  # Searches over parameters and seeds (10,000 runs) and calculates the mismatch for each. Slow to run: ~1hr
              'finialisefit', # Filters the 10,000 runs from the previous step, selects the best-fitting ones, and runs these
              'scens', # Takes the best-fitting runs and projects these forward under different mask and TTI assumptions
              'tti_sweeps', # Sweeps over future testing/tracing values to create data for heatmaps
              ]
whattorun = runoptions[1] #Select which of the above to run

# Filepaths
data_path = 'dk_data.csv'
resfolder = 'results'

# Important dates
start_day = '2020-02-01'
end_day = '2021-03-31'
data_end = '2021-01-04' # Final date for calibration


########################################################################
# Create the baseline simulation
########################################################################

def make_sim(seed, beta, calibration=True, scenario=None, delta_beta=0.6, future_symp_test=None, future_t_eff=None, end_day=None, verbose=0):

    # Set the parameters
    total_pop    = 5.8e6 # Danish population size
    pop_size     = 100e3 # Actual simulated population
    pop_scale    = int(total_pop/pop_size)
    pop_type     = 'hybrid'
    pop_infected = 120
    beta         = beta
    if end_day is None: end_day = '2021-03-31'

    pars = sc.objdict(
        pop_size     = pop_size,
        pop_infected = pop_infected,
        pop_scale    = pop_scale,
        pop_type     = pop_type,
        start_day    = start_day,
        end_day      = end_day,
        beta         = beta,
        rescale      = True,
        rand_seed    = seed,
        verbose      = verbose,
        iso_factor   = dict(h=0.7, s=0.05, w=0.05, c=0.1),  # Multiply beta by this factor for people in isolation
        quar_factor  = dict(h=1.0, s=0.3, w=0.3, c=0.2),  # Multiply beta by this factor for people in quarantine
    #        rel_death_prob = 2.,  # Calibration parameter due to outbreaks in LTCF
    )

    sim = cv.Sim(pars=pars, datafile=data_path, location='denmark')

    # 1. Lockdowns and NPIs
    interventions = [
        cv.clip_edges(days=['2020-03-13', '2020-04-15', '2020-06-15', '2020-09-01', '2020-12-09', '2020-12-19'], changes=[0.05, 0.8, 0.1, 1.0, 0.5, 0.0], layers=['s']), # School closure and reopening
        cv.change_beta(['2020-03-13', '2020-09-01', '2020-10-29'], [0.5, 0.65, 0.6], layers=['s']), # Assume precautions in place after school returns

        cv.clip_edges(days=['2020-03-13', '2020-04-15', '2020-06-15', '2020-09-01', '2020-12-09', '2020-12-19'], changes=[0.1, 0.6, 0.5, 0.8, 0.5, 0.1], layers=['w']),
        cv.change_beta(['2020-03-13', '2020-09-01', '2020-10-29'], [0.5, 0.65, 0.6], layers=['w']),  # Assume precautions in place for workers

        cv.change_beta(['2020-03-13', '2020-09-01', '2020-10-29', '2020-12-09', '2020-12-19'], [0.4, 0.7, 0.65, 0.6, 0.2], layers=['c']), # Precautions in community lapsed after low case counts
    ]

    # 2. Testing assumptions
    interventions += [
#        cv.test_num(daily_tests=sim.data['new_tests'], start_day=0, end_day=sim.day(data_end), test_delay=1,
#                    symp_test=50.0, quar_test=50.),
        cv.test_prob(symp_prob=0.025, asymp_quar_prob=0.1, start_day=0, end_day='2020-04-30', test_delay=2),
        cv.test_prob(symp_prob=0.05, asymp_quar_prob=0.25, start_day='2020-05-01', end_day='2020-08-31', test_delay=1),
        cv.test_prob(symp_prob=0.10, asymp_quar_prob=0.35, start_day='2020-09-01', test_delay=1)]

    # 3. Assume some amount of contact tracing
    interventions += [cv.contact_tracing(start_day='2020-03-01',
                                         trace_probs={'h': 1, 's': 0.8, 'w': 0.5, 'c': 0.1},
                                         trace_time={'h': 0, 's': 1, 'w': 2, 'c': 7},
                                         quar_period=7)]

    # 4. Change death and critical probabilities
    interventions += [cv.dynamic_pars({'n_imports': {'days': [sim.day('2020-06-01'), sim.day('2020-12-09')], 'vals': [2, 0]},
                                       'rel_death_prob': {'days': [sim.day('2020-06-01')], 'vals': [0.5]},
                                       'rel_crit_prob': {'days': [sim.day('2020-06-01')], 'vals': [0.5]},
                                       'rel_severe_prob': {'days': [sim.day('2020-06-01')], 'vals': [0.5]}})
                      ]

    # 5. Add the new variant
    # Add a new change in beta to represent the takeover of the novel variant VOC 202012/01
    # Assume that the new variant is 60% more transmisible (https://cmmid.github.io/topics/covid19/uk-novel-variant.html,
    # Assume that between Nov 1 and Jan 30, the new variant grows from 0-100% of cases
    voc_days   = np.linspace(sim.day('2020-12-01'), sim.day('2020-12-01')+60, 31)
    voc_prop   = 1./(1+np.exp(-0.15*(voc_days-(sim.day('2020-12-01')+30)))) # Use a logistic growth function to approximate fig 2A of https://cmmid.github.io/topics/covid19/uk-novel-variant.html
    voc_change = voc_prop*(1+delta_beta) + (1-voc_prop)*1.
    interventions += [cv.change_beta(days=voc_days, changes=voc_change)]

    # Finally, update the parameters
    sim.update_pars(interventions=interventions)
    for intervention in sim['interventions']:
        intervention.do_plot = False

    sim.initialize()

    return sim


########################################################################
# Run calibration and scenarios
########################################################################
if __name__ == '__main__':

    betas = [i / 10000 for i in range(158, 163, 1)]

    # Quick calibration
    if whattorun=='quickfit':
        s0 = make_sim(seed=1, beta=0.016, end_day=data_end, verbose=0.1)
        sims = []
        for seed in range(6):
            sim = s0.copy()
            sim['rand_seed'] = seed
            sim.set_seed()
            sim.label = f"Sim {seed}"
            sims.append(sim)
        msim = cv.MultiSim(sims)
        msim.run()
        msim.reduce()
        if do_plot:
            msim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'denmark.png',
                      legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=50, n_cols=2)

    # Full parameter/seed search
    elif whattorun=='fullfit':
        fitsummary = []
        for beta in betas:
            sc.blank()
            print('---------------\n')
            print(f'Beta: {beta}... ')
            print('---------------\n')
            s0 = make_sim(seed=1, beta=beta, end_day=data_end)
            sims = []
            for seed in range(n_runs):
                sim = s0.copy()
                sim['rand_seed'] = seed
                sim.set_seed()
                sim.label = f"Sim {seed}"
                sims.append(sim)
            msim = cv.MultiSim(sims)
            msim.run()
            fitsummary.append([sim.compute_fit().mismatch for sim in msim.sims])

        sc.saveobj(f'{resfolder}/fitsummary.obj',fitsummary)

    # Run calibration with best-fitting seeds and parameters
    elif whattorun=='finialisefit':
        sims = []
        fitsummary = sc.loadobj(f'{resfolder}/fitsummary.obj')
        for bn, beta in enumerate(betas):
            goodseeds = [i for i in range(n_runs) if fitsummary[bn][i] < 300]
            sc.blank()
            print('---------------\n')
            print(f'Beta: {beta}, goodseeds: {len(goodseeds)}')
            print('---------------\n')
            if len(goodseeds) > 0:
                s0 = make_sim(seed=1, beta=beta, end_day=data_end)
                for seed in goodseeds:
                    sim = s0.copy()
                    sim['rand_seed'] = seed
                    sim.set_seed()
                    sim.label = f"Sim {seed}"
                    sims.append(sim)

        msim = cv.MultiSim(sims)
        msim.run()

        if save_sim:
            msim.save(f'{resfolder}/denmark_sim.obj')
        if do_plot:
            msim.reduce()
            msim.plot(to_plot=to_plot, do_save=do_save, do_show=False, fig_path=f'denmark.png',
                      legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=50, n_cols=2)


    # Run scenarios with best-fitting seeds and parameters
    elif whattorun=='scens':

        # Define scenario to run
        scenarios = sc.odict({'masks15': 0.15,
                              'masks30': 0.07,
                              'masks15_notschools': 0.17,
                              'masks30_notschools': 0.095})

        for scenname, future_symp_test in scenarios.iteritems():

            print('---------------\n')
            print(f'Beginning scenario: {scenname}')
            print('---------------\n')
            sc.blank()
            sims_cur, sims_opt = [], []
            fitsummary = sc.loadobj(f'{resfolder}/fitsummary.obj')

            for bn, beta in enumerate(betas):
                goodseeds = [i for i in range(n_runs) if fitsummary[bn][i] < 163]
                if len(goodseeds) > 0:
                    s_cur = make_sim(1, beta, calibration=False, scenario=scenname, future_symp_test=None, end_day='2021-12-31')
                    s_opt = make_sim(1, beta, calibration=False, scenario=scenname, future_symp_test=future_symp_test, end_day='2021-12-31')
                    for seed in goodseeds:
                        sim_cur = s_cur.copy()
                        sim_cur['rand_seed'] = seed
                        sim_cur.set_seed()
                        sim_cur.label = f"Sim {seed}"
                        sims_cur.append(sim_cur)
                        sim_opt = s_opt.copy()
                        sim_opt['rand_seed'] = seed
                        sim_opt.set_seed()
                        sim_opt.label = f"Sim {seed}"
                        sims_opt.append(sim_opt)

            msim_cur = cv.MultiSim(sims_cur)
            msim_cur.run()
            msim_opt = cv.MultiSim(sims_opt)
            msim_opt.run()

            if save_sim:
                msim_cur.save(f'{resfolder}/denmark_sim_{scenname}_current.obj')
                msim_opt.save(f'{resfolder}/denmark_sim_{scenname}_optimal.obj')
            if do_plot:
                msim_cur.reduce()
                msim_cur.plot(to_plot=to_plot, do_save=do_save, do_show=False, fig_path=f'denmark_{scenname}_current.png',
                          legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=50, n_cols=2)
                msim_cur.reduce()
                msim_cur.plot(to_plot=to_plot, do_save=do_save, do_show=False, fig_path=f'denmark_{scenname}_optimal.png',
                          legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=50, n_cols=2)

            print(f'... completed scenario: {scenname}')


    # Run scenarios with best-fitting seeds and parameters
    elif whattorun=='tti_sweeps':

        symp_test_vals = np.linspace(0, 1, 21)
        trace_eff_vals = np.linspace(0, 1, 21)
        scenarios = ['masks30_notschools','masks15','masks30','masks15_notschools']

        # Define scenario to run
        for scenname in scenarios:
            sweep_summary = {'cum_inf':[],'peak_inf':[],'cum_death':[]}
            for future_symp_test in symp_test_vals:
                daily_test = np.round(1 - (1 - future_symp_test) ** (1 / 10), 3) if future_symp_test<1 else 0.4
                cum_inf, peak_inf, cum_death = [], [], []
                for future_t_eff in trace_eff_vals:

                    sc.blank()
                    print('---------------')
                    print(f'Scenario: {scenname}, testing: {future_symp_test}, tracing: {future_t_eff}')
                    print('--------------- ')
                    sims = []
                    fitsummary = sc.loadobj(f'{resfolder}/fitsummary.obj')

                    for bn, beta in enumerate(betas):
                        goodseeds = [i for i in range(n_runs) if fitsummary[bn][i] < 125.5] # Take the best 10
                        if len(goodseeds) > 0:
                            s0 = make_sim(1, beta, calibration=False, scenario=scenname, future_symp_test=daily_test, future_t_eff=future_t_eff, end_day='2021-12-31')
                            for seed in goodseeds:
                                sim = s0.copy()
                                sim['rand_seed'] = seed
                                sim.set_seed()
                                sim.label = f"Sim {seed}"
                                sims.append(sim)

                    msim = cv.MultiSim(sims)
                    msim.run(verbose=-1)
                    msim.reduce()

                    # Store results
                    data_end_day = msim.sims[0].day(data_end)

                    cum_inf.append(msim.results['cum_infections'].values[-1])
                    peak_inf.append(max(msim.results['new_infections'].values[data_end_day:]))
                    cum_death.append(msim.results['cum_deaths'].values[-1])

                sweep_summary['cum_inf'].append(cum_inf)
                sweep_summary['peak_inf'].append(peak_inf)
                sweep_summary['cum_death'].append(cum_death)

            sc.saveobj(f'{resfolder}/denmark_tti_sweeps_{scenname}.obj', sweep_summary)
