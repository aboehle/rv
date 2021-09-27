import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle

from orbits import kepler_py3 as k


def generate_obs_times(dt=365/2.,
                       num_meas=25,
                       time_std=1.0,
                       plot=False):
    """
    Generate observing times that are roughly evenly spaced over a given time range.

    :param dt: time span of observations (days)
    :param num_meas: number of RV measurements over dT days
    :param time_std: std of random gaussian noise added to the regularly spaced observations
    :param plot: if True, plot test plots
    :return: array of observing times
    """

    # Get evenly spaced observing times
    obs_times = np.linspace(0,
                            dt,
                            num_meas)

    dt_reg = obs_times[1] - obs_times[0]

    # Add noise to observing times, keeping them within dt
    obs_times[0] += np.abs(np.random.normal(loc=0,
                                            scale=time_std,
                                            size=1))

    obs_times[-1] -= np.abs(np.random.normal(loc=0,
                                             scale=time_std,
                                             size=1))

    obs_times[1:-1] += np.random.normal(loc=0,
                                        scale=time_std,
                                        size=num_meas-2)  # add random noise

    if plot:
        print(f'regularly spaced measurements are {dt_reg:.3f} days apart for num_meas = {num_meas}')
        print(f'min_time = {obs_times.min():.3f}, max time = {obs_times.max():.3f}')

        plt.figure(figsize=(6,7))
        plt.subplot(211)
        for t in obs_times:
            plt.axvline(t)
        plt.xlabel('Observing times (days)')

        plt.subplot(212)
        plt.hist(obs_times[1:] - obs_times[0:-1], histtype='step')
        plt.axvline(dt_reg)
        plt.xlabel('Difference in subsequent observing times (days)')

    return obs_times

def generate_rv_data(obs_times,
                     rv_std):
    """
    Generate RV data for no planet signal
    at the inputted times and with the inputted noise.

    :param obs_times: times of RV measurements (days)
    :param rv_std: noise of the generated RV measurements (m/s)
    :return:
    """

    # ## Generate RV data
    rv_data = np.random.normal(loc=0,
                               scale=rv_std,
                               size=len(obs_times))

    return rv_data

def gls_fap(obs_times,
            rv_data,
            n_bootstrap=1000,
            f_data=[],
            min_f = None,
            max_f = None,
            samples_per_peak=1,
            plot=False):
    """
    Determine the 99% false alarm probability (FAP) for the inputted RV data set
    for periodic signals using the generalized Lomb-Scargle Periodogram.
    Any signals with power > the 99% FAP would be detected.

    :param obs_times: times of RV measurements (days)
    :param rv_data: RV measurements (m/s)
    :param n_bootstrap: number of bootstrap trials to determine 99% FAP
    :param f_data: values of frequencies to test (if not give, code determines these automatically)
    :param min_f: minimum frequency, otherwise code uses hardcoded value
    :param min_f: maximum frequency, otherwise code uses hardcoded value
    :param samples_per_peak: GLS parameter, number of points computed per peak in periodogram
    :param plot: if True, do diagnostic plots

    :return: (array of tested frequencies, array of FAP for 99%)
    """


    # ## Bootstrap RV data
    rv_bootstrap = np.random.choice(rv_data,
                                    replace=True,
                                    size=(n_bootstrap,len(obs_times)))

    # ## L-S periodogram

    # frequency range from Christophe's code
    if not min_f:
        min_f = 1/(4.0*(obs_times[-1]-obs_times[0]))
    if not max_f:
        max_f = 1/0.65

    if len(f_data) == 0:
        f_data,p_data=LombScargle(obs_times,
                                  rv_data).autopower(minimum_frequency=min_f,
                                                     maximum_frequency=max_f,
                                                     samples_per_peak=samples_per_peak)
    else:
        p_data=LombScargle(obs_times,
                           rv_data).power(f_data)

    p_bootstrap = np.zeros((n_bootstrap,
                            len(f_data)))

    for n in range(n_bootstrap):
        p_bootstrap[n] = LombScargle(obs_times,
                                     rv_bootstrap[n]).power(f_data)

    # ## Get 99% power for each frequency

    p_bootstrap_sorted = np.sort(p_bootstrap,
                                 axis=0)
    fap_99 = p_bootstrap_sorted[990-1,:]

    if plot:
        print(f'min period = {1/max_f:1.2f} days, max period = {1/min_f:1.2f} days ({1/min_f/365.:1.2f} years)')

        alpha_bootstrap = 0.02

        plt.figure(figsize=(12, 7))
        plt.subplot(221)
        plt.errorbar(obs_times, rv_data, yerr=np.std(rv_data),
                     marker='.', color='black', linestyle='none')
        plt.xlabel('Time (days)')
        plt.ylabel('RV (m/s)')

        plt.subplot(222)
        plt.plot(1 / f_data, p_data,
                 color='black')
        plt.gca().set_xscale('log')
        plt.xlabel('Period (days)')
        plt.ylabel('Power')

        #plt.figure(figsize=(6, 7))
        plt.subplot(223)
        for n in range(n_bootstrap):
            plt.plot(f_data,p_bootstrap[n],
                     color='gray',alpha=alpha_bootstrap)
        plt.plot(f_data,p_data,
                 color='black')
        plt.plot(f_data,fap_99,
                 linestyle='dotted',color='red')
        plt.xlabel('Frequency (1/days)')
        plt.ylabel('Power')

        plt.subplot(224)
        for n in range(n_bootstrap):
            if n == 0:
                plt.plot(1/f_data,p_bootstrap[n],
                         color='gray',alpha=alpha_bootstrap,label='GLS for bootstrap')
            else:
                plt.plot(1 / f_data, p_bootstrap[n],
                         color='gray', alpha=alpha_bootstrap)
        plt.plot(1/f_data,p_data,
                 color='black', label='GLS for RV data')
        plt.plot(1/f_data,fap_99,
                 linestyle='dotted',color='red',label='99% FAP')
        plt.gca().set_xscale('log')
        plt.xlabel('Period (days)')
        plt.ylabel('Power')
        plt.legend(loc='upper right')

    return f_data, fap_99


def mass_limits(obs_times,
                rv_data,
                f_data,
                fap_99,
                m_star,
                plot=False):
    """
    Get the mass limits for the RV data set given the previously computed 99% FAP.
    Loops over 12 evenly spaced orbital phases and returns the mass limits for all of them.
    You can chose to take the average or the maximum mass limit.

    The code assumes circular orbits (a common assumption for RV mass limits,
    from what I understand).

    :param obs_times: times of RV measurements (days)
    :param rv_data: RV measurements (m/s)
    :param f_data: array of tested frequencies (output of gls_fap)
    :param fap_99: array of 99% FAP
    :param m_star: mass of star in solar masses
    :param plot: if True, then the results are plotted
    :return:
    """

    f_idx_arr  = np.arange(0,len(f_data),dtype=int)

    w_arr = np.linspace(0, 360, 12, endpoint=False)

    k_fap_99 = np.zeros((len(f_idx_arr),len(w_arr)))

    # single freq test
    #f_idx = int(len(f_data)/50)

    for i, f_idx in enumerate(f_idx_arr):
        #print(i)
        #print(f'Period = {1/f_data[f_idx]:1.2f} days')

        # LS of data at this frequency
        ls_data = LombScargle(obs_times,rv_data)

        # get model parameter for this frequency
        # (offset, ampl of sin, ampl of cos)
        theta = ls_data.model_parameters(f_data[f_idx])

        # amplitude of sum of sin and cos with same freq,
        # minus 0.5 so it is definitely below 99% FAP
        #k_inject = np.sqrt(theta[1]**2. + theta[2]**2.)/2.

        #print(k_inject)
        #k_inject = 0.5

        # loop over orbital phase
        for j, w in enumerate(w_arr):
            k_inject = np.sqrt(theta[1] ** 2. + theta[2] ** 2.) / 2.

            # initial injection
            rv_sim = k.rv_curve(obs_times,
                                P=1/f_data[f_idx],
                                w=w,
                                K=k_inject,
                                e=0,  # circular only
                                t0=0)  # fixed since degenerate with w for e = 0

            rv_sim_noise = rv_sim + rv_data

            p_sim = LombScargle(obs_times,
                                rv_sim_noise).power(f_data[f_idx])

            if p_sim > fap_99[f_idx]:
                if plot:
                    print(f'warning for P = {1/f_data[f_idx]}, w = {w}: initial k_inject is too high')
                    #print(k_inject,p_sim)

                p_sim = -np.inf
                k_inject = 0.5

                #print()

            # find k_inject that is has least a power of FAP 99%
            while p_sim < fap_99[f_idx]:
                k_inject += 0.05

                rv_sim = k.rv_curve(obs_times,
                                    P=1/f_data[f_idx],
                                    w=w,
                                    K=k_inject,
                                    e=0,  # circular only
                                    t0=0) # fixed since degenerate with w for e = 0

                rv_sim_noise = rv_sim + rv_data

                p_sim = LombScargle(obs_times,
                                    rv_sim_noise).power(f_data[f_idx])
            #print(f'num_iter = {i}')

            k_fap_99[i,j] = k_inject

    mlimits_99 = (k_fap_99/28.4329) * (m_star ** (2 / 3.)) * (((1 / np.reshape(f_data, (len(f_data), 1))) / 365.) ** (1 / 3.))

    return(mlimits_99)
