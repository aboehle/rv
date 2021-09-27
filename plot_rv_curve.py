import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits,ascii
import os
import glob


def calc_rv_curve(t, m_p, m_star, a_p, e, w, i, t0):
    """
    Calculate RV curve given an array of times t (in JD).

    :param t: array of times at which to calculate RV (in JD)
    :param float m_p: in jupiter masses
    :param float m_star: in solar masses
    :param a: semi-major axis of planet orbit in AU
    :param e: eccentricity
    :param w: argument of periapse in degrees
    :param i: inclination in degrees
    :param t0: time of periastron passage in JD

    :return rv_t: rv at the inputted times
    """

    # convert a_p to a and then a_p to p, angles to radians
    a = ( ((m_p*954.7919e-6) + m_star) / m_star )*a_p
    print(a, a_p)
    p = np.sqrt(a**3./(m_star+m_p*954.7919e-6))*365.
    print(p)
    w = w*(np.pi/180.)
    i = i*(np.pi/180.)

    f = radvel.orbit.true_anomaly(t, t0, p, e)
    f_mat = np.transpose(f)

    K = 28.4329*m_p*np.sin(i)*( (m_p*954.7919e-6)+m_star)**-0.5*(a**-0.5)/(np.sqrt(1-e**2.))
    print(K)

    # make matrix with width equal to len(t) and height equal to len(K) - i.e., number of different orbital parameter sets.  each row has a value K.
    #K_mat = np.transpose(np.matrix(K))
    #print K_mat
    #K_mat = np.repeat(K,len(t),axis=1)
    
    rv_f = lambda f:K*(np.cos(w + f) + e*np.cos(w))
    rv_t = rv_f(f)

    return rv_t


def plot_rv_curve(m_p, m_star, a, e, w, i):
    """
    Plot example RV curve given set of orbital parameters.  Plot the curve as RV versus time over 1 orbital period.

    :param float m_p: in jupiter masses
    :param float m_star: in solar masses
    :param a: semi-major axis in AU
    :param e: eccentricity
    :param w: argument of periapse in degrees
    :param i: inclination in degrees
    """

    # convert a to p
    p = np.sqrt(a**3./m_star)*365.
    
    t0 = 2458196.0  # arbitrary t0 for plotting
    # t = np.linspace(t0-p/2.,t0+p/2.,10000)
    t = np.arange(t0-p/2.,t0+p/2.,0.1)

    rv_t = calc_rv_curve(t, m_p, m_star, a, e, w, i, t0)

    plt.plot((t-t0)/365., rv_t)
    plt.xlabel('Time from To (years)')
    plt.ylabel('RV (m/s)')
    

def get_max_delv(tau, m_p, m_star, a, e, w, i, plot=False):

    """
    Get the maximum delta radial velocity over a given time baseline.
    Checks the time baseline over the entire orbit.

    :param float tau: time baseline in days
    :param float m_p: in jupiter masses
    :param float m_star: in solar masses
    :param a: semi-major axis in AU
    :param e: eccentricity
    :param w: argument of periapse in degrees
    :param i: inclination in degrees

    :return max_delv: maximum RV change within the given time baseline
    """

    # convert a to p
    p = np.sqrt(a**3./m_star)*365.
    
    
    t0 = 2458196.0  # arbitrary t0
    #t = np.linspace(t0-p,t0,10000)  # times spanning orbital period
    #t = np.arange(t0-p,t0,0.1)
    t = np.arange(t0-p,t0+p+0.001,0.001)
    idx_tau = np.argmin(np.abs( (t-t[0])-tau))

    #rv_oneperiod = calc_rv_curve(t, m_p, m_star, a, e, w, i, t0)
    #rv_t = np.concatenate((rv_oneperiod,rv_oneperiod))

    rv_t = calc_rv_curve(t, m_p, m_star, a, e, w, i, t0)

    if tau > p:  # if full period is covered by time baseline
        K = 28.4329*m_p*np.sin(i)*( (m_p*954.7919e-6)+m_star)**0.5*(a**-0.5)/(np.sqrt(1-e**2.))
        max_delv = 2*K
        
    else:
        delv = np.zeros(len(t)/2)
        for i in range(len(t)/2):  # search over a whole orbital period
            rv_subset = rv_t[i:(i+idx_tau)]  # take subset starting from i to i+idx_tau
            delv[i] = np.max(rv_subset) - np.min(rv_subset)
    
        max_delv = np.max(delv)
        max_window = [t[np.argmax(delv)],t[(np.argmax(delv)+idx_tau)]]

        if plot:
            #plt.plot(np.concatenate( (np.linspace(t0-p,t0,10000),np.linspace(t0,t0+p,10000)))-t0,rv_t)
            #plt.plot(np.concatenate( (np.arange(t0-p,t0,0.1),np.arange(t0,t0+p,0.1)))-t0,rv_t,marker='.')
            plt.plot(t-t0,rv_t,marker='.')
            #plt.plot(t-t0, rv_oneperiod)
            plt.axvline(max_window[0]-t0)
            plt.axvline(max_window[1]-t0)    
            plt.xlabel('Time from T0 (days)')
            plt.ylabel('RV (m/s)')
            plt.xlim(-1,1)
            plt.ylim(-1,1)

    return max_delv


def get_delv_timewindow(tau, m_p, m_star, a, e, w, i, t0, start_time, plot=False):
    """
    Get the maximum delta radial velocity over a given time baseline (tau),
    starting at start_time.

    :param float tau: time baseline in days
    :param float m_p: in jupiter masses
    :param float m_star: in solar masses
    :param a: semi-major axis in AU
    :param e: eccentricity
    :param w: argument of periapse in degrees
    :param i: inclination in degrees
    :param t0: time of periastron passage in JD
    :param start_time: start time of the time window over which to determine max_delv

    :return: max_delv: maximum delta RV
    """

    # convert a to p
    p = np.sqrt(a**3./m_star)*365.

    t = np.arange(start_time,start_time+tau+0.01,0.01)

    rv_t = calc_rv_curve(t, m_p, m_star, a, e, w, i, t0)

    max_delv = np.max(rv_t) - np.min(rv_t)
    #max_delv_times = [t[np.argmin(rv_t)],t[np.argmax(rv_t)]]

    if plot:
        plt.plot(t, rv_t)
        plt.axhline(np.max(rv_t))
        plt.axhline(np.min(rv_t))
        plt.show()
    
    return max_delv #, max_delv_times


def get_mlimit_a(t_baseline, rv_std, m_star, a=10, plot=False, sample_e=True):
    """
    Get the mass limit for given semi-major axis for an RV data set with
    time baseline t_baseline and standard deviation rv_std.
    The mass limit corresponds to the mass that sets the maximum delta RV in the inputted time baseline
    equal to the 1 times the rv standard deviation.
    This is for cases that the orbital periods are too long compared to the time baseline
    for the signal to be periodic in the RV data.

    Inclination is fixed to 90 deg so the output is minimum mass limit.
    Other orbital parameters are sampled randomly (w,
    e if sample_e = True, phase of the orbit via start_time).

    :param t_baseline: time baseline of RV data set in days
    :param rv_std: standard deviation of RV data set (in m/s)
    :param m_star: mass of star in solar masses
    :param a: semi-major axis of interest in AU
    :param plot: if True, then plot the results
    :param sample_e: if True, then sample the eccentricity orbital parameter
    per the distribution in Kipping et al. from RV planet population; otherwise e = 0

    :return: (maximum delta vels, array of limiting planet masses for all orbits,
             w distribution, e distribution, average of limiting mass)
    """

    #start_time = time.time()

    # convert a to p
    p = np.sqrt(a**3./m_star)*365.  # days
    
    # sample orbital parameters
    num_samples = 10000

    # sample w
    w = np.random.uniform(low=0,high=360.,size=num_samples)

    # sample e
    if sample_e:
        a_beta, b_beta = 0.867,3.03  # Kipping et al.
        e = np.random.beta(a=a_beta,b=b_beta,size=num_samples)
    else:
        e = np.zeros(num_samples)

    # sample start time
    t0 = 2458196.0 # arbitrary
    start_time = np.random.uniform(low=t0-p/2.,high=t0+p/2.,size=num_samples)

    # get max delta v for m_p sin(i) = 1 (m_p = 1, i = 90)
    max_delvs = np.zeros(num_samples)    
    for o in range(num_samples):
        max_delvs[o] = get_delv_timewindow(t_baseline, 1, m_star, a, e[o], w[o], i=90, t0=t0, start_time=start_time[o])

    # get max delta v for m_p sin(i) = 1 (m_p = 1, i = 90)
    #max_delvs = np.zeros(num_samples)
    #for o in range(num_samples):

    #    max_delvs[o] = get_max_delv(t_baseline, 1, m_star, a, e[o], w[o], i=90)

    #end_time = time.time()

    #print end_time-start_time,'sec'
        
    # get scale factor that sets max_delvs equal to 5*rv_std - this is (roughly only) the corresponding m_p
    scale_fac = (5*rv_std)/max_delvs

    if plot:
        plt.hist(scale_fac)
        plt.show()

    # convert velocity scale factor to corresponding planet mass
    m_p_lim = get_mplim_from_scalefac(scale_fac,m_star)

    return max_delvs, m_p_lim, w, e, start_time, np.average(m_p_lim)


def get_mlimit_curve(star, a_arr = [], save_indv=True, save_avg=False, sample_e=True,
                     rvinfo_filename = '/Users/annaboehle/research/proposals/jwst_cycle1/targets/nearest_solar_dist_age.txt',
                     outdir='/Users/annaboehle/research/data/rv/mlimits_tbaseline/', file_start_idx=0):
    """

    Get RV mass limit for a range of semi-major axes.

    :param star: name of the star (whose details are in the rvinfo_filename file)
    :param a_arr: array of semi-major axes in AU
    :param save_indv: if True, individual mass limits for the sampled orbital parameters are saved for every semi-major axis (in a separate file)
    :param save_avg: if True, the average mass limit over the sampled orbital parameters are saved for each semi-major axis
    :param sample_e: if True, then sample the eccentricity orbital parameter
    :param rvinfo_filename: file with star names and RV info (time baseline of RV observations, rv standard deviation, and mass of star)
    :param outdir: output directory for files
    :param file_start_idx: index of first individual output file (only used if save_indv = True)
    :return:
    """

    # values for tau ceti for testing
    #t_baseline = 6998  # days
    #rv_std = 1.73 # m/s
    #num_ameas = 1582
    #m_star = 0.82 # solar masses

    star_rvinfo = ascii.read(rvinfo_filename)
    row = np.where(star_rvinfo['object_name'] == star)[0][0]

    t_baseline = star_rvinfo['t_baseline'][row]
    rv_std = star_rvinfo['rv_std'][row]
    m_star = star_rvinfo['m_star'][row]

    # default array of semi-major axes
    if len(a_arr) == 0:
        a_baseline = ((t_baseline/365.)**2.*m_star)**(1/3.)

        a_arr = np.arange(a_baseline,a_baseline+14.,2.0)
        print(star, a_arr)

    # create output dirs, check if output file names exist
    if save_indv or save_avg:
        if sample_e:
            fileout = '{:s}{:s}_rvlim.txt'.format(outdir, star)
            outdir_indv = '{:s}/indv_samples/{:s}/'.format(outdir,star)

        else:
            fileout = '{:s}{:s}_rvlim_e0.txt'.format(outdir, star)                        
            outdir_indv = '{:s}/indv_samples_e0/{:s}/'.format(outdir,star)

        if save_indv:
            # create star directory inside indv_samples/ if it doesn't exist
            if not os.path.exists(outdir_indv):
                print('Creating directory {:s}'.format(outdir_indv))
                os.system('mkdir {:s}'.format(outdir_indv))
    
        if save_avg:
            # check if fileout exists; if so, ask if overwriting is ok
            if os.path.exists(fileout):
                ans = raw_input('Output file {:s} exists; overwrite? (y/[n])'.format(fileout))

                if ans != 'y':
                    print('Halting mass limit calculation.')
                    return            
            


    # loop over semi-major axes
    lim_mass = np.zeros(len(a_arr))
    for idx, a in enumerate(a_arr):
        max_delvs, m_p_lim, w, e, start_times, lim_mass[idx] = get_mlimit_a(t_baseline,rv_std, m_star, a, sample_e=sample_e)

        if save_indv:
            # write out individual results
            fileout_indv = '{:s}{:s}_delvs_limmasses_{:02d}.txt'.format(outdir_indv,star,idx+file_start_idx)
            np.savetxt(fileout_indv, np.column_stack( (max_delvs, m_p_lim, w, e, start_times)),
                        header='{:s}: separation = {:2.5f} AU, rv_std = {:2.2f}, t_baseline = {:2.2f}\nmax_delvs\tmasses\tw\te\tstart_times'.format(star, a, rv_std, t_baseline))

    if save_avg:
        # write out average results
        lim_mass = np.array([lim_mass])
        a_arr = np.array([a_arr])
        np.savetxt(fileout, np.concatenate( (a_arr.transpose(), lim_mass.transpose()), axis=1 ) )

    # plot results
    #plt.plot(a_arr, lim_mass)
    #plt.xlabel('Semi-major axis (AU)')
    #plt.ylabel('Limiting mass (M_J)')

    #if save:
    #    plt.savefig('{:s}{:s}_rvlim.png'.format(outdir, star))
    
        
def calc_mlimit(star,sample_e=True,sample_i=True,rvsigma=1.0):
    """
    Using the average and the individual mass limits from get_mlimit_curve,
    get the mass limits for a given confidence level (1, 2, or 3 sigma).

    :param star: star name
    :param sample_e: set True if this was true in other function
    :param sample_i: set True if this was true in other function
    :param rvsigma: 1, 2, or 3 sigma to set the confidence level
    :return:
    """

    # get average mass limits:
    rvtab = np.loadtxt('/Users/annaboehle/research/data/rv/mlimits_tbaseline/{:s}_rvlim.txt'.format(star))
    a_rv = rvtab[:,0]
    #mass_limits_rv = rvtab[:,1]

    mass_medians = np.zeros(len(a_rv))
    mass_mins = np.zeros(len(a_rv))
    mass_maxs = np.zeros(len(a_rv))
    # get mass limit samples for each separation
    for idx in range(len(a_rv)):
        if sample_e and sample_i:
            masstab = np.loadtxt('/Users/annaboehle/research/data/rv/mlimits_tbaseline/indv_samples_randomi/{:s}/{:s}_delvs_limmasses_{:02d}.txt'.format(star,star,idx))
            incl = masstab[:,5]
        elif sample_e and not sample_i:
            masstab = np.loadtxt('/Users/annaboehle/research/data/rv/mlimits_tbaseline/indv_samples/{:s}/{:s}_delvs_limmasses_{:02d}.txt'.format(star,star,idx))
        elif not sample_e and not sample_i:
            masstab = np.loadtxt('/Users/annaboehle/research/data/rv/mlimits_tbaseline/indv_samples_e0/{:s}/{:s}_delvs_limmasses_{:02d}.txt'.format(star,star,idx))
        else:
            print('No random incl for e = 0 yet.')
            return
                
        masses = masstab[:,1]
        num_samples = len(masses)
        if sample_e and sample_i:
            masses /= np.sin(incl)

        # get median and cl's
        if rvsigma == 1:
            cl = 0.6827  
        elif rvsigma == 2:
            cl = 0.9545
        elif rvsigma == 3:
            cl = 0.9974
        else:
            print('Set rvsigma = 1, 2, or 3!')
        mass_medians[idx] = np.median(masses)
        masses_sorted = np.sort(masses)
        med_idx = np.argmin(np.abs(masses_sorted - mass_medians[idx]))
        num_cl = int(np.ceil(num_samples*cl/2.))

        mass_mins[idx] = masses_sorted[med_idx-num_cl]
        mass_maxs[idx] = masses_sorted[med_idx+num_cl]

    return a_rv, mass_medians, mass_mins, mass_maxs


def plot_rv_mlimit(star, sample_e=True, rvsigma=1.0, save=False, filename='',
                   rvinfo_filename = '/Users/annaboehle/research/proposals/jwst_cycle1/targets/nearest_solar_dist_age.txt'):

    a_rv, mass_medians, mass_mins, mass_maxs = calc_mlimit(star, sample_e, rvsigma)

    star_rvinfo = ascii.read(rvinfo_filename)
    row = np.where(star_rvinfo['object_name'] == star)[0][0]

    dist = star_rvinfo['dist'][row]
    
    star_name = star.split('_')
    star_name[1] = star_name[1].capitalize()
    star_name = ' '.join(star_name)
    
    plt.figure()
    ax_AU = plt.subplot(111)
    ax_arc = ax_AU.twiny()
    
    ax_AU.fill_between(a_rv,mass_mins,mass_maxs,interpolate=True,color='gray')
    ax_AU.plot(a_rv,mass_medians, linestyle='dotted', color='black',label=star_name)

    ax_arc.plot(a_rv/dist,mass_medians, linestyle='dotted', color='black')

    ax_AU.set_xlabel('Semi-major axis (AU)')
    ax_AU.set_ylabel('Minimum mass (M_J)')
    
    ax_arc.set_xlabel('Semi-major axis (arcsec)')

    ax_AU.legend(loc='upper left')

    if save:
        if not filename:
            filename = '{:s}_rvlim'
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.eps')        


def add_random_i(star,sample_e=True,save=True):
    '''
    Add column of randomly determined inclinations to convert minimum mass distribution to physical mass distribution and then resave table to a new directory.
    '''

    outDir = '/Users/annaboehle/research/data/rv/mlimits_tbaseline/indv_samples_randomi/'
    
    # read in current files
    if sample_e:
        file_ls_orig = glob.glob('/Users/annaboehle/research/data/rv/mlimits_tbaseline/indv_samples/{:s}/{:s}_delvs_limmasses_*.txt'.format(star,star))
    else:
        file_ls_orig = glob.glob('/Users/annaboehle/research/data/rv/mlimits_tbaseline/indv_samples_e0/{:s}/{:s}_delvs_limmasses_*.txt'.format(star,star))
        
    for filename in file_ls_orig:
        masstab_orig = np.loadtxt(filename)
        masstab_wincl = np.zeros( (masstab_orig.shape[0], masstab_orig.shape[1] + 1))
        
        masstab_wincl[:,0:masstab_orig.shape[1]] = masstab_orig
            
        unif = np.random.uniform(size=masstab_orig.shape[0])
        i = np.arccos(unif)

        masstab_wincl[:, masstab_orig.shape[1]] = i
            
        if save:
            # get header
            masstab_f = open(filename)
            header1 = masstab_f.readline()
            header2 = masstab_f.readline()
            masstab_f.close()

            header2_new = header2.rstrip('\n') + '\tinclination (radians)'
                
            # write out new table
            print('Saving new table to:',outDir+'/{:s}/{:s}'.format(star,filename.split('/')[-1]))
            np.savetxt(outDir+'/{:s}/{:s}'.format(star,filename.split('/')[-1]), masstab_wincl,header=header1+header2_new)
            

def add_seps(star,max_a=10.,outdir='/Users/annaboehle/research/data/rv/mlimits_tbaseline/',sample_e = True,
             get_mlimit=True, add_to_avgfile=True):

    if sample_e:
        fileout = '{:s}{:s}_rvlim.txt'.format(outdir, star)
        outdir_indv = '{:s}/indv_samples/{:s}/'.format(outdir,star)

    else:
        fileout = '{:s}{:s}_rvlim_e0.txt'.format(outdir, star)                        
        outdir_indv = '{:s}/indv_samples_e0/{:s}/'.format(outdir,star)

    # get star info
    rvinfo_filename = '/Users/annaboehle/research/proposals/jwst_cycle1/targets/nearest_solar_dist_age.txt'
    star_rvinfo = ascii.read(rvinfo_filename)
    row = np.where(star_rvinfo['object_name'] == star)[0][0]

    t_baseline = star_rvinfo['t_baseline'][row]
    rv_std = star_rvinfo['rv_std'][row]
    m_star = star_rvinfo['m_star'][row]

    dist = star_rvinfo['dist'][row]

    # get current max AU
    fileout_tab = np.loadtxt(fileout)
    a_arr_old = fileout_tab[:,0]
    max_a_orig = a_arr_old.max()
    delta_a = a_arr_old[1] - a_arr_old[0]
    
    a_arr=np.arange(max_a_orig+delta_a,dist*max_a+delta_a,delta_a)
    print(a_arr,'AU')

    # get file_start_idx
    indv_file_ls = glob.glob('{:s}/*txt'.format(outdir_indv))
    file_start_idx = len(indv_file_ls)
    print(file_start_idx)

    if get_mlimit:
        get_mlimit_curve(star, a_arr = a_arr, save_indv=True, save_avg=False, sample_e=sample_e,
                         rvinfo_filename = '/Users/annaboehle/research/proposals/jwst_cycle1/targets/nearest_solar_dist_age.txt',
                         outdir='/Users/annaboehle/research/data/rv/mlimits_tbaseline/', file_start_idx=file_start_idx)

    if add_to_avgfile:
        fileout_tab_new = np.zeros((fileout_tab.shape[0] + len(a_arr),fileout_tab.shape[1]))
        fileout_tab_new[0:fileout_tab.shape[0],:] = fileout_tab

        fileout_tab_new[fileout_tab.shape[0]:,0] = a_arr
        fileout_tab_new[fileout_tab.shape[0]:,1] = np.zeros(len(a_arr))

        np.savetxt(fileout,fileout_tab_new)


def get_mlimit_curve_allstarttimes(star, a_arr = [], rvinfo_filename = '/Users/annaboehle/research/proposals/jwst_cycle1/targets/nearest_solar_dist_age.txt',
                                   outdir='/Users/annaboehle/research/data/rv/mlimits_tbaseline/'):

    # setup table
    tab = np.zeros((len(a_arr),2))
    
    
    # get star info
    star_rvinfo = ascii.read(rvinfo_filename)
    row = np.where(star_rvinfo['object_name'] == star)[0][0]

    t_baseline = star_rvinfo['t_baseline'][row]
    rv_std = star_rvinfo['rv_std'][row]
    m_star = star_rvinfo['m_star'][row]
    
    for idx,a in enumerate(a_arr):
        m_limit = get_mlimit_a_allstarttimes(t_baseline, rv_std, m_star, a)

        tab[idx,0] = a
        tab[idx,1] = m_limit

    np.savetxt('{:s}{:s}_rvlim_allstarttimes.txt'.format(outdir,star), tab)
    
        
def get_mlimit_a_allstarttimes(t_baseline, rv_std, m_star, a=10, plot=False):
    '''
    Get the limiting minimum mass by taking regular intervals of the start time, 
    and scaling the mass until RV curve max velocity is 5-sigma about the RMS for ALL start times.
    This function assumes circular orbits (e = 0).
    '''
    
     # convert a to p
    p = np.sqrt(a**3./m_star)*365.  # days

    # check only for circular orbits
    e = 0
    
    # sample start time
    t0 = 2458196.0 # arbitrary
    start_time_range = np.linspace(t0-p/2., t0+p/2., 12)   # 12 evenly-spaced start times

    # get max delta v for m_p sin(i) = 1 (m_p = 1, i = 90)
    max_delvs = np.zeros(len(start_time_range))
    max_delvs_orig = np.zeros(len(start_time_range))        
    for o in range(len(max_delvs)):
        # fix e, i, w (w could also be other values, I don't think it matters)    
        max_delvs_orig[o] = get_delv_timewindow(t_baseline, 1.0, m_star, a, e=e, w=90, i=90, t0=t0, start_time=start_time_range[o],plot=False)
        
    # get scale factors that sets max_delvs equal to 5*rv_std - this is the corresponding m_p (exact for e = 0)
    scale_fac_orig = (5*rv_std)/max_delvs_orig

    m_p_lim = get_mplim_from_scalefac(scale_fac_orig,m_star)
    
    if plot:
        fig = plt.figure(figsize=(10,8))        
        for o in range(len(max_delvs)):
            
            plt.subplot(211)
            
            #plot_rv_curve(scale_fac[o],m_star,a,0,0,90)
            plt.axhline(5*rv_std,linestyle='dotted',color='black')
            plt.axhline(0,linestyle='dotted',color='black')
            #plt.axvline( (start_time_range[o]-t0)/365.)
            #plt.axvline((start_time_range[o]-t0+t_baseline)/365.)

            t = np.arange(start_time_range[o],start_time_range[o]+t_baseline+0.1,0.1)
            #rv_t = calc_rv_curve(t, scale_fac_orig[o], m_star, a, 0, 90, 90, t0)
            rv_t = calc_rv_curve(t, m_p_lim[o], m_star, a, e, 90, 90, t0)
            plt.plot(t-start_time_range[o],rv_t-rv_t.min())

            plt.subplot(212)
            
            #rv_t = calc_rv_curve(t, np.max(scale_fac_orig), m_star, a, 0, 90, 90, t0)
            rv_t = calc_rv_curve(t, np.max(m_p_lim), m_star, a, e, 90, 90, t0)            
            plt.plot(t-start_time_range[o],rv_t-rv_t.min())
            
            plt.axhline(5*rv_std,linestyle='dotted',color='black')
            plt.axhline(0,linestyle='dotted',color='black')

    return np.max(m_p_lim) #np.max(scale_fac)


def get_mplim_from_scalefac(scale_fac,m_star):
    
    mS_over_mJ = 954.7919e-6
    
    # get limiting mass planet from scale factor
    A = 1.0
    B = -(mS_over_mJ*scale_fac**2.)/(m_star+mS_over_mJ)
    C = -(m_star*scale_fac**2.)/(m_star+mS_over_mJ)
    
    m_p_lim = (-B + np.sqrt(B**2 - 4*A*C))/(2.*A)

    return m_p_lim
