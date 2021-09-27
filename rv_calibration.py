import numpy as np
import glob
import matplotlib.pyplot as plt
from astropy.io import ascii
import matplotlib as mpl
import os


def compute_acc(plx,mua,mud):                                      # [mas], [mas/yr]
    """
    Code to compute the secular acceleration from Christophe Lovis (via email on 27/5/19).

    This code should no longer be needed in the more recent versions of DACE,
    since the secular acceleration correction is applied automatically.
    But potentially worth double checking that the correction is applied correctly (lol)
    if you see a linear trend in the RV data.

    :param plx: parallax of star (milliarcsec)
    :param mua: proper motion in R.A. (milliarcsec/year)
    :param mud: proper motion in declination (milliarcsec/year)
    :return: secular acceleration (meters/sec/day)
    """


    d = 1000./plx*3.08567758e16                                     # [m]
    mu = np.sqrt(mua**2+mud**2)/1000./3600.*2*np.pi/360./86400./365.256   # [radians/s]
    return d*mu**2*86400.                                           # [m/s/day]


def calibrate_offsets_fulloverlap(star,save=False,plot=True):
    """
    Calibrates RV offsets between different instruments
    for cases where there is some overlap between the data sets.

    :param star: name of star from Boehle et al. 2019 paper sample
    :param save: if True, write out the calibrated data sets
    :param plot: if True, plot the calibration process
    :return: None
    """

    if plot:
        inst_params = {'COR98': [plt.rcParams['axes.prop_cycle'].by_key()['color'][3], '^', 5.0],
                       'COR07': [plt.rcParams['axes.prop_cycle'].by_key()['color'][9], '<', 8.0],
                       'COR14': [plt.rcParams['axes.prop_cycle'].by_key()['color'][7], '>', 3.0],
                       'HARPS03': [plt.rcParams['axes.prop_cycle'].by_key()['color'][0], 'o', 0.75],
                       'HARPS15': [plt.rcParams['axes.prop_cycle'].by_key()['color'][1], 'D', 0.75],
                       'HIRES': [plt.rcParams['axes.prop_cycle'].by_key()['color'][2], 's', 2.5],
                       }

    outputDir = '/Users/annaboehle/research/analysis/archival_stars/rv_calibrated'

    rvDir = '/Users/annaboehle/research/data/rv/dace/nightly_binning'
    file_ls = glob.glob('{:s}/{:s}/timeseries/{:s}*.rdb'.format(rvDir, star, star))

    star_info = ascii.read('/Users/annaboehle/research/code/directimaging_code/plotting/nearest_solar_dist_age.txt')
    row = np.where(star_info['object_name'] == star)[0][0]
    plx = star_info['plx'][row]
    mua = star_info['pm_ra'][row]
    mud = star_info['pm_dec'][row]

    sec_acc = compute_acc(plx,mua,mud)  # m/s/day

    inst_ls = []
    t_range = []
    rv_times = []
    rv_data = []
    output_fname = []

    for f in file_ls:
        rv_tab = ascii.read(f)

        if ('COR' in f and not 'CORAVEL' in f) or 'HARP' in f:
            if 'COR' in f or 'HARPN' in f:
                inst = f[-17:-12]
            else:
                inst = f[-19:-12]
        else:
            inst = 'HIRES'

        #if inst == 'HARPS15':
            # apply hard-coded offset

        #times = np.array(rv_tab['rjd'][1:], dtype=float) + 2400000 - 2400000.5  # times in MJD

        if len(rv_tab['rjd'][1:]) != 0:
            inst_ls.append(inst)

            rv_times.append(np.array(rv_tab['rjd'][1:], dtype=float))
            rv_data.append(np.array(rv_tab['vrad'][1:], dtype=float))

            t_min, t_max = np.min(rv_times[-1]), np.max(rv_times[-1])
            t_range.append([t_min,t_max])

            output_fname.append(f.split('/')[-1].rstrip('.rdb') + '_cal.dat')

    t_range = np.array(t_range)

    offsets = np.zeros(len(inst_ls))

    # first apply known offset between HARPS03 and HARPS15 for these stars
    if 'HARPS03' in inst_ls and 'HARPS15' in inst_ls and star in ['hd42581', 'kapteyns']:
        idx_harps15 = np.where(np.array(inst_ls) == 'HARPS15')[0][0]

        # apply hard-coded offset derived from the M stars in Table 3 of https://www.eso.org/sci/publications/messenger/archive/no.162-dec15/messenger-no162-9-15.pdf
        offset_mstars = np.array([0.736, -2.281])
        rv_data[idx_harps15] -= np.average(offset_mstars)

        if star == 'kapteyns' and plot:
            idx_harps15 = np.where(np.array(inst_ls) == 'HARPS15')[0][0]
            idx_harps03 = np.where(np.array(inst_ls) == 'HARPS03')[0][0]

            plt.scatter(rv_times[idx_harps03], rv_data[idx_harps03],label=inst_ls[idx_harps03])
            plt.scatter(rv_times[idx_harps15], rv_data[idx_harps15],label=inst_ls[idx_harps15])

            slope_times = np.arange(rv_times[idx_harps03][0],rv_times[idx_harps15][-1],100)
            plt.plot(slope_times,slope_times*sec_acc - np.mean(rv_times[idx_harps03]*sec_acc) + np.mean(rv_data[idx_harps03]),
                     color='black',label='Secular acceleration')
            plt.xlabel('Time (RJD)')
            plt.ylabel('Measured RV (m/s)')
            plt.legend()

            plt.figure(3)
            plt.scatter(rv_times[idx_harps03], rv_data[idx_harps03] - sec_acc*rv_times[idx_harps03],label=inst_ls[idx_harps03])
            plt.scatter(rv_times[idx_harps15], rv_data[idx_harps15] - sec_acc*rv_times[idx_harps15],label=inst_ls[idx_harps15])
            plt.xlabel('Time (RJD)')
            plt.ylabel('RV corrected for secular acceleration (m/s)')
            plt.legend()

            print 'HARPS03 average - HARPS15 average:', \
                   np.mean(rv_data[idx_harps03] - sec_acc*rv_times[idx_harps03]) - np.mean(rv_data[idx_harps15] - sec_acc*rv_times[idx_harps15])

    # loop over each instrument to check for overlaps and calculate offsets
    for ii in range(len(inst_ls)):

        # correct for secular acceleration
        if inst_ls[ii] != 'HIRES':
            rv_data[ii] = rv_data[ii] - sec_acc*rv_times[ii]

        overlap_mask = np.zeros(len(rv_times[ii]),dtype=int) + 1 # default is to be masked

        # loop over the other time ranges to ID indices that have some overlap with at least one other instrument's time range
        if len(inst_ls) > 1:
            for tt in range(len(t_range)):
                if tt != ii:
                    idx = np.where( (rv_times[ii] > t_range[tt][0]) & (rv_times[ii] < t_range[tt][1]))

                    overlap_mask[idx] = 0
        else:
            overlap_mask[:] = 0

        rv_data_masked = np.ma.masked_array(rv_data[ii],mask=overlap_mask)
        rv_times_masked = np.ma.masked_array(rv_times[ii],mask=overlap_mask)

        # check if this is a HARPS15 data set that doesn't overlap with anything from one of the stars below
        if overlap_mask.all() and inst_ls[ii] == 'HARPS15' and star in ['hd42581','kapteyns']:
            idx_harps03 = np.where(np.array(inst_ls) == 'HARPS03')[0][0]
            offsets[ii] = offsets[idx_harps03]

        # otherwise check if all elements are masked out (i.e., no overlapping measurement epochs)
        elif overlap_mask.all():
            offsets[ii] = np.nan

        else:
            offsets[ii] = np.ma.mean(rv_data_masked)

        if plot:
            plt.figure(2)

            if inst_ls[ii] == 'HARPS15':
                yerr =  np.sqrt(inst_params[inst_ls[ii]][2]**2. + 2.**2.)
            else:
                yerr = inst_params[inst_ls[ii]][2]

            plt.errorbar(rv_times[ii],rv_data[ii] - offsets[ii],color=inst_params[inst_ls[ii]][0],
                        marker='o',label=inst_ls[ii],yerr=yerr,linestyle='none')

            r = mpl.patches.Rectangle(
                    (t_range[ii][0],np.min(rv_data[ii])-offsets[ii]),
                    t_range[ii][1] - t_range[ii][0],
                    np.max(rv_data[ii]) - np.min(rv_data[ii]),
                    alpha = 0.1,
                    color=inst_params[inst_ls[ii]][0]
            )

            plt.gca().add_patch(r)

            if ii == len(inst_ls) - 1:
                plt.scatter(rv_times_masked[~rv_times_masked.mask], rv_data_masked.data[~rv_data_masked.mask] - offsets[ii],
                        color='black',marker='.',label='overlapping points')
                plt.xlabel('Time (RJD)')
                plt.ylabel('Calibrated RV (m/s)')

            else:
                plt.scatter(rv_times_masked[~rv_times_masked.mask],
                            rv_data_masked.data[~rv_data_masked.mask] - offsets[ii],
                            color='black', marker='.')

        print

        if save:
            # write out the individual RV data sets in separate files
            if not os.path.exists('{:s}/{:s}'.format(outputDir,star)):
                os.mkdir('{:s}/{:s}'.format(outputDir,star))
            np.savetxt('{:s}/{:s}/{:s}'.format(outputDir,star,output_fname[ii]),
                       np.column_stack((rv_times[ii],rv_data[ii] - offsets[ii])))





    plt.legend()

    #if save:
        # NEXT: need to save out calibrated RV data itself! use same file naming like raw data
    #    np.savetxt('rv_calibrated_{:s}.dat'.format(star), np.column_stack((rv_times, rv_data - offsets)))

    if plot:
        plt.show()


def calibrate_offsets(star,save=False,plot=True):
    """
    Calibrates RV offsets between different instruments.
    Includes treatments of the two stars that had no overlap
    between HARPS03 and HARPS15, so a pre-determined offset was applied.

    :param star: name of star from Boehle et al. 2019 paper sample
    :param save: if True, write out the calibrated data sets
    :param plot: if True, plot the calibration process
    :return: None
    """

    if plot:
        inst_params = {'COR98': [plt.rcParams['axes.prop_cycle'].by_key()['color'][3], '^', 5.0],
                       'COR07': [plt.rcParams['axes.prop_cycle'].by_key()['color'][9], '<', 8.0],
                       'COR14': [plt.rcParams['axes.prop_cycle'].by_key()['color'][7], '>', 3.0],
                       'HARPS03': [plt.rcParams['axes.prop_cycle'].by_key()['color'][0], 'o', 0.75],
                       'HARPS15': [plt.rcParams['axes.prop_cycle'].by_key()['color'][1], 'D', 0.75],
                       'HIRES': [plt.rcParams['axes.prop_cycle'].by_key()['color'][2], 's', 2.5],
                       }

    outputDir = '/Users/annaboehle/research/analysis/archival_stars/rv_calibrated'

    rvDir = '/Users/annaboehle/research/data/rv/dace/nightly_binning'
    file_ls = glob.glob('{:s}/{:s}/timeseries/{:s}*.rdb'.format(rvDir, star, star))

    star_info = ascii.read('/Users/annaboehle/research/code/directimaging_code/plotting/nearest_solar_dist_age.txt')
    row = np.where(star_info['object_name'] == star)[0][0]
    plx = star_info['plx'][row]
    mua = star_info['pm_ra'][row]
    mud = star_info['pm_dec'][row]

    sec_acc = compute_acc(plx,mua,mud)  # m/s/day

    inst_ls = []
    #t_range = []
    rv_times = []
    rv_data = []
    output_fname = []
    rv_times_sum = 0
    n_rv_times = 0

    for f in file_ls:
        rv_tab = ascii.read(f)

        if ('COR' in f and not 'CORAVEL' in f) or 'HARP' in f:
            if 'COR' in f or 'HARPN' in f:
                inst = f[-17:-12]
            else:
                inst = f[-19:-12]
        else:
            inst = 'HIRES'

        #if inst == 'HARPS15':
            # apply hard-coded offset

        #times = np.array(rv_tab['rjd'][1:], dtype=float) + 2400000 - 2400000.5  # times in MJD

        if len(rv_tab['rjd'][1:]) != 0:
            inst_ls.append(inst)

            rv_times.append(np.array(rv_tab['rjd'][1:], dtype=float))
            rv_data.append(np.array(rv_tab['vrad'][1:], dtype=float))

            #t_min, t_max = np.min(rv_times[-1]), np.max(rv_times[-1])
            #t_range.append([t_min,t_max])

            output_fname.append(f.split('/')[-1].rstrip('.rdb') + '_cal.dat')

            rv_times_sum += np.sum(rv_times[-1])
            n_rv_times += len(rv_times[-1])

    t_ref = rv_times_sum/n_rv_times

    # first apply known offset between HARPS03 and HARPS15 for these stars
    if 'HARPS03' in inst_ls and 'HARPS15' in inst_ls and star in ['hd42581', 'kapteyns']:
        idx_harps15 = np.where(np.array(inst_ls) == 'HARPS15')[0][0]
        idx_harps03 = np.where(np.array(inst_ls) == 'HARPS03')[0][0]


        # apply hard-coded offset derived from the M stars in Table 3 of https://www.eso.org/sci/publications/messenger/archive/no.162-dec15/messenger-no162-9-15.pdf
        offset_mstars = np.array([0.736, -2.281])
        rv_data[idx_harps15] -= np.average(offset_mstars)

        if False:
            rv_data[idx_harps03] = np.concatenate((rv_data[idx_harps03],rv_data[idx_harps15]))
            rv_times[idx_harps03] = np.concatenate((rv_times[idx_harps03],rv_times[idx_harps15]))

            rv_data.pop(idx_harps15)
            rv_times.pop(idx_harps15)
            inst_ls.pop(idx_harps15)
            output_fname.pop(idx_harps15)

        # here also merge the two data sets so they are treated the same

        if star == 'kapteyns' and plot:
            idx_harps15 = np.where(np.array(inst_ls) == 'HARPS15')[0][0]

            if True:
                idx_harps03 = np.where(np.array(inst_ls) == 'HARPS03')[0][0]

                plt.scatter(rv_times[idx_harps03], rv_data[idx_harps03],label=inst_ls[idx_harps03])
                plt.scatter(rv_times[idx_harps15], rv_data[idx_harps15],label=inst_ls[idx_harps15])

                slope_times = np.arange(rv_times[idx_harps03][0],rv_times[idx_harps15][-1],100)
                plt.plot(slope_times,slope_times*sec_acc - np.mean(rv_times[idx_harps03]*sec_acc) + np.mean(rv_data[idx_harps03]),
                     color='black',label='Secular acceleration')
                plt.xlabel('Time (RJD)')
                plt.ylabel('Measured RV (m/s)')
                plt.legend()

            plt.figure(3)
            plt.scatter(rv_times[idx_harps03], rv_data[idx_harps03] - sec_acc*rv_times[idx_harps03],label=inst_ls[idx_harps03])
            plt.scatter(rv_times[idx_harps15], rv_data[idx_harps15] - sec_acc*rv_times[idx_harps15],label=inst_ls[idx_harps15])
            plt.xlabel('Time (RJD)')
            plt.ylabel('RV corrected for secular acceleration (m/s)')
            plt.legend()

            #print 'HARPS03 average - HARPS15 average:', \
            #       np.mean(rv_data[idx_harps03] - sec_acc*rv_times[idx_harps03]) - np.mean(rv_data[idx_harps15] - sec_acc*rv_times[idx_harps15])

            plt.savefig('./kapteyns_harps03_harps15_orig.png')

    rv_data_cal = []
    rv_times_cal = []

    sorted_idx = np.argsort(np.array([times[0] for times in rv_times]))
    offsets = np.zeros(len(inst_ls))

    # loop over each instrument from earlier to later times to check for overlaps and calculate offsets
    for ii in sorted_idx:

        # correct for secular acceleration
        if inst_ls[ii] != 'HIRES':
            rv_data[ii] = rv_data[ii] - sec_acc*(rv_times[ii] - t_ref)

        # get current t_range for calibrated rv data
        overlap_mask = np.zeros(len(rv_times[ii]), dtype=int)

        if ii != sorted_idx[0]:
            t_range = [np.min(np.array(rv_times_cal)),np.max(np.array(rv_times_cal))]

            #idx = np.where((rv_times[ii] >= t_range[0]) & (rv_times[ii] <= t_range[1]))[0]
            idx = ((rv_times[ii] >= t_range[0]) & (rv_times[ii] <= t_range[1]))

            overlap_mask[~idx] = 1
            #t_range = [np.min(rv_times[ii]),np.max(rv_times[ii])]
            #overlap_mask = np.zeros(len(rv_times[ii]),dtype=int) + 1 # default is to be masked

        #print

        rv_data_masked = np.ma.masked_array(rv_data[ii], mask=overlap_mask)
        rv_times_masked = np.ma.masked_array(rv_times[ii], mask=overlap_mask)


        if overlap_mask.all() and inst_ls[ii] == 'HARPS15' and star in ['hd42581', 'kapteyns']:
            #offset_mstars = np.array([0.736, -2.281])
            #offsets[ii] = np.average(offset_mstars)

            idx_harps03 = np.where(np.array(inst_ls) == 'HARPS03')[0][0]
            offsets[ii] = offsets[idx_harps03]

        else:
            offsets[ii] = np.ma.mean(rv_data_masked)
        #print rv_data_masked


        if plot:
            if ii != sorted_idx[0]:
                plt.figure()
                plt.scatter(rv_times_cal,rv_data_cal,label='Calibrated RV data')
                plt.scatter(rv_times[ii],rv_data[ii]-offsets[ii],label='{:s} RV data'.format(inst_ls[ii]))


                plt.scatter(rv_times_masked[~rv_times_masked.mask], rv_data_masked.data[~rv_data_masked.mask] - offsets[ii],
                        color='black', marker='.', label='overlapping points')
                plt.xlabel('Time (RJD)')
                plt.ylabel('Calibrated RV (m/s)')
                #plt.xlim(np.min(rv_times),np.max(rv_times))
                plt.legend()

        # very end of loop:
        rv_data_cal.extend(rv_data[ii] - offsets[ii])
        rv_times_cal.extend(rv_times[ii])

        if save:
            # write out the individual RV data sets in separate files
            if not os.path.exists('{:s}/{:s}'.format(outputDir,star)):
                os.mkdir('{:s}/{:s}'.format(outputDir,star))
            np.savetxt('{:s}/{:s}/{:s}'.format(outputDir,star,output_fname[ii]),
                       np.column_stack((rv_times[ii],rv_data[ii] - offsets[ii])))

    if plot:
        plt.show()

        # where current inst's data overlaps with calibrated RV data
        #idx = np.where((rv_times[ii] >= t_range[ii][0]) & (rv_times[ii] <= t_range[ii][1]))

        #overlap_mask[idx] = 0

        #if len(inst_ls) > 1:
            #for tt in range(len(t_range)):
            #    if tt != ii:
            #        idx = np.where( (rv_times[ii] > t_range[tt][0]) & (rv_times[ii] < t_range[tt][1]))

            #        overlap_mask[idx] = 0
        #else:
        #    overlap_mask[:] = 0

    '''
        

        # check if this is a HARPS15 data set that doesn't overlap with anything from one of the stars below
        if overlap_mask.all() and inst_ls[ii] == 'HARPS15' and star in ['hd42581','kapteyns']:
            idx_harps03 = np.where(np.array(inst_ls) == 'HARPS03')[0][0]
            offsets[ii] = offsets[idx_harps03]

        # otherwise check if all elements are masked out (i.e., no overlapping measurement epochs)
        elif overlap_mask.all():
            offsets[ii] = np.nan

        else:
            offsets[ii] = np.ma.mean(rv_data_masked)

        if plot:
            plt.figure(2)

            if inst_ls[ii] == 'HARPS15':
                yerr =  np.sqrt(inst_params[inst_ls[ii]][2]**2. + 2.**2.)
            else:
                yerr = inst_params[inst_ls[ii]][2]

            plt.errorbar(rv_times[ii],rv_data[ii] - offsets[ii],color=inst_params[inst_ls[ii]][0],
                        marker='o',label=inst_ls[ii],yerr=yerr,linestyle='none')

            r = mpl.patches.Rectangle(
                    (t_range[ii][0],np.min(rv_data[ii])-offsets[ii]),
                    t_range[ii][1] - t_range[ii][0],
                    np.max(rv_data[ii]) - np.min(rv_data[ii]),
                    alpha = 0.1,
                    color=inst_params[inst_ls[ii]][0]
            )

            plt.gca().add_patch(r)

            if ii == len(inst_ls) - 1:
                plt.scatter(rv_times_masked[~rv_times_masked.mask], rv_data_masked.data[~rv_data_masked.mask] - offsets[ii],
                        color='black',marker='.',label='overlapping points')
                plt.xlabel('Time (RJD)')
                plt.ylabel('Calibrated RV (m/s)')

            else:
                plt.scatter(rv_times_masked[~rv_times_masked.mask],
                            rv_data_masked.data[~rv_data_masked.mask] - offsets[ii],
                            color='black', marker='.')

        print
    '''




    #print offsets


    #plt.legend()

    #if save:
        # NEXT: need to save out calibrated RV data itself! use same file naming like raw data
    #    np.savetxt('rv_calibrated_{:s}.dat'.format(star), np.column_stack((rv_times, rv_data - offsets)))



