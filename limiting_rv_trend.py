import matplotlib.pyplot as plt
import numpy as np

def get_limiting_rv_trend(t_baseline, rv_std, num_meas):
    '''
    [t_baseline] = days
    [rv_std] = m/s
    '''

    # limiting rv trend: assume delta_v over t_baseline = 5*rv_std
    delta_v = 5*rv_std
    rv_trend = delta_v/(t_baseline/365.)

    # simulate this RV trend, assuming equal spacing of num_meas data points
    #np.poly1d

    
