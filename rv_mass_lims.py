import astropy
import numpy as np
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
import astropy.constants as const

from orbits import kepler_py3 as k


# Get random observation times in the time period of 1 semester. You'd probably schedule them roughly evenly spaced but with some noise.<br>
# -> **make even spacing over 6 months with some buffer, add noise to that?**
# |
# Make one set of fake data to go with those measurement times.<br>
# -> **generate gaussian noise with mean = 0 and assuming an rv_std (that can be changed later)**
# 
# Bootstrap randomize the data 1000 times.<br>
# -> **np.random.choise with replace=True**
# 
# Do generalized L-S periodogram for each bootstrapped data set, collecting the distributions of power for each frequency.<br>
# -> **astropy.timeseries.LombScargle with default normalization.**<br>
# 
# Finally: record the power level for which 99% of the powers fall below, that is the 1% FAP level.
# 
# refs:<br>
# https://docs.astropy.org/en/stable/timeseries/lombscargle.html<br>
# http://jakevdp.github.io/blog/2017/03/30/practical-lomb-scargle/<br>

# ## Input

# In[345]:

#def gls_fap(dT=365/2., num_meas=25, rv_std=4, m_star=0.3):
    '''
'''
    
num_meas = 25 # number of measurements over the 

rv_std = 4.0 # m/s, noise of the rv measurements

m_star = 0.3 # solar masses


# ## Get observation times

# In[85]:


dT_semester = 365/2. # days
obs_times = np.linspace(0,dT_semester,num_meas)
print(f'regularly spaced measurements are {obs_times[1]-obs_times[0]:.3f} days apart for num_meas = {num_meas}')

obs_times += np.random.normal(loc=0,scale=3,size=num_meas)  # add random noise
print(f'min_time = {obs_times.min():.3f}, max time = {obs_times.max():.3f}')


# In[86]:


for t in obs_times:
    plt.axvline(t)


# ## Generate RV data

# In[173]:


rv_data = np.random.normal(loc=0,scale=rv_std,size=num_meas)


# In[174]:


plt.scatter(obs_times,rv_data,marker='o')


# ## Bootstrap RV data

# In[175]:


n_bootstrap = 1000
rv_bootstrap = np.random.choice(rv_data,replace=True,size=(n_bootstrap,num_meas))


# In[176]:


for n in range(n_bootstrap):
    plt.plot(obs_times,rv_bootstrap[n])


# ## L-S periodogram

# In[326]:


# frequency range from Christophe's code
min_f = 1/(4.0*dT_semester)
max_f = 1/0.65

f_data,p_data=LombScargle(obs_times,rv_data).autopower(minimum_frequency=min_f, 
                                                       maximum_frequency=max_f)


# In[327]:


plt.plot(1/f_data,p_data,color='black')
plt.gca().set_xscale('log')
plt.xlabel('Period (days)')
plt.ylabel('Power')

print(f'min period = {1/max_f} days, max period = {1/min_f} days')


# In[328]:


p_bootstrap = np.zeros((n_bootstrap,len(f_data)))

for n in range(n_bootstrap):
    _, p_bootstrap[n] = LombScargle(obs_times,rv_bootstrap[n]).autopower(minimum_frequency=1/(4.0*dT_semester), 
                                                                         maximum_frequency=1/0.65)


# In[329]:


for n in range(n_bootstrap):
    plt.plot(f_data,p_bootstrap[n],color='gray',alpha=0.2)
plt.plot(f_data,p_data,color='black')


# ## Get 99% power for each frequency

# In[330]:


p_bootstrap_sorted = np.sort(p_bootstrap,axis=0)


# In[331]:


fap_99 = p_bootstrap_sorted[990+1,:]


# In[332]:


for n in range(n_bootstrap):
    plt.plot(f_data,p_bootstrap[n],color='gray',alpha=0.2)
plt.plot(f_data,p_data,color='black')
plt.plot(f_data,fap_99,linestyle='dotted',color='red')


# In[333]:


for n in range(n_bootstrap):
    plt.plot(1/f_data,p_bootstrap[n],color='gray',alpha=0.2)
plt.plot(1/f_data,p_data,color='black')
plt.plot(1/f_data,fap_99,linestyle='dotted',color='red')
plt.gca().set_xscale('log')


# ## Try injecting a planet signal

# In[342]:


P_test_idx = int(len(f_data)/20)

P_test = 1/f_data[P_test_idx]
rv_sim_plt = k.rv_curve(np.linspace(0,P_test,100),
                    P=P_test,
                    w=0,
                    K=3,
                    e=0,  # circular only
                    t0=0) # fixed since degenerate with w for e = 0


# In[335]:


plt.plot(np.linspace(0,P_test,100),rv_sim_plt,
         marker='.',linestyle='none')


# In[363]:


rv_sim = k.rv_curve(obs_times,
                    P=P_test,
                    w=0,
                    K=3.8,
                    e=0,  # circular only
                    t0=0) # fixed since degenerate with w for e = 0


# Take Zechmeister et al. 2009 approach and consider data as noise.

# In[364]:


rv_simplusdata = rv_sim + rv_data


# In[400]:


# L-S at single frequency!
p_sim=LombScargle(obs_times,rv_simplusdata).power(f_data[P_test_idx])


# In[399]:


print(p_sim)
print(fap_99[P_test_idx])


# In[391]:


plt.plot(obs_times,rv_simplusdata,marker='.',linestyle='none')


# In[401]:


#plt.plot(f_sim,p_sim,color='black')
#plt.plot(f_data,fap_99,linestyle='dotted',color='red')


# RV semi-amplitude formula:
# 
# K(m_p, m_star, P) -> so need to assume m_star and can solve this for m_p min!

# In[387]:


m_p = 5*(const.M_earth/const.M_jup)
K = 28.4329*(m_p)*((m_p/(const.M_sun/const.M_jup)).value + m_star)**(-2./3)*(P_test/365.)**(-1./3)
print(K)


# For a range of frequencies (for which you have the FAP already):<br>
# For a 12 evenly spaces orbital phases (w):
# 
# - inject planet with given K
# - add to actual data to simulate noise
# - calculate power in that frequency in simulated data with GLS
# - iterate in fixed steps until 99% FAP is reached
# <br>
# <br>
# - average over all phases
# - convert K to m_p for each frequency

# In[ ]:




