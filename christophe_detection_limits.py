
# WARNING: this script only works if there are no significant signals in the data!!!

import os,sys,string
import compute,pyrdb
from numpy import *
from matplotlib import pyplot

jitter = 0.0007
N = 1000
n = 12
K_start = 0.00050
dK = 0.00001

filename = sys.argv[1]
m1 = float(sys.argv[2])
data = pyrdb.read_rdb(filename)
name = string.split(os.path.split(filename)[-1],'_')[0]
result_file = string.replace(os.path.split(filename)[-1],'.rdb','_detection_limits.rdb')

jdb = asarray(data['jdb'])
vrad = asarray(data['vrad'])
svrad = asarray(data['svrad'])
svrad = sqrt(svrad**2+jitter**2)

dT = jdb[-1]-jdb[0]
fstart = 1.0/(4.0*dT)
fend = 1.0/0.65
df = 1.0/dT/8.0
f = arange(fstart,fend,df)
logP = log10(1.0/f)

power = compute.gls(jdb,vrad,svrad,f)[0]

f2 = arange(0.,fend,df)
f2 = concatenate((-1.*f2[::-1][:-1],f2))
wf = f2*0.
for i in range(len(f2)):
	wf[i] = sum(cos(2*pi*f2[i]*jdb)/svrad**2)**2+sum(sin(2*pi*f2[i]*jdb)/svrad**2)**2
	wf[i] = wf[i]/sum(1.0/svrad**2)**2
for i in wf.argsort()[::-1]:
	if abs(f2[i]) > (2.0/dT): break
main_alias = abs(f2[i])

print main_alias

if not locals().has_key('threshold'):
	ppd = zeros(N,'d')
	for i in range(N):
		perm = random.permutation(len(jdb))
		nvrad = vrad[perm]
		nsvrad = svrad[perm]
		npower = compute.gls(jdb,nvrad,nsvrad,f)[0]
		ppd[i] = npower.max()
	threshold = sort(ppd)[int(0.99*N)]

print threshold

P = 10**arange(log10(1./fend),log10(1./fstart),0.01)
K_100_limit = zeros(len(P),'d')+K_start
K_50_limit = zeros(len(P),'d')+K_start

for i in range(len(P)):
	print P[i]
	f_range = zeros(len(f),'bool')
	for ff in [1.0/P[i],abs(1.0/P[i]+main_alias),abs(1.0/P[i]-main_alias)]:
		f_range = f_range | (abs(f-ff) < (1.0/dT))
	pp = zeros(n,'d')
	K_100 = K_100_limit[i-1]
	K_50 = K_50_limit[i-1]
	for j in range(n):
		T0 = jdb[0]+float(j)/n*P[i]
		nvrad = vrad + K_100*cos(2*pi/P[i]*(jdb-T0))
		npower = compute.gls(jdb,nvrad,svrad,f)[0]
		pp[j] = npower[f_range].max()
	T0_100 = jdb[0]+float(pp.argmin())/n*P[i]
	T0_50 = jdb[0]+float(pp.argsort()[len(pp)/2])/n*P[i]
	pp_100 = pp.min()
	pp_50 = sort(pp)[len(pp)/2]
	while pp_100 > threshold:
		K_100 = K_100-dK
		nvrad = vrad + K_100*cos(2*pi/P[i]*(jdb-T0_100))
		npower = compute.gls(jdb,nvrad,svrad,f)[0]
		pp_100 = npower[f_range].max()
	while pp_100 < threshold:
		K_100 = K_100+dK
		nvrad = vrad + K_100*cos(2*pi/P[i]*(jdb-T0_100))
		npower = compute.gls(jdb,nvrad,svrad,f)[0]
		pp_100 = npower[f_range].max()
	K_100_limit[i] = K_100
	print K_100_limit[i]
	while pp_50 > threshold:
		K_50 = K_50-dK
		nvrad = vrad + K_50*cos(2*pi/P[i]*(jdb-T0_50))
		npower = compute.gls(jdb,nvrad,svrad,f)[0]
		pp_50 = npower[f_range].max()
	while pp_50 < threshold:
		K_50 = K_50+dK
		nvrad = vrad + K_50*cos(2*pi/P[i]*(jdb-T0_50))
		npower = compute.gls(jdb,nvrad,svrad,f)[0]
		pp_50 = npower[f_range].max()
	K_50_limit[i] = K_50

a = (m1*(P/365.25)**2)**(1./3.)
m2sini_100 = 317.8*K_100_limit/0.0284329*sqrt(m1)*sqrt(a)
m2sini_50 = 317.8*K_50_limit/0.0284329*sqrt(m1)*sqrt(a)

data = {}
data['P'] = P.tolist()
data['a'] = a.tolist()
data['K_100'] = K_100_limit.tolist()
data['K_50'] = K_50_limit.tolist()
data['m2sini_100'] = m2sini_100.tolist()
data['m2sini_50'] = m2sini_50.tolist()
pyrdb.write_rdb(result_file,data,['P','a','K_100','m2sini_100','K_50','m2sini_50'],'%e\t%e\t%e\t%e\t%e\t%e\n')

pyplot.loglog(a,m2sini_100)
pyplot.loglog(a,m2sini_50)
pyplot.xlabel('Semi-major axis [AU]')
pyplot.ylabel('M sini [$M{_\oplus}$]')
pyplot.title(name)

