import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, linregress

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

def optoResponse(t, Vss, tau_rise, tau_decay, latency, V0):
    pre_onset = (t<latency)*V0
    onset = (t>=latency)*(t<=(50+latency))*((V0-Vss)*np.exp(-(t-latency)/tau_rise)+Vss)
    offset = (t>(50+latency))*((np.max(onset[onset!=0])-V0)*np.exp(-(t-50-latency)/tau_decay)+V0)
    response = pre_onset + onset + offset
    return response

def generalLinearDiscontinuousFn(t,baseline,*args):
    if len(args) == 2:
        slope,latency = args
        x = baseline + (t>=latency)*((t-latency)*slope)
    else:
        slopes = args[::2]
        latencies = args[1::2]
        x = baseline + (t-latencies[0])*slopes[0]*(t>=latencies[0])
        for mprev,m,l in zip(slopes[:-1],slopes[1:],latencies[1:]):
            x = x + (t-l)*(m-mprev)*(t>=l)
    return x

def aicFromRSS(RSS,n,k):
    return n*np.log(np.array(RSS)/n) + 2*k

def bicFromRSS(RSS,n,k):
    return n*np.log(np.array(RSS)/n) + k*np.log(n)

def chi2_gof(RSS_alt,RSS_null,n,dof):
    D = -n*np.log(RSS_alt/RSS_null)
    return chi2.cdf(D,dof)

def fitBaselineFn(t,x):
    xbase = x[:15]
    tbase = t[:15]
    
    baseline = np.mean(xbase[:5])*np.ones_like(tbase)
    m,c = linregress(tbase, xbase)[:2]
    drift = tbase*m + c
    
#     baselineFn = lambda t,m : generalLinearDiscontinuousFn(t,m,0)
#     popt,pcov = curve_fit(baselineFn,tbase,xbase,p0=np.array([np.mean(np.diff(xbase))*10]))
#     drift = baselineFn(tbase,*popt)
    
    RSSmean = len(xbase)*xbase.var()
    RSSdrift = np.sum((drift-xbase)**2)
    
    pdrift = chi2_gof(RSSdrift,RSSmean,n=len(x),dof=1)
    
    if m<=0 and pdrift > 0.95: # significant and negative drift
        baseline = np.mean(xbase[:5])*np.ones_like(t)
        baseline = t*m + c
        RSSnull = np.sum((baseline-x)**2)
        dofnull = 1
    else:
        baseline = np.mean(x[:10])*np.ones_like(t)
        RSSnull = len(x)*x.var()
        dofnull = 0
    return baseline,RSSnull,dofnull

def fitPiecewiseLinearFunction(t, x):
    
    '''First need to determine appropropriate baseline'''
    
    baseline,RSSnull,dofnull = fitBaselineFn(t,x)
    gldf_baseline = lambda t, *args : generalLinearDiscontinuousFn(t, baseline, *args)
    
    maxcomps = 6
    
    p0_all = [[np.mean(np.diff(x[(i+1)*len(x)/(maxcomps+1):(i+2)*len(x)/(maxcomps+1)]))*10,
               (i+1.5)*len(x)/(10*(maxcomps+1))]
              for i in range(maxcomps)]
    p0_all = np.ravel(p0_all)
    
    bounds_lo = [0.0,0.5,0.0,2.01,-np.inf,2.01,-np.inf,2.01,-np.inf,2.01,-np.inf,2.01]
    bounds_hi = [np.inf,2.0,np.inf,t[-5],np.inf,t[-5],np.inf,t[-5],np.inf,t[-5],np.inf,t[-5]]
    
    p0_all = np.clip(p0_all,bounds_lo,bounds_hi)
    
    RSS = [RSSnull]
    
    RSSmodel = []
    
    minBIC = np.inf
    minBICReign = 0
    
    for ix in range(1,maxcomps+1):
        p0 = p0_all[:2*ix]
        bounds = [bounds_lo[:2*ix],bounds_hi[:2*ix]]
        try:
            popt,pcov = curve_fit(gldf_baseline,t,x,p0=p0,bounds=bounds,maxfev=10000)
        except RuntimeError:
            continue
        xhat = gldf_baseline(t,*popt)
        RSSalt = np.sum((xhat-x)**2)
        bic = bicFromRSS(RSSalt,n=len(x),k=2*ix)
        if bic < minBIC:
            minBIC = bic
            pars = popt
            dofalt = 2*ix
        else:
            minBICReign +=1
            if minBICReign == 5:
                RSSmodel.append(RSSalt)
                break
        
        RSSmodel.append(RSSalt)
        
    RSSalt = min(RSSmodel)
    pmodel = chi2_gof(RSSalt,RSSnull,n=len(x),dof=dofalt-dofnull)
    
    return pars, pmodel, RSS+RSSmodel, baseline