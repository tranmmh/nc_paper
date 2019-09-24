'''
Author: Luke Prince
Date: 11 January 2019
'''
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Imputer

from itertools import combinations

def entropyEstimation(x,bins='auto'):
    px,bx = np.histogram(x,bins=bins,density=True)
    dx = np.diff(bx)
    h = -np.sum(dx*px*np.ma.log2(px))
    return h

def jointEntropyEstimation2D(x,y,bins='auto'):
    pxy,bx,by = np.histogram2d(x,y,bins=bins,normed=True)
    dx = np.diff(bx)[0]
    dy = np.diff(by)[0]
    hxy = -np.sum(dx*dy*pxy*np.ma.log2(pxy))
    return hxy

def jointEntropyEstimation3D(x,y,z,bins='auto'):
    pxyz,(bx,by,bz) = np.histogramdd([x,y,z],bins=bins,normed=True)
    dx = np.diff(bx)[0]
    dy = np.diff(by)[0]
    dz = np.diff(bz)[0]
    hxyz = -np.sum(dx*dy*dz*pxyz*np.ma.log2(pxyz))
    return hxyz

def mutualInformation(s,r):
    
    bins_r = np.histogram(r,bins='auto')[1]
    bins_s = np.arange(-0.5, max(s) + 1.5, 1)
    
    hr = entropyEstimation(r, bins=bins_r)
    hs = entropyEstimation(s, bins=bins_s)
    hrs = jointEntropyEstimation2D(r,s,bins=[bins_r, bins_s])
    
    I = hr + hs - hrs
    return I

def conditionalMutualInformation(r, s, c):

    bins_r = np.histogram(r,bins='auto')[1]
    bins_s = np.arange(-0.5, max(s)+1.5, 1)
    bins_c = np.arange(-0.5, max(c)+1.5, 1)
    
    hrc  = jointEntropyEstimation2D(r, c, bins=[bins_r, bins_c]);
    hsc  = jointEntropyEstimation2D(s, c, bins=[bins_s, bins_c]);
    hrsc = jointEntropyEstimation3D(r, s, c, bins=[bins_r, bins_s, bins_c]);
    hc   = entropyEstimation(c, bins= bins_c);
    
    I = hrc + hsc - hrsc - hc
    
    return I

def spikeConditionalMutualInformation(postSpikes, signal, preSpikes):
    
    maxPostSpikes = max([max(post) for post in postSpikes])
    
    br = np.arange(-0.5, maxPostSpikes+1.5, 1) # Bins for POST-synaptic spike count
    bs = np.arange(-0.5, 2.5, 1)  # Bins for signal state (0- low, 1-high)
    bc = np.arange(-0.5, 11.5, 1) # Bins for PRE-synaptic spike count
    
    prc  = []
    prc0 = []
    prc1 = []
    
    for postSpikes_i in postSpikes:
        prc.append(np.histogram2d(preSpikes, postSpikes_i, bins=[bc, br], normed=True)[0])
        prc0.append(np.histogram2d(preSpikes[signal==0], postSpikes_i[signal==0],
                                  bins= [bc, br],
                                  normed= True)[0])
        prc1.append(np.histogram2d(preSpikes[signal==1], postSpikes_i[signal==1],
                                  bins=[bc, br],
                                  normed=True)[0])
        
    prc = np.array(prc).mean(axis=0)                # P(pre, post)
    prc0 = np.array(prc0).mean(axis=0)              # P(pre, post | signal = low)
    prc1 = np.array(prc1).mean(axis=0)              # P(pre, post | signal = high)


    p1 = np.sum(signal==1)/float(len(signal)) # P(signal = high)
    prsc = np.array([(1-p1)*prc0,p1*prc1])          # P(pre, post, signal)

    hrc = -np.sum(prc*np.ma.log2(prc))                                  # H(pre, post)
    hsc = jointEntropyEstimation2D(signal, preSpikes, bins=[bs, bc])    # H(signal, pre)
    hrsc = -np.sum(prsc*np.ma.log2(prsc))                               # H(pre, post, signal)
    hc = entropyEstimation(preSpikes, bins=bc)                          # H(pre)

    return hrc + hsc - hrsc - hc # I(signal; post | pre) = H(post, pre) + H(signal, pre) - H(signal, post, pre) - H(pre)

def spikeMutualInformation(postSpikes, signal):
    maxPostSpikes = max([max(postSpikes_i) for postSpikes_i in postSpikes])
    
    br = np.arange(-0.5, maxPostSpikes+1.5, 1) # Bins for POST-synaptic spike count
    bs = np.arange(-0.5, 2.5, 1)  # Bins for signal state (0- low, 1-high)
    
    prs  = []
    pr   = []
    
    for postSpikes_i in postSpikes:
        prs.append(np.histogram2d(postSpikes_i, signal, bins=[br, bs], normed= True)[0])

        pr.append(np.histogram(postSpikes_i, bins=br, normed=True)[0])

    prs = np.array(prs).mean(axis=0)                # P(post, signal)
    pr = np.array(pr).mean(axis=0)                  # P(post)
    p1 = np.sum(signal==1)/float(len(signal))       # P(signal = high)

    
    hrs = -np.sum(prs*np.ma.log2(prs))            # H(post, signal)
    hr = -np.sum(pr*np.ma.log2(pr))               # H(post)
    hs = -(p1*np.log2(p1) +(1-p1)*np.log2(1-p1))  # H(signal)

    return hr + hs - hrs                            # I(signal; post) = H(post) + H(signal) - H(post, signal)

def hypothesis_tests(df):
    
    fluors = list(df['Fluorescence'].unique())
    fluor_pairs = list(combinations(iterable=fluors, r=2))
    colname = [col for col in df.columns if col is not 'Fluorescence'][0]
    grouped_data = [df.groupby('Fluorescence').get_group(fl)[colname] for fl in fluors]
    
    test_dict = {'Main' : {}, 'Pairs' : {}}
    
    _,p_levene = stats.levene(*grouped_data) # equal variances test
    test_dict['Main']['Levenes'] = p_levene
    
    test, test_string = (stats.f_oneway, 'One-Way ANOVA') if p_levene > 0.05 else (stats.kruskal, 'Kruskal-Wallis')
    
    F, p_test = test(*grouped_data)
    test_dict['Main']['Test'] = test_string + " - " + str(F)
    test_dict['Main']['p'] = p_test
    
    for pair in fluor_pairs:
        fl1 = pair[0]; fx1 = [fx for fx,fl in enumerate(fluors) if fl==fl1][0]
        fl2 = pair[1]; fx2 = [fx for fx,fl in enumerate(fluors) if fl==fl2][0]
        pair_string = '%s vs. %s'%(fl1, fl2)
        test_dict['Pairs'][pair_string] = {}
        
        _, p_levene = stats.levene(grouped_data[fx1],grouped_data[fx2])
        test_dict['Pairs'][pair_string]['Levenes'] = p_levene
        
        test_string = 'Ind t' if p_levene > 0.05 else 'Welch t'
        tstat, p_ttind = stats.ttest_ind(a = grouped_data[fx1], b = grouped_data[fx2], equal_var=p_levene>0.05)
        test_dict['Pairs'][pair_string]['Test'] = test_string + " - " + str(tstat)
        test_dict['Pairs'][pair_string]['p'] = p_ttind
        
    return test_dict

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def ephys_pca(df_ephys):
    features = df_ephys.columns.values[2:]
    x = df_ephys.loc[:,features].values
    y = df_ephys.loc[:,['Fluorescence']].values

    #standardize data, used Imputer to handle NaN entries
    imp = Imputer(strategy="mean", axis=0)
    scale = StandardScaler()
    x = scale.fit_transform(imp.fit_transform(x))
    
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    df_pca = pd.DataFrame(data = principalComponents,
                          columns = ['principal component 1', 'principal component 2', 'principal component 3'],
                          index = df_ephys.index)
    
    return pd.concat([df_pca, df_ephys[['Fluorescence']]], axis = 1)
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------

def spikeTimes_to_phase(spikeTimes, frequency):
    '''
    Convert spike times to phases with respect to a given frequency
    
    Arguments:
        spike times (np.array) : array of spike times for a particular cell in a particular trial
        frequency (float)      : desired frequency to obtain phases
    
    Returns:
        phases (np.array)      : array of phases
        
    '''
    dt = 1./frequency
    phase = 2*np.pi*(spikeTimes%dt)/dt
    return phase

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def gaussian(x, mu=0, sigma=1):
    return np.exp(-0.5 * (x - mu)**2 * sigma**-2) * (2 * np.pi * sigma**2)**-0.5

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def gaussian_mixture(x, M):
    px = np.zeros_like(x)
    for k in range(M.n_components):
        px += gaussian(x, mu=M.means_[k, 0], sigma=np.sqrt(M.covariances_[k,0,0])) * M.weights_[k]
    return px

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def log_gaussian(x, mu=0, sigma=1):
    return np.exp(-0.5 * (np.log(x) - mu)**2 * sigma**-2) * (2 * np.pi * (sigma * x)**2) ** -0.5
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def log_gaussian_mixture(x, M):
    px = np.zeros_like(x)
    for k in range(M.n_components):
        px += log_gaussian(x, mu=M.means_[k, 0], sigma=np.sqrt(M.covariances_[k,0,0])) * M.weights_[k]
    return px