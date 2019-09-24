import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, Imputer
from stats import *

def add_inset_ax(ax,rect):
    
    '''Add an inset subplot'''
    
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    ax_inset = fig.add_axes([x,y,width,height])
    return ax_inset

def plot_fICurves(cells):

    avFI = [[],[],[]]
    fluors = ['NF','PV+','SST+']
    fluors_acc = ['GFP-','GAD67-GFP+','GIN-GFP+']
    colors = ['#F5A623', '#4A90E2', '#7ED321']

    fig,axs = plt.subplots(nrows=1,ncols=3,sharey=True,figsize=(12,3))

    for cellid, celln in cells.items():
        if celln.layer == 'L2/3':
            fluor_idx = [idx for idx, fl in enumerate(fluors) if celln.fluor == fl][0]
            avFI[fluor_idx].append(celln.ephys.spikeFreq)
            plt.sca(axs[fluor_idx])
            plt.plot(celln.ephys.I_inj, celln.ephys.spikeFreq, '.-', ms=5, lw=1, alpha=0.5, color=colors[fluor_idx])

    for ix, f in enumerate(avFI):
        plt.sca(axs[ix])
        plt.plot(celln.ephys.I_inj, np.mean(f, axis=0), 'o-', color=colors[ix], ms=8, lw=4)
        plt.title(fluors_acc[ix]+ ' (n= %i)'%len(f))
        plt.xlabel('I$\mathrm{_{inj}}$ (nA)')

    plt.sca(axs[0]); plt.ylabel('Spike Frequency (Hz)');
    return fig

def plot_ephys_summary(df_ephys, figsize=(12,12)):
    fig,axs=plt.subplots(nrows=4, ncols=3, sharey=False, figsize=figsize)
    axs[-1][-1].axis('off');
    fluors=['NF', 'PV+', 'SST+']
    colors = ['#F5A623', '#4A90E2', '#7ED321']
    cols = df_ephys.columns[2:]

    for ix,col in enumerate(cols):
        plt.sca(np.ravel(axs)[ix])
        df_tmp = df_ephys[['Fluorescence', col]]
        plt.title(col, fontsize=14)

        for jx,fl in enumerate(fluors):
            vals = df_tmp[df_tmp['Fluorescence']==fl][col]
            plt.plot(jx + np.random.randn(len(vals))*0.1, vals, '.', ms=8, color=colors[jx], alpha=0.5)

            # Reduce range of plot with outliers
            if col=='Adaptation Ratio':
                plt.ylim(0,11)
                plt.plot(jx, vals.median(), 's', ms=10, color=colors[jx])
            else:
                plt.plot(jx, vals.mean(), 's', ms=10, color=colors[jx])

        plt.xticks(range(3), fluors)
    fig.subplots_adjust(hspace=0.4)
    return fig

def plot_ephys_scatter(df, figsize=(12,12)):
    axs = scatter_matrix(df, alpha=0.2,figsize=(13,13), diagonal='kde')
    for ax in np.ravel(axs):
        ax.set_xlabel(ax.get_xlabel().replace(' ', '\n'), fontsize=11)
        ax.set_ylabel(ax.get_ylabel().replace(' ', '\n'), fontsize=11)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_label_coords(-0.4, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.4)
    return plt.gcf()

def plot_syn_reliability(df_rel, figsize=(3,4)):
    fluors = ['NF', 'PV+', 'SST+']
    fig = plt.figure(figsize=figsize)
    colors = ['#F5A623', '#4A90E2', '#7ED321']
    for ix,fl in enumerate(fluors):
        rel = df_rel[df_rel['Fluorescence'] == fl]
        plt.plot(ix+np.random.randn(len(rel))*0.1, rel['Reliability'], '.', ms=10, color=colors[ix])

    plt.hlines(y=0.9, xmin=-0.25, xmax=2.25, lw=1, linestyle='--')
    plt.xticks(range(3), fluors)
    plt.xlabel('Cell type')
    plt.ylabel('Reliability')
    plt.title('Monosynaptic\nConnection\nReliability')
    return fig

def plot_syn_delays(df_syn, kind='Mean', figsize=(10, 3)):
    fluors = ['NF', 'PV+', 'SST+']
    fig, axs = plt.subplots(figsize=figsize, ncols=3, sharex= True, sharey = True)
    colors = ['#F5A623', '#4A90E2', '#7ED321']
    for ix,fl in enumerate(fluors):
        dly = df_syn[df_syn['Fluorescence'] == fl]
        plt.sca(axs[ix])
        plt.hist(dly['%s Delay'%kind], density = True, color=colors[ix], histtype='stepfilled', alpha=0.5)
        plt.ylabel('P(Delay)')
        plt.xlabel('Response Delay (ms)')
        plt.title(fl)
    return fig

def plot_bar(df, num_comparisons=1, figsize=(3,4), tests=None):
    fluors = ['NF', 'PV+', 'SST+']
    colors = ['#F5A623', '#4A90E2', '#7ED321']
    width = 0.8
    alpha = 0.6
    spread = 0.1

    fig = plt.figure(figsize=(3,4))
    plt.bar(x = np.arange(len(fluors)),
            height = df.groupby('Fluorescence').mean().squeeze(),
            # yerr = df.groupby('Fluorescence').std().squeeze(),
            width = width,
            alpha=alpha,
            color=colors)

    plt.xticks(range(len(fluors)), fluors)
    
    colname = [col for col in df.columns if col is not 'Fluorescence'][0]

    for fx, fl in enumerate(fluors):
        vals = df[df['Fluorescence'] == fl][colname]
        locs = np.random.randn(len(vals))*spread + fx
        plt.plot(locs, vals, 'o', color=colors[fx])
    plt.xlabel('Cell Type')
    if tests:
        test_string = test_to_string(tests, num_comparisons = num_comparisons )
        fig.set_size_inches((6,4))
        fig.subplots_adjust(right=0.5)
        ax_results = fig.add_axes([0.6,0.125,0.375,0.775])
        plt.sca(ax_results)
        n_cells = df.groupby('Fluorescence').count().as_matrix().squeeze()
        n_cells_string = ''.join(['%s n= %i\n'%(fl, nc) for fl, nc in zip(fluors, n_cells)])
        plt.text(s= n_cells_string + test_string, x= 0.0, y= 0.15, fontsize= 14)
        plt.axis('off')
    return fig

def plot_MI(df, title_string, num_comparisons, figsize=(3,4), tests=None):
    fig = plot_bar(df, num_comparisons=num_comparisons, figsize=(3,4), tests=tests)
    plt.title(title_string)
    plt.ylabel('%sMutual Information (bits)'%('Conditional\n' if 'Rate' in title_string else ''))
    return fig

def plot_cutoffs(df, figsize=(3,4), tests=None):
    fig = plot_bar(df, figsize=(3,4), tests=tests)
    plt.title('Early-Late Cutoff')
    plt.ylabel('Time (ms)')
    return fig

def test_to_string(test_dict, num_comparisons):
    result =  'Bonferroni corrected\n%s%s\n'%(test_dict['Main']['Test'],
                                               ' (Levene\'s p = %.3f)'%test_dict['Main']['Levenes']
                                               if test_dict['Main']['Levenes'] <= 0.05
                                               else '')
    
    result += 'p = %.3f,\n%s\n\n'%(test_dict['Main']['p'],
                                   'Significant at p<=%.3f'%(0.05/num_comparisons)
                                   if test_dict['Main']['p']/num_comparisons else 'n.s.')
    
    result += 'post-hoc tests (significant at p<= %.3f)'%(0.05/len(test_dict['Pairs']))
    for pair in test_dict['Pairs'].keys():
        result += '\n%s: %s %.3f'%(pair,
                                   '%s%s'%(test_dict['Pairs'][pair]['Test'],
                                           ' (Levene\'s p = %.3f)'%test_dict['Pairs'][pair]['Levenes']
                                           if test_dict['Pairs'][pair]['Levenes'] <= 0.05
                                           else ''),
                                   test_dict['Pairs'][pair]['p'])
        
    return result
            
def plot_pca(df, figsize=(15, 5)):
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=figsize)
    pairs_ix = list(combinations(range(len(df.columns)-1), r=2))
    Fluorescence = ['NF', 'PV+', 'SST+']
    colors = ['#F5A623', '#4A90E2', '#7ED321']
    for ax, (ix1, ix2) in zip(axs, pairs_ix):
        plt.sca(ax)
        plt.xlabel('Principal Component %i'%(ix1+1), fontsize = 15)
        plt.ylabel('Principal Component %i'%(ix2+1), fontsize = 15)
        
        for target, color in zip(Fluorescence,colors):
            indicesToKeep = df['Fluorescence'] == target
            plt.scatter(df.loc[indicesToKeep, 'principal component %i'%(ix1+1)],
                       df.loc[indicesToKeep, 'principal component %i'%(ix2+1)],
                       c = color,
                       s = 100)
    plt.legend(Fluorescence)
    return fig

def plot_dendrogram(df, figsize=(25, 10)):
    
    x = df.select_dtypes(np.number).as_matrix()
    imp = Imputer(strategy="mean", axis=0)
    scale = StandardScaler()
    x = scale.fit_transform(imp.fit_transform(x))

    #ward's linkage dendrogram
    Z = linkage(x, 'ward')
    fig= plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels=df['Fluorescence']
    )
    ax = fig.axes[0]
    
    fluors = ['NF', 'PV+', 'SST+']
    colors = ['#F5A623', '#4A90E2', '#7ED321']
    for tl in ax.xaxis.get_ticklabels():
        color = [c for c,fl in zip(colors, fluors) if fl == tl.get_text()][0]
        tl.set_color(color)

    ax.set_xlabel('Cell type', fontsize=16)
    ax.set_ylabel('Distance', fontsize=16)
    
    return fig

def pi_tick_labels(nticks = 7):
    '''
    Return ticks and tick labels in the bound of [0, 2 pi]
    
    Arguments :
        nticks (int) : number of ticks
    
    Returns :
        ticks (np.array)   : tick locations
        tick_labels (list) : list of strings containing tick labels
    '''
    
    from fractions import Fraction
    tick_labels = ['0']
    for ix in range(1, nticks):
        F = Fraction(2*ix, nticks-1)
        num = F.numerator
        den = F.denominator
        tick_string = r'$\pi$'
        if num > 1:
            tick_string = r'%i'%num + tick_string
        if den > 1:
            tick_string = tick_string + r'/%i'%den
        tick_labels.append(tick_string)
        
    ticks = np.linspace(0, 2*np.pi, nticks)
    return ticks, tick_labels

def plot_mixture_cutoff(X, M, lognormal=False, linspace=1000, cutoff=None, figsize=(6, 4)):
    x = np.linspace(0.001, 2*np.pi, 1000)
    if lognormal:
        X = np.exp(X)
        px = log_gaussian_mixture(x, M)
    else:
        px = gaussian_mixture(x, M)
    
    fig = plt.figure(figsize=figsize)
    plt.hist(X, bins='auto', density=True)
    plt.plot(x, px)
    if cutoff:
        plt.vlines(x=cutoff*np.pi/25, ymin=0, ymax=px.max(), color='C7')
    plt.xlabel('Spike Phase')
    plt.ylabel('Probability')
    
    return fig