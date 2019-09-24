'''
Author: Luke Prince
Date: 11 January 2019
'''

import os
import sys
from collections import OrderedDict

from ephys import *
from synapses import *
from stats import *
from .plots import add_inset_ax, plot_mixture_cutoff
from .utils import load_episodic, resample, hcf

from quantities import pF, MOhm, nA, mV, ms, Hz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.optimize import brentq

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

class Cell(object):
    def __init__(self, celldir):
        self.celldir  = celldir
        self.cellid   = os.path.basename(self.celldir)
        self.fluor    = pd.read_csv(celldir+'/Metadata_%s.csv'%self.cellid, nrows=1)['Celltype'].squeeze()
        self.layer    = pd.read_csv(celldir+'/Metadata_%s.csv'%self.cellid, nrows=1)['Layer'].squeeze()
        
    def extract_ephys(self, pulse_on, pulse_len, sampling_frequency, I_min, I_max, I_step):
        self.ephys   = Electrophysiology(celldir = self.celldir, 
                                         pulse_on = pulse_on, pulse_len = pulse_len,
                                         sampling_frequency = sampling_frequency,
                                         I_min = I_min, I_max = I_max, I_step = I_step)
        self.ephys.extract_features()
        if not os.path.exists(self.celldir+'/ephys_features.svg'):
            self.ephys.plot(closefig=True)
        
        return self.ephys.results
    
    def extract_syns(self, mono_winsize, total_winsize, sampling_frequency):
        self.syns = Synapses(celldir = self.celldir,
                             mono_winsize = mono_winsize, total_winsize=total_winsize,
                             sampling_frequency = sampling_frequency)
        self.syns.extract_features()
        if not os.path.exists(self.celldir+'/syn_features.svg'):
            self.syns.plot(closefig=True)
        
    def initialize_MI(self, stim_winsize, mono_winsize, delay, start, dur, sampling_frequency):
        
        self.rate_code = Code(celldir= self.celldir, code_type= 'rate', delay = delay,
                              mono_winsize= mono_winsize, stim_winsize= stim_winsize,
                              start = start, dur = dur, sampling_frequency = sampling_frequency)
        
        self.temp_code = Code(celldir= self.celldir, code_type= 'temp', delay = delay,
                              mono_winsize= mono_winsize, stim_winsize= stim_winsize,
                              start = start, dur = dur, sampling_frequency = sampling_frequency)
        
        self.cutoff = self.estimate_early_late_cutoff(bayes=True, fit_lognormal=True, n_inits=30, plot=True)
        self.rate_code.early_winsize = self.cutoff
        self.temp_code.early_winsize = self.cutoff
        
    def estimate_MI(self):
        
        self.rate_code.estimate_mutual_information()
        if self.rate_code.data_exists:
            self.rate_code.plot(closefig=True)
            self.rate_code.hist(closefig=True, savefig=True)
        
        self.temp_code.estimate_mutual_information()
        if self.temp_code.data_exists:
            self.temp_code.plot(closefig=True)
            self.temp_code.hist(closefig=True)
            
        plt.close('all')
        
    def estimate_early_late_cutoff(self, bayes=False, fit_lognormal=False, n_inits=30, plot=True):
        spikeTimes = np.array([])
        if hasattr(self, 'rate_code'):
            if hasattr(self.rate_code, 'spikeTimes'):
                spikeTimes = np.concatenate((spikeTimes, np.concatenate(self.rate_code.spikeTimes['all'])))
        if hasattr(self, 'temp_code'):
            if hasattr(self.temp_code, 'spikeTimes'):
                spikeTimes = np.concatenate((spikeTimes, np.concatenate(self.temp_code.spikeTimes['all'])))

        if np.any(spikeTimes) and len(spikeTimes) > 10:
            phases = spikeTimes_to_phase(spikeTimes= spikeTimes, frequency=20)

            if fit_lognormal:
                X = np.log(phases)
            else:
                X = phases

            if bayes:
                model = BayesianGaussianMixture
            else:
                model = GaussianMixture

            M1 = model(n_components=1, n_init=n_inits).fit(X.reshape(-1, 1))
            M2 = model(n_components=2, n_init=n_inits).fit(X.reshape(-1, 1))

            M2_accept = M2.lower_bound_ > M1.lower_bound_
            M2_reject_1 = np.logical_or(np.all(M2.predict(M2.means_)==1),
                                        np.all(M2.predict(M2.means_)==0))
            M2_reject_2 = np.isclose(*M2.means_.squeeze(), rtol=1e-1)
            M2_reject = M2_reject_1 or M2_reject_2


            if M2_accept and not M2_reject:
                f = lambda x : M2.predict_proba(np.reshape(x, (1, -1)))[:, 0] - 0.5
                cutoff = brentq(f, a= M2.means_.squeeze()[0], b = M2.means_.squeeze()[1])
                if fit_lognormal:
                    cutoff = np.exp(cutoff) * 25 / np.pi

                else:
                    cutoff*= 25/np.pi

                if plot:
                    fig = plot_mixture_cutoff(X, M2, lognormal=fit_lognormal, cutoff=cutoff)
                    plt.title('%s %s'%(self.fluor, self.cellid))
                    fig.savefig(self.celldir+'/mixture_cutoff.svg')

                return cutoff
            else:
                if plot:
                    fig = plot_mixture_cutoff(X, M1, lognormal=fit_lognormal)
                    plt.title('%s %s'%(self.fluor, self.cellid))
                    fig.savefig(self.celldir+'/mixture_cutoff.svg')
                return np.nan

        return np.nan
        
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
    
class CellCollection(OrderedDict):
    def __init__(self, data_path, *args):
        OrderedDict.__init__(self, args)
        self.data_path = data_path
        
    def __getitem__(self, key):
        
        if type(key) is int:
            key = OrderedDict.keys()[key]
            
        return OrderedDict.__getitem__(self, key)
    
    def __setitem__(self, cell):
        OrderedDict.__setitem__(self, cell.cellid, cell)
        
    def add(self, cell):
        self.__setitem__(cell)
                
    def collect_results(self):
        self.collect_metadata()
        self.collect_ephys()
        self.collect_syns()
        self.collect_MI()
                
    def collect_metadata(self):
        self.metadata = {'Fluorescence':[],'Layer':[]}
        
        for cellid, celln in self.items():
            self.metadata['Fluorescence'].append(celln.fluor)
            self.metadata['Layer'].append(celln.layer)
                
    def collect_ephys(self):
        self.ephys = {'Vrest (mV)': [],'Input Resistance (megaohm)': [],'Cell Capacitance (pF)': [],
                      'Rheobase (nA)':[],'fI slope (Hz/nA)':[],
                      'Adaptation Ratio':[],'Sag Amplitude (mV)':[],
                      'Spike Threshold (mV)':[],'Spike Amplitude (mV)':[],
                      'Spike Halfwidth (ms)':[],'Membrane Time Constant (ms)':[]}
        
        for cellid, celln in self.items():
            if hasattr(celln, 'ephys'):
                for key, val in celln.ephys.results.items():
                    try:
                        self.ephys[key+' (%s)'%(str(val.units).split(' ')[1])].append(val.item())
                    except AttributeError:
                        self.ephys[key].append(val)
                        
    def collect_syns(self):
        self.syns = {'Reliability' : [], 'Mean Delay' : [], 'Max Delay' : []}
        for cellid, celln in self.items():
            if hasattr(celln, 'syns'):
                self.syns['Reliability'].append(celln.syns.reliability)
                self.syns['Mean Delay'].append(celln.syns.mean_delay)
                self.syns['Max Delay'].append(celln.syns.max_delay)
                
    def collect_MI(self):
        self.MI = {'Rate' : {'Spikes'     : {'mono' : [], 'poly' : [], 'early': [] , 'late' : [], 'all' : []},
                             'Average Vm' : {'mono' : [], 'poly' : [], 'all' :  []},
                             'Initial Slope' : []},
                   'Temporal' : {'Spikes' : {'mono' : [], 'poly' : [], 'early': [] , 'late' : [], 'all' : []},
                             'Average Vm' : {'mono' : [], 'poly' : [], 'all' :  []},
                             'Initial Slope' : []},
                   'cutoff' : [] }
        
        for cellid, celln in self.items():
            self.MI['cutoff'].append(celln.cutoff)
            
            if hasattr(celln, 'rate_code'):
                for key_a, item_a in self.MI['Rate'].items():
                    if type(item_a) is list:
                        self.MI['Rate'][key_a].append(celln.rate_code.results[key_a])
                    elif type(item_a) is dict:
                        for key_b, item_b in self.MI['Rate'][key_a].items():
                            self.MI['Rate'][key_a][key_b].append(celln.rate_code.results[key_a][key_b]) 
                                
            if hasattr(celln, 'temp_code'):
                for key_a, item_a in self.MI['Temporal'].items():
                    if type(item_a) is list:
                        self.MI['Temporal'][key_a].append(celln.temp_code.results[key_a])
                    elif type(item_a) is dict:
                        for key_b, item_b in self.MI['Temporal'][key_a].items():
                            self.MI['Temporal'][key_a][key_b].append(celln.temp_code.results[key_a][key_b]) 

class Synapses(Cell):
    def __init__(self, celldir, mono_winsize, total_winsize, sampling_frequency):
        super(Synapses, self).__init__(celldir)
        
        self.mono_winsize = mono_winsize
        self.mono_winsteps = int(mono_winsize*sampling_frequency)
        self.total_winsize = total_winsize
        self.total_winsteps = int(total_winsize*sampling_frequency)
        self.sampling_frequency = sampling_frequency # kHz
        self.time = np.arange(self.total_winsteps, dtype=float)/self.sampling_frequency
        
        self.presyn_ids = np.loadtxt(self.celldir+'/presyn_ids.csv', delimiter=',', dtype=int)
        self.synapses = dict([(ix, {'id' : self.presyn_ids[ix]}) for ix in range(10)])

    def get_data(self):
        data = load_episodic(self.celldir + '/Synapse_finder.abf')
        Vm = data[0][:, self.presyn_ids, 0]
        pulse = data[0][:, :, 2]
        return Vm, pulse
        
    def extract_features(self):
        Vm, pulse = self.get_data()
        self.pulse_on = np.arange(1, len(pulse))[np.logical_and(pulse[1:, 0]>0.1, pulse[:-1, 0]<=0.1)]
        
        for synapse_id, synapse in self.synapses.items():
            synapse['data']     = np.array([Vm[p:p+self.total_winsteps, synapse_id].T for p in self.pulse_on])
            synapse['pEarlyAP'] = 0.0
            synapse['pMono']    = 0.0
            synapse['delay']    = []
            synapse['slope']    = []
            synapse['mono_fit'] = []
            
            '''Estimate direct response properties'''
            
            for trace in synapse['data']:
                pars, pmodel, RSS, baseline = fitPiecewiseLinearFunction(self.time[:self.mono_winsteps], trace[:self.mono_winsteps])
                synapse['pEarlyAP'] += np.any(trace[:self.mono_winsteps*2]>-25)*1.0/len(synapse['data'])
                if pmodel >= 0.99:
                    synapse['pMono'] += 1./len(synapse['data'])
                    synapse['slope'].append(pars[0])
                    synapse['delay'].append(pars[1])
                    synapse['mono_fit'].append(generalLinearDiscontinuousFn(self.time[:self.mono_winsteps], baseline, *pars))
                else:
                    synapse['delay'].append(np.nan)
                    synapse['slope'].append(0.0)
                    synapse['mono_fit'].append(baseline)
                
            synapse['delay'] = np.nanmean(synapse['delay'])
            synapse['slope'] = np.nanmean(synapse['slope'])
                
            '''Estimate network properties'''
            avTrace = np.median(synapse['data'], axis=0)
            Vstart  = np.mean(avTrace[:10])
            V0      = np.mean(avTrace[-50:])
            latency = 5.0 if np.isnan(synapse['delay']) else synapse['delay']
            
            optoResponse_fixdelay = lambda t, Vss, taur, taud : optoResponse(t, Vss, taur, taud, latency, V0)
            popt, pcov = curve_fit(optoResponse_fixdelay, self.time, avTrace, p0=[np.percentile(trace,90),10,25])
            synapse['taur']    = popt[1]
            synapse['taud']    = popt[2]
            synapse['Vinc']    = popt[0] - Vstart
            synapse['net_fit'] = optoResponse_fixdelay(self.time, popt[0], popt[1], popt[2])
            
        self.reliability = np.mean([self.synapses[ix]['pMono'] for ix in range(len(self.synapses))])
        self.mean_delay  = np.nanmean([self.synapses[ix]['delay'] for ix in range(len(self.synapses))])
        self.max_delay   = np.nanmax([self.synapses[ix]['delay'] for ix in range(len(self.synapses))])
    
    def plot(self, figsize=(12, 4.5), savefig=True, closefig=False):
        fig,axs = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,figsize=figsize)
        axs = np.ravel(axs)
        for synapse_id, synapse in self.synapses.items():
            plt.sca(axs[synapse_id])
            plt.plot(self.time, synapse['data'].T, lw=1, alpha=0.5, color='C0')
            plt.title('Input #%i'%(synapse['id']+1))
            
            if synapse_id >= 5:
                plt.xlabel('Time (ms)')
            if synapse_id%5 == 0:
                plt.ylabel('Voltage (mV)')
                
            plt.plot(self.time, synapse['net_fit'], color='C1')
            
            ax_inset = add_inset_ax(axs[synapse_id], rect=[0.5, 0.5, 0.4, 0.4])
            plt.sca(ax_inset)
            plt.plot(self.time[:self.mono_winsteps], synapse['data'][:, :self.mono_winsteps].T, lw=1, alpha=0.5, color='C0')
            plt.title('P(mono)=%.1f'%synapse['pMono'], fontsize=10)

            for rfit in synapse['mono_fit']:
                plt.plot(self.time[:self.mono_winsteps], rfit, color='C1', lw=1, alpha=0.5)
                
        if savefig:
            fig.savefig(self.celldir+'/syn_features.svg')
            
        if closefig:
            fig.clf()
            plt.close()
            
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------            
            
class Electrophysiology(Cell):
    
    def __init__(self, celldir, pulse_on, pulse_len, sampling_frequency, I_min, I_max, I_step):
        super(Electrophysiology, self).__init__(celldir)
        self.pulse_on           = pulse_on
        self.pulse_off          = pulse_on + pulse_len
        self.pulse_len          = pulse_len
        self.pulse_off_ix       = int(self.pulse_off*sampling_frequency)
        self.pulse_on_ix        = int(self.pulse_on*sampling_frequency)
        self.sampling_frequency = sampling_frequency
        self.I_min              = I_min
        self.I_max              = I_max
        self.I_step             = I_step
        self.I_inj              = np.arange(I_min, I_max + I_step, I_step)
        
    #------------------------------------------------------------------
    #------------------------------------------------------------------
        
    def get_data(self):
        return load_episodic(self.celldir+'/Ic_step.abf')[0][:,:,0]
        
    #------------------------------------------------------------------
    #------------------------------------------------------------------

    def extract_features(self):
        data            = self.get_data()
        self.total_time = len(data)/self.sampling_frequency
        self.time       = np.arange(0, self.total_time, 1./self.sampling_frequency)

        '''True/False threshold crossing at each index'''
        threshCross = np.array([findThreshCross(trace) for trace in data.T])

        '''Count number of threshold crossings'''
        self.spikeCount = np.array([np.count_nonzero(tc) for tc in threshCross])

        '''Estimate Spike Frequency from number of threshold crossings'''
        self.spikeFreq = self.spikeCount*1000./self.pulse_len # in Hz

        '''Estimate resting Membrane Potential'''
        self.v_rest = resting_MP(data[:self.pulse_on_ix]) # mV

        '''Estimate Sag Amplitude'''
        self.v_sag = sag_amplitude(data[self.pulse_on_ix:self.pulse_off_ix,0]) # mV

        '''Input resistance measurement from traces where there was no threshold cross, and I>0'''
        self.Rin = np.mean([input_resistance(trace[self.pulse_on_ix:self.pulse_off_ix],I,self.v_rest) 
                            for (trace, I, crossed) in zip(data.T, self.I_inj, np.any(threshCross,axis=1))
                            if ((np.abs(I)>0) and not crossed)]) # MOhm

        '''Membrane time constant measurement from traces for 100ms after pulse off with no threshold cross, and I>0'''
        self.taum = np.mean([membrane_tau(trace[self.pulse_off_ix:self.pulse_off_ix+int(self.sampling_frequency*100)],
                                          self.sampling_frequency)
                             for (trace, I, crossed) in zip(data.T, self.I_inj, np.any(threshCross,axis=1))
                             if ((np.abs(I)>0) and not crossed)]) # ms

        '''Membrane capacitance calculation from Rin and taum'''
        self.Cm = 1000*self.taum/self.Rin # pF

        '''Estimate Slope of FI curve. Use I_inj at first non-zero spike count and initial increase as
           starting parameters for search. Use only max 5 points above rheobase to prevent attempting
           to fit to adapting FI curves.
        '''
        if self.spikeFreq.any():
            rheo_p0       = rheobase(self.spikeFreq, self.I_inj)
            slope_p0      = np.diff(np.nonzero(self.spikeFreq)[0])[0]/self.I_step

            rheo, fISlope = estimatefISlope(self.spikeFreq, self.I_inj, p0=[rheo_p0,slope_p0])

            self.rheobase = rheo
            self.fISlope  = fISlope
        else:
            self.rheobase = np.nan
            self.fISlope = np.nan

        '''Get spike times from threshold crossings'''
        self.spikeTimes = [self.time[1:][tc] for tc in threshCross]

        '''Estimate adaptation ratio if there are at least three spikes in a sweep'''
        self.adaptation_ratio = adaptation_ratio(self.spikeTimes[-1])

        '''Indices for spike times'''
        spikeix = [(sp*self.sampling_frequency).astype('int') for sp in self.spikeTimes]

        '''For each spike...'''
        ''' TODO: Make spike window selection smarter'''
        v_thresh = []
        v_amp = []
        v_ahp = []
        spikeHalfWidth = []
        for sp,trace in zip(spikeix, data.T):
            if np.any(sp):
                for ix in sp:
                    spikeTrace = trace[ix-29:ix+30] # Extract spike
                    v_thresh.append(spike_threshold(spikeTrace[:30])) # estimate spike threshold
                    v_amp.append(spike_amplitude(spikeTrace, v_thresh[-1])) # estimate spike amplitude
                    v_ahp.append(ahp_amplitude(spikeTrace, v_thresh[-1])) # estimate ahp amplitude
                    spikeHalfWidth.append(spike_halfwidth(spikeTrace, v_thresh[-1], v_amp[-1], self.sampling_frequency)) # estimate spike halfwidth

        self.v_thresh = np.mean(v_thresh)
        self.v_amp = np.mean(v_amp)
        self.v_ahp = np.mean(v_ahp)
        self.spikeHalfWidth = np.mean(spikeHalfWidth)

        self.results = {'Vrest': self.v_rest*mV,'Input Resistance': self.Rin*MOhm,'Cell Capacitance': self.Cm*pF,
                        'Rheobase':self.rheobase*nA,'fI slope':self.fISlope*Hz/nA, #'AHP amplitude' : self.ahpAmp*mV,
                        'Adaptation Ratio':self.adaptation_ratio,'Sag Amplitude':self.v_sag*mV,
                        'Spike Threshold':self.v_thresh*mV,'Spike Amplitude':self.v_amp*mV,
                        'Spike Halfwidth':self.spikeHalfWidth*ms,'Membrane Time Constant':self.taum*ms}
        
    #------------------------------------------------------------------
    #------------------------------------------------------------------

    def plot(self, num_traces=3, fig_height=4, include_fI = True, include_results = True, savefig = True, closefig=False):
        data = self.get_data()

        width_ratio = 3;
        if include_fI or include_results:
            include_ratio = ((width_ratio, True), (1, include_fI), (1, include_results))
            width_ratio = tuple([width for width,inc in include_ratio if inc])

            fig_width = fig_height * 4. * np.sum(width_ratio)/5.
            fig, axs = plt.subplots(ncols=len(width_ratio), figsize=(fig_width, fig_height), gridspec_kw={'width_ratios' : width_ratio})

            plt.sca(axs[0])

        else:
            fig_width = fig_height * 2.4
            fig = plt.figure(figsize=(fig_width, fig_height))

        idxs = np.array([i in np.percentile(self.I_inj,
                                            np.linspace(0, 100, num_traces),
                                            interpolation='nearest')
                         for i in self.I_inj])
        
        for ii,trace in enumerate(data.T[idxs]):
            plt.plot(self.time, trace, color = plt.cm.plasma(np.linspace(0, 1, num_traces)[ii]), lw=1, zorder=-ii)
            
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend(['I = %.2f'%I for I in self.I_inj[idxs]])
        
        plt.title('Cell ID: %s'%self.celldir+' %s'%self.fluor +' %s'%self.layer,fontsize=20)
        
        if include_fI:
            plt.sca(axs[1])
            plt.plot(self.I_inj, self.spikeFreq, 'o', ms=10)
            I_fine = np.linspace(self.I_min-self.I_step, self.I_max+self.I_step, 1000)
            plt.plot(I_fine, fICurve(I_fine, self.rheobase, self.fISlope), color='rebeccapurple', zorder=-1)
            plt.xlabel('I$\mathrm{_{inj}} (nA)$')
            plt.ylabel('Spike Frequency (Hz)')
            
        if include_results:
            plt.sca(axs[-1])
            plt.axis('off')
            Y = np.linspace(0.05,0.9,len(self.results.keys()))
            for ix,key in enumerate(self.results.keys()):
                plt.text(x = 0.1,y = Y[ix],s = key+' = ' + str(np.round(self.results[key], 2)),
                         transform=axs[-1].axes.transAxes, fontsize=14)
                
        if savefig:
            fig.savefig(self.celldir+'/ephys_features.svg')
            
        if closefig:
            fig.clf()
            plt.close()
            
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

class Code(Cell):
    def __init__(self, celldir, code_type, delay, stim_winsize, mono_winsize, start, dur, sampling_frequency):
        super(Code, self).__init__(celldir)
        assert code_type == 'rate' or code_type == 'temp', 'code_type \'%s\' not recognised'%code_type
        
        self.code_type          = code_type
        self.file_type          = 'Rate' if self.code_type == 'rate' else 'Temporal' if self.code_type == 'temp' else None
        
        self.sampling_frequency = sampling_frequency # kHz
        
        self.start              = start # secs
        self.dur                = dur   # secs
        
        self.exp_start          = int(start * self.sampling_frequency * 1000)
        self.exp_dur            = int(dur * self.sampling_frequency * 1000)
        self.exp_end            = self.exp_start + self.exp_dur
        
        self.stim_winsize       = stim_winsize
        self.mono_winsize       = mono_winsize
        self.delay              = delay
        
        self.stim_winsteps      = int(self.stim_winsize*self.sampling_frequency)
        self.mono_winsteps      = int(self.mono_winsize*self.sampling_frequency)
        self.delaysteps         = int(self.delay*self.sampling_frequency)
        
        self.time               = np.arange(0, self.dur, self.stim_winsize/1000.)
        self.time_trace         = np.arange(self.exp_dur, dtype=float)/(self.sampling_frequency*1000)
        
        self.stimulus           = np.loadtxt('./stimulus/%scode.csv'%self.code_type, delimiter=',', skiprows=1)[:, 1:]/50
        
        sigBins,signal          = np.loadtxt('./stimulus/stim_signal.csv', delimiter= ',')
        self.signal             = resample(signal,
                                           cf= 1./np.diff(sigBins).min(),
                                           nf= 1000/self.stim_winsize)

        self.data_exists        = os.path.exists('./%s/%s.abf'%(self.celldir, self.file_type))
        
        self.get_spikeTimes()
            
    def get_data(self):
        if self.data_exists:
            return load_episodic('./%s/%s.abf'%(self.celldir, self.file_type))[0][self.exp_start:self.exp_end, :, 0]
        
    def get_spikeTimes(self):
        if self.data_exists:
            data = self.get_data()
            threshCrosses = findThreshCross(data)
            self.spikeTimes = {'all' : [self.time_trace[1:][tc] for tc in threshCrosses.T]}
    
    def estimate_mutual_information(self):
        if self.data_exists:
            
            data = self.get_data()
            
            self.results = {'Spikes'     : {},
                            'Average Vm' : {},
                            'Initial Slope' : []}
            
            delay_secs         = self.delay/1000.
            stim_winsize_secs  = self.stim_winsize/1000.
            mono_winsize_secs  = self.mono_winsize/1000.
            early_winsize_secs = self.early_winsize/1000.
            hcf_winsteps = hcf(self.stim_winsteps, self.mono_winsteps)
            
            hcf_winsize_secs = hcf_winsteps/(self.sampling_frequency*1000.)
            

            self.extract_slopes()
            bins_all  = np.append(arr= self.time, values=self.dur).astype('float')
            bins_mono = np.arange(0, self.dur + hcf_winsize_secs, hcf_winsize_secs, dtype=float)
            binned_vm = binned_statistic(x = self.time_trace - delay_secs,
                                         values = data.T,
                                         statistic='mean',
                                         bins=bins_mono)[0]
            mono_mask = (bins_mono[:-1]/hcf_winsize_secs).round().astype('int') % int(self.stim_winsteps/hcf_winsteps) < int(self.mono_winsteps/hcf_winsteps)

            self.average_Vm = {}

            if int(self.mono_winsteps/hcf_winsteps) == 1:
                self.average_Vm['mono'] = binned_vm[:, mono_mask]
            else:
                self.average_Vm['mono'] = binned_statistic(x = bins_mono[:-1][mono_mask],
                                                           values = binned_vm[:, mono_mask],
                                                           statistic=np.nanmean, bins=bins_all)[0]
            self.average_Vm['poly'] = binned_statistic(x = bins_mono[:-1][~mono_mask],
                                                       values=binned_vm[:, ~mono_mask],
                                                       statistic=np.nanmean, bins=bins_all)[0]
            self.average_Vm['all']  = binned_statistic(x = bins_mono[:-1],
                                                       values=binned_vm,
                                                       statistic=np.nanmean, bins=bins_all)[0]


            threshCrosses = findThreshCross(data)
            self.spikeTimes['mono']  = [spT[spT % stim_winsize_secs <  (mono_winsize_secs + delay_secs)] for spT in self.spikeTimes['all']]
            self.spikeTimes['poly']  = [spT[spT % stim_winsize_secs >= (mono_winsize_secs + delay_secs)] for spT in self.spikeTimes['all']]
            
            self.spikeTimes['early'] = [spT[spT % stim_winsize_secs < early_winsize_secs] for spT in self.spikeTimes['all']]
            self.spikeTimes['late']  = [spT[spT % stim_winsize_secs >= early_winsize_secs] for spT in self.spikeTimes['all']]
            
            self.spikeCounts = {}

            for key, times in self.spikeTimes.items():
                self.spikeCounts[key] = [np.histogram(times_i, bins_all)[0] for times_i in times]


            if self.code_type == 'rate':
                # estimate mutual information conditioned on pre-synaptic spike count
                preSpikes = self.stimulus.sum(axis=1)

                # Initial Slope
                self.results['Initial Slope'] = conditionalMutualInformation(s = self.signal[self.slopes.nonzero()],
                                                                             r = self.slopes[self.slopes.nonzero()],
                                                                             c = preSpikes[self.slopes.nonzero()])

                # Spikes
                for key, postSpikes in self.spikeCounts.items():
                    self.results['Spikes'][key] = spikeConditionalMutualInformation(postSpikes=postSpikes,
                                                                                    signal=self.signal,
                                                                                    preSpikes=preSpikes)

                for key, Vm in self.average_Vm.items():
                    self.results['Average Vm'][key] = conditionalMutualInformation(s = np.tile(self.signal, Vm.shape[0]),
                                                                                   r = np.ravel(Vm),
                                                                                   c = np.tile(preSpikes, Vm.shape[0]))

            elif self.code_type == 'temp':
                # estimate mutual information

                # Initial Slope
                self.results['Initial Slope'] = mutualInformation(s = self.signal[self.slopes.nonzero()],
                                                                  r = self.slopes[self.slopes.nonzero()])

                # Spikes
                for key, postSpikes in self.spikeCounts.items():
                    self.results['Spikes'][key] = spikeMutualInformation(postSpikes=postSpikes,
                                                                         signal=self.signal)

                for key, Vm in self.average_Vm.items():
                    self.results['Average Vm'][key] = mutualInformation(s = np.tile(self.signal, Vm.shape[0]),
                                                                        r = np.ravel(Vm))
        else:
            self.results = {'Spikes'     : {'mono' : np.nan, 'poly' : np.nan, 'early' : np.nan, 'late' : np.nan, 'all' : np.nan},
                            'Average Vm' : {'mono' : np.nan, 'poly' : np.nan, 'all' : np.nan},
                            'Initial Slope' : np.nan}
            
                
    def extract_slopes(self):
        if self.data_exists:
            if os.path.exists('%s/%s_slopes.csv'%(self.celldir, self.file_type)):
                self.slopes = np.loadtxt('%s/%s_slopes.csv'%(self.celldir, self.file_type), delimiter=',')
            else:
                data = self.get_data()
                
                time = np.arange(self.mono_winsteps, dtype='float')/self.sampling_frequency
                self.slopes = []
                for ix,c in enumerate(self.stimulus):
                    if c.any():
                        sys.stdout.write('%i %% Done \r'%(np.round(ix*100./len(self.stimulus))))
                        syn_slope = []
                        for trace in data[self.stim_winsteps*ix:self.stim_winsteps*ix+self.mono_winsteps].T:
                            pars, pmodel, RSS, baseline = fitPiecewiseLinearFunction(time, trace)
                            if pmodel > 0.99:
                                syn_slope.append(pars[0])
                            else:
                                syn_slope.append(0)
                        self.slopes.append(np.mean(syn_slope))

                codeSlopes = np.zeros(len(self.stimulus))
                codeSlopes[self.stimulus.any(axis=1)] = np.array(self.slopes)
                self.slopes = codeSlopes
                np.savetxt('%s/%s_slopes.csv'%(self.celldir, self.file_type), codeSlopes, delimiter=',')
        else:
            raise IOError('Cannot extract slopes. No data exists for cell %s'%self.cellid)
    
    def hist(self, figsize=(6, 4), savefig=True, closefig=False):
        if self.data_exists:
            signal          = np.tile(self.signal, len(self.spikeCounts['all']))
            preSpikeCounts  = np.tile(self.stimulus.sum(axis=1), len(self.spikeCounts['all']))
            postSpikeCounts = np.concatenate(self.spikeCounts['all'])
            average_Vm       = np.concatenate(self.average_Vm['all'])
            
            bins_signal = np.arange(0, np.max(signal) + 1.5)
            bins_pre    = np.arange(0, np.max(preSpikeCounts) + 1.5)
            bins_post   = np.arange(0, np.max(postSpikeCounts) + 1.5)
            bins_avVm   = np.histogram(average_Vm, bins='auto')[1]
            
            fig_width, fig_height= figsize
            
            if self.code_type is 'rate':
                
                fig, axs = plt.subplots(figsize=(fig_width * 2, fig_height * 3), nrows=3, ncols=2)
                
                plt.sca(axs[0,0])
                plt.bar(0, 1-np.mean(signal))
                plt.bar(1, np.mean(signal))
                plt.xlabel('Signal State')
                
                plt.xticks([0, 1], ['Low', 'High'])
                plt.ylabel('Probability(State)')
                
                plt.sca(axs[0,1])
                plt.hist(preSpikeCounts[signal==0], bins=bins_pre, alpha=0.5, density=True)
                plt.hist(preSpikeCounts[signal==1], bins=bins_pre, alpha=0.5, density=True)

                plt.hist(preSpikeCounts[signal==0], bins=bins_pre, histtype='step', lw=5, color='C0', density=True)
                plt.hist(preSpikeCounts[signal==1], bins=bins_pre, histtype='step', lw=5, color='C1', density=True)

                plt.xlabel('Presynaptic Stimulus Count')
                plt.ylabel('Probability (Count | State)')

                legend = plt.legend(['Low', 'High'])
                plt.setp(legend.get_title(),fontsize=16)
                
                plt.sca(axs[1,0])
                plt.hist2d(preSpikeCounts[signal==0], postSpikeCounts[signal==0], bins=[bins_pre, bins_post], normed=True)
                plt.xlabel('Presynaptic Stimulus Count')
                plt.ylabel('Postsynaptic Stimulus Count')
                plt.colorbar(fraction=0.05, label='Pr(Pre Count, Post Count | State = Low)')
                
                plt.sca(axs[1,1])
                plt.hist2d(preSpikeCounts[signal==1], postSpikeCounts[signal==1], bins=[bins_pre, bins_post], normed=True)
                plt.xlabel('Presynaptic Stimulus Count')
                plt.ylabel('Postsynaptic Stimulus Count')
                plt.colorbar(fraction=0.05, label='Pr(Pre Count, Post Count | State = High)')
                
                plt.sca(axs[2,0])
                plt.hist2d(preSpikeCounts[signal==0], average_Vm[signal==0], bins=[bins_pre, bins_avVm], normed=True)
                plt.xlabel('Presynaptic Stimulus Count')
                plt.ylabel('Average Membrane Potential (mV)')
                plt.colorbar(fraction=0.05, label='Pr(Pre Count, V$_{mem}$ | State = Low)')
                
                plt.sca(axs[2,1])
                plt.hist2d(preSpikeCounts[signal==1], average_Vm[signal==1], bins=[bins_pre, bins_avVm], normed=True)
                plt.xlabel('Presynaptic Stimulus Count')
                plt.ylabel('Average Membrane Potential (mV)')
                plt.colorbar(fraction=0.05, label='Pr(Pre Count, V$_{mem}$ | State = High)')
                
            elif self.code_type is 'temp':
                
                fig, axs = plt.subplots(figsize=(fig_width * 2, fig_height), nrows=1, ncols=2)
                
                plt.sca(axs[0])
                plt.hist(postSpikeCounts[signal==0], bins=bins_post, alpha=0.5, density=True)
                plt.hist(postSpikeCounts[signal==1], bins=bins_post, alpha=0.5, density=True)
                plt.hist(postSpikeCounts[signal==0], bins=bins_post, histtype='step', lw=5, color='C0', density=True)
                plt.hist(postSpikeCounts[signal==1], bins=bins_post, histtype='step', lw=5, color='C1', density=True)
                
                plt.xlabel('Post-Synaptic Spike Count')
                plt.ylabel('Probability')
                legend = plt.legend(['Low', 'High'])
                plt.setp(legend.get_title(),fontsize=16)
                
                plt.sca(axs[1])
                plt.hist(average_Vm[signal==0], bins=bins_avVm, alpha=0.5, density=True)
                plt.hist(average_Vm[signal==1], bins=bins_avVm, alpha=0.5, density=True)
                plt.hist(average_Vm[signal==0], bins=bins_avVm, histtype='step', lw=5, color='C0', density=True)
                plt.hist(average_Vm[signal==1], bins=bins_avVm, histtype='step', lw=5, color='C1', density=True)
                
                plt.xlabel('Average Membrane Potential (mV)')
                plt.ylabel('Probability')
                legend = plt.legend(['Low', 'High'])
                plt.setp(legend.get_title(),fontsize=16)
            
            fig.subplots_adjust(wspace=0.35, hspace=0.3)
                
            if savefig:
                fig.savefig(self.celldir+'/%s_code_hists.svg'%self.code_type)
                
            if closefig:
                fig.clf()
                plt.close()
                
    def plot(self, figsize=(12, 10), savefig=True, closefig=False):
        if self.data_exists:
            data = self.get_data()
            
            fig, axs    = plt.subplots(nrows = 9, sharex=True, sharey=False, figsize=figsize,
                                       gridspec_kw = {'height_ratios' : (2, 1, 6, 1, 3, 3, 3, 3, 3)})
            
            ax_sig = axs[0]; ax_stim = axs[2]; ax_resp = axs[4:];
            axs[1].set_visible(False)
            axs[3].set_visible(False)
            
            plt.sca(ax_sig)
            ax_sig.xaxis.set_visible(False)
            ax_sig.spines['bottom'].set_visible(False)
            plt.plot(self.time, self.signal)
            plt.yticks([0,1],['Low', 'High'])
            plt.title('Signal', y=1.15)
            
            plt.sca(ax_stim)
            ix, n = np.nonzero(self.stimulus)
            plt.plot(self.time[ix], n+1, '|', ms=5)
            plt.yticks([1,10])
            plt.ylabel('Input #')
            plt.title('%s Code'%self.file_type)
            
            for ix, trace in enumerate(data.T):
                plt.sca(ax_resp[ix])
                plt.plot(self.time_trace, trace, lw=1)
                plt.yticks(np.linspace(-75, 75, 3))
                plt.ylabel('Trial %i'%(ix+1), fontsize=12)
                
            ax_resp[2].text(s='Membrane Potential (mV)', x = -0.125, y=0.5, fontsize=16, transform=ax_resp[2].transAxes,
                     rotation='vertical', verticalalignment='center', horizontalalignment='center')
            ax_resp[0].set_title('Response')
            plt.suptitle('Cell ID: %s %s %s, %s Code'%(self.cellid, self.fluor, self.layer, self.file_type))
            plt.xlabel('Time (secs)')
            
            if savefig:
                fig.savefig(self.celldir+'/%s_code_response.svg'%self.code_type)

            if closefig:
                fig.clf()
                plt.close()
            
        else:
            raise IOError('Cannot plot data. No data exists for cell %s'%self.cellid)