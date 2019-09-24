'''
Author: Luke Prince
Date: 11 January 2019
'''

import os
import yaml
import re
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cell
import ephys
import plots
from .utils import load_episodic, convert_quantity, merge_dicts
from .stats import hypothesis_tests, ephys_pca

class Analysis(object):
    
    def __init__(self, data_path):
        
        self.data_path = data_path
        
        self.cell_ids = [os.path.basename(dirname[0])
                         for dirname in os.walk(self.data_path)
                         if re.match(r'\d{2}\w{1}\d{2}\-\d{1}', os.path.basename(dirname[0]))]
        self.cell_ids.sort()
        self.cells = cell.CellCollection(data_path= self.data_path)
        
        for cell_id in self.cell_ids:
            self.cells.add(cell.Cell('%s/%s'%(self.data_path,cell_id)))

    def load(self, redo_ephys=False, redo_syns=False, redo_MI=False):
        tqdm.tqdm.write('Loading...')

        self.incomplete = {'ephys' : [],
                           'syns' : [],
                           'MI' : []}
        
        
        for cellid in tqdm.tqdm(self.cells.keys()):
            celln = self.cells[cellid]
            if os.path.exists(celln.celldir + '/ephys.yaml') and not redo_ephys:
                state = yaml.load(open(celln.celldir + '/ephys.yaml', 'r'))
                celln.ephys = cell.Electrophysiology(celldir  = celln.celldir,
                                                     pulse_on =state['Experiment']['pulse_on'],
                                                     pulse_len=state['Experiment']['pulse_len'], 
                                                     sampling_frequency=state['Experiment']['sampling_frequency'],
                                                     I_min    = state['Experiment']['I_min'],
                                                     I_max    = state['Experiment']['I_max'],
                                                     I_step   = state['Experiment']['I_step'])
                celln.ephys.results = dict([(key, convert_quantity(val)) for key, val in state['Results'].items()])
                celln.ephys.spikeFreq = np.array(state['SpikeFreq'])

                celln.ephys.total_time = len(celln.ephys.get_data())/celln.ephys.sampling_frequency
                celln.ephys.time       = np.arange(0, celln.ephys.total_time, 1./celln.ephys.sampling_frequency)

                celln.ephys.v_rest           = celln.ephys.results['Vrest'].item()
                celln.ephys.Rin              = celln.ephys.results['Input Resistance'].item()
                celln.ephys.Cm               = celln.ephys.results['Cell Capacitance'].item()
                celln.ephys.rheobase         = celln.ephys.results['Rheobase'].item()
                celln.ephys.adaptation_ratio = celln.ephys.results['Adaptation Ratio']
                celln.ephys.fISlope          = celln.ephys.results['fI slope'].item()
                celln.ephys.v_sag            = celln.ephys.results['Sag Amplitude'].item()
                celln.ephys.v_thresh         = celln.ephys.results['Spike Threshold'].item()
                celln.ephys.v_amp            = celln.ephys.results['Spike Amplitude'].item()
                celln.ephys.spikeHalfWidth   = celln.ephys.results['Spike Halfwidth'].item()
                celln.ephys.taum             = celln.ephys.results['Membrane Time Constant'].item()
            else:
                self.incomplete['ephys'].append(cellid)
                
            if os.path.exists(celln.celldir + '/syns.yaml') and not redo_syns:
                state = yaml.load(open(celln.celldir+'/syns.yaml', 'r'))
                celln.syns = cell.Synapses(celldir=celln.celldir,
                                           mono_winsize=state['Experiment']['mono_winsize'],
                                           total_winsize=state['Experiment']['total_winsize'],
                                           sampling_frequency = state['Experiment']['sampling_frequency'])
                celln.syns.synapses    = state['Synapses']
                celln.syns.reliability = state['Results']['Reliability']
                celln.syns.mean_delay  = state['Results']['Mean Delay']
                celln.syns.max_delay   = state['Results']['Max Delay']
            else:
                self.incomplete['syns'].append(cellid)
                
            if os.path.exists(celln.celldir + '/MI.yaml') and not redo_MI:
                state = yaml.load(open(celln.celldir+'/MI.yaml', 'r'))
                
                celln.cutoff = state['Results']['cutoff']
                celln.rate_code = cell.Code(celldir= celln.celldir, code_type='rate', delay=celln.syns.max_delay,
                                            mono_winsize       = state['Params']['Rate']['mono_winsize'],
                                            stim_winsize       = state['Params']['Rate']['stim_winsize'],
                                            start              = state['Params']['Rate']['start'],
                                            dur                = state['Params']['Rate']['dur'],
                                            sampling_frequency = state['Params']['Rate']['sampling_frequency'])
                
                celln.rate_code.early_winsize  = state['Params']['Rate']['early_winsize']
                celln.rate_code.results        = state['Results']['Rate']
                
                if celln.rate_code.data_exists:                
                    celln.rate_code.spikeTimes = state['SpikeTimes']['Rate']
                
                celln.temp_code = cell.Code(celldir= celln.celldir, code_type='temp', delay=celln.syns.max_delay,
                                            mono_winsize       = state['Params']['Temporal']['mono_winsize'],
                                            stim_winsize       = state['Params']['Temporal']['stim_winsize'],
                                            start              = state['Params']['Temporal']['start'],
                                            dur                = state['Params']['Temporal']['dur'],
                                            sampling_frequency = state['Params']['Temporal']['sampling_frequency'])
                
                celln.temp_code.early_winsize  = state['Params']['Temporal']['early_winsize']
                celln.temp_code.results = state['Results']['Temporal']
                
                if celln.temp_code.data_exists:
                    celln.temp_code.spikeTimes = state['SpikeTimes']['Temporal']
            else:
                self.incomplete['MI'].append(cellid)

        
    def save(self):
        tqdm.tqdm.write('Saving...')
        for cellid in tqdm.tqdm(self.cells.keys()):
            celln = self.cells[cellid]
            ephys_state = {'Results' : dict([(key, convert_quantity(val)) for key, val in celln.ephys.results.items()]),
                           'Experiment' : {'I_min'     : celln.ephys.I_min,
                                           'I_max'     : celln.ephys.I_max,
                                           'I_step'    : celln.ephys.I_step,
                                           'pulse_on'  : celln.ephys.pulse_on,
                                           'pulse_len' : celln.ephys.pulse_len,
                                           'sampling_frequency' : celln.ephys.sampling_frequency},
                           'SpikeFreq': celln.ephys.spikeFreq.tolist()}
            yaml.dump(ephys_state, open(celln.celldir + '/ephys.yaml', 'w'))
            
            syns_state = {'Synapses' : celln.syns.synapses,
                          'Experiment' : {'mono_winsize' : celln.syns.mono_winsize,
                                          'total_winsize': celln.syns.total_winsize,
                                          'presyn_ids'   : celln.syns.presyn_ids,
                                          'sampling_frequency' : celln.syns.sampling_frequency},
                          'Results' : {'Reliability' : celln.syns.reliability,
                                       'Mean Delay' : celln.syns.mean_delay,
                                       'Max Delay' : celln.syns.max_delay}}
            
            yaml.dump(syns_state, open(celln.celldir + '/syns.yaml', 'w'))
            
            MI_state = {'Results' : {'Rate' : celln.rate_code.results,
                                     'Temporal' : celln.temp_code.results, 
                                     'cutoff' : celln.cutoff},
                        'Params'  : {'Rate' : {'stim_winsize'  : celln.rate_code.stim_winsize,
                                               'mono_winsize'  : celln.rate_code.mono_winsize,
                                               'early_winsize' : celln.rate_code.early_winsize,
                                               'start' : celln.rate_code.start,
                                               'dur' : celln.rate_code.dur,
                                               'sampling_frequency' : celln.rate_code.sampling_frequency},
                                     'Temporal' : {'stim_winsize' : celln.temp_code.stim_winsize,
                                                   'mono_winsize' : celln.temp_code.mono_winsize,
                                                   'early_winsize' : celln.temp_code.early_winsize,
                                                   'start' : celln.temp_code.start,
                                                   'dur' : celln.temp_code.dur,
                                                   'sampling_frequency' : celln.temp_code.sampling_frequency}},
                       'SpikeTimes' : {}}
            
            if celln.rate_code.data_exists:
                MI_state['SpikeTimes']['Rate'] = celln.rate_code.spikeTimes
            if celln.temp_code.data_exists:
                MI_state['SpikeTimes']['Temporal'] = celln.temp_code.spikeTimes
            yaml.dump(MI_state, open(celln.celldir + '/MI.yaml', 'w'))
        
    def update(self):
        tqdm.tqdm.write('Updating...')
        for cellid in tqdm.tqdm(self.cells.keys()):
            celln = self.cells[cellid]
            if cellid in self.incomplete['ephys']:
                celln.extract_ephys(pulse_on = 266, pulse_len = 500, sampling_frequency = 10.,
                                    I_min = -0.08, I_max = 0.4, I_step = 0.04)
            if cellid in self.incomplete['syns']:
                df = pd.read_csv('./data/stimcellids.csv', index_col=0, header=None)
                df = df.loc[celln.cellid] - 1
                df.to_csv(celln.celldir+'/presyn_ids.csv', index=False)
                
                celln.extract_syns(mono_winsize=5, total_winsize=200, sampling_frequency=10.)
                
            if cellid in self.incomplete['MI']:
                celln.initialize_MI(stim_winsize=50, mono_winsize=5,
                                    start=5, dur=50, sampling_frequency=10, delay=celln.syns.max_delay)
                
                
        for cellid in tqdm.tqdm(self.cells.keys()):
            celln = self.cells[cellid]
            if np.isnan(celln.cutoff):
                mean_cutoff = np.nanmean([self.cells[idx].cutoff for idx in self.cells.keys() if self.cells[idx].fluor == celln.fluor])
                if hasattr(celln, 'rate_code'):
                    celln.rate_code.early_winsize = mean_cutoff
                if hasattr(celln, 'temp_code'):
                    celln.temp_code.early_winsize = mean_cutoff
                
            if cellid in self.incomplete['MI']:
                celln.estimate_MI()
                
        self.incomplete = {'ephys' : [],
                           'syns'  : [],
                           'MI'    : []}
        
    def run(self, redo_ephys=False, redo_syns=False, redo_MI=False, reliability_threshold=0.9, replace_mono_with_slopes=False, fig_dir='./'):
        self.load(redo_ephys=redo_ephys, redo_syns=redo_syns, redo_MI=redo_MI)
        self.update()
        self.cells.collect_results()
        
        # Create summary of electrophysiological features
        
        df = pd.DataFrame(data=merge_dicts(self.cells.metadata, self.cells.ephys), index=self.cells.keys());

        # Reorder columns

        cols = ['Fluorescence','Layer',                                                # Cell Type
                'Vrest (mV)','Input Resistance (megaohm)',                             # Passive Properties
                'Cell Capacitance (pF)','Membrane Time Constant (ms)',                 # Passive Properties
                'Rheobase (nA)','fI slope (Hz/nA)',                                    # Active Properties
                'Adaptation Ratio','Sag Amplitude (mV)',                               # Slow Properties
                'Spike Threshold (mV)','Spike Amplitude (mV)','Spike Halfwidth (ms)']  # Action Potential Properties

        df = df[cols]
        self.df_ephys      = df[df['Layer']=='L2/3']
        self.fig_ephys     = plots.plot_ephys_summary(self.df_ephys)
        self.fig_fi        = plots.plot_fICurves(self.cells)
        
        df_ephys_transform = self.df_ephys.drop(['Adaptation Ratio', 'Vrest (mV)'], 1)
        df_ephys_transform['log(Adaptation Ratio)'] = self.df_ephys['Adaptation Ratio'].apply(np.log)
        self.fig_scatter   = plots.plot_ephys_scatter(df_ephys_transform)
        
        self.df_pca        = ephys_pca(df_ephys_transform)
        self.fig_pca       = plots.plot_pca(self.df_pca)
        
        self.fig_dend      = plots.plot_dendrogram(df_ephys_transform)
        del df_ephys_transform
        
        df = pd.DataFrame(data=merge_dicts(self.cells.metadata, self.cells.syns), index=self.cells.keys())
        self.df_syns       = df[df['Layer']=='L2/3']
        self.fig_monorel   = plots.plot_syn_reliability(self.df_syns)
        self.fig_mean_dly  = plots.plot_syn_delays(self.df_syns, kind='Mean')
        self.fig_max_dly   = plots.plot_syn_delays(self.df_syns, kind='Max')
        
        self.results       = {}
        
        codes = ['Rate', 'Temporal']
        resps = ['Spikes', 'Average Vm']
        wins  = ['mono', 'poly', 'all', 'early', 'late']
        
        df_criteria = pd.DataFrame(merge_dicts(self.cells.metadata, self.cells.syns, self.cells.ephys), 
                                   index=self.cells.keys())
        
        ## Inclusion Criteria ##
        # Include if cell in L2/3 and input reliability > 0.9
        inclusion_criteria = np.logical_and(df_criteria['Layer']=='L2/3',
                                            df_criteria['Reliability'] > reliability_threshold)
        # Exlude if fast-spiking NF cell
        exclusion_criteria = np.logical_and(df_criteria['Fluorescence']=='NF',
                                            df_criteria['fI slope (Hz/nA)'] > 500)
        # Combine Criteria
        criteria = np.logical_and(inclusion_criteria, ~exclusion_criteria)

        df_criteria['include'] = criteria
        self.df_criteria = df_criteria

        for code in self.cells.MI.keys():
            if type(self.cells.MI[code]) is dict:
                self.results[code] = {}
                for resp in self.cells.MI[code].keys():
                    if type(self.cells.MI[code][resp]) is dict:
                        self.results[code][resp] = {}
                        for win in self.cells.MI[code][resp].keys():
                            self.results[code][resp][win] = {}
                            df_MI = df

                            if replace_mono_with_slopes and resp=='Average Vm' and win=='mono': 
                                df_MI['MI'] = self.cells.MI[code]['Initial Slope']
                            else:
                                df_MI['MI'] = self.cells.MI[code][resp][win]

                            df_MI = df_MI[criteria]
                            df_MI = df_MI[['Fluorescence', 'MI']]
                            df_MI = df_MI.dropna()
                            self.results[code][resp][win]['df']    = df_MI
                            self.results[code][resp][win]['tests'] = hypothesis_tests(df_MI)
                            self.results[code][resp][win]['fig']   = plots.plot_MI(df_MI, num_comparisons=len(codes)*len(wins),
                                                                                   title_string='%s - %s - %s'%(code, resp, win),
                                                                                   tests=self.results[code][resp][win]['tests'])
                            self.results[code][resp][win]['fig'].savefig(fig_dir + '/%s_%s_%s_MI.svg'%(code,
                                                                                                      resp.replace(' ', ''),
                                                                                                      win))
                            
        df_cutoff = df
        df_cutoff['cutoff'] = self.cells.MI['cutoff']
        df_cutoff = df_cutoff[['Fluorescence', 'cutoff']]
        df_cutoff = df_cutoff.dropna()
        self.results['cutoff'] = {}
        self.results['cutoff']['df'] = df_cutoff
        self.results['cutoff']['tests'] = hypothesis_tests(df_cutoff)
        self.results['cutoff']['fig'] = plots.plot_cutoffs(df_cutoff, tests=self.results['cutoff']['tests'])
        self.results['cutoff']['fig'].savefig(fig_dir + '/cutoff.svg')

        self.fig_ephys.savefig(fig_dir+'/ephys_features.svg')
        self.fig_scatter.savefig(fig_dir+'/ephys_scatter.svg')
        self.fig_pca.savefig(fig_dir+'/ephys_pca.svg')
        self.fig_dend.savefig(fig_dir+'/ephys_dendrogram.svg')
        
        self.fig_fi.savefig(fig_dir+'/fI_curves.svg')
        self.fig_monorel.savefig(fig_dir+'/reliability.svg')
        self.fig_mean_dly.savefig(fig_dir+'/mean_delay.svg')
        self.fig_max_dly.savefig(fig_dir+'/max_delay.svg')
        
        all_slopes = []
        self.fig_slopes = plt.figure()
        for cellid, celln in self.cells.items():
            if celln.syns.reliability >= 0.9:
                for ix in range(10):
                    slope = celln.syns.synapses[ix]['slope']
                    all_slopes.append(celln.syns.synapses[ix]['slope'])

        plt.hist(all_slopes, bins='auto', density=True);
        plt.xlabel('Postsynaptic response slope (mV/ms)')
        plt.ylabel('Probability');
        plt.xlim(0, 9)
        
        self.fig_slopes.savefig(fig_dir+'/slopes.svg')
        
        plt.close('all')
        self.save()