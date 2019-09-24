'''
Author: Luke Prince
Date: 22 November 2018

Analysis of electrophysiological properties of neurons by current injection.

Traces are membrane potentials over the course of an injected current pulse of a known amplitude and duration.
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import zscore
    
#------------------------------------------------------------------
#------------------------------------------------------------------

def steadyStateVoltage(trace,start_ix,end_ix):
    '''
    Calculate the steady state voltage of a slice between two timepoints of a trace

    Arguments:
        - trace (np.array) : array containing voltage trace in mV
        - start_ix (float)     : start time index to calculates steady state
        - end_ix (float)       : end time index to calculate steady state

    Returns:
        - v_ss (np.float)  : steady state voltage in mV of trace between t1 and t2
    '''

    # Calculate mean voltage in slice and return
    return np.mean(trace[start_ix:end_ix])

#------------------------------------------------------------------
#------------------------------------------------------------------

def sag_amplitude(trace):
    '''
    Calculate amplitude of the sag potential in response to a hyperpolarising current injection

    Argument:
        - trace (np.array) : array containing voltage trace in mV with hyperpolarising current injection

    Returns:
        - v_sag (np.float) : sag amplitude in mV

    '''
    # Calculate baseline voltage after current injection
    Vss = steadyStateVoltage(trace,-100,-1)

    # Find lowest voltage in first 1000 time points of trace
    Vmin = np.min(trace[:1000])

    # Return absolute difference between baseline and minimum membrane potential
    return np.abs(Vmin-Vss)

#------------------------------------------------------------------
#------------------------------------------------------------------


def adaptation_ratio(spikeTimes):
    '''
    Calculate adaptation ratio defined asratio between late and early inter-spike intervals in a train).
    Requires a minimum of 3 spikes.

    Arguments:
        - spikeTimes (list) : list of spike times in ms

    Return:
        - AR (float) : adaptation ratio
    '''

    # Set ratio to nan (in the event of insufficient spikes to calculate ratio
    AR = np.nan

    # For spike train with 3-6 spikes, calculate ratio between first and last inter-spike interval
    if len(spikeTimes)>2 and len(spikeTimes)<7:
        # Calculate inter-spike intervals as difference between spike times
        isi = np.diff(spikeTimes)
        # Divide last interval by first interval to obtain adaptation ratio
        AR = isi[-1]/isi[0]

    # For spike train with 7+ spikes, calculate ratio between average of first two and last two 
    # inter-spike intervals
    elif len(spikeTimes)>=7:
        # Calculate inter-spike intervals as difference between spike times
        isi = np.diff(spikeTimes)
        # Divide mean of last two intervals by mean of first two intervals to obtain adaptation ratio
        AR = np.mean(isi[-2:])/np.mean(isi[:2])

    # Return adaptation ratio
    return AR

#------------------------------------------------------------------
#------------------------------------------------------------------

def findThreshCross(trace,thresh=-20):
    '''
    Find threshold crosses of a trace

    Arguments:
        - trace (np.array) : membrane potential trace in mV in response to current injection
        - thresh (float)   : threshold to cross (default = -20)

    Returns:
        - bool_idx (np.array) : threshold crossings returned as a boolean index of length (len(trace)-2).
    '''

    # Determine time points greater than or equal to threshold, and time points less than threshold and take
    # union shifted by one time point
    return np.logical_and(trace[1:]>=thresh,trace[:-1]<thresh)

#------------------------------------------------------------------
#------------------------------------------------------------------

def resting_MP(trace):
    '''
    Estimate resting membrane potential

    Arguments:
        - trace (np.array) : membrane potential trace in mV in response to current injection

    Returns:
        - v_rest (np.float) : resting membrane potential in mV
    '''

    # Estimate resting membrane potential as average voltage in first 200 time points
    return steadyStateVoltage(trace,0,2000)

#------------------------------------------------------------------
#------------------------------------------------------------------

def fICurve(I,thresh,slope):
    '''
    Function to describe relationship between current injection in nA (I) and firing rate (f) in Hz of a neuron.
    A basic f-I curve assumes a relationship whereby the frequency increases linearly with current with a defined
    slope after a threshold is reached. Subthreshold current induces no firing (0 Hz).

    Arguments:
        - I (np.array)      : current in nA over which to define f-I curve
        - thresh (np.float) : threshold current in nA to induce non-zero firing rate
        - slope (np.float)  : slope in Hz/nA to define super-threshold frequency current relationship

    Returns:
        - f (np.array)      : firing rate in Hz
    '''

    return np.clip(slope*I - slope*thresh,a_min=0,a_max=np.inf)

#------------------------------------------------------------------
#------------------------------------------------------------------

def estimatefISlope(f,I,p0=[100,10], num_nonzero=5):
    '''
    Fit parameters (threshold and slope) of f-I curve to firing rate in Hz of sampled neurons
    with current injection values I. Uses curve_fit from scipy.optimize

    Arguments:
        - f (np.array)      : observed firing rates in Hz of neurons in response to current injection
        - I (np.array)      : current injection values in nA
        - p0 (list)         : initial parameter values (default=[100,10])
        - num_nonzero (int) : number of non-zero points to fit up to (default = 5)

    Returns:
        - [fitted_thresh, fitted_slope]
    '''
    ix = int(len(f) - np.maximum(np.count_nonzero(f) - num_nonzero, 0))
    popt,pcov = curve_fit(fICurve,I[:ix],f[:ix],p0=p0)
    return popt
#------------------------------------------------------------------
#------------------------------------------------------------------

def rheobase(f,I):
    '''
    Estimate rheobase from observed firing rate (f) and current injection values (I).
    Returns first current value for which firing rate is greater than zero

    Arguments:
        - f (np.array) : observed firing rates in Hz of neurons in response to current injection
        - I (np.array) : current injection values in nA

    Returns:
        - rheobase (np.float)
    '''

    return I[f>0][0]

#------------------------------------------------------------------
#------------------------------------------------------------------

def expDecay(t,V0,tau,offset):
    '''
    Function describing exponential decay from V0+offset to V0

    Arguments:
        - t (np.array)       : time-steps in ms
        - V0 (np.float)      : baseline voltage in mV
        - tau (np.float)     : time constant in ms
        - offset (np.float)  : offset voltage in mV

    Returns: 
        - v_decay (np.array) : exponentially decaying voltage
    '''

    return (V0-offset)*np.exp(-t/tau) + offset

#------------------------------------------------------------------
#------------------------------------------------------------------

def membrane_tau(trace, sampling_frequency):
    '''
    Fit membrane time constant of a trace.

    Arguments:
        - trace (np.array) : trace of exponentially decaying membrane potential

    Returns:
        - fitted_tau (np.float) : fitted time constant
    '''
    # Create array of time-steps in ms
    t = np.arange(0,len(trace)/sampling_frequency,1./sampling_frequency) # in ms

    # Fit time constant using curve_fit from scipy.optimize
    popt,pcov = curve_fit(expDecay,t,trace,p0=[trace[0],1.0,trace[-1]])
    return popt[1]

#------------------------------------------------------------------
#------------------------------------------------------------------

def input_resistance(trace,I,vrest):
    '''
    Estimate input resistance using Ohm's Law (V = IR)

    R_in    =     Change in steady state voltage due to current injection
                  _______________________________________________________

                                    Injected Current

    Arguments:
        - trace (np.array) : trace of membrane potential in mV
        - I (np.float)     : injected current in nA
        - vrest (np.float) : resting membrane potential in mV

    Returns:
        - Rin (np.float) : input resistance in MOhm
    '''
    vss = steadyStateVoltage(trace,-100,-1)
    return (vss-vrest)/I

#------------------------------------------------------------------
#------------------------------------------------------------------

def spike_threshold(spikeTrace,zslopeThresh=0.5):
    '''
    Estimate threshold membrane potential in mV of action potential by finding voltage at which membrane potential
    slope drastically increases

    Arguments:
        - spikeTrace (np.array)   : trace of action potential in mV
        - zslopeThresh (np.float) : z-scored membrane potential first derivative threshold

    Returns: 
        - v_thresh (np.float) : threshold membrane potential in mV
    '''

    # Calculate z-score of action potential slope
    zslope = zscore(np.diff(spikeTrace))
    # find slope z-score threshold crossings 
    ap_thresh = [np.logical_and(zslope[1:]>=zslopeThresh,zslope[:-1]<zslopeThresh)]

    # Return first membrane potential crossing slope z-score threshold
    return spikeTrace[1:-1][ap_thresh][0]

#------------------------------------------------------------------
#------------------------------------------------------------------

def spike_amplitude(spikeTrace,Vthresh):
    '''
    Estimate action potential amplitude in mV given an estimated action potential threshold

    Arguments:
        - spikeTrace (np.array) : trace of action potential in mV
        - Vthresh (np.float) : action potential threshold

    Return:
        - v_ap (np.float) : action potential amplitude in mV
    '''

    Vpeak = np.max(spikeTrace)
    return Vpeak - Vthresh

#------------------------------------------------------------------
#------------------------------------------------------------------

def ahp_amplitude(spikeTrace,Vthresh):
    '''
    Estimate after-hyperpolarization amplitude in mV given an estimated action potential threshold

    Arguments:
        - spikeTrace (np.array) : trace of action potential in mV
        - Vthresh (np.float) : action potential threshold

    Return:
        - v_ahp (np.float) : after-hyperpolarization amplitude in mV

    '''
    Vtrough = np.min(spikeTrace)
    return Vthresh - Vtrough

#------------------------------------------------------------------
#------------------------------------------------------------------

def spike_halfwidth(spikeTrace, Vthresh, spikeAmplitude, sampling_frequency):
    '''
    Estimate spike half-width in ms given an estimate of action potential threshold and spike amplitude.
    Spike half-width is the duration above mid-voltage of the action potential
    '''

    # time-steps of action potential trace
    time = np.arange(0,len(spikeTrace)/sampling_frequency,1./sampling_frequency)

    # Calculate mid-voltage
    Vhalf = Vthresh + spikeAmplitude/2.

    # Find time points above mid-voltage
    time = time[spikeTrace>=Vhalf]

    # Return difference between first and last time points above mid-voltage + additional time step for correction
    return time[-1] - time[0] + 1./sampling_frequency