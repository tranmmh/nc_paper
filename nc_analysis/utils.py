import numpy as np
from neo import io
import os
import pdb
from quantities import Quantity
from fractions import Fraction

'''
ABF_IO.py

A simple library with functions for loading ABF files using the NEO library.
'''


def load_episodic(filename):
    
	'''
	load_episodic(filename)

	Loads episodic recordings from pClamp data in 'filename'.

	Returns the following:

	trace: a numpy array of size [t, n, c], where t is the number of samples per episode,
	n is the number of episodes (sweeps) and c is the number of channels.

	cinfo: a dictionary containing lists with the names and units of each of the channels, 
	keys are 'names' and 'units'.

	'''

	# open the file
	try:
		r = io.AxonIO(filename=filename)
	except IOError as e:
		print('Problem loading specified file')

	# read file into blocks
	bl = r.read_block(lazy=False,cascade=True)

	# read in the header info
	head = r.read_header()

	# determine the input channels and their info
	chans  = head['listADCInfo']
	nchans = len(chans)
	cinfo  = {'names' : [], 'units' : []}
	for c in chans:
		cinfo['names'].append(c['ADCChNames'])
		cinfo['units'].append(c['ADCChUnits'])

	# determine the number of sweeps and their length
	nsweeps  = np.size(bl.segments)
	nsamples = head['protocol']['lNumSamplesPerEpisode']/nchans

	# initialize an array to store the data
	trace = np.zeros((nsamples,nsweeps,nchans))

	# load the data into the traces
	bl = r.read_block(lazy=False,cascade=True)
	for c in range(nchans):
		for s in range(nsweeps):
			#pdb.set_trace()
			trace[:,[s],[c]] = bl.segments[s].analogsignals[c]



	return (trace, cinfo)

def merge_dicts(*dict_args):
    results = {}
    for d in dict_args:
        results.update(d)
    return results

def convert_quantity(x):
    if type(x) is Quantity:
        return [x.item(), str(x.units).split()[1]]
    
    elif type(x) is list:
        if type(x[0]) is float and type(x[1]) is str:
            return Quantity(*x, dtype=float)
        else:
            raise TypeError('Do not recognise type of %s'%x)
    else:
        return x
    
def resample(x, cf, nf):
    if nf>cf:
        xtmp = np.repeat(x,np.ceil(nf/cf))
        mask = np.ones(len(xtmp),dtype=bool)
        if np.ceil(nf/cf) != nf/cf:
            skipeach = Fraction((nf/cf)/np.ceil(nf/cf)).limit_denominator().denominator
            mask[::skipeach] = 0
        xnew = xtmp[mask]
    elif nf<cf:
        xtmp = x[::int(cf/hcf(cf,nf))]
        xnew = np.repeat(xtmp,int(nf/hcf(cf,nf)))
    elif nf==cf:
        xnew = x
    return xnew

def highestCommonFactor(x, y):
    while y != 0:
        (x, y) = (y, x % y)
    return x

def hcf(x,y):
    return highestCommonFactor(x, y)