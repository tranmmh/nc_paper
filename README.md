# Analysis code for Tran et al (in press) 'Rate and temporal code multiplexing in neocortical microcircuits'

This repository contains the analysis code for data published in Tran et al (2019). This code has been written in python 2.7, and has the following requirements:

matplotlib == 2.1.2  
numpy == 1.14.1  
scipy == 1.0.0  
pandas == 0.22.0  
neo == 0.5.2  
sklearn == 0.19.1  
tqdm == 4.29.1  
yaml == 3.13  

If you have any issues with compatibility, please fill out an issue report.

The analysis functions are split into three sections: electrophysiology (ncstudy.ephys) used to measure the electrophysiological properties of recorded cells in response to step currents, synaptic (ncstudy.synapses) used to measure the fidelity of the optogenetically stimulated presynaptic inputs to the electrophysiological recorded postsynaptic cell, and statistics (ncstudy.stats) used to estimate the mutual information between the underlying signal and the postsynaptic cell response.

Cell objects (ncstudy.cell.Cell) are used to store the results of each experiment (ncstudy.cell.Electrophysiology, ncstudy.cell.Synapses, ncstudy.cell.Code), and Cell objects are stored in extended OrderedDict class (ncstudy.cell.CellCollection). The full analysis pipeline is run through an Analysis object (ncstudy.analysis.Analysis) that handles loading and updating data from previous and new cells, and plotting the results.

The use of this code is demonstrated in the Neural_Coding.ipynb notebook.

Although many of these objects have been specialised for analysing the data from the experiments reported in Tran et al (in press), the functions in ncstudy.ephys and ncstudy.stats may have more general use.

For questions, please email luke.prince@utoronto.ca
