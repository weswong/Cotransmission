# Cotransmission
Direct questions to: wesleywong@fas.harvard.edu or DFWirth@hsph.harvard.edu 

Requirements
Standard Python modules: numpy, collections,math, sys, itertools, json

The Genome module from: https://github.com/edwenger/genepi


P falciparum cotransmission model used in the paper titled "Modeling the genetic relatedness of Plasmodium falciparum parasites following meiotic recombination and cotransmission."

The script is split into three components, a section regarding meiosis, an Infection class, and a sSimulation class.
The meiosis part of the script governs the process of sexual recombination.
Infection class is a container representing the parasites found in the human host.
Simulation class governs the actual running of the simulation.


User decides the coi of the previous infection, the number of oocysts in the mosquito midgut, and the number of hepatocytes formed in the subsequent human host.

Simulation is run on the commandline with the command (Please ensure that the cotx_util_github.py script is in the same directory or somewhere in the PYTHONPATH):

python co-transmission_relatedness_simulation_github.py {coi} {n_oocysts} {n_ihepatocytes}

By default, it will only run 10 iteration. This can be changed by altering the value at line 397:
s = Simulation.simulation(coi,n_oocysts,n_ihepatocytes,10)
where the 10 indicates the number of repetitions to perform.

Output is a json file that can be loaded back in as a dictionary using the json package (https://docs.python.org/2/library/json.html).

The keys to this dictionary are:
params = sim_params

coi = proportion of cotransmission simulations ending with a final COI of {1,2,3,4...}

relationships = proportions of each of the pedigrees describing cotransmitted parasites 

average_ihep_relatedness = average pairwise relatedness of all parasites in the infected hepatocytes (includes genetically identical parasites)

std_ihep_relatedness = std deviation ofpairwise relatedness of all parasites in the infected hepatocytes (includes genetically identical parasites) 

average_obs_relatedness = average pairwise relatedness of genetically distinct parasites in the infected hepatocytes

std_obs_relatedness = std deviation of pairwise relatedness of genetically distinct parasites in the infected hepatocytes

xover_block_distribution = interarrival distance block distribution

n_chiasma_distribution = sim_n_chiasma_distribution
