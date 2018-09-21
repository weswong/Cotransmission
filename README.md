# Cotransmission
Direct questions to: w2w.wong@gmail.com or DFWirth@hsph.harvard.edu 

Requirements
Standard Python modules: numpy, collections,math, sys, itertools, json

P falciparum cotransmission model used in the paper titled "Modeling the genetic relatedness of Plasmodium falciparum parasites following meiotic recombination and cotransmission."

The script is split into three components, a section regarding meiosis, an Infection class, and a Simulation class.
The meiosis part of the script governs the process of sexual recombination.
Infection class is a container representing the parasites found in the human host.
Simulation class governs the actual running of the simulation.


User decides the coi of the previous infection, the number of oocysts in the mosquito midgut, and the number of hepatocytes formed in the subsequent human host. COI is referred to as poi throughout the script (used to stand for polygenomicity of infection).

Simulation is run on the commandline with the command (Please ensure that the cotx_util_github.py script and the genome folder is in the same directory):

python transmission_relatedness_simulation_PLOS.py {coi} {n_oocysts} {n_ihepatocytes} {strain_differential} {n_repetitions}
The first argument is the COI, the second argument is the # oocysts in the mosquito midgut, the third argument is the number of infected hepatocytes in the subsequent human host, the fourth argument is the difference in strain proportion between the most frequent and least frequent strain in the sampled infection, and the last is the number of repetitions to run.

Also included is the serial transmission script. To run:
python cotx_serial.py {coi} {strain_differential} {backcross} {n_repetitions}

For the serial transmission script, n_oocysts and n_ihepatocytes are not specified and are drawn from distributions observed from actual infections. The backcross parameter is a flag indicating whether to : {0: no backcross oppportunity (each new infection on an uninfected host, 1: new infection on a host previously infected with one of the original parental strains (the same each time) 2: a new infection on a host previously infected with a unrelated strain.

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
