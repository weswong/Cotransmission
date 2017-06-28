import numpy as np
from collections import defaultdict



chr_lengths = {1:643292,
               2:947102,
               3:1060087,
               4:1204112,
               5:1343552,
               6:1418244,
               7:1501717,
               8:1419563,
               9:1541723,
               10: 1687655,
               11:2038337,
               12:2271478,
               13:2895605,
               14:3291871}


# Utility functions ---------------------------------------------------
def sim_find_flip_points(genome_object):
    '''find xover locations'''
    xover_locations = {}
    for chromosome in range(1,15):
        sequence = genome_object.return_chromosome(chromosome)
        flip_points = np.where(sequence[:-1] != sequence[1:])[0]
        xover_locations[chromosome] = flip_points
    return xover_locations

def calculate_abc_metrics(genome_object):
    '''calculate the intercrossover distance and chiasma count on each chromosome of the genome'''
    sim_xover_lengths = defaultdict(list)
    sim_chiasma_count = defaultdict(list)
    tmp_count = []
    xover_locations = sim_find_flip_points(genome_object)
    for chromosome in range(1,15):
        sim_xover_lengths[chromosome]+= calculate_crossover_length(chromosome, xover_locations[chromosome])
        sim_chiasma_count[chromosome].append(len(xover_locations[chromosome]))
    return sim_xover_lengths, sim_chiasma_count

def calculate_crossover_length(chromosome, chiasma_pos_array):
    '''calculates intercrossover distance'''
    segment_lengths = []
    n_chiasma = len(chiasma_pos_array)
    if n_chiasma == 0:
        segment_lengths.append(chr_lengths[chromosome]) # if no crossovers found on the chromosome
    elif n_chiasma == 1:
        segment_lengths+=[chiasma_pos_array[0], chr_lengths[chromosome] - chiasma_pos_array[0]] #the two lengths if there is one crossover
    else: #in the case of multiple crossovers
        start = 0
        end = 0
        for i, chiasma_pos in enumerate(chiasma_pos_array):
            if i == 0:
                d = chiasma_pos_array[i]
            elif i != len(chiasma_pos_array) -1:
                d = chiasma_pos_array[i] - chiasma_pos_array[i-1]
            else:
                d = chiasma_pos_array[i] - chiasma_pos_array[i-1]
                segment_lengths.append(d)
                d = chr_lengths[chromosome] - chiasma_pos_array[i-1]
                
        
            segment_lengths.append(d)
    return segment_lengths

def find_unique_arrays(genome_pool):
    '''utility function to quickly find unique genomes'''
    a= np.asarray([g.genome for g in genome_pool])
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx, counts = np.unique(b, return_index=True, return_counts = True)
    unique_a = a[idx]
    unique_genomes = [genome_pool[i] for i in idx]
    return len(unique_a), unique_genomes

def fractionize_counts(counts):
    '''utility function to convert counts to proportions'''
    denominator = np.sum(counts.values(), dtype = np.float64)
    for k in counts:
        counts[k] = counts[k] / denominator
    return counts

def collapse_dictionary(dictionary_array):
    '''utility function to combine simulation results'''
    summarized_dictionary = defaultdict(list)
    for d in dictionary_array:
        for chromosome in d:
            summarized_dictionary[chromosome]+= d[chromosome]
    return summarized_dictionary

def summarize_xover_data(collapsed_xover_dict, collapsed_n_chiasma_dict):
    '''summarize data'''
    n_chiasma_bins = np.arange(0,6)
    interarrival_bins = np.arange(0,250, 10)
    bp_per_cM = 15000.0
    
    xover_block_distribution = {}
    n_chiasma_distribution = {}
    for chromosome in range(1,15):
        frequencies, xover_bins = np.histogram([bp/bp_per_cM for bp in collapsed_xover_dict[chromosome]],interarrival_bins)
        xover_block_distribution[chromosome] = list(frequencies)
        
        frequencies, chiasma_bins = np.histogram([n for n in collapsed_n_chiasma_dict[chromosome]],n_chiasma_bins)
        n_chiasma_distribution[chromosome] = list(frequencies)
    xover_block_distribution['bins'] = list(xover_bins)
    n_chiasma_distribution['bins'] = list(chiasma_bins)
    
    return xover_block_distribution, n_chiasma_distribution


