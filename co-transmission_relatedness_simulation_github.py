import numpy as np
import json
from collections import defaultdict, Counter
import math
import sys
import itertools

#from github respository: https://github.com/edwenger/genepi
from old_genepi import genome
from old_genepi.genome import Genome
from old_genepi import utils
genome.initialize_from('sequence')

from cotx_utils_github import *
#Meiosis -------------------------------------------------------

bp_per_cM     = 15000. #~15kb/cM Su et al 1998
                                 
def gamma_interarrival_time(v=1):   #v=1 means no interference
    '''returns the distance in map units of the next chiasma event in the 4 chromatid bundle
    interrarival times specify the next event (in Morgan), does not incorporate obligate chiasma formation
    allows for different crossover interference levels'''
    interarrival_time = np.random.gamma(shape=v, scale =1./(2*v))
    #rate parameter must be constrained to equal 2*shape
    #scale = 1/shape
    #for rate to be constrained to 2*shape, scale must = 1/(2v)
    #conversion to bp, rounded up
    d = int(math.ceil(interarrival_time * bp_per_cM* 100))
    return d
                                 
def get_crossover_points(v,chrom_length):
    '''does not incorporate obligate chiasma formation'''
    next_point=-100000 #make it a stationary renewal process
    xpoints=[]
    while next_point < chrom_length:
        if next_point > 0.:
            xpoints.append(next_point)
        d = gamma_interarrival_time(v)
        next_point+=d
    return xpoints


#equation relating v to the scale in obligate chiasma version
obligate_co_scale_fn = np.poly1d(np.array([  2.86063522e-05,  -1.28111927e-03,   2.42373279e-02,
                                            -2.52092360e-01,   1.57111461e+00,  -5.99256708e+00,
                                             1.36678013e+01,  -1.72133175e+01,   9.61531678e+00]))

def oc_get_crossover_points(v, chrom_length):
    '''obligate chiasma version
    Generate the first obligate chiasma by drawing from a Uniform Distribution
    Expand outwards from that point until you reach both ends of the chromosome'''
    xpoints=[]
    obligate_chiasma_pos = int(math.ceil(np.random.uniform(low=0., high= float(chrom_length))))
    xpoints.append(obligate_chiasma_pos)
    
    scale = obligate_co_scale_fn(v)
    #move to the right
    interarrival_time = np.random.gamma(shape=v, scale =scale)
    d = int(math.ceil(interarrival_time * bp_per_cM* 100))
    right_point = d + obligate_chiasma_pos
    while right_point < chrom_length:
        xpoints.append(right_point)
        interarrival_time = np.random.gamma(shape=v, scale =scale)
        d = int(math.ceil(interarrival_time * bp_per_cM* 100))
        right_point += d
    
    #move to the left
    interarrival_time = np.random.gamma(shape=v, scale =scale)
    d = int(math.ceil(interarrival_time * bp_per_cM* 100))
    left_point = obligate_chiasma_pos - d
    while left_point > 0:
        xpoints.append(left_point)
        interarrival_time = np.random.gamma(shape=v, scale =scale)
        d = int(math.ceil(interarrival_time * bp_per_cM* 100))
        left_point -= d
        
    return xpoints

def crossover(g1,g2,xpoints):
    #S phase, DNA duplication time
    
    #sister chromatids on homologous chromosome #1
    c1 = np.copy(g1)
    c2 = np.copy(g1)
    
    #sister chromatids on homologous chromosome #2
    c3 = np.copy(g2)
    c4 = np.copy(g2)
    if not xpoints:
        return c1,c2, c3,c4
    
    for breakpoint in xpoints:
        probability = np.random.random()
        if probability < 0.25: # c1 and c3
            t = np.copy(c1[breakpoint:])
            c1[breakpoint:] = c3[breakpoint:]
            c3[breakpoint:] = t
        elif probability >= 0.25 and probability < 0.5: #c1 and c4
            t = np.copy(c1[breakpoint:])
            c1[breakpoint:] = c4[breakpoint:]
            c4[breakpoint:] = t
        elif probability >= 0.5 and probability < 0.75: #c2 and c3
            t = np.copy(c2[breakpoint:])
            c2[breakpoint:] = c3[breakpoint:]
            c3[breakpoint:] = t
        elif probability >=0.75: #c2 and c4
            t = np.copy(c2[breakpoint:])
            c2[breakpoint:] = c4[breakpoint:]
            c4[breakpoint:] = t
    return c1, c2, c3, c4

def calculate_relatedness(o1, o2):
    return np.sum(o1.genome == o2.genome, dtype = float) / len(o1.genome)

def meiosis(in1,in2,N=4,v=2, oc=True):
    '''v defines the shape of the gamma distribution, it is required to have a non-zero shape parameter
    if v = 0, we assume user means no crossover model
    v =1 corresponds to no interference
    obligate crossover means use the obligate crossover version'''
    if N > 4:
        raise IndexError('Maximum of four distinct meiotic products to sample.')
    genomes=[genome.reference_genome() for _ in range(4)]
    
    if v != 0:
        if oc:
            crossover_fn = oc_get_crossover_points
        else:
            crossover_fn = get_crossover_points
    else:
        crossover_fn = lambda x,y: []
    for idx,(start,end) in enumerate(utils.pairwise(Genome.chrom_breaks)):
        c1,c2=in1.genome[start:end],in2.genome[start:end]
        xpoints = crossover_fn(v, len(c1))

        #log.debug('Chr %d, xpoints=%s',chrom_names[idx],xpoints)
        c1, c2, c3, c4=crossover(c1,c2,xpoints)
        
        #independent assortment
        outputs=sorted([c1,c2,c3,c4], key=lambda *args: random.random())       
        for j in range(4):
            genomes[j][start:end]=outputs[j]
    return [Genome(genomes[j]) for j in range(N)]

#-----------------------------------------------------------------------------------------------------------------------

class Infection:
    def __init__(self, genome_pool):
        self.genome_pool = genome_pool
        self.coi,self.unique_genome_pool = find_unique_arrays(genome_pool)[0], find_unique_arrays(genome_pool)[1]
        
    @classmethod
    def initiate_infection(cls,coi):
        genomes = []
        for _ in range(coi):
            genome = Genome.from_reference()
            genome.genome = genome.genome + _
            genome.id = _
            genomes.append(genome)
        return genomes
    
    @classmethod
    def find_unique_arrays(cls,a):
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, idx, counts = np.unique(b, return_index=True, return_counts = True)
        unique_a = a[idx]
        count_dictionay={}
        for i,c in zip(idx,counts):
            count_dictionary[a[i].id] = c
        return len(unique_a), count_dictionary
    
    @classmethod
    def mosquito_transmission(cls,genome_pool,n_oocysts, n_ihepatocytes):
        sporozoite_pool = []
        for _ in range(n_oocysts):
            p1, p2= np.random.choice(genome_pool, 2)
            meiotic_products = meiosis(p1,p2)
            for o in meiotic_products:
                o.p1 = p1.id
                o.p2 = p2.id
                o.oocyst = _
            sporozoite_pool += meiotic_products
        
        hep_pool = np.random.choice(sporozoite_pool, n_ihepatocytes)
        return Infection(hep_pool) 
        
    def interpret_ihep_relationships(self):
        #relationship and relatedness of the n_ihepatocyte strains. Not necessarily unique. 
        #Gives us and idea of the type of pedigree trees accessed
        relationship_types = []
        relatedness= []
        #1 = same oocyst, selfed origin
        #2 = same oocyst, outcross origin
        #3 = different oocyst, both self, same parent
        #4 = different oocyst, both self, different parents
        #5 = different oocyst, 1 self 1 outcross, 1 parent shared
        #6 = different oocyst, 1 self 1 outcross, 0 parents shared
        #7 = different oocyst, both outcross, same parents
        #8 = different oocyst, both outcross, 1 parent shared
        #9 = different oocyst, both outcross, different parents shared
        #10 = unknown
        combos = itertools.combinations(self.genome_pool,2)
        for g1,g2 in combos:
            relatedness.append(calculate_relatedness(g1,g2))
                               
            if g1.oocyst == g2.oocyst: #both originate from same oocyst
                if g1.p1 == g1.p2:
                    relationship_types.append(1) #selfed oocyst
                else:
                    relationship_types.append(2) #outcross oocyst
                    
            else: # both originate from different oocysts
                
                if (g1.p1 == g1.p2) and (g2.p1 == g2.p2):
                    # both selfed oocysts
                    if (g1.p1 == g2.p1): 
                        #same origins
                        relationship_types.append(3)
                    elif (g1.p1 != g2.p1):
                        #different origins
                        relationship_types.append(4)
                
                elif ((g1.p1 == g1.p2) and (g2.p1 != g2.p2)) or ((g1.p1 != g1.p2) and (g2.p1 == g2.p2)):
                    #1 self oocyst, 1 outcross oocyst
                    num_unique_parents = len(np.unique([g1.p1, g1.p2, g2.p1, g2.p2]))
                    if num_unique_parents == 2:
                        #1 parent shared with outcross oocyst
                        relationship_types.append(5)
                    elif num_unique_parents == 3:
                        #outcross oocyst composed of different parasite strains
                        relationship_types.append(6)
                
                elif (g1.p1 != g1.p2) and (g2.p1 != g2.p2):
                    #two outcross oocysts
                    num_unique_parents = len(np.unique([g1.p1, g1.p2, g2.p1, g2.p2]))
                    if num_unique_parents == 2:
                        #oocyst share both parents
                        relationship_types.append(7)
                    elif num_unique_parents ==3:
                        #share one parent
                        relationship_types.append(8)
                    elif num_unique_parents == 4:
                        relationship_types.append(9)
                
                else: #debug
                    print g1.oocyst, g1.p1, g1.p2
                    print g2.oocyst, g2.p1, g2.p2   
                    relationship_types.append(10)
        self.relationship_types = Counter(relationship_types)
        self.ihep_relatedness = np.average(relatedness)
        
    def calculate_observable_relatedness(self):
        #working definition of relatedness: average relatedness amongst UNIQUE strains
        #this is the relatedness we observe in the field
        if len(self.unique_genome_pool) != len(self.genome_pool):
            combos = itertools.combinations(self.unique_genome_pool,2)
            relatedness= []
            for g1, g2 in combos:
                relatedness.append(calculate_relatedness(g1,g2))
            self.observable_relatedness = np.average(relatedness)
        else:
            self.observable_relatedness = self.ihep_relatedness
    
    def calculate_xover_stats(self):
        #calculates xover only amongst the UNIQUE strains
        #these will represent the xover maps generated in the field
        n_chiasma_array  = []
        xover_length_array = []
        for g in self.unique_genome_pool:
            xover_length, chiasma_count = calculate_abc_metrics(g)
            n_chiasma_array.append(chiasma_count)
            xover_length_array.append(xover_length)
        n_chiasma_dict = collapse_dictionary(n_chiasma_array)
        xover_length_dict = collapse_dictionary(xover_length_array)
        self.n_chiasma= n_chiasma_dict
        self.xover_length= xover_length_dict
            
    
    def return_stats(self):
        return [self.relationship_types, self.ihep_relatedness, self.coi,
                self.observable_relatedness,self.xover_length, self.n_chiasma]

#--------------------------------------------------------------------------------------------

class Simulation:
    def __init__(self, sim_params, sim_coi, 
                sim_average_ihep_relatedness, sim_std_ihep_deviation,  sim_relationship_types,
                sim_average_obs_relatedness,  sim_std_obs_relatedness,
                sim_xover_block_distribution, sim_n_chiasma_distribution):
        
        self.params = sim_params
        self.coi = sim_coi
        self.average_ihep_relatedness = sim_average_ihep_relatedness
        self.std_ihep_relatedness = sim_std_ihep_deviation
        self.relationships = sim_relationship_types
        self.average_obs_relatedness = sim_average_obs_relatedness
        self.std_obs_relatedness = sim_std_obs_relatedness
        self.xover_block_distribution = sim_xover_block_distribution
        self.n_chiasma_distribution = sim_n_chiasma_distribution
    
    @classmethod
    def summarize_relationship_types(cls, simulation_relationship_output):
        summarized_relationship_dict = defaultdict(list)
        
        for sim_counts in simulation_relationship_output:
            for count in sim_counts:
                summarized_relationship_dict[count].append(sim_counts[count])
        for count in range(1,10):
            summarized_relationship_dict[count] = np.sum(summarized_relationship_dict[count],
                                                                  dtype=np.float64)
            
        return fractionize_counts(summarized_relationship_dict)
                    
    
    @classmethod
    def simulation(cls, coi, n_oocysts, n_ihepatocytes, n_repetitions):
        simulation_array = []
        for _ in range(n_repetitions):
            print _
            genome_pool = Infection.initiate_infection(coi)
            host = Infection(genome_pool)
            transmission = Infection.mosquito_transmission(host.genome_pool,n_oocysts,n_ihepatocytes)
            transmission.interpret_ihep_relationships()
            transmission.calculate_observable_relatedness()
            transmission.calculate_xover_stats()
            
            simulation_array.append(transmission.return_stats())
        # [self.relationship_types, self.ihep_relatedness, self.coi,
        # self.observable_relatedness,self.xover_length, self.n_chiasma]
        
        sim_params = {'n_oocysts': n_oocysts,
                  'n_ihepatocytes': n_ihepatocytes,
                  'coi' : coi}  
        
        relationship_types_list = []
        coi_list = []
        ihep_relatedness_list = []
        obs_relatedness_list = []
        xover_block_list = []
        n_chiasma_list = []
        for s in simulation_array:
            s_type = s[0]
            s_ihep_relatedness = s[1]
            s_coi =s[2]
            s_obs_relatedness = s[3]
            s_xover_length = s[4]
            s_n_chiasma = s[5]
            
            relationship_types_list.append(s_type)
            coi_list.append(s_coi)
            
            if int(s_coi) > 1:
                ihep_relatedness_list.append(s_ihep_relatedness)
                obs_relatedness_list.append(s_obs_relatedness)
                xover_block_list.append(s_xover_length)
                n_chiasma_list.append(s_n_chiasma)
        
        sim_relationship_types = Simulation.summarize_relationship_types(relationship_types_list)
        sim_coi = fractionize_counts(Counter(coi_list))
        
        sim_average_ihep_relatedness =   np.average(ihep_relatedness_list)
        sim_std_ihep_deviation = np.std(ihep_relatedness_list)
        
        sim_average_obs_relatedness = np.average(obs_relatedness_list)
        sim_std_obs_relatedness = np.std(obs_relatedness_list)
        
        sim_xover_block_distribution,sim_n_chiasma_distribution = summarize_xover_data(collapse_dictionary(xover_block_list),collapse_dictionary(n_chiasma_list))
        
                                     
        return Simulation(sim_params, sim_coi, 
                          sim_average_ihep_relatedness, sim_std_ihep_deviation,  sim_relationship_types,
                          sim_average_obs_relatedness,  sim_std_obs_relatedness,
                          sim_xover_block_distribution, sim_n_chiasma_distribution)
    
    def write_results(self):
        fout = open('{coi}_{n_oocysts}_{n_ihepatocytes}.json'.format(coi = self.params['coi'],
                                                                n_oocysts = self.params['n_oocysts'],
                                                                n_ihepatocytes = self.params['n_ihepatocytes']),'w')
        json.dump(self.__dict__, fout)
        fout.close()
        
    def write_results_serial(self):
        fout = open('{coi}_{n_oocysts1}_{n_ihepatocytes1}_{n_oocysts2}_{n_ihepatocytes2}.json'.format(coi = self.params['coi'],
                                                                n_oocysts1 = self.params['n_oocysts'][0],n_oocysts2 = self.params['n_oocysts'][1],
                                                                n_ihepatocytes1 = self.params['n_ihepatocytes'][0], n_ihepatocytes2=self.params['n_ihepatocytes'][1]
                                                                ),'w')
        json.dump(self.__dict__, fout)
        fout.close()
                        
                        
if __name__ == '__main__':
    coi = int(sys.argv[1])
    n_oocysts = int(sys.argv[2])
    n_ihepatocytes = int(sys.argv[3])
    s = Simulation.simulation(coi,n_oocysts,n_ihepatocytes,10) #last argument is the number of repetitions
    s.write_results()
    
