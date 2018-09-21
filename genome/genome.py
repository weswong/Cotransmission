import math
import bisect
import random
import itertools
from collections import defaultdict
import hashlib
import sys
sys.path.insert(0, 'genome/snp/')

#import logging
#log = logging.getLogger(__name__)

import numpy as np # for fast meiosis operations on arrays

import utils

from snp import SNP
from transmission import Transmission
import barcode as snp_barcode_mod



bp_per_morgan = 1.5e6
bp_per_cM     = bp_per_morgan/1e2
bp_per_Mbp    = 1e6

mutation_prob = 9.7e-9 / 2.0  # (per day, per position) Bopp et al, 2013. Mitotic Evolution of Plasmodium falciparum Shows a Stable Core Genome but Recombination in Antigen Families.

chrom_names       = range(1,15) # + ['MT','Api']
chrom_lengths_Mbp = [ 0.643, 0.947,  1.1,  1.2,   1.35,  1.42,  1.45,
                      1.5,   1.55,   1.7,  2.05,  2.3,   2.95,  3.3 ]
chrom_lengths_bp  = [ int(bp_per_Mbp*c) for c in chrom_lengths_Mbp ]
Pf_chrom_lengths  = dict(zip(chrom_names,chrom_lengths_bp))
#log.debug('Chromosome lengths:\n%s',Pf_chrom_lengths)

def initialize_from(SNP_source,bin_size=None,min_allele_freq=0):
    SNPs=SNP.initialize_from(SNP_source,min_allele_freq)
    set_bin_size(SNPs,bin_size)
    set_chrom_breaks()
    set_binned_SNPs(SNPs)
    
def set_bin_size(SNPs,bin_size):
    if bin_size:
        Genome.bin_size_bp=bin_size
    else:
        min_dist=find_closest_SNPs(SNPs)
        Genome.bin_size_bp=get_rounded_binning(min_dist,2)

def find_closest_SNPs(SNPs):
    snp_pos_by_chrom=defaultdict(list)
    for snp in SNPs:
        snp_pos_by_chrom[snp.chrom].append(snp.pos)
    min_dist=[]
    for c,s in snp_pos_by_chrom.items():
        p=sorted(s)
        d=[j-i for i, j in itertools.izip(p[:-1], p[1:])]
        if d:
            min_dist=min(d+[min_dist])
    #log.debug('Closest SNPs = %d bp', min_dist)
    return min_dist

def get_rounded_binning(min_dist, ndigits=1):
    if min_dist<1:
        raise Exception('Genome binning must be a postive integer.')
    log10_floor = math.floor(math.log10(min_dist))
    trunc_digits = -int(log10_floor) + (ndigits - 1)
    bin_size = round(min_dist - math.pow(10,-trunc_digits)//2, trunc_digits)
    #log.debug('Rounded genome-discretization binning = %d bp', bin_size)
    return bin_size

def set_chrom_breaks():
    Genome.chrom_idxs={v:i for i,v in enumerate(chrom_names)}
    bins_per_chrom = [n_bins_from(bp) for bp in chrom_lengths_bp]
    Genome.chrom_breaks = list(utils.cumsum([0]+bins_per_chrom))
    #log.debug('chrom_breaks:',Genome.chrom_breaks)

def n_bins_from(bp):
    return int(math.ceil(bp/Genome.bin_size_bp))

def snp_bin_from_chrom_pos(chrom,pos):
    return int(pos/Genome.bin_size_bp) + Genome.chrom_breaks[Genome.chrom_idxs[chrom]]

def set_binned_SNPs(SNPs):
    Genome.SNP_bins=[]
    Genome.SNP_freqs=[]
    Genome.SNP_names=[]
    last_snp,last_bin,max_freq=(None,0,0)
    for snp in SNPs:
        snp_bin=snp_bin_from_chrom_pos(snp.chrom,snp.pos)
        if last_snp and (snp.chrom,snp_bin)==(last_snp.chrom,last_bin):
            log.debug('overlap at chrom %d bin %d',snp.chrom,snp_bin)
            if snp.freq > max_freq:
                Genome.SNP_freqs[-1]=snp.freq # replace
                Genome.SNP_names[-1]='Pf.%d.%d'%(snp.chrom,snp.pos)
                max_freq=snp.freq
        else:
            Genome.SNP_bins.append(snp_bin) # append
            Genome.SNP_freqs.append(snp.freq)
            Genome.SNP_names.append('Pf.%s.%s'%(snp.chrom,snp.pos))
            last_snp,last_bin=(snp,snp_bin)

    Genome.bin_fitness = np.ones(genome_length())

    #log.info('%d of %d unique SNPs after discretization at %dbp binning',
             #len(Genome.SNP_bins),len(SNPs),Genome.bin_size_bp)

def add_locus(chrom,pos,name,fitness,freq=0):
    add_bin = snp_bin_from_chrom_pos(chrom,pos)
    if add_bin in Genome.SNP_bins:
        idx = bisect.bisect_left(Genome.SNP_bins, add_bin)
        #log.warning('SNP already exists at bin %d: chrom,pos=(%d,%d)' % (add_bin, chrom, pos))
        Genome.SNP_names[idx] = name
        Genome.SNP_freqs[idx] = freq
    else:
        Genome.SNP_bins.append(add_bin)
        Genome.SNP_names.append(name)
        Genome.SNP_freqs.append(freq)
    Genome.bin_fitness[add_bin] = fitness
                  
def reference_genome():
    return np.zeros(genome_length(),dtype=np.uint8)

def num_SNPs():
    return len(Genome.SNP_bins)

def genome_length():
	return Genome.chrom_breaks[-1]

def display_bit(b):
    return '*' if b else '-'

obligate_co_scale_fn = np.poly1d(np.array([  2.86063522e-05,  -1.28111927e-03,   2.42373279e-02,
                                            -2.52092360e-01,   1.57111461e+00,  -5.99256708e+00,
                                             1.36678013e+01,  -1.72133175e+01,   9.61531678e+00]))
                                 
def gamma_interarrival_time(v=1):   #v=1 means no interference
    '''returns the distance in map units of the next chiasma event in the 4 chromatid bundle
    interrarival times specify the next event (in Morgan)'''
    interarrival_time = np.random.gamma(shape=v, scale =1./(2*v))
    #rate parameter must be constrained to equal 2*shape
    #scale = 1/shape
    #for rate to be constrained to 2*shape, scale must = 1/(2v)
    
    #conversion to bp, rounded up
    d = int(math.ceil(interarrival_time * bp_per_cM* 100))
    return d
                                 
def get_crossover_points(v,chrom_length):
    '''not ever used, generates chiasma locations on 4-chromatid bundle'''
    next_point=-100000 #make it a stationary renewal process
    xpoints=[]
    while next_point < chrom_length:
        if next_point > 0.:
            xpoints.append(next_point)
        d = gamma_interarrival_time(v)
        next_point+=d
    return xpoints

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
    left_point = obligate_chiasma_pos - d
    
    while right_point < chrom_length:
        xpoints.append(right_point)
        d = gamma_interarrival_time(v)
        right_point += d
        
    while left_point > 0:
        xpoints.append(left_point)
        d= gamma_interarrival_time(v)
        left_point -= d
        
    return xpoints

def crossover(g1,g2,xpoints):
    #S phase, DNA duplication time
    c1 = np.copy(g1)
    c2 = np.copy(g1)
    
    c3 = np.copy(g2)
    c4 = np.copy(g2)
    if not xpoints:
        return c1,c2, c3,c4
    
    for breakpoint in xpoints:
        probability = np.random.random()
        if probability < 0.25: 
            # c1 and c3
            t = np.copy(c1[breakpoint:])
            c1[breakpoint:] = c3[breakpoint:]
            c3[breakpoint:] = t
        elif probability >= 0.25 and probability < 0.5: 
            #c1 and c4
            t = np.copy(c1[breakpoint:])
            c1[breakpoint:] = c4[breakpoint:]
            c4[breakpoint:] = t
        elif probability >= 0.5 and probability < 0.75: 
            #c2 and c3
            t = np.copy(c2[breakpoint:])
            c2[breakpoint:] = c3[breakpoint:]
            c3[breakpoint:] = t
        elif probability >=0.75:
            #c2 and c4
            t = np.copy(c2[breakpoint:])
            c2[breakpoint:] = c4[breakpoint:]
            c4[breakpoint:] = t
    return c1, c2, c3, c4

def meiosis(in1,in2,N=4,v=2, oc=True):
    '''v defines the shape of the gamma distribution, it is required to have a non-zero shape parameter
    if v = 0, we assume user means no crossover model
    v =1 corresponds to no interference
    obligate crossover means use the obligate crossover version'''
    if N > 4:
        raise IndexError('Maximum of four distinct meiotic products to sample.')
    genomes=[reference_genome() for _ in range(4)]
    
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


def distinct_sporozoites_from(gametocyte_pairs,n_products):
    transmitted_sporozoites=[]
    for (g1,g2),N in zip(gametocyte_pairs,n_products):
        if g1.id==g2.id:
            #log.debug('Selfing of gametocytes (id=%d)\n%s',g1.id,g1)
            t=Transmission((g1.id,g2.id),g1)
            transmitted_sporozoites.append(t)
        else:
            meiotic_products=meiosis(g1,g2,N)
            #log.debug('Meiosis: %s',[str(mp) for mp in meiotic_products])
            tt=[Transmission((g1.id,g2.id),g) for g in meiotic_products]
            transmitted_sporozoites.extend(tt)
    return distinct(transmitted_sporozoites,id_fn=lambda t:t.genome.id)


def distinct(genomes,
             id_fn=lambda g:g.id):
             #id_fn=lambda g: hash(g)):
    distinct=[]
    seen = set()
    for g in genomes:
        h=id_fn(g)
        if h not in seen:
            seen.add(h)
            distinct.append(g)
    return distinct

class Genome:
    '''
    The discretized representation of SNPs on chromosomes
    '''

    id=itertools.count()
    hash_to_id   = {}

    chrom_idxs    = {} # chrom_break indices by chromosome name
    chrom_breaks  = [] # locations of chromosome breakpoints on genome
    SNP_bins      = [] # locations of variable positions on genome
    SNP_names     = [] # chrom.pos encoding of binned SNPs
    SNP_freqs     = [] # minor-allele frequency of binned SNPs
    bin_fitness   = [] # relative fitness at each binned site
    bin_size_bp   = [] # base pairs per genome bin

    # TODO: find a better thread-safe way of letting Genome know what
    #       Simulation to notify on reportable events
    sim = None
    @classmethod
    def set_simulation_ref(cls,sim):
        cls.sim=sim

    def __init__(self,genome,mod_fns=[]):
        self.genome=genome
        for fn in mod_fns:
            fn(self.genome)
        h=hash(self)
        id=Genome.hash_to_id.get(h,None)
        #id=None
        if id:
            self.id=id
        else:
            self.id=Genome.id.next()
            Genome.hash_to_id[h]=self.id
            if Genome.sim:
                Genome.sim.notify('genome.init',self)
        #log.debug('Genome: id=%d', self.id)
        #log.debug('%s', self)

    def __repr__(self):
        return 'Genome(%s)'%self.genome

    def __str__(self):
        return self.display_barcode()
        #return self.display_genome()

    def __hash__(self):
        h=hashlib.sha1
        #h=hashlib.md5
        return int(h(self.genome.view(np.uint8)).hexdigest(),16)

    @classmethod
    def from_reference(cls):
        return cls(reference_genome())

    @classmethod
    def from_allele_freq(cls,mod_fns=[]):
        rands=np.random.random_sample((num_SNPs(),))
        barcode=rands<Genome.SNP_freqs
        return cls.from_barcode(barcode,mod_fns)

    @classmethod
    def from_barcode(cls,barcode,mod_fns=[]):
        genome=reference_genome()
        np.put(genome,Genome.SNP_bins,barcode)
        return cls(genome,mod_fns)

    def fitness(self):
        m=self.bin_fitness[self.genome!=0] # NB: assuming binary SNPs
        return np.product(m) if m.size else 1.

    def barcode(self,sites=None):
        if not sites:
            sites=Genome.SNP_bins
        return self.genome[sites]

    def display_barcode(self):
        return ''.join([display_bit(b) for b in self.barcode()])

    def display_genome(self):
        s=[]
        for idx,(start,end) in enumerate(utils.pairwise(Genome.chrom_breaks)):
            s.append('Chromosome %s' % chrom_names[idx])
            s.append(''.join([display_bit(b) for b in self.genome[start:end]]))
        return '\n'.join(s)
        
    def return_chromosome(self, chromosome):
        start = Genome.chrom_breaks[chromosome -1]
        end = Genome.chrom_breaks[chromosome]
        return self.genome[start:end]
        
    def reference_genome(self):
        return reference_genome()
        