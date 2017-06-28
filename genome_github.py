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
        log.debug('Genome: id=%d', self.id)
        log.debug('%s', self)

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
        