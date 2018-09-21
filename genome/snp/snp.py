from importlib import import_module
import sys

class SNP:
    '''
    The properties of a single nucleotide polymorphism
    '''

    def __init__(self,chrom,pos,freq=0.5,bin=None):
        self.chrom=chrom
        self.pos=pos
        self.freq=freq

    def __repr__(self):
        return 'genome.SNP(%d,%d)'%self.to_tuple()

    def to_tuple(self):
        return (self.chrom,self.pos)

    @staticmethod
    def initialize_from(SNP_source,min_allele_freq):
        try:
            mod=import_module(SNP_source)

            return mod.init(min_allele_freq)
        except ImportError as e:
            sys.exit("ImportError for SNP_source: %s"%e)
