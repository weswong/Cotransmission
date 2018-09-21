import os

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from snp import SNP

allele_freqs=[0.274725, 0.0378378,      0.265537, 0.187166, 0.212766, 0.298343, 
              0.414365, 0.320261269683, 0.108108, 0.472527, 0.483516, 0.423841, 
              0.491525, 0.338889,       0.39779,  0.278689, 0.494444, 0.33871, 
              0.34,     0.210526,       0.426966, 0.0,      0.473684, 0.0]

def positions_from_txt_table(filename,allele_freqs=[]):
    '''
    Read SNP positions from file in following format:
    CHR    POS
    Pf3D7_01_v3    130339
    '''
    log.info('Reading SNPs from file: %s', filename)
    SNPs=[]
    with open(os.path.join(os.path.dirname(__file__),filename)) as f:
        for content in f.readlines()[1:]:
            CHR,POS = content.split()
            snp=SNP(chrom=int(CHR.split('_')[1]),
                       pos=int(POS))
            SNPs.append(snp)

    if allele_freqs:
        if len(allele_freqs) != len(SNPs):
            raise Exception('Incompatible lengths of \
                             SNP positions and allele frequencies')
        for i,s in enumerate(SNPs):
            s.freq=allele_freqs[i]

    return SNPs

def init(min_allele_freq=0):
    if min_allele_freq>0:
        raise Exception('Filtering on rare SNPs not supported for barcode.')
    SNPs=positions_from_txt_table('barcode_loci.txt',allele_freqs)
    return SNPs
