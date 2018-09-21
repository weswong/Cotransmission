import math
import random
from itertools import tee,izip,takewhile


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def by_pair(l):
    "l -> (l0,l1), (l2,l3), (l4, l5), ..."
    if len(l) % 2:
        l.append(None)
    a = iter(l)
    return izip(a, a)

def cumsum(it):
    total = 0
    for x in it:
        total += x
        yield total

def accumulate_cdf(iterable):
    cdf,subtotal=[],0
    norm=float(sum(iterable))
    if not norm:
        return []
    for it in iterable:
        subtotal += it/norm
        cdf.append(subtotal)
    cdf[-1]=1.0
    return cdf

def nextTime(rateParameter):
    if rateParameter<=0:
        raise Exception('Cannot calculate next time from zero rate.')
    return -math.log(1.0 - random.random()) / rateParameter

def poissonRandom(lam):
    if lam<=0:
        return 0
    sumTime=0
    N=0
    while True:
        sumTime+=nextTime(lam)
        if sumTime>1:
            break
        N+=1
    return N



def binomialApproxRandom(n,p):
    '''
    Small numbers: exact Binomial
    Near edges: Poisson approximation
    Intermediate probabilities: normal approximation
    '''
    if n<0 or p<0:
        raise Exception('Binomial requires positive (n,p)=(%d,%f)'%(n,p))
    if n==0 or p==0:
        return 0
    if p>1:
        log.warning('Fixing probability %f>1.0 to one.',p)
        return n
    if n<10:
        return(sum([random.random()<p for _ in range(n)]))
    lt50pct=p<0.5
    p_tmp = p if lt50pct else (1-p)
    if n < 9*(1-p_tmp)/p_tmp:
        poisson_tmp=poissonRandom(n*p_tmp)
        return poisson_tmp if lt50pct else n-poisson_tmp
    normal_tmp=int(round(random.gauss(mu=n*p,sigma=math.sqrt(n*p*(1-p)))))
    return max(0,min(n,normal_tmp))

def weighted_choice(cumwts):
    R = random.random()
    idx=sum(takewhile(bool, (cw < R for cw in cumwts)))
    return idx

def choose_with_replacement(M,N):
    choose = random.choice
    indices = range(N)
    return [choose(indices) for _ in range(M)]

def choose_without_replacement(M,N):
    '''
    Floyd algorithm: O(M) in choose M from N scenario,
    which can be much faster for some typical use cases
    than random.sample, which seems to be O(N)
    '''
    if M>N:
        raise Exception('Cannot sample %d from %d without replacement',(M,N))
    if M==N:
        return range(M)
    chosen_idxs=set()
    for j in range(N-M,N):
        t = random.randint(0,j)
        idx = t if t not in chosen_idxs else j
        chosen_idxs.add(idx)
    return list(chosen_idxs)
