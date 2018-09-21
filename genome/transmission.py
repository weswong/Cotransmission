class Transmission:
    '''
    Characteristics of transmitted sporozoite and parent gametocytes
    '''
    def __init__(self,parentGenomeIds=(None,None),genome=None,
                      parentInfection=None,infection=None,
                      populationId=None,day=None,type=None):
        # sporozoite properties
        self.genome=genome
        self.infection=infection
        # male + female gametocyte properties
        self.parentGenomeIds=parentGenomeIds
        self.parentInfection=parentInfection
        # other info
        self.populationId=populationId
        self.day=day
        
        self.type = type
        #types grouped into 4 categories
        #       1) new: 1 strain, 1 mosquito, naiive host
        #       2) cotx: 2+ strains, 1 mosquito, naive host
        #       3) i_super (idealized_superinfection): 1 strain, previously infected hosts
        #       4) cotx_super (mix superinfection): cotx of 2 strains to previously infected host
        #       simultaneous infection not considered

    def to_tuple(self):
        return (self.day,self.populationId,
                getattr(self.infection,'id',None),
                getattr(self.parentInfection,'id',None),
                self.parentGenomeIds[0],self.parentGenomeIds[1],
                self.genome.id, self.type)
