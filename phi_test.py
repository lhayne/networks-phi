import numpy as np
from itertools import chain,combinations
from scipy import stats
from pyemd import emd
import pyphi

class phi():
    def __init__(self,tpm,state=None):
        self.ss_tpm = np.array(tpm)
        self.num_nodes = int(np.log2(tpm.shape[0]))
        self.state = np.array(state)
        self.sn_tpm = self.ss_to_sn(tpm)

        self.node_tpm = dict()

        for node in range(self.num_nodes):
            node_tpm = np.zeros([2 for _ in range(self.num_nodes+1)])
            node_tpm = np.sum(node_tpm,axis=node,keepdims=True)

            # we only marginalize out the node itself, because there are no self loops
            node_tpm[...,1] = self.marginalize_out(self.sn_tpm,[node])[...,node]
            node_tpm[...,0] = 1 - node_tpm[...,1]

            self.node_tpm[node] = node_tpm

        # this tracks the inputs to a node, which is assumed to be all other nodes in the network
        self.node_inputs = {i : np.setdiff1d(np.arange(self.num_nodes), [i]) for i in range(self.num_nodes)}


    def ss_to_sn(self,tpm):
        tpm = np.array(tpm)
        
        # number of states and number of nodes in given state by state tpm
        S = tpm.shape[0]
        N = int(np.log2(S))

        # this is a N x S matrix where cell (i,j) is 1 if node i is on in purview state j
        on_mask = np.array([[self.index_to_state(i,N)[n] for i in range(S)] for n in range(N)])

        # this is the new tensor tpm with N+1 dimensions. The first N dimensions are of size 2 (on, off)
        # the last dimension is size N indicating the probability of node n being on at time t+1
        sn_tpm = np.zeros(([2 for _ in range(N)] + [N]))
        
        # We find the likelihood of node n being on at time t+1 by multiplying by the on_mask and summing
        # accross rows. Because the rows normally sum to 1 it works out that summing across rows after multiplying
        # by the on_mask gives us the probability that node n is on at time t+1
        distributions = [np.sum(tpm * on_mask[n],axis=1) for n in range(N)]
        
        # populate the new tensor tpm. The paper warns that the mapping from a state by state tpm to a state by node
        # tpm is not one to one, but many to one, so you have to be careful not to lose information.
        for state_index in range(S):
            for node_index in range(N):
                sn_tpm[self.index_to_state(state_index,N)][node_index] = distributions[node_index][state_index]
        
        return sn_tpm


    # Converts a list of states to a row number, so that we can index the tpm
    # input : list of states, Ex. [0,1,0,0]
    # output : row number based on little endian notation, so index zero in list is least
    #          significant bit and index n-1 is most significant bit. Ex. [0,1,0,0] -> 2
    def state_to_index(self,states):
        decimal = 0
        for i,state in enumerate(states):
            decimal += (2**i) * (state)
        return decimal
    
    # Converts a row number from the tpm into a list of states
    # input : row number, total number of states. Ex. (2, 4)
    # output : list of states that row number represents. Ex. (2, 4) -> [0,1,0,0]
    #          We include the number of states so that we know how many zeros to append at the end.
    def index_to_state(self,index,num_states):
        binary = bin(index)[2:]
        states = [int(i) for i in binary[::-1]] # flip to make little endian with [::-1]

        if (len(states) < num_states): 
            states = states + [0 for _ in range(num_states-len(states))]

        return tuple(states)

    # return the powerset of a list of nodes including full set
    # function courtesy of stack overflow: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    def powerset_all(self,iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(0,len(s)+1))

    def marginalize_out(self,tpm,nodes):
        # marginalize out nodes indexes from tpm (state x node)

        normalizer = np.prod(np.array(tpm.shape)[list(nodes)])

        # This sums over the nodes listed
        tpm = np.sum(tpm,tuple(nodes),keepdims=True)

        # this re-normalizes the distribution. We can simply divide by number of nodes we marginalized out
        # because any two 1's that get summed should now be a 1 (dividing by 2)
        tpm = tpm / normalizer

        return tpm

    """
    This function is currently borrowed from pyphi!
    """
    def condition_tpm(self,tpm, fixed_nodes, state):
        """Return a TPM conditioned on the given fixed node indices, whose states
        are fixed according to the given state-tuple.
        The dimensions of the new TPM that correspond to the fixed nodes are
        collapsed onto their state, making those dimensions singletons suitable for
        broadcasting. The number of dimensions of the conditioned TPM will be the
        same as the unconditioned TPM.
        """
        conditioning_indices = [[slice(None)]] * len(state)
        for i in fixed_nodes:
            # Preserve singleton dimensions with `np.newaxis`
            conditioning_indices[i] = [state[i], np.newaxis]
        # Flatten the indices.
        conditioning_indices = list(chain.from_iterable(conditioning_indices))
        # Obtain the actual conditioned TPM by indexing with the conditioning
        # indices.
        return tpm[tuple(conditioning_indices)]


    def repertoire_shape(self,purview,size):
        return [2 if i in purview else 1 for i in range(size)]

    def effect_repertoire_single(self,mechanism,purview):
        # The effect repertoire is calculated by conditioning the purview's nodes tpm on the state
        # of the mechanism nodes that are also parents of the purview node.
        mechanism_parents = np.intersect1d(mechanism,self.node_inputs[purview[0]])

        # condition on the mechanism nodes that are also parents of the purview node
        tpm = self.condition_tpm(self.node_tpm[purview[0]],mechanism_parents,self.state)

        # and marginalize out non mechanism nodes
        tpm = self.marginalize_out(tpm,np.setdiff1d(np.arange(self.num_nodes),mechanism))

        # before returning the repertoire needs to be in the shape of the next state
        return tpm.reshape(self.repertoire_shape(purview,self.num_nodes))



    # conveniently, the effect repertoire is the tensor product of all the individual purview tensors
    def effect_repertoire(self,mechanism,purview):
        effect_repertoire = np.ones(self.repertoire_shape(purview,self.num_nodes))

        for node in purview:
            effect_repertoire = effect_repertoire * self.effect_repertoire_single(mechanism,tuple([node]))

        return effect_repertoire


    # here we find the cause repertoire for a single mechanism node
    def cause_repertoire_single(self,mechanism,purview):

        # get the tpm for the mechanism and condition on the current mechanism's state, so this is still a conditional
        # probability distribution P(mechanism_node in current state | all parent nodes)
        tpm = self.node_tpm[mechanism[0]][...,self.state[mechanism]]

        # before we return we marginalize out the nodes which are parents of the mechanism, but not part of the purview
        # so now we have the distribution P(mechanism_node in current state | parents of node also in the purview)
        return self.marginalize_out(tpm,np.setdiff1d(self.node_inputs[mechanism[0]],purview))


    # normalize all the entries in the matrix, by dividing by the sum, unless the sum is zero.
    def normalize(self,tpm):
        return tpm / (1 if np.sum(tpm) == 0 else np.sum(tpm))

    # conveniently, the cause repertoire is the tensor product of all the individual purview tensors
    def cause_repertoire(self,mechanism,purview):

        # we know the size of the distribution should be dependent on the purview [2] possible states for each
        # purview node, [1] otherwise
        cause_repertoire = np.ones(self.repertoire_shape(purview,self.num_nodes))

        # multiply together all the distributions of the individual mechanism nodes the same way we did it for
        # the effect repertoire.
        for node in mechanism:
            cause_repertoire = cause_repertoire * self.cause_repertoire_single(tuple([node]),purview)

        # Each row in the tpm sums to 1, but to get a distribution over past states we must sum over some of 
        # the rows, so we lose that property and have to re-normalize
        return self.normalize(cause_repertoire)



    def effect_mip(self,system,purview):        
        # We need to find the cut the makes the least difference to the tpm of the system,
        # so we generate the original tpm first. We'll want to compare the original tpm
        # with the tpm generated after making our cut to determine the cut that makes the
        # least difference.
        min_cut = None
        cut_distance = float('inf')
        full_tpm = self.effect_repertoire(system,purview)
        
        # We need to loop through two powersets to determine all the cuts for this particular 
        # system + purview pair. We need to loop through the system powerset and the purview powerset
        # See factors.png below for an example.
        
        seen = set()
        
        for system_subset in self.powerset_all(system):
            
            for purview_subset in self.powerset_all(purview):
                
                # We haven't already seen the complement of the subset and the length of both of the factors
                # aren't zero
                if system_subset not in seen and not (len(system_subset) == 0 and len(purview_subset) == 0):
                    system_factor_1 = tuple(system_subset)
                    purview_factor_1 = tuple(purview_subset)
                    
                    system_factor_2 = tuple(np.setdiff1d(system,system_subset))
                    purview_factor_2 = tuple(np.setdiff1d(purview,purview_subset))
                    
                    tpm_1 = self.effect_repertoire(system_factor_1,purview_factor_1)
                    tpm_2 = self.effect_repertoire(system_factor_2,purview_factor_2)
                    
                    cut_tpm = np.multiply(tpm_1,tpm_2)
                    
                    # assess the distance between the cut_tpm and the full_tpm
                    new_cut_distance = pyphi.distance.effect_emd(full_tpm, cut_tpm)
                    
                    # print ((system_factor_1,purview_factor_1,system_factor_2,purview_factor_2)," ",new_cut_distance)
                    
                    # update the distance with the lowest distance
                    if new_cut_distance < cut_distance:
                        cut_distance = new_cut_distance
                        min_cut = (system_factor_1,purview_factor_1,system_factor_2,purview_factor_2)
            
            # add the complement of the system_subset to the seen set to ensure we don't double count certain
            # partitions
            seen.add(tuple(np.setdiff1d(system,system_subset)))
        
        return cut_distance,min_cut

# TODO: implement this. Right now it's not working when the purview is smaller than the full
    # list of nodes because the tensor product assumes the second dimension will be the full length
    # of all the states given all the nodes
    def mie(self,system):
        phi = float('-inf')
        mie = None
        for purview in self.powerset_all(np.arange(self.num_nodes)):
            if (len(purview) > 0):
                cut_distance,min_cut = self.effect_mip(system,purview)
                # print (cut_distance)
                if cut_distance > phi:
                    phi = cut_distance
                    mie = purview
        return phi,mie
    
    def concept(self,system):
        
        effect_phi,mie = self.mie(system)
        cause_phi,mic = self.mic(system) # to be implemented
        
        return (min(effect_phi,cause_phi),mic,mie)
                        

    def cause_mip(self,system,purview):        
        # We need to find the cut the makes the least difference to the tpm of the system,
        # so we generate the original tpm first. We'll want to compare the original tpm
        # with the tpm generated after making our cut to determine the cut that makes the
        # least difference.
        min_cut = None
        cut_distance = float('inf')
        full_tpm = self.cause_repertoire(system,purview)
        
        # We need to loop through two powersets to determine all the cuts for this particular 
        # system + purview pair. We need to loop through the system powerset and the purview powerset
        # See factors.png below for an example.
        
        seen = set()
        
        for system_subset in self.powerset_all(system):
            
            for purview_subset in self.powerset_all(purview):
                
                # We haven't already seen the complement of the subset and the length of both of the factors
                # aren't zero
                if system_subset not in seen and not (len(system_subset) == 0 and len(purview_subset) == 0):
                    system_factor_1 = tuple(system_subset)
                    purview_factor_1 = tuple(purview_subset)
                    
                    system_factor_2 = tuple(np.setdiff1d(system,system_subset))
                    purview_factor_2 = tuple(np.setdiff1d(purview,purview_subset))
                    
                    tpm_1 = self.cause_repertoire(system_factor_1,purview_factor_1)
                    tpm_2 = self.cause_repertoire(system_factor_2,purview_factor_2)
                    
                    cut_tpm = np.multiply(tpm_1,tpm_2)
                    # assess the distance between the cut_tpm and the full_tpm
                    new_cut_distance = pyphi.distance.hamming_emd(full_tpm, cut_tpm)
                    
#                     print ((system_factor_1,purview_factor_1,system_factor_2,purview_factor_2)," ",new_cut_distance)
                    
                    # update the distance with the lowest distance
                    if new_cut_distance < cut_distance:
                        cut_distance = new_cut_distance
                        min_cut = (system_factor_1,purview_factor_1,system_factor_2,purview_factor_2)
            
            # add the complement of the system_subset to the seen set to ensure we don't double count certain
            # partitions
            seen.add(tuple(np.setdiff1d(system,system_subset)))
        
        return cut_distance,min_cut

    def mic(self,system):
        phi = float('-inf')
        mic = None
        for purview in self.powerset_all(np.arange(self.num_nodes)):
            if (len(purview) > 0):
                cut_distance,min_cut = self.cause_mip(system,list(purview))
                # print(str(cut_distance) + ' ' + str(min_cut) + ' ' + str(purview))
                if cut_distance >= phi:
                    phi = cut_distance
                    mic = purview
        return phi,mic


















        