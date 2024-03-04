import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor import *
from quimb.tensor.tensor_2d import *
from quimb.tensor.tensor_1d import *
import time
from constants import *
from mcmc import *


def main():
    
    L = 25
    ta = 0
    tb = np.pi/4
    num_samples = 12000
    T = T_()
    op = [Bp(ta, tb), Bm(ta, tb)]   
    
    def proposal(state):
        loc = np.random.randint(0, 2*L*(L-1))
        if state[loc] == 1:
            state[loc] = 0
        else:
            state[loc] = 1
        return state
    
    def p(state):
        '''
            p({s}) of a given configuration {s}.
        '''
        dummy = qtn.PEPS.ones(Lx=L, Ly=L, phys_dim= 1, bond_dim=1)
        peps = qtn.PEPS.ones(Lx=L, Ly=L, phys_dim= 1, bond_dim=2)
        i = 0
        for y in range(L):
            for x in range(L):
                if x == 0 and y == 0:
                    peps[x,y].modify(data = np.expand_dims(top_left(T, op[state[i]]), axis = 2))
                    # print(peps[x,y].data)
                    i+=1
                elif x == L-1 and y == 0:
                    peps[x,y].modify(data = np.expand_dims(top_right(T), axis =2))
                elif y == 0:
                    peps[x,y].modify(data = np.expand_dims(top(T, op[state[i]]), axis =3))
                    i+=1
                elif x == 0 and y == L-1:
                    peps[x,y].modify(data = np.expand_dims(down_left(T, op[state[i]], op[state[i+1]]), axis =2))
                    i+=2
                elif x == 0:
                    peps[x,y].modify(data = np.expand_dims(left(T, op[state[i]], op[state[i+1]]), axis =3))
                    i+=2
                elif x == L-1 and y == L-1:
                    peps[x,y].modify(data = np.expand_dims(down_right(T, op[state[i]]), axis =2))
                    i+=1   
                elif x == L-1:
                    peps[x,y].modify(data = np.expand_dims(right(T, op[state[i]]), axis =3))
                    i+=1
                elif y == L-1:
                    peps[x,y].modify(data =  np.expand_dims(down(T, op[state[i]], op[state[i+1]]), axis =3))
                    i+=2
                else:                
                    peps[x,y].modify(data = np.expand_dims(bulk(T, op[state[i]], op[state[i+1]]), axis =4))
                    i+=2
                    
        norm = dummy & peps
        
        res = norm.contract_boundary(max_bond=32)
        print(res)
        # exit()
        return res
        

    # in_state = np.random.randint(2, size=2*L*(L-1))
    in_state = np.ones([2*L*(L-1)],dtype = np.int16)
    print(in_state)
    samples = metropolis_hastings(p, proposal, in_state, num_samples, burnin=0.2)
    np.save("samples_test", samples)
    
    # a = time.time()
    # p(state)
    # b = time.time()
    # print(b-a)

    
    

if __name__=='__main__':
    # memory size of numpy array in bytes
    # create a numpy 1d-array
    # x = np.ones([10000,25,25])
    # print("Size of the array: ",x.size)
    
    # print("Memory size of one array element in bytes: ",
    #     x.itemsize)
    
    # # memory size of numpy array in bytes
    # print("Memory size of numpy array in bytes:",
    #     x.size * x.itemsize)
    main()