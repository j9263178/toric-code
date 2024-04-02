import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor import *
from quimb.tensor.tensor_2d import *
from quimb.tensor.tensor_1d import *
import time
from constants import *
from mcmc import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--p", type=float, default=1.)
parser.add_argument("--L", type=int, default=6)
parser.add_argument("--num_samples", type=int, default=12000)
parser.add_argument("--init_type", default="ONES")
parser.add_argument("--insample_npy", default="")
parser.add_argument("--insample_index", type=int, default=-1)
parser.add_argument("--outsample", default="out_sample")
parser.add_argument("--outp", default="out_p")
args = parser.parse_args()

def main():
    
    print(args)
    beta = np.arctanh(1-2*args.p)
    L = args.L
    num_samples = args.num_samples
    T2 = T_2()
    T2_up = T_2_up()
    T3 = T_3()
    T4 = T_4()
    op = [Wp(beta), Wm(beta)]
    
    n = np.max([np.max(op[0]),np.max(op[1])])
    #n = np.max([np.linalg.norm(op[0]),np.linalg.norm(op[1])])
    op[0]/=n
    op[1]/=n
    
    print(op[0])
    print(op[1])
    
    def proposal(state):
        new_state = state.copy()
        loc = np.random.randint(0, 2*L*(L-1))
        if new_state[loc] == 1:
            new_state[loc] = 0
        else:
            new_state[loc] = 1
        return new_state
    
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
                    peps[x,y].modify(data = np.expand_dims(top_left(T2, op[state[i]]), axis = 2))
                    # print(peps[x,y].data)
                    i+=1
                elif x == L-1 and y == 0:
                    peps[x,y].modify(data = np.expand_dims(top_right(T2), axis =2))
                elif y == 0:
                    peps[x,y].modify(data = np.expand_dims(top(T3, op[state[i]]), axis =3))
                    i+=1
                elif x == 0 and y == L-1:
                    peps[x,y].modify(data = np.expand_dims(down_left(T2_up, op[state[i]], op[state[i+1]]), axis =2))
                    i+=2
                elif x == 0:
                    peps[x,y].modify(data = np.expand_dims(left(T3, op[state[i]], op[state[i+1]]), axis =3))
                    i+=2
                elif x == L-1 and y == L-1:
                    peps[x,y].modify(data = np.expand_dims(down_right(T2, op[state[i]]), axis =2))
                    i+=1   
                elif x == L-1:
                    peps[x,y].modify(data = np.expand_dims(right(T3, op[state[i]]), axis =3))
                    i+=1
                elif y == L-1:
                    peps[x,y].modify(data =  np.expand_dims(down(T3, op[state[i]], op[state[i+1]]), axis =3))
                    i+=2
                else:                
                    peps[x,y].modify(data = np.expand_dims(bulk(T4, op[state[i]], op[state[i+1]]), axis =4))
                    i+=2
                    
        norm = dummy & peps
        res = norm.contract_boundary(max_bond=32)
        # print(res)
        return res
        
    in_state = []
    
    if args.init_type == 'ONES':
        in_state = np.ones([2*L*(L-1)],dtype = np.int16)
    elif args.init_type == 'RAND':
        in_state = np.random.randint(2, size=2*L*(L-1))
            
    print(in_state)
    
    samples, ps = metropolis_hastings(p, proposal, in_state, num_samples, burnin=0.2)
    # print(ps)
    np.save(args.outsample, samples)
    np.save(args.outp, ps)

    
    

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