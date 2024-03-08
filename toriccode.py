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
parser.add_argument("--ta_", type=float, default=1.)
parser.add_argument("--tb_", type=float, default=1.)
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
    
    L = args.L
    ta = args.ta_*np.pi/4
    tb = args.tb_*np.pi/4
    num_samples = args.num_samples
    T = T_()
    op = [Bp(ta, tb), Bm(ta, tb)]   
    print(op[0])
    print(op[1])
    
    prec = []
    
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
        # print(norm[L-1,0])
        res = norm.contract_boundary(max_bond=32)
        # print(i)
        prec.append(res)
        return res
        
    in_state = []
    
    if args.init_type == 'ONES':
        in_state = np.ones([2*L*(L-1)],dtype = np.int16)
    elif args.init_type == 'RAND':
        in_state = np.random.randint(2, size=2*L*(L-1))
            
    print(in_state)
    
    samples = metropolis_hastings(p, proposal, in_state, num_samples, burnin=0.2)
    
    np.save(args.outsample, samples)
    np.save(args.outp, prec)

    
    

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