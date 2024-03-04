import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor import *
from quimb.tensor.tensor_2d import *
from quimb.tensor.tensor_1d import *
import time
from constants import *
from mcmc import *
from tqdm import tqdm, trange


def main():
    
    L = 25
    ta = 0
    tb = np.pi/4
    num_samples = 12000
    T = T_()
    op = [Bp(ta, tb), Bm(ta, tb)]   
    X = np.zeros([2,2])
    X[0,1] =1
    X[1,0] =1
    
    def spin_x(state):
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
                    if x == 12 and y==12:           
                        peps[x,y].modify(data = np.expand_dims(bulk(T, X, op[state[i+1]]), axis =4))
                    else:
                        peps[x,y].modify(data = np.expand_dims(bulk(T, op[state[i]], op[state[i+1]]), axis =4))
                    i+=2
        norm = dummy & peps
        return norm.contract_boundary(max_bond=32)
        
    obs = []
    samples = np.load("samples2.npy")
    for sample in tqdm(samples):
        obs.append(spin_x(sample))
        
    np.save("spin_x", obs)
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