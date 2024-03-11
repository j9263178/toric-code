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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ta_", type=float, default=1.)
parser.add_argument("--tb_", type=float, default=1.)
parser.add_argument("--L", type=int, default=6)
parser.add_argument("--insample", default="")
parser.add_argument("--out", default="out_EA")
args = parser.parse_args()

def main():
    
    L = args.L
    ta = args.ta_*np.pi/4
    tb = args.tb_*np.pi/4
    T2 = T_2()
    T2_up = T_2_up()
    T3 = T_3()
    T4 = T_4()
    Tc = T_c()
    op = [Bp(ta, tb), Bm(ta, tb)]
    
    n = np.max([np.max(op[0]),np.max(op[1])])
    #n = np.max([np.linalg.norm(op[0]),np.linalg.norm(op[1])])
    op[0]/=n
    op[1]/=n
    
    X = np.zeros([2,2])
    X[0,1] =1
    X[1,0] =1
    Z = np.zeros([2,2])
    Z[0,0] =1
    Z[1,1] =-1
    Y = np.zeros([2,2], dtype = np.complex128)
    Y[0,1] =-1j
    Y[1,0] =1j
    

    # def spin_x(state):
    #     '''
    #         p({s}) of a given configuration {s}.
    #     '''
    #     dummy = qtn.PEPS.ones(Lx=L, Ly=L, phys_dim= 1, bond_dim=1)
    #     peps = qtn.PEPS.ones(Lx=L, Ly=L, phys_dim= 1, bond_dim=2)
    #     i = 0
    #     for y in range(L):
    #         for x in range(L):
    #             if x == 0 and y == 0:
    #                 peps[x,y].modify(data = np.expand_dims(top_left(T, op[state[i]]), axis = 2))
    #                 i+=1
    #             elif x == L-1 and y == 0:
    #                 peps[x,y].modify(data = np.expand_dims(top_right(T), axis =2))
    #             elif y == 0:
    #                 peps[x,y].modify(data = np.expand_dims(top(T, op[state[i]]), axis =3))
    #                 i+=1
    #             elif x == 0 and y == L-1:
    #                 peps[x,y].modify(data = np.expand_dims(down_left(T, op[state[i]], op[state[i+1]]), axis =2))
    #                 i+=2
    #             elif x == 0:
    #                 peps[x,y].modify(data = np.expand_dims(left(T, op[state[i]], op[state[i+1]]), axis =3))
    #                 i+=2
    #             elif x == L-1 and y == L-1:
    #                 peps[x,y].modify(data = np.expand_dims(down_right(T, op[state[i]]), axis =2))
    #                 i+=1   
    #             elif x == L-1:
    #                 peps[x,y].modify(data = np.expand_dims(right(T, op[state[i]]), axis =3))
    #                 i+=1
    #             elif y == L-1:
    #                 peps[x,y].modify(data =  np.expand_dims(down(T, op[state[i]], op[state[i+1]]), axis =3))
    #                 i+=2
    #             else:
    #                 if x == 12 and y==12:           
    #                     peps[x,y].modify(data = np.expand_dims(bulk(T, X, op[state[i+1]]), axis =4))
    #                 else:
    #                     peps[x,y].modify(data = np.expand_dims(bulk(T, op[state[i]], op[state[i+1]]), axis =4))
    #                 i+=2
    #     norm = dummy & peps
    #     return norm.contract_boundary(max_bond=32)
    
    def EA(state):
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
                    if x == int(L/2) and y== int(L/2):           
                        peps[x,y].modify(data = np.expand_dims(bulk(Tc, op[state[i]], op[state[i+1]]), axis =4))
                    else:
                        peps[x,y].modify(data = np.expand_dims(bulk(T4, op[state[i]], op[state[i+1]]), axis =4))
                    i+=2
        norm = dummy & peps
        return norm.contract_boundary(max_bond=32)
    
    # def SSSS(state, O):
    #     dummy = qtn.PEPS.ones(Lx=L, Ly=L, phys_dim= 1, bond_dim=1)
    #     peps = qtn.PEPS.ones(Lx=L, Ly=L, phys_dim= 1, bond_dim=2)
    #     i = 0
    #     for y in range(L):
    #         for x in range(L):
    #             if x == 0 and y == 0:
    #                 peps[x,y].modify(data = np.expand_dims(top_left(T, O), axis = 2))
    #                 i+=1
    #             elif x == L-1 and y == 0:
    #                 peps[x,y].modify(data = np.expand_dims(top_right(T), axis =2))
    #             elif y == 0:
    #                 peps[x,y].modify(data = np.expand_dims(top(T, op[state[i]]), axis =3))
    #                 i+=1
    #             elif x == 0 and y == L-1:
    #                 peps[x,y].modify(data = np.expand_dims(down_left(T, op[state[i]], op[state[i+1]]), axis =2))
    #                 i+=2
    #             elif x == 0:
    #                 peps[x,y].modify(data = np.expand_dims(left(T, op[state[i]], op[state[i+1]]), axis =3))
    #                 i+=2
    #             elif x == L-1 and y == L-1:
    #                 peps[x,y].modify(data = np.expand_dims(down_right(T, op[state[i]]), axis =2))
    #                 i+=1   
    #             elif x == L-1:
    #                 peps[x,y].modify(data = np.expand_dims(right(T, op[state[i]]), axis =3))
    #                 i+=1
    #             elif y == L-1:
    #                 peps[x,y].modify(data =  np.expand_dims(down(T, op[state[i]], op[state[i+1]]), axis =3))
    #                 i+=2
    #             else:
    #                 if i+1 == 21:
    #                     peps[x,y].modify(data = np.expand_dims(bulk(T, op[state[i]], O), axis =4))
    #                 if i == 31:
    #                     peps[x,y].modify(data = np.expand_dims(bulk(T, O, O), axis =4))
    #                 if i == 33:
    #                     peps[x,y].modify(data = np.expand_dims(bulk(T, O, op[state[i+1]]), axis =4))
    #                 i+=2
    #     norm = dummy & peps
    #     res = norm.contract_boundary(max_bond=32)
    #     # print(res)
    #     return res
    
    obs = []
    samples = np.load(args.insample)
    for sample in tqdm(samples):
        # print(EA(sample,Z))
        obs.append(EA(sample))
        #obs.append(SSSS(sample,Z))
    np.save(args.out, obs)
    
    # sample = np.ones([2*L*(L-1)], dtype = np.int16)
    # sample = np.random.randint(2, size=2*L*(L-1))
    # sample = np.zeros([2*L*(L-1)], dtype = np.int16)
    # print(p(sample))
    # print(EA(sample,X+Y+Z))
    
        

    
    

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