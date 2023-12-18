import numpy as np

def delta(a,b):
    return 1 if a == b else 0
    
def ising():
    #T_l,u,r,d = \delta_l,u  \delta_l,r \delta_l,d

    # for l in range(0,2):
    #     for u in range(0,2):
    #         for r in range(0,2):
    #             for d in range(0,2):
    #                 T[l,u,r,d] = delta(l,u) *delta(l,r) *delta(l,d)
    T = np.zeros([2,2,2,2])
    Bp = np.zeros([2,2])

    T[0,0,0,0] = 1
    T[1,1,1,1] = 1

    ta, tb = 0, 0

    Bp[0,0] = cos(ta+tb)**2
    Bp[0,1] = cos(ta-tb)**2
    Bp[1,0] = cos(ta-tb)**2
    Bp[1,1] = cos(ta+tb)**2

    Bm[0,0] = sin(ta+tb)**2
    Bm[0,1] = sin(ta-tb)**2
    Bm[1,0] = sin(ta-tb)**2
    Bm[1,1] = sin(ta+tb)**2
    
    return T, Bp, Bm

ising()