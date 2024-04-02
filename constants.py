import numpy as np

def delta(a,b):
    return 1 if a == b else 0
    
def T_2_up():
    T = np.zeros([2,2])
    
    T[0,0] = 1

    return T

def T_2():
    T = np.zeros([2,2])
    
    T[0,0] = 1
    T[1,1] = 1

    return T

def T_3():
    T = np.zeros([2,2,2])
    
    T[0,0,0] = 1
    T[1,1,1] = 1

    return T

def T_4():
    T = np.zeros([2,2,2,2])
    
    T[0,0,0,0] = 1
    T[1,1,1,1] = 1

    return T

def T_c():
    T = np.zeros([2,2,2,2])
    
    T[0,0,0,0] = 1
    T[1,1,1,1] = -1

    return T

def Bp(ta, tb):
    
    Bp = np.zeros([2,2])
    
    Bp[0,0] = np.cos(ta+tb)**2
    Bp[0,1] = np.cos(ta-tb)**2
    Bp[1,0] = np.cos(ta-tb)**2
    Bp[1,1] = np.cos(ta+tb)**2
    
    return Bp
    
def Bm(ta, tb):
    
    Bm = np.zeros([2,2])
    
    Bm[0,0] = np.sin(ta+tb)**2
    Bm[0,1] = np.sin(ta-tb)**2
    Bm[1,0] = np.sin(ta-tb)**2
    Bm[1,1] = np.sin(ta+tb)**2
    
    return Bm


def Wp(beta):
    
    Bp = np.zeros([2,2])
    
    Bp[0,0] = np.exp(-beta)
    Bp[0,1] = np.exp(beta)
    Bp[1,0] = np.exp(beta)
    Bp[1,1] = np.exp(-beta)
    
    return Bp
    
def Wm(beta):
    
    Bm = np.zeros([2,2])
    
    Bm[0,0] = np.exp(beta)
    Bm[0,1] = np.exp(-beta)
    Bm[1,0] = np.exp(-beta)
    Bm[1,1] = np.exp(beta)
    
    return Bm

    
def bulk(T, B1,B2):
    return np.einsum('ijdl,iu,jr->urdl',T, B1, B2)

def top_left(T, B1):
    return np.einsum('id,ir->rd',T, B1)

def top(T, B1):
    return np.einsum('idl,ir->rdl',T, B1)

def top_right(T):
    return np.einsum('dl',T)

def left(T, B1, B2):
    return np.einsum('ijd,iu,jr->urd',T, B1, B2)

def right(T, B1):
    return np.einsum('idl,iu->udl',T, B1)

def down_left(T, B1, B2):
    return np.einsum('ij,iu,jr->ur',T, B1, B2)

def down(T, B1, B2):
    return np.einsum('ijl,iu,jr->url',T, B1, B2)

def down_right(T, B1):
    return np.einsum('il,iu->ul',T, B1)


    