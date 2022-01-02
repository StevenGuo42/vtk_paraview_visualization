import numpy as np
from evtk.hl import pointsToVTK

import math
sin, cos, sqrt = math.sin, math.cos, math.sqrt
pi = math.pi

##############################################################################

def StyblinskiTang(x):
    """StyblinskiTang
    :param x: n points, d dimension
    :type x: 2D [n, d] float array
    
    :return: f(x)
    :rtype: 1D [n] float array
    """

    (n,d) = x.shape 
    return np.asarray([[np.sum(np.array([0.5*(x[i,j]**4 - 16*x[i,j]**2 + 5*x[i,j]) for j in range(d)])) for i in range(n)]])



def Rosenbrock(x):
    """Rosenbrock
    :param x: n points, d dimension
    :type x: 2D [n, d] float array
    
    :return: f(x)
    :rtype: 1D [n] float array
    """
    
    (n,d) = x.shape 
    return np.asarray([[np.sum(np.array([(100.0 * (x[i,j+1] - x[i,j]**2)**2 + (1 - x[i,j])**2) for j in range(d-1)])) for i in range(n)]]).T


def Rastrigin(x):
    """Rastrigin
    :param x: n points, d dimension
    :type x: 2D [n, d] float array
    
    :return: f(x)
    :rtype: 1D [n] float array
    """
    
    (n,d) = x.shape 
    
    return np.asarray([[np.sum(np.array([10+(x[i,j]**2 - 10*cos(2*pi*x[i,j])) for j in range(d)])) for i in range(n)]]).T

def Sphere(x):
    """Sphere
    :param x: n points, d dimension
    :type x: 2D [n, d] float array
    
    :return: f(x)
    :rtype: 1D [n] float array
    """
    
    (n,d) = x.shape 
    
    return np.asarray([[np.sum(np.array([x[i,j]**2 for j in range(d)])) for i in range(n)]]).T

def Protein(M, TH):
    """Toy Protein Folding Model
    
        A--A--B--A
        4 monomer, 2 angles
        
    :param M: monomers in the chain
    :type M: string/1D [d] char array
    :param th: angles between the connection
    :type th: 2D [n, d-2] float array
    
    :return: Phi, potential-energy
    :rtype: 1D [n] float array
    """
    
    Marray = [ord(m)-ord('A') for m in list(M)]
    n = len(M); npt = TH.shape[0]
    TH = np.append(np.zeros((npt,1)), TH, axis=1)
    PHI = np.zeros((npt,1))
    C = [[1.0, -1/2], [-1/2, 1/2]]
    
    for ip in range(npt): 
        th = TH[ip, :]
        V1 = sum([0.25 * (1 - cos(x)) for x in th[1:n-1]])
        V2 = 0.0

        for i in range(0, n-2):
            for j in range(i+2, n):
                ii = Marray[i]
                jj = Marray[j]
    
                rij = 0.0 
                t1 = 0.0
                t2 = 0.0
                for k in range(i+1, j):
                    thtemp = sum(th[i+1:k+1])
                    t1 = t1 + cos(thtemp)
                    t2 = t2 + sin(thtemp)
                rij = sqrt((1.0 + t1)**2 + t2**2)
                
                V2 = V2 + (4.0 * (rij**-12.0 - C[ii][jj]*rij**(-6.0)))
        PHI[ip, 0] = V1 + V2;
    return PHI

##############################################################################

class CSRBF_struct():
    Nn = 0; Nd = 0  # num of points; dimension
    a = 0           # scaling factor
    x = None        # points on surrogate model
    lam = None      # lambda for the surrogate model
    
def CSRBF_3_1 (V, st):
    ''' phi_3,1(r) = (1-r)^4_+ * (4r+1)
    :param V: points to be evaluated on the surrogate
    :type V: 2D [npt, D] float array
    :param st: parameter for setting up CSRBF
    :type st: CSRBF_struct
    
    :return: CSRBF(V_0), ..., CSRBF(V_n)
    :rtype: 1D [npt,] float array

    '''
    
    npt = V.shape[0]
    fx = np.zeros((npt, 1))
    for ip in range(npt): 
        s = 0.0
        for i in range (st.Nn): # (i = 0; i < st->Nn; i++) {
            r = 0.0;
            for j in range(st.Nd): # (j = 0; j < st->Nd; j++) {
                r += (V[ip, j] - st.x[i, j])**2
    
            r = sqrt(r) / st.a
    
            if r <= 1.0: 
                temp0 = (1.0 - r) ** 4.0
                temp1 = (4.0 * r) + 1.0
                s += st.lam[i] * (temp0 * temp1)
        fx[ip, 0] = s 
        if ip%1000 == 0:
            print(ip,"/", npt)
    return fx

def setup_csrbf(X, a, f):
    '''coompute CSRBF parameters
    :param X: points used for the surrogate
    :type X: 2D [N, D] float array
    :param a: scaling factor 
    :type a: float
    :param f: value returned by true cost function
    :type f: 1D [N,] array
    
    :return: parameters for CSRBF
    :rtype: CSRBF_struct
    '''
    N, D = X.shape
    z = f(X)
    
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            R[i, j] = np.linalg.norm(X[i, :] - X[j, :]);
    R /= a
    PHI = ((1 - R)**4) * (4*R + 1)
    PHI *= (R < 1.0)
    lam = np.linalg.lstsq(PHI, z.flatten())[0] # lam = PHI\z
    
    params = CSRBF_struct()
    params.Nn = N; params.Nd = D; params.a = a
    params.x = X; params.lam = lam
    return params






##############################################################################
##
##   for toy protein
##
##############################################################################
f = Protein
M = "ABABAB"; n = 500000
lb = -pi; ub = pi

d = len(M); d-=2; less = 2
name = "Protein_"+M

Ngrid = 30 

    
xmin = -3.14
xmax = 3.14


fixed = np.asarray([np.nan, np.nan, np.nan, 1.0])

d_indx = np.asarray(np.where(np.isnan(fixed))).reshape((-1,))
d_grid = len(d_indx); D = len(fixed)

name+="_"+str(D)+"D_"+str(Ngrid)+"ptsGrid_"+str(list(fixed))

## Grid for evaluating the cost function or (CS)RBF....
GRID = np.linspace(xmin, xmax, Ngrid)


## Rows....
N = Ngrid**D

## Output....
pts = np.zeros((N, D+1))

## Parameter of the cost function....
x = np.zeros(D)

## Powers of Ngrid.... 
Ngrid_to_i = Ngrid ** np.arange(0, D)
    
for ii in range(0, N):

    ## Get all indices....
    for i in range(0, D):
        if i in d_indx:
            indx = int(ii / Ngrid_to_i[i]) % Ngrid
    
            ## Construct the D-D point....
            x[i] = GRID[indx]
        else:
            x[i] = fixed[i]
    pts[ii][1:D+1] = x
    
pts[:, 0] = f(M, pts[:, 1:D+1]).flatten()


# Remove points with high f(X)
if less != np.nan | less != None: 
    pts = np.asarray([pt for pt in pts if pt[0] <= less])
    name += "_less2"
    
pointsToVTK(name, np.ascontiguousarray(pts[:, d_indx[0]+1]), np.ascontiguousarray(pts[:, d_indx[1]+1]), np.ascontiguousarray(pts[:, d_indx[2]+1]), data = {"fx" : np.ascontiguousarray(pts[:, 0].flatten())} )








##############################################################################
##
##   for CSRBF
##
##############################################################################

n_rbf = 1000; D = 4
lb = -5; ub = 5
a = 10; f = Sphere

X = np.random.uniform(low=lb, high=ub, size=(n_rbf,D))
params = setup_csrbf(X, a, f)

name = f.__name__+"_CSRBF"

Ngrid=30
fixed = np.asarray([np.nan, np.nan, np.nan, 0.0])#-2.903534


name += "_"+str(D)+"D_"+str(Ngrid)+"ptsGrid_"+str(list(fixed))
name += "_"+str(n_rbf)+"ptsRBF"


d_indx = np.asarray(np.where(np.isnan(fixed))).reshape((-1,))
d_grid = len(d_indx)

## Grid for evaluating the cost function or (CS)RBF....
GRID = np.linspace(xmin, xmax, Ngrid)

## Rows....
N = Ngrid**d_grid

## Output....
pts = np.zeros((N, D+1))

## Parameter of the cost function....
x = np.zeros(D)

## Powers of Ngrid.... 
Ngrid_to_i = Ngrid ** np.arange(0, d_grid)


## Loop through all grid coordinates....
for ii in range(0, N):

    ## Get all indices....
    for i in range(0, D):
        if i in d_indx:
            indx = int(ii / Ngrid_to_i[i]) % Ngrid
    
            ## Construct the D-D point....
            x[i] = GRID[indx]
        else:
            x[i] = fixed[i]
    ## Evaluate the cost function or (CS)RBF....
    #y = StyblinskiTang(x)
            
    #pts[ii][0] = y
    pts[ii][1:D+1] = x

#pts[:, 0] = StyblinskiTang(pts[:, 1:D+1]).flatten()
pts[:, 0] = CSRBF_3_1(pts[:, 1:D+1], params).flatten()



pointsToVTK(name, np.ascontiguousarray(pts[:, d_indx[0]+1]), 
            np.ascontiguousarray(pts[:, d_indx[1]+1]), 
            np.ascontiguousarray(pts[:, d_indx[2]+1]), 
            data = {"fx" : np.ascontiguousarray(pts[:, 0].flatten())} )




##############################################################################



    


