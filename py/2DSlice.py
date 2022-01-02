########################################################
##
##  Test for high-dimensional visualization
##  COSC 3307 WI 2021
##
##  NIST cost functions: MGH17, ENSO, Rat43
##  Other cost functions: Styblinski-Tang, Rastrigin, Sphere,
##      Rosenbrock, Qing, Michalewicz, Salomon, Deb3
##
##  See (e.g. for MGH17):
##      https://www.itl.nist.gov/div898/strd/nls/data/mgh17.shtml
##      https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/MGH17.dat
##      https://www.itl.nist.gov/div898/strd/nls/data/LINKS/v-mgh17.shtml
##      https://www.itl.nist.gov/div898/strd/nls/data/LINKS/s-mgh17.shtml
##
##  For other NIST functions, see:
##      https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
##
##  Dr. M. Wachowiak
##  4/28/21 -- 5/1/21
##
##
########################################################
import numpy as np

## For TSNE....
from sklearn.manifold import TSNE

## For interpolation....
from scipy.interpolate import griddata

## For plotting....
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

## For Plotly graphics....
import plotly.graph_objects as go



############################################
##
##  Cost functions....
##
############################################

##########################################
##  Styblinski-Tang
##########################################
def StybTang(x):
    d = len(x)

    s = sum(x**4 - 16*x**2 + 5*x) / 2.0

    return(s)


##########################################
##  Qing
##########################################
def Qing(x):
    d = len(x)

    s = sum((x**2 - (np.arange(0, d)+1)) ** 2)

    return(s)

##########################################
##  Rosenbrock
##########################################
def Rosenbrock(x):
    d = len(x)

    s = 0.0

    for i in range(0, d-1):
        t0 = (x[i+1] - (x[i])**2)**2
        t1 = (1.0 - x[i])**2
        s += (100.0*t0 + t1)

    return(s)


##########################################
##  Sphere
##########################################
def Sphere(x):
    ## d = len(x)

    s = sum(x**2)

    return(s)


##########################################
##  Michalewicz
##########################################
def Michalewicz(x):

    d = len(x)
    m = 10
    
    s = (np.sin((np.linspace(1, d, num = d) * (x**2)) / np.pi)) ** (2*m)
    s = s * np.sin(x)
    s = -sum(s)

    return(s)

    


##########################################
##  Rastrigin
##########################################
def Rastrigin(x):
    d = len(x)
    A = 10

    s = 0.0

    for i in range(0, d):
        s += ((x[i])**2 - A*np.cos(2.0 * np.pi * x[i]))

    s += (A * d)

    return(s)


##########################################
##  Salomon
##########################################
def Salomon(x):
    d = len(x)

    s0 = np.sqrt(sum(x ** 2))
    s = 1.0 - np.cos(2.0 * np.pi * s0) + (0.1 * s0)

    return(s)


##########################################
##  Deb3
##########################################
def Deb3(x):
    d = len(x)

    s = -1.0/d * sum((np.sin(5.0 * np.pi * ((x ** (3/4)) - 0.05))) ** 6)

    return(s)



##########################################
##  MGH17 (NIST)
##########################################
def MGH17(beta):

    d = len(beta)           ## 5

    N = len(RESP_VARS)      ## Global
    
    s = 0.0
        
    for i in range(0, N):
        y0 = RESP_VARS[i][0]
        x = RESP_VARS[i][1]

        y = beta[0] + beta[1]*np.exp(-beta[3]*x) + beta[2]*np.exp(-beta[4]*x)

        s += ((y0 - y)**2)/N

    s = np.sqrt(s)

    return(s)


##########################################
##  Rat43 (NIST)
##########################################
def Rat43(beta):


    d = len(beta)           ## 5

    N = len(RAT43_DATA)     ## Global
    
    s = 0.0
        
    for i in range(0, N):
        y0 = RAT43_DATA[i][0]
        x = RAT43_DATA[i][1]

        y = beta[0] / ((1.0 + np.exp(beta[1] - beta[2]*x))**(1.0/beta[3]))
        
        ##y = beta[0] + beta[1]*np.exp(-beta[3]*x) + beta[2]*np.exp(-beta[4]*x)

        s += ((y0 - y)**2)/N

    s = np.sqrt(s)

    return(s)


##########################################
##  ENSO (NIST)
##########################################
def ENSO(beta):
    
    d = len(beta)       ## 9

    N = len(ENSO_DATA)  ## Global
    
    TwoPI = 2.0 * np.pi
        
    s = 0.0

    for i in range(0, N):
        y0 = ENSO_DATA[i][0]
        x = ENSO_DATA[i][1]

        y = beta[0] + beta[1]*np.cos(TwoPI * x / 12.0) + beta[2]*np.sin(TwoPI * x / 12.0) + beta[4]*np.cos(TwoPI * x / beta[3]) + beta[5]*np.sin(TwoPI * x / beta[3]) + beta[7]*np.cos(TwoPI * x / beta[6]) + beta[8]*np.sin(TwoPI * x / beta[6])
        
        ##y = beta[0] + beta[1]*np.exp(-beta[3]*x) + beta[2]*np.exp(-beta[4]*x)

        s += ((y0 - y)**2)/N

    s = np.sqrt(s)

    return(s)


###################################################################
##
##  Get a slice from the volume....
##
###################################################################
def getVolumeSlice(plane, islice, X, Xinterp, gx, gy, gz):

    Ngrid = len(gx)
    
    ## This can be made more general....
    if (plane == 0):
        vslice = gx[islice]
    elif (plane == 1):
        vslice = gy[islice]
    else:
        vslice = gz[islice]

    indx = np.where(X[:, plane] == vslice)

    ## This needs to be cleaned up....
    Z = Xinterp[indx]

    ## Reshape into an image....
    Z = np.reshape(Z, ((Ngrid, Ngrid)))

    return(Z)


#####################################################
##
##  Get a 2D slice directly from the points....
##
#####################################################
def getPointCloudSlice(plane, BETA, rmse, Ngrid):
    ## Assume that 'plane' is 1 x 2.
    B = BETA[:, plane]

    xmin, xmax = np.min(B[:,0]), np.max(B[:, 0])
    ymin, ymax = np.min(B[:,1]), np.max(B[:, 1])

    gx, gy = np.meshgrid(np.linspace(xmin, xmax, num = Ngrid),
                         np.linspace(ymin, ymax, num = Ngrid))

    Z = griddata(B, rmse, (gx, gy), method = 'linear')
                   
    return(Z)



#####################################################
##
##  Evaluate the grid directly....
##
#####################################################
def getFunctionSlice(f, v, Ngridx, Ngridy, LB, UB):
    
    ## First NaN is x, second NaN is y....
    indx = np.where(np.isnan(v))
    indx = indx[0]

    Z = np.zeros((Ngridy, Ngridx))

    gx = np.linspace(LB[indx[0]], UB[indx[0]], num = Ngridx)
    gy = np.linspace(LB[indx[1]], UB[indx[1]], num = Ngridy)

    x = np.copy(v)
    for j in range(0, Ngridy):
        for i in range(0, Ngridx):
            x[indx[0]] = gx[i]
            x[indx[1]] = gy[j]

            Z[i][j] = f(x)


    return(Z, gx, gy)



###################################################################
##
##  Begin program....
##
###################################################################

###################################################
##  MGH17 data
##  See https://www.itl.nist.gov/div898/strd/nls/data/LINKS/s-mgh17.shtml
###################################################
RESP_VARS = np.array([
      [8.440000E-01,    0.000000E+00],
      [9.080000E-01,    1.000000E+01],
      [9.320000E-01,    2.000000E+01],
      [9.360000E-01,    3.000000E+01],
      [9.250000E-01,    4.000000E+01],
      [9.080000E-01,    5.000000E+01],
      [8.810000E-01,    6.000000E+01],
      [8.500000E-01,    7.000000E+01],
      [8.180000E-01,    8.000000E+01],
      [7.840000E-01,    9.000000E+01],
      [7.510000E-01,    1.000000E+02],
      [7.180000E-01,    1.100000E+02],
      [6.850000E-01,    1.200000E+02],
      [6.580000E-01,    1.300000E+02],
      [6.280000E-01,    1.400000E+02],
      [6.030000E-01,    1.500000E+02],
      [5.800000E-01,    1.600000E+02],
      [5.580000E-01,    1.700000E+02],
      [5.380000E-01,    1.800000E+02],
      [5.220000E-01,    1.900000E+02],
      [5.060000E-01,    2.000000E+02],
      [4.900000E-01,    2.100000E+02],
      [4.780000E-01,    2.200000E+02],
      [4.670000E-01,    2.300000E+02],
      [4.570000E-01,    2.400000E+02],
      [4.480000E-01,    2.500000E+02],
      [4.380000E-01,    2.600000E+02],
      [4.310000E-01,    2.700000E+02],
      [4.240000E-01,    2.800000E+02],
      [4.200000E-01,    2.900000E+02],
      [4.140000E-01,    3.000000E+02],
      [4.110000E-01,    3.100000E+02],
      [4.060000E-01,    3.200000E+02]
      ])

###################################################
##  RAT43 data
###################################################
RAT43_DATA = np.array([[16.08E0,     1.0E0],
      [33.83E0,     2.0E0],
      [65.80E0,     3.0E0],
      [97.20E0,     4.0E0],
     [191.55E0,     5.0E0],
     [326.20E0,     6.0E0],
     [386.87E0,     7.0E0],
     [520.53E0,     8.0E0],
     [590.03E0,     9.0E0],
     [651.92E0,    10.0E0],
     [724.93E0,    11.0E0],
     [699.56E0,    12.0E0],
     [689.96E0,    13.0E0],
     [637.56E0,    14.0E0],
     [717.41E0,    15.0E0]
])


###################################################
##  ENSO data
###################################################
ENSO_DATA = np.array([[12.9,1],
[11.3,2],
[10.6,3],
[11.2,4],
[10.9,5],
[7.5,6],
[7.7,7],
[11.7,8],
[12.9,9],
[14.3,10],
[10.9,11],
[13.7,12],
[17.1,13],
[14,14],
[15.3,15],
[8.5,16],
[5.7,17],
[5.5,18],
[7.6,19],
[8.6,20],
[7.3,21],
[7.6,22],
[12.7,23],
[11,24],
[12.7,25],
[12.9,26],
[13,27],
[10.9,28],
[10.4,29],
[10.2,30],
[8,31],
[10.9,32],
[13.6,33],
[10.5,34],
[9.2,35],
[12.4,36],
[12.7,37],
[13.3,38],
[10.1,39],
[7.8,40],
[4.8,41],
[3,42],
[2.5,43],
[6.3,44],
[9.7,45],
[11.6,46],
[8.6,47],
[12.4,48],
[10.5,49],
[13.3,50],
[10.4,51],
[8.1,52],
[3.7,53],
[10.7,54],
[5.1,55],
[10.4,56],
[10.9,57],
[11.7,58],
[11.4,59],
[13.7,60],
[14.1,61],
[14,62],
[12.5,63],
[6.3,64],
[9.6,65],
[11.7,66],
[5,67],
[10.8,68],
[12.7,69],
[10.8,70],
[11.8,71],
[12.6,72],
[15.7,73],
[12.6,74],
[14.8,75],
[7.8,76],
[7.1,77],
[11.2,78],
[8.1,79],
[6.4,80],
[5.2,81],
[12,82],
[10.2,83],
[12.7,84],
[10.2,85],
[14.7,86],
[12.2,87],
[7.1,88],
[5.7,89],
[6.7,90],
[3.9,91],
[8.5,92],
[8.3,93],
[10.8,94],
[16.7,95],
[12.6,96],
[12.5,97],
[12.5,98],
[9.8,99],
[7.2,100],
[4.1,101],
[10.6,102],
[10.1,103],
[10.1,104],
[11.9,105],
[13.6,106],
[16.3,107],
[17.6,108],
[15.5,109],
[16,110],
[15.2,111],
[11.2,112],
[14.3,113],
[14.5,114],
[8.5,115],
[12,116],
[12.7,117],
[11.3,118],
[14.5,119],
[15.1,120],
[10.4,121],
[11.5,122],
[13.4,123],
[7.5,124],
[0.6,125],
[0.3,126],
[5.5,127],
[5,128],
[4.6,129],
[8.2,130],
[9.9,131],
[9.2,132],
[12.5,133],
[10.9,134],
[9.9,135],
[8.9,136],
[7.6,137],
[9.5,138],
[8.4,139],
[10.7,140],
[13.6,141],
[13.7,142],
[13.7,143],
[16.5,144],
[16.8,145],
[17.1,146],
[15.4,147],
[9.5,148],
[6.1,149],
[10.1,150],
[9.3,151],
[5.3,152],
[11.2,153],
[16.6,154],
[15.6,155],
[12,156],
[11.5,157],
[8.6,158],
[13.8,159],
[8.7,160],
[8.6,161],
[8.6,162],
[8.7,163],
[12.8,164],
[13.2,165],
[14,166],
[13.4,167],
[14.8,168]])


## Default bounds....
LB = np.array([-10, -10, -10, -10, -10, -10, -10])
UB = np.array([10, 10, 10, 10, 10, 10, 10])

  

################################################################
##
##  Certified values and bounds for the NIST functions....
##
################################################################

cert_MGH17 = np.array([3.7541005211E-01,1.9358469127E+00,
                 -1.4646871366E+00,1.2867534640E-02,
                 2.2122699662E-02])

cert_Rat43 = np.array([6.9964151270E+02,5.2771253025E+00,7.5962938329E-01,1.2792483859E+00])

cert_ENSO = np.array([1.0510749193E+01,3.0762128085E+00,5.3280138227E-01,4.4311088700E+01,-1.6231428586E+00,5.2554493756E-01,2.6887614440E+01,2.1232288488E-01,1.4966870418E+00])


LB_ENSO = np.array([0.,0.,0.,20.,-3.,-5.,5.,-1.,0.])
UB_ENSO = np.array([20.,10.,2.,80.,0.05,5,40.,2.,5.])

LB_MGH17 = np.array([0, 0, -100, 0, 0, 0])
UB_MGH17 = np.array([50, 150, -0.1, 10, 2])

LB_Rat43 = np.array([100, 0, 0, 0])
UB_Rat43 = np.array([900, 10, 2, 5])




##########################################################################
##
##  Sample graphics session....
##
##  NOTE:  The Z matrix (the 2D function evaluation values)
##  can be displayed as they are (Plotly will adjust the mapping)
##  or they can be transformed to assist visualizing subtle features.
##
##  E.g. if all the values are > 0, then np.log(Z) can be displayed, as
##  can np.sqrt(Z), or even Z**(1/4).
##  Alternately, Z can be scaled to [0, 1] and subsequently transformed.
##  E.g. Z1 = (Z - np.min(Z)) / (np.max(Z) - np.min(Z)).
##  Then, Z1 can be transformed; e.g. np.sqrt(Z1) or Z1 ** (1/4).
##
##########################################################################

###################################
##  Example 1 -- ENSO
###################################
v = np.copy(cert_ENSO)
v[[0, 3]] = np.nan      ## Grid is on dimensions 0 and 3....

LB = np.copy(LB_ENSO)
UB = np.copy(UB_ENSO)

## Number of points (pixels) on the grid....
Ngridx = 140
Ngridy = 140

## Compute the 2D slice....
Z, gx, gy = getFunctionSlice(ENSO, v, Ngridx, Ngridy, LB, UB)

## Plot with Plotly....
fig = go.Figure(data =
     go.Heatmap(x = gx, y = gy, z = Z))
  
fig.show()

## Plot the transformed output with Plotly....
## NOTE: RMSE > 0 (the data do not fit the model perfectly to obtain RMSE = 0).
fig = go.Figure(data =
     go.Heatmap(x = gx, y = gy, z = np.log(Z)))
  
fig.show()

###################################
##  Example 2 -- Michalewicz
###################################
## Search space ranges from 0 to Pi.
D = 10
LB = np.zeros(D)
UB = np.zeros(D) + np.pi

v = np.random.rand(D) * np.pi

v[[4, 8]] = np.nan      ## Grid is on dimensions 4 and 8....

## Number of points (pixels) on the grid....
Ngridx = 200
Ngridy = 200

## Compute the 2D slice....
Z, gx, gy = getFunctionSlice(Michalewicz, v, Ngridx, Ngridy, LB, UB)

## Plot with Plotly....
fig = go.Figure(data =
     go.Heatmap(x = gx, y = gy, z = Z))
  
fig.show()
   

###################################
##  Example 3 -- Deb3
###################################
## Search space ranges from 0 to 1.
D = 8
LB = np.zeros(D)
UB = np.zeros(D) + 1

v = np.random.rand(D)

v[[1, 7]] = np.nan      ## Grid is on dimensions 1 and 7....

## Number of points (pixels) on the grid....
Ngridx = 200
Ngridy = 200

## Compute the 2D slice....
Z, gx, gy = getFunctionSlice(Deb3, v, Ngridx, Ngridy, LB, UB)

## Plot with Plotly....
fig = go.Figure(data =
     go.Heatmap(x = gx, y = gy, z = Z))
  
fig.show()
   

###################################
##  Example 4 -- Rosenbrock
###################################
## Search space ranges from -5 to 5.
D = 8
LB = np.zeros(D) - 5
UB = np.zeros(D) + 5

v = np.zeros(D) + 1

v[[2, 6]] = np.nan      ## Grid is on dimensions 2 and 6....

## Number of points (pixels) on the grid....
Ngridx = 200
Ngridy = 200

## Compute the 2D slice....
Z, gx, gy = getFunctionSlice(Rosenbrock, v, Ngridx, Ngridy, LB, UB)

## Plot the transformed cost function with Plotly....
fig = go.Figure(data =
     go.Heatmap(x = gx, y = gy, z = np.log(Z)))
  
fig.show()
   
###################################
##  Example 5 -- Salomon
###################################
## Search space ranges from -100 to 100.
D = 12
LB = np.zeros(D) - 100
UB = np.zeros(D) + 100

v = np.random.rand(D) * 200 - 100

v[[3, 9]] = np.nan      ## Grid is on dimensions 3 and 9....

## Number of points (pixels) on the grid....
Ngridx = 200
Ngridy = 200

## Compute the 2D slice....
Z, gx, gy = getFunctionSlice(Salomon, v, Ngridx, Ngridy, LB, UB)

## Plot the transformed cost function with Plotly....
fig = go.Figure(data =
     go.Heatmap(x = gx, y = gy, z = np.log(Z)))
  
fig.show()
