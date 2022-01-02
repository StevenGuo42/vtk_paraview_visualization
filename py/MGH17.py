import numpy as np

def MGH17(beta, RESP_VARS):

    d = 5

    N = len(RESP_VARS)
    
    s = 0.0
        
    for i in range(0, N):
        y0 = RESP_VARS[i][0]
        x = RESP_VARS[i][1]

        y = beta[0] + beta[1]*np.exp(-beta[3]*x) + beta[2]*np.exp(-beta[4]*x)

        s += ((y0 - y)**2)/N

    s = np.sqrt(s)

    return(s)

    

## See https://www.itl.nist.gov/div898/strd/nls/data/LINKS/s-mgh17.shtml
def MGH17_rmse(N = 5000, fixed = np.asarray([np.nan, np.nan, np.nan, 1.2867534640E-02, 2.2122699662E-02]), ERR_MAX = 50):
    D = 5
    '''
    Parameter	    Certified          Certified
                     Estimate	    Std. Dev. of Est.
                                    
     beta(1)		3.7541005211E-01		2.0723153551E-03
     beta(2)		1.9358469127E+00		2.2031669222E-01
     beta(3)		-1.4646871366E+00	2.2175707739E-01
     beta(4)		1.2867534640E-02		4.4861358114E-04
     beta(5)		2.2122699662E-02		8.9471996575E-04
     '''
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
    
    ## Bounds....
    LB = np.array([-10, -10, -10, -10, -10])
    UB = np.array([10, 10, 10, 10, 10])
    
    ##LB = LB + 5
    ##UB = UB - 5
    
    

    
    ## Parameters....
    BETA = np.zeros((N, D))
    
    ## RMSE....
    RMSE = np.zeros(N)
    
        
    ## Evaluate the function....
    beta = np.zeros(D)
    
    ipt = 0             ## Index to points....
    attempts = 0        ## To check evaluation procedure....
    while (ipt < N):
        attempts += 1
        for d in range(0, D):
            if fixed[d] == np.nan:
                beta[d] = np.random.rand(1) * (UB[d] - LB[d]) + LB[d]
            else:
                beta[d] = fixed[d]
    
        temp = MGH17(beta, RESP_VARS)
        if (temp <= ERR_MAX):
            BETA[ipt, :] = beta
            RMSE[ipt] = temp
            ipt += 1
    return BETA, RMSE