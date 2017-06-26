def getSmoothingWeights(dt=25, windowsize=500, halflife=5000):
    tau = (halflife/dt)/ np.log(2) # half-life = 5000ns and time between 2 consecutive time frame = 25ns
    alpha = 1 - np.exp(-1/tau)
    s = 0;
    wt = np.zeros((windowsize,1));
    for i in range (0,windowsize):
        wt[i] = (alpha * np.exp(-(windowsize-i)/tau))/0.96499303
        s = s + wt[i];
    return wt;
