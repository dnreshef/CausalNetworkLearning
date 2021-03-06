Meeting Notes
=============

### 2/26/2015
- plot both directions & figure out why GP gets stuff wrong
- normalize scoring
- try decreasing theta0 and see if it takes longer
- future: generate plots like the one in email
- future: graph search

### 3/2/2015
- make sure that GP is correct. find some data from tutorial and recreate GP
- plot residuals as sanity check
- try transforming data to 0 mean 1 variance

### 3/9/2015
- transition to using sheffield GPy
- length scale is median euclidean distance
- also try .1 quantile and .9 quantile
- score GP on variance on normalized data, and MI
- output 6 column table. {var, MI, log likelihood} x {no change, normalized}. save plots
- for MI, use sqrt(n)/2 x sqrt(n)/2 bins

### 3/13/2015
- run optimize on sets with <= x points where x is max computationally feasible
- find bottleneck
- try to plot accuracy vs number of decisions ranked
- investigate why error margins differ for un-normalized data. make sure to optimize
- look at wrong examples and explain why they are wrong

### 3/20/2015
- Pick best bandwidth between 10, 50, 90 for both directions independently
- If MI most promising, try to estimate it better
- Use David's better measure of dependence by calling java exe

### 3/27/2015
- Change lengthscale to use squared values of euclidean distance
- Increase threshold to 5000 data points
- Start using TIC
- Set var parameter for kern as var of y data, stop running on normalized(?)

### 4/21/2015
- Downsample, get optimized kernel, and use optimized kernel on full dataset (check against original approach)
- Remove outliers
- Run optimize on 20 different starting conditions between 10th and 90th percentile indices in distances array
