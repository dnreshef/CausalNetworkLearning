library('pcalg')
data(gmG)
mydata = gmG8$x
#mydata = read.csv('pair0053.csv')
n <- nrow (mydata)
V <- colnames(mydata) # labels aka node names
## estimate CPDAG
pc.fit <- pc(suffStat = list(C = cor(mydata), n = n),
             indepTest = gaussCItest, ## indep.test: partial correlations
             alpha=0.01, labels = V, verbose = TRUE)
if (require(Rgraphviz)) {
    ## show estimated CPDAG
    par(mfrow=c(1,2))
    plot(pc.fit, main = "Estimated CPDAG")
    #plot(mydata, main = "True DAG")
}
