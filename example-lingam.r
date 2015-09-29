library('pcalg')
mydata = read.csv('pair0053.csv')
estDAG = LINGAM(mydata)
show(estDAG)
