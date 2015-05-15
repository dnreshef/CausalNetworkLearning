import numpy as np
import matplotlib.pyplot as plt

prefix = "/Users/georgedu/Dropbox/Dave and George Shared/results/"
gpss_prefix = "/Users/georgedu/Dropbox/Dave and George Shared/gpss_results/"
metrics = ["var", "MI", "nll", "MIC", "varnorm", "MInorm", "nllnorm", "MICnorm"]
with open("data/truths.txt") as f:
    truths = [line.rstrip() for line in f]

def plot_confidence(datafile):
    data = np.loadtxt(open(datafile,"rb"),delimiter=",",skiprows=1)
    indices = data[:,[0]].ravel()
    for i in xrange(8):
        X = data[:,[2*i+1]].ravel()
        Y = data[:,[2*i+2]].ravel()
        suffix = metrics[i]
        confidence = [(max(Y[i]/X[i], X[i]/Y[i]),\
            truths[int(indices[i])-1] == "x" if X[i] < Y[i] else truths[i] == "y") for i in xrange(X.size)]
        a = np.argmax(confidence, axis=0)[0]
        print a, indices[a], confidence[a]
        confidence.sort(key=lambda x:x[0], reverse=True)
        correct = 0.0
        graph = []
        for x in xrange(len(confidence)):
            if (confidence[x][1]):
                correct += 1.0
            graph.append(correct / (x+1))
        plt.plot([(i+1.0) / len(graph) for i in xrange(len(graph))], graph)
        plt.xlabel("Decision Rate")
        plt.ylabel("Accuracy")
        plt.ylim([0,1])
        #plt.show()
        plt.savefig(prefix + suffix)
        plt.clf()

def compare(datafile1, datafile2):
    data1 = np.loadtxt(open(datafile1,"rb"),delimiter=",",skiprows=1)
    data2 = np.loadtxt(open(datafile2,"rb"),delimiter=",",skiprows=1)
    sorted1 = data1[data1[:,0].argsort()]
    sorted2 = data2[data2[:,0].argsort()]
    for i in xrange(8):
        X1 = sorted1[:,[2*i+1]].ravel()
        Y1 = sorted1[:,[2*i+2]].ravel()
        X2 = sorted2[:,[2*i+1]].ravel()
        Y2 = sorted2[:,[2*i+2]].ravel()
        same_results = (((Y1 / X1) - np.ones(Y1.size)) * ((Y2 / X2) - np.ones(Y1.size))) > 0
        print "For metric %s, data matched %d / %d of the time" % (metrics[i], same_results.sum(), same_results.size)

plot_confidence(prefix + "scores_test_new.csv")
#compare(prefix + "scores_test.csv", gpss_prefix + "scores_test.csv")