import numpy as np
import matplotlib.pyplot as plt

prefix = "/Users/georgedu/Dropbox/Dave and George Shared/results/"
suffixes = ["var", "MI", "nll", "MIC", "varnorm", "MInorm", "nllnorm", "MICnorm"]

def plot_confidence(indices, X, Y, truths, suffix):
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

data = np.loadtxt(open(prefix + "scores_test.csv","rb"),delimiter=",",skiprows=1)
with open("data/truths.txt") as f:
    truths = [line.rstrip() for line in f]
print truths
for i in xrange(8):
    plot_confidence(data[:,[0]].ravel(), data[:,[2*i+1]].ravel(), data[:,[2*i+2]].ravel(), truths, suffixes[i])
