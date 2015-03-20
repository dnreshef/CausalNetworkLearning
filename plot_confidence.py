import numpy as np
import matplotlib.pyplot as plt

def plot_confidence(X, Y, truths):
    confidence = [(max(Y[i]/X[i], X[i]/Y[i]),\
        truths[i] == "x" if X[i] < Y[i] else truths[i] == "y") for i in xrange(X.size)]
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
    plt.show()

data = np.loadtxt(open("results/scores_full.csv","rb"),delimiter=",",skiprows=1)
with open("data/truths.txt") as f:
    truths = [line.rstrip() for line in f]
for i in xrange(6):
    plot_confidence(data[:,[2*i+1]].ravel(), data[:,[2*i+2]].ravel(), truths)