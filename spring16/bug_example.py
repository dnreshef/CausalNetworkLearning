import numpy as np
from matplotlib import pyplot as plt
import GPy

# Make 2D data, sample size 10
x = np.random.rand(10, 1)
y = x.copy()

kernel = GPy.kern.RBF(1)
model = GPy.models.GPRegression(x, y, kernel)
model.optimize()

# Plot variance

plot_x = np.linspace(0, 1, 200)
mu, s2 = model.predict(plot_x.reshape((200, 1)))
plt.plot(plot_x, s2.flatten())
plt.show()
'''
model.plot()
plt.show()
'''
