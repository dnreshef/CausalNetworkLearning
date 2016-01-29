import matplotlib.pyplot as plt
import numpy as np
a = np.array([0.5129212,0.5009723,0.41527545,0.45671479,0.40607978,0.19216456,0.21578227,0.26939123,0.32495763,0.47882888])
b = np.array([0.3254711,0.44107469,-0.08433706,0.16211347,-0.36015472,-0.34862665,-0.42213365,0.63236687,0.60396961,-0.42902998])
a = sorted(a)
plt.plot(a, b, ls='None', marker='+')
plt.fill_between(a, b + 0.01*np.ones(10), b - 0.01 * np.ones(10))
plt.show()
