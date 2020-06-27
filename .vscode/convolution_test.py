import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# %%

""" t_total = 100  # unit: second
number_of_interval = 300  # discretization: total number of intervals
index_of_interval = np.arange(0, 300, 1)
t_interval = t_total/number_of_interval
# the end is 299 intervals, 299 * 0.333 = 99.666
t = index_of_interval * t_interval """
# another method of contruction t
t0 = 0
t_end = 10
t_step = 0.05
t = np.arange(t0, t_end, t_step)
print(pd.DataFrame(t))


def func_f(x):
    # this is a step function, when x >=0, f(x) = constant
    constant = 1
    if type(x) not in [float, int]:
        result = [constant for idx in range(len(x))]
        result = np.array(result)
    else:
        result = constant
    return result


plt.plot(t, func_f(t))
plt.grid(True)


def func_g(x):
    # this is a square function
    result = signal.square(2*np.pi*x, duty=0.5)
    return result


plt.plot(t, func_g(t))
plt.grid(True)


n = 1
y_of_nT_total = 0
for idx in range(1000):
    #idx = 2
    y_iT_of_nT = t_step * func_f(idx * t_step) * func_g(t[n] - idx * t_step)
    y_of_nT_total = y_of_nT_total + y_iT_of_nT
