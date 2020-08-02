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
t_step = 0.05  # this is the smallest time interval
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
y_of_nT_total_list = []
func_g_result = func_g(t)
for n in range(200):
    y_of_nT_total = 0
    for idx in range(1000):
        #idx = 2
        y_iT_of_nT = t_step * func_f(idx * t_step) * \
            func_g_result[int(t[n]/t_step - idx * t_step)]
        y_of_nT_total = y_of_nT_total + y_iT_of_nT
    y_of_nT_total_list.append(y_of_nT_total)

plt.figure()
plt.plot(t, y_of_nT_total_list)
plt.grid()

# %% using buit-in function to solve convolution
vector1 = func_f(t)
vector2 = func_g(t)
func_after_convolution = np.convolve(vector1, vector2, "same")
plt.plot(t, func_after_convolution)

# %% using scipy
func_after_convolution2 = signal.convolve(vector1, vector2, "same")
plt.plot(t, func_after_convolution2)

# %% simulate matlab conv, find document in good_example_convolution.pdf and try to write code for covolution
n = np.arange(-3, 8, 1)
x = 0.55**(n+3)
h = [1 for i in range(11)]
y = np.convolve(x, h)
plt.figure()
plt.plot(n, x, "ro")
plt.plot(n, h, "ko")
plt.plot(y, "bo")

# %% my own code for convolution

t = np.arange(-3, 8 ,1)
f_function = 0.55 ** (t+3)
plt.stem(t, f_function)



g_function = np.array([1 for i in range(11)])
g_function_reversed = np.array(list(reversed(g_function)))
g_reversed_t = np.array(list(reversed(t*(-1))))
# convert g_function[n] to g_function[-n]


plt.stem(t, g_function,"r")
plt.stem(g_reversed_t,g_function_reversed,"b")







# %% second thought
# 1. g(t) reverse its t vector. eg. g(t) = [1,2,3,4,5],
# while t = [0, 0.1, 0.2, 0.3, 0.4]. reverse its t vector means:
# -t = [-0.4,-0.3,-0.2,-0.1,0], for n - t becomes a new vector
# find the overlap of the new vector and the t vector of f(t)，
# then sumup the area
