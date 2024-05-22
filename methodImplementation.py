# importing packages
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cmath
from scipy import integrate
import time
import ito_diffusions
import scipy
import sklearn
import pip
import metrics
import sklearn.metrics as metrics
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import datetime
import sympy as sy
from numpy import linalg

# ito_diffusions, brownian_1d_diffusions - dX(t) = mdt + sdW(t), m,s - cons
T = 1
SCHEME_STEPS = int(1e4)
sim = ito_diffusions.BM(x0=0, T=T, scheme_steps=SCHEME_STEPS, vol=1)
x = sim.simulate()
plt.plot(x['spot'])

# generating basis functions
# 1. Haar system


def haar_wavelet(k, l):
    # k;l;

    def haar_wavelet2(t):
        x = t*(2**k)-l
        # y = 0
        if x < 0:
            y = 0
        elif x < 1/2:
            y = 1
        elif x < 1:
            y = -1
        else:
            y = 0
        return (2**(k/2))*y
    return haar_wavelet2


def generate_kl(basis_length):
    if basis_length > 1:
        k_list = []
        l_list = []
        idx = 2
        k = 0
        while idx <= basis_length:
            l = 0
            while idx <= basis_length and l <= (2**k-1):
                k_list.append(k)
                l_list.append(l)
                l = l+1
                idx = idx+1
            k = k+1
        return [k_list, l_list]

# 2. Fourier (trigonometric) system


def trig_real(n):
    # n;
    def trig_real2(t):
        if n % 2 == 0:
            return math.sqrt(2)*math.cos(2*math.pi*n*t)
        else:
            return math.sqrt(2)*math.sin(2*math.pi*n*t)
    return trig_real2


def trig_complex(n):
    # n;
    def trig_complex2(t):
        return math.e**(1J*n*2*math.pi*t)
    return trig_complex2


def generate_basis(basis_length, basis_type):
    if basis_type == 'haar':
        b0 = lambda t: 1
        b = [b0]
        if basis_length > 1:
            kl = generate_kl(basis_length)
            for i in range(len(kl[1])):
                b.append(haar_wavelet(kl[0][i], kl[1][i]))
        return b
    elif basis_type == 'trig_real':
        b0 = lambda t: 1
        b = [b0]
        if basis_length > 1:
            n = 1
            for j in range(1, basis_length):
                b.append(trig_real(n))
                n += 1
        return b
    elif basis_type == 'trig_complex':
        b0 = lambda t: 1
        b = [b0]
        if basis_length > 1:
            n = 1
            for j in range(1, basis_length):
                b.append(trig_complex(n))
                b.append(trig_complex(-n))
                n += 1
        return b


# examples

# example 1
b6_haar = generate_basis(6, 'haar')

x = np.linspace(0, 1, 100)

y1 = [b6_haar[0](k) for k in x]
y2 = [b6_haar[1](k) for k in x]
y3 = [b6_haar[2](k) for k in x]
y4 = [b6_haar[3](k) for k in x]
y5 = [b6_haar[4](k) for k in x]
y6 = [b6_haar[5](k) for k in x]

figure, axis = plt.subplots(2, 3)
axis[0, 0].plot(x, y1)
axis[0, 1].plot(x, y2)
axis[0, 2].plot(x, y3)
axis[1, 0].plot(x, y4)
axis[1, 1].plot(x, y5)
axis[1, 2].plot(x, y6)
plt.show()

# example 2
b6_trig_real = generate_basis(6, 'trig_real')

x = np.linspace(0, 1, 100)

y1 = [b6_trig_real[0](k) for k in x]
y2 = [b6_trig_real[1](k) for k in x]
y3 = [b6_trig_real[2](k) for k in x]
y4 = [b6_trig_real[3](k) for k in x]
y5 = [b6_trig_real[4](k) for k in x]
y6 = [b6_trig_real[5](k) for k in x]

figure, axis = plt.subplots(2, 3)
axis[0, 0].plot(x, y1)
axis[0, 1].plot(x, y2)
axis[0, 2].plot(x, y3)
axis[1, 0].plot(x, y4)
axis[1, 1].plot(x, y5)
axis[1, 2].plot(x, y6)
plt.show()

# example 3
b6_trig_complex = generate_basis(6, 'trig_complex')

x = np.linspace(0, 1, 100)

y1 = [b6_trig_complex[0](k) for k in x]
y2 = [b6_trig_complex[1](k) for k in x]
y3 = [b6_trig_complex[2](k) for k in x]
y4 = [b6_trig_complex[3](k) for k in x]
y5 = [b6_trig_complex[4](k) for k in x]
y6 = [b6_trig_complex[5](k) for k in x]

figure, axis = plt.subplots(2, 3)
axis[0, 0].plot(x, y1)
axis[0, 1].plot(x, y2)
axis[0, 2].plot(x, y3)
axis[1, 0].plot(x, y4)
axis[1, 1].plot(x, y5)
axis[1, 2].plot(x, y6)
plt.show()


# examples of estimated functions

def sig1(t):
    return (t**3)+1


def sig2(t):
    if t <= 1/3:
        return 1
    elif t <= 2/3:
        return 4
    elif t <= 1:
        return 9
    else:
        return 0


def sig3(t):
    return np.sin(10*t)


def sig4(t):
    return np.arcsin(t)


def sig5(t):
    return np.sqrt(t)


def sig6(t):
    if t <= 1/5:
        return 1
    elif t <= 2/5:
        return 9
    elif t <= 3/5:
        return 49
    elif t <= 4/5:
        return 36
    elif t <= 1:
        return 16
    else:
        return 0


def sig7(t):
    return (t**2)


def sig8(t):
    return (t**4)


# trapezoidal rule

def trapezoidal_rule(f, u, a, b, n):  # n - number of dividing points
    s = 0
    k = (b-a)/(n-1)
    for i in range(1, n):
        s += (f(a+(i-1)*k)+f(a+i*k))*(u(a+i*k)-u(a+(i-1)*k))/2
    return s


# Chebyshev nodes

def chebyshev_nodes(a, b, n):
    x = np.empty(n, dtype=float)
    for i in range(1, n+1):
        x[i-1] = 1/2*(a+b)+(1/2*(b-a)*np.cos(((2*i-1)*np.pi)/(2*n)))
    return x


# the proposed volatility estimator

gamma = 1


def volatility_estimation(data, basis, m):
    q = pd.DataFrame(0, index=range(1), columns=range(len(data)))
    for i in range(0, len(data)):
        sum = 0
        for k in range(1, i+1):
            sum += (data[k] - data[(k - 1)]) ** 2
        q[i] = sum
    b = generate_basis(2 * m + 1, basis)  # 'trig_complex'
    S = np.array([])
    for z in range(len(b)):
        s = np.array([])
        for j in range(len(q)):
            trap_rule = 0
            k = 1 / (len(q.columns) - 1)
            for i in range(1, len(q.columns)):
                trap_rule += (q.iloc[j][i] - q.iloc[j][i - 1]) * ((b[z](i * k)/(data[i]**(2*gamma))) + (b[z]((i - 1) * k) / (data[i-1]**(2*gamma)))) / 2
            s_i = trap_rule
            s = np.append(s, s_i)
        S = np.append(S, np.mean(s))

    def sigma(y):
        sum = 0
        for j in range(len(b)):
            sum += S[j]*b[j](abs(y-1))
        return sum
    return sigma


# the competing method estimation


def volatility_estimation_cm(data, m):
    t = np.linspace(0, 1, len(data.columns))
    N = len(t) - 1
    K = math.ceil(N/2)
    M = m

    def trig_fourier_volatility_est_proc(p):
        dp = np.diff(p)
        fourier_coef_est = np.zeros(4 * K + 1, dtype=np.complex_)
        for k in range(-2 * K, 2 * K + 1):
            sucet = 0
            for l in range(0, N):
                pom = cmath.exp(-2 * math.pi * 1J * k * t[l])
                sucet += pom * dp[l]
            fourier_coef_est[k + 2 * K] = sucet
        bohr_conv_est = np.zeros(2 * M + 1, dtype=np.complex_)
        for k in range(-M, M + 1):
            sucet = 0
            for s in range(-K, K + 1):
                sucet += fourier_coef_est[s + 2 * K] * fourier_coef_est[k - s + 2 * K]
            bohr_conv_est[k + M] = sucet / (2 * K + 1)

        def estimator(t2):
            est = 0
            for k in range(-M, M + 1):
                est += (1 - (np.abs(k) / M)) * bohr_conv_est[k + M] * cmath.exp(2 * math.pi * 1J * k * t2)
            return est
        return estimator

    def sigma(y):
        avg = np.array([])
        for j in range(len(data)):
            avg = np.append(avg, trig_fourier_volatility_est_proc(csv.iloc[j])(y))
        return np.mean(avg)

    return sigma

# convergence

a_M_array = []
a_M2_array = []
for k in range(0, 1000):
    data = files[i][k:(k+1)]
    t = np.linspace(0, 1, len(data.columns))
    sigma = volatility_estimation(data, 'trig_complex', M_array[i])
    a_M = linalg.norm(sigma(t) - sig8(t), ord=np.inf)
    a_M_array.append(a_M)
    a_M2 = np.sqrt(integrate.simpson(sigma(t) - sig8(t), t))
    a_M2_array.append(a_M2)

epsilon = 0.2
P_M = sum(l > epsilon for l in a_M_array)/1000
P_M2 = sum(l > epsilon for l in a_M2_array)/1000

# results comparison

def compare(data, m):
    csv = pd.read_csv(data)
    t = np.linspace(0, 1, len(csv.columns))
    start_time1 = time.time()
    sigma = volatility_estimation(data, 'trig_complex', m)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1
    print("Time complexity of the volatility_estimation function: ", elapsed_time1)
    start_time2 = time.time()
    sigma_cm = [volatility_estimation_cm(data, m)(i).real for i in t]
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    print("Time complexity of the volatility_estimation_cm function: ", elapsed_time2)
    print("Time difference: ", elapsed_time1 - elapsed_time2)
    print("MSE: ", metrics.mean_squared_error(sig1(t), sigma(t).real))  # sig1(t)
    print("MSE_IT: ", metrics.mean_squared_error(sig1(t), sigma_cm))
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(sig1(t), sigma(t).real)))
    print("RMSE_IT: ", np.sqrt(metrics.mean_squared_error(sig1(t), sigma_cm)))
    print("MAE: ", metrics.mean_absolute_error(sig1(t), sigma(t).real))
    print("MAE_IT: ", metrics.mean_absolute_error(sig1(t), sigma_cm))
    print("MdAE: ", metrics.median_absolute_error(sig1(t), sigma(t).real))
    print("MdAE_IT: ", metrics.median_absolute_error(sig1(t), sigma_cm))
    print("Max Error: ", metrics.max_error(sig1(t), sigma(t).real))
    print("Max Error IT: ", metrics.max_error(sig1(t), sigma_cm))
    return None

# volatility calculation

def parkinson(H, L):
    return np.sqrt((1/(4*np.log(2)))*(np.log(H/L)**2))


def garman_klass(O, H, L, C):
    return (1/2*(np.log(H) - np.log(L))**2 - (2*np.log(2) - 1)*(np.log(C)-np.log(O))**2)*100


def rogers_satchell(O, H, L, C):
    return np.sqrt(np.log(H/C)*np.log(H/O)-np.log(L/C)*np.log(L/O))


volatility = pd.DataFrame(columns=['Date', 'V'])
volatility['Date'] = data['Date']
for i in range(0, len(volatility)):
    volatility['V'][i] = garman_klass_w_ln(data.loc[i][1], data.loc[i][2], data.loc[i][3], data.loc[i][4])

X = sm.add_constant(X)
regressor_OLS = sm.OLS(endog=Y, exog=X.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
regressor_OLS.summary()