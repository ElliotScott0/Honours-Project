import numpy as np
from pyts.decomposition import SingularSpectrumAnalysis
import matplotlib.pyplot as plt
M = 660
N = 2000
T = 1/10
stdnoise = 1

class SSA_Train():
   

    def train(data):
        t = np.arange(1, N + 1)
        X = np.sin(2 * np.pi * t / T)
        noise = stdnoise * np.random.randn(N)
        X = X + noise
        X = X - np.mean(X)
        X = X / np.std(X, ddof=1)

        plt.figure(1)
        plt.title('Time series X')
        plt.plot(t, X, 'b-')
        plt.show()



    def main(raw_data, results):
        SSA_Train.train(raw_data)