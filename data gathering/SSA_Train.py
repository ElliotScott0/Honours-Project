import numpy as np
from pyts.decomposition import SingularSpectrumAnalysis
import matplotlib.pyplot as plt


class SSA_Train():
   

    def hankel_bet(rca, N, L, K):
        y = np.zeros(N)
        Lp = min(L, K)
        Kp = max(L, K)

        for k in range(Lp - 1):
            for m in range(k + 1):
                y[k] += (1 / (k + 1)) * rca[m, k - m]

        for k in range(Lp - 1, Kp):
            for m in range(Lp):
                y[k] += (1 / Lp) * rca[m, k - m]

        for k in range(Kp, N):
            for m in range(k - Kp + 1, N - Kp + 1):
                y[k] += (1 / (N - k)) * rca[m, k - m]

        return y

    def SSA_1D(x1, first, last):
        # Step1 : Build trayectory matrix
        N = len(x1)
        L = int(N / 40) if int(N / 40) < N else N
        K = N - L + 1
        X = np.zeros((L, K))
        for i in range(K):
            X[:, i] = x1[i:L + i]
        
            
        # Step 2: SVD
        S = X @ X.T
        autoval, U = np.linalg.eig(S)
        i = np.argsort(-autoval)  # Sort indices in descending order
        d = autoval[i]
        U = U[:, i]
        sev = np.sum(d)
        
        # Uncomment the following lines if you want to plot the normalized eigenvalues
        # plt.plot((d / sev) * 100)
        # plt.plot((d / sev) * 100, 'rx')
        # plt.title('Singular Spectrum')
        # plt.xlabel('Eigenvalue Number')
        # plt.ylabel('Eigenvalue (% Norm of trajectory matrix retained)')
        # plt.show()
        
        V = X.T @ U
        
        # Step 3: Grouping
        I = range(first-1, last)  # Python uses 0-based indexing
        Vt = V.T
        rca = U[:, I] @ Vt[I, :]
        
        # Step 4: Reconstruction

        RC2=SSA_Train.hankel_bet(rca,N,L,K)
        
        return RC2
    

    def main(raw_data, first, last):
        matrix = []
        rc2_array = []
        for i in range(len(raw_data)):
           

            rc2 = SSA_Train.SSA_1D(raw_data[i], first, last)
            rc2_array.append(rc2)

            #t = np.arange(0, len(rc2))
            #plt.subplot(3, 1, 2)
            #plt.plot(t, rc2, label='rca')
            #plt.title('Plot of rc2')
            #plt.legend()

           


            #plt.tight_layout()
            #plt.show()

            #print(rca ,rc1, rc3)

        return rc2_array    

    if __name__ == "__main__":
        L=5 # define the value according to your own data
        first=1 # define the value according to your own data
        last=1 # define the value according to your own data
        output = SSA_1D(yourdata, L, first, last)