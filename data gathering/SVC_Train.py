from Process_data import epoch_time, epoch_seconds
from Get_Set import Get_set
from sklearn.metrics import f1_score, roc_curve, auc, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SVC_Train:
    def scatter(data, over):
        names = ['rms_value', 'variance_values', 'std_dev_values',  'log_energy_values', 'normalized_entropy_values', 'mad_values', 'kurtosis_values', 'skewness_values', 'fft_max_frequency', 'fft_max_magnitude']
        
        rms_value = data[0]
        
        #for i in range(len(data) - 1):
            #test = data[i+1]
        test = data[9]
        for k in range(len(rms_value)):
            
            tester = test[k]
            test_rms = rms_value[k]
            for j in range(len(rms_value[0])):
                
                
                    
                if(j <8):
                    plt.scatter(test_rms[j], tester[j], color='green', marker="+", label='pre')
                
                elif(j + over[k]> 15):
                    plt.scatter(test_rms[j], tester[j], color='green', marker="s", label='post')
                else:
                    plt.scatter(test_rms[j], tester[j], color='blue', marker=".", label='during')

                plt.annotate(j+1, (test_rms[j], tester[j]), textcoords="offset points", xytext=(0, 16), ha='center')
            plt.title(k)
            plt.xlabel("kms_value")
            plt.ylabel(names[9])
            #plt.legend(loc='upper left', fontsize='small')
            plt.show()


    
    def train_SVC(df):
        #scores = []
        #for i in range(20):
        X = df.drop('results', axis=1).copy()
        #print(X)
        y = df['results'].copy()
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        
        model = SVC(probability=True)

        model.fit(X_train, y_train)

        

        #model score
        model_score = model.score(X_test, y_test)
        
        # Calculate F1-score
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        #calculate sensitivity
        sensitivity = recall_score(y_test, y_pred)

        # Calculate AUC-ROC score
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        #calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        #calculate specificity
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp)

       
        
        
        #print("Principal Components:\n", pca.components_)
        print("F1-score:", f1)
        print("AUC-ROC Score:", roc_auc)
        print("Sensitivity (Recall): ",sensitivity)
        print("Specificity: ",specificity)
        print("confusion matrix: ", conf_matrix)
        print("Model Score :", model_score)
        #print(np.mean(scores))
        
    def train_PCA(df):

        

        pca = PCA()
        pca.fit_transform(df)

        eigenvalues = pca.explained_variance_

        # Plot the scree plot
        plt.plot(1, len(eigenvalues) + 1, eigenvalues, marker = "o")
        plt.xlabel('Principal Component Number')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues against Priciple Components')
        plt.show()


        
        
        
        




    def dataframe(data, over):
        rms_value = [element for row in data[0] for element in row]
        variance_values= [element for row in data[1] for element in row]
        std_dev_values = [element for row in data[2] for element in row]
        log_energy_values = [element for row in data[3] for element in row]
        normalized_entropy_values = [element for row in data[4] for element in row]
        mad_values = [element for row in data[5] for element in row]
        kurtosis_values = [element for row in data[6] for element in row]
        skewness_values = [element for row in data[7] for element in row]
        fft_max_frequency = [element for row in data[8] for element in row]
        fft_max_magnitude = [element for row in data[9] for element in row]

        default = []
        for _ in range(int((epoch_seconds/epoch_time)/2)):  # Loop 20 times, but only perform actions for the first 10 iterations
            default.append(0)

        for _ in range(int((epoch_seconds/epoch_time)/2),int(epoch_seconds/epoch_time)):  # Loop for the second half of 20 iterations
            default.append(1)

            
        new_array = []
        for i in range(len(over)):
            if(over[i]== 0):
                new_array.append(default)
            else:
                new = []
                for j in range(len(default)):
                    
                    if(j <int((epoch_seconds/epoch_time)/2)):
                        new.append(0)
                    elif(j + over[i]> len(default)-1):
                        #print(i , " ", j , " " , over[i])
                        new.append(0)
                    else:
                        new.append(1)
                new_array.append(new)

        results = [element for row in new_array for element in row]

        df = []
        i =0
        
       
        d = {'rms_value': rms_value, 'variance_values': variance_values, 'std_dev_values': std_dev_values,
            'normalized_entropy_values': normalized_entropy_values,'fft_max_frequency': fft_max_frequency,
            'fft_max_magnitude': fft_max_magnitude, 'results': results }
            
        df = pd.DataFrame(data=d)
        
        return df
    

    def main(data, over):
        #SVC_Train.scatter(data, over)
        data_frame = SVC_Train.dataframe(data, over)
        SVC_Train.train_SVC(data_frame)
        SVC_Train.train_PCA(data_frame)