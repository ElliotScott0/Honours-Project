from Process_data import epoch_time, epoch_seconds
from Get_Set import Get_set
from sklearn.metrics import f1_score, roc_curve, auc, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class DataFrame:
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
                
                
                    
                if(j <int((epoch_seconds/epoch_time)/2)):
                    plt.scatter(test_rms[j], tester[j], color='green', marker="+", label='pre')
                
                elif(j> epoch_seconds/epoch_time-1):
                    plt.scatter(test_rms[j], tester[j], color='green', marker="s", label='post')
                else:
                    plt.scatter(test_rms[j], tester[j], color='blue', marker=".", label='during')

                plt.annotate(j+1, (test_rms[j], tester[j]), textcoords="offset points", xytext=(0, 40), ha='center')
            plt.title(k)
            plt.xlabel("kms_value")
            plt.ylabel(names[9])
            #plt.legend(loc='upper left', fontsize='small')
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

        #results = DataFrame.annotations(over)  #use this if you want doctors annotations
        results = DataFrame.self_anotaion()   #use this for custom annotations
        

        #DataFrame.scatter_graph(data, results) # shows scatterfraph for plotted points and results

        df = []
      

        d = {'rms_value': rms_value, 'variance_values': variance_values, 'std_dev_values': std_dev_values,
            'normalized_entropy_values': normalized_entropy_values,'fft_max_frequency': fft_max_frequency,
            'fft_max_magnitude': fft_max_magnitude, 'results': results }
            
        df = pd.DataFrame(data=d)
        #print(df)
        return df
    


    def scatter_graph(data, results):
        counter = 0
        test = data[9]
        t_rms = data[0]
        for k in range(len(t_rms)):
            
            tester = test[k]
            test_rms = t_rms[k]
            for j in range(len(t_rms[0])):
                
                
                    
                if(results[counter] == 0):
                    plt.scatter(test_rms[j], tester[j], color='blue', marker="+", label='yes')
                
                elif(results[counter] == 1):
                    plt.scatter(test_rms[j], tester[j], color='blue', marker="s", label='yes')
                
               
                counter += 1

            plt.title(k +2)
            plt.xlabel("kms_value")
            plt.ylabel('fft_max_magnitude')
            #plt.legend(loc='upper left', fontsize='small')
            plt.show()




    def annotations(over):

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
        return results

    def self_anotaion():
        results = []
        file_path = 'g:\HONOURS PROJECT\Honours-Project\data gathering\self_anotaion.txt' 
        with open(file_path, 'r') as file:
   
            for line in file: 
                for char in line.strip():
                    results.append(int(char)) 
    
        return results
    
    
    def main(data, over):
        DataFrame.scatter_graph(data, over) # can test multiple different calculations to see which is best for training model
        df = DataFrame.dataframe(data, over)
    
        

        return df