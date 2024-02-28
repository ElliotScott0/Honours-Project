from Get_Set import Get_set
from glob import glob
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import pandas
import csv
import math
from Process_data import epoch_time

#import pandas
#import matplotlib.pyplot as plt
seizureFile = pathlib.Path("G:/eeg_data/edf/train/")


seizure_type = 'absz'
#seizure_type = 'gnsz'
montage = [19]
channel_names = ['F4-C4']

all_file_path = list(seizureFile.rglob("*.edf"))
all_file_path_csv =list(seizureFile.rglob("*.csv"))

secondSample = 250
fourSecondSample = secondSample * 4
pre_sample = secondSample * 4
during_sample = secondSample * 4
total_sample = pre_sample + during_sample



#reads seizure from file by setting channel, start time, for length of seizure
                    
#print(len(abcent_file_path))
class Collect_data:
    def get_data():
        
        patient_count = 0
        old_file_path = 'a'
        pre_seizure = []
        seizure = []
        pre_during_seizure = []
        
        
        over = []
        for i in range(len(montage)):
            for y in range(len(all_file_path)):
                with open(str(all_file_path_csv[y])) as file_obj: 
        
                # Create reader object by passing the file  
                # object to reader method 
                    data = csv.reader(file_obj) 

                    #reads row of csv
                    for row in data:
                        # if finds correct seizure and channel
                        if(seizure_type in row and channel_names[i] in row):
                            #print("s")
                            if (old_file_path != all_file_path_csv[y]):
                                patient_count +=1
                                old_file_path = all_file_path_csv[y]
                             #sets channel
                        
                        
                            #gets the start time in seconds so multiplies by sample rate of 250 to get start point in file
                            start_time = math.ceil(float(row[1]))*secondSample
                            stop_time = math.floor(float(row[2]))*secondSample
                    
                            file_path_str = str(all_file_path[y])
                            file = pyedflib.EdfReader(file_path_str)

                        
                            #begin pre seizure, if 4 seconds before is before start of file then start from 0
                            #pre_seizure_value = file.readSignal(channel, start_time - fourSecondSample, fourSecondSample, True)
                        
                        #if(pre_seizure_value[0] != -163):                      
                            #pre_seizure.append(pre_seizure_value)
                        #seizure.append(file.readSignal(channel, start_time , fourSecondSample, True))
                        
                        
                        
                            pre_during_value = file.readSignal(montage[i], start_time - pre_sample, total_sample, True)
                            #print(pre_during_value , "here")
                            try:
                                if(pre_during_value[0] != -163 and pre_during_value[1] != -163):                      
                                    pre_during_seizure.append(pre_during_value)
                              
                                    #y_axis = pre_during_value
                                    #x_axis = np.linspace(0, 40 , len(pre_during_seizure[0]))
                                    #plt.plot(x_axis,y_axis)
                                    #plt.title(len(pre_during_seizure))
                                    #plt.xlabel("seconds")
                                    #plt.show()

                        
                                    if(start_time + pre_sample  > stop_time):
                                        time_off = (start_time + pre_sample - stop_time)/250
                                        over.append(time_off*(1/epoch_time)
                                

                                    else:
                                        over.append(0)   
                            except Exception as e:
                                # Handle other exceptions
                                print("An error occurred:", e)
                        
                            file.close()
                        
                       

                        
                            
                        
               
            file_obj.close()
        print(pre_during_seizure[0]) 
        with open('output.txt', 'w') as file:
    
            for element in pre_during_seizure[0]:
                file.write(str(element) + '\n')
        #print(len(pre_during_seizure) , "seizure events found for" , seizure_type)
        print(patient_count/len(channel_names) , "patients found for" , seizure_type)
        #data = [pre_during_seizure, over]
        return pre_during_seizure, over

    
    
    

