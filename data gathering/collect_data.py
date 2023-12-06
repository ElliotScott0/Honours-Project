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

#import pandas
#import matplotlib.pyplot as plt
seizureFile = pathlib.Path("G:/eeg_data/edf/train/")

seizure_type = 'absz'

channel_name = 'FP2-F4'

all_file_path = list(seizureFile.rglob("*.edf"))
all_file_path_csv =list(seizureFile.rglob("*.csv"))

secondSample = 250
fourSecondSample = secondSample * 4

seizure_numbers = 0


#reads seizure from file by setting channel, start time, for length of seizure
                    
#print(len(abcent_file_path))
class Collect_data:
    def get_data():
        pre_seizure = []
        seizure = []
        for y in range(len(all_file_path)):
            with open(str(all_file_path_csv[y])) as file_obj: 
        
            # Create reader object by passing the file  
            # object to reader method 
                data = csv.reader(file_obj) 

                #reads row of csv
                for row in data:
                    # if finds correct seizure and channel
                    if(seizure_type in row and channel_name in row):
                        
                        #sets channel
                        channel = 18
                        
                        #gets the start time in seconds so multiplies by sample rate of 250 to get start point in file
                        start_time = math.ceil(float(row[1]))*secondSample
                        #stop_time = math.floor(float(row[2]))*secondSample
                    
                        file_path_str = str(all_file_path[y])
                        file = pyedflib.EdfReader(file_path_str)

                        #seizure_numbers += 1
                        #begin pre seizure, if 4 seconds before is before start of file then start from 0
                        pre_seizure_value = file.readSignal(channel, start_time - fourSecondSample, fourSecondSample, True)
                        
                        if(pre_seizure_value[0] != -163 or pre_seizure_value[0] != 163):                      
                            pre_seizure.append(pre_seizure_value)
                        seizure.append(file.readSignal(channel, start_time , fourSecondSample, True))
                        
                    
                        

                            
                        
                        file.close()
                        
                        # prints data gathered and puts in graph
                        
                            
                        #y_axis = pre_seizure
                        #x_axis = np.linspace(0, 543 , len(pre_seizure))
                        #plt.plot(x_axis,y_axis)
                        
                        #plt.xlabel("seconds")
                        #plt.show()

                        
                            
                        
                
            file_obj.close()
        #print(seizure_numbers)
        Get_set.pre_seizure_data = pre_seizure
        Get_set.seizure_data = seizure
        return data

    
    
    

