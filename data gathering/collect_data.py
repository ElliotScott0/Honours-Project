from glob import glob
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import pandas
import csv


#import pandas
#import matplotlib.pyplot as plt
seizureFile = pathlib.Path("G:/eeg_data/edf/train/")

all_file_path = list(seizureFile.rglob("*.edf"))
all_file_path_csv =list(seizureFile.rglob("*.csv"))

secondSample = 250
fourSecondSample = secondSample *4


print(all_file_path_csv[0])



#abcent_file_path =[str(i) for i in all_file_path if 's012' in str(i).split('\\')[7]]
#print(len(abcent_file_path))
for y in range(len(all_file_path)):
    with open(str(all_file_path_csv[y])) as file_obj: 
      
    # Create reader object by passing the file  
    # object to reader method 
        data = csv.reader(file_obj) 

        #reads row of csv
        for row in data:
            # if finds correct seizure and channel
            if('absz' in row and 'FP2-F4' in row):
                print(y)
                print(str(all_file_path_csv[y]))
                #sets channel
                channel = 18
                
                #gets the start time in seconds so multiplies by sample rate of 240 to get start point in file
                start_time = int(round(float(row[1])*secondSample)) 
                stop_time = int(round(float(row[2])*secondSample)) 

                file_path_str = str(all_file_path[y])
                file = pyedflib.EdfReader(file_path_str)

                #reads seizure from file by setting channel, start time, for length of seizure
                seizure = file.readSignal(channel, start_time , stop_time - start_time, True)

                #begin pre seizure, if 4 seconds before is before start of file then start from 0
               
                pre_seizure = file.readSignal(channel, start_time - fourSecondSample, fourSecondSample, True)
                    
                
                file.close()
                
                # prints data gathered and puts in graph
                
                    
                y_axis = pre_seizure
                x_axis = np.linspace(0, 4 , len(pre_seizure))
                plt.plot(x_axis,y_axis)
                
                plt.xlabel("seconds")
                plt.show()

        
                    
                

    file_obj.close()


 