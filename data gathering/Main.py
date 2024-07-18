from collect_data import *
from Process_data import *
from SVC_Train import *
from SSA_Train import *
from DataFrame import *
from Train_Random_Forest import *
from SVC_Train_standard import *




class MainClass:
    
    print("gathering data and creating dataframe")
    EEG_data, over = Collect_data.get_data()
    processed_data =Process_data.main(EEG_data)
    frequency = processed_data[8]
    magnitude = processed_data[9]
    df = DataFrame.main(processed_data,over)   

    print()
    print("SVM results")
    SVC_Train.main(df)

    print()
    print("Random Forrest results")
    Train_Random_Forest.main(df)

    print()
    print("SVM results using standerdization ")
    SVC_Train_standard.main(df)

    print()
    print("processing SSA and creating dataframe")
    ssa_data = SSA_Train.main(EEG_data ,1, 40)
    processed_data =Process_data.main(ssa_data)
    processed_data[8] = frequency
    processed_data[9] = magnitude
    df = DataFrame.main(processed_data,over)   

    print()
    print("SVM results using SSA ") 
    SVC_Train.main(df)

    print()
    print("Random Forrest results using SSA ")
    Train_Random_Forest.main(df)

    print()
    print("SVM results using SSA and standerdization ")
    SVC_Train_standard.main(df)