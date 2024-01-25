from Collect_data import *
from Process_data import *
from SVC_Train import *
from SSA_Train import *





class MainClass:
    #print("SCV results")
    EEG_data, over = Collect_data.get_data()
    #processed_data =Process_data.main(EEG_data)
       
    #SVC_Train.main(processed_data,over)

    print("SSA results")
    ssa_data = SSA_Train.main(EEG_data ,1, 40)

    processed_data =Process_data.main(ssa_data)
       
    SVC_Train.main(processed_data,over)