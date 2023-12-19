from Collect_data import *
from Process_data import *
from SVC_Train import *
from SSA_Train import *





class MainClass:
    
    data = Collect_data.get_data()
    processed_data =Process_data.main(data[0])
       
    SVC_Train.main(processed_data, data[1])


    #SSA_Train.main(data[0],data[1])