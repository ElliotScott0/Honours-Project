

class Get_set:
    def __init__(self, pre_seizure_data: list, seizure_data: list, pre_seizure_calculations: list, seizure_calculations: list):
        self._pre_seizure_data = pre_seizure_data
        self._seizure_data = seizure_data
        self._pre_seizure_calculations = pre_seizure_calculations
        self._seizure_calculations = seizure_calculations
    
  
    @property   
    def seizure_data(self):
        return self._seizure_data
    
    @seizure_data.setter
    def seizure_data(self, value: list):
        self._seizure_data = value[:]
    
    
    @property    
    def pre_seizure_data(self):
        return self._pre_seizure_data 


    @pre_seizure_data.setter
    def set_pre_seizure_data(self, value: list):
        
        self._pre_seizure_data = value[:]

    @property
    def seizure_calculations(self):
        return self._seizure_calculations
    
    @seizure_calculations.setter
    def seizure_calculations(self, value: list):
        self._seizure_data = value[:]
    


    @property
    def pre_seizure_calculations(self):
        return self._pre_seizure_calculations 

    @pre_seizure_calculations.setter
    def pre_seizure_calculations(self, value: list):
        self._pre_seizure_data = value[:]

        