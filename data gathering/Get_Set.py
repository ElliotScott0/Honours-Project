class Get_set:
    def __init__(self, pre_seizure_data: list, seizure_data: list, pre_during_data: list, pre_seizure_calculations: list, seizure_calculations: list, entire_seizure_calculations: list):
        self._pre_seizure_data = pre_seizure_data
        self._seizure_data = seizure_data
        self._pre_during_data = pre_during_data
        self._pre_seizure_calculations = pre_seizure_calculations
        self._seizure_calculations = seizure_calculations
        self._entire_seizure_calculations = entire_seizure_calculations
    
    # creates a property to return seizure data
    @property   
    def seizure_data(self):
        return self._seizure_data
    
    # sets the property for seizure data
    @seizure_data.setter
    def seizure_data(self, value: list):
        self._seizure_data = value[:]
    
    # creates a property to return pre seizure data
    @property    
    def pre_seizure_data(self):
        return self._pre_seizure_data 

    # sets the property for pre seizure data
    @pre_seizure_data.setter
    def set_pre_seizure_data(self, value: list):      
        self._pre_seizure_data = value[:]


    @property   
    def pre_during_data(self):
        return self._pre_during_data
    
    # sets the property for seizure data
    @pre_during_data.setter
    def pre_during_data(self, value: list):
        self._pre_during_data = value[:]
    


    # creates a property to return seizure calculations
    @property
    def seizure_calculations(self):
        return self._seizure_calculations
    
    # sets the property for seizure calculations
    @seizure_calculations.setter
    def seizure_calculations(self, value: list):
        self._seizure_data = value[:]
    

    

    # creates a property to return pre seizure calculations
    
    @property
    def pre_seizure_calculations(self):
        return self._pre_seizure_calculations 

    # sets the property for pre seizure calculations
    @pre_seizure_calculations.setter
    def pre_seizure_calculations(self, value: list):
        self._pre_seizure_data = value[:]

        
    
    @property
    def entire_seizure_calculations(self):
        return self._entire_seizure_calculations
    
    # sets the property for seizure calculations
    
    @entire_seizure_calculations.setter
    def entire_seizure_calculations(self, value: list):
       
        self._entire_seizure_calculations = value[:]
    
