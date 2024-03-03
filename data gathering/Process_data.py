from collect_data import total_sample
import numpy as np
from Get_Set import Get_set
from scipy.stats import kurtosis, skew
from scipy.stats import entropy
import matplotlib.pyplot as plt

epoch_time = 0.2
epoch_length = int(250*epoch_time)  # change to 250/epoch time if epoch time is > 1 and do same further down when changing plotting
epoch_seconds = total_sample/250
num_epochs = 0


pre_calculations = []
seizure_calculations = []




class Process_data:


    # turns the eeg data into epochs of desired legnth
    def epoch_transformer(data):
        num_epochs = int(len(data)/epoch_length)
        
        # Create an array to store the epochs
        epochs = np.zeros((num_epochs, epoch_length))

        # Extract epochs from the EEG data
        for i in range(num_epochs):
            start_index = i * epoch_length
            end_index = start_index + epoch_length
            epochs[i] = data[start_index:end_index]
            
        return epochs

    #function to calculate rms
    def calculate_rms(data):
        return np.sqrt(np.mean(data**2))
    
    #function to calculate variance
    def calculate_var(data):
        return np.var(data)
    
    #function to calculate STD
    def calculate_std(data):
        return np.std(data)
    
    #function to calculate MAD
    def calculate_MAD(data):
        return np.mean(np.abs(data - np.mean(data)))

    # Function to calculate Log Energy
    def calculate_log_energy(data):
        return np.sum(np.log(1 + data**2))
    
    #function to calculate kurtosis
    def calculate_kurtosis(data):
        return kurtosis(data)
    
    #function to calculate skewness
    def calculate_skew(data):
        return skew(data)
    

   

    # Function to calculate Normalized Entropy
    def calculate_normalized_entropy(data):
        probabilities = np.histogram(data, bins='auto', density=True)[0]
        # Calculate entropy
        ent = entropy(probabilities, base=2)
        # Normalize entropy
        normalized_entropy = ent / np.log(len(probabilities))
        return normalized_entropy
    
    def calculate_fft_mag(data):
        
       
        
        signal_without_dc = data - np.mean(data)
        
        Epoch_fft = np.fft. rfft(signal_without_dc)


            
        return np.max(np.abs(Epoch_fft[:63]))

    def calculate_fft_fre(data):
        
        
        signal_without_dc = data - np.mean(data)
        
        Epoch_fft = np.fft.fft(signal_without_dc)
          
        return np.argmax(np.abs(Epoch_fft[:63]))

  

    #plots graphs with percentiles of calculated data
    def plot_percentiles(data, calculation, title):
        percentile25 = []
        percentile50 = []
        percentile75 = []
        for i in range(len(data)):
            percentile25.append(np.percentile(data[i], 25))
            percentile50.append(np.percentile(data[i], 50))
            percentile75.append(np.percentile(data[i], 75))
        #Plotting

        # change to / when epoch length changes to 1 or over 
        half_second_interval = np.arange(0, len(percentile25) * epoch_time, epoch_time)

        plt.plot(half_second_interval, percentile25, marker='o', label='25')
        plt.plot(half_second_interval, percentile50, marker='s', label='50')
        plt.plot(half_second_interval, percentile75, marker='^', label='75')
        #Plot vertical lines for percentiles
        plt.legend(loc='upper right', fontsize='small')
        plt.axvline(x=len(percentile25) * epoch_time  * 0.5 - epoch_time, color='red', linestyle='--', label='Midpoint')
        plt.xlabel(f"Epoch value, {epoch_time} second intervals")
        plt.ylabel("value")
        plt.title(calculation + title)

        

        #Show the plot
        plt.show()

    # calls calculation functions on every epoch in certain order then puts values into arrays. order is the time in which the epoch is placed within the eeg data
    def get_sorted_epochs(epochs, label):
        data = []
        rms_values = []
        kurtosis_values = []
        skewness_values = []
        mad_values = []
        normalized_entropy_values = []
        log_energy_values = []
        std_dev_values = []
        variance_values = []
        
        fft_max_frequency = []
        fft_max_magnitude = []
        

        for i in range(int(epoch_seconds/epoch_time)):
            partial_rms_values = []
            partial_variance_values = []
            partial_std_dev_values = []
            partial_log_energy_values = []
            partial_normalized_entropy_values = []
            partial_mad_values = []
            partial_kurtosis_values = []
            partial_skewness_values = []
            partial_fft_max_frequency = []
            partial_fft_max_magnitude = []

            for y in range(0, len(epochs), int(epoch_seconds/epoch_time)):
                
                try:
                    partial_rms_values.append(Process_data.calculate_rms(epochs[y+i]))
                    partial_variance_values.append(Process_data.calculate_var(epochs[y+i]))
                    partial_std_dev_values.append(Process_data.calculate_std(epochs[y+i]))
                    partial_log_energy_values.append(Process_data.calculate_log_energy(epochs[y+i]))
                    partial_normalized_entropy_values.append(Process_data.calculate_normalized_entropy(epochs[y+i]))
                    partial_mad_values.append(Process_data.calculate_MAD(epochs[y+i]))  # Mean Absolute Deviation
                    partial_kurtosis_values.append(Process_data.calculate_kurtosis(epochs[y+i]))
                    partial_skewness_values.append(Process_data.calculate_skew(epochs[y+i]))
                    partial_fft_max_frequency.append(Process_data.calculate_fft_fre(epochs[y+i]))
                    partial_fft_max_magnitude.append(Process_data.calculate_fft_mag(epochs[y+i]))
                   
                except IndexError:
                    
                    print(f"Skipping index {y + i}, exceeds length of epochs")

                
            
            rms_values.append(partial_rms_values)
            variance_values.append(partial_variance_values)
            std_dev_values.append(partial_std_dev_values)
            log_energy_values.append(partial_log_energy_values)
            normalized_entropy_values.append(partial_normalized_entropy_values)
            mad_values.append(partial_mad_values)
            kurtosis_values.append(partial_kurtosis_values)
            skewness_values.append(partial_skewness_values)

            fft_max_frequency.append(partial_fft_max_frequency)
            fft_max_magnitude.append(partial_fft_max_magnitude)

        #print(len(rms_values[0]))
        #Process_data.plot_percentiles(rms_values, "rms_value ", label)
        #Process_data.plot_percentiles(variance_values, "variance_value ", label)
        #Process_data.plot_percentiles(std_dev_values, "std_dev_value ", label)
        #Process_data.plot_percentiles(log_energy_values, "log_energy_value ", label)
        #Process_data.plot_percentiles(normalized_entropy_values, "normalized_entropy_values ", label)
        #Process_data.plot_percentiles(mad_values, "mad_value ", label)
        #Process_data.plot_percentiles(kurtosis_values, "kurtosis_value ", label)
        #Process_data.plot_percentiles(skewness_values, "skewness_value ", label)

        #Process_data.plot_percentiles(fft_max_frequency, "fft_max_frequency ", label)
        #Process_data.plot_percentiles(fft_max_magnitude, "fft_max_magnitude ", label)

             
        data = [rms_values, variance_values, std_dev_values, log_energy_values, normalized_entropy_values, mad_values, kurtosis_values, skewness_values, fft_max_frequency, fft_max_magnitude]
            
          
        
        return data
    
    # sorts data into arrays of patients for machine learning
    def format_data(data):  
        
        sorted_data = []
        for i in range(len(data[0])):
            partial_sorted_data = []
            for j in range(len(data)):
                partial_sorted_data.append(data[j][i])
            sorted_data.append(partial_sorted_data)
        return sorted_data
    
    #runs functions        
    def main(data): 

        sorted_seizure_calculations = []
   
        merged_entire_seizure = [element for row in data for element in row]
       
      
        entire_seizure_epochs = Process_data.epoch_transformer(merged_entire_seizure)

    
        entire_seizure_calculations = Process_data.get_sorted_epochs(entire_seizure_epochs, "pre + during seizure")


        for i in range(len(entire_seizure_calculations)):
            sorted_seizure_calculations.append(Process_data.format_data(entire_seizure_calculations[i]))

        return sorted_seizure_calculations

       

