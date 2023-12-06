import numpy as np
from Get_Set import Get_set
from scipy.stats import kurtosis, skew
from scipy.stats import entropy
import matplotlib.pyplot as plt

epoch_time = 0.5
epoch_length = int(250*epoch_time)  # change to 250/epoch time if epoch time is > 1
epoch_seconds = 4
num_epochs = 0


pre_calculations = []
seizure_calculations = []


class Process_data:

    
    def epoch_transformer(data):
        num_epochs = int(len(data)/epoch_length)
        #print(num_epochs)
        # Create an array to store the epochs
        epochs = np.zeros((num_epochs, epoch_length))

        # Extract epochs from the EEG data
        for i in range(num_epochs):
            start_index = i * epoch_length
            end_index = start_index + epoch_length
            epochs[i] = data[start_index:end_index]
            
        return epochs


    def calculate_rms(data):
        return np.sqrt(np.mean(data**2))

    # Function to calculate Log Energy
    def calculate_log_energy(data):
        return np.sum(np.log(1 + data**2))

    # Function to calculate Normalized Entropy
    def calculate_normalized_entropy(data):
        probabilities = np.histogram(data, bins='auto', density=True)[0]
        # Calculate entropy
        ent = entropy(probabilities, base=2)
        # Normalize entropy
        normalized_entropy = ent / np.log(len(probabilities))
        return normalized_entropy


    def calculate_fft(data):
        fft = []
        for i in range(len(data)):
            EEG_fft = np.fft.fft(data[i])

            freq = np.arange(1, 1001) / 1000 * 125
            #print(np.argmax(np.abs(EEG_fft[:1000])))
           # plt.figure()
           # plt.plot(freq, np.abs(EEG_fft[:1000]))
           # plt.xlabel('Frequency (Hz)')
           # plt.ylabel('Magnitude')
           # plt.title('FFT of EEG Data')
           # plt.show()
            fft.append(EEG_fft) 
        return fft

    
    def plot_percentiles(data, calculation, title):
        percentile25 = []
        percentile50 = []
        percentile75 = []
        for i in range(len(data)):
            percentile25.append(np.percentile(data[i], 25))
            percentile50.append(np.percentile(data[i], 50))
            percentile75.append(np.percentile(data[i], 75))
        #Plotting
    
        plt.plot(percentile25, marker='o')
        plt.plot(percentile50, marker='s')
        plt.plot(percentile75, marker='^')
        #Plot vertical lines for percentiles
        
        plt.xlabel("epoch value")
        plt.ylabel("value")
        plt.title(calculation + title)

        

        #Show the plot
        plt.show()


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
        

        for i in range(int(epoch_seconds/epoch_time)):
            partial_rms_values = []
            partial_variance_values = []
            partial_std_dev_values = []
            partial_log_energy_values = []
            partial_normalized_entropy_values = []
            partial_mad_values = []
            partial_kurtosis_values = []
            partial_skewness_values = []

            for y in range(0, len(epochs), int(epoch_seconds/epoch_time)):
                
                try:
                    partial_rms_values.append(Process_data.calculate_rms(epochs[y+i]))
                    partial_variance_values.append(np.var(epochs[y+i]))
                    partial_std_dev_values.append(np.std(epochs[y+i]))
                    partial_log_energy_values.append(Process_data.calculate_log_energy(epochs[y+i]))
                    partial_normalized_entropy_values.append(Process_data.calculate_normalized_entropy(epochs[y+i]))
                    partial_mad_values.append(np.mean(np.abs(epochs[y+i] - np.mean(epochs[y+i]))))  # Mean Absolute Deviation
                    partial_kurtosis_values.append(kurtosis(epochs[y+i]))
                    partial_skewness_values.append(skew(epochs[y+i]))

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


        
        #Process_data.plot_percentiles(rms_values, "rms_value ", label)
        #Process_data.plot_percentiles(variance_values, "variance_value ", label)
        #Process_data.plot_percentiles(std_dev_values, "std_dev_value ", label)
        #Process_data.plot_percentiles(log_energy_values, "log_energy_value ", label)
        #Process_data.plot_percentiles(normalized_entropy_values, "normalized_entropy_values ", label)
        #Process_data.plot_percentiles(mad_values, "mad_value ", label)
        #Process_data.plot_percentiles(kurtosis_values, "kurtosis_value ", label)
        #Process_data.plot_percentiles(skewness_values, "skewness_value ", label)

             
        data = [rms_values, variance_values, std_dev_values, log_energy_values, normalized_entropy_values, mad_values, kurtosis_values]
            
            
        
        return data
        
            
    def main(): 

        pre_data = Get_set.pre_seizure_data
        seizure_data = Get_set.seizure_data

        

        merged_pre_seizure = [element for row in pre_data for element in row]
        merged_seizure = [element for row in seizure_data for element in row]


        pre_epochs = Process_data.epoch_transformer(merged_pre_seizure)
        seizure_epochs = Process_data.epoch_transformer(merged_seizure)

        pre_calculations = Process_data.get_sorted_epochs(pre_epochs, "pre seizure")
        seizure_calculations = Process_data.get_sorted_epochs(seizure_epochs, "seizure")

        pre_fft = Process_data.calculate_fft(pre_data)
        seizure_fft = Process_data.calculate_fft(seizure_data)

        pre_calculations.append(pre_fft)
        seizure_calculations.append(seizure_fft)

        

        Get_set.pre_seizure_calculations = pre_calculations
        Get_set.seizure_calculations = seizure_calculations

       

