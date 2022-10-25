import random
import numpy as np
import matplotlib.pyplot as plt

T = 1002
N_s = 500
p = 0.6
samples = {}
sample_deltas = {}

def initialize_samples(num):
    """Initializes the storage of each sample in a dictionary."""
    for i in range(1, num + 1):
        samples[f"sample {i}"] = [0]

def gen_steps(time, dict):
    """Generates the steps of one random walk."""
    for key in dict:
        for i in range(time):
            step = random.uniform(0, 1)
            if step > 1-p:
                dict[key].append(dict[key][i] + 1)
            else:
                dict[key].append(dict[key][i] - 1)

def get_data(time, dict):
    """Collects data for each sample before a given time."""
    time_t_data = []
    for key in dict:
            time_t_data.append(dict[key][time])
    return time_t_data

def calculate_mean(sample_data):
    """Calculates the mean at a given time."""
    return np.mean(sample_data)

def calculate_variance(sample_data):
    """Calculates the variance at a given time."""
    return np.var(sample_data)

def gen_means(time, dict):
    """Generates the mean value of x at each time step t."""
    means = []
    for i in range(1, time):
        mean_t = calculate_mean(get_data(i, dict))
        means.append(mean_t)
    return np.array(means)

def gen_vars(time, dict):
    """Generates the mean value of x at each time step t."""
    vars = []
    for i in range(1, time):
        var_t = calculate_variance(get_data(i, dict))
        vars.append(var_t)
    return np.array(vars)**(1/2)

def plot_data(data_1, data_2, data_3, data_4, data_5):
        """Genreates plots"""
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(data_1)), data_1, drawstyle='steps-mid', label="data_1")
        ax.plot(np.arange(len(data_2)), data_2, drawstyle='steps-mid', color = 'red', label="data_2")
        ax.plot(np.arange(len(data_3)), data_3, drawstyle='steps-mid', color = 'green', label="data_3")
        ax.plot(np.arange(len(data_4)), data_4, drawstyle='steps-mid', color = 'orange', label="mean")
        ax.plot(np.arange(len(data_5)), data_5, drawstyle='steps-mid', color = 'black', label="variance")

        ax.set_ylabel("position")
        ax.set_xlabel("time")

        plt.legend()
        plt.show()

def gen_indep_deltas(dict):
    """Generates the independent delta value for a given time, for each sample."""
    for key in dict:
        sample_deltas[key] = gen_delta_t(dict[key])

def gen_delta_t(data):
    """Generates the independent delta value for each time."""
    deltas = []
    for i in range(1, len(data)):
        delta_t = data[i] - data[i-1]
        deltas.append(delta_t)
    return deltas
        
def hypothesis_plots(time, sample_mean, sample_var):
    """Generates the plot of the same mean adn its hypothesis along with the sample variance and its hypothesis."""
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(sample_mean)), sample_mean, label="sample mean data")
    ax.plot(np.arange(len(sample_var)), sample_var, label="sample var data")
    ax.plot(np.arange(time), 0.2*np.arange(time), linestyle='dashed', label="hypothesis mean data")
    ax.plot(np.arange(time), (0.96*np.arange(time))**(0.5), linestyle='dashed', label="hypothesis var data")

    ax.set_ylabel("position")
    ax.set_xlabel("time")

    plt.legend()
    plt.show()

def gen_hist_data(time, dict):
    """Bins the histogram data for a given time."""
    position_vals = []
    for key in dict:
        position_vals.append(dict[key][time])
    return np.histogram(position_vals, 6)

def gen_gaussian(time, dict):
    """Generates the Gaussian distribution for the sample data at a given time."""
    mean = gen_means(T, dict)[time]
    variance = gen_vars(T, dict)[time]



    return N_s/(variance*(2*np.pi)**(0.5))*np.exp(-0.5*((np.arange(-2.5*variance+0.2*time, 2.5*variance+0.2*time)-mean)/variance)**2)

def plot_histograms(time, dict):
    """Plots a histogram and continuous Gaussian distribution for a given time and sample set."""
    counts, bins = gen_hist_data(time, dict)
    fig, ax = plt.subplots()
    ax.plot(np.arange(-2.5*gen_vars(T, dict)[time]+0.2*time, 2.5*gen_vars(T, dict)[time]+0.2*time), gen_gaussian(time, dict))
    ax.hist(bins[:-1], weights=counts, bins=bins)
    ax.set_ylabel("counts")
    ax.set_xlabel("position")
    plt.show()


if __name__ == '__main__':
    initialize_samples(N_s)
    gen_steps(T, samples)
    plot_data(samples["sample 1"], samples["sample 2"], samples["sample 3"], gen_means(T, samples), gen_vars(T, samples))
    hypothesis_plots(T, gen_means(T, samples), gen_vars(T, samples) )
    plot_histograms(time=10, dict=samples) 
    plot_histograms(time=500, dict=samples) 
    plot_histograms(time=1000, dict=samples) 
    

    