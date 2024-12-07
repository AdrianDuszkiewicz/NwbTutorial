import pandas as pd
import numpy as np  # needs numpy 2.0 or earlier
import pynapple as nap  # for easier loading of NWB file
import matplotlib.pyplot as plt

def main():

    # Read the NWB file
    nwb_file_path = '/Users/doctordu/Downloads/sub-A1813_behavior+ecephys+ogen.nwb'  # Replace with your actual NWB file path
    data = nap.load_file(nwb_file_path)
    print(data)

    # Read opto stim file
    stim_file_path = '/Users/doctordu/Downloads/Opto_stim.csv'
    stim_timestamps = pd.read_csv(stim_file_path).to_numpy()



    # plot a PSTH of unit spikes on opto stim

    # load some unit spike times
    cell_number = 1
    spikes = data['units'][cell_number].times()

    # Plot PSTH on opto stim
    bin_size = 0.1  # Bin size in seconds
    window = (-2, 2)  # Time window around each stimulus

    aligned_spikes = []  # Align spikes to each stimulus
    for stim in stim_timestamps:
        aligned_spikes.extend(spikes - stim)

    aligned_spikes = np.array([s for s in aligned_spikes if window[0] <= s <= window[1]]) # Filter spikes to keep only those within the window
    bins = np.arange(window[0], window[1] + bin_size, bin_size) # Create histogram bins
    counts, edges = np.histogram(aligned_spikes, bins=bins)  # Compute histogram

    # Plot
    plt.bar(edges[:-1], counts, width=bin_size, align="edge", color="blue", edgecolor="black")
    plt.xlabel("Time (s) relative to stim")
    plt.ylabel("Spike count")
    plt.axvline(0, color="red", linestyle="--", label="Stimulus time")  # Highlight stimulus
    plt.legend()
    plt.show()


    # now plot Event Related Potential of LFP on opto stim

    lfp_chan_number = 50 # pick LFP channel
    lfp = data['LFP']
    sampling_rate = lfp.rate
    lfp_timestamps = lfp.times()
    lfp_one_chan = lfp[:, lfp_chan_number]

    # Parameters for ERP
    window = (-2, 2)  # Time window around each stimulus (in seconds)
    window_samples = int((window[1] - window[0]) * sampling_rate)  # Samples per window

    # Align LFP to each stimulus
    aligned_lfp = []
    for stim in stim_timestamps:
        # Find LFP indices within the window of the current stimulus
        start_idx = np.searchsorted(lfp_timestamps, stim + window[0])
        end_idx = np.searchsorted(lfp_timestamps, stim + window[1])
        if end_idx - start_idx == window_samples:
            aligned_lfp.append(lfp_one_chan[start_idx:end_idx])

    aligned_lfp = np.array(aligned_lfp)

    # Compute the average and standard error
    mean_lfp = aligned_lfp.mean(axis=0)
    std_error = aligned_lfp.std(axis=0) / np.sqrt(len(aligned_lfp))

    # Generate time vector relative to stimulus
    time_vector = np.linspace(window[0], window[1], mean_lfp.shape[0])

    # Plot the ERP
    plt.figure(figsize=(8, 5))
    plt.plot(time_vector, mean_lfp, label="Average LFP", color="blue")
    plt.fill_between(time_vector, mean_lfp - std_error, mean_lfp + std_error, color="blue", alpha=0.3, label="SEM")
    plt.axvline(0, color="red", linestyle="--", label="Stimulus")
    plt.xlabel("Time (s) relative to stimulus")
    plt.ylabel("LFP Signal")
    plt.title("Event-Related LFP")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()