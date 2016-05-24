#
# Analyze the data from the simulation
#
# Output from simulation file format:
#    PID, t submitted, t started, t completed, \
#    total t spent in queues, total t executing, \
#    total t waiting for I/O
#
# Analysis consists of:
#    Average turnaround (T_t)
#    Average wait in queues (W_t)
#    Average response (R_t)
#    Average throughput (Th)
#
import re
import os
import sys
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import andrews_curves

# Make them look prettier
plt.style.use('ggplot')
#sns.set(style="ticks")
sns.set_style("whitegrid")

# For reproducibility
random.seed(0)
np.random.seed(0)

# Evaluate the metrics on the desired files and create a DataFrame so that we
# can easily plot the data
def processFiles(directory, files):
    results = pd.DataFrame()

    for i, [fn, cores] in enumerate(files):
        df = pd.read_csv(os.path.join(directory,fn))

        turnaround = df['Completed'] - df['Submitted']
        wait = df['Queues']
        response = df['Started'] - df['Submitted']
        avgTurnaround = turnaround.mean()
        avgWait = wait.mean()
        avgResponse = response.mean()
        throughput = df['Completed'].count() / df['Completed'].max()

        results = results.append(pd.DataFrame([[cores, avgTurnaround, avgWait,
            avgResponse, throughput]], index=[i],
            columns=['Cores','AvgTurnaround','AvgWait','AvgResponse','Throughput']))

    return results

if __name__ == "__main__":
    # Plot all the data
    dirs = sys.argv[1:]

    if not dirs:
        print("Pass in result directory names as arguments.")

    for d in dirs:
        files = os.listdir(d)

        # Match the different tests we're doing
        re_fcfs = re.compile(".*_fcfs_cpu(.*)\.csv")
        fcfs_files = []
        for f in files:
            m = re_fcfs.match(f)

            if m and len(m.groups()):
                # filename, # of cores
                fcfs_files.append((f, int(m.groups()[0])))

        # FCFS with different numbers of cores
        fcfs = processFiles(d, fcfs_files)

        # Plots
        def combPlot(x, y, data):
            #sns.violinplot(x=x, y=y, data=data, inner=None)
            #sns.swarmplot(x=x, y=y, data=data, color="w", alpha=.5)
            sns.swarmplot(x=x, y=y, data=data)

        fig = plt.figure()
        fig.suptitle("FCFS - "+d)

        ax1 = fig.add_subplot(2,2,1)
        combPlot(x="Cores", y="AvgTurnaround", data=fcfs)
        ax1.set_title("Avg Turnaround")

        ax2 = fig.add_subplot(2,2,2)
        combPlot(x="Cores", y="AvgWait", data=fcfs)
        ax2.set_title("Avg Wait")

        ax3 = fig.add_subplot(2,2,3)
        combPlot(x="Cores", y="AvgResponse", data=fcfs)
        ax3.set_title("Avg Response")

        ax4 = fig.add_subplot(2,2,4)
        combPlot(x="Cores", y="Throughput", data=fcfs)
        ax4.set_title("Throughput")

        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Show all plots at the end
    plt.show()
