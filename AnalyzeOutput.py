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

    for i, [fn, queues, cores] in enumerate(files):
        df = pd.read_csv(os.path.join(directory,fn))

        turnaround = df['Completed'] - df['Submitted']
        wait = df['Queues']
        response = df['Started'] - df['Submitted']
        avgTurnaround = turnaround.mean()
        avgWait = wait.mean()
        avgResponse = response.mean()
        throughput = df['Completed'].count() / df['Completed'].max()

        results = results.append(pd.DataFrame([[queues, cores,
            avgTurnaround, avgWait, avgResponse, throughput]], index=[i],
            columns=['Queues', 'Cores', 'AvgTurnaround', 'AvgWait',
                'AvgResponse', 'Throughput']))

    return results

if __name__ == "__main__":
    # Plot all the data
    dirs = sys.argv[1:]

    if not dirs:
        print("Pass in result directory names as arguments.")

    for d in dirs:
        files = os.listdir(d)

        # FCFS with different numbers of cores
        re_fcfs = re.compile(".*_fcfs(.*)_cpu(.*)\.csv")
        fcfs_files = []
        for f in files:
            m = re_fcfs.match(f)

            if m and len(m.groups()) == 2:
                queues = m.groups()[0] # FCFS
                coreCount = int(m.groups()[1])
                fcfs_files.append((f, queues, coreCount))

        fcfs = processFiles(d, fcfs_files)

        # RRRRFCFS
        re_rrrrfcfs = re.compile(".*_RR(.*)RR(.*)FCFS_cpu(.*)\.csv")
        rrrrfcfs_files = []
        for f in files:
            m = re_rrrrfcfs.match(f)

            if m and len(m.groups()) == 3:
                tq1 = m.groups()[0]
                tq2 = m.groups()[1]
                queues = tq1+","+tq2 # RR, RR, FCFS
                coreCount = int(m.groups()[2])
                rrrrfcfs_files.append((f, queues, coreCount))

        rrrrfcfs = processFiles(d, rrrrfcfs_files)

        # RRRRSPN
        re_rrrrspn = re.compile(".*_RR(.*)RR(.*)SPN_cpu(.*)\.csv")
        rrrrspn_files = []
        for f in files:
            m = re_rrrrspn.match(f)

            if m and len(m.groups()) == 3:
                tq1 = m.groups()[0]
                tq2 = m.groups()[1]
                queues = tq1+","+tq2 # RR, RR, SPN
                coreCount = int(m.groups()[2])
                rrrrspn_files.append((f, queues, coreCount))

        rrrrspn = processFiles(d, rrrrspn_files)

        # Plots
        def combPlot(x, y, data, hue=None):
            #sns.violinplot(x=x, y=y, data=data, hue=hue, inner=None)
            #sns.swarmplot(x=x, y=y, data=data, hue=hue, color="w", alpha=.5)
            sns.swarmplot(x=x, y=y, hue=hue, data=data)
            #sns.stripplot(x=x, y=y, hue=hue, data=data, jitter=False, alpha=0.5)

        #
        # Single FCFS Queue
        #
        fig = plt.figure()
        fig.suptitle("Single FCFS Queue - "+d)

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

        plt.subplots_adjust(wspace=0.3, hspace=0.4)

        #
        # Multiple FCFS Queues
        #
        fig = plt.figure()
        fig.suptitle("Multiple FCFS Queues - "+d)
        oneCore = fcfs.loc[lambda df: df.Cores == 1]

        ax1 = fig.add_subplot(2,2,1)
        combPlot(x="Queues", y="AvgTurnaround", data=oneCore)
        ax1.set_title("Avg Turnaround")

        ax2 = fig.add_subplot(2,2,2)
        combPlot(x="Queues", y="AvgWait", data=oneCore)
        ax2.set_title("Avg Wait")

        ax3 = fig.add_subplot(2,2,3)
        combPlot(x="Queues", y="AvgResponse", data=oneCore)
        ax3.set_title("Avg Response")

        ax4 = fig.add_subplot(2,2,4)
        combPlot(x="Queues", y="Throughput", data=oneCore)
        ax4.set_title("Throughput")

        plt.subplots_adjust(wspace=0.3, hspace=0.4)

        #
        # RR, RR, FCFS Queues
        #
        #for core in rrrrfcfs['Cores'].unique():
        if False:
            oneCore = rrrrfcfs.loc[lambda df: df.Cores == core]

            fig = plt.figure()
            fig.suptitle("RR RR FCFS Queues for "+str(core)+" cores - "+d)

            ax1 = fig.add_subplot(2,2,1)
            combPlot(x="Queues", y="AvgTurnaround", data=oneCore)
            ax1.set_title("Avg Turnaround")

            ax2 = fig.add_subplot(2,2,2)
            combPlot(x="Queues", y="AvgWait", data=oneCore)
            ax2.set_title("Avg Wait")

            ax3 = fig.add_subplot(2,2,3)
            combPlot(x="Queues", y="AvgResponse", data=oneCore)
            ax3.set_title("Avg Response")

            ax4 = fig.add_subplot(2,2,4)
            combPlot(x="Queues", y="Throughput", data=oneCore)
            ax4.set_title("Throughput")

            plt.subplots_adjust(wspace=0.3, hspace=0.4)

        fig = plt.figure()
        fig.suptitle("RR RR FCFS Queues - "+d)

        ax1 = fig.add_subplot(2,2,1)
        combPlot(hue="Queues", x="Cores", y="AvgTurnaround", data=rrrrfcfs)
        ax1.set_title("Avg Turnaround")

        ax2 = fig.add_subplot(2,2,2)
        combPlot(hue="Queues", x="Cores", y="AvgWait", data=rrrrfcfs)
        ax2.set_title("Avg Wait")

        ax3 = fig.add_subplot(2,2,3)
        combPlot(hue="Queues", x="Cores", y="AvgResponse", data=rrrrfcfs)
        ax3.set_title("Avg Response")

        ax4 = fig.add_subplot(2,2,4)
        combPlot(hue="Queues", x="Cores", y="Throughput", data=rrrrfcfs)
        ax4.set_title("Throughput")

        plt.subplots_adjust(wspace=0.3, hspace=0.4)


        #
        # RR, RR, SPN Queues
        #
        fig = plt.figure()
        fig.suptitle("RR RR SPN Queues - "+d)

        ax1 = fig.add_subplot(2,2,1)
        combPlot(hue="Queues", x="Cores", y="AvgTurnaround", data=rrrrspn)
        ax1.set_title("Avg Turnaround")

        ax2 = fig.add_subplot(2,2,2)
        combPlot(hue="Queues", x="Cores", y="AvgWait", data=rrrrspn)
        ax2.set_title("Avg Wait")

        ax3 = fig.add_subplot(2,2,3)
        combPlot(hue="Queues", x="Cores", y="AvgResponse", data=rrrrspn)
        ax3.set_title("Avg Response")

        ax4 = fig.add_subplot(2,2,4)
        combPlot(hue="Queues", x="Cores", y="Throughput", data=rrrrspn)
        ax4.set_title("Throughput")

        plt.subplots_adjust(wspace=0.3, hspace=0.4)
    # Show all plots at the end
    plt.show()
