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
    outdir_prefix = "plots"
    outdir_cores = "cores"
    outdir_queues = "queues"
    outdir_single = "single"

    if not dirs:
        print("Pass in result directory names as arguments.")

    for d in dirs:
        files = os.listdir(d)
        outdir = outdir_prefix + "_" + d.strip().replace("/","")

        # Make sure output directory exists
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not os.path.exists(os.path.join(outdir, outdir_cores)):
            os.makedirs(os.path.join(outdir, outdir_cores))
        if not os.path.exists(os.path.join(outdir, outdir_queues)):
            os.makedirs(os.path.join(outdir, outdir_queues))
        if not os.path.exists(os.path.join(outdir, outdir_single)):
            os.makedirs(os.path.join(outdir, outdir_single))

        # FCFS with different numbers of cores
        re_fcfs = re.compile(".*_fcfs_cpu([0-9]*)\.csv")
        fcfs_files = []
        for f in files:
            m = re_fcfs.match(f)

            if m and len(m.groups()) == 1:
                queues = "FCFS" # FCFS
                coreCount = int(m.groups()[0])
                fcfs_files.append((f, queues, coreCount))

        fcfs = processFiles(d, fcfs_files)

        # HRRN with different numbers of cores
        re_hrrn = re.compile(".*_hrrn_cpu([0-9]*)\.csv")
        hrrn_files = []
        for f in files:
            m = re_hrrn.match(f)

            if m and len(m.groups()) == 1:
                queues = "HRRN" # HRRN
                coreCount = int(m.groups()[0])
                hrrn_files.append((f, queues, coreCount))

        hrrn = processFiles(d, hrrn_files)

        # SPN with different numbers of cores
        re_spn = re.compile(".*_spn_cpu([0-9]*)\.csv")
        spn_files = []
        for f in files:
            m = re_spn.match(f)

            if m and len(m.groups()) == 1:
                queues = "SPN" # SPN
                coreCount = int(m.groups()[0])
                spn_files.append((f, queues, coreCount))

        spn = processFiles(d, spn_files)

        # RRRRFCFS
        re_rrrrfcfs = re.compile(".*_RR([0-9]*)RR([0-9]*)FCFS_cpu([0-9]*)\.csv")
        rrrrfcfs_files = []
        for f in files:
            m = re_rrrrfcfs.match(f)

            if m and len(m.groups()) == 3:
                tq1 = m.groups()[0]
                tq2 = m.groups()[1]
                queues = tq1+","+tq2+",F" # RR, RR, FCFS
                coreCount = int(m.groups()[2])
                rrrrfcfs_files.append((f, queues, coreCount))

        rrrrfcfs = processFiles(d, rrrrfcfs_files)

        # RRRRSPN
        re_rrrrspn = re.compile(".*_RR([0-9]*)RR([0-9]*)SPN_cpu([0-9]*)\.csv")
        rrrrspn_files = []
        for f in files:
            m = re_rrrrspn.match(f)

            if m and len(m.groups()) == 3:
                tq1 = m.groups()[0]
                tq2 = m.groups()[1]
                queues = tq1+","+tq2+",S" # RR, RR, SPN
                coreCount = int(m.groups()[2])
                rrrrspn_files.append((f, queues, coreCount))

        rrrrspn = processFiles(d, rrrrspn_files)

        # RRRRHRRN
        re_rrrrhrrn = re.compile(".*_RR([0-9]*)RR([0-9]*)HRRN_cpu([0-9]*)\.csv")
        rrrrhrrn_files = []
        for f in files:
            m = re_rrrrhrrn.match(f)

            if m and len(m.groups()) == 3:
                tq1 = m.groups()[0]
                tq2 = m.groups()[1]
                queues = tq1+","+tq2+",H" # RR, RR, HRRN
                coreCount = int(m.groups()[2])
                rrrrhrrn_files.append((f, queues, coreCount))

        rrrrhrrn = processFiles(d, rrrrhrrn_files)

        # RRRRSPNFCFS
        re_rrrrspnfcfs = re.compile(".*_RR([0-9]*)RR([0-9]*)SPN_FCFS_cpu([0-9]*)\.csv")
        rrrrspnfcfs_files = []
        for f in files:
            m = re_rrrrspnfcfs.match(f)

            if m and len(m.groups()) == 3:
                tq1 = m.groups()[0]
                tq2 = m.groups()[1]
                queues = tq1+","+tq2+",SF" # RR, RR, SPN, FCFS
                coreCount = int(m.groups()[2])
                rrrrspnfcfs_files.append((f, queues, coreCount))

        rrrrspnfcfs = processFiles(d, rrrrspnfcfs_files)

        # RRRRHRRNFCFS
        re_rrrrhrrnfcfs = re.compile(".*_RR([0-9]*)RR([0-9]*)HRRN_FCFS_cpu([0-9]*)\.csv")
        rrrrhrrnfcfs_files = []
        for f in files:
            m = re_rrrrhrrnfcfs.match(f)

            if m and len(m.groups()) == 3:
                tq1 = m.groups()[0]
                tq2 = m.groups()[1]
                queues = tq1+","+tq2+",HF" # RR, RR, HRRN, FCFS
                coreCount = int(m.groups()[2])
                rrrrhrrnfcfs_files.append((f, queues, coreCount))

        rrrrhrrnfcfs = processFiles(d, rrrrhrrnfcfs_files)

        # Plots
        def combPlot(x, y, data, hue=None, onlyAverage=False):
            # For Alexander, if she really only wanted to see one value for
            # each x value, averaging averages.
            #
            # Note: the reset_index() is required to make Seaborn be able to
            # plot the data for some reason.
            # http://stackoverflow.com/a/10374456
            if onlyAverage:
                if hue:
                    data = pd.DataFrame(data.groupby([x,hue]).mean().reset_index())
                else:
                    data = pd.DataFrame(data.groupby(x).mean().reset_index())

            #sns.violinplot(x=x, y=y, data=data, hue=hue, inner=None)
            #sns.swarmplot(x=x, y=y, data=data, hue=hue, color="w", alpha=.5)
            sns.swarmplot(x=x, y=y, hue=hue, data=data)
            #sns.stripplot(x=x, y=y, hue=hue, data=data, jitter=False, alpha=0.9)

        figsize = (12,5)

        if len(fcfs.index):
            #
            # Single FCFS Queue
            #
            fig = plt.figure(figsize=figsize)
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
            plt.savefig(os.path.join(outdir, outdir_single, "Single FCFS Queue.png"))
            plt.close(fig)

        if len(spn.index):
            #
            # Single SPN Queue
            #
            fig = plt.figure(figsize=figsize)
            fig.suptitle("Single SPN Queue - "+d)

            ax1 = fig.add_subplot(2,2,1)
            combPlot(x="Cores", y="AvgTurnaround", data=spn)
            ax1.set_title("Avg Turnaround")

            ax2 = fig.add_subplot(2,2,2)
            combPlot(x="Cores", y="AvgWait", data=spn)
            ax2.set_title("Avg Wait")

            ax3 = fig.add_subplot(2,2,3)
            combPlot(x="Cores", y="AvgResponse", data=spn)
            ax3.set_title("Avg Response")

            ax4 = fig.add_subplot(2,2,4)
            combPlot(x="Cores", y="Throughput", data=spn)
            ax4.set_title("Throughput")

            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.savefig(os.path.join(outdir, outdir_single, "Single SPN Queue.png"))
            plt.close(fig)

        if len(hrrn.index):
            #
            # Single HRRN Queue
            #
            fig = plt.figure(figsize=figsize)
            fig.suptitle("Single HRRN Queue - "+d)

            ax1 = fig.add_subplot(2,2,1)
            combPlot(x="Cores", y="AvgTurnaround", data=hrrn)
            ax1.set_title("Avg Turnaround")

            ax2 = fig.add_subplot(2,2,2)
            combPlot(x="Cores", y="AvgWait", data=hrrn)
            ax2.set_title("Avg Wait")

            ax3 = fig.add_subplot(2,2,3)
            combPlot(x="Cores", y="AvgResponse", data=hrrn)
            ax3.set_title("Avg Response")

            ax4 = fig.add_subplot(2,2,4)
            combPlot(x="Cores", y="Throughput", data=hrrn)
            ax4.set_title("Throughput")

            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.savefig(os.path.join(outdir, outdir_single, "Single HRRN Queue.png"))
            plt.close(fig)

        # Compare SPN, HRRN, and FCFS for 3 cores on one plot
        if len(fcfs.index) and len(hrrn.index) and len(spn.index):
            core = 3
            comparison = pd.concat([fcfs, hrrn, spn])
            oneCore = comparison.loc[lambda df: df.Cores == core]

            fig = plt.figure(figsize=figsize)
            fig.suptitle("FCFS vs. HRRN vs. SPN Queues for "+str(core)+" cores - "+d)

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
            plt.savefig(os.path.join(outdir, outdir_queues,
                "FCFS vs. HRRN vs. SPN Queues for "+str(core)+" cores.png"))
            plt.close(fig)

        # Compare everything for 3 cores on one plot
        if len(fcfs.index) and len(hrrn.index) and len(spn.index) and \
        len(rrrrfcfs.index) and len(rrrrspn.index) and len(rrrrhrrn.index) and \
        len(rrrrspnfcfs.index) and len(rrrrhrrnfcfs.index):
            core = 3
            comparison = pd.concat([fcfs, hrrn, spn, rrrrfcfs, rrrrspn,
                rrrrhrrn, rrrrspnfcfs, rrrrhrrnfcfs])

            oneCore = comparison.loc[lambda df: df.Cores == core]

            fig = plt.figure(figsize=(figsize[0]*5, figsize[1]))
            fig.suptitle("Queue Algorithm Variations for "+str(core)+" cores - "+d)

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
            plt.savefig(os.path.join(outdir, outdir_queues,
                "Queue Algorithm Variations for "+str(core)+" cores.png"))
            plt.close(fig)

        if len(rrrrfcfs.index):
            #
            # RR, RR, FCFS Queues
            #
            #for core in rrrrfcfs['Cores'].unique():
            for core in [3]:
                comparison = pd.concat([fcfs, hrrn, spn, rrrrfcfs])
                oneCore = comparison.loc[lambda df: df.Cores == core]

                fig = plt.figure(figsize=(figsize[0]*1.5, figsize[1]))
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

                plt.subplots_adjust(wspace=0.15, hspace=0.4)
                plt.savefig(os.path.join(outdir, outdir_queues,
                    "RR RR FCFS Queues for "+str(core)+" cores.png"))
                plt.close(fig)

            fig = plt.figure(figsize=figsize)
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
            plt.savefig(os.path.join(outdir, outdir_cores, "RR RR FCFS Queues.png"))
            plt.close(fig)

        if len(rrrrspn.index):
            #
            # RR, RR, SPN Queues
            #
            #for core in rrrrspn['Cores'].unique():
            for core in [3]:
                comparison = pd.concat([fcfs, hrrn, spn, rrrrspn])
                oneCore = comparison.loc[lambda df: df.Cores == core]

                fig = plt.figure(figsize=(figsize[0]*1.5, figsize[1]))
                fig.suptitle("RR RR SPN Queues for "+str(core)+" cores - "+d)

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

                plt.subplots_adjust(wspace=0.15, hspace=0.4)
                plt.savefig(os.path.join(outdir, outdir_queues,
                    "RR RR SPN Queues for "+str(core)+" cores.png"))
                plt.close(fig)

            fig = plt.figure(figsize=figsize)
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
            plt.savefig(os.path.join(outdir, outdir_cores, "RR RR SPN Queues.png"))
            plt.close(fig)

        if len(rrrrhrrn.index):
            #
            # RR, RR, HRRN Queues
            #
            #for core in rrrrhrrn['Cores'].unique():
            for core in [3]:
                comparison = pd.concat([fcfs, hrrn, spn, rrrrhrrn])
                oneCore = comparison.loc[lambda df: df.Cores == core]

                fig = plt.figure(figsize=(figsize[0]*1.5, figsize[1]))
                fig.suptitle("RR RR HRRN Queues for "+str(core)+" cores - "+d)

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

                plt.subplots_adjust(wspace=0.15, hspace=0.4)
                plt.savefig(os.path.join(outdir, outdir_queues,
                    "RR RR HRRN Queues for "+str(core)+" cores.png"))
                plt.close(fig)

            fig = plt.figure(figsize=figsize)
            fig.suptitle("RR RR HRRN Queues - "+d)

            ax1 = fig.add_subplot(2,2,1)
            combPlot(hue="Queues", x="Cores", y="AvgTurnaround", data=rrrrhrrn)
            ax1.set_title("Avg Turnaround")

            ax2 = fig.add_subplot(2,2,2)
            combPlot(hue="Queues", x="Cores", y="AvgWait", data=rrrrhrrn)
            ax2.set_title("Avg Wait")

            ax3 = fig.add_subplot(2,2,3)
            combPlot(hue="Queues", x="Cores", y="AvgResponse", data=rrrrhrrn)
            ax3.set_title("Avg Response")

            ax4 = fig.add_subplot(2,2,4)
            combPlot(hue="Queues", x="Cores", y="Throughput", data=rrrrhrrn)
            ax4.set_title("Throughput")

            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.savefig(os.path.join(outdir, outdir_cores, "RR RR HRRN Queues.png"))
            plt.close(fig)

        if len(rrrrspnfcfs.index):
            #
            # RR, RR, SPN, FCFS Queues
            #
            #for core in rrrrspnfcfs['Cores'].unique():
            for core in [3]:
                comparison = pd.concat([fcfs, hrrn, spn, rrrrspnfcfs])
                oneCore = comparison.loc[lambda df: df.Cores == core]

                fig = plt.figure(figsize=(figsize[0]*1.5, figsize[1]))
                fig.suptitle("RR RR SPN FCFS Queues for "+str(core)+" cores - "+d)

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

                plt.subplots_adjust(wspace=0.15, hspace=0.4)
                plt.savefig(os.path.join(outdir, outdir_queues,
                    "RR RR SPN FCFS Queues for "+str(core)+" cores.png"))
                plt.close(fig)

            fig = plt.figure(figsize=figsize)
            fig.suptitle("RR RR SPN FCFS Queues - "+d)

            ax1 = fig.add_subplot(2,2,1)
            combPlot(hue="Queues", x="Cores", y="AvgTurnaround", data=rrrrspnfcfs)
            ax1.set_title("Avg Turnaround")

            ax2 = fig.add_subplot(2,2,2)
            combPlot(hue="Queues", x="Cores", y="AvgWait", data=rrrrspnfcfs)
            ax2.set_title("Avg Wait")

            ax3 = fig.add_subplot(2,2,3)
            combPlot(hue="Queues", x="Cores", y="AvgResponse", data=rrrrspnfcfs)
            ax3.set_title("Avg Response")

            ax4 = fig.add_subplot(2,2,4)
            combPlot(hue="Queues", x="Cores", y="Throughput", data=rrrrspnfcfs)
            ax4.set_title("Throughput")

            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.savefig(os.path.join(outdir, outdir_cores, "RR RR SPN FCFS Queues.png"))
            plt.close(fig)

        if len(rrrrhrrnfcfs.index):
            #
            # RR, RR, HRRN, FCFS Queues
            #
            #for core in rrrrhrrnfcfs['Cores'].unique():
            for core in [3]:
                comparison = pd.concat([fcfs, hrrn, spn, rrrrhrrnfcfs])
                oneCore = comparison.loc[lambda df: df.Cores == core]

                fig = plt.figure(figsize=(figsize[0]*1.5, figsize[1]))
                fig.suptitle("RR RR HRRN FCFS Queues for "+str(core)+" cores - "+d)

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

                plt.subplots_adjust(wspace=0.15, hspace=0.4)
                plt.savefig(os.path.join(outdir, outdir_queues,
                    "RR RR HRRN FCFS Queues for "+str(core)+" cores.png"))
                plt.close(fig)

            fig = plt.figure(figsize=figsize)
            fig.suptitle("RR RR HRRN FCFS Queues - "+d)

            ax1 = fig.add_subplot(2,2,1)
            combPlot(hue="Queues", x="Cores", y="AvgTurnaround", data=rrrrhrrnfcfs)
            ax1.set_title("Avg Turnaround")

            ax2 = fig.add_subplot(2,2,2)
            combPlot(hue="Queues", x="Cores", y="AvgWait", data=rrrrhrrnfcfs)
            ax2.set_title("Avg Wait")

            ax3 = fig.add_subplot(2,2,3)
            combPlot(hue="Queues", x="Cores", y="AvgResponse", data=rrrrhrrnfcfs)
            ax3.set_title("Avg Response")

            ax4 = fig.add_subplot(2,2,4)
            combPlot(hue="Queues", x="Cores", y="Throughput", data=rrrrhrrnfcfs)
            ax4.set_title("Throughput")

            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.savefig(os.path.join(outdir, outdir_cores, "RR RR HRRN FCFS Queues.png"))
            plt.close(fig)

    # Show all plots at the end
    #plt.show()
