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
def processFiles(files):
    results = pd.DataFrame()

    for i, [fn, cores] in enumerate(files):
        df = pd.read_csv(os.path.join(directory,fn))

        turnaround = df['Completed'] - df['Submitted']
        wait = df['Queues']
        response = df['Started'] - df['Submitted']
        avgTurnaround = turnaround.mean()
        avgWait = wait.mean()
        avgResponse = response.mean()
        throughput = df['Completed'].count() / df['Completed'].iloc[-1]

        results = results.append(pd.DataFrame([[cores, avgTurnaround, avgWait,
            avgResponse, throughput]], index=[i],
            columns=['Cores','AvgTurnaround','AvgWait','AvgResponse','Throughput']))

    return results

# Plot all the data
directory = "results"
files = os.listdir(directory)

# Match the different tests we're doing
re_fcfs = re.compile(".*_fcfs_cpu(.*)\.csv")
fcfs_files = []
for f in files:
    m = re_fcfs.match(f)

    if m and len(m.groups()):
        # filename, # of cores
        fcfs_files.append((f, int(m.groups()[0])))

# FCFS with different numbers of cores
fcfs = processFiles(fcfs_files)

plt.figure()
sns.violinplot(x="Cores", y="AvgTurnaround", data=fcfs)
plt.title("Average Turnaround with FCFS varying number of cores")

plt.figure()
sns.violinplot(x="Cores", y="AvgWait", data=fcfs)
plt.title("Average Wait with FCFS varying number of cores")

plt.figure()
sns.violinplot(x="Cores", y="AvgResponse", data=fcfs)
plt.title("Average Response with FCFS varying number of cores")

plt.figure()
sns.violinplot(x="Cores", y="Throughput", data=fcfs)
plt.title("Average Throughput with FCFS varying number of cores")

plt.show()
