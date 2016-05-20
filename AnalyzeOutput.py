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
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import andrews_curves

# Make them look prettier
plt.style.use('ggplot')
sns.set(style="ticks")

# For reproducibility
random.seed(0)
np.random.seed(0)

# Plot all the data
df = pd.read_csv('processes/example_output.csv')

turnaround = df['Completed'] - df['Submitted']
wait = df['Queues']
response = df['Started'] - df['Submitted']
avgTurnaround = turnaround.mean()
avgWait = wait.mean()
avgResponse = response.mean()
throughput = df['Completed'].count() / df['Completed'].iloc[-1]

print("AvgTurnaround:", avgTurnaround)
print("AvgWait:", avgWait)
print("AvgResponse:", avgResponse)
print("Throughput:", throughput)
