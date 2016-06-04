#
# Generate the files of processes arriving at the CPU at different times and
# taking certain amounts of CPU and I/O time
#
# Format:
#    PID #, Arrival t, CPU t, IO t, CPU t, IO t, CPU t
#    PID #, Arrival t, CPU t
#    PID #, Arrival t, CPU t, IO t, CPU t, IO t
#
import os
import numpy as np

np.random.seed(0)

if __name__ == "__main__":
    generateInput = True
    generateExampleOutput = False
    directory = "processes"
    outputdirectory = "results"

    #
    # Generate process files for running through the simulation
    #
    if generateInput:
        for maxArrivalInc in [5, 500]:
            # Parameters for each file
            maxFiles = 5
            count = 1000
            maxTimeExec = 50
            maxTimeIO = 10
            maxSwitches = 5
            #maxArrivalInc = 5

            d = directory + "." + str(maxTimeExec) + "_" + str(maxTimeIO) + "_" \
                + str(maxSwitches) + "_" + str(maxArrivalInc)

            # Make sure the directory exists
            if not os.path.exists(d):
                os.makedirs(d)

            for fn in range(0,maxFiles):
                with open(os.path.join(d,str(fn)+".txt"), "w") as f:
                    # Initialize for each file
                    arrival = 0

                    for pid in range(0, count):
                        # How many times we have, starting with CPU and then alternating
                        # between CPU and IO
                        switches = np.random.randint(1,maxSwitches)

                        # Choose how long after the last process arived to make this
                        # process arrive
                        arrival += np.random.randint(1,maxArrivalInc)

                        # Create the proccess line
                        l = []
                        l.append(pid)
                        l.append(arrival)

                        for j in range(0,switches):
                            if j%2 == 0:
                                l.append(np.random.randint(1,maxTimeExec))
                            else:
                                l.append(np.random.randint(1,maxTimeIO))

                        f.write(",".join(str(x) for x in l) + "\n")

    #
    # Generate example output file for initial plotting
    #
    if generateExampleOutput:
        # Make sure the directory exists
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)

        with open(os.path.join(outputdirectory,"example_output.csv"), "w") as f:
            # Initialize for each file
            arrival = 0

            # Some maximum values for random number generation
            maxStarted = 100
            maxCompleted = 1000
            maxWaitQueues = 50
            maxExecuting = 500
            maxWaitIO = 10

            f.write("PID,Submitted,Started,Completed,Queues,Executing,IO\n")

            for pid in range(0, count):
                arrival += np.random.randint(1,maxArrivalInc)
                started = arrival + np.random.randint(1,maxStarted)
                completed = arrival + np.random.randint(1,maxCompleted)
                waitingQueues = np.random.randint(1,maxWaitQueues)
                executing = np.random.randint(1,maxExecuting)
                waitingIO = np.random.randint(1,maxWaitIO)

                # Create the proccess line
                l = []
                l.append(pid)
                l.append(arrival)
                l.append(started)
                l.append(completed)
                l.append(waitingQueues)
                l.append(executing)
                l.append(waitingIO)
                f.write(",".join(str(x) for x in l) + "\n")
