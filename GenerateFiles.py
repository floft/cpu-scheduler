import pandas
import numpy as np

# Format:
#    PID #, Arrival t, CPU t, IO t, CPU t, IO t, CPU t
#    PID #, Arrival t, CPU t
#    PID #, Arrival t, CPU t, IO t, CPU t, IO t

np.random.seed(0)

if __name__ == "__main__":
    #f = open(filename, 'w')
    #f.write(line)

    # Parameters for each file
    maxFiles = 10
    directory = "processes"
    count = 2000
    maxTime = 100
    maxSwitches = 5
    maxArrivalInc = 5

    for fn in range(0,maxFiles):
        with open(directory+"/" + str(fn) + ".csv", "w") as f:
            # Initialize for each file
            pid = 0
            arrival = 0

            for i in range(0, count):
                # How many times we have, starting with CPU and then alternating
                # between CPU and IO
                switches = np.random.randint(1,maxSwitches)

                # Choose how long after the last process arived to make this
                # process arrive
                arrival += np.random.randint(1,maxArrivalInc)

                # Create the proccess line
                processFile = []
                processFile.append(pid)
                processFile.append(arrival)

                for j in range(0,switches):
                    processFile.append(np.random.randint(1,maxTime))

                pid += 1

                f.write(",".join(str(x) for x in processFile) + "\r\n")
