#
# Intial prototype of the overall simulation structure
#
# Note: will be re-coded in C++ when I get something initally working.
#
import os.path

# The different queueing algorithms
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

class QueueFCFS(Queue):
    pass

# Stores information about a process after reading from the file but before it
# arrives and gets added to the process table
class Process:
    def __init__(self, pid, arrivalTime, times):
        self.pid = pid
        self.arrivalTime = arrivalTime
        self.times = times

    def __lt__(self, other):
         return self.arrivalTime < other.arrivalTime

    def __repr__(self):
        return "PID:{0} Arrival:{1} Times:{2}".format(
                self.pid,self.arrivalTime,self.times)

# Handle information about what is running on each CPU, context switches,
# starting/stopping proceses, etc.
class CPU:
    def __init__(self, processTable, completedProcesses, contextSwitchTime):
        self.processTable = processTable
        self.completedProcesses = completedProcesses
        self.running = False
        self.pid = 0 # PID of running process

        # To implement a context switch, and note that we start off in a
        # context switch
        self.contextSwitchTime = contextSwitchTime
        self.contextSwitch = True
        self.contextSwitchCount = 0

    def startContextSwitch(self):
        self.running = False
        self.contextSwitch = True
        self.contextSwitchCount = 0

    def inContextSwitch(self):
        if self.contextSwitch and self.contextSwitchCount < self.contextSwitchTime:
            self.contextSwitchCount += 1
            return True
        else:
            self.contextSwitch = False
            self.contextSwitchCount = 0
            return False

    def start(self, clock, pid):
        self.running = True
        self.pid = pid

        p = self.processTable[pid]
        p.started = clock

    def done(self, clock, p):
        p.completed = clock
        self.completedProcesses.append(p)
        del self.processTable[p.pid]

# All the information about a single process
class PCB:
    def __init__(self, pid, times, submitted):
        # Identification
        self.pid = pid

        # Not normally in an OS, but since we're not running real
        # processes, we need to know how long each of these processes
        # would take if they were to do something
        self.times = times

        # A priority that will increase over time so we don't get
        # starvation
        self.priority = 0

        # Important timestamps
        self.submitted = submitted
        self.started = 0
        self.completed = 0

        # Time spent in each of the following
        self.queues = 0 # Not including I/O time
        self.executing = 0 # Resets after hitting each I/O
        self.executingTotal = 0
        self.io = 0 # Resets after hitting each I/O
        self.ioTotal = 0

    def __repr__(self):
        return ",".join([str(i) for i in [self.pid, self.submitted,
            self.started, self.completed, self.queues, self.executingTotal,
            self.ioTotal]])

    # Check if I/O or execution has completed, and if so then remove that from
    # the list of times and reset the I/O or execution time accordingly, which
    # we use to determine when we've finished the next time item
    def isDoneIO(self, clock):
        isDone = self.io == self.times[0]

        if isDone:
            self.times.pop()
            self.io = 0

        return isDone

    def isDoneExec(self):
        isDone = self.executing == self.times[0]

        if isDone:
            self.times.pop()
            self.executing = 0

        return isDone

    # Increment the times when doing I/O, executing, in queues
    def incIO(self):
        self.io += 1
        self.ioTotal += 1

    def incExec(self):
        self.executing += 1
        self.executingTotal += 1

    def incQueues(self):
        self.queues += 1

# We will re-run the same files multiple times with different input so we can
# compare different algorithms
def loadProcessesFromCSV(fn):
    allProcesses = []

    # Load from file
    #
    # Format: PID, ArrivalTime, CPU t, IO t, CPU t, ...
    with open(fn, 'r') as f:
        for line in f:
            pid, arrivalTime, times = line.split(",", 2)
            allProcesses.append(Process(int(pid), int(arrivalTime),
                [int(x.strip()) for x in times.split(",")]))

    # Sort based on arrival times
    allProcesses.sort()

    return allProcesses

# Write outputs to a file that we can then make nice plots with. Separate the
# analysis from the simulation since the simulation will likely take a while.
def writeResultsToCSV(results, fn):
    with open(fn, 'w') as f:
        f.write("PID,Submitted,Started,Completed,Queues,Executing,IO\r\n")

        for p in results:
            f.write(repr(p) + "\r\n")

def runSimulation(filename, queue, cpuCount, contextSwitchTime=2, debug=False):
    # Initialize
    clock = 0
    allProcesses = loadProcessesFromCSV(filename)

    # Process table, i.e. list of all processes running, indexed by the PID
    processTable = {}

    # Move processes here when they're done
    completedProcesses = []

    # Create a CPU object for each CPU to keep track of what is running on each
    cpus = [CPU(processTable, completedProcesses, contextSwitchTime)
            for i in range(0,cpuCount)]

    # We have one I/O queue that is FCFS
    io = []

    while True:
        # Add processes as they arrive to the process table
        arrived = [p for p in allProcesses if p.arrivalTime == clock]

        for p in arrived:
            # Remove it since it only arrives once. Not really needed, but
            # searches for equality on a smaller array are faster.
            allProcesses.remove(p)

            # Add to the process table and add to our queue to start executing
            processTable[p.pid] = PCB(p.pid, p.times, clock)
            queue.enqueue(p.pid)

            if debug:
                print(clock, "Process", p.pid, "arrived")

        # Manage what is waiting in the I/O queue, which is FCFS
        if io:
            pid = io[-1]
            p = processTable[pid]
            p.incIO()

            # If it's done with I/O, then remove it from this queue and put it
            # back in the queue to be executed
            if p.isDoneIO(clock):
                if debug:
                    print(clock, "Process", pid, "done with I/O")
                io.pop()

                # If the I/O wasn't the last operation, then we have more CPU
                # time needed for this process
                if p.times:
                    queue.enqueue(pid)
                else:
                    cpu.done(clock, p)
                    if debug:
                        print(clock, "Process", p.pid, "completed")

        # Run each CPU
        for cpuIndex, cpu in enumerate(cpus):
            # If a process is running, increment it's executing time
            if cpu.running:
                p = processTable[cpu.pid]
                p.incExec()

                # If it's done, then remove it
                #if debug:
                #   print(clock, "Executing", p.pid, "Time left", p.times[0]-p.executing)
                if p.isDoneExec():
                    # If there are more times, then it's waiting for I/O now
                    if p.times:
                        if debug:
                            print(clock, "Process", cpu.pid, "performing I/O")
                        io.insert(0, p.pid)
                    # Otherwise, we're done and start a context switch
                    else:
                        cpu.done(clock, p)
                        if debug:
                            print(clock, "Process", p.pid, "completed")

                    # In either case, this core is no longer running any
                    # process and now it's in a context switch
                    cpu.startContextSwitch()

            # If no process is running, check if we're done with any context
            # switch, and if not but there's another process in the global
            # queue, start running it
            elif not cpu.inContextSwitch() and queue.size():
                cpu.start(clock, queue.dequeue())
                if debug:
                    print(clock, "Running process", cpu.pid, "on CPU", cpuIndex)

        # Increment all the processes that are currently waiting in a queue
        for pid in queue.items:
            p = processTable[pid]
            p.incQueues()

        # We're done with this clock cycle
        clock += 1

        # Quit once all processes have finished running and nothing in the queues
        if not allProcesses and not queue.size() and not io and not [c
                    for c in cpus if c.running or c.contextSwitch]:
            break

    return completedProcesses

if __name__ == "__main__":
    # Run on each of the 10 randomly-generated input files of processes
    for i in range(0,10):
        # Test different numbers of CPUs with FCFS
        queue = QueueFCFS()
        for j in range(1,10):
            infile="processes/"+str(i)+".txt"
            outfile="results/"+str(i)+"_fcfs_cpu"+str(j)+".csv"

            # Does the input exist?
            if not os.path.isfile(infile):
                print("Doesn't exist:", infile)
                continue

            # Skip if the output exists already
            if not os.path.isfile(outfile):
                print("Running:", infile)
                writeResultsToCSV(runSimulation(infile, queue, cpuCount=j),
                        outfile)
                print("Results:", outfile)
            else:
                print("Skipping:", outfile)
