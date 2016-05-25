#
# Discrete event simulation of CPU scheduling
#
import os
import numpy as np
import multiprocessing

# The different queueing algorithms
#
# Base class here copied and pasted from:
#  http://interactivepython.org/runestone/static/pythonds/BasicDS/ImplementingaQueueinPython.html
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    # By default, never preempt a running process, only used in RR and other
    # preemptive algorithms
    def preemptTime(self):
        return np.Inf

# The base class is FCFS
class QueueFCFS(Queue):
    pass

# RR is FCFS but with preemption
class QueueRR(Queue):
    def __init__(self, preempt):
        Queue.__init__(self)
        self.preempt = preempt

    def preemptTime(self):
        return self.preempt

# SPN is FCFS but selects which is next based on the smallest remaining time
class QueueSPN(Queue):
    def dequeue(self):
        # Find one with the least amount of time remaining
        timesRemaining = [sum(p.times) for p in self.items]
        index = np.argmin(timesRemaining)
        item = self.items[index]

        # Delete it from the queue
        del self.items[index]

        return item

# Allow for working with multiple queues
class MultilevelQueue():
    def __init__(self, queues):
        assert len(queues) > 0, "MultilevelQueue must have at least one queue"
        self.grabFromNext = 0
        self.queues = queues

    def isEmpty(self):
        return all(q.isEmpty() for q in self.queues)

    # Insert into the first queue if priority = 0, which is used if this is the
    # first time this is added to this multilevel queue. However, if this has
    # been preempted (in RR for instance), then next time it runs it'll get a
    # lower priority and be moved to the second if priority=1, etc. queue.
    def enqueue(self, item, priority=0):
        self.queues[min(priority,len(self.queues)-1)].enqueue(item)

    # Grab one from the queues just cycling through each, so first from the
    # first queue, next from the second queue, etc.
    def dequeue(self):
        assert not self.isEmpty(), "Can't dequeue() when no items in queue"

        # Skip a queue if a queue doesn't have anything in it
        while self.queues[self.grabFromNext].isEmpty():
            self.incGrabNext()

        item = self.queues[self.grabFromNext].dequeue()

        # Save the current priority, i.e. which queue this came from
        fromQueue = self.grabFromNext
        item.priority = fromQueue

        # Save the max time it can run before being preempted, changes
        # depending on which queue this came out of, e.g. may have come from a
        # RR with a time quantum of 2 or another one with a time quantum of 10
        item.preemptTime = self.queues[fromQueue].preemptTime()

        # Next time we'll grab from the next queue
        self.incGrabNext()

        return item

    def incGrabNext(self):
        self.grabFromNext = (self.grabFromNext+1)%len(self.queues)

    def size(self):
        return sum(q.size() for q in self.queues)

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
        self.p = None # A reference to the PCB of the process being run

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

    def start(self, clock, p):
        self.running = True
        self.p = p

        # Only set started if this is the first time it's been started. We'll
        # call this function both when it starts and when it starts up again
        # after performing I/O.
        if self.p.started == None:
            self.p.started = clock

    # TODO maybe this should be a PCB method? It doesn't even use self.p and is
    # called also when done with I/O even if not on a CPU
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

        # A priority that will increase over time so we don't get starvation
        #
        # Note: at the moment this stores which queue in the multilevel queues
        # this process came from so that next time it will be moved to a later
        # queue.
        self.priority = 0

        # When this process will be preempted next, by default it won't
        self.preemptTime = np.Inf

        # In addition to executing and executingTotal, we need to record how
        # long it was since we were last preempted so that we know when to
        # preempt next
        self.sinceLastPreempt = 0

        # Important timestamps
        self.submitted = submitted
        self.started = None # None means not started yet, 0 could be started at zero
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
        isDone = self.io == self.times[-1]

        if isDone:
            self.times.pop()
            self.io = 0

        return isDone

    def isDoneExec(self):
        isDone = self.executing == self.times[-1]

        if isDone:
            self.times.pop()
            self.executing = 0

        return isDone

    def isPreempted(self):
        isPreempt = self.sinceLastPreempt == self.preemptTime

        if isPreempt:
            self.sinceLastPreempt = 0

        return isPreempt

    # Increment the times when doing I/O, executing, in queues
    def incIO(self):
        self.io += 1
        self.ioTotal += 1

    def incExec(self):
        self.executing += 1
        self.sinceLastPreempt += 1
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

            # Reverse the times list so that popping off the back is doable via
            # the .pop() function
            allProcesses.append(Process(int(pid), int(arrivalTime),
                [int(x.strip()) for x in reversed(times.split(","))]))

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

# Load a file, run the simulation, output the results to a file
def runSimulation(infile, outfile, queues, cpuCount, contextSwitchTime, debug):
    # Initialize
    clock = 0
    queue = MultilevelQueue(queues)
    allProcesses = loadProcessesFromCSV(infile)

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

        for process in arrived:
            # Remove it since it only arrives once. Not really needed, but
            # searches for equality on a smaller array are faster.
            allProcesses.remove(process)

            # Add to the process table and add to our queue to start executing
            p = PCB(process.pid, process.times, clock)
            processTable[process.pid] = p
            queue.enqueue(p, priority=0)

            if debug:
                print(clock, "Process", p.pid, "arrived")

        # Manage what is waiting in the I/O queue, which is FCFS
        if io:
            p = io[-1]
            p.incIO()

            if debug:
                oldTime = p.times[-1]

            # If it's done with I/O, then remove it from this queue and put it
            # back in the queue to be executed
            if p.isDoneIO(clock):
                if debug:
                    print(clock, "Process", p.pid, "done with I/O after", oldTime)
                io.pop()

                # If the I/O wasn't the last operation, then we have more CPU
                # time needed for this process
                if p.times:
                    queue.enqueue(p, priority=0)
                else:
                    cpu.done(clock, p)
                    if debug:
                        print(clock, "Process", p.pid, "completed after",
                                p.executingTotal, "and", p.ioTotal, "I/O")

        # Run each CPU
        for cpuIndex, cpu in enumerate(cpus):
            # If a process is running, increment it's executing time
            if cpu.running:
                cpu.p.incExec()

                if debug:
                    print(clock, "Executing", cpu.p.pid, "Time left",
                        cpu.p.times[-1]-cpu.p.executing)

                # If it's done, then remove it
                if cpu.p.isDoneExec():
                    # If there are more times, then it's waiting for I/O now
                    if cpu.p.times:
                        if debug:
                            print(clock, "Process", cpu.p.pid,
                                    "performing I/O for", cpu.p.times[-1])
                        io.insert(0, cpu.p)
                    # Otherwise, we're done and start a context switch
                    else:
                        cpu.done(clock, cpu.p)
                        if debug:
                            print(clock, "Process", cpu.p.pid, "completed after",
                                    cpu.p.executingTotal, "and", cpu.p.ioTotal, "I/O")

                    # In either case, this core is no longer running any
                    # process and now it's in a context switch
                    cpu.startContextSwitch()

                # Otherwise, if it's preempted, then move it to the next queue
                # and let the processor move onto another process next clock
                # cycle
                elif cpu.p.isPreempted():
                    newPriority = min(cpu.p.priority,len(queue.queues)-1)
                    queue.enqueue(cpu.p, priority=newPriority)
                    cpu.startContextSwitch()

                    if debug:
                        print(clock, "Preempting", cpu.p.pid, "after",
                                cpu.p.sinceLastPreempt, "moving to priority",
                                newPriority)

            # If no process is running, check if we're done with any context
            # switch, and if not but there's another process in the global
            # queue, start running it
            elif not cpu.inContextSwitch() and queue.size():
                cpu.start(clock, queue.dequeue())

                if debug:
                    print(clock, "Running process", cpu.p.pid, "on CPU", cpuIndex)

        # Increment all the processes that are currently waiting in a queue.
        # We're dealing with a multilevel queue, so it's 2D.
        for q in queue.queues:
            for p in q.items:
                p.incQueues()

        # We're done with this clock cycle
        clock += 1

        # TODO increment by time to get us to the next event

        # Quit once all processes have finished running and nothing in the queues
        if not allProcesses and not queue.size() and not io and not [c
                    for c in cpus if c.running or c.contextSwitch]:
            break

    writeResultsToCSV(completedProcesses, outfile)

    # For ease of printing out what we're done with
    return outfile

# Run all the tests
if __name__ == "__main__":
    debug = False
    outdir = "results"

    # We'll just set this here for all the simulations
    contextSwitchTime = 2

    if debug:
        maxProcesses = 1
    else:
        maxProcesses = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=maxProcesses)
    results = []

    print("Will use", maxProcesses, "threads")

    # Make sure output directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Run on each of the 5 randomly-generated input files of processes
    for fn in range(0,5):
        testfile = str(fn)
        infile = os.path.join("processes", testfile+".txt")

        # Does the input exist?
        if not os.path.isfile(infile):
            print("Doesn't exist:", infile)
            continue

        # Run each of the tests
        #
        # Defined here to just simplify the code, not having to pass in
        # additional variables defined in this scope that are the same for all
        # tests
        def runTest(testname, queues, cpuCount):
            outfile = os.path.join(outdir, testfile+"_"+testname+".csv")

            # Skip if the output exists already
            if not os.path.isfile(outfile):
                print("Running:", outfile)
                results.append(pool.apply_async(runSimulation, [infile,
                    outfile, queues, cpuCount, contextSwitchTime, debug]))
            else:
                print("Skipping:", outfile)

        # Test different numbers of FCFC queues, also varying number of cores
        for fcfsCount in [1,3,5,7]:
            for cores in range(1,18,2):
                queues = [QueueFCFS() for i in range(0,fcfsCount)]
                runTest("fcfs"+str(fcfsCount)+"_cpu"+str(cores), queues, cores)

        # Test RR, RR, FCFS, also varying number of cores
        for tq1, tq2 in [(2,10), (10,2), (5,5), (10,10), (50,50), (50,10), (10,50)]:
            for cores in range(1,18,2):
                queues = [QueueRR(tq1), QueueRR(tq2), QueueFCFS()]
                runTest("RR"+str(tq1)+"RR"+str(tq2)+"FCFS_cpu"+str(cores), queues, cores)

        # Test RR, RR, SPN, also varying number of cores
        for tq1, tq2 in [(2,10), (10,2), (5,5), (10,10), (50,50), (50,10), (10,50)]:
            for cores in range(1,18,2):
                queues = [QueueRR(tq1), QueueRR(tq2), QueueSPN()]
                runTest("RR"+str(tq1)+"RR"+str(tq2)+"SPN_cpu"+str(cores), queues, cores)

    # Print when each finishes
    for r in results:
        print("Results:", r.get())
