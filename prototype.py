#
# Intial prototype of the overall simulation structure
#
# Note: will be re-coded in C++ when I get something initally working.
#
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

class Process:
    def __init__(self, pid, arrivalTime, times):
        self.pid = pid
        self.arrivalTime = arrivalTime
        self.times = times

    def __lt__(self, other):
         return self.arrivalTime < other.arrivalTime

    def __repr__(self):
        return "PID:{0} Arrival:{1} Times:{2}".format(self.pid,self.arrivalTime,self.times)

class CPU:
    def __init__(self, contextSwitchTime):
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
        self.queues = 0
        self.executing = 0 # Resets after hitting each I/O
        self.executingTotal = 0
        self.io = 0 # Resets after hitting each I/O
        self.ioTotal = 0

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

def processComplete(clock, p, processTable, completedProcesses):
    p.completed = clock
    print(clock, "Process", p.pid, "completed")
    completedProcesses.append(p)
    del processTable[p.pid]

if __name__ == "__main__":
    # Parameters
    cpuCount = 4
    filename = "processes/0.txt"
    contextSwitchTime = 2

    # Initialize
    clock = 0
    allProcesses = loadProcessesFromCSV(filename)

    # Create a CPU object for each CPU to keep track of what is running on each
    cpus = [CPU(contextSwitchTime) for i in range(0,cpuCount)]

    # Process table, i.e. list of all processes running, indexed by the PID
    processTable = {}

    # Move processes here when they're done
    completedProcesses = []

    # Queue of the desired algorithm
    queue = QueueFCFS()

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

            print(clock, "Process", p.pid, "arrived")

        # Manage what is waiting in the I/O queue
        if io:
            pid = io[-1]
            p = processTable[pid]
            p.io += 1
            p.ioTotal += 1

            # If it's done with I/O, then remove it from this queue and put it
            # back in the queue to be executed
            if p.io == p.times[0]:
                p.times.pop()
                p.io = 0
                io.pop()
                print(clock, "Process", p.pid, "done with I/O")

                # If the I/O wasn't the last operation, then we have more CPU
                # time needed for this process
                if p.times:
                    queue.enqueue(pid)
                else:
                    processComplete(clock, p, processTable, completedProcesses)

        # Run each CPU
        for cpuIndex, cpu in enumerate(cpus):
            # If a process is running, increment it's executing time
            if cpu.running:
                p = processTable[cpu.pid]
                p.executing += 1
                p.executingTotal += 1

                # If it's done, then remove it
                #print(clock, "Executing", p.pid, "Time left", p.times[0]-p.executing)
                if p.executing == p.times[0]:
                    p.times.pop()
                    p.executing = 0

                    # If there are more times, then it's waiting for I/O now
                    if p.times:
                        print(clock, "Process", cpu.pid, "performing I/O")
                        io.insert(0, p.pid)
                    # Otherwise, we're done and start a context switch
                    else:
                        processComplete(clock, p, processTable, completedProcesses)

                    # In either case, this core is no longer running any
                    # process and now it's in a context switch
                    cpu.startContextSwitch()

            # If no process is running, grab another process from the global
            # queue if there is one
            #
            # Note: this is another elif since if it stopped running a process
            # on this clock cycle, then we've already spent this clock cycle
            # executing that cycle and can't count this cycle as part of a
            # context switch
            elif queue.size() and not cpu.running and not cpu.inContextSwitch():
                cpu.running = True
                cpu.pid = queue.dequeue()
                print(clock, "Running process", cpu.pid, "on CPU", cpuIndex)

                p = processTable[cpu.pid]
                p.started = clock

        # We're done with this clock cycle
        clock += 1

        # Quit once all processes have finished running
        if not allProcesses and not [c for c in cpus if c.running]:
            break
