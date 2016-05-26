# CPU Scheduler Simulation

## Running

To generate all the process files:

    python GenerateFiles.py

To run the simulation (will take a while):

    python simulation.py

To show plots of the simulation results (argument is directory of all the CSV
files):

    python AnalyzeOutput.py results

## Installing Python and libraries on Windows

Download and install Python 3 from
[Anaconda](https://www.continuum.io/downloads). Then run the following command
in the terminal to install the one library that isn't installed by default:

    conda install seaborn
