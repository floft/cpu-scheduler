# CPU Scheduler Simulation

## Running

To generate all the process files:

    python3 GenerateFiles.py

To run the simulation (will take a while):

    python3 simulation.py processes.50_10_5_5 results5
    python3 simulation.py processes.50_10_5_500 results500

To show plots of the simulation results (argument is directory of all the CSV
files), run the following then look in the newly-created *plots_{results5,results500}* directories:

    python3 AnalyzeOutput.py results5
    python3 AnalyzeOutput.py results500

## Installing Python and libraries on Windows

Download and install Python 3 from
[Anaconda](https://www.continuum.io/downloads). Then run the following command
in the terminal to install the one library that isn't installed by default:

    conda install seaborn
