import numpy as np
import sys
import os
from dataclasses import dataclass
import subprocess
import LC_Helpers as lch


@dataclass
class RunInstance:
    muval: float
    run_location: str
    run_command: str

def WriteParamsFile(run, file):
    file.write(str(run.muval))


base_name = "viscous_tests"
# Adding in a minimum value but I will be appending 0. I just want the stepping to be log scale
N_Steps = 40
mu_min = 0.0 * 1.0e-4
mu_max = 1.0e0
mu_vals = np.linspace(mu_min, mu_max, N_Steps)
# mu_vals = np.logspace(mu_min, mu_max, N_Steps - 1)
# mu_vals = np.insert(mu_vals, 0, 0.0, axis=0)
# print(mu_vals)

current_directory = os.getcwd()
foldername = os.path.join(current_directory, base_name)
# os.mkdir(foldername)
lch.create_folder(foldername)
# Building Run Instances
runs = []
for i in range(N_Steps):
    runs.append(RunInstance(mu_vals[i], "", ""))

# Building Folder Structure
for i in range(N_Steps):
    run_directory = os.path.join(foldername, str(i))
    # os.mkdir(run_directory)
    lch.create_folder(run_directory)
    params_file = os.path.join(run_directory, "params.txt")
    file = open(params_file, 'w')
    runs[i].run_location = run_directory
    commandpath = os.path.abspath(os.path.join("..", "viscous_lattice_parametrized"))
    runs[i].run_command = f"{commandpath} -p {params_file} -o {run_directory}"
    WriteParamsFile(runs[i], file)
    file.close()

    
# Run 

run_time = "02:00:00"
bank = "sentmat"
# threads = 32
nodes_per_run = 1
for run in runs:
    print(f"Running job {run}")
    lch.RunJob(run.run_location, f"viscous_runs_{run.muval}", nodes_per_run, run_time, bank, run.run_command)
    