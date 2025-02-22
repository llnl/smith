import numpy as np
import sys
import os
from dataclasses import dataclass
import subprocess



def GetHostName():
    my_hostname = str(subprocess.check_output(['hostname']))
    if "ruby" in my_hostname:
        my_hostname = "ruby"
    elif "lassen" in my_hostname:
        my_hostname = "lassen"
    elif "rzhound" in my_hostname:
        my_hostname = "rzhound"
    elif "tuo" in my_hostname:
        my_hostname = "tuo"
    return my_hostname

my_hostname = GetHostName() 
print(f"Running on: {my_hostname}")


def create_folder(folder_path):
    """
    Creates a folder if it does not already exist.
    
    :param folder_path: The path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")




def WriteSBATCHFile(filename, job_name, output_location, nodes, run_time, bank, command):
    file = open(filename, "w")
    file.write("#!/bin/bash\n")    
    file.write(f"#SBATCH --job-name={job_name}\n")
    file.write(f"#SBATCH --time={run_time}\n")
    file.write(f"#SBATCH --nodes={nodes}\n")
    file.write(f"#SBATCH --account={bank}\n")
    file.write(f"#SBATCH -o {output_location}\n")
    file.write(f"NumProcs=$(($SLURM_NNODES * $SLURM_CPUS_ON_NODE))\n")
    file.write(f"srun -n $NumProcs {command}")
    file.close()

def WriteBSUBFile(filename, job_name, output_location, nodes, run_time, bank, command):
    file = open(filename, "w")
    file.write(f"#BSUB -nnodes {nodes}\n")
    file.write("#BSUB -q pbatch\n")
    file.write(f"#BSUB -u ${{USER}}@llnl.gov\n")
    file.write(f"#BSUB -W {run_time}\n")
    file.write(f"#BSUB -o {output_location}\n")
    file.write(f"lrun -N{nodes} -T1 {command}")


def RunJob(foldername, job_name, nodes, run_time, bank, command):
    if my_hostname == "ruby":
        file_name = os.path.join(foldername, "sbatch.sbatch")
        WriteSBATCHFile(file_name, job_name, foldername, nodes, run_time, bank, command)
        os.system(f"sbatch {file_name}")
    elif my_hostname == "lassen":
        file_name = os.path.join(foldername, "job.bsub")
        WriteBSUBFile(file_name, job_name, foldername, nodes, run_time, bank, command)
        os.system(f"bsub {file_name}")
    elif my_hostname == "rzhound":
        file_name = os.path.join(foldername, "sbatch.sbatch")
        WriteSBATCHFile(file_name, job_name, foldername, nodes, run_time, bank, command)
        os.system(f"sbatch {file_name}")
    elif my_hostname == "tuo":
        print("not implemented yet")
    else:
        exit("error not implmented job runner")

