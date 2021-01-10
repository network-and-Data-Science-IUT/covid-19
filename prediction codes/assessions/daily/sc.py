from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom

def Run(i):
    subprocess.call("python ./prediction.py "+str(i), shell=True)

def main():
    # loom = ProcessLoom(max_runner_cap = 8)
    for i in range(22):
        print(200 * '*')
        print(i)
        Run(i)
    subprocess.call("python ./errors.py", shell=True)
    #     print(i)
    #     loom.add_function(Run,[i])
    # loom.execute()


if __name__ == "__main__":

    main()