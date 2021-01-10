from sys import argv
import sys
import os
import subprocess
from pexecute.process import ProcessLoom

r = 7

def Run(i):
    subprocess.call("python ./prediction.py "+str(i), shell=True)

def main():
    for i in range(min((2 * (10 - r)) + 1, 7)):
        print(200 * '*')
        print(i)
        Run(i)



if __name__ == "__main__":

    main()