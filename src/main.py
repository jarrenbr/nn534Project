import sys
import os

def main(args):
    allArgs = []
    for index, arg in enumerate(args):
        if index > 0:
            allArgs.append(arg)
    execute(allArgs)

def execute(args):
    for arg in args:
        print("Executing: " + arg)
        

if __name__ == "__main__":
    main(sys.argv)