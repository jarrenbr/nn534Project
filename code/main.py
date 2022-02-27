import sys

def main(argv):
    arguments = argv[1:]
    for arg in arguments:
        print(arg)

if __name__ == "__main__":
    main(sys.argv)