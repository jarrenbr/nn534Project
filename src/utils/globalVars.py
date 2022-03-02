import sys

def get_debug_flag():
    return sys.gettrace() is not None

DEBUG = get_debug_flag()

DATA_AMT = int(1e5) if DEBUG else None