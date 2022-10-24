import sys

def get_debug_flag():
    return sys.gettrace() is not None

DEBUG = get_debug_flag()

DATA_AMT = None #int(2**14) if DEBUG else None