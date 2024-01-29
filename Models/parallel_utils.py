import time
from concurrent.futures import ProcessPoolExecutor
import sys
from multiprocessing.reduction import ForkingPickler, AbstractReducer
import multiprocessing as mp

from tqdm import tqdm

CHUNK_SIZE = 16


class ForkingPickler4(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)

def dump(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump

def run_parallel(func, args, chunksize=10):
    ctx = mp.get_context()
    ctx.reducer = Pickle4Reducer()

    executor = ProcessPoolExecutor()
    num_args = len(args)
    results = [i for i in tqdm(executor.map(func,*list(zip(*args)),chunksize=chunksize),total=num_args)]
    return results
