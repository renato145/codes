from tqdm import tqdm
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

# Review
def tqdm_parallel_map(fn, x, workers=8):
    '''Parallel execute and track with tqdm'''
    with ThreadPoolExecutor(workers) as ex:
        futures_list = [ex.submit(fn, i) for i in x]
    
    for f in tqdm(futures.as_completed(futures_list), total=len(futures_list)):
        yield f.result()