import time
import numpy as np      # for numpy
# import cupy as np     # for cupy

#
# Bayesian network "brute force" simulation
# - Introduction to Data Mining 2nd Edition (Tan, et al./)
#   Exercise 4-11 
#
def bayesian_network():
    ROWS = 10_000_000
    # R = np.random.rand(ROWS, 4, dtype=np.float32) # for cupy
    R = np.random.rand(ROWS, 4) # for numpy
    B = np.zeros((ROWS, 4), dtype=bool)

    # battery,fuel,gauge,start = 0,1,2,3
    B[:,0] = R[:,0] < 0.1   # battery: bad=True, good=False
    B[:,1] = R[:,1] < 0.2   # fuel: empty=True, not empty=False
    B[:,2] = np.where(B[:,0],   # gauge: empty=True, not empty = False
                    np.where(B[:,1], R[:,2] < 0.9, R[:,2] < 0.2),
                    np.where(B[:,1], R[:,2] < 0.8, R[:,2] < 0.1))
    B[:,3] = np.where(B[:,0],   # start: not start=True, start=False
                    np.where(B[:,1], R[:,3] < 1.0, R[:,3] < 0.9),
                    np.where(B[:,1], R[:,3] < 0.8, R[:,3] < 0.1))
    # (a)
    a = len(B[~B[:,0] & B[:,1] & B[:,2] & ~B[:,3]]) / len(B)
    # (b)
    b = len(B[B[:,0] & B[:,1] & ~B[:,2] & B[:,3]]) / len(B)
    # (c)
    c = len(B[B[:,0] & ~B[:,3]]) / len(B[B[:,0]])

    return a,b,c

def main():
    COUNT = 10
    answers = np.zeros((COUNT, 3), dtype=np.float32)
    start = time.time()
    for i in range(10):
        answers[i] = np.array(bayesian_network())
    print(time.time() - start)
    print(np.mean(answers, axis=0))

if __name__ == '__main__':
    main()