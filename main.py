import numpy as np
import time

def print_hi(name):
    v = [1., 2., 3.]
    w = [1., 2., 3.]

    vnp = np.array(v)
    wnp = np.array(w)

    start = time.time()
    for i in range(1000):
        k = vnp[0]*wnp[0] + vnp[1]*wnp[1] + vnp[2]*wnp[2]
    print((time.time() - start) * 1e3)

    vnp = np.array(v)
    wnp = np.array(w)

    start = time.time()
    for i in range(1000):
        k = np.dot(vnp, wnp)
    print((time.time() - start) * 1e3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
