"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly

##
def test():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    print("gyf:opt={}".format(opt))
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    print("gyf:model.netg={}".format(model.netg))
    # print("gyf:model.netd={}".format(model.netd))
    ##
    # TRAIN MODEL
    model.test()

if __name__ == '__main__':
    test()
