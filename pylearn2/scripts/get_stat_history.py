#!/usr/bin/env python
"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

def print_monitor(args):
    from pylearn2.utils import serial
    import gc
    model_path = args[0]
    key = args[1]
    model = serial.load(model_path)
    monitor = model.monitor
    del model
    gc.collect()
    channels = monitor.channels
    if not hasattr(monitor, '_epochs_seen'):
        print 'old file, not all fields parsed correctly'
    else:
        print 'epochs seen: ',monitor._epochs_seen
    print 'time trained: ',max(channels[key].time_record[-1] for key in
            channels)
    for i,val in enumerate(channels[key].val_record):
        print "{0}: {1}".format(i,val)


if __name__ == '__main__':
    import sys
    print_monitor(sys.argv[1:])
