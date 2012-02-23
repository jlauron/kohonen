#!/usr/bin/env python
# encoding: utf-8
"""
main.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

import sys
import getopt

import algorithm

from test import *

help_message = '''
Support Vector Machine
'''

class Usage (Exception):
    def __init__ (self, msg):
        self.msg = msg

def main (argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
            raise Usage (msg)

        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage (help_message)

        test_kfold_linear ()
        test_kfold_gaussian ()
        test_kfold_polynomial ()

    except Usage, err:
        print >> sys.stderr, str (err.msg)
        return 1


if __name__ == "__main__":
    sys.exit (main ())
