#!/usr/bin/python
# -*- coding: UTF-8 -*-

#TODO: use argparse to parse passed params (see https://github.com/tyiannak/pyAudioAnalysis/blob/master/audioAnalysis.py)
import argparse

import sys 
import os
import re
import json

from loader import Loader

# RED   = "\033[1;31m"  
# BLUE  = "\033[1;34m"
# CYAN  = "\033[1;36m"
# GREEN = "\033[0;32m"
# RESET = "\033[0;0m"
# BOLD    = "\033[;1m"
# REVERSE = "\033[;7m"

def load_json_data(input_data):
    loader = Loader(input_data)
    loader.print_data()

def print_help():
    print "Usage:"
    print "1. python cmup.py load                           - loads input data"
    print "   Arguments: --file=<PATH_TO_FILE_WITH_DATA> | --data=<DATA_IN_JSON_FORMAT>"
    print "   Input data should be defined in JSON format."
    print "   Structure of input data: \n", \
          "     {\n", \
          "         \"ID1\":[<path_to_music_file1>, <path_to_music_file2>, ..., <path_to_music_fileN>],\n", \
          "         \"ID2\":[<path_to_music_file1>, <path_to_music_file2>, ..., <path_to_music_fileN>],\n", \
          "         ...,\n", \
          "         \"IDN\":[<path_to_music_file1>, <path_to_music_file2>, ..., <path_to_music_fileN>]\n", \
          "     }"
    print "2. python cmup.py calculate_features             - runs feature calculating for all stored user IDs"
    print "   Possible arguments: --ids=<ID1,ID2,..>        - runs feature calculating on specified user IDs"
    print "3. python cmup.py update_features                - runs feature updating for all stored user IDs"
    print "   Possible arguments: --ids=<ID1,ID2,..>        - runs feature updating for specified user IDs"
    print "4. python cmup.py run_classifier                 - runs classifier on all stored user IDs"
    print "5. python cmup.py get_group -ids=<ID1,ID2,..>    - returns classified group whom defined users belongs to (JSON)"
    print "6. python cmup.py get_result                     - returns result of classification (JSON)"
    print "   Arguments: --file=<PATH_TO_RESULT_FILE>"
    print "   If no arguments defined prints result into stdout"

try:
    sys.argv[1]
except IndexError:
    print_help()
    sys.exit()

files=[]
is_file = 0 
file_name = ""
directory=""

try:
    m1 = re.search("load", sys.argv[1])
    m2 = re.search("2 (.*)", sys.argv[1])
    if m1:
        m1 = re.search("--file=(.*)", sys.argv[2])
        if m1:
            print "file => " + m1.group(1)
            file_name = os.path.abspath(m1.group(1))
            if not os.path.exists(file_name):
                print "File '" + file_name + "' doesn't exist!"
                sys.exit()

            data = ''
            with open(file_name, 'r') as datafile:
                data=datafile.read().replace('\n', '')

            load_json_data(data)
            sys.exit()

        m1 = re.search("--data=(.*)", sys.argv[2])
        if m1:
            load_json_data(m1.group(1))
            sys.exit()

        raise Exception('invalid params')

    elif m2:
        print "2"
        #directory=m2.group(1)
        #for f in os.listdir(m2.group(1)):
        #    f = os.path.join(m2.group(1), f)
        #    if os.path.isfile(f) and re.search("\.mp3", f):
        #        files.append(f)
    else:
        raise Exception('invalid params')
except IndexError:
    print '\033[1;31mInvalid params passed! \033[0;0m'
    print_help()
    sys.exit()

print "OK1"
