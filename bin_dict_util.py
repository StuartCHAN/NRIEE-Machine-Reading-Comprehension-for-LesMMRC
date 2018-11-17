# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 21:31:05 2018
    https://stackoverflow.com/questions/19232011/convert-dictionary-to-bytes-and-back-again-python
@author: Lenovo
"""

import json

def dict_to_binary(the_dict):
    str = json.dumps(the_dict)
    binary = ' '.join(format(ord(letter), 'b') for letter in str)
    return binary;


def binary_to_dict(the_binary):
    jsn = ''.join(chr(int(x, 2)) for x in the_binary.split())
    d = json.loads(jsn)  
    return d;

#my_dict = {'key' : [1,2,3]}
#
#bin = dict_to_binary(my_dict)
#print bin
#
#dct = binary_to_dict(bin)
#print dct