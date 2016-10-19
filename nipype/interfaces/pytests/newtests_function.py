# -*- coding: utf-8 -*- 
from __future__ import print_function, unicode_literals
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-                    
# vi: set ft=python sts=4 ts=4 sw=4 et:                                                      
import os
import pytest 

from nipype.interfaces import utility
import nipype.pipeline.engine as pe

import pdb

def adding(val1, val2):
    return val1 + val2


def test_function1(tmpdir):
    """ Testing if Function works when input_names is the same as function arguments"""
    os.chdir(str(tmpdir))
    fun = utility.Function(input_names = ["val1", "val2"],
                          output_names = ["out"],
                          function = adding)
    fun.inputs.val1 = 3
    fun.inputs.val2 = 2
    run = fun.run()
    out = run.outputs.out
    assert out == 5


def test_function2(tmpdir):
    """ Testing if Function raises an exception when input_names is not a function argument"""
    os.chdir(str(tmpdir))

    with pytest.raises(Exception):
        fun = utility.Function(input_names = ["val1", "_val2"],
                               output_names = ["out"],
                               function = adding)


def test_function3(tmpdir):
    os.chdir(str(tmpdir))
    with pytest.raises(TypeError):  #powinno tu sie wywolywac, a nie przy run dopiero  
        fun = utility.Function(input_names = ["val1"],
                               output_names = ["out"],
                               function = adding)


#zmienic tak f-cje aby dzialalo
def test_function4(tmpdir):
    os.chdir(str(tmpdir))

    fun = utility.Function(output_names = ["out"],
                          function = adding)
    fun.inputs.inp = 3
    run= fun.run()
    out= run.outputs.out
    #pdb.set_trace()                                                                                   


def adding_args(*inp):
    sum= 0
    for i in inp:
        sum += i
    return sum

def atest_function5(tmpdir):
    os.chdir(str(tmpdir))

    fun = utility.Function(input_names = ["inp1", "inp2"],
                          output_names = ["out"],
                          function = adding_args)
    pdb.set_trace()
    fun.inputs.inp1 = 3
    fun.inputs.inp2 = 2
    run= fun.run()
    out= run.outputs.out
    #pdb.set_trace()   
