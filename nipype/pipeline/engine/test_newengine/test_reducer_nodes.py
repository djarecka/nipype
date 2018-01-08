import pytest, pdb
from .state import State
import numpy as np
import itertools
from collections import namedtuple

from .supernodes import Node
from ....interfaces import base as nib
from ....interfaces.utility import Function

#TODO: list as input should be also ok 

def fun1(a):
    return a**2

fun1_interf = Function(input_names=["a"],
                        output_names=["out"],
                        function=fun1)


def fun2(a):
    import numpy as np
    pow = np.arange(4)
    return a**pow

fun2_interf = Function(input_names=["a"],
                        output_names=["out"],
                        function=fun2)


def fun3(a, b):
    return a * b

fun3_interf = Function(input_names=["a", "b"],
                        output_names=["out"],
                        function=fun3)


@pytest.mark.parametrize("inputs_dict, expected_order, expected_output, reducer", [
        ({"a": np.array([3, 4, 5])}, ["a"], [[("state(a=3)", 9)],[("state(a=4)", 16)], [("state(a=5)", 25)]], "a"),
        #do we want to allow 2D inputs a, when mapper="a"?
        ({"a": np.array([[3, 4], [5, 6]])}, ["a"], 
         [[("state(a=3)", 9)],[("state(a=4)", 16)], [("state(a=5)", 25)], [("state(a=6)", 36)]], "a"),
        ({"a": np.array([3, 4, 3])}, ["a"], [[("state(a=3)", 9), ("state(a=3)", 9)],[("state(a=4)", 16)]], "a"),
        ])
def test_singlenode_reducer_1(inputs_dict, expected_order, expected_output, reducer):
    nn  = Node(inputs=inputs_dict, mapper="a", interface=fun1_interf,
               name="single_node_1", reducer=reducer)
    nn.run()
    state = namedtuple("state", expected_order)
    for (ii, out_red) in enumerate(nn.result):
        #dj: problems wit set
        #assert out_red[0] == "{} = {}".format(reducer, set(inputs_dict[reducer].flatten())[ii])
        for (jj, out) in enumerate(out_red[1]):
            assert out[0] == eval(expected_output[ii][jj][0]) # checking state values
            assert (out[1].out == expected_output[ii][jj][1]).all() # assuming that output value is an array (all() is used)


@pytest.mark.parametrize("inputs_dict, expected_order, expected_output, reducer", [
        # mapper is axb, so output is 2dimensional array (but results is flat for now)
        ({"a":np.array([3, 1]), "b":np.array([1, 2, 4])}, ["a", "b"],
         [[("state(a=3, b=1)", 3), ("state(a=3, b=2)", 6), ("state(a=3, b=4)", 12)],
          [("state(a=1, b=1)", 1), ("state(a=1, b=2)", 2), ("state(a=1, b=4)", 4)]], "a"),
        ])
def test_singlenode_reducer_2(inputs_dict, expected_order, expected_output, reducer):
    nn = Node(interface=fun3_interf, name="single_node_4", mapper=['a','b'],
              inputs=inputs_dict, reducer=reducer)
    nn.run()
    state = namedtuple("state", expected_order)
    for (ii, out_red) in enumerate(nn.result):
        for (jj, out) in enumerate(out_red[1]):
            assert out[0] == eval(expected_output[ii][jj][0])
            assert (out[1].out == expected_output[ii][jj][1]).all()

