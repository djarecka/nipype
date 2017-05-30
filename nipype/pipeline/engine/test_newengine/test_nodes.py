import pytest, pdb
from state import State
import numpy as np
import itertools
from collections import namedtuple

from supernodes import Node
from ....interfaces import base as nib


# TODO: check for a simpler mapper and no mapper
#TODO: list as input should be also ok 


def fun3(a, b):
    return a * b


@pytest.mark.parametrize("inputs_dict, expected_order, expected_output", [
        # mapper is a.b
        # result[0] is for a[0]=3 and b[0]=0, etc.
        ({"a":np.array([3, 1, 8]), "b":np.array([0, 1, 2])}, ["a", "b"],
         [("state(a=3, b=0)", 0), ("state(a=1, b=1)", 1), ("state(a=8, b=2)", 16)]),
        #({"a":np.array([3, 1, 8]), "b":np.array([2])}, ["a", "b"], #dj: should this be allowed?
        # [("state(a=3, b=2)", 6), ("state(a=1, b=2)", 2), ("state(a=8, b=2)", 16)]),
        ({"a":np.array([[3, 1], [6, 8]]), "b":np.array([[0, 1], [2, 3]])}, ["a", "b"],
         [("state(a=3, b=0)", 0), ("state(a=1, b=1)", 1), ("state(a=6, b=2)", 12), 
          ("state(a=8, b=3)", 24)])
        ])
def test_single_node_3(inputs_dict, expected_order, expected_output):
    nn = Node(interface=fun3, name="single_node_3", mapper=("a", "b"), 
              inputs=inputs_dict)
    nn.run()
    state = namedtuple("state", expected_order)

    for (i, out) in enumerate(nn.result):
        assert out[0] == eval(expected_output[i][0]) 
        assert out[1] == expected_output[i][1] #dj: do i need ".all()"?

#TODO !!! should I keep the 2d array structure?? - see atest_single_node.py
#TODO: what should be the order of results
@pytest.mark.parametrize("inputs_dict, expected_order, expected_output", [
        # mapper is axb, so output is 2dimensional array (but results is flat for now)
        ({"a":np.array([3, 1]), "b":np.array([1, 2, 4])}, ["a", "b"],
         [("state(a=3, b=1)", 3), ("state(a=3, b=2)", 6), ("state(a=3, b=4)", 12),
          ("state(a=1, b=1)", 1), ("state(a=1, b=2)", 2), ("state(a=1, b=4)", 4)]),
         # a is 2dimensional array, so mapper axb is 3dimensional
        ({"a":np.array([[3, 1], [30, 10]]), "b":np.array([1, 2, 4])}, [ "a", "b"],
         [("state(a=3, b=1)", 3), ("state(a=3, b=2)", 6), ("state(a=3, b=4)", 12),
          ("state(a=1, b=1)", 1), ("state(a=1, b=2)", 2), ("state(a=1, b=4)", 4),
          ("state(a=30, b=1)", 30), ("state(a=30, b=2)", 60), ("state(a=30, b=4)", 120),
          ("state(a=10, b=1)", 10), ("state(a=10, b=2)", 20), ("state(a=10, b=4)", 40)]),
        ({"a":np.array([3, 1]), "b":np.array([2])}, ["a", "b"], 
         [("state(a=3, b=2)", 6), ("state(a=1, b=2)", 2)]),
        ])
def test_single_node_4(inputs_dict, expected_order, expected_output):
    nn = Node(interface=fun3, name="single_node_4", mapper=['a','b'],
              inputs=inputs_dict)
    nn.run()
    state = namedtuple("state", expected_order)

    for (i, out) in enumerate(nn.result):
        assert out[0] == eval(expected_output[i][0])
        assert (out[1] == expected_output[i][1]).all()


@pytest.mark.parametrize("inputs_dict", [
        {"a":np.array([[3, 1], [0,0]]), "b":np.array([1, 2, 0])},
        {"a":np.array([[3, 1], [0,0], [1, 1]]), "b":np.array([1, 2, 0])}, # think if this should work
        {"a":np.array([[3, 1, 1], [0,0, 0]]), "b":np.array([1, 2, 0])},  # think if this should work
        ])
def test_single_node_wrong_input(inputs_dict):
    with pytest.raises(Exception) as excinfo:
        nn = Node(interface=fun3, name="single_node_exception",
                  mapper=('a','b'), inputs=inputs_dict)
        nn.run()
    assert "should have the same size" in str(excinfo.value)


def test_single_node_wrong_key():
    # TODO: should i specify error?
    with pytest.raises(KeyError):
        nn = Node(interface=fun3, name="single_node_exception", 
                  mapper=('a','b'), inputs={"a":[3], "c":[0]})
        nn.run()

