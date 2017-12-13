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


@pytest.mark.parametrize("inputs_dict, expected_order, expected_output", [
        ({"a": np.array([3, 4, 5])}, ["a"], [("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25)]),
        # do we want to allow 2D inputs a, when mapper="a"?
        ({"a": np.array([[3, 4], [5, 6]])}, ["a"], 
         [("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25), ("state(a=6)", 36)])
        ])
def test_singlenode_1(inputs_dict, expected_order, expected_output):
    """testing a single node for function that returns only one value"""
    nn = Node(inputs=inputs_dict, mapper="a", interface=fun1_interf,
               name="single_node_1")
    nn.run()
    state = namedtuple("state", expected_order)

    for (i, out) in enumerate(nn.result):
        assert out[0] == eval(expected_output[i][0]) # checking state values
        assert (out[1] == expected_output[i][1]).all() # assuming that output value is an array (all() is used)


@pytest.mark.parametrize("inputs_dict, expected_order, expected_output", [
        ({"a": np.array([3, 4, 5])}, ["a"],
         [("state(a=3)", [1, 3, 9, 27]),("state(a=4)", [1, 4, 16, 64]), ("state(a=5)", [1, 5, 25, 125])])
        ])
def test_singlenode_2(inputs_dict, expected_order, expected_output):
    """testing a single node for a function that returns a list/array"""
    nn  = Node(inputs=inputs_dict, mapper="a", interface=fun2_interf,
               name="single_node_2")
    nn.run()
    state = namedtuple("state", expected_order)
    
    for (i, out) in enumerate(nn.result):
        assert out[0] == eval(expected_output[i][0])
        assert (out[1] == expected_output[i][1]).all()



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
    """testing for a single node with two input fields, scalar mapping"""
    nn = Node(interface=fun3_interf, name="single_node_3", mapper=("a", "b"),
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
    """testing for a single node with two input fields, outer mapping"""
    nn = Node(interface=fun3_interf, name="single_node_4", mapper=['a','b'],
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
    """testing if error is raised when the inputs doesn't meet the mapper"""
    with pytest.raises(Exception) as excinfo:
        nn = Node(interface=fun3_interf, name="single_node_exception",
                  mapper=('a','b'), inputs=inputs_dict)
        nn.run()
    assert "should have the same size" in str(excinfo.value)


def test_single_node_wrong_key():
    """testing if the wrror is raised when inputs key don't match mapper"""
    # TODO: should i specify error?
    with pytest.raises(KeyError):
        nn = Node(interface=fun3_interf, name="single_node_exception",
                  mapper=('a','b'), inputs={"a":[3], "c":[0]})
        nn.run()

