import pytest, pdb
from .state import State
import numpy as np
import itertools
from collections import namedtuple

from .supernodes import Node, Workflow
from ....interfaces import base as nib
from ....interfaces.utility import Function

#TODO: list as input should be also ok 

def fun1(a):
    return a**2

fun1_interf = Function(input_names=["a"],
                        output_names=["out"],
                        function=fun1)


def fun1a(a):
    return a + 10

fun1a_interf = Function(input_names=["a"],
                        output_names=["out"],
                        function=fun1a)


def fun2(a):
    import numpy as np
    pow = np.arange(4)
    return a**pow

fun2_interf = Function(input_names=["a"],
                        output_names=["out"],
                        function=fun2)


def fun2a(a):
    import numpy as np
    return a + np.array(4*[10])

fun2a_interf = Function(input_names=["a"],
                        output_names=["out"],
                        function=fun2a)



def fun3(a, b):
    return a * b

fun3_interf = Function(input_names=["a", "b"],
                        output_names=["out"],
                        function=fun3)


@pytest.mark.parametrize("inputs_dict, fun, expected_order, expected_output", [
        ({"a": np.array([3, 4, 5])}, "fun1_interf", ["a"],
         [("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25)]),
        ({"a": np.array([[3, 4], [5, 6]])}, "fun1_interf", ["a"],
         [("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25), ("state(a=6)", 36)]),
        ({"a": np.array([3, 4, 5])}, "fun2_interf", ["a"],
         [("state(a=3)", [1, 3, 9, 27]), ("state(a=4)", [1, 4, 16, 64]), ("state(a=5)", [1, 5, 25, 125])])

])
def test_workflow_singlenode_1(inputs_dict, fun, expected_order, expected_output):
    """testing workflow witha a single node provided in the __init__"""
    nn = Node(inputs=inputs_dict, mapper="a", interface=eval(fun),
              name="single_node_1")
    wf = Workflow(nodes=[nn], name="workflow_1")
    wf.run()

    state = namedtuple("state", expected_order)

    for (i, out) in enumerate(wf.nodes[0].result):
        assert out[0] == eval(expected_output[i][0]) # checking state values
        assert (out[1].out == expected_output[i][1]).all() # assuming that output value is an array (all() is used)



@pytest.mark.parametrize("inputs_dict, fun, expected_order, expected_output", [
        ({"a": np.array([3, 4, 5])}, "fun1_interf", ["a"],
         [("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25)]),
        ({"a": np.array([[3, 4], [5, 6]])}, "fun1_interf", ["a"],
         [("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25), ("state(a=6)", 36)]),
        ({"a": np.array([3, 4, 5])}, "fun2_interf", ["a"],
         [("state(a=3)", [1, 3, 9, 27]), ("state(a=4)", [1, 4, 16, 64]), ("state(a=5)", [1, 5, 25, 125])])

])
def test_workflow_singlenode_1a(inputs_dict, fun, expected_order, expected_output):
    """testing workflow witha a single node provided with add_nodes"""
    nn = Node(inputs=inputs_dict, mapper="a", interface=eval(fun),
              name="single_node_1")
    wf = Workflow(name="workflow_1a")
    wf.add_nodes([nn])
    wf.run()

    state = namedtuple("state", expected_order)

    for (i, out) in enumerate(wf.nodes[0].result):
        assert out[0] == eval(expected_output[i][0]) # checking state values
        assert (out[1].out == expected_output[i][1]).all() # assuming that output value is an array (all() is used)


@pytest.mark.parametrize("inputs_dict, fun1, fun2, expected_order, expected_output", [
        ({"a": np.array([3, 4, 5])}, "fun1_interf", "fun1a_interf", ["a"],
         [[("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25)],
          [("state(a=3)", 13),("state(a=4)", 14), ("state(a=5)", 15)]]),
         ({"a": np.array([3, 4, 5])}, "fun2_interf", "fun2a_interf", ["a"],
         [[("state(a=3)", [1, 3, 9, 27]), ("state(a=4)", [1, 4, 16, 64]), ("state(a=5)", [1, 5, 25, 125])],
          [("state(a=3)", [13, 13, 13, 13]), ("state(a=4)", [14, 14, 14, 14]), ("state(a=5)", [15, 15, 15, 15])]])
])
def test_workflow_not_connected_nodes_1(inputs_dict, fun1, fun2, expected_order, expected_output):
    """testing workflow with two nodes that have no connections"""
    nn1 = Node(inputs=inputs_dict, mapper="a", interface=eval(fun1),
              name="node_1")
    nn2 = Node(inputs=inputs_dict, mapper="a", interface=eval(fun2),
              name="node_2")
    wf = Workflow(name="workflow_1a")
    wf.add_nodes([nn1, nn2])
    wf.run()

    state = namedtuple("state", expected_order)

    for ni in range(2):
        for (i, out) in enumerate(wf.nodes[ni].result):
            assert out[0] == eval(expected_output[ni][i][0]) # checking state values
            assert (out[1].out == expected_output[ni][i][1]).all() # assuming that output value is an array (all() is used)


@pytest.mark.parametrize("inputs_dict, fun1, fun2, expected_order, expected_output", [
        ({"a": np.array([3, 4, 5])}, "fun1_interf", "fun1a_interf", ["a"],
         [[("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25)],
          [("state(a=3)", 19),("state(a=4)", 26), ("state(a=5)", 35)]]),
         ({"a": np.array([3, 4])}, "fun2_interf", "fun1a_interf", ["a"],
         [[("state(a=3)", [1, 3, 9, 27]), ("state(a=4)", [1, 4, 16, 64])],
          [("state(a=3)", [11, 13, 19, 37]), ("state(a=4)", [11, 14, 26, 74])]])
])
def test_workflow_connected_nodes_1(inputs_dict, fun1, fun2, expected_order, expected_output):
    """testing workflow with two nodes that have no connections"""
    nn1 = Node(inputs=inputs_dict, mapper="a", interface=eval(fun1),
              name="node_1")
    nn2 = Node(mapper="a", interface=eval(fun2),
              name="node_2")
    wf = Workflow(name="workflow_1a")
    wf.add_nodes([nn1, nn2])
    wf.connect(nn1, "out", nn2, "a")
    wf.run()

    state = namedtuple("state", expected_order)
    for ni in range(2):
        for (i, out) in enumerate(wf.nodes[ni].result):
            assert out[0] == eval(expected_output[ni][i][0]) # checking state values
            assert (out[1].out == expected_output[ni][i][1]).all() # assuming that output value is an array (all() is used)


@pytest.mark.parametrize("inputs_dict, functions, mappers, expected_order, expected_output", [
        ([{"a": np.array([3, 4, 5])}, {"b": np.array([1, 2, 3])}], ["fun1_interf", "fun3_interf"],
         [["a"], ("a", "b")], [["a"], ["a", "b"]],
         [[("state(a=3)", 9),("state(a=4)", 16), ("state(a=5)", 25)],
          [("state(a=3, b=1)", 9),("state(a=4, b=2)", 32), ("state(a=5, b=3)", 75)]]),
        ([{"a": np.array([3, 4, 5])}, {"b": np.array([1, 2, 3])}], ["fun1_interf", "fun3_interf"],
         [["a"], ["a", "b"]], [["a"], ["a", "b"]],
         [[("state(a=3)", 9), ("state(a=4)", 16), ("state(a=5)", 25)],
           [("state(a=3, b=1)", 9), ("state(a=3, b=2)", 18), ("state(a=3, b=3)", 27),
            ("state(a=4, b=1)", 16), ("state(a=4, b=2)", 32), ("state(a=4, b=3)", 48),
            ("state(a=5, b=1)", 25), ("state(a=5, b=2)", 50), ("state(a=5, b=3)", 75)]]),
        ([{"a": np.array([3, 4, 5])}, {"b": np.array([1, 2, 3])}], ["fun1_interf", "fun3_interf"],
         [["a"], ["a"]], [["a"], ["a", "b"]],
         [[("state(a=3)", 9), ("state(a=4)", 16), ("state(a=5)", 25)],
         [("state(a=3, b=np.array([1, 2, 3]))", np.array([9, 18, 27])),
          ("state(a=4, b=np.array([1, 2, 3]))", np.array([16, 32, 48])),
          ("state(a=5, b=np.array([1, 2, 3]))", np.array([25, 50, 75]))]])
])
def test_workflow_connected_nodes_2(inputs_dict, functions, mappers, expected_order, expected_output):
    """testing workflow with two nodes and a mapper"""
    nn1 = Node(inputs=inputs_dict[0], mapper=mappers[0], interface=eval(functions[0]),
              name="node_1")
    nn2 = Node(mapper=mappers[1], inputs=inputs_dict[1], interface=eval(functions[1]),
              name="node_2")
    wf = Workflow(name="workflow_1a")
    wf.add_nodes([nn1, nn2])
    wf.connect(nn1, "out", nn2, "a")
    wf.run()

    for ni in range(2):
        state = namedtuple("state", expected_order[ni])
        for (i, out) in enumerate(wf.nodes[ni].result):
            assert out[0].a == eval(expected_output[ni][i][0]).a
            if ni == 1:
                assert (out[0].b == eval(expected_output[ni][i][0]).b).all()
            assert (out[1].out == expected_output[ni][i][1]).all() # assuming that output value is an array (all() is used)
