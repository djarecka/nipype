import pytest, pdb
from .state import State
import numpy as np
import itertools
from collections import namedtuple


@pytest.mark.parametrize("mapper, mapper_rpn, input, axis_for_inp, inp_for_axis, ndim",[
        ("d", ["d"], {"d":np.array([3,4,5])}, {'d': [0]}, [["d"]], 1),

        ("d", ["d"], {"d":np.array([[3,4],[5,6]])}, {'d': [0, 1]}, [["d"], ["d"]], 2),

        (('d', 'r'), ['d', 'r', '.'], {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, 
         {'r': [0], 'd': [0]}, [["r", "d"]], 1),

        (('d', 'r'), ['d', 'r', '.'], {"d":np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])}, 
         {'r': [0, 1], 'd': [0, 1]}, [["r", "d"], ["r", "d"]], 2),

        ((("d", "r"), "e"), ['d', 'r', '.', "e", "."], 
         {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2,3])}, 
         {'r': [0], 'e': [0], 'd': [0]}, [["r", "d", "e"]], 1),

        (("d", ("r", "e")), ['d', 'r', "e", ".", "."], 
         {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2,3])}, 
         {'r': [0], 'e': [0], 'd': [0]}, [["r", "d", "e"]], 1),

        (["d", "r"], ['d', 'r', '*'], {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, 
         {'r': [1], 'd': [0]}, [["d"], ["r"]], 2),

        ([('d', 'r'), "e"], ['d', 'r', '.', "e", "*"], 
         {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2,3])}, 
         {'r': [0], 'e': [1], 'd': [0]}, [["r", "d"], ["e"]], 2),

        (['d', ('r', 'e')], ['d', 'r', "e", ".", "*"], 
         {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2,3])}, 
         {'r': [1], 'e': [1], 'd': [0]}, [["d"], ["r", "e"]], 2),

        (('d', ['e', 'r']), ['d', 'e', 'r', '*', '.'], 
         {"d":np.array([[3,4,5], [3,4,5]]), "r":np.array([1,2,3]), "e":np.array([1,1])}, 
         {'r': [1], 'e': [0], 'd': [0, 1]}, [["d", "e"], ["d", "r"]], 2),

        ((['e', ('r', 'w')], 'd'), ['e', 'r', 'w', '.', '*', 'd', '.'], 
         {"d":np.array([[3,4,5], [3,4,5]]), "r":np.array([1,2,3]), "e":np.array([1,1]), 
          'w':np.array([1,2,3])}, 
         {'r': [1], 'e': [0], 'd': [0, 1], 'w': [1]}, [["d", "e"], ["d", "r", "w"]], 2),

        (['d', ('r', 'w')], ['d', 'r', 'w', '.', '*'], 
         {"d":np.array([[3,4],[5,6]]), 'w':np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])}, 
         {'r': [2, 3], 'd': [0, 1], 'w': [2, 3]}, [["d"], ["d"], ["r", "w"], ["r", "w"]], 4),

        ([('d', 'r'), 'w'], ['d', 'r', '.', 'w', '*'], 
         {"d":np.array([[3,4],[5,6]]), 'w':np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])}, 
         {'r': [0, 1], 'd': [0, 1], 'w': [2, 3]}, [["d", "r"], ["d", "r"], ["w"], ["w"]], 4)
])
def test_mapping_axis(mapper, mapper_rpn, input, axis_for_inp, inp_for_axis, ndim):
    """testing if State() create a proper mapper representation,
        and connect proper variables with proper axes"""
    st = State(state_inputs=input, mapper=mapper)

    assert st._mapper_rpn == mapper_rpn
    assert st.axis_for_input == axis_for_inp 
    assert st.ndim == ndim

    for i, inp in enumerate(inp_for_axis):
        assert sorted(st.input_for_axis[i]) == sorted(inp) 
    #pdb.set_trace()


@pytest.mark.parametrize("mapper, input, expected_order, expected_state_values",[
        (('d', 'r'), {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, ["d", "r"], 
         ["state(d=3, r=1)", "state(d=4, r=2)", "state(d=5, r=3)"]),

        (('r', 'd'), {"d":np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])}, ['d', 'r'],
         ["state(r=1, d=3)", "state(r=2, d=4)", "state(r=3, d=5)", "state(r=3, d=6)"]),

        ((("d", "r"), "e"), {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2,3])},
         ['d', 'e', 'r'],
         ["state(d=3, r=1, e=1)", "state(d=4, r=2, e=2)", "state(d=5, r=3, e=3)"]),

        (["d", "r"], {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, ['d', 'r'], 
         ["state(d=3, r=1)", "state(d=3, r=2)", "state(d=3, r=3)", 
          "state(d=4, r=1)", "state(d=4, r=2)", "state(d=4, r=3)", 
          "state(d=5, r=1)", "state(d=5, r=2)", "state(d=5, r=3)"]),

        ([('d', 'r'), "e"], {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2])},
         ["d", "e", "r"],
         ["state(d=3, r=1, e=1)", "state(d=3, r=1, e=2)", "state(d=4, r=2, e=1)",
          "state(d=4, r=2, e=2)", "state(d=5, r=3, e=1)", "state(d=5, r=3, e=2)"]),

        (['d', ('r', 'e')], {"d":np.array([3,4]), "r":np.array([1,2,3]), "e":np.array([1,2,3])}, 
         ["d", "e", "r"],
         ["state(d=3, r=1, e=1)", "state(d=3, r=2, e=2)", "state(d=3, r=3, e=3)",
          "state(d=4, r=1, e=1)", "state(d=4, r=2, e=2)", "state(d=4, r=3, e=3)"]),

        (('d', ['e', 'r']), 
         {"d":np.array([[3,4,5], [6,7,8]]), "r":np.array([1,2,3]), "e":np.array([1,2])}, ['d', 'e', 'r'],
         ["state(d=3, r=1, e=1)", "state(d=4, r=2, e=1)", "state(d=5, r=3, e=1)",
          "state(d=6, r=1, e=2)", "state(d=7, r=2, e=2)", "state(d=8, r=3, e=2)"]),

        (['d', ('w', 'r')],
         {"d":np.array([[3,4],[5,6]]), "r":np.array([1,2]), "w":np.array([3,4])}, ['d', 'r', 'w'],
         ["state(d=3, w=3, r=1)", "state(d=3, w=4, r=2)", "state(d=4, w=3, r=1)", 
          "state(d=4, w=4, r=2)", "state(d=5, w=3, r=1)", "state(d=5, w=4, r=2)", 
          "state(d=6, w=3, r=1)", "state(d=6, w=4, r=2)"])
         ])
def test_state_values(mapper, input, expected_order, expected_state_values):
    """testing state_values(ind) method, should return the correct values for chosen index"""
    st = State(state_inputs=input, mapper=mapper)
    state = namedtuple("state", expected_order) #used to evaluate expected_state_value

    for i, ind in enumerate(itertools.product(*st.all_elements)):
        state_dict = st.state_values(ind)
        assert state_dict == eval(expected_state_values[i])


@pytest.mark.parametrize("mapper, input, elements, expected_order, expected_state_values",[
        (('d', 'r'), {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, ["st[0]", "st[2]"], 
         ["d", "r"], ["state(d=3, r=1)", "state(d=5, r=3)"]),

        (('r', 'd'), {"d":np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])},
         ["st[0,1]", "st[1,0]"], ["d", "r"], ["state(r=2, d=4)", "state(r=3, d=5)"]),

        (["d", "r"], {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, 
         ["st[0,1]", "st[2,1]", "st[1,1]"], ["d", "r"],
         ["state(d=3, r=2)", "state(d=5, r=2)", "state(d=4, r=2)"]),
        ])
def test_state_ind(mapper, input, elements, expected_order, expected_state_values):
    """testing if I can access the correct elements from State()
         using __getitem__"""
    st = State(state_inputs=input, mapper=mapper)
    state = namedtuple("state", expected_order)

    for i, el in enumerate(elements):
        assert eval(el) == eval(expected_state_values[i])


@pytest.mark.parametrize("mapper, input, elements",[
        (('d', 'r'), {"d":np.array([3,4]), "r":np.array([1,2])}, ["st[3]", "st[0,1]"]),

        (('d', 'r'), {"d":np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])},
         ["st[2,1]", "st[1,3]", "st[1,1,1]"]),

        (["d", "r"], {"d":np.array([3,4,5]), "r":np.array([1,2,3])},
         ["st[0,3]", "st[3,1]", "st[1,1,1]"]), 
        ])
def test_state_indexerror(mapper, input, elements):
    """testing if incorrect indexing os State() will raise an error"""
    st = State(state_inputs=input, mapper=mapper)

    for i, el in enumerate(elements):
        with pytest.raises(IndexError):
            eval(el) 
