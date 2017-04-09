import pytest, pdb
from state import State
import numpy as np
import itertools



@pytest.mark.parametrize("mapper, mapper_rpn, input, axis_for_inp, inp_for_axis, ndim",[
        (('d', 'r'), ['d', 'r', '.'], {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, 
         {'r': [0], 'd': [0]}, [["r", "d"]], 1),

        (('d', 'r'), ['d', 'r', '.'], {"d":np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])}, {'r': [0, 1], 'd': [0, 1]}, [["r", "d"], ["r", "d"]], 2),

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
    st = State(state_inputs=input, mapper=mapper)

    assert st.mapper_rpn == mapper_rpn
    assert st.axis_for_input == axis_for_inp 
    assert st.ndim == ndim

    for i, inp in enumerate(inp_for_axis):
        assert sorted(st.input_for_axis[i]) == sorted(inp) 



@pytest.mark.parametrize("mapper, input, state_values_list",[
        (('d', 'r'), {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, 
         [{'d': 3, 'r': 1}, {'d': 4, 'r': 2}, {'d': 5, 'r': 3}]),

        (('d', 'r'), {"d":np.array([[3,4],[5,6]]), "r":np.array([[1,2],[3,3]])}, 
         [{'d': 3, 'r': 1}, {'d': 4, 'r': 2}, {'d': 5, 'r': 3}, {'d': 6, 'r': 3}]),

        ((("d", "r"), "e"), {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2,3])}, 
         [{'d': 3, 'r': 1, 'e': 1}, {'d': 4, 'r': 2, 'e': 2}, {'d': 5, 'r': 3, 'e': 3}]),

        (["d", "r"], {"d":np.array([3,4,5]), "r":np.array([1,2,3])}, 
         [{'d': 3, 'r': 1}, {'d': 3, 'r': 2}, {'d': 3, 'r': 3}, {'d': 4, 'r': 1}, {'d': 4,'r': 2}, 
          {'d': 4, 'r': 3}, {'d': 5, 'r': 1}, {'d': 5, 'r': 2}, {'d': 5, 'r': 3}]),

        ([('d', 'r'), "e"], {"d":np.array([3,4,5]), "r":np.array([1,2,3]), "e":np.array([1,2])}, 
         [{'d': 3, 'r': 1, 'e': 1}, {'d': 3, 'r': 1, 'e': 2}, {'d': 4, 'r': 2, 'e': 1},
          {'d': 4, 'r': 2, 'e': 2}, {'d': 5, 'r': 3, 'e': 1}, {'d': 5, 'r': 3, 'e': 2}]),

        (['d', ('r', 'e')], {"d":np.array([3,4]), "r":np.array([1,2,3]), "e":np.array([1,2,3])}, 
         [{'d': 3, 'r': 1, 'e': 1}, {'d': 3, 'r': 2, 'e': 2}, {'d': 3, 'r': 3, 'e': 3},
          {'d': 4, 'r': 1, 'e': 1}, {'d': 4, 'r': 2, 'e': 2}, {'d': 4, 'r': 3, 'e': 3}]),

        (('d', ['e', 'r']), 
         {"d":np.array([[3,4,5], [6,7,8]]), "r":np.array([1,2,3]), "e":np.array([1,2])}, 
         [{'d': 3, 'r': 1, 'e': 1}, {'d': 4, 'r': 2, 'e': 1}, {'d': 5, 'r': 3, 'e': 1},
          {'d': 6, 'r': 1, 'e': 2}, {'d': 7, 'r': 2, 'e': 2}, {'d': 8, 'r': 3, 'e': 2}]),

        (['d', ('r', 'w')],
         {"d":np.array([[3,4],[5,6]]), "r":np.array([1,2]), "w":np.array([3,4])}, 
         [{'d': 3, 'r': 1, 'w': 3}, {'d': 3, 'r': 2, 'w': 4}, {'d': 4, 'r': 1, 'w': 3},
          {'d': 4, 'r': 2, 'w': 4}, {'d': 5, 'r': 1, 'w': 3}, {'d': 5, 'r': 2, 'w': 4},
          {'d': 6, 'r': 1, 'w': 3}, {'d': 6, 'r': 2,  'w': 4}])
         ])
def test_state_values(mapper, input, state_values_list):
    st = State(state_inputs=input, mapper=mapper)
    for i, ind in enumerate(itertools.product(*st.all_elements)):
        state_dict = st.state_values(ind)
        assert state_dict == state_values_list[i]
