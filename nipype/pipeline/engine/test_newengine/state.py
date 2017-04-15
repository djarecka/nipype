import numpy as np
import itertools
import pdb

import auxiliary as aux

class State:
    def __init__(self, state_inputs, mapper=None):
        
        self.state_inputs = state_inputs

        self.mapper = mapper
        if self.mapper:
            # changing mapper (as in rpn), so I can read from left to right
            # e.g. if mapper=('d', ['e', 'r']), _mapper_rpn=['d', 'e', 'r', '*', '.']
            self._mapper_rpn = aux.mapper2rpn(self.mapper)
        
        # dictionary[key=input names] = list of axes related to
        # e.g. {'r': [1], 'e': [0], 'd': [0, 1]}
        # ndim - int, number of dimension for the "final array" (that is not created)
        self.axis_for_input, self.ndim = aux.mapping_axis(self.state_inputs, self._mapper_rpn)

        # list of inputs variable for each axis
        # e.g. [['e', 'd'], ['r', 'd']]
        # shape - list, e.g. [2,3]
        self.input_for_axis, self.shape = aux.converting_axis2input(self.state_inputs, 
                                                                    self.axis_for_input, self.ndim)
        
        # list of all possible indexes in each dim, will be use to iterate
        # e.g. [[0, 1], [0, 1, 2]]
        self.all_elements = [range(i) for i in self.shape]

    def __getitem__(self, key):
        if type(key) is int:
            key = (key,)
        return self.state_values(key)


    # it should be probably move to auxiliary 
    def _converting_axis2input(self):
        for i in range(self.ndim):
            self.input_for_axis.append([])
            self.shape.append(0)

        for inp, axis in self.axis_for_input.items():
            for (i, ax) in enumerate(axis):
                self.input_for_axis[ax].append(inp)
                self.shape[ax] =  self.state_inputs[inp].shape[i]


    def state_values(self, ind):
        state_dict = {}
        for input, ax in self.axis_for_input.items():
            # checking which axes are important for the input
            sl_ax = slice(ax[0], ax[-1]+1)
            # taking the indexes related to the axes 
            ind_inp = ind[sl_ax]
            state_dict[input] = self.state_inputs[input][ind_inp]
        return state_dict


    #this should be in the node claslss, just an example how the state_values can be used
    def yielding_state(self):
        for ind in itertools.product(*self.all_elements):
            state_dict = self.state_values(ind)
            print("State", state_dict)
            
