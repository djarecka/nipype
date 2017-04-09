import numpy as np
import itertools
import pdb

import auxiliary as aux

class State:
    def __init__(self, state_inputs, mapper=None):
        
        self.state_inputs = state_inputs

        self.mapper = mapper
        if self.mapper:
            self.mapper_rpn = aux.mapper2rpn(self.mapper)
        
        # dictionary[key=input names] = list of axis related to
        self.axis_for_input, self.ndim = aux.mapping_axis(self.state_inputs, self.mapper_rpn)

        # dictionary[key=axis] = list of axis related to
        self.input_for_axis = []
        self.shape = []
        self._converting_axis2input()

        self.all_elements = [range(i) for i in self.shape]



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
            sl = slice(ax[0], ax[-1]+1)
            state_dict[input] = self.state_inputs[input][ind[sl]]
        return state_dict


    #this should be in the node class
    def yielding_state(self):
        for ind in itertools.product(*self.all_elements):
            state_dict = self.state_values(ind)
            print("State", state_dict)
            
