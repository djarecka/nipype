import numpy as np
import itertools
from collections import namedtuple
import pdb

import auxiliary as aux

class State(object):
    def __init__(self, state_inputs, mapper=None):
        
        self.state_inputs = state_inputs

        self._mapper = mapper
        # TODO: what if mapper=None, more things should be in IF
        if self._mapper:
            # changing mapper (as in rpn), so I can read from left to right
            # e.g. if mapper=('d', ['e', 'r']), _mapper_rpn=['d', 'e', 'r', '*', '.']
            self._mapper_rpn = aux.mapper2rpn(self._mapper)

        self._input_names = [i for i in self._mapper_rpn if i not in ["*", "."]]
        self._state_tuple = namedtuple("state_tuple", self._input_names)

        #pdb.set_trace()
        # dictionary[key=input names] = list of axes related to
        # e.g. {'r': [1], 'e': [0], 'd': [0, 1]}
        # ndim - int, number of dimension for the "final array" (that is not created)
        self._axis_for_input, self._ndim = aux.mapping_axis(self.state_inputs, self._mapper_rpn)

        # list of inputs variable for each axis
        # e.g. [['e', 'd'], ['r', 'd']]
        # shape - list, e.g. [2,3]
        self._input_for_axis, self._shape = aux.converting_axis2input(self.state_inputs, 
                                                                    self._axis_for_input, self._ndim)
        
        # list of all possible indexes in each dim, will be use to iterate
        # e.g. [[0, 1], [0, 1, 2]]
        self._all_elements = [range(i) for i in self._shape]


    def __getitem__(self, key):
        if type(key) is int:
            key = (key,)
        return self.state_values(key)


    @property
    def input_for_axis(self):
        return self._input_for_axis


    @property
    def axis_for_input(self):
        return self._axis_for_input


    @property
    def all_elements(self):
        return self._all_elements


    @property
    def mapper(self):
        return self._mapper


    @property
    def ndim(self):
        return self._ndim


    @property
    def shape(self):
        return self._shape


    # it should be probably move to auxiliary 
    def _converting_axis2input(self):
        for i in range(self._ndim):
            self._input_for_axis.append([])
            self._shape.append(0)

        for inp, axis in self._axis_for_input.items():
            for (i, ax) in enumerate(axis):
                self._input_for_axis[ax].append(inp)
                self._shape[ax] =  self.state_inputs[inp].shape[i]


    def state_values(self, ind):
        if len(ind) > self._ndim:
            raise IndexError("too many indices")

        for ii, index in enumerate(ind):
            if index > self._shape[ii] - 1:
                raise IndexError("index {} is out of bounds for axis {} with size {}".format(index, ii, self._shape[ii]))

        state_dict = {}
        for input, ax in self._axis_for_input.items():
            # checking which axes are important for the input
            sl_ax = slice(ax[0], ax[-1]+1)
            # taking the indexes related to the axes 
            ind_inp = ind[sl_ax]
            state_dict[input] = self.state_inputs[input][ind_inp]

        # returning a named tuple
        return self._state_tuple(**state_dict)


    #this should be in the node claslss, just an example how the state_values can be used
    def yielding_state(self, function, reducer_key=None): # TODO should move to interface 
        results_list = []
        if reducer_key:
            if reducer_key in self._input_names:
                reducer_value_dict = {}
            else:
                # dj: reducer_key can be at the end also an output (?)
                raise Exception("reducer_key is not a valid input name")
        for ind in itertools.product(*self._all_elements):
            state_dict = self.state_values(ind)
            if reducer_key:
                val = state_dict.__getattribute__(reducer_key)
                if val in reducer_value_dict.keys():
                    #pdb.set_trace()
                    results_list[reducer_value_dict[val]][1].append((state_dict, function(*state_dict)))
                else:
                    #pdb.set_trace()
                    reducer_value_dict[val] = len(results_list)
                    results_list.append(("{} = {}".format(reducer_key, val), [(state_dict, function(*state_dict))]))
            else:        
                # TODO: it will be later interface.run or something similar
                results_list.append((state_dict, function(*state_dict)))
        return results_list
