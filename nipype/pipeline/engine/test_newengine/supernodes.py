#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Defines functionality for pipelined execution of interfaces

The `EngineBase` class implements the more general view of a task.

  .. testsetup::
     # Change directory to provide relative paths for doctests
     import os
     filepath = os.path.dirname(os.path.realpath( __file__ ))
     datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
     os.chdir(datadir)

"""
from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import object
from collections import namedtuple

from future import standard_library
standard_library.install_aliases()

from copy import deepcopy
import re
import numpy as np
import networkx as nx
from .... import logging
import itertools
from ....interfaces.base import DynamicTraitedSpec
from ....utils.filemanip import loadpkl, savepkl

from . import state

import pdb

logger = logging.getLogger('workflow')


class Node(object):
    """Defines common attributes and functions for workflows and nodes."""

    #dj: can mapper be None??
    def __init__(self, interface, name, mapper=None, reducer=None, reducer_interface=None,
                 inputs=None, base_dir=None):
        """ Initialize base parameters of a workflow or node

        Parameters
        ----------
        interface : Interface (mandatory)
            node specific interface
        inputs: dictionary
            inputs fields
        mapper: string, tuple (for scalar) or list (for outer product)
            mapper used with the interface
        reducer: string
            field used to group results
        reducer_interface: Interface
            interface used to reduce results
        name : string (mandatory)
            Name of this node. Name must be alphanumeric and not contain any
            special characters (e.g., '.', '@').
        base_dir : string
            base output directory (will be hashed before creations)
            default=None, which results in the use of mkdtemp

        """
        self._mapper = mapper
        # contains variables from the state (original) variables
        self._state_mapper = self._mapper
        self._reducer = reducer
        self._reducer_interface = reducer_interface
        if inputs:
            self._inputs = inputs
            # extra input dictionary needed to save values of state inputs
            self._state_inputs = self._inputs.copy()
        else:
            self._inputs = {}
            self._state_inputs = {}

        self._interface = interface
        self.base_dir = base_dir
        # dj TODO
        self.config = None
        self._verify_name(name)
        self.name = name
        # dj TODO: do I need _id and _hierarchy?
        # for compatibility with node expansion using iterables
        self._id = self.name
        self._hierarchy = None


    @property
    def result(self):
        if self._result:
            return self._result
        else:
            cwd = self.output_dir()
            # dj TODO: no self._load_resultfile
            result, _, _ = self._load_resultfile(cwd)
            return result

    @property
    def inputs(self):
        """Return the inputs of the underlying interface"""
        #return self._interface.inputs
        # dj: temporary will use self._inputs
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        # Basic idea, though I haven't looked up the actual way to do this:
        # self._interface.inputs.clear()
        # self._interface.inputs.update(inputs)
        #print("IN SETTER")
        self._inputs = inputs
        self._state_inputs = self._inputs.copy()


    @property
    def outputs(self):
        """Return the output fields of the underlying interface"""
        return self._interface._outputs()

    @property
    def interface(self):
        """Return the underlying interface object"""
        return self._interface

    @property
    def fullname(self):
        fullname = self.name
        if self._hierarchy:
            fullname = self._hierarchy + '.' + self.name
        return fullname

    # dj TODO: it's not the same as fullname? (self._id is self._name  in init)
    @property
    def itername(self):
        itername = self._id
        if self._hierarchy:
            itername = self._hierarchy + '.' + self._id
        return itername


    # dj TODO: when do we need clone?
    def clone(self, name):
        """Clone an EngineBase object

        Parameters
        ----------

        name : string (mandatory)
            A clone of node or workflow must have a new name
        """
        if (name is None) or (name == self.name):
            raise Exception('Cloning requires a new name')
        self._verify_name(name)
        clone = deepcopy(self)
        clone.name = name
        clone._id = name
        clone._hierarchy = None
        return clone


    def _check_outputs(self, parameter):
        return hasattr(self.outputs, parameter)


    # dj TODO: don't use it
    def _check_inputs(self, parameter):
        if isinstance(self.inputs, DynamicTraitedSpec):
            return True
        return hasattr(self.inputs, parameter)


    # dj TODO: don't use it
    def _verify_name(self, name):
        valid_name = bool(re.match('^[\w-]+$', name))
        if not valid_name:
            raise ValueError('[Workflow|Node] name \'%s\' contains'
                             ' special characters' % name)

    def __repr__(self):
        if self._hierarchy:
            return '.'.join((self._hierarchy, self._id))
        else:
            return '{}'.format(self._id)

    # dj TODO: don't use it
    def save(self, filename=None):
        if filename is None:
            filename = 'temp.pklz'
        savepkl(filename, self)

    # dj TODO: don't use it
    def load(self, filename):
        return loadpkl(filename)


    def run_interface(self):
        """ running interface for each element generated from node state.
            checks self._reducer and reduce the final result.
            returns a list with results (TODO: should yield)
        """
        results_list = []
        if self._reducer:
            # to save values for the reducer
            reducer_val_l = []
            if self._reducer in self.node_states._input_names:
                reducer_value_dict = {}
            elif self._reducer == "all":
                results_list.append(("all", []))
            else:
                raise Exception("reducer is not a valid input name")

        # this should yield at the end, not append to the list
        for ind in itertools.product(*self.node_states._all_elements):
            inputs_dict = self.node_states_inputs.state_values(ind)
            state_dict = self.node_states.state_values(ind)
            res = self._interface.run(**inputs_dict._asdict())
            output = res.outputs
            if self._reducer and self._reducer != "all":
                val = state_dict.__getattribute__(self._reducer)
                if val in reducer_value_dict.keys():
                    results_list[reducer_value_dict[val]][1].append((state_dict, output))
                else:
                    reducer_value_dict[val] = len(results_list)
                    results_list.append(("{} = {}".format(self._reducer, val), [(state_dict, output)]))
                    reducer_val_l.append(val)
            elif self._reducer == "all":
                results_list[0][1].append((state_dict, output))
            else:
                # TODO: it will be later interface.run or something similar
                results_list.append((state_dict, output))

        # TODO: this probably should be in another method, after I moved to use generators
        # if self._reducer_interface I have to run an extra interface for every reducer value
        if self._reducer_interface:
            results_list_red = []
            # TODO assuming for now that is only one field in the reducer
            for ii, (st_el, res_el) in enumerate(results_list):
                values_l = [i[1].out for i in res_el]
                # TODO: should work for other arguments names
                res_red = self._reducer_interface.run(out=values_l) #assuming one val for now
                if self._reducer == "all":
                    #state_tuple_red = namedtuple("state_tuple", ["all"])
                    results_list_red.append(("all",res_red.outputs))
                else:
                    state_tuple_red = namedtuple("state_tuple", sorted(self._reducer))
                    results_list_red.append((state_tuple_red(reducer_val_l[ii]), res_red.outputs))
            return results_list_red

        return results_list



    def run(self):
        # contains value of inputs
        #pdb.set_trace()
        self.node_states_inputs = state.State(state_inputs=self._inputs, mapper=self._mapper)
        #pdb.set_trace()
        # contains value of state inputs (values provided in original input)
        self.node_states = state.State(state_inputs=self._state_inputs, mapper=self._state_mapper)
        self._result = self.run_interface()


class Workflow(object):
    #allow_flattening = False #not used for now

    def __init__(self, nodes=None, **kwargs):
        self.graph = nx.DiGraph()
        if nodes:
            self._nodes = nodes
            self.graph.add_nodes_from(nodes)
        else:
            self._nodes = []
        self.connected_var = {}


    @property
    def nodes(self):
        return self._nodes


    def add_nodes(self, nodes):
        """adding nodes without defining connections"""
        self._nodes += nodes
        self.graph.add_nodes_from(nodes)
        for nn in nodes:
            self.connected_var[nn] = {}


    def connect(self, from_node, from_socket, to_node, to_socket):
        self.graph.add_edges_from([(from_node, to_node)])
        if not to_node in self.nodes:
            self.add_nodes(to_node)
        self.connected_var[to_node][to_socket] = (from_node, from_socket)


    def run(self, monitor_consumption=True): #dj TODO: monitor consumption not used
        self.graph_sorted = list(nx.topological_sort(self.graph))
        for nn in self.graph_sorted:
            try:
                for inp, out in self.connected_var[nn].items():
                    (node_nm, var_nm) = self.connected_var[nn][inp]
                    nn.inputs.update({inp: np.array([getattr(ii[1], var_nm) for ii in node_nm.result])})
                    nn._state_inputs.update(node_nm._state_inputs)
                    #pdb.set_trace()
                    if not nn._state_mapper or nn._state_mapper == out[1]:
                        #pdb.set_trace()
                        nn._state_mapper = node_nm._state_mapper
                        nn._mapper = inp
                    elif out[1] in nn._state_mapper: #_state_mapper or _mapper?? TODO
                        if type(nn._state_mapper) is tuple:
                            #pdb.set_trace()
                            nn._state_mapper = tuple([node_nm._state_mapper if ii==out[1] else ii for ii in nn._state_mapper])
                            nn._mapper = tuple([inp if ii==out[1] else ii for ii in nn._mapper])
                        elif type(nn._state_mapper) is list:
                            nn._state_mapper = [node_nm._state_mapper if ii==out[1] else ii for ii in nn._state_mapper]
                            nn._mapper = [inp if ii==out[1] else ii for ii in nn._mapper]
                    #TODO!!!!
                    elif [out[1], inp] in nn._state_mapper:
                        if type(nn._state_mapper) is tuple:
                            nn._state_mapper = tuple([[node_nm._state_mapper, inp+"_ind"] if ii==[out[1], inp] else ii for ii in nn._state_mapper])
                            nn._mapper = tuple([inp if ii==[out[1], inp] else ii for ii in nn._mapper])
                        if type(nn._state_mapper) is list:
                            nn._state_mapper = [[node_nm._state_mapper, inp+"_ind"] if ii==[out[1], inp] else ii for ii in nn._state_mapper]
                            nn._mapper = [inp if ii==[out[1], inp] else ii for ii in nn._mapper]
                        nn._state_inputs.update({inp+"_ind": np.array(range(nn.inputs[inp].shape[-1]))})
                        # not sure..
                        #pdb.set_trace()
                        #nn.inputs.update({inp: np.array([getattr(ii[1], var_nm) for ii in node_nm.result])})
                        #nn._mapper = [node_nm._state_mapper, nn._mapper]
                        #pdb.set_trace()
                        pass
                    elif inp in nn._state_mapper:
                        raise Exception("{} can be in the mapper only together with {}, i.e. {})".format(inp, out[1], [out[1], inp])) 
                    
            except(KeyError):
                # tmp: we don't care about nn that are not in self.connected_var
                pass

            nn.run()
