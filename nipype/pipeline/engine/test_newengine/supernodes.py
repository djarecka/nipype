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

        name : string (mandatory)
            Name of this node. Name must be alphanumeric and not contain any
            special characters (e.g., '.', '@').
        base_dir : string
            base output directory (will be hashed before creations)
            default=None, which results in the use of mkdtemp

        """
        self._mapper = mapper
        # dj TODO: check when it should be moved to self._state_*
        self._state_mapper = self._mapper
        self._reducer = reducer
        self._reducer_interface = reducer_interface
        if inputs:
            self._inputs = inputs
            # dj: i would need extra dictionaries for state in workflow
            self._state_inputs = self._inputs.copy()
        else:
            self._inputs = {}
            self._state_inputs = {}

        self._interface = interface
        self.base_dir = base_dir
        # dj TODO: do i need it?
        self.config = None
        self._verify_name(name)
        self.name = name
        # dj TODO: do I need _id and _hierarchy?
        # for compatibility with node expansion using iterables
        self._id = self.name
        self._hierarchy = None


    @property
    def result(self):
        # dj TODO: think if we want to have self._result
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
        print("IN SETTER")
        self._inputs = inputs
        # dj: i would need extra dictionaries for state in workflow
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

    # dj TODO: don't understand
    def _check_inputs(self, parameter):
        if isinstance(self.inputs, DynamicTraitedSpec):
            return True
        return hasattr(self.inputs, parameter)

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

    def save(self, filename=None):
        if filename is None:
            filename = 'temp.pklz'
        savepkl(filename, self)

    def load(self, filename):
        return loadpkl(filename)


    def run_interface(self):
        """ running interface for each element generated from node state.
            checks self._reducer and reduce the final result.
            returns a list with results (TODO: should yield?)
        """
        results_list = []
        if self._reducer:
            red_val_l = []
        if self._reducer:
            if self._reducer in self.node_states._input_names:
                reducer_value_dict = {}
            else:
                # dj: self._reducer can be at the end also an output (?)
                raise Exception("reducer is not a valid input name")

        # this should yield at the end, not append to the list
        for ind in itertools.product(*self.node_states._all_elements):
            inputs_dict = self.node_states_inputs.state_values(ind)
            state_dict = self.node_states.state_values(ind)
            res = self._interface.run(**inputs_dict._asdict())
            output = res.outputs
            if self._reducer:
                val = state_dict.__getattribute__(self._reducer)
                if val in reducer_value_dict.keys():
                    results_list[reducer_value_dict[val]][1].append((state_dict, output))
                else:
                    reducer_value_dict[val] = len(results_list)
                    results_list.append(("{} = {}".format(self._reducer, val), [(state_dict, output)]))
                    red_val_l.append(val)
            else:
                # TODO: it will be later interface.run or something similar
                results_list.append((state_dict, output))

        if self._reducer_interface:
            results_list_red = []
            # TODO assuming for now that is only one field in the reducer
            for ii, (st_el, res_el) in enumerate(results_list):
                values_l = [i[1].out for i in res_el]
                res_red = self._reducer_interface.run(out=values_l) #assuming one val for now
                state_tuple_red = namedtuple("state_tuple", sorted(self._reducer))
                results_list_red.append((state_tuple_red(red_val_l[ii]), res_red.outputs))
            return results_list_red

        return results_list



    def run(self):
        # dj: contains value of inputs
        self.node_states_inputs = state.State(state_inputs=self._inputs, mapper=self._mapper)
        # dj: contains value of state inputs
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
        self._nodes += nodes
        self.graph.add_nodes_from(nodes)
        for nn in nodes:
            self.connected_var[nn] = {}


    def connect(self, from_node, from_socket, to_node, to_socket):
        self.graph.add_edges_from([(from_node, to_node)])
        if not to_node in self.nodes:
            self.add_nodes(to_node)
        self.connected_var[to_node][to_socket] = (from_node, from_socket)


    def run(self, monitor_consumption=True):
        self.graph_sorted = list(nx.topological_sort(self.graph))
        for nn in self.graph_sorted:
            try:
                for inp, out in self.connected_var[nn].items():
                    (node_nm, var_nm) = self.connected_var[nn][inp]
                    nn.inputs.update({inp: np.array([getattr(ii[1], var_nm) for ii in node_nm.result])})
                    nn._state_inputs.update(node_nm._state_inputs)
                    if type(nn._state_mapper) is str:
                        nn._state_mapper = nn._state_mapper.replace(inp, node_nm._state_mapper)
                    elif type(nn._state_mapper) is tuple:
                        nn._state_mapper = tuple([node_nm._state_mapper if ii==inp else ii for ii in nn._state_mapper])
                    elif type(nn._state_mapper) is list:
                        nn._state_mapper = [node_nm._state_mapper if ii == inp else ii for ii in nn._state_mapper]
            except(KeyError):
                pass

            nn.run()
