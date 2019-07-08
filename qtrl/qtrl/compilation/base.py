# Copyright (c) 2018-2019, UC Regents

from collections import OrderedDict, namedtuple
import copy
import numpy as np

AlgorithmNode = namedtuple('AlgorithmNode', ['Name', 'QubitNames', 'past', 'future'])


class Algorithm:
    """Class which builds a directed acylcic graph representation of an algorithm.
    This can be extended to represent QASM code, and allows for easy composite gate
    and randomized compiling tools."""

    def __init__(self, qubits=2):
        """Similar to QASM you have to instantiate the number of qudits you have"""

        self.graph = OrderedDict()

        if isinstance(qubits, list):
            start_pointers = qubits
            self.n_qubits = len(qubits)
        else:
            start_pointers = ['Q{}'.format(i) for i in range(int(qubits))]
            self.n_qubits = qubits

        self.qubit_names = start_pointers
        self.start_nodes = ['{} Start'.format(p) for p in start_pointers]
        self.qubit_pointers = dict(zip(start_pointers, self.start_nodes))

        for pointer in start_pointers:
            self.graph.update({self.qubit_pointers[pointer]: [AlgorithmNode(*[None, [pointer], [None], [None]]), 0]})

        self.clock_cycles = None

    def add_op(self, name, *qubits):
        """Operations are recorded by their name and qubits they are being applied to,
        Qudits are named 'Q0', 'Q1' etc. The order of qudits supplied here is important, as for
        CNOT gates or similar non-symmetric gates order matters."""
        virtex_pointer = self.new_ident()
        previous_pointers = []
        for i, qubit in enumerate(qubits):
            qubit_index = self.graph[self.qubit_pointers[qubit]][0].QubitNames.index(qubit)
            self.graph[self.qubit_pointers[qubit]][0].future[qubit_index] = virtex_pointer
            previous_pointers.append(self.qubit_pointers[qubit])
            self.qubit_pointers[qubit] = virtex_pointer
        new_node = AlgorithmNode(*[name, list(qubits), previous_pointers, len(qubits) * [None]])
        self.graph[virtex_pointer] = [new_node, 0]

    def new_ident(self, ident=None):
        """Generates a unique ID not currently in the computational graph"""

        # shouldnt need more then 10^9 nodes.... if we do for the love of god
        # stop using python
        if ident is None:
            ident = str(np.random.randint(1e9))

        if ident in self.graph:
            ident = self.new_ident(str(int(ident) + 1))
        return ident

    def def_gate(self, gate):
        """Define the name of a gate"""
        self.__dict__[gate] = lambda *y: self.add_op(gate, *y)

    def calculate_priorities(self):
        """Calculate the gate depths of every node in the graph
        final gates will have 0 priority, the next clock cycle back is 1 etc
        2 qubit gates complicate things as they effect multiple qubits."""

        self._clear_priorities()

        # keep track of final positions
        final_pointers = copy.deepcopy(self.qubit_pointers)

        # until all the pointers point back to the start, keep traversing upward
        while not all([x in self.start_nodes for x in self.qubit_pointers.values()]):
            # move all qubits up until hitting a multi-qubit node
            for qubit in self.qubit_names:
                self.advance_pointer(qubit)

            # correctly calculate the priorities and step the
            # points up to the next node
            for qubit in self.qubit_names:
                pointer = self.qubit_pointers[qubit]
                node = self.graph[pointer]
                q_index = node[0].QubitNames.index(qubit)
                past = node[0].past[q_index]
                if past is None or self.graph[past] is None:
                    next
                else:
                    self.graph[past][1] = np.max([node[1] + 1, self.graph[past][1]])
                    self.qubit_pointers[qubit] = past

        self.qubit_pointers = final_pointers

    def reset_pointers(self):
        """Resets to qubit pointers to the end of the graph"""

        # for each qubit traverse the graph and find the end
        for i, qubit in enumerate(self.qubit_names):

            cur_node_pointer = self.start_nodes[i]

            while cur_node_pointer is not None:
                node = self.graph[cur_node_pointer][0]
                q_index = node.QubitNames.index(qubit)
                if node.future[q_index] is None:
                    break
                cur_node_pointer = node.future[q_index]
            self.qubit_pointers[qubit] = cur_node_pointer

    def advance_pointer(self, qubit):
        """Given a current qubit, move the pointer for the qubit up the graph
        until it hits a multi-qubit node, then stop.

        Updates priorities as it goes.

        """
        pointer = self.qubit_pointers[qubit]

        cur_node = self.graph[pointer]

        cur_priority = cur_node[1]

        chain = [pointer]
        if (cur_node is None or cur_node[0].past[0] is None):
            return chain

        while len(cur_node[0].QubitNames) == 1:

            pointer = cur_node[0].past[0]
            if (pointer is None):
                return chain[:-1]

            chain.append(pointer)

            cur_priority += 1
            self.graph[pointer][1] = np.max([cur_priority, self.graph[pointer][1]])

            cur_node = self.graph[pointer]

        self.qubit_pointers[qubit] = pointer

        return chain[:-1]

    def compile(self):
        """Takes the graph and builds the gate sequence to be run"""
        self.reset_pointers()
        self.calculate_priorities()

        n_qubits = self.n_qubits

        self.gate_depth = max([self.graph[x][1] for x in self.start_nodes])

        clock_cycles = []
        current_pointers = [self.qubit_pointers[qubit] for qubit in self.qubit_names]

        # we have an expected number of layers based on the priorities calculated
        # for each layer we can find all the correct gates to be run
        for layer in range(self.gate_depth):

            # For a given clock cycle (IE: layer) we find all gates at that depth
            cur_cycle = []

            # we do this by climbing the tree, since we keep track of the last
            # pointer to each qubit, we can iterate up the branch and check
            # to see if the priority is the same as the layer we are on
            # if it is, we record the gate, and move the pointer to the
            # next highest node in the graph
            for i, qubit in enumerate(self.qubit_names):
                # First we get the current node for each qubit
                node = self.graph[current_pointers[i]]

                # if the priority is the same or less than the layer, and not the start
                # we record the gate
                if node[1] == layer and node[0].past[0] is not None:
                    # since the qubits are recorded internally in the node
                    # we have to find which index the qubit we are currently
                    # looking at is at in the node, IE: for a cnot, 0 or 1
                    # for the target and control as an example
                    q_ind = node[0].QubitNames.index(qubit)

                    # if we have a multi qubit gate we want to record the order of operations
                    # so that we can apply the multi qubit gate in the correct direction
                    # this is a string which just adds a 0, 1, 2... as appropriate
                    # to the gate name to indicate which part of the multi-qubit gate
                    # the qubit is effected by, IE: 0 might be target and 1 might be control
                    ident = '' if len(node[0].QubitNames) == 1 else str(q_ind)

                    # Now we build a gate name using our standard convention, Q3_X90 for example
                    cur_cycle.append(str(node[0].QubitNames[q_ind]) + "_" + node[0].Name + ident)

                    # increment our pointers as appropriate
                    current_pointers[i] = node[0].past[q_ind]

            # record our clock cycles, this should never be empty
            clock_cycles.append(cur_cycle)

        # re-order since we built end to start
        clock_cycles = clock_cycles[::-1]

        # Record results and return
        self.clock_cycles = clock_cycles
        return clock_cycles

    def replace_node(self, replace_pointer, other_alg):
        """Replace a node with another Algorithm, this can
        be used for composite pulses or randomized compiling."""

        subgraph = copy.deepcopy(other_alg)

        original_node = self.graph[replace_pointer][0]

        past_pointers = original_node.past
        future_pointers = original_node.future

        # now we need to add all the nodes from the sub_graph into the original graph
        # renaming nodes as we go and preserving the structure

        # new_subgraph_qubit_names = dict(zip(subgraph.qubit_names, self.graph[replace_pointer][0].QubitNames))

        # before doing anything important
        assert len(past_pointers) == len(subgraph.qubit_names), 'Not the same number of qubits'

        # change qubit names as appropriate
        # subgraph.qubit_names = [new_subgraph_qubit_names[x] for x in subgraph.qubit_names]
        # sg_pointers = list(subgraph.qubit_pointers.keys())
        # for key in sg_pointers:
        #     subgraph.qubit_pointers[new_subgraph_qubit_names[key]] = subgraph.qubit_pointers[key]
        #     del subgraph.qubit_pointers[key]

        # for g in subgraph.graph:
        #     subgraph.graph[g][0] = subgraph.graph[g][0]._replace(
        #         QubitNames=[new_subgraph_qubit_names[x] for x in subgraph.graph[g][0].QubitNames])

        # Merge the subgraph with the graph and deal with hash collisions
        sub_graph_connections = self.merge_graphs(subgraph)

        # move the pointers over to the subgraph
        for i, qubit in enumerate(original_node.QubitNames):
            # move the connection from the start first
            past_pointer = original_node.past[i]
            past_node = self.graph[past_pointer][0]
            new_target = sub_graph_connections[qubit][0]

            past_node_qubit_index = past_node.QubitNames.index(qubit)
            new_target_q_index = self.graph[new_target][0].QubitNames.index(qubit)

            past_node.future[past_node_qubit_index] = new_target
            self.graph[new_target][0].past[new_target_q_index] = past_pointer

            # move the exit points to the new points
            future_pointer = original_node.future[i]
            if future_pointer is not None:
                future_node = self.graph[future_pointer][0]
                new_target = sub_graph_connections[qubit][1]

                future_node_qubit_index = future_node.QubitNames.index(qubit)
                new_target_q_index = self.graph[new_target][0].QubitNames.index(qubit)

                future_node.past[future_node_qubit_index] = new_target
                self.graph[new_target][0].future[new_target_q_index] = future_pointer

            # update the qubit_pointers to point to the new stuff as necessary
            if self.qubit_pointers[qubit] == replace_pointer:
                self.qubit_pointers[qubit] = subgraph.qubit_pointers[subgraph.qubit_names[i]]

        # remove the old node
        del self.graph[replace_pointer]

        # if no errors, go drink a beer because you deserve it

    def del_node(self, del_pointer):
        """Given a reference, delete the node"""
        assert del_pointer in self.graph, 'Node {} not present in graph'.format(del_pointer)

        node = self.graph[del_pointer]

        for ind, qubit in enumerate(node[0].QubitNames):
            past_node_ref = node[0].past[ind]
            future_node_ref = node[0].future[ind]

            if past_node_ref is not None:
                past_q_index = self.graph[past_node_ref][0].QubitNames.index(qubit)
                self.graph[past_node_ref][0].future[past_q_index] = future_node_ref

            if future_node_ref is not None:
                future_q_index = self.graph[future_node_ref][0].QubitNames.index(qubit)
                self.graph[future_node_ref][0].past[future_q_index] = past_node_ref
            elif self.qubit_pointers[qubit] == del_pointer:
                self.qubit_pointers[qubit] = past_node_ref

        del self.graph[del_pointer]

    def _clear_priorities(self):
        for node in self.graph:
            self.graph[node][1] = 0

    def merge_graphs(self, other_graph):
        """Merges a graph with the pre-existing graph in the alg
        If there are hash collisions, it renames the nodes in the graph
        to ID's that are not present in the current algorithm

        Returns:
            dict of qubit names in the other_graph
                entries of dict are a 2-tuple of the entry
                and exit nodes of the other_graph"""

        other_graph = copy.deepcopy(other_graph)

        # move the other_graph start pointers to the first non-start node present
        for i, pointer in enumerate(other_graph.start_nodes):
            old_start = pointer
            other_graph.start_nodes[i] = other_graph.graph[pointer][0].future[0]
            other_graph.graph.pop(old_start, None)

        # check for hash collisions
        hash_collisions = [x for x in other_graph.graph if x in self.graph]

        for i in hash_collisions:
            # get a new id number
            new_index = self.new_ident()

            # out with the old, in with the new ID
            for g in other_graph.graph:
                other_graph.graph[g][0] = other_graph.graph[g][0]._replace(
                    future=[x if x != i else new_index for x in other_graph.graph[g][0].future])
                other_graph.graph[g][0] = other_graph.graph[g][0]._replace(
                    past=[x if x != i else new_index for x in other_graph.graph[g][0].past])

                # update start_node pointers to the new ID if we need
            for j, start_node in enumerate(other_graph.start_nodes):
                if start_node == i:
                    other_graph.start_nodes[j] = new_index

            # same with end pointers
            for p in other_graph.qubit_pointers:
                if other_graph.qubit_pointers[p] == i:
                    other_graph.qubit_pointers[p] = new_index

            # update graph and remove old reference
            other_graph.graph[new_index] = other_graph.graph[i]
            other_graph.graph.pop(i, None)

        connection_points = {}
        for qubit in other_graph.qubit_pointers:
            qubit_index = other_graph.qubit_names.index(qubit)
            connection_points[qubit] = [other_graph.start_nodes[qubit_index],
                                        other_graph.qubit_pointers[qubit]]

        self.graph.update(other_graph.graph)

        return connection_points

    def simplify(self, simplifications, qubit=None):
        if qubit is None:
            for qubit in self.qubit_names:
                self.simplify(simplifications, qubit)

        self.reset_pointers()

        q_pointers = copy.deepcopy(self.qubit_pointers)

        chain_pointers = self.advance_pointer(qubit)

        for l in range(0, len(chain_pointers), 1):
            for k in range(len(chain_pointers), l, -1):
                cur_pointers = chain_pointers[l:k]
                names = [self.graph[x][0].Name for x in cur_pointers]

                if tuple(names) in simplifications:
                    simp_names = simplifications[tuple(names)]
                    for p in cur_pointers[1:]:
                        self.del_node(p)
                    if len(simp_names) == 0:
                        self.del_node(cur_pointers[0])
                        return self.simplify(simplifications, qubit)

                    t_alg = Algorithm(qubits=['T0'])
                    for gate in simp_names:
                        t_alg.add_op(gate, 'T0')
                    if cur_pointers[0] is not None:
                        self.replace_node(cur_pointers[0], t_alg)
                        return self.simplify(simplifications, qubit)

        return simplifications
