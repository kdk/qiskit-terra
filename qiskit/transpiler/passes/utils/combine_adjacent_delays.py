import logging

from pprint import pformat, pprint
from dataclasses import dataclass
from collections import defaultdict

from typing import List

from qiskit.circuit import Delay, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import TransformationPass, CouplingMap

logger = logging.getLogger(__name__)

MIN_JOINABLE_DELAY_DURATION = 500

class CombineAdjacentDelays(TransformationPass):
    def __init__(self, cmap):
        self.cmap = cmap
        super().__init__()


    def run(self, dag):
        bit_idx_locs = {bit: idx for idx, bit in enumerate(dag.qubits)}

        # Collect and sort a list of all times where a delay starts or ends
        # These will be the places we'll examine to split/combine delay ops.
        sorted_delay_edges = sorted(
            (
                (event_type,
                 start_time if event_type == 'begin' else start_time + op_node.op.duration,
                 op_node)
                for op_node, start_time in self.property_set['node_start_time'].items()
                if (
                    op_node.op.name == 'delay'
                    and start_time != 0  # Skip delays at start of circuit
                    # and start_time + op_node.op.duration < dag.duration  # Skip delays at end of circuit
                    and op_node.op.duration > MIN_JOINABLE_DELAY_DURATION
                )
                for event_type in ('begin', 'end')
            ),
        key=lambda itm: itm[1])
        joinable_delay_nodes = set(op_node for _, __, op_node in sorted_delay_edges)

        logger.info(pformat(sorted_delay_edges))

        @dataclass
        class ReplacementDelay:
            new_delay_op: Delay
            start_time: int
            end_time: int
            replacing_delay_nodes: List[DAGOpNode]

        open_delays = []  # new_delay_op start_time, replacing_delay_nodes
        closed_delays = []  # new_delay_op, start_time, end_time, replacing_delay_nodes

        def _open_delay(start_time, replacing_delay_nodes):
            qubits = [
                qarg.index for delay_node in replacing_delay_nodes for qarg in delay_node.qargs
            ]
            reduced = self.cmap.reduce(qubits)
            # If the delay is not connected, break it up into its components and open new 
            # delya for each 
            if not reduced.is_connected():
                # If every single component of the mapping contains only one element,
                # then reduce() returns empty
                if reduced.size() > 0:
                    connected_components = reduced.connected_components()
                    for component in connected_components:
                        new_replacing_delay_nodes = [
                            replacing_delay_nodes[idx] for idx in component
                        ]
                        __open_delay(start_time, new_replacing_delay_nodes)
                else:
                    for node in replacing_delay_nodes:
                        __open_delay(start_time, [node])
            else:
                __open_delay(start_time, replacing_delay_nodes)

        def __open_delay(start_time, replacing_delay_nodes):
            delay_op = Delay(0)
            delay_op.num_qubits = len(replacing_delay_nodes)
            open_delays.append((delay_op, start_time, replacing_delay_nodes))

        def _expand_delay(start_time, existing_delay, new_replacing_delay_node):
            delay_op = Delay(0)
            delay_op.num_qubits = 1 + existing_delay[0].num_qubits

            _close_delay(existing_delay, start_time, closing_op_node=None)  # Don't auto-open a new delay.
            open_delays.append((delay_op, start_time, existing_delay[2] + [new_replacing_delay_node]))

        def _close_delay(delay, end_time, closing_op_node=None):
            delay_op, start_time, replacing_delay_nodes = delay
            open_delays.remove(delay)

            # When merging delays, we might end up opening and immediately closing a delay at the same time it started.
            # So in this case, just drop it.
            if start_time != end_time:
                delay_op.duration = end_time - start_time
                closed_delays.append((delay_op, start_time, end_time, replacing_delay_nodes))

            # If we're closing a delay on 1 qubit out of many, re-open a N-1 qubit delay on the remaining qubits
            if closing_op_node is not None and len(replacing_delay_nodes) > 1:
                _open_delay(end_time, [node for node in replacing_delay_nodes if node is not closing_op_node])

        def _combine_delays(old_delay_node, start_time, delays):
            # If we find a delay opening which is adjacent two two or more open delays.

            for delay in delays:
                _close_delay(delay, start_time)
            _open_delay(start_time,
                        [old_delay_node] + [node for delay in delays for node in delay[2]])

        for edge_type, edge_time, edge_node in sorted_delay_edges:
            adjacent_open_delays = [open_delay for open_delay in open_delays
                                    if any(
                                        self.cmap.distance(
                                            bit_idx_locs[edge_node.qargs[0]],
                                            bit_idx_locs[open_delay_qarg]) <= 1
                                           for open_delay_node in open_delay[2]
                                           for open_delay_qarg in open_delay_node.qargs)]
            if edge_type == 'begin':
                # If crossing a begin edge, check if there are any open delays that are adjacent.
                # If so, close those, and open a new delay spanning those qubits and this one.
                # If not, open a new delay.

                if len(adjacent_open_delays) == 0:
                    # Make a new delay
                    _open_delay(edge_time, [edge_node])
                else:
                    # Combine that with this
                    _combine_delays(edge_node, edge_time, adjacent_open_delays)

            if edge_type == 'end':
                # If crossing a end edge, close any open delay on this qubit (and re-open a delay on any qubits this delay had shared)
                if len(adjacent_open_delays) != 1:
                    import pdb; pdb.set_trace()
                    raise Exception("closing edge w/o an open delay?")
                else:
                    _close_delay(adjacent_open_delays[0], edge_time, edge_node)

            logger.info(pformat((
                '---exit---',
                [edge_type, edge_time, edge_node],
                ('open_delays: ', open_delays),
                ('closed_delays: ', closed_delays),
            )))

        logger.info(pformat((
            '---post_collect---',
            ('open_delays: ', open_delays),
            ('len(closed_delays): ', len(closed_delays)),
            ('len(multi_q_delays): ', len([delay for delay in closed_delays if delay[0].num_qubits > 1])),
        ))) #Expect ~30 for example qv 256

        # Try something simpler:
        # 1) Split each doomed delay in to N 1-q delays of new duration
        # 2) Combine blocks of new multi-qubit delays

        # create lookup from delay to replacements sorted by start time

        replacement_ops = defaultdict(list)  # existing_delay_node, [sorted_list_of_replacement_delays]
        for closed_delay in sorted(closed_delays, key=lambda k: (k[1], k[2])):  # Might already be in right order?
            (delay_op, start_time, end_time, replacing_delay_nodes) = closed_delay

            for replacing_delay_node in replacing_delay_nodes:
                replacement_ops[replacing_delay_node].append(closed_delay)

        delay_op_placeholders = defaultdict(list)

        for doomed_delay_node, replacements in replacement_ops.items():
            oneq_delay = QuantumCircuit(1)
            for (delay_op, _, __, ___) in replacements:
                oneq_delay.delay(delay_op.duration, 0)

            oneq_dag = circuit_to_dag(oneq_delay)
            oneq_delay_node_ids = [ node._node_id for node in oneq_dag.topological_op_nodes()]

            out_node_map = dag.substitute_node_with_dag(doomed_delay_node, oneq_dag)
            out_oneq_delay_nodes = [out_node_map[node_id] for node_id in oneq_delay_node_ids]

            for (delay_op, _, __, ___), out_oneq_delay_node in zip(replacements, out_oneq_delay_nodes):
                delay_op_placeholders[id(delay_op)].append(out_oneq_delay_node)

        for (delay_op, start_time, end_time, replacing_delay_nodes) in closed_delays:
            doomed_placeholder_nodes = delay_op_placeholders[id(delay_op)]
            dag.replace_block_with_op(
                doomed_placeholder_nodes,
                delay_op,
                {node.qargs[0]: idx for idx, node in enumerate(doomed_placeholder_nodes)},
                cycle_check=True)



        return dag
