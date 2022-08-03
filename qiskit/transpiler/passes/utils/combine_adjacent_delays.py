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

        @dataclass
        class DelayEvent:
            type: str
            time: int
            op_node: DAGOpNode
            
        # These will be the places we'll examine to split/combine delay ops.
        sorted_delay_edges = sorted(
            (
                DelayEvent(
                    event_type,
                    start_time if event_type == 'begin' else start_time + op_node.op.duration,
                    op_node
                )
                for op_node, start_time in self.property_set['node_start_time'].items()
                if (
                    op_node.op.name == 'delay'
                    and start_time != 0  # Skip delays at start of circuit
                    # and start_time + op_node.op.duration < dag.duration  # Skip delays at end of circuit
                    and op_node.op.duration > MIN_JOINABLE_DELAY_DURATION
                )
                for event_type in ('begin', 'end')
            ),
            key=lambda event: (
                event.time,                        # Sort by event time
                0 if event.type == 'end' else 1,  # With 'end' events before 'begin'
            )
        )

        logger.info(pformat(sorted_delay_edges))

        # Approach before was to collect open and close edges and process more or less in place.
        # As delays open, expand any open delays, and as they close, replace them with narrower delays.
        #
        # Now, still collect open and closed events to group into idle adjacent idle time
        # Might want to sort to say which is widest adjacnet group.
        # Start there: if it's width is > MIN_JOIN, make a delay, add to output remove from input.
        # If it's less than MIN_JOIN, panic.
        # Look at start and end qubit, if same, remove it, and give it its own 1q delay. Re-insert remiander in queue.
        # If start and end are different (will probably need to group these to multi-q) so above may be "find intersection"
        # Will be hard to know what to do next. Could split L and split R, and maybe this is the best since we're looking
        # for "widest delays only".
        #
        # 

        @dataclass
        class DelayGroup:
            events: List[DelayEvent]
            active_qubits: Set[Qubit]
        
        # Collect grouped delays by concurrency in time and adjacencey on device.

        open_delays = []
        closed_delays = []

        def _open_delay(delay_edge):
            open_delay_groups.append(
                DelayGroup(events=[delay_edge],
                           active_quits=set(delay_event.op_node.qargs)))

        def _close_delay(open_delay_group, delay_edge):
            open_delay_group.events.append(delay_edge)
            open_delay_group.active_qubits -= set(delay_edge.op_node.qargs)

            if not open_delay_group.active_qubits:
                open_delays.remove(open_delay_group)
                closed_delays.append(open_delay_group)

        def _combine_delays(delay_edge, open_delay_groups):
            survivor, *doomed = open_delay_groups

            survivor.events.append(delay_edge)
            survivor.active_qubits.update(delay_edge.op_node.qargs)

            for doomed_delay_group in doomed:
                if survivor.active_qubits.intersection(doomed_delay_group.active_qubits):
                    raise RuntimeError('More than one open delay on a qubit?')

                survivor.events.extend(doomed_delay_group.events)
                survivor.active_qubits.update(doomed_delay_group.active_qubits)

                open_delays.remove(doomed_delay_group)
                survivor.events.sort(key=lambda event: event.time)  # Maintain sorted event order
            
        for delay_edge in sorted_delay_edges:
            adjacent_open_delays = [open_delay for open_delay in open_delays
                                    if any(
                                            self.cmap.distance(
                                                bit_idx_locs[delay_edge.op_node.qargs[0]],
                                                bit_idx_locs[open_delay_qarg]) <= 1
                                            for open_delay_node in open_delay.active_qubits)]

            if delay_edge.type == 'begin':
                # If crossing a begin edge, check if there are any open delays that are adjacent.
                # If so, add current event to that group.
            
                if len(adjacent_open_delays) == 0:
                    # Make a new delay
                    _open_delay(delay_edge)
                else:
                    # Combine that with this
                    _combine_delays(delay_edge, adjacent_open_delays)

            if delay_edge.type == 'end':
                # If crossing a end edge, remove this qubit from the actively delaying qubits"
                if len(adjacent_open_delays) != 1:
                    import pdb; pdb.set_trace()
                    raise Exception("closing edge w/o an open delay?")
                else:
                    _close_delay(adjacent_open_delays[0], delay_edge)

            logger.info(pformat((
                '---exit---',
                delay_edge,
                ('open_delays: ', open_delays),
                ('closed_delays: ', closed_delays),
            )))

        logger.info(pformat((
            '---post_collect---',
            ('open_delays: ', open_delays),
            ('len(closed_delays): ', len(closed_delays)),
            ('len(multi_q_delays): ', len([delay for delay in closed_delays if delay[0].num_qubits > 1])),
        ))) #Expect ~30 for example qv 256

        if open_delays:
            raise RuntimeError('Failed to close all open delays.')
        for closed_delay in closed_delays:
            if closed_delay.active_qubits:
                raise RuntimeError('Failed remove active qubits on closed delays.')

        @dataclass
        class GroupedDelayEvent:
            type: str
            time: int
            op_nodes: List[DAGOpNode]
        
        # Within each closed delay, collect events by time, type.
        grouped_closed_delays = []
        
        for closed_delay in closed_delays:
            grouped_delay_edges = []

            for delay_edge in closed_delay.events:
                # Look back through known groups.
                for existing_grouped_delay in reversed(grouped_delay_edges):
                    # If newest grouped event is older than current, need a new group.
                    if event_time > existing_grouped_delay.time:
                        grouped_delay_edges.append(
                            GroupedDelayEvent(delay_edge.type,
                                              delay_edge.time,
                                              [delay_edge.op_node]))
                        break

                    # If newest grouped event is _newer_ than current, question our worldview.
                    if event_time < existing_grouped_delay.time:
                        raise RuntimeError('Did not find a matching event group.')

                    if event_type == existing_grouped_delay.type:
                        existing_grouped_delay.op_nodes.append(delay_edge.op_node)
                        break

            grouped_closed_delays.append(grouped_delay_edges)

        grouped_delay_events = grouped_closed_delays
        logger.info(pformat(grouped_delay_events))

        
        # Assume we've kept the time/event ordering from above.
        # Rely on this in block below.
        # So verify.
        for idx, grouped_delay in enumerate(grouped_delay_events):
            if idx < len(grouped_delay_events)-1:
                if grouped_delay.time > grouped_delay_events[idx+1].time:
                    raise RuntimeError('grouped_delay_events not ordered by time')

                if grouped_delay.time == grouped_delay_events[idx+1].time:
                    if (grouped_delay.type, grouped_delay_events[idx+1].type) != ('begin', 'end'):
                        raise RuntimeError('grouped_delay_events not ordered by event type')
        
        # Split grouped delays into blocks where we have largest openings.
        
        closed_joined_delays = []

        while grouped_delay_events:
            grouped_delay = grouped_delay_events.pop(0)
            
            # Find widest point, incr on open, decr on close
            current_open_qubits = set()
            current_open_op_nodes = []

            widest = [len(current_open_qubits),  # Width
                      None,                      # Begin GroupedDelayEvent
                      None,                      # End GroupedDelayEvent
                      current_open_op_nodes]     # List[DAGOpNode]

            for grouped_delay_event in grouped_delay_events:
                qubits = set(qarg for op_node in grouped_delay_event.op_nodes
                             for qarg in op_node.qargs)

                if grouped_delay_event.type == 'begin':
                    if not current_open_qubits.isdisjoint(qubits):
                        raise RuntimeError('re-opening open qubits?')

                    current_open_qubits.update(qubits)
                    current_open_op_nodes.extend(grouped_delay_event.op_nodes)

                    if len(current_open_qubits) > widest[0]:
                        widest = [len(current_open),
                                  grouped_delay_event,
                                  None,
                                  current_open_op_nodes]
                elif grouped_delay_event.type == 'end':
                    if not qubits.issubset(current_open_qubits):
                        raise RuntimeError('closing qubits that arent open')

                    if widest[2] is None:
                        widest[2] = grouped_delay_event

                    current_open_qubits -= qubits
                    for op_node in gropued_delay_event.op_nodes:
                        current_open_op_nodes.remove(op_node)
                else:
                    raise RuntimeError('bad event type?')

            new_delay_width, new_delay_begin, new_delay_end, replacing_op_nodes = widest
            if new_delay_width == 1:
                closed_joined_delays.append(widest)

                if len(grouped_delay_events) > 2:
                    raise RuntimeError('widest[0] == 1 but more than one open/close pair')
            else:
                new_delay_duration = new_delay_end.time - new_delay_begin.time

                if new_delay_duration >= MIN_JOINABLE_DELAY_DURATION:
                    closed_joined_delays.append(widest)

                    split_left_idx = grouped_delay_events.index(new_delay_begin)
                    split_right_idx = grouped_delay_events.index(new_delay_end)

                    if split_right_idx != split_left_idx + 1:
                        raise RuntimeError('widest begin and end non-adjacent')

                    new_left_delay_group = grouped_delay_events[:split_left_idx]
                    new_right_delay_group = grouped_delay_events[split_right_idx+1:]

                    new_left_delay_group.append(
                        GroupedDelayEvent(type='end',
                                          time=new_delay_begin.time,
                                          op_nodes=[op_node for op_node in replacing_op_nodes
                                                    if op_node not in new_delay_begin.op_nodes]))

                    new_right_delay_group.append(
                        GroupedDelayEvent(type='begin',
                                          time=new_delay_end.time,
                                          op_nodes=[op_node for op_node in replacing_op_nodes
                                                    if op_node not in new_delay_end.op_nodes]))

                    grouped_delay_events.append(new_left_delay_group)
                    grouped_delay_events.append(new_right_delay_group)
                    
                else:
                    split_left_idx = grouped_delay_events.index(new_delay_begin)
                    split_right_idx = grouped_delay_events.index(new_delay_end)

                    if split_right_idx != split_left_idx + 1:
                        raise RuntimeError('widest begin and end non-adjacent')

                    # Single out qubit which are opened by the 'begin' event and closed by the 'end'
                    # event so that they can be given 1q delays.
                    narrow_single_delay_qubits = set(qarg
                                                     for op_node in new_delay_begin.op_nodes
                                                     for qarg in qargs
                                                ).intersection(qarg
                                                               for op_node in new_delay_end.op_nodes
                                                               for qarg in qargs)

                    narrow_single_delay_op_nodes = [op_node for op_node in replacing_op_nodes
                                                    if narrow_single_delay_qubits.issuperset(op_node.qargs)]
                    new_delay_begin.op_nodes = narrow_single_delay_op_nodes
                    new_delay_end.op_nodes = narrow_single_delay_op_nodes
                    
                    narrow_single_delay = [len(narrow_single_delay_qubits),
                                           new_delay_begin,
                                           new_delay_end,
                                           new_single_delay_op_nodes]
                    closed_joined_delays.append(narrow_single_delay)

                    grouped_delay_events[split_left_idx] = GroupedDelayEvent(type='end',
                                                                             time=new_delay_end.time,
                                                                             op_nodes=[op_node for op_node in new_delay_end.op_nodes
                                                                                       if not narrow_single_delay_qubits.issuperset(op_node.qargs)])
                    
                    grouped_delay_events[split_right_idx] = GroupedDelayEvent(type='begin',
                                                                              time=new_delay_start.time,
                                                                              op_nodes=[op_node for op_node in new_delay_start.op_nodes
                                                                                       if not narrow_single_delay_qubits.issuperset(op_node.qargs)])

                    grouped_delay_events.append(grouped_delay_events[:split_left_idx-1])
                    grouped_delay_events.append(grouped_delay_events[split_right_idx:])

        @dataclass
        class ReplacementDelay:
            new_delay_op: Delay
            start_time: int
            end_time: int
            replacing_delay_nodes: List[DAGOpNode]  # List of DelayOpNodes on the original DAG which overlap with this new delay.
                    
        #closed_joined_delays
        #             widest = [len(current_open_qubits),  # Width
                      # None,                      # Begin GroupedDelayEvent
                      # None,                      # End GroupedDelayEvent
                      # current_open_op_nodes]     # List[DAGOpNode]

        closed_delays = []
        for widest in closed_joined_delays:
            start_time = widest[1].time
            end_time = widest[2].time

            replacement = ReplacementDelay(
                Delay(end_time - start_time),
                start_time,
                end_time,
                widest[3])
            replacement.delay.num_qubits = len(qarg for op_node in widest[3] for qarg in op_node.qargs)

            closed_delay.append(replacement)

        # Replacement strategy:
        # 1) Split each doomed delay in to N 1-q delays with duration of multi-qubit delays
        # 2) Combine blocks of new multi-qubit delays

        # Create lookup from delay to replacements sorted by start time

        replacement_ops = defaultdict(list)  # existing_delay_node, [sorted_list_of_replacement_delays]
        for closed_delay in sorted(closed_delays, key=lambda k: (k.start_time, k.end_time)):  # Might already be in right order?
            for replaced_delay_node in closed_delay.replacing_delay_nodes:
                replacement_ops[replaced_delay_nodes].append(closed_delay)

        delay_op_placeholders = defaultdict(list)

        for doomed_delay_node, replacements in replacement_ops.items():
            oneq_delay = QuantumCircuit(1)
            for replacement in replacements:
                oneq_delay.delay(replacement.new_delay_op.duration, 0)

            oneq_dag = circuit_to_dag(oneq_delay)
            oneq_delay_node_ids = [ node._node_id for node in oneq_dag.topological_op_nodes()]

            out_node_map = dag.substitute_node_with_dag(doomed_delay_node, oneq_dag)
            out_oneq_delay_nodes = [out_node_map[node_id] for node_id in oneq_delay_node_ids]

            for replacement, out_oneq_delay_node in zip(replacements, out_oneq_delay_nodes):
                delay_op_placeholders[id(replacement.new_delay_op)].append(out_oneq_delay_node)

        for replacement in closed_delays:
        # for (delay_op, start_time, end_time, replacing_delay_nodes) in closed_delays:
            doomed_placeholder_nodes = delay_op_placeholders[id(replacement.new_delay_op)]
            dag.replace_block_with_op(
                doomed_placeholder_nodes,
                delay_op,
                {node.qargs[0]: idx for idx, node in enumerate(doomed_placeholder_nodes)},
                cycle_check=True)

        return dag
