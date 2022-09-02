import logging

from pprint import pformat, pprint
from dataclasses import dataclass
from collections import defaultdict

from typing import List, Set

from qiskit.circuit import Delay, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import TransformationPass, CouplingMap

logger = logging.getLogger(__name__)

MIN_JOINABLE_DELAY_DURATION = 2500


class CombineAdjacentDelays(TransformationPass):
    def __init__(self, cmap):
        self.cmap = cmap
        super().__init__()


    def run(self, dag):
        bit_idx_locs = {bit: idx for idx, bit in enumerate(dag.qubits)}

        @dataclass
        class DelayEvent:
            "Represent the start or eding event for each delay in the input circuit."
            type: str  # One of 'begin' or 'end'
            time: int  # Time in the circuit, in dt
            op_node: DAGOpNode  # The node for the circuit delay

            @property
            def qargs(self):
                return set(op_node.qargs)

            @property
            @staticmethod
            def sort_lambda():
                return lambda event: (
                    event.time,                        # Sort by event time
                    0 if event.type == 'end' else 1,  # With 'end' events before 'begin'
                )
            
        # Find and sort every leading/trailing edge of a delay in the circuit.
        # These will be the places we'll examine to split/combine delay ops.
        sorted_delay_events = sorted(
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
            key=DelayEvent.sort_lambda
        )

        logger.info(f'sorted_delay_events: {pformat(sorted_delay_events)}')

        # Now, still collect open and closed events to group into idle adjacent idle time
        # Might want to sort to say which is widest adjacnet group.
        # Start there: if it's width is > MIN_JOIN, make a delay, add to output remove from input.
        # If it's less than MIN_JOIN, panic.
        # Look at start and end qubit, if same, remove it, and give it its own 1q delay. Re-insert remainder in queue.
        # If start and end are different (will probably need to group these to multi-q) so above may be "find intersection"
        # Will be hard to know what to do next. Could split L and split R, and maybe this is the best since we're looking
        # for "widest delays only".
        #
        # 

        # Collect grouped delays by concurrency in time and adjacencey on device.

        @dataclass
        class AdjacentDelayBlock:
            events: List[DelayEvent]
            active_qubits: Set[Qubit]

        open_delay_blocks = []
        closed_delay_blocks = []

        def _open_delay_block(delay_event):
            open_delay_blocks.append(
                AdjacentDelayBlock(events=[delay_event],
                                   active_qubits=set(delay_event.qargs)))
            return open_delay_blocks[-1]

        def _update_delay_block(open_delay_block, delay_event):
            open_delay_block.events.append(delay_event)

            if delay_event.type == 'begin':
                open_delay_block.active_qubits += set(delay_event.qargs)
            elif delay_event.type == 'end':
                open_delay_block.active_qubits -= set(delay_event.qargs)
            else:
                raise RuntimeError('bad event type')

            if not open_delay_block.active_qubits:
                open_delay_blocks.remove(open_delay_block)
                closed_delay_blocks.append(open_delay_block)

        def _combine_delay_blocks(open_delay_blocks):
            survivor, *doomed = open_delay_blocks

            for doomed_delay_group in doomed:
                if survivor.active_qubits.intersection(doomed_delay_group.active_qubits):
                    raise RuntimeError('More than one open delay on a qubit?')

                survivor.events.extend(doomed_delay_group.events)
                survivor.active_qubits.update(doomed_delay_group.active_qubits)

                open_delay_blocks.remove(doomed_delay_group)
                survivor.events.sort(key=DelayEvent.sort_lambda)  # Maintain sorted event order


        for delay_edge in sorted_delay_events:
            adjacent_open_delay_blocks = [open_delay for open_delay in open_delay_blocks
                                    if any(
                                            self.cmap.distance(
                                                bit_idx_locs[delay_edge.op_node.qargs[0]],
                                                bit_idx_locs[open_delay_qubit]) <= 1
                                            for open_delay_qubit in open_delay.active_qubits)]

            if delay_edge.type == 'begin':
                # If crossing a begin edge, check if there are any open delays that are adjacent.
                # If so, add current event to that group.
            
                if len(adjacent_open_delay_blocks) == 0:
                    # Make a new delay
                    _open_delay_block(delay_edge)
                else:
                    # Make a new block and combine that with this
                    new_block = _open_delay_block(delay_edge)
                    _combine_delay_blocks(adjacent_open_delay_blocks + [new_block])

            if delay_edge.type == 'end':
                # If crossing a end edge, remove this qubit from the actively delaying qubits"
                if len(adjacent_open_delay_blocks) != 1:
                    import pdb; pdb.set_trace()
                    raise Exception("closing edge w/o an open delay?")
                else:
                    _close_delay_block(adjacent_open_delay_blocks[0], delay_edge)

            logger.info(pformat((
                '---exit---',
                delay_edge,
                ('open_delays: ', open_delay_blocks),
                ('closed_delays: ', closed_delay_blocks),
            )))

        logger.info(pformat((
            '---post_collect---',
            ('open_delays: ', open_delay_blocks),
            ('len(closed_delays): ', len(closed_delay_blocks)),
        )))

        if open_delay_blocks:
            raise RuntimeError('Failed to close all open delays.')
        for closed_delay in closed_delay_blocks:
            if closed_delay.active_qubits:
                raise RuntimeError('Failed remove active qubits on closed delays.')

        adjacent_delay_blocks = closed_delay_blocks

        @dataclass
        class AdjacentGroupedDelayBlock:
            events:  List[GroupedDelayEvent]

        @dataclass
        class GroupedDelayEvent:
            type: str
            time: int
            _op_nodes: List[DAGOpNode]

            def __init__(self, type, time, _op_nodes):
                self.type = type
                self.time = time
                self.op_nodes = _op_nodes

            @property
            def op_nodes(self):
                return self._op_nodes

            @op_nodes.setter
            def op_nodes(self, val):
                if not val:
                    import pdb; pdb.set_trace()
                self._op_nodes = val

            @property
            def qargs(self):
                return set(qarg for op_node in self.op_nodes
                           for qargs in op_node.qargs)
        
        # Within each adjacnet delay block, collect events by time, type.
        
        grouped_delay_blocks = []

        for adjacent_delay_block in adjacent_delay_blocks:
            grouped_delay_block = AdjacentGroupedDelayBlock([])

            for delay_event in adjacent_delay_block.events:
                # If we're on the first edge, it gets a pass.
                if not grouped_delay_block.events:
                    grouped_delay_block.events.append(
                        GroupedDelayEvent(delay_event.type,
                                          delay_event.time,
                                          [delay_event.op_node]))
                    continue

                # Look at last known group.
                # KDK do we need to check adjacency here?
                # i.e. even though we're in a block, not all event groups will be adjacent
                # KDK Maybe dump this grouping, it seems to make life hard and doesn't solve any problems (because we'll still need to check adjacency down the road.
                for existing_grouped_delay in reversed(grouped_delay_events):
                    # If newest grouped event is earlier than current, need a new group.
                    if delay_event.time > existing_grouped_delay.time:
                        grouped_delay_block.events.append(
                            GroupedDelayEvent(delay_event.type,
                                              delay_event.time,
                                              [delay_event.op_node]))
                        break

                    # If newest grouped event is _later_ than current, question our worldview.
                    if delay_event.time < existing_grouped_delay.time:
                        raise RuntimeError('Did not find a matching event group.')

                    if delay_event.type == existing_grouped_delay.type:
                        existing_grouped_delay.op_nodes.append(delay_event.op_node)
                        break


            grouped_delay_blocks.append(grouped_delay_block)

        logger.info(pformat(grouped_delay_blocks))

        
        # Assume we've kept the time/event ordering from above.
        # Rely on this in block below.
        # So verify.
        for grouped_closed_delay in grouped_closed_delays:
            for idx, grouped_delay in enumerate(grouped_closed_delay):
                if idx < len(grouped_closed_delay)-1:
                    if grouped_delay.time > grouped_closed_delay[idx+1].time:
                        raise RuntimeError('grouped_delay_events not ordered by time')

                    if grouped_delay.time == grouped_closed_delay[idx+1].time:
                        if (grouped_delay.type, grouped_closed_delay[idx+1].type) != ('begin', 'end'):
                            raise RuntimeError('grouped_closed_delay not ordered by event type')

        # realization: grouped_closed_delays is a list of all the delay blocks in our circuit
        # within those lists is a list of groupeddelayevents for every edge
        # we had renamed grouped_closed_delays to grouped_delay_events above, but that was wrong
        # because at this stage, still have a list of blocks, need to update the below

        
        # KDK below super confusing becuase we have *closed* from above
        # And we're closing again below, but call both closed
        
        # Split grouped delays into blocks where we have largest openings.
        logger.info(pformat((
            '---post_group_by_time/event---',
            ('grouped_closed_delays: ', pformat(grouped_closed_delays)),
            ('len(closed_delays): ', len(closed_delay_groups)),
        )))

        logger.info('Begin finding and isolating widest delays within groups.')

        closed_joined_delays = []

        while grouped_closed_delays:
            grouped_delay_events = grouped_closed_delays.pop(0)

            logger.info(f'Examining grouped_delay_events: {pformat(grouped_delay_events)}')

            # Find widest point, incr on open, decr on close
            current_open_qubits = set()
            current_open_op_nodes = []

            widest = [len(current_open_qubits),  # Width
                      None,                      # Begin GroupedDelayEvent
                      None,                      # End GroupedDelayEvent
                      current_open_op_nodes.copy()]     # List[DAGOpNode]

            for grouped_delay_event in grouped_delay_events:
                logger.debug(f'Considering event: {pformat(grouped_delay_event)} '
                             f'with widest at entry: {widest}')

                qubits = set(qarg for op_node in grouped_delay_event.op_nodes
                             for qarg in op_node.qargs)

                if grouped_delay_event.type == 'begin':
                    if not current_open_qubits.isdisjoint(qubits):
                        raise RuntimeError('re-opening open qubits?')

                    current_open_qubits.update(qubits)
                    current_open_op_nodes.extend(grouped_delay_event.op_nodes)

                    if len(current_open_qubits) > widest[0]:
                        widest = [len(current_open_qubits),
                                  grouped_delay_event,
                                  None,
                                  current_open_op_nodes.copy()]
                elif grouped_delay_event.type == 'end':
                    if not qubits.issubset(current_open_qubits):
                        raise RuntimeError('closing qubits that arent open')

                    if widest[2] is None:
                        widest[2] = grouped_delay_event

                    current_open_qubits -= qubits
                    for op_node in grouped_delay_event.op_nodes:
                        current_open_op_nodes.remove(op_node)
                else:
                    raise RuntimeError('bad event type?')

            logger.info(f'Found widest segment: {widest}')

            new_delay_width, new_delay_begin, new_delay_end, replacing_op_nodes = widest
            if new_delay_width == 1:
                logger.info('Widest segment has width 1. Adding to closed delays.')
                closed_joined_delays.append(widest)

                if len(grouped_delay_events) > 2:
                    raise RuntimeError('widest[0] == 1 but more than one open/close pair')
            elif new_delay_end is None:
                import pdb; pdb.set_trace()
                raise RuntimeError(f'widest was never closed?: {widest}')
            else:
                new_delay_duration = new_delay_end.time - new_delay_begin.time

                if new_delay_duration >= MIN_JOINABLE_DELAY_DURATION:
                    closed_joined_delays.append(widest)

                    split_left_idx = grouped_delay_events.index(new_delay_begin)
                    split_right_idx = grouped_delay_events.index(new_delay_end)

                    logger.info('Widest exceeds MIN_JOINABLE_DELAY_DURATION, '
                                'splitting group_delay_events at index '
                                f'{split_left_idx} and {split_right_idx}.')

                    if split_right_idx != split_left_idx + 1:
                        raise RuntimeError('widest begin and end non-adjacent')

                    new_left_delay_group = grouped_delay_events[:split_left_idx]
                    new_right_delay_group = grouped_delay_events[split_right_idx+1:]

                    if split_left_idx > 0:
                        new_left_delay_group.append(
                            GroupedDelayEvent(type='end',
                                              time=new_delay_begin.time,
                                              _op_nodes = [op_node for op_node in replacing_op_nodes
                                                          if op_node not in new_delay_begin.op_nodes]))
                        logger.info('Creating group from left-of widest: '
                                    f'{pformat(new_left_delay_group)}')
                        if not new_left_delay_group:
                            import pdb; pdb.set_trace()
                        grouped_closed_delays.append(new_left_delay_group)

                    if split_right_idx < len(grouped_delay_events) - 1:

                        new_right_delay_group.insert(0,
                            GroupedDelayEvent(type='begin',
                                              time=new_delay_end.time,
                                              _op_nodes=[op_node for op_node in replacing_op_nodes
                                                        if op_node not in new_delay_end.op_nodes]))
                        logger.info('Creating group from right-of widest: '
                                    f'{pformat(new_right_delay_group)}')

                        if not new_right_delay_group:
                            import pdb; pdb.set_trace()
                        grouped_closed_delays.append(new_right_delay_group)
                    
                else:
                    split_left_idx = grouped_delay_events.index(new_delay_begin)
                    split_right_idx = grouped_delay_events.index(new_delay_end)
                    logger.info(f'Widest segment duration {new_delay_duration} '
                                f'shorter than MIN_JOINABLE_DELAY_DURATION {MIN_JOINABLE_DELAY_DURATION}.'
                                'Splitting group_delay_events at index '
                                f'{split_left_idx} and {split_right_idx}.')

                    if split_right_idx != split_left_idx + 1:
                        raise RuntimeError('widest begin and end non-adjacent')

                    # Single out qubit which are opened by the 'begin' event and closed by the 'end'
                    # event so that they can be given 1q delays.
                    narrow_single_delay_qubits = set(qarg
                                                     for op_node in new_delay_begin.op_nodes
                                                     for qarg in op_node.qargs
                                                ).intersection(qarg
                                                               for op_node in new_delay_end.op_nodes
                                                               for qarg in op_node.qargs)

                    # If there were qubits in new_delay_begin/new_delay_end that didn't get cut out into singles
                    # we need to retain their begin/end events
                    residual_begin_qubits = set(qarg
                                                for op_node in new_delay_begin.op_nodes
                                                for qarg in op_node.qargs
                                                if qarg not in narrow_single_delay_qubits)

                    residual_end_qubits = set(qarg
                                              for op_node in new_delay_end.op_nodes
                                              for qarg in op_node.qargs
                                              if qarg not in narrow_single_delay_qubits)

                    if narrow_single_delay_qubits:
                        narrow_single_delay_op_nodes = [op_node for op_node in replacing_op_nodes
                                                        if narrow_single_delay_qubits.issuperset(op_node.qargs)]

                        single_delay_begin = GroupedDelayEvent(
                            type='begin',
                            time = new_delay_begin.time,
                            _op_nodes = narrow_single_delay_op_nodes)

                        single_delay_end = GroupedDelayEvent(
                            type='end',
                            time = new_delay_end.time,
                            _op_nodes = narrow_single_delay_op_nodes)
                    
                        narrow_single_delay = [len(narrow_single_delay_qubits),
                                               single_delay_begin,
                                               single_delay_end,
                                               narrow_single_delay_op_nodes]
                        logger.info('Adding single-qubit delays on splintered qubits. '
                                    f'{pformat(narrow_single_delay)}')

                        closed_joined_delays.append(narrow_single_delay)

                    new_left_delay_group = grouped_delay_events[:split_left_idx]
                    new_right_delay_group = grouped_delay_events[split_right_idx+1:]

                    if residual_end_qubits:
                        new_left_delay_group.append()
                    # KDK Having these merged seems like more pain than it's worth
                    # e.g. if I have residual end qubits, do I need to merge them into a single GroupedDElayEvent with below, if it exists?
                    # which make the split_left_idx > 0 look real suspicious
                    # Also, are replacing op nodes... ever multi-qubit? I wouldn't think so but \_(o,o)-/
                    
                    if split_left_idx > 0:
                        new_left_delay_group.append(GroupedDelayEvent(type='end',
                                                                      time=new_delay_end.time,
                                                                      _op_nodes=[op_node for op_node in replacing_op_nodes
                                                                                 if not narrow_single_delay_qubits.issuperset(op_node.qargs)]))

                    if new_left_delay_group:
                        logger.info('Creating group from left-of widest: '
                                    f'{pformat(new_left_delay_group)}')

                        grouped_closed_delays.append(new_left_delay_group)

                    if split_right_idx < len(grouped_delay_events) - 1:
                        new_right_delay_group.insert(0,
                                                     GroupedDelayEvent(type='begin',
                                                                       time=new_delay_begin.time,
                                                                       _op_nodes=[op_node for op_node in replacing_op_nodes
                                                                                  if not narrow_single_delay_qubits.issuperset(op_node.qargs)]
))

                    if new_right_delay_group:
                        logger.info('Creating group from right-of widest: '
                                    f'{pformat(new_right_delay_group)}')

                        grouped_closed_delays.append(new_right_delay_group)

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
            replacement.new_delay_op.num_qubits = len([qarg for op_node in widest[3] for qarg in op_node.qargs])

            closed_delays.append(replacement)

        # Replacement strategy:
        # 1) Split each doomed delay in to N 1-q delays with duration of multi-qubit delays
        # 2) Combine blocks of new multi-qubit delays

        # Create lookup from delay to replacements sorted by start time

        replacement_ops = defaultdict(list)  # existing_delay_node, [sorted_list_of_replacement_delays]
        for closed_delay in sorted(closed_delays, key=lambda k: (k.start_time, k.end_time)):  # Might already be in right order?
            for replaced_delay_node in closed_delay.replacing_delay_nodes:
                replacement_ops[replaced_delay_node].append(closed_delay)

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
                replacement.new_delay_op,
                {node.qargs[0]: idx for idx, node in enumerate(doomed_placeholder_nodes)},
                cycle_check=True)

        return dag
