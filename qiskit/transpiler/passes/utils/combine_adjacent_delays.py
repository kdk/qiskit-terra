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


class CombineAdjacentDelays(TransformationPass):
    def __init__(self, cmap, min_joinable_delay_duration=None):
        self.cmap = cmap

        self._min_joinable_delay_duration = min_joinable_delay_duration or 0
        super().__init__()


    def run(self, dag):
        bit_idx_locs = {bit: idx for idx, bit in enumerate(dag.qubits)}

        @dataclass
        class DelayEvent:
            "Represent the start or ending event for each delay in the input circuit."
            type: str  # One of 'begin' or 'end'
            time: int  # Time in the circuit, in dt
            op_node: DAGOpNode  # The node for the circuit delay

            @property
            def qargs(self):
                return set(self.op_node.qargs)

            @staticmethod
            def sort_key(event):
                return (
                    event.time,                       # Sort by event time
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
                    and op_node.op.duration > self._min_joinable_delay_duration
                )
                for event_type in ('begin', 'end')
            ),
            key=DelayEvent.sort_key
        )

        logger.debug(f'sorted_delay_events: {pformat(sorted_delay_events)}')

        # Collect grouped delays by concurrency in time and adjacency on device.

        @dataclass
        class AdjacentDelayBlock:
            "Group of circuit delays which are collectively adjacent in time and on device."
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
            "Add another delay event to an existing block to either extend or close it."
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

        def _combine_delay_blocks(delay_blocks):
            survivor, *doomed = delay_blocks

            for doomed_delay_group in doomed:
                # Add events and qubits from doomed block to survivor.
                if survivor.active_qubits.intersection(doomed_delay_group.active_qubits):
                    raise RuntimeError('More than one open delay on a qubit?')

                survivor.events.extend(doomed_delay_group.events)
                survivor.active_qubits.update(doomed_delay_group.active_qubits)

                open_delay_blocks.remove(doomed_delay_group)
                survivor.events.sort(key=DelayEvent.sort_key)  # Maintain sorted event order

        for delay_event in sorted_delay_events:
            # This could be avoided by keeping a map of device qubit to open
            # block and only considering neighbors of current event.
            adjacent_open_delay_blocks = [open_delay for open_delay in open_delay_blocks
                                    if any(
                                            self.cmap.distance(
                                                bit_idx_locs[delay_event.op_node.qargs[0]],
                                                bit_idx_locs[open_delay_qubit]) <= 1
                                            for open_delay_qubit in open_delay.active_qubits)]

            if delay_event.type == 'begin':
                # If crossing a begin edge, check if there are any open delays that are adjacent.
                # If so, add current event to that group.
            
                if len(adjacent_open_delay_blocks) == 0:
                    # Make a new delay
                    _open_delay_block(delay_event)
                else:
                    # Make a new block and combine that with adjacent open blocks
                    new_block = _open_delay_block(delay_event)
                    _combine_delay_blocks(adjacent_open_delay_blocks + [new_block])

            if delay_event.type == 'end':
                # If crossing a end edge, remove this qubit from the actively delaying qubits"
                if len(adjacent_open_delay_blocks) != 1:
                    import pdb; pdb.set_trace()
                    raise Exception("closing edge w/o an open delay?")
                else:
                    _update_delay_block(adjacent_open_delay_blocks[0], delay_event)

            logger.debug(pformat((
                '---exit---',
                delay_event,
                ('open_delays: ', open_delay_blocks),
                ('closed_delays: ', closed_delay_blocks),
            )))

        logger.debug(pformat((
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

        # Assume we've kept the time/event ordering from above.
        # Rely on this in block below.
        # So verify.
        for adjacent_delay_block in adjacent_delay_blocks:
            for idx, event in enumerate(adjacent_delay_block.events):
                if idx < len(adjacent_delay_block.events)-1:
                    if event.time > adjacent_delay_block.events[idx+1].time:
                        raise RuntimeError('adjacent_delay_block.events not ordered by time')

                    if event.time == adjacent_delay_block.events[idx+1].time:
                        # At same time, can either be ('begin', 'begin'), ('begin', 'end') or ('end', 'end')
                        if (event.type, adjacent_delay_block.events[idx+1].type) == ('end', 'begin'):
                            raise RuntimeError('adjacent_delay_block.events not ordered by event type')


        # Within each adjacent delay block, collect events by time, type.
        # This will faciltate knowing where to split them into joint N-qubit delays.

        @dataclass
        class GroupedDelayEvent:
            "Grouped event of single type and time, but spanning one or more qubits/Delay instructions."
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

        @dataclass
        class AdjacentGroupedDelayBlock:
            "Collection of circuit delay groups which are collectively adjacent in time and on device."
            events:  List[GroupedDelayEvent]

            def validate(self):
                # events in order start end, type
                last_time=-1
                last_type = None
                for grouped_delay_event in self.events:
                    if grouped_delay_event.time > last_time:
                        last_time = grouped_delay_event.time
                        continue

                    if grouped_delay_event.time < last_time:
                        raise RuntimeError('out of time order')

                    if (last_type, grouped_delay_event.type) != ('end', 'begin'):
                        raise RuntimeError('out of type order')

                # every qubit has matching start/end pairs
                all_qubits = set(qarg for event in self.events for qarg in event.qargs)

                for qubit in all_qubits:
                    qubit_event_types = [event.type for event in self.events if qubit in event.qargs]

                    if len(qubit_event_types) % 2 != 0:
                        raise RuntimeError('odd number of qubit events')

                    if qubit_event_types != ['begin', 'end'] * len(qubit_event_types)/2:
                        raise RuntimeError('qubit events dont strictly follow open/close/open/close/...')

                return self

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

                # Look at last known grouped event.
                # KDK do we need to check adjacency here?
                # i.e. even though we're in a block, not all event groups will be adjacent
                # KDK Maybe dump this grouping, it seems to make life hard and doesn't solve any problems (because we'll still need to check adjacency down the road.
                newest_grouped_delay = grouped_delay_block.events[-1]

                # If newest grouped event is earlier than current, need a new group.
                if delay_event.time > newest_grouped_delay.time:
                    grouped_delay_block.events.append(
                        GroupedDelayEvent(delay_event.type,
                                          delay_event.time,
                                          [delay_event.op_node]))
                    continue

                # If newest grouped event is _later_ than current, question our worldview.
                if delay_event.time < newest_grouped_delay.time:
                    raise RuntimeError('Did not find a matching event group.')

                # If newest grouped has same time and type, add current event to that group.
                if delay_event.type == newest_grouped_delay.type:
                    newest_grouped_delay.op_nodes.append(delay_event.op_node)
                    continue

                #If we've passed all the above, have same time but different type, so add a new group.
                grouped_delay_block.events.append(
                    GroupedDelayEvent(delay_event.type,
                                      delay_event.time,
                                      [delay_event.op_node]))

            grouped_delay_blocks.append(grouped_delay_block)

        logger.debug(pformat((
            '---post_group_by_time/event---',
            ('grouped_delays_blocks: ', grouped_delay_blocks),
            ('len(grouped_delay_blocks): ', len(grouped_delay_blocks)),
        )))

        # Assume we've kept the time/event ordering from above.
        # Rely on this in block below.
        # So verify.
        for grouped_delay_block in grouped_delay_blocks:
            for idx, grouped_delay_event in enumerate(grouped_delay_block.events):
                if idx < len(grouped_delay_block.events)-1:
                    if grouped_delay_event.time > grouped_delay_block.events[idx+1].time:
                        raise RuntimeError('grouped_delay_block not ordered by time')

                    if grouped_delay_event.time == grouped_delay_block.events[idx+1].time:
                        if (grouped_delay_event.type, grouped_delay_block.events[idx+1].type) != ('begin', 'end'):
                            raise RuntimeError('grouped_delay_block not ordered by event type')

        logger.info('Begin finding and isolating widest delays within groups.')

        # Walk through existing delay blocks, finding interval where the most adjacent qubits are joinly idle.
        # If that interval is longer than self._min_joinable_delay_duration, combine events which begin and end it
        # into a new AdjacentGroupedDelayBlock and add to output.
        # Split remainder of original delay block into two new delay blocks and add those onto queue to be
        # further split if possible.

        @dataclass
        class CollectedJointDelay:
            "Collection of Delay ops from input circuit to be output as a joint delay."
            num_qubits: int
            begin_delay_event: GroupedDelayEvent
            end_delay_event: GroupedDelayEvent
            open_op_nodes: List[DAGOpNode]  # N.B. These op_nodes in general will extend beyond {begin,end}_delay_event.

            def __iter__(self):
                return iter((self.num_qubits,
                             self.begin_delay_event,
                             self.end_delay_event,
                             self.open_op_nodes))

        closed_joint_delays = []

        while grouped_delay_blocks:
            grouped_delay = grouped_delay_blocks.pop(0)

            logger.debug(f'Examining grouped_delay: {pformat(grouped_delay)}')

            # Find widest point, incr on open, decr on close

            current_open_qubits = set()
            current_open_op_nodes = []

            widest = CollectedJointDelay(
                num_qubits = len(current_open_qubits),
                begin_delay_event = None,
                end_delay_event = None,
                open_op_nodes = current_open_op_nodes.copy())

            for grouped_delay_event in grouped_delay.events:
                logger.debug(f'Considering event: {pformat(grouped_delay_event)} '
                             f'with widest at entry: {widest}')

                event_qubits = set(qarg for op_node in grouped_delay_event.op_nodes
                                   for qarg in op_node.qargs)

                if grouped_delay_event.type == 'begin':
                    if not current_open_qubits.isdisjoint(event_qubits):
                        raise RuntimeError('re-opening open qubits?')

                    current_open_qubits.update(event_qubits)
                    current_open_op_nodes.extend(grouped_delay_event.op_nodes)

                    if len(current_open_qubits) > widest.num_qubits:
                        widest = CollectedJointDelay(
                            num_qubits = len(current_open_qubits),
                            begin_delay_event = grouped_delay_event,
                            end_delay_event = None,
                            open_op_nodes = current_open_op_nodes.copy())

                elif grouped_delay_event.type == 'end':
                    if not event_qubits.issubset(current_open_qubits):
                        import pdb; pdb.set_trace()
                        raise RuntimeError('closing qubits that arent open')

                    if widest.end_delay_event is None:
                        widest.end_delay_event = grouped_delay_event

                    current_open_qubits -= event_qubits
                    for op_node in grouped_delay_event.op_nodes:
                        current_open_op_nodes.remove(op_node)
                else:
                    raise RuntimeError('bad event type?')

            logger.debug(f'Found widest interval: {widest}')

            new_delay_width, new_delay_begin, new_delay_end, replacing_op_nodes = widest

            if new_delay_width == 1:
                logger.debug('Widest segment has width 1. Adding to closed delays.')
                closed_joint_delays.append(widest)

                if len(grouped_delay.events) > 2:
                    raise RuntimeError('widest[0] == 1 but more than one open/close pair')
            elif new_delay_end is None:
                import pdb; pdb.set_trace()
                raise RuntimeError(f'widest was never closed?: {widest}')
            else:
                new_delay_duration = new_delay_end.time - new_delay_begin.time

                if new_delay_duration >= self._min_joinable_delay_duration:
                    closed_joint_delays.append(widest)

                    split_left_idx = grouped_delay.events.index(new_delay_begin)
                    split_right_idx = grouped_delay.events.index(new_delay_end)

                    logger.debug('Widest exceeds self._min_joinable_delay_duration, '
                                 'splitting group_delay_events at index '
                                 f'{split_left_idx} and {split_right_idx}.')

                    if split_right_idx != split_left_idx + 1:
                        raise RuntimeError('widest begin and end non-adjacent')

                    new_left_delay_group = AdjacentGroupedDelayBlock(grouped_delay.events[:split_left_idx])
                    new_right_delay_group = AdjacentGroupedDelayBlock(grouped_delay.events[split_right_idx+1:])

                    if split_left_idx > 0:
                        new_left_delay_group.events.append(
                            GroupedDelayEvent(type='end',
                                              time=new_delay_begin.time,
                                              _op_nodes = [op_node for op_node in replacing_op_nodes
                                                          if op_node not in new_delay_begin.op_nodes]))
                        logger.debug('Creating group from left-of widest: '
                                     f'{pformat(new_left_delay_group)}')
                        if not new_left_delay_group:
                            import pdb; pdb.set_trace()
                        grouped_delay_blocks.append(new_left_delay_group)

                    if split_right_idx < len(grouped_delay.events) - 1:

                        new_right_delay_group.events.insert(0,
                            GroupedDelayEvent(type='begin',
                                              time=new_delay_end.time,
                                              _op_nodes=[op_node for op_node in replacing_op_nodes
                                                        if op_node not in new_delay_end.op_nodes]))
                        logger.debug('Creating group from right-of widest: '
                                     f'{pformat(new_right_delay_group)}')

                        if not new_right_delay_group:
                            import pdb; pdb.set_trace()
                        grouped_delay_blocks.append(new_right_delay_group)
                    
                else:
                    # If new_delay_duration is shorter than self._min_joinable_delay_duration,
                    # attempt to split current group into two grouped delays which will
                    # admit the widest possible delay when they are each processed.

                    split_left_idx = grouped_delay.events.index(new_delay_begin)
                    split_right_idx = grouped_delay.events.index(new_delay_end)

                    logger.debug(f'Widest segment duration {new_delay_duration} '
                                 f'shorter than self._min_joinable_delay_duration {self._min_joinable_delay_duration}.'
                                 'Splitting group_delay_events at index '
                                 f'{split_left_idx} and {split_right_idx}.')

                    if split_right_idx != split_left_idx + 1:
                        raise RuntimeError('widest begin and end non-adjacent')

                    opening_event_qubits = set(qarg for op_node in new_delay_begin.op_nodes
                                               for qarg in op_node.qargs)
                    closing_event_qubits = set(qarg for op_node in new_delay_end.op_nodes
                                               for qarg in op_node.qargs)
                    # If there are qubits which both opened and closed this delay group,
                    # they must have a delay < self._min_joinable_delay_duration, so pull them out of
                    # this group and give them all 1q Delays.

                    narrow_single_delay_qubits = opening_event_qubits.intersection(closing_event_qubits)

                    for single_delay_qubit in narrow_single_delay_qubits:
                        narrow_single_delay_op_nodes = [op_node for op_node in replacing_op_nodes
                                                        if single_delay_qubit in op_node.qargs]

                        single_delay_begin = GroupedDelayEvent(
                            type='begin',
                            time = new_delay_begin.time,
                            _op_nodes = narrow_single_delay_op_nodes)

                        single_delay_end = GroupedDelayEvent(
                            type='end',
                            time = new_delay_end.time,
                            _op_nodes = narrow_single_delay_op_nodes)
                    
                        narrow_single_delay = CollectedJointDelay(
                            num_qubits = 1,
                            begin_delay_event = single_delay_begin,
                            end_delay_event = single_delay_end,
                            open_op_nodes = narrow_single_delay_op_nodes)

                        logger.debug('Adding single-qubit delays on splintered qubit. '
                                     f'{pformat(narrow_single_delay)}')

                        closed_joint_delays.append(narrow_single_delay)

                    # If there were qubits in new_delay_begin/new_delay_end that didn't get cut out into singles
                    # retain their begin/end events
                    residual_begin_qubits = opening_event_qubits - narrow_single_delay_qubits
                    residual_end_qubits = closing_event_qubits - narrow_single_delay_qubits

                    new_left_delay_group = AdjacentGroupedDelayBlock(grouped_delay.events[:split_left_idx])
                    new_right_delay_group = AdjacentGroupedDelayBlock(grouped_delay.events[split_right_idx+1:])

                    if residual_end_qubits:
                        if split_left_idx == 0:
                            raise RuntimeError('')

                        new_left_delay_group.events.append(
                            GroupedDelayEvent(type='end',
                                              time=new_delay_end.time,
                                              _op_nodes=[op_node for op_node in replacing_op_nodes
                                                         if residual_end_qubits.intersection(op_node.qargs)]))

                    if residual_begin_qubits:
                        if split_right_idx == len(grouped_delay.events):
                            raise RuntimeError('')

                        new_right_delay_group.events.insert(
                            0,
                            GroupedDelayEvent(type='begin',
                                              time=new_delay_begin.time,
                                              _op_nodes=[op_node for op_node in replacing_op_nodes
                                                         if residual_begin_qubits.intersection(op_node.qargs)]))

                    if new_left_delay_group.events:
                        logger.debug('Creating group from left-of widest: '
                                    f'{pformat(new_left_delay_group)}')

                        grouped_delay_blocks.append(new_left_delay_group)

                    if new_right_delay_group.events:
                        logger.debug('Creating group from right-of widest: '
                                    f'{pformat(new_right_delay_group)}')

                        grouped_delay_blocks.append(new_right_delay_group)

        collected_joint_delays = closed_joint_delays

        logger.debug(pformat((
            'collected_joint_delays',
            collected_joint_delays)))

        # Create new N-Qubit delay ops for joint delays.

        @dataclass
        class ReplacementDelay:
            new_delay_op: Delay
            start_time: int
            end_time: int
            replacing_delay_nodes: List[DAGOpNode]  # List of DelayOpNodes on the original DAG which overlap with this new delay.

        replacement_delays = []
        for widest in collected_joint_delays:
            start_time = widest.begin_delay_event.time
            end_time = widest.end_delay_event.time

            replacement = ReplacementDelay(
                Delay(end_time - start_time),
                start_time,
                end_time,
                widest.open_op_nodes)
            replacement.new_delay_op.num_qubits = widest.num_qubits

            replacement_delays.append(replacement)

        # Replacement strategy:
        # 1) Split each doomed delay in to N single-q placeholder delays with duration of multi-qubit delays
        # 2) Combine blocks of new multi-qubit delays

        # Have mapping from new joint delay ops to original delays they replace, but
        # need inverse mapping (original delay to replacements) to know lengths of
        # placeholder delays.
        doomed_delay_to_replacements_map = defaultdict(list)  # existing_delay_node, [sorted_list_of_replacement_delays]
        for replacement_delay in sorted(replacement_delays,
                                        key=lambda k: (k.start_time, k.end_time)):  # Might already be in right order?

            for doomed_delay_node in replacement_delay.replacing_delay_nodes:
                doomed_delay_to_replacements_map[doomed_delay_node].append(replacement_delay)

        # Need a map from each new delay to the corresponding placeholders to know which to replace.
        # (Order doesn't matter.)
        replacement_delay_to_placeholder_map = defaultdict(list)

        for doomed_delay_node, replacements in doomed_delay_to_replacements_map.items():
            placeholder_delay = QuantumCircuit(1)
            for replacement in replacements:
                placeholder_delay.delay(replacement.new_delay_op.duration, 0)

            placeholder_dag = circuit_to_dag(placeholder_delay)
            placeholder_delay_node_ids = [node._node_id for node in placeholder_dag.topological_op_nodes()]

            out_node_map = dag.substitute_node_with_dag(doomed_delay_node, placeholder_dag)
            placeholder_delay_nodes = [out_node_map[node_id] for node_id in placeholder_delay_node_ids]

            for replacement, placeholder_delay_node in zip(replacements, placeholder_delay_nodes):
                replacement_delay_to_placeholder_map[id(replacement.new_delay_op)].append(placeholder_delay_node)

        for replacement in replacement_delays:
            placeholder_nodes = replacement_delay_to_placeholder_map[id(replacement.new_delay_op)]
            dag.replace_block_with_op(
                placeholder_nodes,
                replacement.new_delay_op,
                {node.qargs[0]: idx for idx, node in enumerate(placeholder_nodes)},
                cycle_check=True)

        for delay_node in dag.named_nodes('delay'):
            if delay_node.op.num_qubits > 1 and delay_node.op.duration < self._min_joinable_delay_duration:
                raise RuntimerError('')

        return dag
