import time
import logging

from heapq import heappush, heappop
from itertools import count as iter_count
from collections import defaultdict

import numpy as np
import networkx as nx

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)

basic_insts = ['measure', 'reset', 'barrier', 'snapshot']


class BasisTranslator(TransformationPass):
    def __init__(self, equivalence_library, target_basis):
        """Unroller initializer.

        Args:
            equivalence_library:
            target_basis (list[str]):
        """
        super().__init__()

        self._equiv_lib = equivalence_library        
        self._target_basis = target_basis

    def run(self, dag):
        """Run the Unroller pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError:

        Returns:
            DAGCircuit: output unrolled dag
        """

        target_basis = set(self._target_basis).union(basic_insts)

        dag_ops = dag.count_ops()
        source_basis = set(dag_ops.keys())

        logger.info('Begin BasisTranslator from source basis {} to target '
                    'basis {}.'.format(source_basis, target_basis))

        basis_transform = simple_astar(self._equiv_lib, source_basis,
                                       target_basis, basis_dist)
        
        if basis_transform is None:
            raise QiskitError(
                'Unable to map source basis {} to target basis {} over library {}.'.format(
                    source_basis, target_basis, self._equiv_lib))        

        # basis maps is an ordered list of transformations that move from src to target basis
        compose_start_time = time.time()
        mapped_ops = {}
        for source_op in dag_ops:
            example_source_op = dag.named_nodes(source_op)[0].op
            empty_dag = DAGCircuit()
            qr = QuantumRegister(example_source_op.num_qubits)
            empty_dag.add_qreg(qr)
            empty_dag.apply_operation_back(example_source_op, qr[:], [])
            
            for src_gate_name, dest_circ in basis_transform:
                from qiskit.converters import circuit_to_dag
                dest_dag = circuit_to_dag(dest_circ)
                doomed_nodes = empty_dag.named_nodes(src_gate_name)
                for node in doomed_nodes:
                    empty_dag.substitute_node_with_dag(node, dest_dag) # wires=None ?

            mapped_ops[source_op] = empty_dag

        compose_end_time = time.time()
        logger.info('Basis translation path composed in {%.3f}s.'.format(
            compose_end_time - compose_start_time))
            
        for s, v in mapped_ops.items():
            print(s)
            from qiskit.converters import dag_to_circuit
            print(dag_to_circuit(v))
        return dag
        for node in dag.op_nodes():
            target_node = mapped_ops[node.name]
            if target_node.name == node.name:
                continue
            
            if len(target_node) == 1 and len(node.qargs) == len(rule[0][1]):
                dag.substitute_node(node, rule[0][0], inplace=True)
            else:
                dag.substitute_node_with_dag(node, target_node)

        return dag


def basis_dist(basis, target):
    # Maybe pre-compute for every gate src basis, tree of minimum dist from any gate to any other gate in basis
    # Tree might be necessariy anyway, to identify gates that will never be able to make it to the target set
    # Question of how to procede here, best effort unrolling? Or raise?

    # Note: can search for minimum number of substitutions, or if we start with total gate count, can minimize for final gate count, final fidelity score
    # each a different function

    return 0
    
def simple_astar(edge_graph, src_basis, tgt_basis, heuristic):
    # https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/shortest_paths/astar.html
    search_start_time = time.time()

    src_basis = frozenset(src_basis)
    tgt_basis = frozenset(tgt_basis)

    open_set = set()
    open_set.add(src_basis)
    closed_set = set()
    open_heap = [] # Priority queue for inspection order of open_set
    count = iter_count()

    cameFrom = {}
    heappush(open_heap, (0, next(count), src_basis))

    gScore = defaultdict(lambda: np.inf) # hash-to-val map, lowest found cost from start to key
    gScore[src_basis] = 0

    fScore = defaultdict(lambda: np.inf) # hash-to-val map, gScore[key] + heuristic from key to goal
    fScore[src_basis] = heuristic(src_basis, tgt_basis)

    logger.debug('Begining basis search from {} to {}.'.format(src_basis,
                                                               tgt_basis))

    while open_set:
        (_, __, current_basis) = heappop(open_heap)

        if current_basis in closed_set:
            # When we close a node, we don't remove it from the heap,
            # so skip here.
            continue

        if current_basis.issubset(tgt_basis):
            search_end_time = time.time()
            logger.info('Basis translation path found in {%.3f}s.'.format(
                search_end_time - search_start_time))

            rtn = []
            last_basis = current_basis
            while last_basis != src_basis:
                prev_basis, xform_gate, xform = cameFrom[last_basis]
            
                rtn.append((xform_gate, xform))
                last_basis = prev_basis
            rtn.reverse()

            logger.debug('Transformation path:')
            for xform_gate, xform in rtn:
                logger.debug(xform_gate, '=>', xform)
            return rtn

        logger.debug('Examining basis {}.'.format(current_basis))
        open_set.remove(current_basis)
        closed_set.add(current_basis)

        for gate_name in current_basis:
            from qiskit.circuit import Gate
            # basis_gates has only string names but some gates are vari-width. :(
            xforms = [ form for n in range(10)
                       for form in edge_graph.get_entry(Gate(gate_name, n, []))]

            basis_remain = frozenset(current_basis - {gate_name})
            neighbors = [ (basis_remain | xform.count_ops().keys(), xform)
                          for xform in xforms ]

            for neighbor, xform in neighbors:
                if neighbor in closed_set:
                    continue

                tentative_gScore = gScore[current_basis] + 1

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_gScore >= gScore[neighbor]:
                    continue

                cameFrom[neighbor] = (current_basis, gate_name, xform)
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + heuristic(neighbor, tgt_basis)
                heappush(open_heap, (fScore[neighbor], -1 * next(count), neighbor))

    search_end_time = time.time()
    logger.info('Basis translation found no solution in {%.3f}s.'.format(
        search_end_time - search_start_time))

    return None
