from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError

import networkx as nx

from qiskit.circuit import SessionEquivalenceLibrary

class BasisTranslator(TransformationPass):

    def __init__(self, target_basis, equivalence_library=None, source_basis=None):
        """Unroller initializer.

        Args:
            basis (list[str]): Target basis names to unroll to, e.g. `['u3', 'cx']` .
        """
        super().__init__()
        basic_insts = ['measure', 'reset', 'barrier', 'snapshot']
        self.target_basis = set(target_basis).union(basic_insts)

        if equivalence_library is not None:
            self._eq_lib = equivalence_library
        else:
            self._eq_lib = SessionEquivalenceLibrary

        if source_basis is not None:
            # Can front load building of 
            pass

    def run(self, dag):
        """Run the Unroller pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """

        dag_ops = dag.count_ops()
        if source_basis is not None:
            # Check that input dag matches source_basis
            pass
        
        # Walk through the DAG and expand each non-basis node
        # https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/shortest_paths/astar.html
        basis_map = simple_astar()

        if basis_map is None:
            raise QiskitError(
                'Unable to map source basis {} to target basis {}.'.format(
                source_basis, target_basis))
        
        for node in dag.op_nodes():
            target_node = basis_map[node.name]
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
    return 0
    
def simple_astar(edge_graph, src_basis, tgt_basis, heuristic):
    #src_basis and tgt_basis should both be frozen sets
    
    from heapq import heappush, heappop

    open_set = set()
    open_set.add(src_basis)
    closed_set = set()
    open_heap = [] # Priority queue for inspection order of open_set
    cntr = 0
    heappush(open_heap, (0, cntr, src_basis))

    gScore = defaultdict(lambda: np.inf) # hash-to-val map, lowest found cost from start to key
    gScore[src_basis] = 0

    fScore = defaultdict(lambda: np.inf) # hash-to-val map, gScore[key] + heuristic from key to goal
    fScore[src_basis] = basis_dist(src_basis, tgt_basis)

    while open_set:
        (_, __, current_basis) = heappop(open_heap)

        if current_hash not in open_set:
            # When we close a node, we don't remove it from the heap
            # so skip here.
            continue

        current = dag_hashes[current_hash]

        #print(_)
        #from qiskit.converters import dag_to_circuit
        #print(dag_to_circuit(current))

        if circ_ready_for_ab(current):
            end_time = time.time()
            logger.info('Exiting astar for DAG (time: {:.2f}s, nodes examined: {}, nodes_found: {})\n{}'.format(
                        end_time-start_time,
                        len(closed_set), len(closed_set) + len(open_set),
                dag_to_circuit(current)))

            return current

        open_set.remove(current_hash)
        closed_set.add(current_hash)

        for t_width in identity_set:
            for template in identity_set[t_width]:
                # Should be possible (and faster) to replace this with commutations
                neighbors = identityset.all_template_applications(current, template)

                for neighbor in neighbors:
                    # KDK Would be ideal to collect 2q blocks here, maybe followed by
                    # reduce until irreducible, but you don't want to undo an unrolling
                    # step that might be necessary to find some identity.

                    if _hash_dag(neighbor) in closed_set:
                        continue

                    tentative_gScore = gScore[_hash_dag(current)] + 1

                    if _hash_dag(neighbor) not in open_set:
                        open_set.add(_hash_dag(neighbor))
                        dag_hashes[_hash_dag(neighbor)] = neighbor
                    elif tentative_gScore >= gScore[_hash_dag(neighbor)]:
                        continue

                    cameFrom[_hash_dag(neighbor)] = current
                    gScore[_hash_dag(neighbor)] = tentative_gScore
                    fScore[_hash_dag(neighbor)] = 0.1*tentative_gScore + dist_to_ab_heuristic(neighbor)
                    cntr += 1
                    heappush(open_heap, (fScore[_hash_dag(neighbor)], cntr, _hash_dag(neighbor)))
