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
