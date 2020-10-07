# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unroll a circuit to a given basis."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.exceptions import QiskitError
from qiskit.circuit import ControlledGate
from qiskit.converters.circuit_to_dag import circuit_to_dag

from qiskit.qasm.qasm import Qasm
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler.passes.synthesis.unitary_synthesis import _choose_euler_basis
    

class TwoQToNativeEntangler(TransformationPass):
    """Unroll a circuit to a given basis.

    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    # Already in device basis, already in one or twoq gates
    
    def __init__(self, basis_gates, gate_configurations):
        """Unroller initializer.

        Args:
            basis (list[str] or None): Target basis names to unroll to, e.g. `['u3', 'cx']` . If
                None, does not unroll any gate.
        """
        super().__init__()
        self.basis_gates = basis_gates
        self.gate_configurations = gate_configurations

        self.euler_basis = _choose_euler_basis(basis_gates)

        self.native_entanglers = {
            tuple(gate.coupling_map[0]): QuantumCircuit.from_qasm_str(
                    QuantumCircuit.header
                    + QuantumCircuit.extension_lib
                    + gate.qasm_def
                    + 'qreg q[2]; n2q q[0],q[1];')
            for gate in gate_configurations
            if gate.name == 'n2q'
        }

        for qargs in self.native_entanglers:
            self.native_entanglers[qargs].name = 'n2q'

        self.native_decomposers = {
            qargs: TwoQubitBasisDecomposer(
                entangler, 
                euler_basis=self.euler_basis)
            for qargs, entangler in self.native_entanglers.items()
        }

        self.decomposition_cache = {}

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

        if not self.native_decomposers:
            return dag

        for node in dag.two_qubit_ops():
            q_idxes = tuple(q.index for q in node.qargs)
            key = (node.name, q_idxes)
            if key not in self.decomposition_cache:
                self.decomposition_cache[key] = circuit_to_dag(
                    self.native_decomposers[q_idxes](Operator(node.op).data))

            dag.substitute_node_with_dag(
                node,
                self.decomposition_cache[key],
                wires=None)

        return dag
