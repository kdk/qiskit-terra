import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class TestEquivalence(QiskitTestCase):
    """Qiskit Compiler Tests."""

    def test_circuit_equiv(self):

        class MyGate(g):
            
