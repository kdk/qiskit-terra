from inspect import signature
from collections import defaultdict

from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate
from qiskit.circuit import ParameterVector

class CircuitEquivalenceLibrary():
    def __init__(self, *, base=None):
        self._base = base
        # need to be careful about how we check if a gate is present
        self._map = defaultdict(list)

    # gate instance
    def add_equiv(self, gate, circ):
        self._map[gate.name].append(circ)

    def get_equiv(self, gate):
        if gate.name in self._map:
            return self._map[gate.name]
        if self._base is not None:
            return self._base.get_equiv(gate)
        raise RuntimeError('no known decomp')
        return None # or Raise?
    
        

StandardEquivalenceLibrary = CircuitEquivalenceLibrary()
from qiskit.extensions import standard
gates = [ g for g in standard.__dict__.values() if type(g) is type and issubclass(g, Gate) ] # Should be Instruction? support cbits? Not a problem in stdlib other than barrier, which is already weird
for g in gates:
    if g.__name__ == 'MSGate' or g.__name__ == 'Barrier':
        continue
    n_params = len(signature(g.__init__).parameters) - 1
    th = ParameterVector('th', n_params) # since we're inspecting param name, could re-use already
    gate = g(*th)
    n_qubits = gate.num_qubits
    reg = QuantumRegister(n_qubits, 'q')
    circ = QuantumCircuit(reg)
    print(gate, reg)
    circ.append(gate, [*reg], [])
    StandardEquivalenceLibrary.add_equiv(gate, circ.decompose())


# A catch, for gates, params are ordered (and thus so are Parameters)
# But for circuits they're unordered

SessionEquivalenceLibrary = CircuitEquivalenceLibrary(base=StandardEquivalenceLibrary)
