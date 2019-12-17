from inspect import signature
from collections import defaultdict, namedtuple

from .gate import Gate
from .parametervector import ParameterVector
from .quantumcircuit import QuantumCircuit
from .quantumregister import QuantumRegister

Entry = namedtuple('Entry', ['search_base',
                             'equivs'])

class CircuitEquivalenceLibrary():
    """A library storing equivalence translation rules."""
    
    def __init__(self, *, base=None):
        """Create a new equivalence library.

        Args:
            base - Optional[CircuitEquivalenceLibrary]: Base equivalence library
                which will be referenced if an entry is not found in this library.
        """
        self._base = base

        # keyformat: (gate.label, gate.name)
        # entryformat: (search_base, entries)
        
        self._map = defaultdict(lambda: Entry(True, []))

    def add_entry(self, gate, equivalent_circuit):
        """Add one new equivalence definition to the library.

        Will be added to all existing equalilities (including base)

        Args:
            gate - Gate: \ldots
            equivalent_circuit - QuantumCircuit: \ldots

        """
    
        self._map[(gate.label, gate.name)].equivs.append(equivalent_circuit.copy())

    def set_entry(self, gate, entry):
        """Set 

        Will override existing definitions.

        Args:
            gate - Gate: \ldots
            entry - List[QuantumCircuit]: \ldots

        """

        self._map[gate.label, gate.name] = Entry(False, [q.copy() for q in entry])
        
    def get_entry(self, gate):
        """Get

        Args:
            gate - Gate: \ldots

        Returns: List[qc], if empty list, does not mean gate cannot be decomposed,
        only that library contains no known decompositions

        """

        if (gate.label, gate.name) in self._map:
            search_base, equivs = self._map[gate.label, gate.name]
            if search_base:
                return equivs + self._base.get_entry(gate)
            return equivs

        return []
