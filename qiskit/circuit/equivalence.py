from inspect import signature
from collections import defaultdict, namedtuple

from .gate import Gate
from .parametervector import ParameterVector
from .quantumcircuit import QuantumCircuit
from .quantumregister import QuantumRegister

Entry = namedtuple('Entry', ['search_base',
                             'equivs'])

class EquivalenceLibrary():
    """A library storing equivalence translation rules."""
    
    def __init__(self, *, base=None):
        """Create a new equivalence library.

        Args:
            base - Optional[CircuitEquivalenceLibrary]: Base equivalence library
                which will be referenced if an entry is not found in this library.
        """
        self._base = base

        # keyformat: (gate.label, gate.name, gate.num_qubits, gate_instance)
        # entryformat: (search_base, entries)
        
        self._map = defaultdict(lambda: Entry(True, []))

    def add_entry(self, gate, equivalent_circuit):
        """Add one new equivalence definition to the library.

        Will be added to all existing equalilities (including base).

        Args:
            gate - Gate: \ldots
            equivalent_circuit - QuantumCircuit: \ldots

        """
    
        self._map[(gate.label, gate.name, gate.num_qubits)].equivs.append((gate.params, equivalent_circuit.copy()))

    def set_entry(self, gate, entry):
        """Set 

        Will override existing definitions.

        Args:
            gate - Gate: \ldots
            entry - List[QuantumCircuit]: \ldots

        """
        
        # Should verify gate and entry have same number of free para
        self._map[gate.label, gate.name, gate.num_qubits] = Entry(False, [(gate.params, q.copy()) for q in entry])
        
    def get_entry(self, gate):
        """Get

        Args:
            gate - Gate: \ldots

        Returns: List[qc], if empty list, library contains no known decompositions

        """

        if (gate.label, gate.name, gate.num_qubits) in self._map:
            search_base, equivs = self._map[gate.label, gate.name, gate.num_qubits]

            equivs = [ equiv_circ.bind_parameters({param: val for param, val in zip(params, gate.params)})
                       for params, equiv_circ in equivs]
            
            if search_base and self._base is not None:
                return equivs + self._base.get_entry(gate)
            return equivs

        # Can't just return equivs, need to parameter map (and copy)
        
        if self._base is None:
            return []

        return self._base.get_entry(gate)

    def _build_basis_graph(self):
        # could be deferred, and calculated dynamically, only useful for visualization, high level analysis
        # But could be useful for reachability analysis
        import networkx as nx
        if self._base is None:
            graph = nx.MultiDiGraph()
        else:
            graph = self._base._build_basis_graph()

        # KDK bug here in that if we set at a higher level, we'll still pull lower level defn's into graph
        for (gate_label, gate_name, gate_num_qubits), (_, decomps) in self._map.items():
            gate_basis = frozenset([gate_name])
            for params, decomp in decomps:
                decomp_basis = frozenset(decomp.count_ops())

                graph.add_edge(gate_basis, decomp_basis, decomp=decomp)

        return graph

    def draw_basis_graph(self):
        import io
        import networkx as nx
        import pydot
        from PIL import Image

        dot = nx.drawing.nx_pydot.to_pydot(self._build_basis_graph())
        png = dot.create_png(prog='dot')

        return Image.open(io.BytesIO(png))
