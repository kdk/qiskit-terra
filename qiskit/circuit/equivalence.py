# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Gate equivalence library."""

from collections import namedtuple

from .exceptions import CircuitError
from .parameterexpression import ParameterExpression

Key = namedtuple('Key', ['name',
                         'num_qubits'])

Entry = namedtuple('Entry', ['search_base',
                             'equivalences'])

Equivalence = namedtuple('Equivalence', ['params',  # Ordered to match Gate.params
                                         'circuit'])


class EquivalenceLibrary():
    """A library providing a one-way mapping of Gates to their equivalent
    implementations as QuantumCircuits."""

    def __init__(self, *, base=None):
        """Create a new equivalence library.

        Args:
            base (Optional[EquivalenceLibrary]):  Base equivalence library to
                will be referenced if an entry is not found in this library.
        """
        self._base = base

        self._map = {}

    def add_equivalence(self, gate, equivalent_circuit, validate=True):
        """Add a new equivalence to the library. Future queries for the Gate
        will include the given circuit, in addition to all existing equivalences
        (including those from base).

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            equivalent_circuit (QuantumCircuit): A circuit equivalently
                implementing the given Gate.
            validate (Bool): Optional. If True (default), attempts to check that
                added entires are equivalent to those already present for this
                gate. Raises a CircuitError if not. This argument does not
                rigorously check if parameterized definitions are equivalent, and
                so has a non-zero false negative rate in these cases.

        Raises:
            CircuitError: If added circuit and gate are not of the same number of
               qubits and parameters. Or, if validate=True, if added definition
               is determined non-equivalent to an existing entry.
        """

        _raise_if_shape_mismatch(gate, equivalent_circuit)
        _raise_if_param_mismatch(gate.params, equivalent_circuit.parameters)

        key = Key(name=gate.name,
                  num_qubits=gate.num_qubits)

        equiv = Equivalence(params=gate.params.copy(),
                            circuit=equivalent_circuit.copy())

        if validate:
            self._validate_equiv_against_existing(key, equiv)

        if key not in self._map:
            self._map[key] = Entry(search_base=True, equivalences=[])

        self._map[key].equivalences.append(equiv)

    def has_entry(self, gate):
        """Check if a library contains any decompositions for gate.

        Args:
            gate (Gate): A Gate instance.

        Returns:
            Bool: True if gate has a known decomposition in the library.
                False otherwise.
        """

        key = Key(name=gate.name,
                  num_qubits=gate.num_qubits)

        return (key in self._map or
                (self._base.has_entry(gate) if self._base is not None else False))

    def _validate_equiv_against_existing(self, key, equiv, check_symbolic=False):
        existing_equivs = self._get_equivalences(key)

        if any(circ.parameters for circ in equiv + existing_equvis) and check_symbolic:
            # Assume every entry in the library will have the same number of params as us.
            # KDK Raise if attempting to register a partial parameterization or handle here.
            pes = [(idx, p)
                   for idx, p in enumerate(existing_equiv.params)
                   if isinstance(p, ParameterExpression)]
            # KDK above if should always be True (see previous KDK)

            test_p_vals = [0.1 * n for n in range(len(pes))]

            test_op = Operator(_rebind_equiv(equiv, test_p_vals))
            test_exist_op = Operator(_rebind_equiv(existing_equiv, test_p_vals))

            return
        else:
            for existing_equiv in existing_equivs:
                from qiskit.quantum_info import Operator

                test_op = Operator(equiv)
                test_exist_op = Operator(existing_equiv)

                if test_op != test_exist_op:
                    raise CircuitError(
                        'Attemping to add entry not equal to existing entries. '
                        'Key: {}. New equiv: {} Existing equiv: {}. Param values: {}.'.format(
                            key,
                            equiv,
                            existing_equiv,
                            test_p_vals))

    def set_entry(self, gate, entry, validate=True):
        """Set the equivalence record for a Gate. Future queries for the Gate
        will return only the circuits provided.

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            entry (List['QuantumCircuit']) : A list of QuantumCircuits, each
                equivalently implementing the given Gate.
            validate (Bool): Optional. If True (default), attempts to check that
                added entires are equivalent to those already present for this
                gate. Raises a CircuitError if not. This option has no effect if
                the added entry is parameterized.

        Raises:
            CircuitError: If added circuit and gate are not of the same number of
               qubits and parameters. Or, if validate=True, if added definition
               is determined non-equivalent to an existing entry.
        """

        for equiv in entry:
            _raise_if_shape_mismatch(gate, equiv)
            _raise_if_param_mismatch(gate.params, equiv.parameters)

        key = Key(name=gate.name,
                  num_qubits=gate.num_qubits)

        equivs = [Equivalence(params=gate.params.copy(),
                              circuit=equiv.copy())
                  for equiv in entry]

        if validate:
            for equiv in equivs:
                self._validate_equiv_against_existing(key, equiv)

        self._map[key] = Entry(search_base=False,
                               equivalences=equivs)

    def get_entry(self, gate):
        """Gets the set of QuantumCircuits circuits from the library which
        equivalently implement the given Gate.

        Parameterized circuits will have their parameters replaced with the
        corresponding entries from Gate.params.

        Args:
            gate (Gate) - Gate: A Gate instance.

        Returns:
            List[QuantumCircuit]: A list of equivalent QuantumCircuits. If empty,
                library contains no known decompositions of Gate.

                Returned circuits will be ordered according to their insertion in
                the library, from earliest to latest, from top to base. The
                ordering of the StandardEquivalenceLibrary will not generally be
                consistent across Qiskit versions.
        """

        key = Key(name=gate.name,
                  num_qubits=gate.num_qubits)

        query_params = gate.params

        if key in self._map:
            entry = self._map[key]
            search_base, equivs = entry

            rtn = [_rebind_equiv(equiv, query_params) for equiv in equivs]

            if search_base and self._base is not None:
                return rtn + self._base.get_entry(gate)
            return rtn

        if self._base is None:
            return []

        return self._base.get_entry(gate)


def _raise_if_param_mismatch(gate_params, circuit_parameters):
    gate_parameters = [p for p in gate_params
                       if isinstance(p, ParameterExpression)]

    if set(gate_parameters) != circuit_parameters:
        raise CircuitError('Cannot add equivalence between circuit and gate '
                           'of different parameters. Gate params: {}. '
                           'Circuit params: {}.'.format(
                               gate_parameters,
                               circuit_parameters))


def _raise_if_shape_mismatch(gate, circuit):
    if (gate.num_qubits != circuit.num_qubits
            or gate.num_clbits != circuit.num_clbits):
        raise CircuitError('Cannot add equivalence between circuit and gate '
                           'of different shapes. Gate: {} qubits and {} clbits. '
                           'Circuit: {} qubits and {} clbits.'.format(
                               gate.num_qubits, gate.num_clbits,
                               circuit.num_qubits, circuit.num_clbits))


def _rebind_equiv(equiv, query_params):
    equiv_params, equiv_circuit = equiv

    param_map = dict(zip(equiv_params, query_params))
    equiv = equiv_circuit.assign_parameters(param_map, inplace=False)

    return equiv
