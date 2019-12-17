# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Standard gates."""
from .barrier import Barrier
from .ccx import ToffoliGate
from .cswap import FredkinGate
from .cx import CnotGate
from .cy import CyGate
from .cz import CzGate
from .swap import SwapGate
from .h import HGate
from .iden import IdGate
from .s import SGate
from .s import SdgGate
from .t import TGate
from .t import TdgGate
from .u1 import U1Gate
from .u2 import U2Gate
from .u3 import U3Gate
from .x import XGate
from .y import YGate
from .z import ZGate
from .r import RGate
from .rx import RXGate
from .ry import RYGate
from .rz import RZGate
from .cu1 import Cu1Gate
from .ch import CHGate
from .crz import CrzGate
from .cu3 import Cu3Gate
from .rzz import RZZGate
from .rxx import RXXGate
from .ms import MSGate

from qiskit.circuit import CircuitEquivalenceLibrary as _cel
from inspect import signature
from qiskit.circuit import ParameterVector as _pv, QuantumCircuit as _qc, QuantumRegister as _qr
StandardEquivalenceLibrary = _cel()
# Exclude vari-arity  MSGate, Barrier

# gates = [ g for g in standard.__dict__.values() if type(g) is type and issubclass(g, Gate) ] # Should be Instruction? support cbits? Not a problem in stdlib other than barrier, which is already weird
# for g in gates:
#     if g.__name__ == 'MSGate' or g.__name__ == 'Barrier':
#         continue

for g in [# #Barrier, # No Barrier, Instruction and variadic
          # ToffoliGate,
          # FredkinGate,
          # CnotGate,
          # CyGate,
          # CzGate,
          # SwapGate,
          # HGate,
          # IdGate,
          # SGate,
          # SdgGate,
          # TGate,
          # TdgGate,
          # U1Gate,
          # U2Gate,
          # U3Gate,
          XGate,
          YGate,
          RGate,
          # RXGate,
          # RYGate,
          # RZGate,
          # Cu1Gate,
          # CHGate,
          # CrzGate,
          # Cu3Gate,
          # RZZGate,
          # RXXGate,
          #MSGate, # No MSGate, variadic
]:
    n_params = len(signature(g.__init__).parameters) - 1
    th = _pv('th', n_params) # since we're inspecting param name, could re-use already
    gate = g(*th)
    n_qubits = gate.num_qubits
    reg = _qr(n_qubits, 'q')
    circ = _qc(reg)
    #print(gate, reg)
    circ.data.extend(gate.definition)
    StandardEquivalenceLibrary.add_entry(gate, circ)

reg = _qr(2, 'q')
circ = _qc(reg)
circ.h(1)
circ.cx(0,1)
circ.h(1)
StandardEquivalenceLibrary.add_entry(CnotGate(), circ)

from math import pi
reg = _qr(1, 'q')
circ = _qc(reg)
p = _pv('th', 3)
circ.rz(p[0], 0)
circ.rx(pi/2, 0)
circ.rz(p[1]+pi, 0)
circ.rx(pi/2, 0)
circ.rz(p[2]+pi, 0)

StandardEquivalenceLibrary.add_entry(U3Gate(*p), circ)


# # A catch, for gates, params are ordered (and thus so are Parameters)
# # But for circuits they're unordered

# SessionEquivalenceLibrary = CircuitEquivalenceLibrary(base=StandardEquivalenceLibrary)
