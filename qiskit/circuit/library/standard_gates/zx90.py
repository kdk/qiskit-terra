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


"""
Cross-resonance gate.
"""

import numpy as np
from qiskit.circuit.gate import Gate


class ZX90Gate(Gate):
    r"""Controlled-X gate.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤0      ├
             │  ZX90 │
        q_1: ┤1      ├
             └───────┘

    **Matrix representation:**

    .. math::

        CX\ q_0, q_1 =
            I \otimes |0\rangle\langle0| + X \otimes |1\rangle\langle1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0
            \end{pmatrix}

    **Expanded Circuit:**

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import GMS
        import qiskit.tools.jupyter
        import numpy as np
        circuit = GMS(num_qubits=3, theta=[[0, np.pi/4, np.pi/8],
                                           [0, 0, np.pi/2],
                                           [0, 0, 0]])
        %circuit_library_info circuit.decompose()

    The Mølmer–Sørensen gate is native to ion-trap systems. The global MS
    can be applied to multiple ions to entangle multiple qubits simultaneously [1].

    In the two-qubit case, this is equivalent to an XX(theta) interaction,
    and is thus reduced to the RXXGate. The global MS gate is a sum of XX
    interactions on all pairs [2].

    .. math::

        GMS(\chi_{12}, \chi_{13}, ..., \chi_{n-1 n}) =
        exp(-i \sum_{i=1}^{n} \sum_{j=i+1}^{n} X{\otimes}X \frac{\chi_{ij}}{2}) =

    **References:**

    [1] Chow, Jerry M. and C\'orcoles, A. D. and Gambetta, Jay M. and Rigetti, Chad and Johnson, B. R. and Smolin, John A. and Rozen, J. R. and Keefe, George A. and Rothwell, Mary B. and Ketchen, Mark B. and Steffen, M., Simple All-Microwave Entangling Gate for Fixed-Frequency Superconducting Qubits. Physical Review Letters. 107 (8): 080502.
    `arXiv:1106.0553 <https://arxiv.org/abs/1106.0553>`_

    """

    def __init__(self, label=None):
        """Create new CX gate."""
        super().__init__('zx90', 2, [], label=label)

    def to_matrix(self):
        """Return a numpy.array for the CX gate."""
        sq2 = 1/np.sqrt(2)
        return np.array([[       sq2, -1j * sq2,        0,        0 ],
                         [ -1j * sq2,       sq2,        0,        0 ],
                         [         0,         0,      sq2, 1j * sq2 ],
                         [         0,         0, 1j * sq2,      sq2 ]], dtype=complex)
