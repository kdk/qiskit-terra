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
"""

import numpy as np
from qiskit.circuit.gate import Gate


class ZZGate(Gate):
    r"""
    """

    def __init__(self, label=None):
        """Create new CX gate."""
        super().__init__('zz', 2, [], label=label)

    def to_matrix(self):
        """Return a numpy.array for the CX gate."""
        return np.array([[ 0,   0,   0, 1 ],
                         [ 0,   0, -1j, 0 ],
                         [ 0, -1j,   0, 0 ],
                         [ 1,   0,   0, 0 ]], dtype=complex)
