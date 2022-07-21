from qiskit.test import QiskitTestCase
from qiskit.transpiler import timing_constraints

from qiskit.transpiler.passes.utils.combine_adjacent_delays import CombineAdjacentDelays

from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.scheduling import TimeUnitConversion, ConstrainedReschedule, PadDelay
from qiskit.transpiler import CouplingMap, InstructionDurations, PassManager, instruction_durations
from qiskit.transpiler.passes import ALAPScheduleAnalysis

from qiskit.test.mock import FakeMumbai
from qiskit.visualization import timeline_drawer
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.compiler import transpile

def schedule_alap(c, backend):
    instruction_durations = InstructionDurations.from_backend(backend)
    timing_constraints = TimingConstraints()#**backend.configuration().timing_constraints)

    pm = PassManager(
        [
            TimeUnitConversion(instruction_durations),
            ALAPScheduleAnalysis(instruction_durations),
            ConstrainedReschedule(acquire_alignment=timing_constraints.acquire_alignment,
                                  pulse_alignment=timing_constraints.pulse_alignment),
            PadDelay(),
            CombineAdjacentDelays(CouplingMap(backend.configuration().coupling_map))
        ]
    )
    return pm.run(c)

class TestCombineAdjacentDelays(QiskitTestCase):
    def test_subcomponent_delays(self):
        backend=FakeMumbai()
        cmap = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]])
        instruction_durations = InstructionDurations.from_backend(backend)
        timing_constraints = TimingConstraints()
        test_qc = QuantumCircuit(4)
        test_qc.x(range(4))
        test_qc.delay(1000, range(4))
        test_qc.cx(1, 2)
        test_qc.delay(1000, [0, 3])
        pm = PassManager(
            [
                TimeUnitConversion(instruction_durations),
                ALAPScheduleAnalysis(instruction_durations),
                ConstrainedReschedule(acquire_alignment=timing_constraints.acquire_alignment,
                                    pulse_alignment=timing_constraints.pulse_alignment),
                PadDelay(),
                CombineAdjacentDelays(cmap)
            ]
        )
        return pm.run(test_qc)

    def test_4q_vchain(self):
        N=4
        backend=FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]

        test_qc = QuantumCircuit(backend.configuration().num_qubits)
        for i in range(N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,-1,-1):
            test_qc.cx(line[i], line[i+1])

        g_test_qc = schedule_alap(test_qc, backend)

    def test_6q_vchain(self):
        N=6
        backend=FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]

        test_qc = QuantumCircuit(backend.configuration().num_qubits)
        for i in range(N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,-1,-1):
            test_qc.cx(line[i], line[i+1])

        g_test_qc = schedule_alap(test_qc, backend)

    def test_4q_wchain(self):
        N=4
        backend=FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]

        test_qc = QuantumCircuit(backend.configuration().num_qubits)
        test_qc.cx(0,1)
        for i in range(1,N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,0,-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(2,N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,0,-1):
            test_qc.cx(line[i], line[i+1])

        test_qc.cx(0,1)
        g_test_qc = schedule_alap(test_qc, backend)

    def test_6q_wchain(self):
        N=6
        backend=FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]

        test_qc = QuantumCircuit(backend.configuration().num_qubits)
        test_qc.cx(0,1)
        for i in range(1,N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,0,-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(2,N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,0,-1):
            test_qc.cx(line[i], line[i+1])
        test_qc.cx(0,1)

        qvx = transpile(test_qc,
                        coupling_map=CouplingMap(backend.configuration().coupling_map),
                        basis_gates=['rz', 'sx', 'cx'],
                        #basis_gates=['unitary', 'swap'],
                        seed_transpiler=4,
                        optimization_level=0)

        #qvx._data = qvx._data[:3]

        qvx2 = transpile(qvx,
                         initial_layout=line[:N],
                         basis_gates=['rz', 'sx', 'cx'],
                         instruction_durations = InstructionDurations.from_backend(backend),
                         scheduling_method='alap')

        g_test_qc = schedule_alap(qvx2, backend)

    def test_qv16(self):
        # import logging
        # logging.basicConfig(level=logging.INFO)

        N = 4
        backend = FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]
        basis_gates = ['cx', 'sx', 'rz']

        qv = QuantumVolume(N, seed=0).decompose()
        cmap = CouplingMap.from_line(N)
        instruction_durations = InstructionDurations.from_backend(backend)

        # qv = QuantumCircuit(2)
        # qv.cx(0, 1)
        # qv.rz(1.234, 0)
        # qv.sx(1)
        # qv.rz(0.123, 1)
        # qv.cx(0, 1)

        qvx = transpile(qv,
                        coupling_map=cmap,
                        #basis_gates=['rz', 'sx', 'cx'],
                        basis_gates=['unitary', 'swap'],
                        routing_method='sabre',
                        layout_method='sabre',
                        seed_transpiler=4,
                        optimization_level=0)

        #qvx._data = qvx._data[:3]

        qvx2 = transpile(qvx,
                         initial_layout=line[:N],
                         basis_gates=basis_gates,
                         instruction_durations=instruction_durations,
                         scheduling_method='alap')

        qvx2.draw('mpl', idle_wires=False)


        g_test_qc = schedule_alap(qvx2, backend)

    def test_qv16(self):
        # import logging
        # logging.basicConfig(level=logging.INFO)

        N = 8
        backend = FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]
        basis_gates = ['cx', 'sx', 'rz']

        qv = QuantumVolume(N, seed=0).decompose()
        cmap = CouplingMap.from_line(N)
        instruction_durations = InstructionDurations.from_backend(backend)


        qvx = transpile(qv,
                        coupling_map=cmap,
                        #basis_gates=['rz', 'sx', 'cx'],
                        basis_gates=['unitary', 'swap'],
                        routing_method='sabre',
                        layout_method='sabre',
                        seed_transpiler=4,
                        optimization_level=0)

        qvx2 = transpile(qvx,
                         initial_layout=line[:N],
                         basis_gates=basis_gates,
                         instruction_durations=instruction_durations,
                         scheduling_method='alap')

        #qvx2.draw('mpl', idle_wires=False)
        # timeline_drawer(qvx2, show_idle=False, show_delays=True)

        g_test_qc = schedule_alap(qvx2, backend)

        # g_test_qc.draw('mpl', fold=-1, idle_wires=False)
        # timeline_drawer(g_test_qc, show_idle=False, show_delays=True)

        # import matplotlib.pyplot as plt
        # plt.show()



