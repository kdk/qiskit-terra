from ddt import ddt, data

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

import logging
logging.basicConfig(level='DEBUG')

def schedule_alap(c, backend, m_j_d_d=None):
    instruction_durations = InstructionDurations.from_backend(backend)
    timing_constraints = TimingConstraints()#**backend.configuration().timing_constraints)

    pm = PassManager(
        [
            TimeUnitConversion(instruction_durations),
            ALAPScheduleAnalysis(instruction_durations),
            ConstrainedReschedule(acquire_alignment=timing_constraints.acquire_alignment,
                                  pulse_alignment=timing_constraints.pulse_alignment),
            PadDelay(),
            CombineAdjacentDelays(CouplingMap(backend.configuration().coupling_map), m_j_d_d)
        ]
    )
    return pm.run(c)

@ddt
class TestCombineAdjacentDelays(QiskitTestCase):
    def _run_and_plot(self, test_qc, backend, m_j_d_d=None):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2,2)

        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,:])

        test_qc.draw('mpl', idle_wires=False, ax=ax1)
        timeline_drawer(test_qc, show_idle=False, show_delays=True)
        g_test_qc = schedule_alap(test_qc, backend, m_j_d_d)

        g_test_qc.draw('mpl', fold=-1, idle_wires=False, ax=ax2)
        timeline_drawer(g_test_qc, show_idle=False, show_delays=True, axis=ax3)

        fig.suptitle(f'MIN_JOINABLE_DELAY_DURATION={m_j_d_d}', fontsize=16)
        
        plt.show()


        #print(test_qc.draw(idle_wires=False))
        #g_test_qc = schedule_alap(test_qc, backend)

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

    @data(100, 1600, 2500)
    def test_4q_vchain(self, m_j_d_d):
        N=4
        backend=FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]

        test_qc = QuantumCircuit(backend.configuration().num_qubits)
        for i in range(N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,-1,-1):
            test_qc.cx(line[i], line[i+1])

        self._run_and_plot(test_qc, backend, m_j_d_d)

    @data(100, 1600, 2500)
    def test_6q_vchain(self, m_j_d_d):
        N=6
        backend=FakeMumbai()
        line = [0, 1, 4, 7, 10, 12, 15, 18, 21]

        test_qc = QuantumCircuit(backend.configuration().num_qubits)
        for i in range(N-1):
            test_qc.cx(line[i], line[i+1])
        for i in range(N-3,-1,-1):
            test_qc.cx(line[i], line[i+1])

        self._run_and_plot(test_qc, backend, m_j_d_d)

    @data(100, 1600, 2500)
    def test_4q_wchain(self, m_j_d_d):
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

        self._run_and_plot(test_qc, backend, m_j_d_d)

    @data(100, 1600, 2500)
    def test_6q_wchain(self, m_j_d_d):
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

        self._run_and_plot(test_qc, backend, m_j_d_d)

    @data(100, 1600, 2400)
    def test_qv16(self, m_j_d_d):

        N = 4
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

        self._run_and_plot(qvx2, backend, m_j_d_d)

    @data(100, 1600, 2400)
    def test_qv32(self, m_j_d_d):
        N = 5
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

        self._run_and_plot(qvx2, backend, m_j_d_d)

    @data(100, 1600, 2400)
    def test_qv256(self, m_j_d_d):
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

        self._run_and_plot(qvx2, backend, m_j_d_d)
