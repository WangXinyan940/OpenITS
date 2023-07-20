import openmm.unit as unit
import openmm as mm
import openmm.app as app

class OpenITSException(BaseException):
    ...

class EnergyGroupReporter:
    def __init__(self, file, reportInterval, egroups=[0, 1]):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._output_groups = egroups

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, False, True, None)

    def report(self, simulation, state):
        for egrp in self._output_groups:
            s0 = simulation.context.getState(getEnergy=True, groups={egrp})
            e0 = s0.getPotentialEnergy().value_in_unit(unit.kilojoules/unit.mole)
            self._out.write(f"{e0:16.8e} ")
        self._out.write("\n")
        self._out.flush()