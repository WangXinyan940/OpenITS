import openmm.unit as unit
import openmm as mm
import openmm.app as app
import numpy as np

class OpenITSException(BaseException):
    ...

class EnergyGroupReporter:
    def __init__(self, file, reportInterval, egroups=[0, 1], log_nk = None, log_nk2 = None, temperature_list = None):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._output_groups = egroups

        # report temperature list
        if temperature_list is None:
            self._out.write(f"# Temperature list: None\n")
        else:
            self._out.write(f"# Temperature list: ")
            for t_val in temperature_list:
                self._out.write(f"{t_val:.8f} ")
            self._out.write("\n")
        
        # report log_nk
        if log_nk is None:
            self._out.write(f"# log_nk: None\n")
        else:
            self._out.write(f"# log_nk: ")
            for lnk in log_nk:
                self._out.write(f"{lnk:.10f} ")
            self._out.write("\n")

        # report log_nk2
        if log_nk2 is None:
            self._out.write(f"# log_nk2: None\n")
        else:
            self._out.write(f"# log_nk2: ")
            for lnk in log_nk2:
                self._out.write(f"{lnk:.10f} ")
            self._out.write("\n")

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


def load_its_log(filename):
    with open(filename, "r") as f:
        head, data = [], []
        for line in f:
            if line[0] == "#":
                head.append(line)
            else:
                data.append(line)
        try:
            temperature_line = [i for i in head if "Temperature" in i][0]
            temperature_list = np.array([float(i) for i in temperature_line.strip().split(":")[1].strip().split()])
        except IndexError as e:
            raise OpenITSException(f"Temperature list not found in log file '{filename}'.")

        try:
            lognk = [i for i in head if "log_nk" in i][0]
            log_nk_list = np.array([float(i) for i in lognk.strip().split(":")[1].strip().split()])
        except IndexError as e:
            raise OpenITSException(f"Log_nk not found in log file '{filename}'.")
        
        try:
            lognk2 = [i for i in head if "log_nk2" in i][0]
            log_nk2_list = np.array([float(i) for i in lognk2.strip().split(":")[1].strip().split()])
        except IndexError as e:
            log_nk2_list = np.zeros(log_nk_list.shape)
        
        e_mat = [[float(j) for j in i.strip().split()] for i in data]
        e_mat = np.array(e_mat)
    return e_mat, temperature_list, log_nk_list, log_nk2_list
