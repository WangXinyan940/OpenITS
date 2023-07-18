import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any


class ITSLangevinIntegratorGenerator:
    def __init__(
        self,
        temperature_list: List[float],
        friction: float = 5.0,
        dt: float = 0.002,
        nk: List[float] = None,
    ):
        """Initialize the ITS Langevin Integrator"""
        self.temperature_list = temperature_list
        self.friction = friction
        self.dt = dt
        self.integrator = None
        self.nk = nk if nk is not None else [1.0] * len(temperature_list)
        self.set_integrator()

    def set_integrator(self):
        """Set the ITS Langevin Integrator"""

        temperature = min(self.temperature_list)  # K
        friction = self.friction  # 1/ps
        dt = self.dt  # ps
        kB = 8.314 / 1000.0  # J/mol/K
        kT = kB * temperature

        self.integrator = mm.CustomIntegrator(dt)
        self.integrator.setConstraintTolerance(1e-5)
        self.integrator.addGlobalVariable("a", np.exp(-friction * dt))
        self.integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * friction * dt)))
        self.integrator.addGlobalVariable("one_beta", kT)

        # calc A
        A_up, A_down, vmax = [], [], []
        for i in range(len(self.temperature_list)):
            temp = self.temperature_list[i]
            log_nk = np.log(self.nk[i])
            log_nk_over_log_n0 = log_nk / np.log(self.nk[0])
            beta_k = 1 / (kB * temp)
            beta_k_over_beta_0 = temp / temperature
            A_up.append(
                f"{beta_k_over_beta_0:.8e}*exp(-{beta_k:.8e}*energy+{log_nk:.8e}-vmax)"
            )
            A_down.append(f"exp(-{beta_k:.8e}*energy+{log_nk:.8e}-vmax)")
            vmax.append(f"{beta_k:.8e}*energy+{log_nk:.8e}")

        emax = ",".join(vmax)

        self.integrator.addGlobalVariable("vmax", f"max({emax})")
        self.integrator.addGlobalVariable("Aup", "+".join(A_up))
        self.integrator.addGlobalVariable("Adown", "+".join(A_down))
        self.integrator.addGlobalVariable("A", "Aup/Adown")
        self.integrator.addUpdateContextState()
        self.integrator.addComputePerDof("v", "v + dt*f/m")
        self.integrator.addConstrainVelocities()
        self.integrator.addComputePerDof("x", "x + 0.5*dt*v")
        self.integrator.addComputePerDof("v", "a*v + b*sqrt(one_beta/m)*gaussian")
        self.integrator.addComputePerDof("x", "x + 0.5*dt*v")
        self.integrator.addComputePerDof("x1", "x")
        self.integrator.addConstrainPositions()
        self.integrator.addComputePerDof("v", "v + (x-x1)/dt")
