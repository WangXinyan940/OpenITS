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
        log_nk: List[float] = None,
        boost_e1_only: bool = False
    ):
        """Initialize the ITS Langevin Integrator"""
        self.temperature_list = temperature_list
        self.friction = friction
        self.dt = dt
        self.integrator = None
        if log_nk is not None:
            self.log_nk = np.array(log_nk)
        else:
            self.log_nk = np.zeros((len(temperature_list),))
        print("Use log_nk", self.log_nk)
        self.boost_e1_only = boost_e1_only
        self.set_integrator(boost_e1_only=boost_e1_only)

    def set_integrator(self, boost_e1_only: bool = False):
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
        self.integrator.addGlobalVariable("vmax", 0.0)
        self.integrator.addGlobalVariable("Aup", 0.0)
        self.integrator.addGlobalVariable("Adown", 0.0)
        self.integrator.addGlobalVariable("A", 0.0)
        self.integrator.addPerDofVariable("x1", 0.0)
        for i in range(len(self.temperature_list)):
            self.integrator.addGlobalVariable(f"hyper_e_{i}", 0.0)
        for i in range(len(self.temperature_list)-1):
            self.integrator.addGlobalVariable(f"max_{i}", 0.0)

        # calc A
        A_up, A_down, vmax = [], [], []
        for i in range(len(self.temperature_list)):
            temp = self.temperature_list[i]
            log_nk = self.log_nk[i]
            beta_k = 1 / (kB * temp)
            beta_k_over_beta_0 = temp / temperature
            
            if boost_e1_only:
                self.integrator.addComputeGlobal(f"hyper_e_{i}", f"-{beta_k:.12e}*energy1+{log_nk:.12e}")
            else:
                self.integrator.addComputeGlobal(f"hyper_e_{i}", f"-{beta_k:.12e}*energy+{log_nk:.12e}")
            A_up.append(
                f"{beta_k_over_beta_0:.12e}*exp(hyper_e_{i}-vmax)"
            )
            A_down.append(f"exp(hyper_e_{i}-vmax)")
            vmax.append(f"hyper_e_{i}")

        for nmax in range(len(vmax)-1):
            if nmax == 0:
                self.integrator.addComputeGlobal(f"max_{nmax}", f"max({vmax[nmax]}, {vmax[nmax+1]})")
            else:
                self.integrator.addComputeGlobal(f"max_{nmax}", f"max(max_{nmax-1}, {vmax[nmax+1]})")

        self.integrator.addComputeGlobal("vmax", f"max_{len(vmax)-2}")
        self.integrator.addComputeGlobal("Aup", "+".join(A_up))
        self.integrator.addComputeGlobal("Adown", "+".join(A_down))
        self.integrator.addComputeGlobal("A", "Aup/Adown")
        self.integrator.addUpdateContextState()
        if boost_e1_only:
            self.integrator.addComputePerDof("v", "v + dt*A*f1/m")
            self.integrator.addComputePerDof("v", "v + dt*f0/m")
        else:
            self.integrator.addComputePerDof("v", "v + dt*A*f/m")
        self.integrator.addConstrainVelocities()
        self.integrator.addComputePerDof("x", "x + 0.5*dt*v")
        self.integrator.addComputePerDof("v", "a*v + b*sqrt(one_beta/m)*gaussian")
        self.integrator.addComputePerDof("x", "x + 0.5*dt*v")
        self.integrator.addComputePerDof("x1", "x")
        self.integrator.addConstrainPositions()
        self.integrator.addComputePerDof("v", "v + (x-x1)/dt")

    def update_nk(self, energies, ratio=0.2):
        # calculate new nk from energies
        new_log_nk = np.zeros(self.log_nk.shape)
        emin = np.min(energies)
        for n in range(new_log_nk.shape[0]):
            beta_k = 1. / (8.314 / 1000.0 * self.temperature_list[n])
            new_log_nk[n] = - np.log(np.exp(- beta_k * (energies - emin)).mean()) + beta_k * emin
        # update nk
        self.log_nk = self.log_nk * (1. - ratio) + new_log_nk * ratio
        self.log_nk = self.log_nk - self.log_nk.mean()
        print("Use log_nk", self.log_nk)
        self.set_integrator(self.boost_e1_only)
        

