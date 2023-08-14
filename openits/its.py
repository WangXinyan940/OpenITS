import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
from enum import Enum

class EnhancedGroup(Enum):
    ALL = 0
    E1 = 1
    E1_AND_E2 = 2


class ITSLangevinIntegratorGenerator:
    def __init__(
        self,
        temperature_list: List[float],
        friction: float = 5.0,
        dt: float = 0.002,
        log_nk: List[float] = None,
        log_nk2: List[float] = None,
        its_log: str = None,
        boost_group: int = EnhancedGroup.ALL,
        verbose: bool = False
    ):
        """Initialize the ITS Langevin Integrator"""
        
        self.friction = friction
        self.dt = dt
        self.integrator = None
        self.verbose = verbose

        if its_log is not None:
            temperature_list, log_nk, log_nk2 = self.load_log(its_log)
            self.temperature_list = temperature_list
            self.log_nk = np.array(log_nk)
            self.log_nk2 = np.array(log_nk2)
            print("Use log_nk", self.log_nk)
            print("Use log_nk2", self.log_nk2)
        else:
            self.temperature_list = temperature_list
            if log_nk is not None:
                self.log_nk = np.array(log_nk)
            else:
                self.log_nk = np.zeros((len(temperature_list),))
            if log_nk2 is not None:
                self.log_nk2 = np.array(log_nk2)
            else:
                self.log_nk2 = np.zeros((len(temperature_list),))
            print("Use log_nk", self.log_nk)
            if log_nk2 is not None:
                print("Use log_nk2", self.log_nk2)

        self.boost_group = boost_group
        self.set_integrator(boost_group=boost_group)

    def integ_addGlobalVariable(self, name, value):
        if self.verbose:
            print(f">>> global_init: {name} = {value}")
        self.integrator.addGlobalVariable(name, value)

    def integ_addPerDofVariable(self, name, value):
        if self.verbose:
            print(f">>> per_dof_init: {name} = {value}")
        self.integrator.addPerDofVariable(name, value)

    def integ_addComputeGlobal(self, name, value):
        if self.verbose:
            print(f">>> global_set: {name} = {value}")
        self.integrator.addComputeGlobal(name, value)
    
    def integ_addComputePerDof(self, name, value):
        if self.verbose:
            print(f">>> per_dof_set: {name} = {value}")
        self.integrator.addComputePerDof(name, value)

    def set_integrator(self, boost_group: int = None):
        """Set the ITS Langevin Integrator"""
        if boost_group is None:
            boost_group = self.boost_group

        temperature = min(self.temperature_list)  # K
        friction = self.friction  # 1/ps
        dt = self.dt  # ps
        kB = 8.314 / 1000.0  # J/mol/K
        kT = kB * temperature

        self.integrator = mm.CustomIntegrator(self.dt)
        self.integrator.setConstraintTolerance(1e-5)
        self.integ_addGlobalVariable("a", np.exp(-friction * dt))
        self.integ_addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * friction * dt)))
        self.integ_addGlobalVariable("one_beta", kT)
        self.integ_addPerDofVariable("x1", 0.0)

        if boost_group == EnhancedGroup.E1 or boost_group == EnhancedGroup.ALL:
            self.integ_addGlobalVariable("vmax", 0.0)
            self.integ_addGlobalVariable("Aup", 0.0)
            self.integ_addGlobalVariable("Adown", 0.0)
            self.integ_addGlobalVariable("A", 0.0)
            for i in range(len(self.temperature_list)):
                self.integ_addGlobalVariable(f"hyper_e_{i}", 0.0)
            for i in range(len(self.temperature_list)-1):
                self.integ_addGlobalVariable(f"max_{i}", 0.0)

            # calc A
            A_up, A_down, vmax = [], [], []
            for i in range(len(self.temperature_list)):
                temp = self.temperature_list[i]
                log_nk = self.log_nk[i]
                
                if boost_group == EnhancedGroup.E1:
                    self.integ_addComputeGlobal(f"hyper_e_{i}", f"-1/{kB * temp:.12e}*energy1+{log_nk:.12e}")
                elif boost_group == EnhancedGroup.ALL:
                    self.integ_addComputeGlobal(f"hyper_e_{i}", f"-1/{kB * temp:.12e}*energy+{log_nk:.12e}")
                A_up.append(
                    f"{temperature:.12e}/{temp:.12e}*exp(hyper_e_{i}-vmax)"
                )
                A_down.append(f"exp(hyper_e_{i}-vmax)")
                vmax.append(f"hyper_e_{i}")

            for nmax in range(len(vmax)-1):
                if nmax == 0:
                    self.integ_addComputeGlobal(f"max_{nmax}", f"max({vmax[nmax]}, {vmax[nmax+1]})")
                else:
                    self.integ_addComputeGlobal(f"max_{nmax}", f"max(max_{nmax-1}, {vmax[nmax+1]})")

            self.integ_addComputeGlobal("vmax", f"max_{len(vmax)-2}")
            self.integ_addComputeGlobal("Aup", "+".join(A_up))
            self.integ_addComputeGlobal("Adown", "+".join(A_down))
            self.integ_addComputeGlobal("A", "Aup/Adown")
            self.integ_addUpdateContextState()
            if boost_group == EnhancedGroup.E1:
                self.integ_addComputePerDof("v", "v + dt*A*f1/m")
                self.integ_addComputePerDof("v", "v + dt*f0/m")
            else:
                self.integ_addComputePerDof("v", "v + dt*A*f/m")

        elif boost_group == EnhancedGroup.E1_AND_E2:
            self.integ_addGlobalVariable("vmax_1", 0.0)
            self.integ_addGlobalVariable("Aup_1", 0.0)
            self.integ_addGlobalVariable("Adown_1", 0.0)
            self.integ_addGlobalVariable("A_1", 0.0)
            self.integ_addGlobalVariable("vmax_2", 0.0)
            self.integ_addGlobalVariable("Aup_2", 0.0)
            self.integ_addGlobalVariable("Adown_2", 0.0)
            self.integ_addGlobalVariable("A_2", 0.0)
            # prepare for E1
            for i in range(len(self.temperature_list)):
                self.integ_addGlobalVariable(f"hyper_e_{i}_1", 0.0)
            for i in range(len(self.temperature_list)-1):
                self.integ_addGlobalVariable(f"max_{i}_1", 0.0)
            # prepare for E2
            for i in range(len(self.temperature_list)):
                self.integ_addGlobalVariable(f"hyper_e_{i}_2", 0.0)
            for i in range(len(self.temperature_list)-1):
                self.integ_addGlobalVariable(f"max_{i}_2", 0.0)

            # calc A1
            A_up_1, A_down_1, vmax_1 = [], [], []
            for i in range(len(self.temperature_list)):
                temp = self.temperature_list[i]
                log_nk = self.log_nk[i]
                
                self.integ_addComputeGlobal(f"hyper_e_{i}_1", f"-1/{kB * temp:.12e}*energy1+{log_nk:.12e}")
                A_up_1.append(
                    f"{temperature:.12e}/{temp:.12e}*exp(hyper_e_{i}_1-vmax_1)"
                )
                A_down_1.append(f"exp(hyper_e_{i}_1-vmax_1)")
                vmax_1.append(f"hyper_e_{i}_1")

            for nmax in range(len(vmax_1)-1):
                if nmax == 0:
                    self.integ_addComputeGlobal(f"max_{nmax}_1", f"max({vmax_1[nmax]}, {vmax_1[nmax+1]})")
                else:
                    self.integ_addComputeGlobal(f"max_{nmax}_1", f"max(max_{nmax-1}_1, {vmax_1[nmax+1]})")

            # calc A2
            A_up_2, A_down_2, vmax_2 = [], [], []
            for i in range(len(self.temperature_list)):
                temp = self.temperature_list[i]
                log_nk = self.log_nk2[i]
                
                self.integ_addComputeGlobal(f"hyper_e_{i}_2", f"-1/{kB * temp:.12e}*energy2+{log_nk:.12e}")
                A_up_2.append(
                    f"{temperature:.12e}/{temp:.12e}*exp(hyper_e_{i}_2-vmax_2)"
                )
                A_down_2.append(f"exp(hyper_e_{i}_2-vmax_2)")
                vmax_2.append(f"hyper_e_{i}_2")
            
            for nmax in range(len(vmax_2)-1):
                if nmax == 0:
                    self.integ_addComputeGlobal(f"max_{nmax}_2", f"max({vmax_2[nmax]}, {vmax_2[nmax+1]})")
                else:
                    self.integ_addComputeGlobal(f"max_{nmax}_2", f"max(max_{nmax-1}_2, {vmax_2[nmax+1]})")

            self.integ_addComputeGlobal("vmax_1", f"max_{len(vmax_1)-2}_1")
            self.integ_addComputeGlobal("Aup_1", "+".join(A_up_1))
            self.integ_addComputeGlobal("Adown_1", "+".join(A_down_1))
            self.integ_addComputeGlobal("A_1", "Aup_1/Adown_1")

            self.integ_addComputeGlobal("vmax_2", f"max_{len(vmax_2)-2}_2")
            self.integ_addComputeGlobal("Aup_2", "+".join(A_up_2))
            self.integ_addComputeGlobal("Adown_2", "+".join(A_down_2))
            self.integ_addComputeGlobal("A_2", "Aup_2/Adown_2")

            self.integrator.addUpdateContextState()
            self.integ_addComputePerDof("v", "v + dt*f0/m")
            self.integ_addComputePerDof("v", "v + dt*A_1*f1/m")
            self.integ_addComputePerDof("v", "v + dt*A_2*f2/m")

        self.integrator.addConstrainVelocities()
        self.integ_addComputePerDof("x", "x + 0.5*dt*v")
        self.integ_addComputePerDof("v", "a*v + b*sqrt(one_beta/m)*gaussian")
        self.integ_addComputePerDof("x", "x + 0.5*dt*v")
        self.integ_addComputePerDof("x1", "x")
        self.integrator.addConstrainPositions()
        self.integ_addComputePerDof("v", "v + (x-x1)/dt")

    def update_nk(self, energies, energies_2 = None, ratio=0.5):
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

        if energies_2 is not None:
            # calculate new nk from energies
            new_log_nk = np.zeros(self.log_nk2.shape)
            emin = np.min(energies_2)
            for n in range(new_log_nk.shape[0]):
                beta_k = 1. / (8.314 / 1000.0 * self.temperature_list[n])
                new_log_nk[n] = - np.log(np.exp(- beta_k * (energies_2 - emin)).mean()) + beta_k * emin
            # update nk
            self.log_nk2 = self.log_nk2 * (1. - ratio) + new_log_nk * ratio
            self.log_nk2 = self.log_nk2 - self.log_nk2.mean()
            print("Use log_nk2", self.log_nk2)

        self.set_integrator(self.boost_group)
        
    def load_log(self, filename):
        # return temp_list and log_nks
        import json
        with open(filename, "r") as f:
            data = json.load(f)
        return data["temp_list"], data["log_nk"], data["log_nk2"]

    def write_log(self, filename):
        data = {
            "temp_list": self.temp_list,
            "log_nk": self.log_nk,
            "log_nk2": self.log_nk2
        }
        import json
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
