import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from openits.its import ITSLangevinIntegratorGenerator
from tqdm import trange
import openits
from openits.energy_group import (
    create_nonbonded_energy_group,
    create_rotamer_torsion_energy_group,
    create_rotamer_14_energy_group,
)
from openits.utils import EnergyGroupReporter
import sys


pdb = app.PDBFile("system.pdb")
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.1 * unit.nanometer,
    constraints=app.HBonds,
)

group1, group2 = [], []
for res in pdb.topology.residues():
    for atom in res.atoms():
        if atom.index < 32:
            group1.append(atom.index)
        else:
            group2.append(atom.index)
system = create_nonbonded_energy_group(system, group1, group2, energy_group=1)

rotamers = [[6, 8], [8, 14]]
system = create_rotamer_torsion_energy_group(system, rotamers, energy_group=2)
system = create_rotamer_14_energy_group(system, rotamers, pdb.topology, energy_group=2)

system.addForce(mm.MonteCarloBarostat(1 * unit.atmosphere, 300 * unit.kelvin, 25))

# Define the temperature list
temp_list = np.arange(300, 601, 10)
int_gen = ITSLangevinIntegratorGenerator(
    temp_list, 2.0, 0.002, boost_group=openits.EnhancedGroup.E1_AND_E2
)

start_state = None
for nloop in range(10):
    simulation = app.Simulation(pdb.topology, system, int_gen.integrator)
    if start_state is None:
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
    else:
        simulation.context.setPeriodicBoxVectors(*start_state.getPeriodicBoxVectors())
        simulation.context.setPositions(start_state.getPositions())
        simulation.context.setVelocities(start_state.getVelocities())
    if nloop == 0:
        simulation.minimizeEnergy()
    energy_list_1, energy_list_2 = [], []
    for nstep in trange(100):
        simulation.step(250)
        state = simulation.context.getState(getEnergy=True, groups={1})
        energy_list_1.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        state = simulation.context.getState(getEnergy=True, groups={2})
        energy_list_2.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    energy_list_1 = np.array(energy_list_1)
    energy_list_2 = np.array(energy_list_2)
    int_gen.update_nk(energy_list_1, energies_2=energy_list_2, ratio=0.5)
    start_state = simulation.context.getState(getPositions=True, getVelocities=True)

# Run the production simulation
for force in system.getForces():
    print(force.getName())

simulation = app.Simulation(pdb.topology, system, int_gen.integrator)
simulation.context.setPeriodicBoxVectors(*start_state.getPeriodicBoxVectors())
simulation.context.setPositions(start_state.getPositions())
simulation.context.setVelocities(start_state.getVelocities())
simulation.reporters.append(app.DCDReporter("boost_all.dcd", 500))
simulation.reporters.append(
    app.StateDataReporter(
        "boost_all.out",
        500,
        step=True,
        potentialEnergy=True,
        temperature=True,
        density=True,
        remainingTime=True,
        totalSteps=5000 * 500,
        speed=True
    )
)
simulation.reporters.append(EnergyGroupReporter("boost_all.eg", 500, egroups=[0, 1]))
simulation.step(5000 * 500)