import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import matplotlib.pyplot as plt
from openits.its import ITSLangevinIntegratorGenerator
from tqdm import trange
from openits.energy_group import (
    create_nonbonded_energy_group,
    create_rotamer_torsion_energy_group,
    create_rotamer_14_energy_group,
)
import sys


pdb = app.PDBFile("system.pdb")
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.9 * unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
    ewaldErrorTolerance=0.0005,
    hydrogenMass=4 * unit.amu
)
system.addForce(mm.MonteCarloBarostat(1 * unit.atmosphere, 300 * unit.kelvin, 25))

integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 2.0/unit.picosecond, 4*unit.femtosecond)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
simulation.minimizeEnergy()
simulation.reporters.append(app.DCDReporter("sim_temp.dcd", 500))
simulation.reporters.append(
    app.StateDataReporter(
        "sim_temp.out",
        500,
        step=True,
        potentialEnergy=True,
        temperature=True,
        density=True,
        remainingTime=True,
        totalSteps=10 * 1000 * 250,
    )
)
sim = app.SimulatedTempering(simulation, numTemperatures=31, minTemperature=300, maxTemperature=600, tempChangeInterval=500, reportInterval=500, reportFile="sim_temp.log")
sim.step(10 * 1000 * 250)