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
    nonbondedCutoff=1.1 * unit.nanometer,
    constraints=app.HBonds,
)
system.addForce(mm.MonteCarloBarostat(1 * unit.atmosphere, 300 * unit.kelvin, 25))

integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 2.0/unit.picosecond, 2*unit.femtosecond)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
simulation.minimizeEnergy()
simulation.reporters.append(app.DCDReporter("ref_dih.dcd", 500))
simulation.reporters.append(
    app.StateDataReporter(
        "ref_dih.out",
        500,
        step=True,
        potentialEnergy=True,
        temperature=True,
        density=True,
        remainingTime=True,
        totalSteps=5000 * 500,
    )
)
simulation.step(5000 * 500)