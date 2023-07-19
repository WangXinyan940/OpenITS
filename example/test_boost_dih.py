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

rotamers = [[8, 14], [14, 16]]
system = create_rotamer_torsion_energy_group(system, rotamers)
system = create_rotamer_14_energy_group(system, rotamers, pdb.topology)

# Define the temperature list
temp_list = np.arange(300, 401, 5)
log_nk = np.array(
    [
        6.49714025,
        5.71877902,
        4.9650229,
        4.2347158,
        3.52677338,
        2.84017755,
        2.17397142,
        1.52725483,
        0.89918009,
        0.28894827,
        -0.30419431,
        -0.88095926,
        -1.442019,
        -1.98800945,
        -2.5195325,
        -3.03715824,
        -3.54142705,
        -4.03285153,
        -4.51191827,
        -4.97908946,
        -5.43480444,
    ]
)
int_gen = ITSLangevinIntegratorGenerator(
    temp_list, 2.0, 0.002, log_nk=log_nk, boost_e1_only=True
)
simulation = app.Simulation(pdb.topology, system, int_gen.integrator)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
simulation.minimizeEnergy()
simulation.reporters.append(app.DCDReporter("boost_dih.dcd", 500))
simulation.reporters.append(
    app.StateDataReporter(
        "boost_dih.out",
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