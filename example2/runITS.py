import openmm.app as app
import openmm as mm
import openmm.unit as unit
from tqdm import tqdm, trange
import numpy as np
import openits


temperature_min = 300.0
temperature_max = 600.0
temperature_reps = 30

max_minimize_step = 1000
nvt_res_length = 100 # ps
npt_res_length = 200 # ps
npt_free_length = 1000 # ps

optimize_samples = 100
optimize_step = 250 # record energy every {optimize_step} steps
optimize_loop = 10
 
production_time = 100 * 1000 # ps

lig_name = "MOL"
bio_residues = ["MG"]


gro = app.GromacsGroFile("init.gro")
top = app.GromacsTopFile("topol.top", periodicBoxVectors=gro.getPeriodicBoxVectors())
system = top.createSystem(app.PME, nonbondedCutoff=0.9*unit.nanometer, constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)
with open("system.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(system))


# OPENMM equilibrate
# 1. NVT with heavy atom restraint
with open("system.xml", "r") as f:
    text = "".join(f.readlines())
    system = mm.XmlSerializer.deserialize(text)
ref_pos = gro.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
ref_idx = [atom.index for atom in top.topology.atoms() if atom.element.symbol != "H" and atom.name != "OW"]
rmsd = mm.RMSDForce(ref_pos, ref_idx)
restraint = mm.CustomCVForce("1000*rmsd^2")
restraint.addCollectiveVariable("rmsd", rmsd)
system.addForce(restraint)
integ = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtosecond)
context = mm.Context(system, integ)
context.setPositions(gro.getPositions())
print(">>> Minimize energy...")
mm.LocalEnergyMinimizer.minimize(context, maxIterations=max_minimize_step)
context.setVelocitiesToTemperature(300*unit.kelvin)
print(">>> Run NVT equilibration with heavy atom restraint")
for nstep in trange(int(nvt_res_length)):
    context._integrator.step(250)
nvt_res_state = context.getState(getPositions=True, getVelocities=True)


# OPENMM equilibrate
# 2. NPT with heavy atom restraint
with open("system.xml", "r") as f:
    text = "".join(f.readlines())
    system = mm.XmlSerializer.deserialize(text)
system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
ref_pos = gro.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
ref_idx = [atom.index for atom in top.topology.atoms() if atom.element.symbol != "H" and atom.name != "OW"]
rmsd = mm.RMSDForce(ref_pos, ref_idx)
restraint = mm.CustomCVForce("500*rmsd^2")
restraint.addCollectiveVariable("rmsd", rmsd)
system.addForce(restraint)
integ = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtosecond)
context = mm.Context(system, integ)
context.setPositions(nvt_res_state.getPositions())
context.setVelocities(nvt_res_state.getVelocities())
context.setPeriodicBoxVectors(*nvt_res_state.getPeriodicBoxVectors())
print(">>> Run NPT equilibration with heavy atom restraint")
for nstep in trange(int(npt_res_length)):
    context._integrator.step(250)
npt_res_state = context.getState(getPositions=True, getVelocities=True)


# OPENMM equilibrate
# 3. NPT with no restraint
with open("system.xml", "r") as f:
    text = "".join(f.readlines())
    system = mm.XmlSerializer.deserialize(text)
system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
integ = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 4*unit.femtosecond)
context = mm.Context(system, integ)
context.setPositions(npt_res_state.getPositions())
context.setVelocities(npt_res_state.getVelocities())
context.setPeriodicBoxVectors(*npt_res_state.getPeriodicBoxVectors())
print(">>> Run NPT equilibration with no restraint")
for nstep in trange(int(npt_free_length)):
    context._integrator.step(250)
npt_free_state = context.getState(getPositions=True, getVelocities=True)


# set three groups: Protein, Ligand, Others
bio_idx, lig_idx, sol_idx = [], [], []
for atom in top.topology.atoms():
    if atom.residue.name == lig_name:
        lig_idx.append(atom.index)
    elif atom.residue.name in app.PDBFile._standardResidues + bio_residues and atom.residue.name != "HOH":
        bio_idx.append(atom.index)
    else:
        sol_idx.append(atom.index)
print(f">>> Group bio: {len(bio_idx)} atoms")
print(f">>> Group ligand: {len(lig_idx)} atoms")
print(f">>> Group sol: {len(sol_idx)} atoms")


# add interaction boost
with open("system.xml", "r") as f:
    text = "".join(f.readlines())
    system = mm.XmlSerializer.deserialize(text)
system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
system = openits.create_nonbonded_energy_group(system, lig_idx, bio_idx+sol_idx, scale=1.0, energy_group=1)
system = openits.create_nonbonded_energy_group(system, bio_idx, sol_idx, scale=0.5, energy_group=2)


# optimize log_nk values
print("Optimize ITS parameters")
temperature_list = np.linspace(temperature_min, temperature_max, temperature_reps)
its_gen = openits.ITSLangevinIntegratorGenerator(temperature_list=temperature_list, friction=1.0, dt=0.004, boost_group=openits.EnhancedGroup.E1_AND_E2, verbose=False)

for nloop in range(optimize_loop):
    energy_1, energy_2 = [], []
    context = mm.Context(system, its_gen.integrator)
    if nloop == 0:
        context.setPositions(npt_free_state.getPositions())
        context.setVelocities(npt_free_state.getVelocities())
        context.setPeriodicBoxVectors(*npt_free_state.getPeriodicBoxVectors())
    else:
        context.setPositions(npt_opt_state.getPositions())
        context.setVelocities(npt_opt_state.getVelocities())
        context.setPeriodicBoxVectors(*npt_opt_state.getPeriodicBoxVectors())
    for nstep in trange(optimize_samples):
        context._integrator.step(optimize_step)
        state_1 = context.getState(getEnergy=True, groups={1})
        state_2 = context.getState(getEnergy=True, groups={2})
        energy_1.append(state_1.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        energy_2.append(state_2.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    its_gen.update_nk(energy_1, energies_2=energy_2, ratio=0.5)
    npt_opt_state = context.getState(getPositions=True, getVelocities=True)


# run production MD
print(">>> Start production")
with open("system.xml", "r") as f:
    text = "".join(f.readlines())
    system = mm.XmlSerializer.deserialize(text)
system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
system = openits.create_nonbonded_energy_group(system, lig_idx, bio_idx+sol_idx, scale=1.0, energy_group=1)
system = openits.create_nonbonded_energy_group(system, bio_idx, sol_idx, scale=0.5, energy_group=2)
simulation = app.Simulation(top.topology, system, its_gen.integrator)
simulation.context.setPeriodicBoxVectors(*npt_free_state.getPeriodicBoxVectors())
simulation.context.setPositions(npt_free_state.getPositions())
simulation.context.setVelocities(npt_free_state.getVelocities())
simulation.reporters.append(app.DCDReporter("boost_all.dcd", 10000))
simulation.reporters.append(
    app.StateDataReporter(
        "boost_all.out",
        10000,
        step=True,
        time=True, 
        potentialEnergy=True,
        temperature=True,
        density=True,
        remainingTime=True,
        totalSteps=production_time * 250,
        speed=True,
    )
)
simulation.reporters.append(
    openits.utils.EnergyGroupReporter("boost_all.eg", 10000, egroups=[0, 1, 2])
)
simulation.step(production_time * 250)