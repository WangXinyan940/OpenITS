import openmm as mm
import openmm.app as app
import openmm.unit as unit
from .its import EnhancedGroup, ITSLangevinIntegratorGenerator
from .utils import OpenITSException
import numpy as np
from tqdm import trange


def optimize_logNk(
    int_gen: ITSLangevinIntegratorGenerator,
    system: mm.System,
    init_pos,
    cell,
    platform=None,
    n_loop=10,
    n_sample=100,
    n_step=250,
    ratio=0.75,
    return_state=False,
    verbose=True,
):
    energy_list_1, energy_list_2 = [], []
    state = None
    for lp in range(n_loop):
        if platform is None:
            context = mm.Context(system, int_gen.integrator)
        else:
            context = mm.Context(system, int_gen.integrator, platform)
        if lp == 0:
            context.setPositions(init_pos)
            low_temp = int_gen.temperature_list[0]
            context.setVelocitiesToTemperature(low_temp * unit.kelvin)
            context.setPeriodicBoxVectors(*cell)
            mm.LocalEnergyMinimizer.minimize(context, maxIterations=1000)
            if verbose:
                if int_gen.boost_group != EnhancedGroup.E1_AND_E2:
                    print(int_gen.log_nk)
                else:
                    print(int_gen.log_nk, int_gen.log_nk2)
        else:
            context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            context.setPositions(state.getPositions())
            context.setVelocities(state.getVelocities())
        for nsample in trange(n_sample):
            context.step(n_step)
            if int_gen.boost_group == EnhancedGroup.ALL:
                state = context.getState(getEnergy=True)
                energy_list_1.append(
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                )
            elif int_gen.boost_group == EnhancedGroup.E1:
                state = context.getState(getEnergy=True, groups={1})
                energy_list_1.append(
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                )
            elif int_gen.boost_group == EnhancedGroup.E1_AND_E2:
                state = context.getState(getEnergy=True, groups={1})
                energy_list_1.append(
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                )
                state = context.getState(getEnergy=True, groups={2})
                energy_list_2.append(
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                )
            else:
                raise OpenITSException(f"Unknown boost group {int_gen.boost_group}")
        energy_list_1 = np.array(energy_list_1)
        energy_list_2 = np.array(energy_list_2)
        if (
            int_gen.boost_group == EnhancedGroup.ALL
            or int_gen.boost_group == EnhancedGroup.E1
        ):
            int_gen.update_nk(energy_list_1, ratio=ratio)
        elif int_gen.boost_group == EnhancedGroup.E1_AND_E2:
            int_gen.update_nk(energy_list_1, energy_list_2, ratio=ratio)
        state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    if return_state:
        return state
    return None
