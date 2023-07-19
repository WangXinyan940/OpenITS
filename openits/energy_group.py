import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
from .utils import OpenITSException


ONE_4PI_EPS0 = 138.93545764438198


def create_nonbonded_energy_group(
    system: mm.System, group1: List[int], group2: List[int]
) -> mm.System:
    nbforce: mm.NonbondedForce = None
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nbforce = force
    if nbforce is None:
        raise OpenITSException(
            "Cannot find openmm.NonbondedForce in openmm.System, this method can only support systems where intermolecular interactions are described solely by openmm.NonbondedForce."
        )
    upforce = mm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6)+ONE_4PI_EPS0*qq/r; sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); qq=q1*q2")
    dnforce = mm.CustomNonbondedForce("-4*epsilon*((sigma/r)^12-(sigma/r)^6)-ONE_4PI_EPS0*qq/r; sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); qq=q1*q2")
    upforce.setName("Positive Interaction")
    dnforce.setName("Negative Interaction")
    upforce.addPerParticleParameter("q")
    upforce.addPerParticleParameter("sigma")
    upforce.addPerParticleParameter("epsilon")
    upforce.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    dnforce.addPerParticleParameter("q")
    dnforce.addPerParticleParameter("sigma")
    dnforce.addPerParticleParameter("epsilon")
    dnforce.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    for nparticle in range(nbforce.getNumParticles()):
        chrg, sig, eps = nbforce.getParticleParameters(nparticle)
        upforce.addParticle([chrg, sig, eps])
        dnforce.addParticle([chrg, sig, eps])
    for nexcl in range(nbforce.getNumExceptions()):
        ii, jj, chrgprod, sig, eps = nbforce.getExceptionParameters(nexcl)
        upforce.addExclusion(ii, jj)
        dnforce.addExclusion(ii, jj)
    if nbforce.getNonbondedMethod() in [app.PME, app.CutoffPeriodic]:
        upforce.setNonbondedMethod(upforce.CutoffPeriodic)
        dnforce.setNonbondedMethod(upforce.CutoffPeriodic)
    else:
        upforce.setNonbondedMethod(upforce.NoCutoff)
        dnforce.setNonbondedMethod(upforce.NoCutoff)
    upforce.addInteractionGroup(group1, group2)
    dnforce.addInteractionGroup(group1, group2)
    upforce.setForceGroup(1)
    dnforce.setForceGroup(0)
    system.addForce(upforce)
    system.addForce(dnforce)
    return system


def create_rotamer_torsion_energy_group(
    system: mm.System, rotamers: List[List[int]]
) -> mm.System:
    pass


def create_rotamer_14_energy_group(
    system: mm.System, rotamers: List[List[int]]
) -> mm.System:
    pass
