import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
from .utils import OpenITSException


ONE_4PI_EPS0 = 138.93545764438198


def create_nonbonded_energy_group(
    system: mm.System, group1: List[int], group2: List[int], scale: float = 1.0
) -> mm.System:
    nbforce: mm.NonbondedForce = None
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nbforce = force
    if nbforce is None:
        raise OpenITSException(
            "Cannot find openmm.NonbondedForce in openmm.System, this method can only support systems where intermolecular interactions are described solely by openmm.NonbondedForce."
        )
    upforce = mm.CustomNonbondedForce(
        f"{scale}*4*epsilon*((sigma/r)^12-(sigma/r)^6)+{scale}*ONE_4PI_EPS0*qq/r; sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); qq=q1*q2"
    )
    dnforce = mm.CustomNonbondedForce(
        f"-{scale}*4*epsilon*((sigma/r)^12-(sigma/r)^6)-{scale}*ONE_4PI_EPS0*qq/r; sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); qq=q1*q2"
    )
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


def check_rotamer_in_list(rot: List[int], rotlist: List[List[int]]) -> bool:
    for r in rotlist:
        if rot[0] == r[0] and rot[1] == r[1]:
            return True
        elif rot[0] == r[1] and rot[1] == r[0]:
            return True
    return False


def create_rotamer_torsion_energy_group(
    system: mm.System, rotamers: List[List[int]], scale: float = 1.0
) -> mm.System:
    torsionforce = None
    for force in system.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            torsionforce = force
    if torsionforce is None:
        raise OpenITSException(
            "Cannot find openmm.PeriodicTorsionForce in openmm.System, this method can only support systems where torsional interactions are described solely by openmm.PeriodicTorsionForce."
        )

    upforce = mm.PeriodicTorsionForce()
    dnforce = mm.PeriodicTorsionForce()
    upforce.setName("Positive Interaction")
    dnforce.setName("Negative Interaction")
    for ntorsion in range(torsionforce.getNumTorsions()):
        i, j, k, l, per, phase, k = torsionforce.getTorsionParameters(ntorsion)
        if check_rotamer_in_list([j, k], rotamers):
            upforce.addTorsion(i, j, k, l, per, phase, scale * k)
            dnforce.addTorsion(i, j, k, l, per, phase, - scale * k)
    upforce.setForceGroup(1)
    dnforce.setForceGroup(0)
    system.addForce(upforce)
    system.addForce(dnforce)
    return system


def create_rotamer_14_energy_group(
    system: mm.System, rotamers: List[List[int]], topology: app.Topology, scale: float = 1.0
) -> mm.System:
    torsionforce = None
    nbforce = None
    for force in system.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            torsionforce = force
        elif isinstance(force, mm.NonbondedForce):
            nbforce = force
    if torsionforce is None:
        raise OpenITSException(
            "Cannot find openmm.PeriodicTorsionForce in openmm.System, this method can only support systems where torsional interactions are described solely by openmm.PeriodicTorsionForce."
        )
    if nbforce is None:
        raise OpenITSException(
            "Cannot find openmm.NonbondedForce in openmm.System, this method can only support systems where intermolecular interactions are described solely by openmm.NonbondedForce."
        )
    
    # build bond connection
    conn = []
    for atom in topology.atoms():
        conn.append([])
    for bond in topology.bonds():
        i1 = bond.atom1.index
        i2 = bond.atom2.index
        conn[i1].append(i2)
        conn[i2].append(i1)       
    
    # find 1-4 pairs around the rotamers
    pairs = []
    for rotamer in rotamers:
        for ii in  [i for i in conn[rotamer[0]] if i not in rotamer]:
            for jj in [j for j in conn[rotamer[1]] if j not in rotamer]:
                pairs.append([ii, jj])


    upforce = mm.CustomBondForce(
        f"step(1.1-r)*{scale}*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+ONE_4PI_EPS0*qq/r)"
    )
    dnforce = mm.CustomBondForce(
        f"step(1.1-r)*{scale}*(-4*epsilon*((sigma/r)^12-(sigma/r)^6)-ONE_4PI_EPS0*qq/r)"
    )
    upforce.setName("Positive Interaction")
    dnforce.setName("Negative Interaction")
    upforce.addPerBondParameter("qq")
    upforce.addPerBondParameter("sigma")
    upforce.addPerBondParameter("epsilon")
    upforce.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    dnforce.addPerBondParameter("qq")
    dnforce.addPerBondParameter("sigma")
    dnforce.addPerBondParameter("epsilon")
    dnforce.addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0)
    for nexcl in range(nbforce.getNumExceptions()):
        ii, jj, chrgprod, sig, eps = nbforce.getExceptionParameters(nexcl)
        if check_rotamer_in_list([ii, jj], pairs):
            upforce.addBond(ii, jj, [chrgprod, sig, eps])
            dnforce.addBond(ii, jj, [chrgprod, sig, eps])
    upforce.setForceGroup(1)
    dnforce.setForceGroup(0)
    system.addForce(upforce)
    system.addForce(dnforce)
    return system