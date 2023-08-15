import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import sys


# compute dihedral-dihedral distribution
dih_list = [
    [4, 6, 8, 14],
    [6, 8, 14, 16]
]


plt.figure(figsize=(14,8))
plt.subplot(2, 3, 1)

traj = md.load("ref_dih.dcd", top="system.pdb")[20:]
dih_ref = md.compute_dihedrals(traj, dih_list)
traj = md.load("boost_dih.dcd", top="system.pdb")
dih_boost = md.compute_dihedrals(traj, dih_list)
plt.scatter(dih_ref[:, 0], dih_ref[:, 1], label="ref", alpha=0.5)
plt.scatter(dih_boost[:, 0], dih_boost[:, 1], label="boost", alpha=0.5)
plt.scatter(dih_boost[-1,0], dih_boost[-1,1])
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

plt.xlabel("Phi angles (degrees)")
plt.ylabel("Psi angles (degrees)")
plt.title("Ref vs dihedral boost")
plt.grid()

plt.subplot(2, 3, 2)

traj = md.load("ref_dih.dcd", top="system.pdb")[20:]
dih_ref = md.compute_dihedrals(traj, dih_list)
traj = md.load("boost_all.dcd", top="system.pdb")
dih_boost = md.compute_dihedrals(traj, dih_list)
plt.scatter(dih_ref[:, 0], dih_ref[:, 1], label="ref", alpha=0.5)
plt.scatter(dih_boost[:, 0], dih_boost[:, 1], label="boost", alpha=0.5)
plt.scatter(dih_boost[-1,0], dih_boost[-1,1])
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

plt.xlabel("Phi angles (degrees)")
plt.ylabel("Psi angles (degrees)")
plt.title("Ref vs dih & nb boost")
plt.grid()

plt.subplot(2, 3, 3)

traj = md.load("ref_dih.dcd", top="system.pdb")[20:]
dih_ref = md.compute_dihedrals(traj, dih_list)
traj = md.load("boost_nb.dcd", top="system.pdb")
dih_boost = md.compute_dihedrals(traj, dih_list)
plt.scatter(dih_ref[:, 0], dih_ref[:, 1], label="ref", alpha=0.5)
plt.scatter(dih_boost[:, 0], dih_boost[:, 1], label="boost", alpha=0.5)
plt.scatter(dih_boost[-1,0], dih_boost[-1,1])
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

plt.xlabel("Phi angles (degrees)")
plt.ylabel("Psi angles (degrees)")
plt.title("Ref vs nb boost")
plt.grid()

plt.subplot(2, 3, 4)

traj = md.load("ref_dih.dcd", top="system.pdb")[20:]
dih_ref = md.compute_dihedrals(traj, dih_list)
traj = md.load("boost_its.dcd", top="system.pdb")
dih_boost = md.compute_dihedrals(traj, dih_list)
plt.scatter(dih_ref[:, 0], dih_ref[:, 1], label="ref", alpha=0.5)
plt.scatter(dih_boost[:, 0], dih_boost[:, 1], label="boost", alpha=0.5)
plt.scatter(dih_boost[-1,0], dih_boost[-1,1])
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

plt.xlabel("Phi angles (degrees)")
plt.ylabel("Psi angles (degrees)")
plt.title("Ref vs all boost")
plt.grid()

plt.subplot(2, 3, 5)

traj = md.load("ref_dih.dcd", top="system.pdb")[20:]
dih_ref = md.compute_dihedrals(traj, dih_list)
traj = md.load("sim_temp.dcd", top="system.pdb")
dih_boost = md.compute_dihedrals(traj, dih_list)
plt.scatter(dih_ref[:, 0], dih_ref[:, 1], label="ref", alpha=0.5)
plt.scatter(dih_boost[:, 0], dih_boost[:, 1], label="simtemp", alpha=0.5)
plt.scatter(dih_boost[-1,0], dih_boost[-1,1])
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

plt.xlabel("Phi angles (degrees)")
plt.ylabel("Psi angles (degrees)")
plt.title("Ref vs all simulating tempering")
plt.grid()


plt.savefig("compare.png")