import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import sys


inp = sys.argv[1]
out = sys.argv[2]
traj = md.load(inp, top="system.pdb")
phi_angles = md.compute_phi(traj)
psi_angles = md.compute_psi(traj)

# 将计算出的角度从弧度转换为度
phi_angles = np.rad2deg(phi_angles[1])
psi_angles = np.rad2deg(psi_angles[1])

# 绘制Ramachandran plot
plt.figure(figsize=(6, 6))
plt.scatter(phi_angles, psi_angles, s=5, alpha=0.5)
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.xlabel("Phi angles (degrees)")
plt.ylabel("Psi angles (degrees)")
plt.title("Ramachandran plot")
plt.grid()
plt.savefig(out)