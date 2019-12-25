"Plots"
import numpy as onp

def W3(i,R1,axs,**kwargs):
  R_plt = onp.array(R1[i])
  axs[0].scatter(R_plt[:, 0], R_plt[:, 1],**kwargs)
  axs[1].scatter(R_plt[:, 0], R_plt[:, 2],**kwargs)
  axs[2].scatter(R_plt[:, 1], R_plt[:, 2],**kwargs)


def W3_animate(i,R1):
  R_plt = onp.array(R1[i])
  ls[0].set_data(R_plt[:, 0], R_plt[:, 1],)
  ls[1].set_data(R_plt[:, 0], R_plt[:, 2],)
  ls[2].set_data(R_plt[:, 1], R_plt[:, 2],)
