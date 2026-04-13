import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from tqdm import tqdm   



def Gerchberg_Saxton(input_amp : np.array,
                     target_amp : np.array,
                     initial_phase : np.array,
                     N_iterations : int = 100,
                     plot_every : int = 100) -> np.array:
    
    phase_beam = initial_phase

    for i in tqdm(range(N_iterations+1)):
        field_beam = input_amp * np.exp(1j*phase_beam)

        field_focus = np.fft.fftshift(np.fft.fft2(field_beam))
        phase_focus = np.angle(field_focus)

        if i % plot_every == 0 or i == N_iterations:
            ticks_res = 4
            ticks = np.arange(0, N+1, N//ticks_res)
            fplane_ticks = np.round(np.arange(-fieldsize_fplane/2, fieldsize_fplane/2+1/N, fieldsize_fplane/ticks_res)*1e3, 2)

            phase_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            phase_tick_names = ['-π', '-π/2', 0, 'π/2', 'π']

            fig, axs = plt.subplots(1, 2, figsize=(7, 14))

            axs[0].set_title(f'Iteration {i}')
            im_phase = axs[0].imshow(phase_beam)
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im_phase, cax=cax)
            cbar.set_ticks(phase_ticks)
            cbar.set_ticklabels(phase_tick_names)

            x1, x2 = 610, 630
            y1, y2 = x1, x2

            axins = inset_axes(axs[0], width="30%", height="30%", loc="upper right")
            axins.imshow(phase_beam)
            axins.set_xlim(x1, x2)
            axins.set_ylim(y2, y1)
            #axins.set_xticks([])
            #axins.set_yticks([])

            mark_inset(axs[0], axins, loc1=2, loc2=4)

            axs[1].imshow(np.abs(field_focus)**2)
            axs[1].set_xticks(ticks, fplane_ticks)
            axs[1].set_yticks(ticks, fplane_ticks)
            axs[1].yaxis.tick_right()
            axs[1].yaxis.set_label_position("right")    
            axs[1].set_xlabel('[mm]')
            plt.show()
        
        field_focus = target_amp * np.exp(1j*phase_focus)

        field_beam = np.fft.ifft2(np.fft.ifftshift(field_focus))
        phase_beam = np.angle(field_beam)
        #

    return phase_beam