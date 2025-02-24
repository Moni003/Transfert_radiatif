### Etape 5 : reprise de l'étape 4 pour faire une animation

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importation des modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dimensions de la grille et paramètres initiaux
nb_pixels = 100                                     # Nombre de pixels par côté (pour définir l'image finale) 
x1, y1, z1 = 0, 0, 0                                # Position de l'étoile 
x2 = 1000                                           # Distance entre l'étoile et la caméra selon l'axe x (en pc)
xmin, xmax = -5000, 5000                            # Limites du plan x_face (en pc)
ymin, ymax = -5000, 5000                            # Limites du plan y_face (en pc)
nb_photons = 100000                                 # Nombre total de photons émis par l'étoile
h = 6.63 * 10**(-34)                                # Constante de Planck (en J.s)
c = 3 * 10**8                                       # Vitesse de la lumière dans le vide (en m/s)
k = 1.38 * 10**(-23)                                # Constante de Boltzmann (en J/K)
T_etoile = 7220                                     # Température effective de l'étoile (en K)
sigma_w = 2.898 * 10**(-3)                          # Constante de Wien (en m.K)
k_s = 0                                             # Coefficient de diffusion (en m-1)
l_max = sigma_w / T_etoile                          # Longueur d'onde de rayonnement (en m)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Valeurs de k_a à faire varier
ka_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Fonction pour simuler la luminance pour un ka donné
def luminance(k_a):
    k_e = k_a + k_s        # Coefficient d'extinction
    a = k_s / (k_a + k_s)  # Calcul de l'albédo

    positions_x = []
    positions_y = []
    luminances  = []

    for i in range(nb_photons):
        x, y, z = x1, y1, z1         # Position initiale du photon
        last_x, last_y = None, None

        while True:
            # Direction de propagation
            theta = np.arccos(-1 + 2 * np.random.uniform(0, 1))
            phi = 2 * np.pi * np.random.uniform(0, 1)
            n_x = np.sin(theta) * np.cos(phi)
            n_y = np.sin(theta) * np.sin(phi)
            n_z = np.cos(theta)
            n = np.array([n_x, n_y, n_z])

            # Normalisation
            norm_n = np.sqrt(n_x**2 + n_y**2 + n_z**2)
            n_x /= norm_n
            n_y /= norm_n
            n_z /= norm_n

            # Calcul de la profondeur optique et distance traversée
            tau_scatter = -np.log(1 - np.random.uniform(0, 1))
            s_x = (x2 - x) / n_x
            tau = k_e * s_x

            # Si le photon atteint directement la caméra
            if tau > tau_scatter: 
                r1 = np.array([x, y, z])
                r2 = r1 + s_x * n
                if xmin <= r2[1] <= xmax and ymin <= r2[2] <= ymax and s_x > 0:
                    last_x, last_y = r2[1], r2[2]
                break
            
            # Si le photon subit une collision : diffusion ou absorption
            else:  
                if np.random.uniform(0, 1) <= a:  # Photon diffusé
                    # Échantillonnage d'une nouvelle direction de propagation
                    theta2 = np.arccos(-1 + 2 * np.random.uniform(0, 1))
                    phi2 = 2 * np.pi * np.random.uniform(0, 1)
                    n_x2 = np.sin(theta2) * np.cos(phi2)
                    n_y2 = np.sin(theta2) * np.sin(phi2)
                    n_z2 = np.cos(theta2)
                    n2 = np.array([n_x2, n_y2, n_z2])

                    # Normalisation de la nouvelle direction de propagation
                    norm_n2  = np.sqrt(n_x2**2 + n_y2**2 + n_z2**2)
                    n_x2 /= norm_n2
                    n_y2 /= norm_n2
                    n_z2 /= norm_n2

                    # Calcul de la profondeur optique et de la distance traversée
                    s_collision = tau_scatter / k_e
                    r1 = np.array([x, y, z]) + s_collision * n
                    x, y, z = r1[1], r1[2], r1[0]

                    # Vérifier si le photon reste dans les limites de la caméra après diffusion 
                    if xmin <= x <= xmax and ymin <= y <= ymax: 
                        last_x, last_y = x, y

                    else: # Hors des limites de la caméra
                        break

                else:  # Photon absorbé
                    break

        if last_x is not None and last_y is not None:
            L = ((2 * h * c**2)/(l_max**5)) * (1 / (np.exp((h * c) / (l_max * k * T_etoile)) - 1))
            positions_x.append(last_x)
            positions_y.append(last_y)
            luminances.append(L)

    hist_luminance, _, _ = np.histogram2d(
        positions_x, positions_y, bins=nb_pixels, range=[[xmin, xmax], [ymin, ymax]], weights=luminances
    )
    return hist_luminance


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure pour l'animation
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.zeros((nb_pixels, nb_pixels)), cmap='hot', origin='lower', extent=[xmin, xmax, ymin, ymax])
cbar = plt.colorbar(im, ax=ax, label='Luminance [W·m⁻²·sr⁻¹·Hz⁻¹]')
ax.set_title(f'Luminance évoluant avec $k_a$', fontsize=14)
ax.set_xlabel('Position X [pc]')
ax.set_ylabel('Position Y [pc]')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mise à jour de chaque image dans l'animation
def update(frame):
    k_a = ka_values[frame]
    hist_luminance = luminance(k_a)
    im.set_data(hist_luminance.T)
    im.set_clim(vmin=0, vmax=np.max(hist_luminance))
    ax.set_title(f'Luminance', fontsize=12)
    return im,


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Création de l'animation
anim = FuncAnimation(fig, update, frames=len(ka_values), interval=500, blit=True)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Enregistrement de l'animation 
anim.save('luminance_animation.gif', dpi=300, writer='pillow')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage de l'animation
plt.show()
