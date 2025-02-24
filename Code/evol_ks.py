### Etape 3 : modéliser l'émission de photons par une étoile en prenant en compte la diffusion et l'absorption et en faisant varier k_s


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importation des modules
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dimensions de la grille et paramètres initiaux
nb_pixels = 100                                     # Nombre de pixels par côté (pour définir l'image finale) 
x1, y1, z1 = 0, 0, 0                                # Position de l'étoile 
x2 = 1000                                           # Distance entre l'étoile et la caméra selon l'axe x (en pc)
xmin, xmax = -5000, 5000                            # Limites du plan x_face (en pc)
ymin, ymax = -5000, 5000                            # Limites du plan y_face (en pc)
nb_photons = 1000000                                # Nombre total de photons émis par l'étoile
h = 6.63 * 10**(-34)                                # Constante de Planck (en J.s)
c = 3 * 10**8                                       # Vitesse de la lumière dans le vide (en m/s)
k = 1.38 * 10**(-23)                                # Constante de Boltzmann (en J/K)
T_etoile = 7220                                     # Température effective de l'étoile (en K)
sigma_w = 2.898 * 10**(-3)                          # Constante de Wien (en m.K)
k_a = 0                                             # Coefficient d'absorption (en m-1)
l_max = sigma_w / T_etoile                          # Longueur d'onde de rayonnement (en m)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Valeurs de k_s à faire varier
ks_values = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
ks_values2 = np.array([1e-6, 1e-5, 1e-1])

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Création de la figure avec 6 sous-graphiques (2 lignes, 3 colonnes)
fig, axes = plt.subplots(1, 3, figsize=(17, 15), sharex=True, sharey=True)
axes = axes.flatten()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Listes de stockage 
all_hist_luminance = []    # Stockage des histogrammes de luminance
max_luminances     = []    # Stockage des valeurs maximales de luminance
nb_photons_direct  = []    # Stockage du nombre de photons qui atteignent directement la caméra
nb_photons_diff    = []    # Stockage du nombre de photons diffusés
nb_photons_abs     = []    # Stockage du nombre de photons absorbés


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Boucle principale
for idx, k_s in enumerate(ks_values2):
    k_e = k_a + k_s        # Coefficient d'extinction
    a = k_s / (k_a + k_s)  # Calcul de l'albédo
    print(f"k_s = {k_s}, albedo (a) = {a}")  ###ajout

    positions_x = []       # Stockage des positions X
    positions_y = []       # Stockage des positions Y
    luminances  = []       # Stockage des luminances

    # Compteurs locaux de photons
    nb_direct = 0
    nb_diff   = 0
    nb_abs    = 0

    for i in range(nb_photons):
        x, y, z = x1, y1, z1         # Position initiale du photon (étoile)
        last_x, last_y = None, None  # Stockage de la dernière position du photon dans l'image

        while True:
            if x==0 and y==0 and z==0: 
                # Échantillonnage d'une direction de propagation 
                theta = np.arccos(-1 + 2 * np.random.uniform(0, 1))
                phi = 2 * np.pi * np.random.uniform(0, 1)
                n_x = np.sin(theta) * np.cos(phi)
                n_y = np.sin(theta) * np.sin(phi)
                n_z = np.cos(theta)
                n = np.array([n_x, n_y, n_z])

                # Normalisation de la direction de propagation
                norm_n = np.sqrt(n_x**2 + n_y**2 + n_z**2)
                n_x /= norm_n
                n_y /= norm_n
                n_z /= norm_n

                # Calcul de la profondeur optique et distance traversée
                tau_scatter = -np.log(1 - np.random.uniform(0, 1))  # Profondeur optique de diffusion
                s_x = (x2 - x) / n_x                                # Distance pour atteindre la caméra
                tau = k_e * s_x                                     # Profondeur optique totale
            
            else:
                tau_scatter = -np.log(1 - np.random.uniform(0, 1))  # Profondeur optique de diffusion
                s_x = (x2 - x) / n_x                                # Distance pour atteindre la caméra
                tau = k_e * s_x                                     # Profondeur optique totale


            # Si le photon atteint directement la caméra
            if tau > tau_scatter:  
                r1 = np.array([x, y, z])
                r2 = r1 + s_x * n

                # On vérifie que le photon n'est pas en dehors des limites de la caméra
                if xmin <= r2[1] <= xmax and ymin <= r2[2] <= ymax and s_x > 0:
                    last_x, last_y = r2[1], r2[2]
                    last_event = 'direct'
                break
            
            # Si le photon subit une collision : diffusion ou absorption
            else:  
                if np.random.uniform(0, 1) <= a:  # Photon diffusé
                    # Échantillonnage d'une nouvelle direction aléatoire 
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
                        last_event = 'diffusion'

                    else: # Hors des limites de la caméra
                        break

                else:  # Photon absorbé
                    last_event = 'absorption'
                    break

         # Calcul et stockage de la luminance 
        if last_x is not None and last_y is not None:
            L = ((2 * h * c**2)/(l_max**5)) * (1 / (np.exp((h * c) / (l_max * k * T_etoile)) - 1))
            positions_x.append(last_x)
            positions_y.append(last_y)
            luminances.append(L)

            # Mise à jour des compteurs selon le dernier événement
            if last_event == 'direct':
                nb_direct += 1
            elif last_event == 'diffusion':
                nb_diff += 1
            elif last_event == 'absorption':
                nb_abs += 1

    # Ajouter les compteurs dans les listes
    nb_photons_direct.append(nb_direct)
    nb_photons_diff.append(nb_diff)
    nb_photons_abs.append(nb_abs)

    # Histogramme des luminances
    hist_luminance, _, _ = np.histogram2d(
        positions_x, positions_y, bins=nb_pixels, range=[[xmin, xmax], [ymin, ymax]], weights=luminances
    )
    all_hist_luminance.append(hist_luminance)
    max_luminance = np.max(hist_luminance)   # Calcul du maximum de luminance
    max_luminances.append(max_luminance)     # Stockage du maximum

    # Affichage des sous-graphiques
    ax = axes[idx]
    im = ax.imshow(hist_luminance.T, cmap='hot', origin='lower', extent=[xmin, xmax, ymin, ymax])
    ax.set_title(f'Luminance ($k_s$={k_s:.1e} m⁻¹)', fontsize=10)
    ax.set_xlabel('Position X [pc]', fontsize=8)
    ax.set_ylabel('Position Y [pc]', fontsize=8)

    # Affichage de la valeur maximale de luminance
    ax.text(0.05, 0.9, f'Max: {max_luminance:.2e}', transform=ax.transAxes, fontsize=10, color='white')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ajuster l'espacement entre les sous-graphiques pour éviter les chevauchements
plt.subplots_adjust(hspace=0.3, wspace=0.3)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ajouter une colorbar 
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
fig.colorbar(im, cax=cbar_ax, label='Luminance [W·m⁻²·sr⁻¹·Hz⁻¹]')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ajouter un titre 
fig.suptitle(f'Luminance avec variation de $k_s$', fontsize=16, ha='center', va='top')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Enregistrement de l'image
plt.savefig('evol_ks.png', dpi=300, bbox_inches='tight')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage de la figure
plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage des compteurs de photons
print("Répartition du nombre de photons:")
for idx, k_s in enumerate(ks_values):
    print(f"k_s = {k_s:.1e}: direct = {nb_photons_direct[idx]}, diffusés = {nb_photons_diff[idx]}, absorbés = {nb_photons_abs[idx]}")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Histogramme pour la répartition du nombre de photons
x = np.arange(len(ks_values))
width = 0.3
plt.figure(figsize=(10, 6))
plt.bar(x - width, nb_photons_direct, width=width, label='Photons directs', color='blue')
plt.bar(x, nb_photons_diff, width=width, label='Photons diffusés', color='magenta')
plt.bar(x + width, nb_photons_abs, width=width, label='Photons absorbés', color='orange')
plt.xticks(x, [f"{k_s:.1e}" for k_s in ks_values])
plt.xlabel("$k_s$ [m$^{-1}$]")
plt.ylabel("Nombre de photons")
plt.title("Répartition du nombre de photons")
plt.legend()
plt.savefig('hist_evol_ks.png', dpi=300, bbox_inches='tight')
plt.show()




