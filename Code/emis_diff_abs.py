### Etape 2 : modéliser l'émission de photons par une étoile en prenant en compte la diffusion et l'absorption (milieu quasi transparent)


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
k_s = 1e-5                                          # Coefficient de diffusion (en m-1)
k_e = k_a + k_s                                     # Coefficient d'extinction (en m-1)
l_max = sigma_w / T_etoile                          # Longueur d'onde de rayonnement (en m)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Listes pour stocker les positions et la luminance des photons
positions_x = []   # Stockage des positions X
positions_y = []   # Stockage des positions Y
luminances  = []   # Stockage des luminances


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calcul de la longueur d'onde de rayonnement de l'étoile
l_max = sigma_w / T_etoile


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calcul de l'albédo
a = k_s / (k_a + k_s)
print("Albedo : ", a)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialisation des compteurs de photons
nb_photons_abs    = 0
nb_photons_diff   = 0
nb_photons_direct = 0


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Boucle principale
for i in range(nb_photons):
    x, y, z = x1, y1, z1                    # Position initiale du photon (étoile)
    last_x, last_y = None, None             # Stockage de la dernière position du photon dans l'image
    last_event = None                       # Dernier événement (émission, diffusion, absorption)

    while True:
        # Échantillonnage d'une direction de propagation 
        theta = np.arccos(-1 + 2 * np.random.uniform(0, 1))
        phi = 2 * np.pi * np.random.uniform(0, 1)
        n_x = np.sin(theta) * np.cos(phi)
        n_y = np.sin(theta) * np.sin(phi)
        n_z = np.cos(theta)
        n = np.array([n_x, n_y, n_z])

        # Normalisation de la direction de propagation
        norm_n  = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        n_x /= norm_n
        n_y /= norm_n
        n_z /= norm_n

        # Calcul de la profondeur optique et de la distance traversée
        tau_scatter = -np.log(1 - np.random.uniform(0, 1))  # Profondeur optique de diffusion
        s_x = (x2 - x) / n_x                                # Distance pour atteindre la caméra
        tau = k_e * s_x                                     # Profondeur optique totale

        # Si le photon atteint directement la caméra
        if tau > tau_scatter and s_x > 0 :
            r1 = np.array([x, y, z])
            r2 = r1 + s_x * n

            # On vérifie que le photon n'est pas en dehors des limites de la caméra
            if xmin <= r2[1] <= xmax and ymin <= r2[2] <= ymax:
                last_x, last_y = r2[1], r2[2]                               # Enregistrer la position finale du photon
                nb_photons_direct += 1
                #print(f"Photon atteint la caméra à {last_x}, {last_y}")    # Décommenter pour afficher la position du photon atteignant directement la caméra
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
                s_collision = tau_scatter / k_e                 # Distance jusqu'à la collision
                r1 = np.array([x, y, z]) + s_collision * n2     # Calcul de la position après diffusion 
                x, y = r1[1], r1[2]                             # Mise à jour de la position (cette position deviendra la nouvelle position initiale où le photon est émis)

                # Vérifier si le photon reste dans les limites de la caméra après diffusion 
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    last_x, last_y = x, y                       # Mise à jour de la dernière position (pour éviter de calculer la luminance pour plusieurs diffusions d'un même photon)
                    last_event = 'diffusion'                    # Le dernier événement est une diffusion
                    #print(f"Photon diffusé à {x}, {y}")        # Décommenter pour afficher la position du photon diffusé
                else:
                    break                                       # Le photon sort de la caméra après diffusion

            else:  # Photon absorbé
                nb_photons_abs += 1
                last_event = 'absorption'
                break 

    # Calcul et stockage de la luminance 
    if last_x is not None and last_y is not None:
        # Incrémenter le compteur de diffusion selon le dernier événement
        if last_event == 'diffusion':
            nb_photons_diff += 1

        if last_event == 'absorption':
            nb_photons_abs += 1

        # Calcul de la luminance du photon pour le dernier événement
        L = ((2 * h * c**2)/(l_max**5)) * (1 / (np.exp((h * c) / (l_max * k * T_etoile)) - 1)) 
        positions_x.append(last_x)  # Stockage de la dernière position X
        positions_y.append(last_y)  # Stockage de la dernière position Y
        luminances.append(L)        # Stockage de la luminance finale


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage des résultats
print(f"Nombre de photons qui continuent leur chemin : {nb_photons_direct}")
print(f"Nombre de photons diffusés : {nb_photons_diff}")
print(f"Nombre de photons absorbés : {nb_photons_abs}")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage de l'histogramme de luminance
hist_luminance, _, _ = np.histogram2d(positions_x, positions_y, bins=nb_pixels, range=[[xmin, xmax], [ymin, ymax]], weights=luminances)
max_lum = np.max(hist_luminance)
print("Max:", max_lum)

plt.figure(figsize=(8, 8))
plt.imshow(hist_luminance.T, cmap='hot', origin='lower', extent=[xmin, xmax, ymin, ymax])
plt.colorbar(label='Luminance [W·m⁻²·sr⁻¹·Hz⁻¹]')
plt.title("Distribution de la luminance")
plt.xlabel("Position X [pc]")
plt.ylabel("Position Y [pc]")
plt.grid(False)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Enregistrement de l'image
plt.savefig('emis_diff_abs.png', dpi=300, bbox_inches='tight')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage de la figure
plt.show()




