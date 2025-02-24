### Etape 1 : modéliser l'émission de photons par une étoile  


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importation des modules
import numpy as np
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dimensions de la grille
nb_pixels = 100                                     # Nombre de pixels par côté (pour définir l'image finale) [meilleure visualisation avec 100x100]
x_etoile, y_etoile, z_etoile = 0, 0, 0              # Position de l'étoile 
x2 = 1000                                           # Distance entre l'étoile et la caméra selon l'axe x (en pc)
xmin, xmax = -5000, 5000                            # Limites du plan x_face (en pc)
ymin, ymax = -5000, 5000                            # Limites du plan y_face (en pc)
nb_photons = 1000000                                # Nombre total de photons émis par l'étoile
h = 6.63 * 10**(-34)                                # Constante de Planck (en J.s)
c = 3 * 10**8                                       # Vitesse de la lumière dans le vide (en m/s)
k = 1.38 * 10**(-23)                                # Constante de Boltzmann (en J/K)
T_etoile = 70220                                     # Température effective de l'étoile (en K)
sigma_w = 2.898 * 10**(-3)                          # Constante de Wien (en m.K)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Listes pour stocker les positions et la luminance des photons
positions_x = []
positions_y = []
luminance   = []


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialisation des composantes de la direction de propagation
n_x = np.zeros(nb_photons)
n_y = np.zeros(nb_photons)
n_z = np.zeros(nb_photons)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calcul de la longueur d'onde de rayonnement de l'étoile
l_max = sigma_w / T_etoile


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initialisation du compteur de photons émis
nb_photons_emis = 0


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Boucle principale
for i in range(nb_photons):
    # Échantillonnage de θ ∈ [0, π] et φ ∈ [0, 2π]
    theta = np.arccos(-1 + 2 * np.random.uniform(0, 1))      
    phi   = 2 * np.pi * np.random.uniform(0, 1)               

    # Conversion en coordonnées cartésiennes avec calcul de n_x, n_y, n_z 
    n_x[i] = np.sin(theta) * np.cos(phi)
    n_y[i] = np.sin(theta) * np.sin(phi)
    n_z[i] = np.cos(theta)

    # Normalisation des directions
    norm_n  = np.sqrt(n_x[i]**2 + n_y[i]**2 + n_z[i]**2)
    n_x[i] /= norm_n
    n_y[i] /= norm_n
    n_z[i] /= norm_n

    # Calcul de s_x 
    s_x = (x2 - x_etoile) / n_x[i]

    # Calcul du vecteur r2
    r1 = np.array([x_etoile, y_etoile, z_etoile])
    n = np.array([n_x[i], n_y[i], n_z[i]])
    r2 = r1 + s_x * n
    
    # Vérification si r2 est dans les limites du plan x_face
    if xmin <= r2[1] <= xmax and ymin <= r2[2] <= ymax and s_x > 0:
        # Incrémenter le compteur de photons émis
        nb_photons_emis = nb_photons_emis + 1

        # Enregistrer les positions x et y pour chaque photon
        positions_x.append(r2[1])
        positions_y.append(r2[2])

        # Calcul de la luminance  
        L = (2 * h * c**2)/(l_max**5) * (1 / (np.exp((h * c) / (l_max * k * T_etoile)) - 1))

        # Enregistrer la luminance
        luminance.append(L)   


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Conversion des listes en tableaux numpy
positions_x = np.array(positions_x)
positions_y = np.array(positions_y)
luminances  = np.array(luminance)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage du nombre de photons émis
print(f"Nombre de photons émis : {nb_photons_emis}")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calcul de la luminance par pixel
hist_luminance, xedges, yedges = np.histogram2d(positions_x, positions_y, bins=nb_pixels, range=[[xmin, xmax], [ymin, ymax]], weights=luminances)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage de l'histogramme de luminance
plt.figure(figsize=(8, 8))
plt.imshow(hist_luminance.T, cmap='hot', origin='lower', extent=[xmin, xmax, ymin, ymax])
plt.colorbar(label='Luminance [W·m⁻²·sr⁻¹·Hz⁻¹]')
plt.title("Distribution de la luminance")
plt.xlabel("Position X [pc]") 
plt.ylabel("Position Y [pc]")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Enregistrement du graphe dans un fichier
plt.savefig('emission.png', dpi=300, bbox_inches='tight')


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Affichage du graphe
plt.show()