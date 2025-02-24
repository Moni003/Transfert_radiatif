# Transfert radiatif dans le disque galactique par la méthode de Monte-Carlo

Afin d'assurer une bonne compréhension des codes il faut suivre dans l'ordre les étapes suivantes : 


1/ emission.py : On prend en compte une émission simple de photons par étoile.

2/ emis_diff_abs.py : Le code précédent ne fait pas appel aux principes de diffusion et d'absorption des photons contrairement à celui-ci.

3/ evol_ks.py : En fixant k_a à zero, on fait varier la valeur de k_s.

4/ evol_ka.py : Cette fois on fixe k_s et on fait varier k_a.

5/ evol_ka_anim.py : On reprend le code précédent mais on en fait une animation (spécifiquement pour la soutenance)

6/ evol_ks_false.py : Dans les codes précédents on a considéré que x=r2[1], y=r2[2], z=r2[0]. 
                      On pourrait naturellement penser qu'on doit avoir x=r2[0], y=r2[1], z=r2[2]. 
                      On verra grâce à ce code que prendre en compte la deuxième option est incorrecte. 
