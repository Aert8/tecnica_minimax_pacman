
import math


class FuncionHeuristica:
    def __init__(self, max_depth, peso_h1=0.5, peso_h2=0.5):
        self.peso_h1 = float(peso_h1)
        self.peso_h2 = float(peso_h2)
        self.max_depth = max_depth

    def distancia_manhattan(self, ghost_x, ghost_y, pacman_x, pacman_y):
        """
        h(n): distancia Manhattan entre fantasma y pacman.
        Se invierte a negativo para que MAX favorezca distancia minima.
        """
        distancia = abs(ghost_x - pacman_x) + abs(ghost_y - pacman_y)
        return -distancia
    
    def distancia_manhattan_normalizada(self, ghost_x, ghost_y, pacman_x, pacman_y):
        """
        h(n): distancia Manhattan normalizada entre fantasma y pacman.
        Se invierte a negativo para que MAX favorezca distancia minima.
        """
        max_distancia = 18
        distancia = abs(ghost_x - pacman_x) + abs(ghost_y - pacman_y)

        return -float(distancia) / max_distancia

    def h_euclidiana(self, ghost_x, ghost_y, pacman_x, pacman_y):
        """
        h2(n): distancia euclidiana entre fantasma y pacman.
        Se invierte a negativo para que MAX favorezca distancia minima.
        """
        distancia = math.sqrt(((ghost_x - pacman_x) ** 2) + ((ghost_y - pacman_y) ** 2))
        return -distancia

    def h_euclidiana_normalizada(self, ghost_x, ghost_y, pacman_x, pacman_y):
        """
        h2(n): distancia euclidiana normalizada entre fantasma y pacman.
        Se invierte a negativo para que MAX favorezca distancia minima.
        """
        max_distancia = 12.727922061357855
        distancia = math.sqrt(((ghost_x - pacman_x) ** 2) + ((ghost_y - pacman_y) ** 2))
        return -float(distancia) / max_distancia

    def evaluar(self, ghost_x, ghost_y, pacman_x, pacman_y, movimientos_fantasma):
        """
        f(n) = peso_h1 * h_manhattan(n) + peso_h2 * h_euclidiana(n) + peso_g * g(n)
        """
        h1_n = self.distancia_manhattan_normalizada(ghost_x, ghost_y, pacman_x, pacman_y)
        h2_n = self.h_euclidiana_normalizada(ghost_x, ghost_y, pacman_x, pacman_y)
        return (
            (self.peso_h1 * h1_n) +
            (self.peso_h2 * h2_n)
        )

    def evaluar_nodo(self, nodo, movimientos_fantasma):
        """
        Atajo para evaluar un nodo con estructura:
        nodo["ghost"]["x"], nodo["ghost"]["y"], nodo["pacman"]["x"], nodo["pacman"]["y"].
        """
        ghost = nodo["ghost"]
        pacman = nodo["pacman"]
        return self.evaluar(
            ghost["x"],
            ghost["y"],
            pacman["x"],
            pacman["y"],
            movimientos_fantasma,
        )
