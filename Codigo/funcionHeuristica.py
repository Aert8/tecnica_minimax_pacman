
class FuncionHeuristica:
    def __init__(self, max_depth, peso_h=0.7, peso_g=0.3):
        self.peso_h = float(peso_h)
        self.peso_g = float(peso_g)
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

    def movimientos_fantasma(self, movimientos_fantasma):
        """
        g(n): numero de movimientos del fantasma.
        Se invierte a negativo para que MAX favorezca menos movimientos.
        """
        return -float((movimientos_fantasma//2)+1)
    
    def movimientos_fantasma_normalizados(self, movimientos_fantasma):
        """
        g(n): numero de movimientos del fantasma normalizado.
        Se invierte a negativo para que MAX favorezca menos movimientos.
        """
        return -float((movimientos_fantasma//2)+1) / self.max_depth

    def evaluar(self, ghost_x, ghost_y, pacman_x, pacman_y, movimientos_fantasma):
        """
        f(n) = peso_h * h(n) + peso_g * g(n)
        """
        h_n = self.distancia_manhattan_normalizada(ghost_x, ghost_y, pacman_x, pacman_y)
        g_n = self.movimientos_fantasma_normalizados(movimientos_fantasma)
        return (self.peso_h * h_n) + (self.peso_g * g_n)

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
