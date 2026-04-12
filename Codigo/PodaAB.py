import math
from funcionHeuristica import FuncionHeuristica
from funcionHeuristicaManada import FuncionHeuristica as FuncionHeuristicaManada

class poda_alpha_beta:
    def __init__(self, arbol_completo, profundidad_maxima=None, heuristica=None, generador=None):
        self.arbol = arbol_completo
        self.modo_manada = self._es_arbol_manada(arbol_completo)
        self.profundidad_maxima = (
            profundidad_maxima
            if profundidad_maxima is not None
            else self._inferir_profundidad_maxima(arbol_completo)
        )
        self.heuristica = (
            heuristica
            if heuristica is not None
            else self._crear_heuristica_por_modo()
        )
        self.podas = 0
        
        # Referencia al agente para expandir ramas en tiempo real
        self.generador = generador
        # Límite para las estrategias adicionales
        self.limite_absoluto = self.profundidad_maxima + 4

    def _es_arbol_manada(self, nodo):
        if nodo is None:
            return False
        if "ghosts" in nodo:
            return True
        hijos = nodo.get("children", [])
        for hijo in hijos:
            if self._es_arbol_manada(hijo):
                return True
        return False

    def _crear_heuristica_por_modo(self):
        heuristica_cls = FuncionHeuristicaManada if self.modo_manada else FuncionHeuristica
        return heuristica_cls(max_depth=max(1, self.profundidad_maxima))

    def _inferir_profundidad_maxima(self, nodo):
        if nodo is None:
            return 0
        hijos = nodo.get("children", [])
        if len(hijos) == 0:
            return nodo.get("depth", 0)
        profundidad_hijos = [self._inferir_profundidad_maxima(hijo) for hijo in hijos]
        return max(profundidad_hijos)

    def _distancia_minima(self, nodo):
        """Calcula la distancia al fantasma más cercano en el nodo actual"""
        pacman = nodo.get("pacman", {})
        ghosts = nodo.get("ghosts", [])
        ghost = nodo.get("ghost", {})
        if len(ghosts) > 0:
            return min(abs(g.get("x",0) - pacman.get("x",0)) + abs(g.get("y",0) - pacman.get("y",0)) for g in ghosts)
        elif ghost:
            return abs(ghost.get("x",0) - pacman.get("x",0)) + abs(ghost.get("y",0) - pacman.get("y",0))
        return 100

    def _es_terminal(self, nodo):
        if nodo is None:
            return True

        #ESTRICTAMENTE AMBOS
        ghosts = nodo.get("ghosts", [])
        ghost = nodo.get("ghost", {})
        pacman = nodo.get("pacman", {})
        
        # Modo Manada
        if len(ghosts) > 0:
            # all() devuelve True SOLO si todos los elementos de la lista cumplen la condición
            if all(g.get("x") == pacman.get("x") and g.get("y") == pacman.get("y") for g in ghosts):
                return True
                
        # Modo Fantasma Solitario (se mantiene igual)
        elif ghost and pacman:
            if ghost.get("x") == pacman.get("x") and ghost.get("y") == pacman.get("y"):
                return True

        profundidad = nodo.get("depth", 0)
        hijos = nodo.get("children", [])

        #ESTRATEGIAS
        if profundidad >= self.profundidad_maxima:
            if self.generador and profundidad < self.limite_absoluto:
                
                distancia = self._distancia_minima(nodo)
                cercania_critica = distancia <= 3
                
                if len(hijos) == 0:
                    if self.modo_manada:
                        self.generador._expand_state_tree_manada(nodo, profundidad + 1)
                    else:
                        self.generador._expand_state_tree(nodo, profundidad + 1)
                    hijos = nodo.get("children", [])

                movimiento_forzado = len(hijos) == 1
                
                if (cercania_critica or movimiento_forzado) and len(hijos) > 0:
                    return False
            
            return True

        return len(hijos) == 0

    def _evaluar_terminal(self, nodo):
        profundidad = nodo.get("depth", 0)
        pacman = nodo.get("pacman", {})
        
        #ESTRICTAMENTE AMBOS:
        ghosts = nodo.get("ghosts", [])
        if len(ghosts) > 0:
            # all() exige que todos los fantasmas cumplan la condición
            if all(g.get("x") == pacman.get("x") and g.get("y") == pacman.get("y") for g in ghosts):
                return 10000.0 - profundidad
                    
        ghost = nodo.get("ghost", {})
        if ghost:
            if ghost.get("x") == pacman.get("x") and ghost.get("y") == pacman.get("y"):
                return 10000.0 - profundidad

        # MODO PERSECUCIÓN
        return float(self.heuristica.evaluar_nodo(nodo, profundidad))

    def _ab(self, nodo, alpha, beta):
        if self._es_terminal(nodo):
            valor = self._evaluar_terminal(nodo)
            nodo["ab_value"] = valor
            return valor

        hijos = nodo.get("children", [])

        if nodo.get("turn") == "MAX":
            for hijo in hijos:
                alpha = max(alpha, self._ab(hijo, alpha, beta))
                if alpha >= beta:
                    self.podas += 1
                    nodo["ab_value"] = beta
                    return beta
            nodo["ab_value"] = alpha
            return alpha

        for hijo in hijos:
            beta = min(beta, self._ab(hijo, alpha, beta))
            if alpha >= beta:
                self.podas += 1
                nodo["ab_value"] = alpha
                return alpha

        nodo["ab_value"] = beta
        return beta

    def ejecutar(self):
        alpha = -math.inf
        beta = math.inf
        return self._ab(self.arbol, alpha, beta)

    def mejor_hijo_raiz(self):
        if self.arbol is None:
            return None, 0.0

        hijos = self.arbol.get("children", [])
        if len(hijos) == 0:
            return None, self.ejecutar()

        alpha = -math.inf
        beta = math.inf
        mejor_hijo = None
        mejor_valor = -math.inf

        for hijo in hijos:
            valor = self._ab(hijo, alpha, beta)
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_hijo = hijo

            alpha = max(alpha, mejor_valor)
            if alpha >= beta:
                self.podas += 1
                break

        self.arbol["ab_value"] = mejor_valor
        return mejor_hijo, mejor_valor