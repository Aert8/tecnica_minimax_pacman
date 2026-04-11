import math
from funcionHeuristica import FuncionHeuristica
from funcionHeuristicaManada import FuncionHeuristica as FuncionHeuristicaManada

class poda_alpha_beta:
    def __init__(self, arbol_completo, profundidad_maxima=None, heuristica=None):
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
        
        # Margen para la ventana de aspiración
        self.margen_ventana = 0.15

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

    def _es_terminal(self, nodo, limite_profundidad=None):
        if nodo is None:
            return True

        # Se utiliza el límite dinámico de la Profundidad Iterativa
        limite = limite_profundidad if limite_profundidad is not None else self.profundidad_maxima

        if limite is not None:
            if nodo.get("depth", 0) >= limite:
                return True

        # Condición de captura (Victoria)
        ghosts = nodo.get("ghosts", [])
        ghost = nodo.get("ghost", {})
        pacman = nodo.get("pacman", {})
        if len(ghosts) > 0:
            for ghost_state in ghosts:
                if ghost_state.get("x") == pacman.get("x") and ghost_state.get("y") == pacman.get("y"):
                    return True

        if ghost and pacman:
            if ghost.get("x") == pacman.get("x") and ghost.get("y") == pacman.get("y"):
                return True

        hijos = nodo.get("children", [])
        return len(hijos) == 0

    def _evaluar_terminal(self, nodo):
        profundidad = nodo.get("depth", 0)
        
        # Extraemos las entidades del nodo
        ghost0 = nodo.get("ghost0", {})
        ghost1 = nodo.get("ghost1", {})
        ghost_solo = nodo.get("ghost", {})
        pacman = nodo.get("pacman", {})
        
        # Verificamos captura para el modo Manada
        if ghost0 and pacman and ghost1:
            if (ghost0.get("x") == pacman.get("x") and ghost0.get("y") == pacman.get("y")) or \
               (ghost1.get("x") == pacman.get("x") and ghost1.get("y") == pacman.get("y")):
                # Se capturó al pacman. Ignoramos la heurística y damos la máxima recompensa.
                # Restamos la profundidad para que el fantasma prefiera atraparlo rápido.
                return 10000.0 - profundidad

        # ---------------------------------------------------------
        # MODO PERSECUCIÓN: Si no hubo captura, usamos la heurística
        # ---------------------------------------------------------
        # Esto solo se ejecuta si el árbol llegó a su max_depth sin atraparlo
        return float(self.heuristica.evaluar_nodo(nodo, profundidad))

    def _ab(self, nodo, alpha, beta, limite_profundidad=None):
        if self._es_terminal(nodo, limite_profundidad):
            valor = self._evaluar_terminal(nodo)
            nodo["ab_value"] = valor
            return valor

        hijos = nodo.get("children", [])

        if nodo.get("turn") == "MAX":
            valor_max = -math.inf
            for hijo in hijos:
                valor_evaluado = self._ab(hijo, alpha, beta, limite_profundidad)
                valor_max = max(valor_max, valor_evaluado)
                alpha = max(alpha, valor_max)

                if alpha >= beta:
                    self.podas += 1
                    break
            
            nodo["ab_value"] = valor_max
            return valor_max

        else: # Turno MIN
            valor_min = math.inf
            for hijo in hijos:
                valor_evaluado = self._ab(hijo, alpha, beta, limite_profundidad)
                valor_min = min(valor_min, valor_evaluado)
                beta = min(beta, valor_min)

                if alpha >= beta:
                    self.podas += 1
                    break

            nodo["ab_value"] = valor_min
            return valor_min

    def mejor_hijo_raiz(self):
        """
        Ejecuta la búsqueda desde la raíz utilizando Profundidad Iterativa 
        y Ventanas de Aspiración dinámicas.
        """
        if self.arbol is None:
            return None, 0.0

        hijos = self.arbol.get("children", [])
        if len(hijos) == 0:
            return None, self._ab(self.arbol, -math.inf, math.inf)

        mejor_hijo_global = None
        mejor_valor_global = -math.inf
        valor_esperado = 0.0 
        
        # Profundidad Iterativa: se evalúa nivel por nivel
        for profundidad_actual in range(1, self.profundidad_maxima + 1):
            
            # 1. Configurar la ventana de aspiración
            if profundidad_actual == 1:
                # En la primera iteración no hay información previa, ventana infinita
                alpha = -math.inf
                beta = math.inf
            else:
                # Ventana cerrada centrada en el resultado de la profundidad anterior
                alpha = valor_esperado - self.margen_ventana
                beta = valor_esperado + self.margen_ventana

            mejor_hijo = None
            mejor_valor = -math.inf

            # 2. Búsqueda con la ventana establecida
            for hijo in hijos:
                valor = self._ab(hijo, alpha, beta, limite_profundidad=profundidad_actual)
                
                if valor > mejor_valor:
                    mejor_valor = valor
                    mejor_hijo = hijo
                
                alpha = max(alpha, mejor_valor)

            # 3. Verificar si hubo Fail-Low o Fail-High (Re-búsqueda de emergencia)
            if mejor_valor <= (valor_esperado - self.margen_ventana) or mejor_valor >= (valor_esperado + self.margen_ventana):
                
                # La ventana falló, abrimos a infinito y volvemos a buscar en esta misma profundidad
                alpha = -math.inf
                beta = math.inf
                mejor_valor = -math.inf
                mejor_hijo = None
                
                for hijo in hijos:
                    valor = self._ab(hijo, alpha, beta, limite_profundidad=profundidad_actual)
                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_hijo = hijo
                    alpha = max(alpha, mejor_valor)

            # 4. Guardar resultados para la siguiente iteración
            valor_esperado = mejor_valor
            mejor_hijo_global = mejor_hijo
            mejor_valor_global = mejor_valor

        self.arbol["ab_value"] = mejor_valor_global
        return mejor_hijo_global, mejor_valor_global