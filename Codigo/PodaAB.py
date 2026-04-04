import math
from funcionHeuristica import FuncionHeuristica


class poda_alpha_beta:
    def __init__(self, arbol_completo, profundidad_maxima=None, heuristica=None):
        self.arbol = arbol_completo
        self.profundidad_maxima = (
            profundidad_maxima
            if profundidad_maxima is not None
            else self._inferir_profundidad_maxima(arbol_completo)
        )
        self.heuristica = (
            heuristica
            if heuristica is not None
            else FuncionHeuristica(max_depth=max(1, self.profundidad_maxima))
        )
        self.podas = 0

    def _inferir_profundidad_maxima(self, nodo):
        if nodo is None:
            return 0

        hijos = nodo.get("children", [])
        if len(hijos) == 0:
            return nodo.get("depth", 0)

        profundidad_hijos = [self._inferir_profundidad_maxima(hijo) for hijo in hijos]
        return max(profundidad_hijos)

    def _es_terminal(self, nodo):
        if nodo is None:
            return True

        if self.profundidad_maxima is not None:
            if nodo.get("depth", 0) >= self.profundidad_maxima:
                return True

        hijos = nodo.get("children", [])
        return len(hijos) == 0

    def _evaluar_terminal(self, nodo):
        movimientos_fantasma = nodo.get("depth", 0)
        return float(self.heuristica.evaluar_nodo(nodo, movimientos_fantasma))

    def _ab(self, nodo, alpha, beta):
        # 1) Si J es terminal, devolver h(j)
        if self._es_terminal(nodo):
            valor = self._evaluar_terminal(nodo)
            nodo["ab_value"] = valor
            return valor

        # 2) k <- 1 (se itera en orden sobre los hijos)
        hijos = nodo.get("children", [])

        # 3) Si J es MAX
        if nodo.get("turn") == "MAX":
            for hijo in hijos:
                # alpha <- max(alpha, ab(Jk, alpha, beta))
                alpha = max(alpha, self._ab(hijo, alpha, beta))

                # Si alpha >= beta, devolver beta (segun pseudocodigo adjunto)
                if alpha >= beta:
                    self.podas += 1
                    nodo["ab_value"] = beta
                    return beta

            # Si se evaluaron todos los hijos, devolver alpha
            nodo["ab_value"] = alpha
            return alpha

        # 4) Si J es MIN
        for hijo in hijos:
            # beta <- min(beta, ab(Jk, alpha, beta))
            beta = min(beta, self._ab(hijo, alpha, beta))

            # Si alpha >= beta, devolver alpha (segun pseudocodigo adjunto)
            if alpha >= beta:
                self.podas += 1
                nodo["ab_value"] = alpha
                return alpha

        # Si se evaluaron todos los hijos, devolver beta
        nodo["ab_value"] = beta
        return beta

    def ejecutar(self):
        # Llamada inicial: alpha = -inf, beta = +inf
        alpha = -math.inf
        beta = math.inf
        return self._ab(self.arbol, alpha, beta)

    def mejor_hijo_raiz(self):
        # Util para seleccionar movimiento del fantasma cuando la raiz es MAX.
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
