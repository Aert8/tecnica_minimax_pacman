import pygame
from pygame.locals import *

# Cargamos las bibliotecas de OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import math
import os
import numpy as np
import pandas as pd
import random
from PodaAB import poda_alpha_beta

class Ghost:
    def __init__(self,mapa, mc, x_mc, y_mc, xini, yini, dir, tipo):
        #Matriz de control que almacena los IDs de las intersecciones
        self.MC = mc
        #Vectores que almacenan las coordenadas 
        self.XPxToMC = x_mc
        self.YPxToMC = y_mc
        #se resplanda el mapa en terminos de pixeles
        self.mapa = mapa
        #se inicializa la posicion del fantasma en terminos de pixeles
        self.position = []
        self.position.append(xini)
        self.position.append(1) #YPos
        self.position.append(yini)
        #se define el arreglo para la posicion en la matriz de control
        self.positionMC = []
        self.positionMC.append(self.XPxToMC[self.position[0] - 20]) #coord en x
        self.positionMC.append(self.YPxToMC[self.position[2] - 20]) #coord en y
        #se inicializa una direccion valida
        self.direction = dir
        #se almacena que tipo de fantasma sera:
        #0: fantasma aleatorio
        #1: fantasma con pathfinding
        self.tipo = tipo
        #arreglo para almacenar las opciones del fantasma
        self.options = [
            [1,2],
            [2,3],
            [0,1],
            [0,3],
            [1,2,3],
            [0,2,3],
            [0,1,3],
            [0,1,2],
            [0,1,2,3],
            [1],
            [3]
        ]
        self.option = []
        self.dir_inv = 0
        self.path_n = 0
        self.state_tree = None
        self.prev_pacman_xy = None
        self.last_ab_value = None
        self.last_ab_prunes = 0

    def loadTextures(self, texturas, id):
        self.texturas = texturas
        self.Id = id

    def drawFace(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(x1, y1, z1)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(x2, y2, z2)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(x3, y3, z3)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(x4, y4, z4)
        glEnd()

    def _inverse_direction(self, direction):
        return (direction + 2) % 4

    def _is_inside_mc(self, x, y):
        return 0 <= y < len(self.MC) and 0 <= x < len(self.MC[0])

    def _direction_to_str(self, direction):
        labels = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        return labels.get(direction, "UNKNOWN")

    def _estimate_pacman_direction(self, pacmanXY):
        if self.prev_pacman_xy is None:
            self.prev_pacman_xy = [pacmanXY[0], pacmanXY[2]]
            return None

        dx = pacmanXY[0] - self.prev_pacman_xy[0]
        dz = pacmanXY[2] - self.prev_pacman_xy[1]
        self.prev_pacman_xy = [pacmanXY[0], pacmanXY[2]]

        if abs(dx) > abs(dz):
            if dx > 0:
                return 1
            if dx < 0:
                return 3
        else:
            if dz > 0:
                return 2
            if dz < 0:
                return 0

        return None

    def _pixel_to_mc(self, px, pz):
        x_idx = px - 20
        y_idx = pz - 20
        if ((x_idx < 0) or (x_idx >= len(self.XPxToMC)) or
            (y_idx < 0) or (y_idx >= len(self.YPxToMC))):
            return None

        x_mc = self.XPxToMC[x_idx]
        y_mc = self.YPxToMC[y_idx]
        if (x_mc == -1) or (y_mc == -1):
            return None

        return [x_mc, y_mc]

    def _project_pixel_to_mc(self, px, pz, direction):
        mapped = self._pixel_to_mc(px, pz)
        if mapped is not None:
            return mapped

        deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        if direction in deltas:
            directions = [direction] + [d for d in [0, 1, 2, 3] if d != direction]
        else:
            directions = [0, 1, 2, 3]

        for dir_candidate in directions:
            dx, dz = deltas[dir_candidate]
            test_x = px
            test_z = pz

            for _ in range(max(len(self.XPxToMC), len(self.YPxToMC))):
                test_x += dx
                test_z += dz
                mapped = self._pixel_to_mc(test_x, test_z)
                if mapped is not None:
                    return mapped

        return None

    def _get_cell_options(self, cell_id, current_dir):
        options_by_cell = {
            10: [1, 2],
            11: [2, 3],
            12: [0, 1],
            13: [0, 3],
            21: [1, 2, 3],
            22: [0, 2, 3],
            23: [0, 1, 3],
            24: [0, 1, 2],
            25: [0, 1, 2, 3],
            26: [1],
            27: [3]
        }

        if cell_id == 0:
            return [] if current_dir is None else [current_dir]

        return list(options_by_cell.get(cell_id, []))

    def _get_available_directions(self, mc_x, mc_y, current_dir, avoid_immediate_reverse):
        cell_id = self.MC[mc_y][mc_x]
        dirs = self._get_cell_options(cell_id, current_dir)

        if avoid_immediate_reverse and len(dirs) > 1 and current_dir is not None:
            inv_dir = self._inverse_direction(current_dir)
            if inv_dir in dirs:
                dirs.remove(inv_dir)

        return dirs

    def _advance_to_next_true_intersection(self, start_x, start_y, direction):
        deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        if direction not in deltas:
            return None

        dx, dy = deltas[direction]
        x = start_x
        y = start_y
        steps = 0

        while True:
            x += dx
            y += dy
            steps += 1

            if not self._is_inside_mc(x, y):
                return None

            if self.MC[y][x] != 0:
                return [x, y, steps]

    def _expand_state_tree(self, node, max_depth):
        if node["depth"] >= max_depth:
            return
        # ---------------------------------------------------------
        # NUEVO: Condición de paro por captura (Estado Terminal)
        # Si el fantasma y pacman están en la misma coordenada, 
        # el juego terminó. Ya no generamos más hijos.
        # ---------------------------------------------------------
        if node["ghost"]["x"] == node["pacman"]["x"] and node["ghost"]["y"] == node["pacman"]["y"]:
            return

        # Se fuerza que el turno actual siempre sea MAX o MIN.
        # Cualquier valor inesperado se interpreta como MAX.
        current_turn = "MIN" if node.get("turn") == "MIN" else "MAX"
        next_turn = "MIN" if current_turn == "MAX" else "MAX"

        if current_turn == "MAX":
            actor = node["ghost"]
            avoid_reverse = True
        else:
            actor = node["pacman"]
            avoid_reverse = False

        available_dirs = self._get_available_directions(
            actor["x"], actor["y"], actor["dir"], avoid_reverse
        )

        for direction in available_dirs:
            next_state = self._advance_to_next_true_intersection(
                actor["x"], actor["y"], direction
            )

            if next_state is None:
                continue

            next_x, next_y, steps = next_state

            child_ghost = dict(node["ghost"])
            child_pacman = dict(node["pacman"])

            if current_turn == "MAX":
                child_ghost["x"] = next_x
                child_ghost["y"] = next_y
                child_ghost["dir"] = direction
            else:
                child_pacman["x"] = next_x
                child_pacman["y"] = next_y
                child_pacman["dir"] = direction

            child_node = {
                "depth": node["depth"] + 1,
                "turn": next_turn,
                "actor": current_turn,
                "move_dir": direction,
                "move_label": self._direction_to_str(direction),
                "travel_steps": steps,
                "ghost": child_ghost,
                "pacman": child_pacman,
                "children": []
            }

            node["children"].append(child_node)
            self._expand_state_tree(child_node, max_depth)

    def generar_arbol_estados(self, pacmanXY, max_depth=6, pacman_dir=None):
        ghost_mc = self._pixel_to_mc(self.position[0], self.position[2])
        if ghost_mc is None:
            return None

        if pacman_dir is None:
            pacman_dir = self._estimate_pacman_direction(pacmanXY)

        pacman_mc = self._project_pixel_to_mc(pacmanXY[0], pacmanXY[2], pacman_dir)
        if pacman_mc is None:
            return None

        # La raiz siempre inicia con MAX: el fantasma mueve primero.
        root_turn = "MAX"

        root = {
            "depth": 0,
            "turn": root_turn,
            "actor": None,
            "move_dir": None,
            "move_label": None,
            "travel_steps": 0,
            "ghost": {
                "x": ghost_mc[0],
                "y": ghost_mc[1],
                "dir": self.direction
            },
            "pacman": {
                "x": pacman_mc[0],
                "y": pacman_mc[1],
                "dir": pacman_dir
            },
            "children": []
        }

        self._expand_state_tree(root, max_depth)
        return root
   
    def sigue_adelante(self):
        #si el fantasma esta en un tunel, no es necesario calcular la siguiente posicion a traves del path
        #solo se sigue la direccion actual y se aumenta el contador que accede a la posicion del path actual
        if self.direction == 0: #up
            self.position[2] -= 1
        elif self.direction == 1: #right
            self.position[0] += 1
        elif self.direction == 2: #down
            self.position[2] += 1
        else: #left
            self.position[0] -= 1
        #se actualiza la variable de posicion sobre el path
        if self.tipo == 1: #fantasma inteligente
            self.path_n += 1
        
    def path_ia(self,pacmanXY, pacmanDir=None):
        # bloque para implementar la IA en los fantasmas
        self.state_tree = self.generar_arbol_estados(pacmanXY, pacman_dir=pacmanDir)

        if self.state_tree is None:
            #print("No se pudo generar el árbol de estados. Se elige movimiento aleatorio.")
            self.interseccion_random()
            return

        if len(self.state_tree.get("children", [])) == 0:
            #print("El árbol de estados no tiene hijos. Se elige movimiento aleatorio.")
            self.interseccion_random()
            return

        motor_ab = poda_alpha_beta(self.state_tree)
        mejor_hijo, mejor_valor = motor_ab.mejor_hijo_raiz()
        self.last_ab_value = mejor_valor
        self.last_ab_prunes = motor_ab.podas

        if (mejor_hijo is None) or (mejor_hijo.get("move_dir") is None):
            print("No se pudo determinar el mejor movimiento. Se elige movimiento aleatorio.")
            self.interseccion_random()
            return

        # Solo se aplica la direccion elegida; el movimiento fisico se mantiene pixel a pixel.
        self.direction = mejor_hijo["move_dir"]

        if self.direction == 0:
            self.position[2] -= 1
        elif self.direction == 1:
            self.position[0] += 1
        elif self.direction == 2:
            self.position[2] += 1
        elif self.direction == 3:
            self.position[0] -= 1
        
    def interseccion_random(self):
        #se determina en que tipo de celda esta el fantasma
        self.positionMC[0] = self.XPxToMC[self.position[0] - 20]
        self.positionMC[1] = self.YPxToMC[self.position[2] - 20]
        celId = self.MC[self.positionMC[1]][self.positionMC[0]]
        #a partir de la celda actual se generan sus opciones posibles
        if celId == 0:
            self.option = [self.direction]
        elif celId == 10: #options = [1, 2]
            self.option = self.options[0]
        elif celId == 11: #options = [2, 3]
            self.option = self.options[1]
        elif celId == 12: #options = [0, 1]
            self.option = self.options[2]
        elif celId == 13: #options = [0, 3]
            self.option = self.options[3]
        elif celId == 21: #options = [1, 2, 3]
            self.option = self.options[4]
        elif celId == 22: #options = [0, 2, 3]
            self.option = self.options[5]
        elif celId == 23: #options = [0, 1, 3]
            self.option = self.options[6]
        elif celId == 24: #options = [0, 1, 2]
            self.option = self.options[7]
        elif celId == 25: #options = [0, 1, 2, 3]
            self.option = self.options[8]
        elif celId == 26: #options = [1]
            self.option = self.options[9]
        elif celId == 27: #options = [3]
            self.option = self.options[10]
        
        #se calcula la direccion inversa a la actual
        if self.direction == 0:
            self.dir_inv = 2
        elif self.direction == 1:
            self.dir_inv = 3
        elif self.direction == 2:
            self.dir_inv = 0
        else:
            self.dir_inv = 1

        #se elimina la direccion invertida a la actual, evitando que el
        #fantasma regrese por el camion por donde llego (rebote)
        if (celId != 0) and (celId != 26) and (celId != 27):
            self.option.remove(self.dir_inv)
        
        #se elige aleatoriamente una opcion entre las disponibles
        size = len(self.option)
        dir_rand = random.randint(0, size - 1)
        
        #se actualiza el vector de direccion y posicion del fantasma
        self.direction = self.option[dir_rand]
        
        if self.direction == 0:
            self.position[2] -= 1
        elif self.direction == 1:
            self.position[0] += 1
        elif self.direction == 2:
            self.position[2] += 1
        elif self.direction == 3:
            self.position[0] -= 1
            
        if (celId != 0) and (celId != 26) and (celId != 27):
            self.option.append(self.dir_inv)    
    
    def update2(self,pacmanXY, pacmanDir=None):
        #Compobamos que el fantasma atrapo a pacman
        #if (self.position[0] == pacmanXY[0]) and (self.position[2] == pacmanXY[2]):
            #print("Pacman atrapado por el fantasma!")

        #si el fantasma se encuentra en una interseccion (valida o "falsa interseccion")
        if ((self.YPxToMC[self.position[2] - 20] != -1) and 
            (self.XPxToMC[self.position[0] - 20] != -1)):
            if self.tipo == 1: #agente inteligente, se manda la posición del objetivo
                self.path_ia(pacmanXY, pacmanDir)
            else:
                self.interseccion_random()
        else: #si no se encuentra en una interseccion o es falsa interseccion
            self.sigue_adelante()
        
    def draw(self):
        glPushMatrix()
        glColor3f(1.0, 1.0, 1.0)
        glTranslatef(self.position[0], self.position[1], self.position[2])
        glScaled(10,1,10)
        #Activate textures
        glEnable(GL_TEXTURE_2D)
        #front face
        glBindTexture(GL_TEXTURE_2D, self.texturas[self.Id])
        self.drawFace(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0)    
        glDisable(GL_TEXTURE_2D)  
        glPopMatrix()        