
from ArcadioCv import * 
from matplotlib import pyplot as plt
import numpy as np
import random
import math

class Kmeans:
    def __init__(self,ruta):
        self.ruta = ruta

    def distancia_euclidiana(d1,d2):
        x0 = d1[0]
        y0 = d1[1]
        z0 = d1[2]

        x1 = d2[0]
        y1 = d2[1]
        z1 =d2[2]

        return ((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)**0.5

    def k_aleatorios(k):
        lista = []
        for i in range(k): 
            x = int(random.uniform(0, 255))
            y = int(random.uniform(0, 255))
            z = int(random.uniform(0, 255))
            coords = (x,y,z)
            lista.append(coords)
        return lista

    def obtener_k(vectores):
        x = []
        y = []
        z = []
        for vector in vectores:
            x.append(vector[0])
            y.append(vector[1])
            z.append(vector[2])
        mean_x = float(np.mean(x))
        mean_y= float(np.mean(y))
        mean_z = float(np.mean(z))
        #print(mean_x)
        return (mean_x,mean_y,mean_z)

    """ def obtener_k(matriz):
        lista = []
        for vector in matriz: 
            punto = k_definidos(vector)
            lista.append(punto)
            
        return lista """

    def redefinir_k(matriz_distancias):
        new_k_points = []
        for k in matriz_distancias:
            valores = k.get('puntos')
            new_point = Kmeans.obtener_k(valores)
            #print(type(new_point[0]))
            if(math.isnan(new_point[0]) ):
                new_point = k.get('k_point')
                #print("XD",new_point)
                """ print(new_point)
                new_point = k_aleatorios(1)[0]
                print("XD",new_point) """
            new_k_points.append(new_point)
        return new_k_points



    """Función que agrupa nuestros puntos al k mas cercano"""
    def agrupamiento(imagen,k_points): 
        rows,cols,dim = imagen.shape
        puntos_agrupados = [[] for k in range(len(k_points))] #crea listas vacías
        indices = [[] for k in range(len(k_points))]
        valores_distancias = [[] for k in range(len(k_points))]
        matriz_distancias = []
        """ for k in range(k_points):
            puntos_agrupados[k] = {
                k_points[k]:[]
            } """

        for i in range(rows): 
            for j in range(cols):
                punto = imagen[i][j] #punto en r,g,b
                rojo = punto[0]
                verde = punto[1]
                azul = punto[2]
                #encontramos la distancia del punto a todos los k
                distancias = []
                for k in k_points: 
                    distancia = Kmeans.distancia_euclidiana(punto,k)
                    distancias.append(distancia)
                #encontramos el indice minimo
                distancias = np.array(distancias)
                indice = np.where(distancias == np.amin(distancias))[0][0] #indice de distancia minima
                distancia_minima = distancias[indice] #encontramos la distancia minima
                #k_minimo = k_points[indice]

                #Guarsamos diccionarios
                indices[indice].append((i,j)) #guardamos el punto en coordenadas
                puntos_agrupados[indice].append((rojo,verde,azul))
                valores_distancias[indice].append(distancia_minima) #guardamos la distancia a ese k

        #guardamos todo en una matriz con los valores


        for k in range(len(k_points)):
            matriz_distancias.append(
                {
                    "k_point":k_points[k],
                    "puntos":puntos_agrupados[k],
                    "indices":indices[k],
                    "distancias":valores_distancias[k]
                }
            )
        #print(matriz_distancias[0])
        #print(len(matriz_distancias[0].get('puntos'))+len(matriz_distancias[1].get('puntos')))
        return matriz_distancias

        #print(len(matriz_distancias[0].get('puntos'))+len(matriz_distancias[1].get('puntos')))
        #print(len(matriz_distancias[0].get('distancias'))+len(matriz_distancias[1].get('distancias')))
        
    def crear_imagen_cluster(imagen,matriz_distancias):
        rows,cols,colors = imagen.shape
        #ecnontramos el promedio de los colores
        imagen_agrupada = np.zeros((rows,cols,colors),np.uint8)
        colores = []
        for k in matriz_distancias: 
            color = np.array(k.get('puntos'))
            if(len(color)>0):
                rojo = np.mean(color[:,0]) #todas las filas columna 0
                verde = np.mean(color[:,1])
                azul = np.mean(color[:,2])
                colores.append((rojo,verde,azul))
            


        for k,color in zip(matriz_distancias,colores): 
            indices = k.get('indices')
            for indice in indices:
                i = indice[0]
                j = indice[1]
                if(indice == (i,j)):
                    #print("xdxd")
                    #guardamos el valor
                    imagen_agrupada[i,j] = np.array(color,np.uint8)

        imagen_agrupada = cv2.cvtColor(imagen_agrupada,cv2.COLOR_RGB2BGR)

        return imagen_agrupada

    def k_means(url_imagen,K=2):
        imagen = ArcadioCv.abrir_imagen_rgb(url_imagen)
        k_points = Kmeans.k_aleatorios(K)
        k_anterior = []
        print(k_points)
        #
        matriz_distancias = Kmeans.agrupamiento(imagen,k_points)
        while(k_points != k_anterior):
            k_anterior = k_points
            k_points = Kmeans.redefinir_k(matriz_distancias)
            print(k_points)
            matriz_distancias = Kmeans.agrupamiento(imagen,k_points)
            

        imagen_final = Kmeans.crear_imagen_cluster(imagen,matriz_distancias)

        return imagen_final,matriz_distancias