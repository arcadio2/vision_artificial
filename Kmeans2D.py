from ArcadioCv import * 
import random
import numpy as np 
import math 

class Kmeans2D:

    def distancia_euclidiana(d1,d2):
        x0=d1[0]
        x1=d2[0]
        y0=d1[1]
        y1=d2[1]
        return ((x1-x0)**2+(y1-y0)**2)**0.5


    def k_aleatorios(k,puntos):
        lista = []
        for i in range(k): 
            #en los rangos de la imagen
            x = random.choice(puntos)
            y = random.choice(puntos)
            #coords = (x,y)
            lista.append(x)
        return lista
    
    def obtener_k(vectores):
        x=[]
        y=[]
        
        for vector in vectores:
            x.append(vector[0])
            y.append(vector[1])
        mean_x = np.mean(x)
        mean_y= np.mean(y)
        return (mean_x,mean_y)

    def redefinir_k(agrupaciones): 
        new_k_points = []
        for group in agrupaciones: 
            valores = group.get('indices')
            new_point = Kmeans2D.obtener_k(valores)
            if(math.isnan(new_point[0]) ):
                print("xd")
                new_point = group.get('k_point')

            new_k_points.append(new_point)
            
        return new_k_points
    
    def agrupamiento(k_points,x,y): 
        k_agrupados = []
        indices = [[] for i in range(len(k_points))]
        for i in range(len(x)): 
            distancias = []
            for k in k_points: 
                distancia = Kmeans2D.distancia_euclidiana((x[i],y[i]),k)
                distancias.append(distancia)
            distancias = np.array(distancias)
            indice = np.where(distancias == np.amin(distancias))[0][0] #indice de distancia minima
            distancia_minima = distancias[indice] #encontramos la distancia minima
            
            indices[indice].append((x[i],y[i]))
        
        #print(indices[1])
        for i in range(len(k_points)):
            k_agrupados.append(
                {
                    'k_point':k_points[i],
                    "indices":indices[i],
                }
            )
        return k_agrupados
        
    def obtener_blancos(imagen):
        x = []
        y = []
        puntos = []
        rows,cols = imagen.shape
        for i in range(rows):
            for j in range(cols):
                if(imagen[i,j]==255):
                    x.append(i)
                    y.append(j)
                    puntos.append( (i,j) )
        return x,y,puntos 
    
    def k_means_2d(imagen,K=2):
        rows,cols = imagen.shape

        x,y,puntos = Kmeans2D.obtener_blancos(imagen)
        #print(puntos)
        k_points = Kmeans2D.k_aleatorios(K,puntos)
        #k_points = [(0,0),(0,cols-1),(rows,cols),(rows,0)]
        
        k_anterior = []
        agrupaciones = Kmeans2D.agrupamiento(k_points,x,y)

        
        print("K1",k_points)
        while(k_points != k_anterior):
            k_anterior = k_points
            k_points = Kmeans2D.redefinir_k(agrupaciones)
            print(k_points)
            agrupaciones = Kmeans2D.agrupamiento(k_points,x,y)
       
        Kmeans2D.separar_imagenes(agrupaciones,imagen)
    
    def separar_imagenes(agrupaciones,imagen):
        rows,cols = imagen.shape
        imagenes = []
        for group in agrupaciones:
            imagen = np.zeros((rows,cols),np.uint8)
            indices = group.get('indices')
            
            for indice in indices:
                i = indice[0]
                j = indice[1]
                if(indice == (i,j)):
                    #print("xdxd")
                    #guardamos el valor
                    imagen[i,j] = 255
            imagenes.append(imagen)

        for imagen in imagenes:
            ArcadioCv.visualizar_imagen(imagen)



if __name__ == "__main__":
    grises = ArcadioCv.abrir_imagen_grises('cosas.webp')
    imagen_cruces, imagen_delta,imagen_final,imagen_log = ArcadioCv.filtro_log(grises,5,1)
    ArcadioCv.visualizar_imagen(imagen_final,"cruces")
    print(imagen_final)
    random.seed(1)
    #umbral = ArcadioCv.umbral_otsu(imagen_cruces)
    #print(umbral)
    #imagen_delta = ArcadioCv.umbralar(imagen_delta,2)
    #ArcadioCv.visualizar_imagen(imagen_delta,"u")

    #ArcadioCv.visualizar_imagen(imagen_cruces)
    #ArcadioCv.visualizar_imagen(imagen_delta)
    #ArcadioCv.visualizar_imagen(imagen_log)

    Kmeans2D.k_means_2d(imagen_final,7)