from ArcadioCv import * 
from matplotlib import pyplot as plt
from  Kmeans import *


def producto_cruz(p1,p2):
    a1,b1,c1 = p1
    a2,b2,c2 = p2

    x = b1*1 - b2*1 #c2,c1
    y = -(a1*1-a2*1)
    z = a1*b2 - a2*b1
    #normalizamos el vector
    x = x/z
    y = y/z
    z = z/z
    #retornamos el vector
    return (x,y,z)

def puntos_lineas(a,b,c,inicio=-100,final=100,saltos=10):
    x = np.array(range(inicio,final))/saltos
    if(inicio>final):
        x = np.array(range(final,inicio))/saltos
    #0 = ax +by +z
    y = (-c-a*x)/b
    #y = a*x +b*x + c 
    print(y)
    return (list(x),list(y))
    #tomamos 100 valores

def interseccion_lineas():
    print("Seleccione las cooordenadas de la linea 1")
    a1 = int(input("Ingrese a1: "))
    b1 = int(input("Ingrese b1: "))

    print("Seleccione las cooordenadas de la linea 2")
    a2 = int(input("Ingrese a2: "))
    b2 = int(input("Ingrese b2: "))

    l1 = (a1,b1,1)
    l2 = (a2,b2,1)

    x,y,z = producto_cruz(l1,l2)
    print(x,y,z)
    #ax+by+z=0, y = -z/
    
    x1,y1 = puntos_lineas(a1,b1,1)
    x2,y2 = puntos_lineas(a2,b2,1)
    #ploteamos el punto y las lineas
    plt.title("Intersección de dos rectas")
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.scatter(x,y,s=30,c="red")

    plt.show()


def linea_entre_2_puntos():
    print("Seleccione las cooordenadas del punto 1")
    a1 = int(input("Ingrese a1: "))
    b1 = int(input("Ingrese b1: "))

    print("Seleccione las cooordenadas del punto 2")
    a2 = int(input("Ingrese a2: "))
    b2 = int(input("Ingrese b2: "))

    m1 = (a1,b1,1)
    m2 = (a2,b2,1)

    x,y,z = producto_cruz(m1,m2)
    if(b2>b1):
        x,y,z = producto_cruz(m2,m1)
    print(x,y)

    x1,y1 = puntos_lineas(x,y,1,a1,a2+1,1)
    print(x1,y1)
    print("PUNTO 1",a1,b1)
    print("PUNTO 2",a2,b2)

    plt.title("Recta entre 2 puntos")
    plt.plot(x1,y1)
    plt.scatter(a1,b1,s=30,c="red")
    plt.scatter(a2,b2,s=30,c="red")

    plt.show()


    pass

def distancia_2_puntos(): 

    pass
    

def analisis_2D():
    print("Seleccione lo que quiere realizar")
    print("1.-Punto de intersección")
    print("2.-Línea entre 2 puntos")
    print("otro.-Distancia 2 puntos")
    seleccion = input()
    if(seleccion == "1"):
        interseccion_lineas()
    elif(seleccion == "2"):
        linea_entre_2_puntos()
    else: 
        distancia_2_puntos()
    print("Desea repetir la selección?")
    print("1.-Si")
    print("Otro.-No")
    repetir = input()
    if(repetir=="1" or repetir.lower()=="si"):
        analisis_2D()

def crear_imagenes(coordenadas,imagen):
    rows,cols,colors = imagen.shape
    imagenes = []
    for k in coordenadas:
        imagen = np.zeros((rows,cols),np.uint8)
        indices = k.get('indices')
        for indice in indices:
            i = indice[0]
            j = indice[1]
            if(indice == (i,j)):
                #print("xdxd")
                #guardamos el valor
                imagen[i,j] = 255
        imagenes.append(imagen)
    for imagen in imagenes:
        #grises = ArcadioCv.rgb_to_grises(imagen)
        imagen_cruces, imagen_delta,imagen_final,imagen_log = ArcadioCv.filtro_log(imagen,5,1)
        ArcadioCv.visualizar_imagen(imagen_log)






def k_means():
    K = int(input("Ingresa el numero de k: "))
    #cielo.webp
    imagen_agrupada,coordenadas = Kmeans.k_means("cielo.webp",K)
    
    print(imagen_agrupada)

    ArcadioCv.visualizar_imagen(imagen_agrupada)
    
    crear_imagenes(coordenadas,imagen_agrupada)


    """ grises = ArcadioCv.rgb_to_grises(imagen_agrupada)
    
    imagen_cruces, imagen_delta,imagen_log = ArcadioCv.filtro_log(grises,5,1)


    ArcadioCv.visualizar_imagen(imagen_log) """


if __name__ == "__main__": 
    print("Bienvenido al programa de segmentación a color, seleccione que quiere hacer")
    print("1.-K means")
    print("otro.-Análisis 2D")
    seleccion = input()
    if(seleccion == "1"):
        k_means()
    else:
        analisis_2D()



    