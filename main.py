from ArcadioCv import *
#import tkinter
from matplotlib import pyplot as plt


def watershed(imagen):
    tam_kernel = int(input("Ingresa el tamaño del kernel: "))
    sigma = float(input("Ingresa sigma: "))
    imagen_gauss = ArcadioCv.filtro_gauss(imagen,tam_kernel,sigma)
 
    umbral = ArcadioCv.umbral_otsu(imagen_gauss)
    print("Umbral",int(umbral))
    umbralada = ArcadioCv.umbralar(imagen_gauss,umbral)


    ArcadioCv.mi_watershed(umbralada,imagen,imagen_gauss,umbralada)

def LoG(imagen):
    """PEDIR DATOS"""
    tam_kernel = int(input("Ingresa el tamaño del kernel: "))
    sigma = float(input("Ingresa sigma: "))


    imagen_cruces, imagen_delta,imagen_log = ArcadioCv.filtro_log(grises,tam_kernel,sigma)

    fig,ax = plt.subplots(1,3)
    plt.title("Segmentación LoG")
    ax[0].set_title("Cruces por 0")
    ax[0].imshow(imagen_cruces, cmap=plt.cm.gray)

    ax[1].set_title("Imagen con delta ")
    ax[1].imshow(imagen_delta, cmap=plt.cm.gray)

    ax[2].set_title("And 0 and Delta")
    ax[2].imshow(imagen_log, cmap=plt.cm.gray)

    plt.show()

    """   ArcadioCv.visualizar_imagen(imagen_cruces,"Cruces por 0")
    ArcadioCv.visualizar_imagen(imagen_delta,"Delta")
    ArcadioCv.visualizar_imagen(imagen_log) """
    print(imagen_log)


def main(imagen):
    print("Binvenido al programa de segmentación, ¿Qué desea hacer?")
    print("1.-Segmentación por Watershed")
    print("otro.-Segmentación por LoG")
    opcion = input()
    if(opcion == "1"):
        watershed(grises)
    else:
        LoG(grises)

    print("¿Desea repetir el programa?")
    print("1.-Si")
    print("Otro.-No")
    repetir = input()
    if(repetir.lower()=="si" or repetir.lower()=="sí"):
        main(imagen)

if __name__ == "__main__":
    imagen = ArcadioCv.abrir_imagen_rgb("lena.png")
    grises = ArcadioCv.rgb_to_grises(imagen)
    #ArcadioCv.visualizar_imagen(grises)
    #umbral = ArcadioCv.umbral_otsu(grises)
    #print("Umbral",umbral)
    main(imagen)
   

    

    