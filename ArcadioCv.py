import numpy as np
import cv2 
import math
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

class ArcadioCv:
    def __init__(self,ruta):
        self.ruta = ruta
    
    def abrir_imagen_rgb(ruta):
        imagen = cv2.imread(ruta,1) 
        imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)

        return imagen 
    
    def abrir_imagen_grises(ruta):
        imagen = cv2.imread(ruta,0)
        return imagen  
        
        #cv2.COLOR_RGB2BGR. 

    def rgb_to_grises(imagen):
        width = imagen.shape[0]
        height = imagen.shape[1]
        print(imagen.shape)
        #np.zeros((100,100,3),np.uint8) 
        grises = np.zeros((width,height))
        #rojo 
        rojo = imagen[:,:,0]
        verde = imagen[:,:,1]
        azul = imagen[:,:,2]
        grises = np.array(rojo*0.299 + 0.587*verde + 0.114*azul,np.uint8)

        return grises

        
    def visualizar_imagen(imagen,text="imagen"):
        cv2.imshow(text,imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def kernel_logc(tam_kernel = 3,sigma=1.0):
        kernel = np.zeros((tam_kernel,tam_kernel))
        centro = (tam_kernel-1)/2
   
        for i in range(tam_kernel):
            for j in range(tam_kernel):
                posx = i-centro
                posy = -(j-centro)
                """ valor1 = -1/(math.pi*sigma**4)
                valor2 = 1 - (posx**2+posy**2)/(2*sigma**2)
                valor3 = math.exp( -(posx**2+posy**2)/(2*sigma**2) ) """

                valor1 = 1/(2*math.pi*sigma**4)
                valor2 = 2-(posx**2+posy**2)/(sigma**2)
                valor3 = math.exp(-(posx**2+posy**2)/(2*sigma**2))

                valor = valor1*valor2*valor3
                kernel[i][j] = valor
        

        return kernel
            



        
    def kernel_gaussiano(tam_kernel = 3,sigma =1):
        kernel = np.zeros((tam_kernel,tam_kernel))
        centro = (tam_kernel-1)/2
   
        for i in range(tam_kernel):
            for j in range(tam_kernel):
                posx = i-centro
                posy = -(j-centro)
                valor1 = 1/(2*math.pi*sigma**2)
                valor2 = math.exp(-(posx**2+posy**2)/(2*sigma**2))

                valor = valor1*valor2
                kernel[i][j] = valor

        #kernel = kernel/np.sum(kernel)

        return kernel

    def generar_borde(imagen, tam_exceso = 2):  
        width = imagen.shape[0]
        height = imagen.shape[1]
        bordeada = np.zeros((width+tam_exceso*2,height+tam_exceso*2),np.uint8)
        for i in range(width):
            for j in range(height):
                bordeada[i+tam_exceso][j+tam_exceso] = imagen[i][j]
        return bordeada

    def operacion_convolucion(i,j,kernel,exceso,imagen):
        tam_kernel = kernel.shape[0]
        resultado = 0
        for k in range(tam_kernel):
            for l in range(tam_kernel):
                valor = imagen[i-exceso+k][j-exceso+l]
                resultado += kernel[k][l]*valor 

        return resultado

    def convolucion(imagen,kernel):
        width = imagen.shape[0]
        height = imagen.shape[1]
        tam_kernel = kernel.shape[0]
        exceso = int((tam_kernel-1)/2)
        print(exceso)
        convolucionada = np.zeros((width-exceso*2,height-exceso*2),np.float64)
        for i in range(width):
            if(i<(width-exceso) and i>=exceso):
                for j in range(height):
                    if(j<(height-exceso) and j>=exceso):
                        #operamos en la matriz
                        resultado =  ArcadioCv.operacion_convolucion(i,j,kernel,exceso,imagen) 
                        convolucionada[i-exceso][j-exceso] = resultado
        #regresa la imagen con valores flotantes, hace falta cambiar a np.uint8 y absoluto        
        return convolucionada

    def histograma_imagen_grises(imagen):
        width = imagen.shape[0]
        height = imagen.shape[1]
        histograma = np.zeros(256,np.uint)

        for i in range(width):
            for j in range(height):
                posicion = imagen[i][j]
                histograma[posicion] +=1
        print(sum(histograma))
        return histograma

    def filtro_log(imagen,tam_kernel=3,sigma=1):
        kernel = ArcadioCv.kernel_logc(tam_kernel,sigma)
        print(kernel)
        exceso = int((tam_kernel-1)/2)
        bordeada = ArcadioCv.generar_borde(imagen,exceso)
        imagen_log = ArcadioCv.convolucion(bordeada,kernel)
        ArcadioCv.visualizar_imagen(imagen_log)
        #convol = cv2.filter2D(imagen, -1, kernel)
        delta = float(input("Ingresa el tamaño de delta: "))
        imagen_cruces, imagen_delta,imagen_final  = ArcadioCv.cruce_por_cero(imagen_log,delta)

        return imagen_cruces, imagen_delta,imagen_final,imagen_log

    def filtro_gauss(imagen,tam_kernel = 3,sigma=1):
        kernel = ArcadioCv.kernel_gaussiano(tam_kernel,sigma)
        exceso = int((tam_kernel-1)/2)
        bordeada = ArcadioCv.generar_borde(imagen,exceso)
        imagen_gauss = ArcadioCv.convolucion(bordeada,kernel)
        imagen_gauss = np.array(imagen_gauss,np.uint8)
        return imagen_gauss

    def umbral_otsu(imagen):
        width = imagen.shape[0]
        height = imagen.shape[1]
        resolucion = width*height
        histograma = ArcadioCv.histograma_imagen_grises(imagen)
        #print(histograma)
        resultado = 0
        valores = np.zeros(256)
        t=0
        for i in range(256):

            """INICIALES"""
            suma_hk = sum(histograma[:t])
            suma_k_hk = sum(histograma[i]*i for i in range(0,t))

            Wb = suma_hk / resolucion

            Mb =suma_k_hk/suma_hk if(suma_hk!=0) else 0

            suma_sigmas = sum((i-Mb)**2*histograma[i] for i in range(0,t))

            Sb = suma_sigmas/suma_hk if(suma_hk!=0) else 0

            """FINALES"""
            suma_hk = sum(histograma[t+1:255])
            suma_k_hk = sum(histograma[i]*i for i in range(t+1,255))

            Wf = suma_hk / resolucion

            Mf =suma_k_hk/suma_hk if(suma_hk!=0) else 0

            suma_sigmas = sum((i-Mf)**2*histograma[i] for i in range(t+1,255))

            Sf  = suma_sigmas/suma_hk if(suma_hk!=0) else 0

            resultado = Wb *Sb + Wf * Sf
            #print(resultado)

            valores[i] = resultado
            t+=1
        #el minimo de valores
        min = np.where(valores == np.amin(valores))[0][0]
        return min
        
    def cruce_por_cero(imagen,delta = 0.37):
        width = imagen.shape[0]
        height = imagen.shape[1]
        bordes = np.zeros((width,height),np.uint8)
        imagen_cruces = np.zeros((width,height),np.uint8)
        for i in range(width):
            for j in range(height):
                #comparamos izquierda con derecha
                #buscamos cambio de positovo a negativo
                
                """ if((i-1)>=0 and (j-1)>=0 and (i+1)<width and (j+1)<height):
                    #derecha
                    resta = imagen[i][j]-imagen[i+1][j]
                    bordes[i+1][j] = 255 if (resta>delta) else 0
                    #izquierda
                    resta = imagen[i][j]-imagen[i-1][j]
                    bordes[i-1][j] = 255 if (resta>delta) else 0
                    #arriba
                    resta = imagen[i][j]-imagen[i][j+1]
                    bordes[i][j+1] = 255 if (resta>delta) else 0
                    #abajo
                    resta = imagen[i][j]-imagen[i][j-1]
                    bordes[i][j-1] = 255 if (resta>delta) else 0
                    #serecha arriba
                    resta = imagen[i][j]-imagen[i+1][j+1]
                    bordes[i+1][j+1] = 255 if (resta>delta) else 0
                    #derecha abajo
                    resta = imagen[i][j]-imagen[i+1][j-1]
                    bordes[i+1][j-1] = 255 if (resta>delta) else 0
                    #izquierda arriba
                    resta = imagen[i][j]-imagen[i-1][j+1]
                    bordes[i-1][j+1] = 255 if (resta>delta) else 0
                    #izquierda abajo
                    resta = imagen[i][j]-imagen[i-1][j-1]
                    bordes[i-1][j-1] = 255 if (resta>delta) else 0
 """
                    
                for a in range(-1,2):
                    for b in range(-1,2):
                        if((i+a)>=0 and (j+b)>=0 and (i+a)<width and (j+b)<height):
                            if(a !=0 and b !=0):
                                #diagonales
                                """IMAGEN con delta"""
                                resta = abs(imagen[i][j]-imagen[i+a][j+b])
                                bordes[i+a][j+b] = 255 if (resta>delta) else 0
                                #lados
                                resta = abs(imagen[i][j]-imagen[i][j+b])
                                bordes[i][j+b] = 255 if (resta>delta) else 0

                                resta = abs(imagen[i][j]-imagen[i+a][j])
                                bordes[i+a][j] = 255 if (resta>delta) else 0

                                """Cruces por 0"""
                                #daigonales
                                if( (imagen[i][j]<0 and imagen[i+a][j+b]>0)
                                    or (imagen[i][j]>0 and imagen[i+a][j+b]<0)
                                    ):
                                    imagen_cruces[i+a][j+b] = 255
                                else:
                                    imagen_cruces[i+a][j+b] = 0
                                #lados
                                if( (imagen[i][j]<0 and imagen[i][j+b]>0)
                                    or (imagen[i][j]>0 and imagen[i][j+b]<0)):
                                    imagen_cruces[i][j+b] = 255
                                else:
                                    imagen_cruces[i][j+b] = 0

                                if( (imagen[i][j]<0 and imagen[i+a][j]>0)
                                    or (imagen[i][j]>0 and imagen[i+a][j]<0)):
                                    imagen_cruces[i+a][j] = 255
                                else:
                                    imagen_cruces[i+a][j] = 0


                                #resta = imagen[i][j]-imagen[i-a][j-b] 
                                #bordes[i-a][j-b] = 255 if (resta>delta) else 0
                                #if(i==1 and j==0):
                                   
                        #print(a,b)
                #resta = imagen[i][j]-imagen[i+1][j] if ((i+1)-width) else imagen[i][j]
                #hacia arriba 

                #bordes[i][j] = 255 if (resta>delta) else 0

        imagen_final = np.zeros((width,height),np.uint8)

        for i in range(width):
            for j in range(height):
                imagen_final[i,j] = 255 if( (imagen_cruces[i,j] and bordes[i,j])==255 ) else 0
        #imagen_final = cv2.bitwise_and(imagen_cruces,bordes)
        return imagen_cruces,bordes,imagen_final
    
    def mi_watershed(imagen,grises,gauss,umbralizada):
        distancia = ndi.distance_transform_edt(imagen)
        cordenadas = peak_local_max(distancia, footprint=np.ones((3,3)),labels=imagen)
        mask = np.zeros(distancia.shape,bool)
        mask[tuple(cordenadas.T)] = True
        markers, _ = ndi.label(mask) #colores
        labels = watershed(-distancia,markers,mask = imagen)

        fig, axes = plt.subplots(ncols=3, figsize=(7, 7),nrows=3, sharex=True, sharey=True)
        ax = axes.ravel()


        ax[0].imshow(grises, cmap=plt.cm.gray)
        ax[0].set_title('Imagen grises')
        ax[1].imshow(gauss, cmap=plt.cm.gray)
        ax[1].set_title('Imagen Gaussiana')
        ax[2].imshow(umbralizada, cmap=plt.cm.gray)
        ax[2].set_title('Umbralizada')

        ax[3].imshow(imagen, cmap=plt.cm.gray)
        ax[3].set_title('Imagen original')
        ax[4].imshow(-distancia, cmap=plt.cm.gray)
        ax[4].set_title('Distancias')
        ax[5].imshow(labels, cmap=plt.cm.nipy_spectral)
        ax[5].set_title('Objetos separados')

        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()
        
    def umbralar(imagen,umbral):
        rows,cols = imagen.shape
        umbralada = np.zeros((rows,cols),np.uint8)
        for i in range(rows): 
            for j in range(cols):
                umbralada[i,j] = 255 if(imagen[i,j]>=umbral) else 0

        return umbralada



               



if __name__ == "__main__":
    imagen = ArcadioCv.abrir_imagen_rgb("carta.png")
    grises = ArcadioCv.rgb_to_grises(imagen)
    #ArcadioCv.visualizar_imagen(grises)
    #umbral = ArcadioCv.umbral_otsu(grises)
    #print("Umbral",umbral)
    ArcadioCv.mi_watershed(grises)

    """PEDIR DATOS"""
    tam_kernel = int(input("Ingresa el tamaño del kernel: "))
    sigma = float(input("Ingresa sigma: "))


    imagen_log = ArcadioCv.filtro_log(grises,tam_kernel,sigma)
    ArcadioCv.visualizar_imagen(imagen_log)
    print(imagen_log)
    #umbral = ArcadioCv.umbral_otsu(imagen)
    #print("umbral",umbral)

