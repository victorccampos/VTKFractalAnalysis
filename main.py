import tkinter as tk
from tkinter import filedialog
import os

import numpy as np
import vtk 
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import time

def ask_for_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='Selecionar a pasta da simulação que contém os arquivos vtk')
    root.destroy()
    return folder_path

def get_vtk_data(vtk_filepath: str , index: int, save_png: bool) -> np.ndarray:
    """
    vtk_filepath : caminho completo do arquivo vtk
    save_png: opcao de salvar ou nao o png
    index: indice da foto PhaseField_*.png
    """
    # Read VTK file
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(vtk_filepath)

    reader.Update()

    # Get the data from VTK file
    vtk_data = reader.GetOutput()
    
    # Check if the data is valid
    if vtk_data.GetPointData() is not None:
        # Convert VTK data to Numpy Array
        data_array = vtk.util.numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetScalars())
        # Reshappe according to dimension of the data
        dims = vtk_data.GetDimensions()
        image_width = dims[1]
        image_height = dims[0]
        numpy_image = np.reshape(data_array, (image_height, image_width))  
    else:
        print("Erro: Não foi possível converter os dados do vtk para array numpy")

    if save_png == True:
        plt.imsave(f"phasefield_{index}.png", numpy_image, cmap='coolwarm', origin='lower')
    
    return numpy_image

def boxcount(image: np.ndarray, plot: False) -> float:
    """
    Recebe como parâmetro uma imagem no formato de array numpy e retorna a dimensão fractal.
    image: np.ndarray
    plot: Por padrão False - Não salva o pdf.
    """
    pixels=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]>0:
                pixels.append((i,j))
    
    Lx=image.shape[1]
    Ly=image.shape[0]
    #print (f'Dimensões: {Lx}, {Ly}')
    pixels = np.array(pixels)
    #print (pixels.shape)
    
    # computing the fractal dimension
    #considering only scales in a logarithmic list
    scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    Ns=[]
    # looping over several scales
    for scale in scales:
        #print ("======= Scale :",scale)
        # computing the histogram
        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))
    
    # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)

    if plot == True:
        plt.plot(np.log(scales),np.log(Ns), 'o', mfc='none')
        plt.plot(np.log(scales), np.polyval(coeffs,np.log(scales)))
        plt.xlabel('log $\epsilon$')
        plt.ylabel('log N')
        # plt.savefig('fractal_dimension.png')
        print ("The Hausdorff dimension is\n", -coeffs[0]) #the fractal dimension is the OPPOSITE of the fitting coefficient
    return -coeffs[0]


def main():
    start_time = time.time()
    folder_path = ask_for_folder()
    vtk_files = [file for file in os.listdir(folder_path) if file.endswith(".vtk")]
    vtk_files.sort(key=len)
    
    FD_T4 = []
    ################### ANIMAÇÃO ##############################
    num_arquivos = len(vtk_files)
    for index in range(num_arquivos):
        vtk_file_path = folder_path + '/' + vtk_files[index]
        image = get_vtk_data(vtk_filepath=vtk_file_path, index= index, save_png=True)
        #fractal_dimension = boxcount(image=image, plot=False)
        #FD_T4.append(fractal_dimension)
    end_time = time.time()
    #plt.plot(FD_T4)
    #plt.title("Triangular j=4")
    #plt.xlabel("Passos")
    #plt.ylabel("Dimensão Fractal")
    #plt.show()
    #print("\033[92mPrograma Concluído!\033[0m")
    execution_time = end_time - start_time
    print(f"Tempo de execução: {execution_time}")




if __name__ == "__main__":
    main()