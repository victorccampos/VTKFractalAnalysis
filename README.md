# VTK Fractal Dimension Calculator and VideoMaker  
#### The following project was used in a presentation at Encontro de Outono da Sociedade Brasileira de Física de 2024 by the name "Phase Field Simulation of Dendritic Crystal Growth".  

The `main.py` file contains a script that uses a Tkinter interface to select a user specified folder that contains two-dimension ".vtk" files.
It organizes the file based on the number, e.g, file_0, file_1,... and corvert them to numpy arrays (images). At each iteration, you can calculate the fractal dimension of each image by removing the coments on line 104.  

The `data` folder contains a collection of .txt files that contain fractal dimension calculations of dentritic growth in phase field simulations.The fractal dimension calculation method used here is 
boxcount method implemented from Francesco Turci at https://francescoturci.net/2016/03/31/box-counting-in-numpy/.  

The `data_analysis.ipynb` is a jupyter notebook used to study the behavior of the fractal dimension of different seed shapes and symmetry parameters.

