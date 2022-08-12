# -*- coding: utf-8 -*-
"""
Created on Fri Apr 2 00:44:09 2022

@author: PC
"""


from matplotlib import pyplot as plt
import numpy as np
import torch 
import kornia as K
from kornia import morphology as morph

splited = 2

#%% Funciones Morfológicas
def Erosion(img_rgb):
    # print("Erosion")
    #print(torch.cuda.memory_allocated())
    #print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    #print(img_rgb.shape)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    #print(len(img_rgb))
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    #kernel = torch.ones(kernel_size,kernel_size).to(device)
    contador=0
    for img in img_rgb:
        #print(img.shape)
        img = img.float()/255.
        torch.cuda.empty_cache()
        img = morph.erosion(img, kernel)
        img = img.float()*255.
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        contador+=1
        del img
        torch.cuda.empty_cache()
    
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    #print(img_rgb.shape)
    
    del kernel
    torch.cuda.empty_cache()
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_allocated())
    #print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    return img_rgb

def Dilation(img_rgb):
    # print("Dilation")
    #print(torch.cuda.memory_allocated())
    #print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    #print(img_rgb.shape)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    #print(len(img_rgb))
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    #kernel = torch.ones(kernel_size,kernel_size).to(device)
    contador=0
    for img in img_rgb:
        img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.dilation(img, kernel)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    #print(img_rgb.shape)
    
    del kernel
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_allocated())
    #print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    return img_rgb

def Closing(img_rgb):
    # print("Closing")
    #print(torch.cuda.memory_allocated())
    #print('closing\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    #print(img_rgb.shape)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    #print(len(img_rgb))
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    #kernel = torch.ones(kernel_size,kernel_size).to(device)
    contador=0
    for img in img_rgb:
        img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.closing(img, kernel)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    #print(img_rgb.shape)
    
    del kernel
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    #print(torch.cuda.memory_allocated())
    #print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    return img_rgb

def Opening(img_rgb):
    """Opening """
    # print("Opening")
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    #kernel = torch.ones(kernel_size,kernel_size).to(device)
    contador=0
    for img in img_rgb:
        img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.opening(img, kernel)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)

    del kernel
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    return img_rgb

#%%Detección de bordes
def Sobel(img_rgb):
    """Sobel"""
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    contador=0
    for img in img_rgb:
        img = img.float()/255
        torch.cuda.empty_cache()
        img = K.filters.sobel(img)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()

    return img_rgb

def LaPlacian(img_rgb):
    """LaPlacian"""
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    #kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    #kernel = torch.ones(kernel_size,kernel_size).to(device)
    contador=0
    for img in img_rgb:
        img = img.float()/255
        torch.cuda.empty_cache()
        ##img = morph.opening(img, kernel)
        img = K.filters.laplacian(img, kernel_size=5)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)

    #del kernel
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    return img_rgb

def Gradient(img_rgb):
    """Gradient function """
    device = "cuda:0"
    # print("Gradiente")
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    #kernel = torch.ones(kernel_size, kernel_size).to(device)
    contador=0
    for img in img_rgb:
        #print(img.shape)
        img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.gradient(img, kernel)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)

    del kernel
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    return img_rgb

#%%Funciones aritméticas
def suma_imgs2(img1, img2):
    device = "cuda:0"
    img1 = img1.to(device)
    img1 = img1.float()/255
    
    img2 = img2.to(device)
    img2 = img2.float()/255
    
    suma = img1+img2
    suma = suma.float()*255
    suma = torch.clamp(suma, min=0.0,max=255.0)
    
    img1 = img1.to("cpu")
    img2 = img2.to("cpu")
    suma = suma.to("cpu")
    return suma

def suma_imgs3(img1, img2, img3):
    device = "cuda:0"
    img1 = img1.to(device)
    img1 = img1.float()/255
    
    img2 = img2.to(device)
    img2 = img2.float()/255
    
    img3 = img3.to(device)
    img3 = img3.float()/255
    
    suma = img1+img2+img3
    suma = suma.float()*255
    suma = torch.clamp(suma, min=0.0,max=255.0)
    
    img1 = img1.to("cpu")
    img2 = img2.to("cpu")
    img3 = img3.to("cpu")
    suma = suma.to("cpu")
    return suma

def resta_imgs(img1, img2):
    device = "cuda:0"
    img1 = img1.to(device)
    img1 = img1.float()/255
    
    img2 = img2.to(device)
    img2 = img2.float()/255
    
    resta = img1-img2
    resta = resta.float()*255
    resta = torch.clamp(resta, min=0.0,max=255.0)
    
    img1 = img1.to("cpu")
    img2 = img2.to("cpu")
    resta = resta.to("cpu")
    
    return resta

def sqrt(img):
    device = "cuda:0"
    img = img.to(device)
    img = img.float()/255
    sq = torch.sqrt(img)
    sq = sq.float()*255
    sq = torch.clamp(sq, min=0.0, max=255.0)
    
    img = img.to("cpu")
    sq = sq.to("cpu")
    
    return sq

#%%Funciones de filtrado y transformaciones de intensidad
def Gaussian_blur_2d(img_rgb):
    """Gaussian blur 2D = GB_2D
    Size kernel should be odd int positive (3,5,7,9,11,13,15,17) are the possible numbers to operate the function
    The seccond parameter is the standard deviation of the kernel. Must be a float number
    """
    # print("Gaussian_blur_2d")
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel_size=7

    contador=0
    for img in img_rgb:
        img = img.float()/255
        torch.cuda.empty_cache()
        img = K.filters.gaussian_blur2d(img, (kernel_size,kernel_size), (10.0, 10.0))
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    return img_rgb


# def Box_blur(img_rgb,kernel_size):
#     """Box blur """
#     print("Box_blur",kernel_size)
#     device = "cuda:0"
#     img_rgb = img_rgb.to(device)
#     img_rgb :torch.Tensor = torch.split(img_rgb,splited)
#     img_rgb = list(img_rgb)
#     contador=0
#     for img in img_rgb:
#         img = img.float()/255
#         torch.cuda.empty_cache()
#         img = K.filters.box_blur(img, (kernel_size, kernel_size))
#         img = img.float()*255
#         img = torch.clamp(img,min=0.0,max=255.0)
#         img_rgb[contador]=img
#         del img
#         torch.cuda.empty_cache()
#         contador+=1
#     img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    
#     img_rgb = img_rgb.to("cpu")
#     #img_rgb = list(map(torch.stack, zip(*img_rgb)))
#     torch.cuda.empty_cache()
#     # print(torch.cuda.memory_allocated())
#     # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
#     return img_rgb


# def Mtn_blur(img_rgb,kernel_size,angle):
#     """Motion blur 
#     img_rgb = tensor in shape (B,C,H,W) 
#     kernel_size = Must be an odd integer positive
#     angle =  Must be a integer [1-90]
#     """
#     print("MTN_blur",kernel_size)
#     device = "cuda:0"
#     img_rgb = img_rgb.to(device)
#     img_rgb :torch.Tensor = torch.split(img_rgb,splited)
#     img_rgb = list(img_rgb)
#     contador=0
#     for img in img_rgb:
#         img = img.float()/255
#         torch.cuda.empty_cache()
#         img = K.filters.motion_blur(img, kernel_size, angle, 1)
#         img = img.float()*255
#         img = torch.clamp(img,min=0.0,max=255.0)
#         img_rgb[contador]=img
#         del img
#         torch.cuda.empty_cache()
#         contador+=1
#     img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    
#     img_rgb = img_rgb.to("cpu")
#     #img_rgb = list(map(torch.stack, zip(*img_rgb)))
#     torch.cuda.empty_cache()
#     # print(torch.cuda.memory_allocated())
#     # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
#     return img_rgb

# def Sharpen(img_rgb,kernel_size,standar_deviation):
#     """Sharpen """
#     print("Sharpen",kernel_size,standar_deviation)
#     device = "cuda:0"
#     img_rgb = img_rgb.to(device)
#     img_rgb :torch.Tensor = torch.split(img_rgb,splited)
#     img_rgb = list(img_rgb)
#     sharpen = K.filters.UnsharpMask((kernel_size,kernel_size), (standar_deviation,standar_deviation))
#     contador=0
#     for img in img_rgb:
#         img = img.float()/255
#         img = sharpen(img)
#         img = img.float()*255
#         img = torch.clamp(img,min=0.0,max=255.0)
#         img_rgb[contador]=img
#         del img
#         torch.cuda.empty_cache()
#         contador+=1
#     img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
#     del sharpen
    
#     img_rgb = img_rgb.to("cpu")
#     #img_rgb = list(map(torch.stack, zip(*img_rgb)))
#     torch.cuda.empty_cache()
#     # print(torch.cuda.memory_allocated())
#     # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

#     return img_rgb

def En_adbright(img_rgb):
    """Enhance adjust brightness
    factor = must be float between [-0.5,0.5]
    """
    # print("Adjust bright")
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    contador=0
    factor=0.2
    for img in img_rgb:
        img = img.float()/255
        torch.cuda.empty_cache()
        img =  K.enhance.adjust_brightness(img, factor)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)

    
    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(),
    

    return img_rgb


def En_equal(img_rgb):
    """Enhance adjust brightness """
    # print("Histogram equalization")
    device = "cuda:0"
    img_rgb = img_rgb.to(device)

    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    contador=0
    for img in img_rgb:
        #print(img.shape)
        img = (img.float()/255)
        img =  K.enhance.equalize(img)
        img = img.float()*255
        img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)

    img_rgb = img_rgb.to("cpu")
    #img_rgb = list(map(torch.stack, zip(*img_rgb)))
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    # print('erosion\nEvaluation before:\t', torch.cuda.memory_allocated(),
    

    return img_rgb    
