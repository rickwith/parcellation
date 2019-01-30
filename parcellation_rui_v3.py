#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:58:12 2018

@author: XRAY
"""
import nibabel as nib
import numpy as np
import sys
import os
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
#import  matplotlib  
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import cm as cm


def Level_1_parcellation(data_list,mask_file, output_folder,cluster_range,num_iteration):
   #  cluster_range = 20
   #  num_iteration = 100
  #--- load mask file ---------
     ma = nib.load(mask_file)
     ma_data = np.asarray(ma._data)
     ma_data_re = np.reshape(ma_data,(np.prod(ma.shape),1))
     index = np.argwhere(ma_data_re>0)
     print("The dimension of mask is %d %d %d" %  ma_data.shape)

  #--- read csv file -----------
     table=np.genfromtxt(data_list, skip_header=1,delimiter=',', dtype=None)

  #--- initialize all_corr_matrx ----
     #all_corr_matrix = np.zeros((len(index),len(index)))
    # sub_ad_matrix = np.zeros((*input_matrix.shape,cluster_range))     
   #  print("+++++++ The dimension of all_ad_matrix is %d %d %d " %  all_ad_matrix.shape)
   
     for ind, row in enumerate(table):
     #--- loop through each subject ---------

         print(">>>%s" % str(row, 'utf-8'))
         fMRI_file = str(row,'utf-8')
   
         img = nib.load(fMRI_file)
         img_data = np.asarray(img._data)
 
         print("The dimension of fMRI is %d %d %d %d" % np.shape(img_data))

     #--- reshape data ---
         img_data_re = np.reshape(img_data, (np.prod(img_data.shape[:-1]), img_data.shape[-1] ))

     #--- Correlation within mask 
#    index = np.argwhere(ma_data_re>0)
    #print(index[:,0])
    #print(index.shape)
         corr_matrix = np.corrcoef(img_data_re[index[:,0],:], rowvar= True)
         print("The dimension of correlation matrix is %d %d " %corr_matrix.shape)
         #np.savetxt('corr_matrix.txt',corr_matrix)
         np.save(os.path.join(output_folder,"corr_matrix_{:02d}.npy".format(ind)),corr_matrix)            
     #----- clustering on each subject --------
         
  #       cluster_range = 20
  #       num_iteration = 100
         sub_ad_matrix = np.zeros((*corr_matrix.shape,cluster_range))
         print("+++++++ The dimension of all_ad_matrix is %d %d %d " %  sub_ad_matrix.shape)

         for num_k in range(2,cluster_range+1):
                sum_matrix = np.zeros((corr_matrix.shape))
                for num_iter in range(num_iteration):
                          input_matrix = np.exp(-corr_matrix / corr_matrix.std())
                          ad_matrix, ct = base_parcellation_SC(input_matrix,num_k,num_iter)
                          sum_matrix = sum_matrix + ad_matrix    
                          if np.mod(num_iter, 10) == 0 :
                            print("------- %d" % num_iter)
                sub_ad_matrix[:,:,num_k-1] =sum_matrix/num_iteration
                print("------Level 1 finished the %d cluster solution " % num_k)   
     
     #--- write out  matrix ----------- 
         print(">>>%d" % ind)
         np.save(os.path.join(output_folder,"sub_matrix_{:02d}.npy".format(ind)),sub_ad_matrix)  
         #sub_ad_matrix = np.load("sub_matrix_{:02d}.npy".format(ind))

     #ad_matrix_1st = np.mean(sub_ad_matrix, axis=2)
     #np.save(os.path.join(output_folder,'ave_ad_matrix_1st.npy'),ad_matrix_1st) 
 
  #Human readable data
    # np.savetxt('ave_matrix.txt', all_corr_matrix/len(table))
    
#     return  ad_matrix_1st


def base_parcellation_SC(corr_matrix, num_k,num_iter):   
    
    #input_matrix = np.load('ave_matrix.npy')
    #input_matrix = np.exp(-corr_matrix / corr_matrix.std())
    input_matrix = corr_matrix
 #   print("+++++++ number of voxels %d" % len(input_matrix))
 #   print("+++++++ num_k is %d  num_iter %d"  % ( num_k, num_iter))

    if np.isnan(input_matrix).any(): 
       print(">>>>>> NAN detected")
       raise SystemExit("AT line 102 program exited")
    else:      
       clustering = SpectralClustering(n_clusters=num_k , assign_labels="discretize",affinity="precomputed" ,random_state=num_iter).fit(input_matrix)
       #clustering = KMeans(n_clusters=num_k ,random_state=num_iter,algorithm ='auto').fit(input_matrix)
       tmp = clustering.labels_
   
    ad_matrix = adjacency_matrix(tmp,num_k)          
    return  ad_matrix, tmp

def base_parcellation_KM(corr_matrix, num_k,num_iter):   
    
    #input_matrix = np.load('ave_matrix.npy')
    #input_matrix = np.exp(-corr_matrix / corr_matrix.std())
    input_matrix = corr_matrix
 #   print("+++++++ number of voxels %d" % len(input_matrix))
 #   print("+++++++ num_k is %d  num_iter %d"  % ( num_k, num_iter))

    if np.isnan(input_matrix).any(): 
       print(">>>>>> NAN detected")
       raise SystemExit("AT line 102 program exited")
    else:      
       #clustering = SpectralClustering(n_clusters=num_k , assign_labels="discretize",affinity="precomputed" ,random_state=num_iter).fit(input_matrix)
       clustering = KMeans(n_clusters=num_k ,random_state=num_iter,algorithm ='auto').fit(input_matrix)
       tmp = clustering.labels_
   
    ad_matrix = adjacency_matrix(tmp,num_k)          
    return  ad_matrix, tmp


def base_parcellation_AG(corr_matrix, num_k):   
    
    #input_matrix = np.load('ave_matrix.npy')
    #input_matrix = np.exp(-corr_matrix / corr_matrix.std())
    input_matrix = corr_matrix
 #   print("+++++++ number of voxels %d" % len(input_matrix))
 #   print("+++++++ num_k is %d  num_iter %d"  % ( num_k, num_iter))

    if np.isnan(input_matrix).any(): 
       print(">>>>>> NAN detected")
       raise SystemExit("AT line 102 program exited")
    else:      
       #clustering = SpectralClustering(n_clusters=num_k , assign_labels="discretize",affinity="precomputed" ,random_state=num_iter).fit(input_matrix)
       #clustering = KMeans(n_clusters=num_k ,random_state=num_iter,algorithm ='auto').fit(input_matrix)
       clustering = AgglomerativeClustering(affinity='precomputed', compute_full_tree='auto',
            connectivity=None, linkage='average', memory=None, n_clusters=num_k,
            pooling_func='deprecated').fit(input_matrix) 
       tmp = clustering.labels_
     
    ad_matrix = adjacency_matrix(tmp,num_k)          
    return  ad_matrix, tmp



def adjacency_matrix(cluster_ind,num_cluster):    
    #---- turn vector into sparse matrix --------
    length_matrix =  len(cluster_ind)
    matrix = np.zeros((length_matrix, length_matrix))
    for i in range(0,num_cluster):
          index = np.argwhere(cluster_ind==i)
          for ii in range(len(index)):
             for jj in range(len(index)):
                matrix[index[ii],index[jj]]=1
    return matrix      

def Level_2_parcellation(subject_list,output_folder,cluster_range,num_iteration):
    #------ 
    #admatrix = np.load('ave_ad_matrix_1st.npy')
    table=np.genfromtxt(subject_list, skip_header=1,delimiter=',', dtype=None)
    num_sub = len(table)
    com_silhouette = np.zeros((num_iteration,num_sub,cluster_range))

    
    for num_k in range(2,cluster_range+1):

        sum_matrix = np.zeros((1106,1106))
        sum_matrix2 = np.zeros((1106,1106))
        sub_ad_matrix2 = np.zeros((1106,1106))
        
        for ind in range(num_sub):
           sub_ad_matrix = np.load(os.path.join(output_folder,"sub_matrix_{:02d}.npy".format(ind)))
           sum_matrix  = sum_matrix + sub_ad_matrix[:,:,num_k-1]

        for num_iter in range(num_iteration):
           
            ad_matrix, ct = base_parcellation_SC(sum_matrix/num_sub,num_k,num_iter)
            #ad_matrix, ct = base_parcellation_AG(sum_matrix/num_sub,num_k)
           #ad_matrix, ct = base_parcellation_KM(ad_matrix,num_k,num_iter)
            sum_matrix2 = sum_matrix2 + ad_matrix
       
     # ---- estimate averaged sihouett ---
            #s_score=silhouette_score(corr_matrix, ct,random_stats=num_iter)
            s_score=silhouette_score(ad_matrix, ct, metric='precomputed',random_stats=num_iter)
            com_silhouette[num_iter,ind,num_k-1]= s_score
            if np.mod(num_iter,10) ==0 :
                    print("------- %d" % num_iter)

        sub_ad_matrix2 =sum_matrix2/num_iteration
        print("------Level 2 finished the %d cluster solution " % num_k)
        np.save(os.path.join(output_folder,"sub_matrix2_{:02d}.npy".format(num_k)),sub_ad_matrix2)


    fin_silhouette = np.mean(np.mean(com_silhouette,axis=0),axis=0)
    #np.save(os.path.join(output_folder,'fin_silhouette.npy'),fin_silhouette) 
    with open('fin_silhouette.txt','w') as f:
         for item in fin_silhouette:
              f.write("%d\n" % item)                 
    for num_k in range(2,cluster_range+1):
             print("For n_clusters =", str(num_k),' The ave sihouette score is :' , 
                  str(fin_silhouette[num_k-1]) )                          
    # determine the optimal cluster 
     
    index_max = np.argmax(fin_silhouette) 
    print(">>>> The optimal solution is ", str(index_max+1)) 
     
def plot_matrix_and_save(input,name):
     plt.imsave(name,input)
    #plt.ioff() 
#    fig = plt.figure()
#    fig, ax1 = plt.subplots(1,1)
#    cmap = cm.get_cmap('jet',30)
#    ax = fig.plot()
#    ax.matshow(input)
    #plot.title('correlation_matrix')
#    plt.show() 
#    plt.title("correlation")
#    plt.colorbar()
#    plt.show() 
#    plt.savefig('fig.png', dpi=fig.dpi)   
        
def Level_3_parcellation(subject_list,output_folder,cluster_range,mask_file): 
    
    #table=np.genfromtxt(subject_list, skip_header=1,delimiter=',', dtype=None)
    #num_sub = len(table)
    
    for num_k in range(2,cluster_range+1):
        
        #sum_matrix = np.zeros((1106,1106))
 
        sub_ad_matrix = np.load(os.path.join(output_folder,"sub_matrix2_{:02d}.npy".format(num_k)))
        ad_matrix, ct = base_parcellation_SC(sub_ad_matrix,num_k,0)
        #ad_matrix, ct = base_parcellation_AG(sub_ad_matrix,num_k)
        print(ct.shape)
        plot_matrix_and_save(ad_matrix,os.path.join(output_folder,"sub_matrix3_{:02d}.png".format(num_k)))
        #-------- write out parcellation image
        img = nib.load(mask_file)
        img_data = np.asarray(img._data)
        img_data_re = np.reshape(img_data,(np.prod(img.shape),))
        index = np.argwhere(img_data_re>0)
        ct=ct+1
        #np.save('ct.npy',ct)
        #print(lct.shape)
        output_pre = np.zeros((np.prod(img.shape),))
        output_pre[np.transpose(index)]= np.transpose(ct)
        
        output_data = np.reshape(output_pre,img.shape)
        
        output_filename= os.path.join(output_folder,"parcel_SC_"+str(num_k)+"_"+mask_file)
        nii = nib.Nifti1Image(output_data, img._affine)
        nii._header = img._header
#      nii._header['cal_max'] = max_output
        print(nii._header)
        nii.to_filename(output_filename)


if __name__ == "__main__":

# parcellation_kmeans usage 
# ---- input_list.txt mask.nii outputfolder

   if len(sys.argv) == 1:
      print("Please provide data, mask, threshold. Be a good boy")

   elif len(sys.argv) > 2:
       print("+++++++ The input fMRI : %s " %str(sys.argv[1]))
       datalist = str(sys.argv[1])
       print("+++++++ The ROI mask : %s" % str(sys.argv[2]))
       maskfile = str(sys.argv[2])

       if not os.path.exists(datalist):
          print("Input file does not exist!")

       if not os.path.exists(maskfile):
          print("Mask file does not exist!")
 
       if len(sys.argv) >3:
            output_folder = str(sys.argv[3])
            print("+++++++ The output folder is %s" % str(sys.argv[3]) )
       else:
            output_folder = os.getcwd()
            print("No output folder specified. Using current folder instead. I got your covered!")

       cluster_range = 6
       num_iteration = 500
       Level_1_parcellation(datalist, maskfile, output_folder,cluster_range, num_iteration)
       Level_2_parcellation(datalist,output_folder,cluster_range,num_iteration)    
       Level_3_parcellation(datalist,output_folder,cluster_range,maskfile)
          
          
