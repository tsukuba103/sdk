import numpy as np
import pandas as pd

print (pd.__version__)
# '0.16.2'

import xray
print (xray.__version__)
# '0.5.2'


from pylab import *
import cv2
import sys


def padding_length(scan_acquisition_time,M):

    print ("    <    padding_length    >")
    j=0;
    actual_scan_len = np.zeros(M)
    start_no = 0;
    repeat_time = np.zeros(len(scan_acquisition_time))
    for i in np.arange(1,len(scan_acquisition_time),+1):
           repeat_time[i] = (scan_acquisition_time[i]-scan_acquisition_time[0])%7.0;
           if repeat_time[i] - repeat_time[i-1] < -1.0:
               actual_scan_len[j]=i-start_no;
               start_no = i;
               j=j+1;


    nonzero_mean_of_actual_scan_len= actual_scan_len.sum().astype(np.float32) / np.nonzero(actual_scan_len)[0].size;
    print ("      \"precise pixel size\" enables the tilt correction, called padding")
    print ("      that is", nonzero_mean_of_actual_scan_len,"\n\n")
    return nonzero_mean_of_actual_scan_len




def decode(nc_fid):


    print ("\n\n\n\n\n")
    print ("@@@@@@@@@@@@@     decode    @@@@@@@@@@@@@@")

    print ("      \"decode\" reads cdf files, and converts them into a serial of pix data \n\n")

    da2 = xray.DataArray(nc_fid.scan_acquisition_time)
    input_scan_acquisition_time = da2.values

    da2 = xray.DataArray(nc_fid.point_count)
    input_point_count = da2.values


    da2 = xray.DataArray(nc_fid.intensity_values)
    input_intensity_values = da2.values


    da2 = xray.DataArray(nc_fid.mass_values)
    input_mass_values = da2.values


    da2 = xray.DataArray(nc_fid.total_intensity)
    input_total_intensity = da2.values


    da2 = xray.DataArray(nc_fid.scan_index)
    input_scan_index = da2.values


    actual_scan_number = nc_fid.actual_scan_number[:] 
    actual_run_time_length=nc_fid.actual_run_time_length
    print ("      actual_run_time_length=",actual_run_time_length)
    actual_delay_time=nc_fid.actual_delay_time
    print ("      actual_delay_time=",actual_delay_time)
    original_dimension_x = (int)((actual_run_time_length   - actual_delay_time)/7.0)
    print ("      original_dimension_x=",original_dimension_x)
    original_dimension_y = (int)(len(actual_scan_number)/double(original_dimension_x))
    print ("      original_dimension_y=",original_dimension_y)

    print ("da2.dims"),
    da2 = xray.DataArray(nc_fid.mass_values)
    print (da2.dims,da2.coords ,da2.values)
#    print "      input_mass_values[0:10]=",mass_values

 
    AD= 0; 

    pix = {}
    for k in arange(0,len(input_scan_acquisition_time),+1):
          #print k,
          target_mass_values =[];
          target_intensity_values =[];
      
#          for i  in arange(0,input_point_count[k],+1):
#             target_mass_values.append(float(input_mass_values[AD+i]))
#             target_intensity_values.append(float(input_intensity_values[AD+i]))

          pix[k]=[k,input_scan_acquisition_time[k],input_point_count[k],target_intensity_values,target_mass_values,input_total_intensity[k],input_scan_index[k]];

          AD= AD + input_point_count[k];




    NN=padding_length(input_scan_acquisition_time,original_dimension_x);



    print ("\n\n\n\n")

    return pix,original_dimension_x,original_dimension_y,NN




def convert_pix2serial(pix):

    print ("@@@@@@@@@@@@@     convert_pix2serial    @@@@@@@@@@@@@@")
    print ("      \"convert_pix2serial\" converts a set of pix data into a serial\n\n")


    list_point_count    =[];
    list_total_intensity =[];
    new_intensity_values ={}
    new_mass_values ={}

    AD=0;
    serial = {}


#    for k in xrange(0,len(pix),+1):
    for k in range(0,len(pix),+1):
      list_point_count.append(pix[k][2]);
      new_mass_values[k] = pix[k][4];
      new_intensity_values[k] = ones(len(new_mass_values[k]));

      serial[k]=[k,pix[k][1],len(new_mass_values[k]),new_intensity_values[k],new_mass_values[k],pix[k][5],AD];
      AD= AD + len(new_mass_values[k]);

    print ("      scan_index(the memory address of intensity_values or mass_values)=", AD)
    print ("");


    print ("\n\n\n\n")

    return serial

import matplotlib.pyplot as plt


def serial2matrix(N,NN,serial,M,name):
    print ("\n\n\n\n\n")
    print ("@@@@@@@@@@@@@    serial2matrix    @@@@@@@@@@@@@@")

    print ("      \"serial2matrix\" converts a serial of pix data into the image matrix")

    matrx =np.zeros(N +1);

    for i in np.arange(0,M-1,+1): 
       matrx_new = np.zeros(N +1);
       decimal, integer = modf(i*NN)

       for j in np.arange(0 , N, +1): 
           matrx_new[N-j] = serial[integer+j][5];
       matrx = np.dstack((matrx,matrx_new));



    fig = plt.figure(figsize=(12,8))
    x = arange(0, M-1, +1)
    y = arange((N) +1, -1, -1)
    Y, X = meshgrid(y, x) 
    ZZ =  0*X+0*Y


    for ix in arange(0, M-1, +1):

     for iy in arange((N-1) +1, (-1) , -1):
      ZZ[ix][iy]=matrx[0][iy][ix+1];
     ave=mean(ZZ[ix]);
     for iy in arange((N-1) +1, (-1) , -1):
      ZZ[ix][iy]=ZZ[ix][iy]-ave;


     
     if ix%10==0:
      text="%f" % ave
      print (text),

    print ('M=',M,'N=',N)
    plt.pcolor(X, Y, ZZ)

    imagefile='../../JPG-RGB/'+name+'.jpg'
    print ("      that is writable to ",imagefile)
#    plt.savefig(imagefile,format='png', figsize=(original_dimension_x, original_dimension_y), dpi=1000)
    plt.savefig(imagefile,format='jpg', figsize=(1200,800), dpi=1000)

    print ("\n\n\n\n")
#    out=name+'_gray'+'.jpg'
    
    out='../../JPG-GRY/'+name+'.jpg'
    gimg = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    gshrink=cv2.resize(gimg,(1200,800))
    cv2.imwrite(out,gshrink)

    cimg = cv2.imread(imagefile)
    cshrink=cv2.resize(cimg,(1200,800))
    cv2.imwrite(imagefile,cshrink)



#    cv2.imshow('detected area',cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print (cimg)

    return matrx



#dir = 'C:/Onizuka/150914'
dir = ''

name='150914-04.cdf_img01'


#http://sinhrks.hatenablog.com/entry/2015/07/26/232141



f=dir+'/' + name+ '.cdf'
ds_disk = xray.open_dataset(f)

pix, original_dimension_x, original_dimension_y ,NN=decode(ds_disk)

serial=convert_pix2serial(pix);

N1=original_dimension_y;

#matrix = serial2matrix(N1,NN,serial,int(original_dimension_x/8),name)
matrix = serial2matrix(N1,NN,serial,int(original_dimension_x),name)



