# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:37:50 2022

@author: says
"""



#Cluster project for DALYs in fruits
import os
import pandas as pd
import numpy as np
import re
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
# from scipy import optimize

def lognorm_from_percentiles(x1, p1, x2, p2):
    """ Return a log-normal distribuion X parametrized by:
    
            P(X < p1) = x1
            P(X < p2) = x2
    """
    x1 = np.log(x1)
    x2 = np.log(x2)
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)
    
    scale = (x2 - x1) / (p2ppf - p1ppf)
    mean = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    
    return lognorm(s=scale, scale=np.exp(mean)), mean
def norm_from_percentiles(x1, p1, x2, p2):
    """ Return a normal distribuion X parametrized by:
    
            P(X < p1) = x1
            P(X < p2) = x2
    """
    p1ppf = norm.ppf(p1)
    p2ppf = norm.ppf(p2)

    location = ((x1 * p2ppf) - (x2 * p1ppf)) / (p2ppf - p1ppf)
    scale = (x2 - x1) / (p2ppf - p1ppf)

    return norm(loc=location, scale=scale), location


os.chdir('C:\\Users\\says\\Projects\\MLcluster_project\\data')

data = pd.read_csv('consumption_alex.csv',  sep=";", header = 0, index_col=False, float_precision= "round_trip",decimal=",")
#data = pd.read_excel('consumption_alex.xlsx',  header = 0, index_col=False)

data1= pd.read_csv('cluster_exp.csv',  sep=",", header = 0, index_col=False, float_precision= "round_trip")
data["cluster"] = 100
data["age"] = 100

for i in range(len(data)):
    if (data.Ipnr[i] in data1['ID'].values):
        ind = data1.index[data1['ID'].values == data.Ipnr[i]][0]
        print(ind)
        data["cluster"][i] = data1.cluster[ind]
        data["age"][i] = data1.age[ind]
    
#Sort data with clusters
data = data.sort_values(by=['cluster'])        
data = data.reset_index(drop=True)
data.drop(data[data.cluster == 100].index, inplace=True)

#Change colnames of LT to X
colum=[]
x=data.columns

for i in range(len(x)):
    if('LTNR' in x[i]):
        colum.append(i)
        a=int(re.split('LTNR',x[i])[1])
        b="X" + str(a)
        data.rename(columns={x[i]: b}, inplace=True)

        
#Select only meat
data2= pd.read_table('C:\\Users\\says\\Projects\\MLcluster_project\\RE__LT_codes_for_food_groups\\meat.txt', delimiter = ' ', header = None)
colum_names=[]
for i in range(len(data2)):
    colum_names.append(data2.values[i][0])
#colum_names.remove('X149')   # Remove any column that has zero consumption
Final_data = data[colum_names]



# for i in colum_names:
#     Final_data[i] = [x.replace(',', '.') for x in Final_data[i]]
#     Final_data[i] = Final_data[i].astype(float)
     

#probabilistic Dose Response


def dose_response_meat_t2d(x):
    # Dose response = 0.98(0.97,1.00)
    dist1, mean = lognorm_from_percentiles(1.08, 0.025, 1.26, 0.975)
    X=[[0], [100] ]
    Y=[[1], [dist1.rvs(1)[0]]]
    Y=np.log(Y)
    model3=linear_model.LinearRegression()
    model3.fit(X,Y)
    if x <= 170:
        rr_pred = np.exp(model3.predict([[x]]))
    else:
        rr_pred = np.exp(model3.predict([[170]]))
    
    return rr_pred, model3.coef_



#Calculate mean of each clusters
Final_data = Final_data.replace(0, np.NaN) 
Final_data['mean'] = Final_data.iloc[:, 0:61].sum(axis=1)
Final_data['cluster'] = data.cluster
Final_data['age'] = data.age

df= Final_data.groupby(['cluster'])['mean'].mean()  #mean intake of each cluster
df_std = Final_data.groupby(['cluster'])['mean'].std()

np.savetxt(r'C:\\Users\\says\\Projects\\MLcluster_project\\mean_meat.txt', df_std, fmt='%.2f\t')

#DALY Calculations


daly_t2d=[0]*100
daly_t2d_upper=[0]*100
daly_t2d_lower=[0]*100


data5 = pd.read_excel('C:\\Users\\says\\Projects\\MLcluster_project\\DALY_rate_health.xls', sheet_name=5, header = 0)


for i in range(100):
    if(i<15):
        daly_t2d[i] = 0
        daly_t2d_upper[i] = 0
        daly_t2d_lower[i] = 0
    if(i <= 19 and i >= 15):
        daly_t2d[i] = data5['DALY/100K'][0]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][0]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][0]
    if(i <= 24 and i >= 20):
        daly_t2d[i] = data5['DALY/100K'][1]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][1]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][1]
    if(i <= 29 and i >= 25):
        daly_t2d[i] = data5['DALY/100K'][2]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][2]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][2]
    if(i <= 34 and i >= 30):
        daly_t2d[i] = data5['DALY/100K'][3]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][3]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][3]
    if(i <= 39 and i >= 35):
        daly_t2d[i] = data5['DALY/100K'][4]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][4]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][4]
    if(i <= 44 and i >= 40):
        daly_t2d[i] = data5['DALY/100K'][5]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][5]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][5]
    if(i <= 49 and i >= 45):
        daly_t2d[i] = data5['DALY/100K'][6]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][6]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][6]
    if(i <= 54 and i >= 50):
        daly_t2d[i] = data5['DALY/100K'][7]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][7]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][7]
    if(i <= 59 and i >= 55):
        daly_t2d[i] = data5['DALY/100K'][8]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][8]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][8]
    if(i <= 64 and i >= 60):
        daly_t2d[i] = data5['DALY/100K'][9]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][9]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][9]
    if(i <= 69 and i >= 65):
        daly_t2d[i] = data5['DALY/100K'][10]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][10]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][10]
    if(i <= 74 and i >= 70):
        daly_t2d[i] = data5['DALY/100K'][11]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][11]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][11]
    if(i <= 79 and i >= 75):
        daly_t2d[i] = data5['DALY/100K'][12]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][12]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][12]
    if(i <= 84 and i >= 80):
        daly_t2d[i] = data5['DALY/100K'][14]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][14]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][14]
    if(i <= 89 and i >= 85):
        daly_t2d[i] = data5['DALY/100K'][15]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][15]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][15]
    if(i <= 94 and i >= 90):
        daly_t2d[i] = data5['DALY/100K'][16]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][16]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][16]
    if(i >= 95):
        daly_t2d[i] = data5['DALY/100K'][17]
        daly_t2d_upper[i] = data5['Upper DALY/100K'][17]
        daly_t2d_lower[i] = data5['Lower DALY/100K'][17]
        

#Daly Simulation
def daly():

    Final_data['Daly_t2d'] = ""     
    for i in range(len(Final_data)):

    
        dist3, mean = norm_from_percentiles(daly_t2d_lower[Final_data.age[i]] , 0.025, daly_t2d_upper[Final_data.age[i]], 0.975)                
        Final_data['Daly_t2d'][i] = dist3.rvs(1)[0]
        
    df4= Final_data.groupby(['cluster'])['Daly_t2d'].apply(np.mean).reset_index()
    return df4
    


#simulate montecarlo impact fractions
    
# Impact Fraction Calculation

impact_fraction_t2d_simulated = []
impact_fraction_t2d_mean =[]
impact_fraction_t2d_std=[]

    

      

l=pd.Series()
df_mean_simulation=[]
df_std_simulation = []
for i in range(50):

    impact_fraction_t2d_simulated=[]
    for j in range(12):

        #Calculate Impact Fraction

        c= (dose_response_meat_t2d(np.mean(np.random.normal(df[j], df_std[j], 100)))[0][0][0] -  dose_response_meat_t2d(15)[0][0][0])/dose_response_meat_t2d(np.mean(np.random.normal(df[j], df_std[j], 1000)))[0][0][0] 
        impact_fraction_t2d_simulated.append(c)
    impact_fraction_t2d_simulated = [0 if x < 0 else x for x in impact_fraction_t2d_simulated]

    #Daly Simulation
    f=daly()

    l=  pd.concat([l, f['Daly_t2d']*impact_fraction_t2d_simulated],axis=1)
    print(i)


l.mean(axis=1)  
            
np.savetxt(r'C:\\Users\\says\\Projects\\MLcluster_project\\Daly_meat_probabilistic_new.txt', list(zip(l.mean(axis=1),l.std(axis=1))), fmt='%.2f\t')


# #Plot common burden
# temp = [0,1,2,3,4,5,6,7,8,9,10,11] 
# t2d = [7.17,0.06,9.69,98.40,4.44,13.99,19.82,41.27,101.83,32.63,16.18,54.70] 
# data6= pd.read_csv('C:\\Users\\says\\Projects\\MLcluster_project\\RE__LT_codes_for_food_groups\\chemcial burden.csv',  sep=";", header = 0, index_col=False, float_precision= "round_trip")
# fig=plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
# plt.bar(temp,t2d , color='red')

# plt.bar(temp, data6['mehg'].values, bottom=t2d , color='green')
# plt.bar(temp, data6['cd'],bottom=t2d +data6['mehg'], color='cyan')
# plt.bar(temp, data6['as'],bottom=t2d +data6['mehg'] + data6['cd'], color='yellow')
# plt.xlabel("Cluster")
# plt.ylabel("Cumulative DALY/100k")
# plt.legend(["T2D","MeHg","Cd","As"])
# plt.title("DALY/100k for Meat")

#plt.savefig('C:\\Users\\says\\Projects\\MLcluster_project\\RE__LT_codes_for_food_groups\\Meat.jpg')







#Calculate the uncertainity bands

#np.add(impact_fraction_t2d_mean, np.multiply(impact_fraction_t2d_std,2))
#np.subtract(impact_fraction_t2d_mean, np.multiply(impact_fraction_t2d_std,2))   
#df4['Daly_t2d']  
#
#fig=plt.figure(num=None, figsize=(6, 4), dpi=150, facecolor='w', edgecolor='k')
#plt.title('Daly Uncertainity for Fruits and T2D')
#plt.xlabel('Cluster number')
#plt.ylabel('DALY/100000k')
#plt.scatter(temp, df4['Daly_t2d']* impact_fraction_t2d_mean)
#yerr = df4['Daly_t2d']* np.add(impact_fraction_t2d_mean, np.multiply(impact_fraction_t2d_std,2)) -  df4['Daly_t2d']* impact_fraction_t2d_mean
#plt.errorbar(temp, df4['Daly_t2d']* impact_fraction_t2d_mean, yerr=yerr, fmt="o")
#plt.savefig('D:\\MLcluster_project\\Daly_uncertainity_fruits_t2d.jpg') 
#    


