# -*- coding: utf-8 -*-
"""
DataScience cs3753 Final project
Prupose:
    Examines a CSV file provided by the city of Austin that tracks all crimes
    committed throughout the year of 2016. This program tears it apart and examines
    each crime by cleaning and then examining for surface observations, correlation
    tendencies and clustering.
@author: Chandler V
"""
#%% I - IMPORTS & GLOBALS
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import numpy as np
import time
import copy

import gmplot    #Will require installation
from pyproj import Proj, transform    #Will require installation

FULLPRINT = False   #Setting to True will print extra information about the program
RUNLIST = ['SUMMARY', 'CORRELATION', 'CLUSTERING']
SAVEMPL = False

#%% II - SUPPORTING FUNCTIONS
#%%
"""     
createGoogleMap(lat, long, MapType, filename)
Purpose:
    This function takes in two arrays, and places them on a google map html
    link that plots information which is dependent on the MapType input. It will
    require an API Key supplied by Google for free off of the Google Cloud Platform.
Inputs:
    lat - Array containing all Latitudes desired to be plotted.
    long - Similar to lat, just with longitudes
    MapyType - accepts S for scatter, H for heatmap or B for both
    filename - straing containing the filename this will save to. Saves with .html
Outputs:
    One html file to be opened in an internet browser.
Notes:
  ->Requires an API key from Google. Their Cloud Platform allows for free keys to
    be used for programs like this with simple queries. In mass quantities however,
    it will charge you for usage. A good tutorial for getting one is right here:
    https://developers.google.com/maps/documentation/javascript/get-api-key
  ->Original Lat/Long arrays are in State PLane format, which is a mess to translate
    to Lat/Long. This is handled in inProj/outProj. lat/long must be in state plane
    when called though.
"""
def createGoogleMap(lat, long, MapType="S", filename="DEFAULT"):
    if FULLPRINT:
        print("Generating google map with plot points now...")
    try:
        T1 = time.process_time()
        #gmap3 is used to do the actual plotting for the google html file
        gmap3 = gmplot.GoogleMapPlotter(30.32106, 
                                    -97.74958, 12,
                                  apikey = 'AIzaSyCWbeUa9G8zEN5R8gqo_e9V7bIlzidg5B8') 
        inProj = Proj(init='epsg:2277', preserve_units = True)
        outProj = Proj(init='epsg:4326')
        x1,y1 = np.array(lat), np.array(long)   #Handles State plane to Lat/long
        x2,y2 = transform(inProj,outProj,x1,y1)
        if MapType.upper() == "S" or MapType.upper() == "B":    
            gmap3.scatter( y2,x2, '# FF0000', size = 50, marker = False ) 
        if MapType.upper() == "H" or MapType.upper() == "B":
            gmap3.heatmap(y2, x2)
        gmap3.draw(filename+ ".html" ) #Finish the file and save it to the cwd
        T2 = time.process_time()
        if FULLPRINT:
            print("Time to generate map: ",T2 - T1, "\nMap Complete")
    except:
        print("Error creating the Google map (%s)" % filename)

"""
createClusteredGoogleMap(data, labels, filename)
Purpose:
    An extension of the original googleMap function. This function focuses on taking
    a dataframe in, assigning it to color labels and then plotting those points to
    the google map. Doing this helps visualize the clusters/groupings of certain
    crimes.
Parameters:
    Data - dataframe containing the X & Y values designated to be plotted
    labels - an array containing the color ID assigned with the same plot point
    filename - a string with the chosen filename (will have HTML at end)
Notes:
    the actual clustering needs to be done outside of this function. Empty labels
    or missing labels will cause a crash. Also, DBSCAN takes years on this function.
"""    
def createClusteredGoogleMap(data, labels, filename):
    T1 = time.process_time()
    googData = data[['XCoord','YCoord']].copy()
    googData['Color'] = labels
    
    gmap4 = gmplot.GoogleMapPlotter(30.32106, 
                                        -97.74958, 12,
                                      apikey = 'AIzaSyCWbeUa9G8zEN5R8gqo_e9V7bIlzidg5B8') 
    inProj = Proj(init='epsg:2277', preserve_units = True)
    outProj = Proj(init='epsg:4326')
    
    colors = ['#ee340c','#3cee0c','#0ceaee','#0c0cee','#ee0ce3','#ee910c','#730cee','#ee0c73']
    
    for i in range(labels.max()+1):
        if FULLPRINT and (i < 10 or i%20==0):
            print("Generating clustering map -- %2.2f done" % (i/labels.max()))
        lat = googData[googData['Color'] == i].XCoord
        long = googData[googData['Color'] == i].YCoord
        x1,y1 = np.array(lat), np.array(long)   #Handles State plane to Lat/long
        x2,y2 = transform(inProj,outProj,x1,y1)
        if i < len(colors):
            gmap4.scatter( y2,x2, colors[i], size = 50, marker = False) 
        else:
            gmap4.scatter(y2, x2, matplotlib.colors.rgb_to_hsv(np.random.rand(len(y2),
                                        3)).all(), size=50, marker = False)    
    gmap4.draw(filename + ".html" ) #Finish the file and save it to the cwd 
    
    T2 = time.process_time()
    if FULLPRINT:
        print("Time to generate map: ",T2 - T1, "\nMap Complete")

"""
createScatterMPL(aX, aY, sTitle)
Purpose:
    Displays scatter plot using Matplotlib quickly. 
"""
def createScatterMPL(aX, aY, sTitle):
    try:
        plt.clf()
        plt.scatter(aX, aY, color='purple')
        xmin, xmax, ymin, ymax = plt.axis()
        plt.text(xmin, ymax-((ymax-ymin)/20), 'Total Points = %.1f' % aX.shape[0], fontsize = 10)
        plt.title(sTitle)
        if SAVEMPL:
            plt.savefig(sTitle + ' Scatter.png')
        plt.show()
    except:
        print("--Error within createScatterMPL--")
"""
createHeatScatterMPL(aX, aY, sTitle)
Purpose:
    Displays a scatter plot that increases color based on overlapped points.
"""
def createHeatScatterMPL(aX, aY, sTitle):
    T1 = time.process_time()
    plt.clf()
    z = np.vstack([X,Y])
    z = gaussian_kde(z)(z)
    fig, ax = plt.subplots()
    ax.scatter(X, Y, c=z, s=100, edgecolor='')
    xmin, xmax, ymin, ymax = plt.axis()    
    plt.text(xmin, ymax-((ymax-ymin)/20), 'Total Points = %.1f' % aX.shape[0], fontsize = 10)
    plt.title(sTitle)
    if SAVEMPL:
        plt.savefig(sTitle + ' HeatScatter.png')
    plt.show()
    T2 = time.process_time()
    if FULLPRINT:
        print("Time to generate Heat map: ",T2 - T1, "\nMap Complete")
"""
createBarMPL(aX, aY, sTitle, log=True)
Purpose:
    Creates a bar graph with custom settings. It plots two graphs, one which is
    unaltered in terms of scale, and one that is either a log scale or linear
    with the average subtracted from the values. Setting log to True will make
    the right one the log scale version, or False for the linear mean version.
"""
def createBarMPL(aX, aY, sTitle, log=True):
    plt.figure(figsize=(10, 8))
    plt.suptitle(sTitle)
    # Regular dist
    plt.subplot(221)
    plt.bar(aX, aY, color='Green')
    plt.title('Unedited Distribution')
    plt.xlabel("Total of crimes")
    plt.ylabel("District")
    plt.grid(True)
    
    # log
    if log:
        plt.subplot(222)
        plt.bar(aX, aY-aY.mean(), color= 'Green')
        plt.xlabel("Total of crimes")
        plt.yscale('log')
        plt.title('log')
        plt.grid(True)
    
    #Deviation
    else:
        plt.subplot(222)
        plt.bar(aX, aY-aY.mean(), color= 'Green')
        plt.xlabel("Total of crimes")
        plt.yscale('linear')
        plt.title('Deviation')
        plt.grid(True)
        
    if SAVEMPL:
        plt.savefig(sTitle + ' Bar.png')
    plt.show()

#%% III - CLEANING DATA
#%%

#Loading the csv file into a dataframe
data = pd.read_csv("2016_Annual_Crime_Data.csv")
MASTERDATA = copy.deepcopy(data)

#Renaming some columns to write them easier/understand
data.rename(columns = {'GO X Coordinate':'XCoord'}, inplace = True)
data.rename(columns = {'GO Y Coordinate':'YCoord'}, inplace = True)
data.rename(columns = {'GO Highest Offense Desc':'HighestOffenseDesc'}, inplace = True)
data.rename(columns = {'GO Report Date':'ReportDate'}, inplace = True)
data.rename(columns = {'GO District':'District'}, inplace = True)

#Count of rows lost through dropping all NaN values, We dont want incomplete crimes

before = data.shape[0]
data = data.dropna()
if FULLPRINT:
    print("Before cleaning: ", before, " After Cleaning: ", data.shape[0])

#Details about the column names and the current counts of each crime
if FULLPRINT:
    print("TOTALS OF ALL CRIME TYPES:\n", data['HighestOffenseDesc'].value_counts())

#Removing outliers based on geographics. We only want values that are downtown
X = data[data['XCoord'] < 16000000]
X = X['XCoord']
Y = data[data['YCoord'] < 16000000]
Y = Y['YCoord']

#Create an original scatter plot that shows us the entirety of all crimes
createScatterMPL(X, Y, "Original Unfiltered")

#Replace all crimes with THEFT in the name to just THEFT
rows = data['HighestOffenseDesc'].value_counts().index.values.tolist()
miscCondensed = []
for crime in rows:
    if 'THEFT' in crime:
        data.loc[data.HighestOffenseDesc == crime, 'HighestOffenseDesc'] = 'THEFT'
    elif 'AGG' in crime:
        data.loc[data.HighestOffenseDesc == crime, 'HighestOffenseDesc'] = 'AGG'
    elif 'ROBBERY' in crime:
        data.loc[data.HighestOffenseDesc == crime, 'HighestOffenseDesc'] = 'ROBBERY'
    elif 'BURG' in crime:
        data.loc[data.HighestOffenseDesc == crime, 'HighestOffenseDesc'] = 'BURGLARY'
    elif 'MURDER' in crime:
        data.loc[data.HighestOffenseDesc == crime, 'HighestOffenseDesc'] = 'MURDER'
    else:
        data.loc[data.HighestOffenseDesc == crime, 'HighestOffenseDesc'] = 'MISC'
        miscCondensed.append(crime.strip())
        
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
rows = data['ReportDate'].value_counts().index.values.tolist()
for row in rows: #Has all dates in it
    for month in months: #has all month names
        if month in row:
            data.loc[data.ReportDate == row, 'ReportDate'] = month

if FULLPRINT:
    print("AFTER REPLACEMENT:\n", data['HighestOffenseDesc'].value_counts())
    print("MISC CONTAINS: ", miscCondensed)
    
#%% IV - SUMMARY STATSTICS
#%%
if "SUMMARY" in RUNLIST:
    #Calculating the totals for each district in total
    tmp = data['District'].value_counts()
    createBarMPL(tmp.index, tmp, "Total Crimes per District")
    
    #Calculating NONviolent crimes per area
    subData1 = data[data['HighestOffenseDesc'].isin(['THEFT', 'BURGLARY', 'ROBBERY']) ]
    tmp2 = subData1['District'].value_counts()
    createBarMPL(tmp2.index, tmp2, "Nonviolent Crimes per District")
    
    #Calculating violent crimes per area
    subData2 = data[data['HighestOffenseDesc'].isin(['AGG', 'MURDER']) ]
    tmp3 = subData2['District'].value_counts()
    tmp3['AP'] = 0
    tmp3['88'] = 0
    createBarMPL(tmp3.index, tmp3, "Violent Crimes per District")
    
    print("Average crime per day: %f" % (data['ReportDate'].count()/365))
    print("Average crime per day per district:")
    print("\t Total - Nonviolent - Violent")
    for district in tmp.index:
        print("\t %2s: %2.f - %2.f - %2.f" % (district, tmp[district]/365, tmp2[district]/365, tmp3[district]/365))
    
    tmp = data['ReportDate'].value_counts()
    createBarMPL(tmp.index, tmp, "Crimes committed each month", False)

#%% V - CORRELATIONS
#%%
    
if "CORRELATION" in RUNLIST:
    districtMap = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'AP':10, '88':11}
    monthMap = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
                        'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    offenseMap = {'THEFT':1, 'BURGLARY':2, 'AGG':3, 'ROBBERY':4, 'MISC':5, 'MURDER':6}
    
    corrData = data[['ReportDate','HighestOffenseDesc','District']]
    corrData.replace({"ReportDate": monthMap, "District":districtMap,
                      "HighestOffenseDesc":offenseMap}, inplace=True)
    print(corrData)
    print("Spearman correlation examination: \n",corrData.corr(method='spearman'),"\n")
    print("Kendall correlation examination: \n",corrData.corr(method='kendall'),"\n")
    print("Pearson correlation examination: \n",corrData.corr(method='pearson'))


#%% VI - MAPPING & CLUSTERING
#%%
    
if "CLUSTERING" in RUNLIST:
    # Loop through each type of crime to see exactly how they look density wise.
    L = data['HighestOffenseDesc'].value_counts()
    rows = L.index.values.tolist()
    TotalTypesRemoved = 0
    for crime in rows:
        data2 = data[data.HighestOffenseDesc.str.contains(str(crime))]
        X1 = data2.XCoord
        Y1 = data2.YCoord
        createScatterMPL(X1, Y1, str(crime))
        
    # ADD/REMOVE TARGETED COLUMN NAMES HERE
    cols = ["THEFT","BURGLARY","AGG","MURDER", "ROBBERY",  "MISC"]
    from sklearn.cluster import KMeans
    # Iterate through and create images
    for i in cols:
        dataZ = data[data.HighestOffenseDesc.str.contains(str(i))]
        X = dataZ.XCoord
        Y = dataZ.YCoord
        filename = str(i) + " CLUSTER MAP"
        createHeatScatterMPL(X, Y, filename)

        #KMeans Clustering of the current crime type
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(dataZ[['XCoord','YCoord']])
        labels = kmeans.predict(dataZ[['XCoord','YCoord']])
        plt.scatter(dataZ['XCoord'], dataZ['YCoord'], c=labels, s=50, cmap='viridis')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
        plt.title("Kmeans Clustering of " + str(i))
        if SAVEMPL:
            plt.savefig(str(i) + ' kmeans.png')
        plt.show()
        createClusteredGoogleMap(dataZ, labels, filename + " - KMeans")
    
        #DBSCAN clustering of the current crime type
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=800, min_samples=3)
        dbscan.fit(dataZ[['XCoord','YCoord']])
        labels = dbscan.labels_
        plt.scatter(dataZ['XCoord'], dataZ['YCoord'], c=labels, s=50, cmap='viridis')
        plt.title("DBSCAN Clustering of " + str(i))
        if SAVEMPL:
            plt.savefig(str(i) + ' dbscan.png')
        plt.show()
        createClusteredGoogleMap(dataZ, labels, filename + " - DBSCAN")















