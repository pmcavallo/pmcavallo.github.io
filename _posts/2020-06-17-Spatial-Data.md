---
layout: post
title: Spatial Data
---

This example uses geocoded (ArcMap) foreign direct investment (FDI) data to analyze and control for spatial autocorrelation in R:

The data for this project comes from the *Finantial Times, BEA, and EMSI*, and the unit of analysis is US states. The dependent variable is greenfield foreign direct investment (FDI) (millions of U$ Dollars). This greenfield data eliminates not only the liquid capital component but also Mergers & Acqusitions (M&A), leaving only the job-creating component state officials are eager to attract. 

The main idea of this project, first, is to investigate if FDI clusters geographically, meaning, do some states’ outcomes depend on the outcomes in other states? In other words, is FDI contagious from states to neighboring or otherwise proximate states? For instance, are Connecticut and New Hampshire receiving more investment than Texas and Colorado due to their proximity to major FDI destinations such as the states of New York and Massachusetts?

The first step is to obtain a shapefile of the US states. Then have the FDI excel file in wide format, meaning each year for the FDI inflow becomes a column (variable). Open the US state shapefile and the FDI CSV file in ArcMap and joing them by a common variable (i.e. state Fips code, geocode, USPS). Finally, you export it back as a shapefile to open in R.

Then, you open the shpefile in R and obtain the states' coordinates (centroids) as follows:

```R
# LIBRARIES
library(maps)
library(maptools)
library(sp)
library(spdep) #this is the package for spatial econometrics modeling
library(RColorBrewer)
library(classInt)
library(rgdal)
library(ggplot2)
library(dplyr)
library(spacetime)
library(foreign)
library(Hmisc)
library(plm)
library(gplots)
library(psych)
library(pastecs)
library(car)
library(splm)
library(fields)
library(stargazer)
library(lmtest)
library(stats)
library(DataCombine)


### US-STATES SHAPEFILE ###
statemap<-readShapePoly("D:/DataScience/Summer2020/Project/Export_Output_2.shp",IDvar="GEOID",proj4string=CRS("+proj=longlat +ellps=WGS84"))

map.centroid<-coordinates(statemap)   

```

The next step is to check if there is spatial autocorrelation in the FDI data. To that end, I first build a spatial weights matrix, the heart of any spatial analyses, where the elements are defined to reflect the suspected nature of the hypothesized spatial relationship between the units. The matrix is composed of elements connecting one metro area to every other metro area in the sample, and reflect the strength of the dependence between them. There are two ways to calculate the spatial weights matrix: contiguity (queen and rook), and inverse distance. In the inverse distance case, distance is measured as the geographic distance between the states' centroids defined by the great-circle distance in miles between one state to the other.

The contiguity weigth matrix (rook and queen ), is calculated as follows:

```R
map.link <- poly2nb(statemap,queen=T)
map.linkr <- poly2nb(statemap,queen=F)

```

The map with the connectivity (queen) between the states looks like this:
```R
plot(statemap,border="blue",axes=TRUE,las=1)
plot(map.link,coords=map.centroid, pch=19,cex=0.1,col="red",add=T)
title("Contiguity Spatial Links Among States")                 
```
![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/queen2.png?raw=true)

The inverse distance weigth matris is calculated as follows:
```R
mydm<-rdist.earth(mycoords)              # computes distance in miles. If true distances are in statute miles if false distances in kilometers. 
for(i in 1:dim(mydm)[1]) {mydm[i,i] = 0} # renders exactly zero all diagonal elements
mydm[mydm > 500] <- 0                    # all distances > 500 miles are set to zero
mydm<-ifelse(mydm!=0, 1/mydm, mydm)      # inverting distances
mydm.lw<-mat2listw(mydm, style="W")      # create a (normalized) listw object
```
And the connectivity between the states looks like this;

```R
plot(statemap,border="blue",axes=TRUE,las=1)
plot(mydm.lw,coords=mycoords, pch=19, cex=0.1,col="red",add=T)
title("Inverse Distance Spatial Links Among States")  
```

![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/inverse.png?raw=true)


