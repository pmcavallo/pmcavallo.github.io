---
layout: post
title: Spatial Data With ArcMap and R
---

This example uses geocoded (ArcMap) foreign direct investment (FDI) data to analyze and control for spatial autocorrelation in R:

The data for this project comes from the *Finantial Times, US Department of Transportation, BEA, and EMSI*, and the unit of analysis is US states. The dependent variable is greenfield foreign direct investment (FDI) (millions of U$ Dollars). This greenfield data eliminates not only the liquid capital component but also Mergers & Acqusitions (M&A), leaving only the job-creating component state officials are eager to attract. 

The main idea of this project is to investigate if FDI clusters geographically, meaning, do some states’ outcomes depend on the outcomes in other states? In other words, is FDI contagious from states to neighboring or otherwise proximate states? For instance, are Connecticut and New Hampshire receiving more investment than Texas and Colorado due to their proximity to major FDI destinations such as the states of New York and Massachusetts?

The first step is to obtain a shapefile of the US states. Then have the FDI excel file in wide format, meaning each year for the FDI inflow becomes a column (variable). Open the US state shapefile and the FDI CSV file in ArcMap and join them by a common variable (i.e. state Fips code, geocode, USPS). Finally, you export it back as a shapefile to open in R.

Then, you open the shapefile in R and obtain the states' coordinates (centroids) as follows:

```R
# LIBRARIES
library(maps)
library(maptools)    #to read the shapefile created in ArcMap
library(sp)          #this is the package for spatial weight matrix
library(spdep)       #this is the package for spatial weight matrix
library(RColorBrewer)
library(classInt)
library(rgdal)
library(ggplot2)
library(dplyr)
library(foreign)
library(Hmisc)
library(plm)
library(splm)        #this is the package for spatial econometrics modeling
library(fields)      #to calculate distance between the units
library(lmtest)
library(stats)



### US-STATES SHAPEFILE ###
statemap<-readShapePoly("<your file path here>/Export_Output_2.shp",IDvar="GEOID",proj4string=CRS("+proj=longlat +ellps=WGS84"))

map.centroid<-coordinates(statemap)   

```

The next step is to check if there is spatial autocorrelation in the FDI data. To that end, I first build a spatial weights matrix, the heart of any spatial analyses, where the elements are defined to reflect the suspected nature of the hypothesized spatial relationship between the units. The matrix is composed of elements connecting one state to every other state in the sample, and reflect the strength of the dependence between them. There are three ways to calculate the spatial weights matrix: contiguity (queen and rook), k nearest neighbors, and inverse distance. Distance is measured as the geographic distance between the states' centroids defined by the great-circle distance in miles between one state to the other.

The contiguity weight matrix (rook and queen), is calculated as follows:

```R
map.link <- poly2nb(statemap,queen=T)
map.linkr <- poly2nb(statemap,queen=F)
nbweights.lw <- nb2listw(map.link, style="W", zero.policy=T)   #creating a matrix with the neighborhood object
```

In the code above, *style="W"* to is used to generate a row-normalized (or standardized) matrix, and *zero.policy=T* allows matrices to be computed even if there are "islands", meaning, no-neighbor areas. The "rook" defines neighbors by the existence of a common edge between two spatial units. The "queen" is somewhat more encompassing and defines neighbors as spatial units sharing a common edge or a common vertex. The map with the connectivity (queen) between the states looks like this:

```R
plot(statemap,border="blue",axes=TRUE,las=1)
plot(map.link,coords=map.centroid, pch=19,cex=0.1,col="red",add=T)
title("Contiguity Spatial Links Among States")                 
```
![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/queen2.png?raw=true)

The k nearest neighbor spatial weight matrix is generated as follows:

```R
col.knn <- knearneigh(mycoords,k=5,longlat=T,RANN=TRUE)
plot(statemap,border="blue",axes=TRUE,las=1)
plot(knn2nb(col.knn), mycoords, pch=19, cex=0.1,col="red",add=T)

```
Where the number of neighbors (k) was set to five, and the connectivity looks like this:

![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/Neighbors.png?raw=true)

The inverse distance weigth matrix is calculated as follows:

```R
mydm<-rdist.earth(mycoords)              # computes distance in miles. 
for(i in 1:dim(mydm)[1]) {mydm[i,i] = 0} # renders exactly zero all diagonal elements
mydm[mydm > 500] <- 0                    # all distances > 500 miles are set to zero
mydm<-ifelse(mydm!=0, 1/mydm, mydm)      # inverting distances
mydm.lw<-mat2listw(mydm, style="W")      # create a (normalized) listw object
```
It's called inverse distance because the weight decreases as distance increases from the points *(w = 1/d)*. In this case, we set the minimun distance between the points to be 500 miles. And the connectivity between the states looks like this:

```R
plot(statemap,border="blue",axes=TRUE,las=1)
plot(mydm.lw,coords=mycoords, pch=19, cex=0.1,col="red",add=T)
title("Inverse Distance Spatial Links Among States")  
```

![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/inverse.png?raw=true)

We then calculate the Moran's I, which is a correlation coefficient that measures the overall spatial autocorrelation of the data. In this case, we are going to check the spatial autocorrelation in FDI at the state-level using the inverse distance spatial matrix.

```R
moran.mc(statemap$FDI2003,mydm.lw,nsim=9999)
```
And the output is something like this:

![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/moran.PNG?raw=true)

Showing spatial autocorrelation for FDI in 2003. Below is a table with the results for some of the years of the data:

![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/moran2.PNG?raw=true)

The results show some significant correlation coefficients. 

We then bring in some variables to control for the quality of a state's market (GDP per capita from BEA), the size of the state (population size from BEA), the level of human capital/skill (education level from EMSI), and the quality of a state's infrastructure (road mileage from the US Departmet of Transportation). Next, we format the data set as panel data:

```R
data <- read.dta(file.choose())
datapd<-pdata.frame(data,index=c("StateID","Year"))
```

Next we test for fixed or random effects in panel data (Hausman test) and employ a couple of Lagrange Multiplier tests for spatial error and/or spatial lag dependence. Depending on what these LM tests find, we run a Spatial Autoregressive model (SAR), a Spatial Error model (SEM), or a combined model (SARAR).

```R
model <- log(FDI) ~ log(Pop) + log(GDPpc) + log(College) + log(Mileage)

fe <- plm(model, data=datapd, model = "within")      #fixed-effects model
re <- plm(model, data=datapd, model = "random")      #random-effects model
phtest(re, fe)                                       #Hausman Test

# LM tests for spatial lag correlation in panel models
slmtest(model, data=datapd, listw=mydm.lw, test="lml") 
```
![spatial lag](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/lm1.PNG?raw=true)

```R
# LM test for spatial error correlation in panel models
slmtest(model, data=datapd, listw=mydm.lw, test="lme")

```
![spatial error](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/lm2.PNG?raw=true)

The test statistics for the Hausman Test is 6.2418 with a p-value of 0.1818, providing evidence for the random effects model. Both Lagrange Multiplier tests also show spatial error and spatial lag dependence in the data. We therefore run a SARAR random-effects model to control for both dependence:

```R
sarar <- spml(model,data=datapd,index=NULL,listw=mydm.lw,model="random",lag=TRUE, spatial.error="kkp",LeeYu=T)
summary(sarar)
```
![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/reg.PNG?raw=true)

Where *model="random"* establihes our random-effects model, *lag=TRUE* ensures the spatial lag of the DV is included, *spatial.error="kkp"* for the KKP-style specification of the spatial error (Kapoor et al. 2007), and *LeeYu=T* allows to transform the data according to Lee and Yu (2010). 

The control variables show the expected direction with the exception of GDP per capita, which is not significant, suggesting FDI is not necessarily attracted to rich states. Population, road mileage, and education are all positive and significant, suggesting FDI is attracted to larger states with high-skilled labor and a good infrastucture. The most important finding is the lambda coefficient, which captures the spatial lag of the dependent variable. 

The lambda coefficient is positive and highly signficant, suggesting there is spatial autocorrelation in FDI, meaning states that are closer to states receiving a large inflow of FDI will receive more FDI.  This provides support for the conjecture that a state’s level of FDI inflows covaries with
the level of FDI inflows among its geographical neighbors.

As a robustness check, and because the rho coefficient of the model shows very weak evidence of a panel regression with spatially correlated errors, we can also run a Spatial Autoregressive model (SAR). Additionally, in a parsimonious modelling approach, the original test statistics (LM for spatial lag or error) with the higher significance could guide the choice of the spatial regression model.

```R
m.lag <-spml(model,data=datapd,index=NULL,listw=mydm.lw,model="random",lag=TRUE, spatial.error="none")
summary(m.lag)
```
![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/reg2.PNG?raw=true)

And the results hold for all variables as well as the lambda coefficient.

Another robustness check could be the same regression model (SARAR) using a contiguity (queen) spatial weight matrix:

```R
m.queen <- spml(model,data=datapd,index=NULL,listw=nbweights.lw,model="random",lag=TRUE, spatial.error="kkp",LeeYu=T)
summary(m.queen)
```
![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/reg3.PNG?raw=true)

And again, the results hold for all variables as well as the lambda coefficient.

So as a final robustness check I will run the same regression model with a *k nearest neighbors* spatial weight matrix. 
 
 ```R
W <- spdep::nb2mat(spdep::knn2nb(spdep::knearneigh(mycoords, k=5,longlat=TRUE))) 
knn.lw<-mat2listw(W, style="W")

m.neigh <- spml(model,data=datapd,index=NULL,listw=knn.lw,model="random",lag=TRUE, spatial.error="kkp",LeeYu=T)
summary(m.neigh)
 ```
 ![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/reg4.PNG?raw=true)
 
The results again hold for all variables as well as the lambda coefficient.
 

