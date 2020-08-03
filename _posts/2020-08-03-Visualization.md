---
layout: post
title: Data Visualization and Animation with R
---

This example visualizes data with *ggplot2* in many different ways and employs animation techniques with *gganimation*.

1. The foreign direct investment (FDI) data for this example comes from the *Financial Times* and it is measured as the number of investment projects per year, at the state-level. 
The other variables employed are the population size and GDP per capita, from the BEA. First, I read in the data from a Stata file. 
Then we run a simple scatterplot to investigate the relationship between GDP per capita and FDI:

```R
library(foreign)
library(ggplot2)
library(ellipsis)
library(gganimate)
library(stringi)
library(gapminder)
library(gifski)

datapd <- read.dta(file.choose())

# We can have a simple scatterplot

scplot <- ggplot(datapd, aes(x=GDPpc, y=Projects)) + 
  geom_point(aes(col=RTW, size=Pop)) + 
  labs(subtitle="GDP per capita Vs FDI Projects", 
       y="FDI Projects", 
       x="GDP per capita", 
       title="Scatterplot", 
       caption = "Source: Financial Times")

plot(scplot)
```
![Contiguity Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/scatter.png?raw=true)
