---
layout: post
title: Spatial Data
---

This example uses geocoded foreign direct investment (FDI) data to analyze and control for spatial correlation:

The data for this project comes from the *Finantial Times, BEA, and EMSI*, and the unit of analysis is Metropolitan Statistical Areas (MSAs) in the US. The dependent variable is greenfield foreign direct investment (FDI) (thousands of U$ Dollars). This greenfield data eliminates not only the liquid capital component but also Mergers & Acqusitions (M&A), leaving only the job-creating component city officials are eager to attract. The FDI data is at the project-level, allowing me to geocode the data at the county-level and then build the metro areas.

The main idea of this project, first, is to investigate if FDI clusters geographcally, meaning, do some MSAs’ outcomes depend on the outcomes in other MSAs? In other words, is FDI contagious from MSAs to neighboring or otherwise proximate MSAs? For instance, are Baltimore and Philadelphia receiving more investment than Dallas and Denver due to their proximity to major FDI destinations such as NYC, Boston, and DC?

The first step is to check if there is spatial autocorrelation in the FDI data. To that end, I first build a spatial weights matrix, the heart of any spatial analyses, where the elements are defined to reflect the suspected nature of the hypothesized spatial relationship between the units. The matrix is composed of elements connecting one metro area to every other metro area in the sample, and reflect the strength of the dependence between them. Distance is measured as the geographic distance between the MSAs defined by the great-circle distance in miles between one MSA to the other.

MORE COMING SOON...
