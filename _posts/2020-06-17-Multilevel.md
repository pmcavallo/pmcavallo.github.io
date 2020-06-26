---
layout: post
title: Hierarchical Models
---

This is an example of hierarchical, or multilevel, models using **STATA**:

The data for this project comes from the Finantial Times, BEA, and the Census (ACS), and the unit of analysis is Metropolitan Statistical Areas (MSAs) in the US. The dependent variable is greenfield foreign direct investment (FDI) (thousands of U$ Dollars). This greenfield data eliminates not only the liquid capital component but also Mergers & Acqusitions (M&A), leaving only the job-creating component city officials are eager to attract. The FDI data is at the project-level, allowing me to geocode the data at the county-level and then build the metro areas.

The idea in the project is to run a multilevel, or hierarchical, model to capture not only MSA-level characteristiocs that influence FDI inflows, but also control for state-level characteristics shared by some of the MSAs in the sample. In order to do so, this project will split the FDI data in sectors (manufacturing, sales, and services) and, using data from EMSI, a labor analytics company, build 2 variables to control for industry clusters. The goal is to test if industry clusters are magnets for new foreign investment coming into metro areas in the US. 

Content coming soon!
