---
layout: post
title: Hierarchical Models
---

This is an example of hierarchical, or multilevel, models using **STATA**:

The data for this project comes from the Finantial Times, EMSI, and the Census Bureau (ACS), and the unit of analysis is Metropolitan Statistical Areas (MSAs) in the US. The dependent variable is greenfield foreign direct investment (FDI) (thousands of U$ Dollars). This greenfield data eliminates not only the liquid capital component but also Mergers & Acqusitions (M&A), leaving only the job-creating component city officials are eager to attract. The FDI data is at the project-level, allowing me to geocode the data at the county-level and then build the metro areas.

The idea in the project is to run a multilevel, or hierarchical, model to capture not only MSA-level characteristiocs that influence FDI inflows, but also control for state-level characteristics shared by some of the MSAs in the sample. In order to do so, this project will split the FDI data in sectors (manufacturing, sales, and services) and, using data from EMSI, a labor analytics company, build 2 variables to control for industry clusters. The goal is to test if industry clusters are magnets for new foreign investment coming into metro areas in the US. 

The project will employ multilevel logistic models, therefore the dependent variable is a dummy with values of 1 if the MSA received an FDI project in a specific sector that year, 0 otherwise. Industry clusters are measured in 2 ways: the first uses the location quotients of manufacturing, services, and sales in each MSA in comparison with all the other MSAs in the country. The second operationalization is the natural logarithm of the number of payrolled business in manufacturing, services, and sales in a given MSA divided by the total number of payrolled businesses in it. Data for both measures come from EMSI.

First, since this is a panel data structure, we have to check if it is more appropriate to use random or fixed-effects. In STATA this is done as follows:

```{stata}
sysuse auto
summarize
```


Content coming soon!
