---
layout: post
title: Webscraping with R
---

This example scrapes web data and cleans it using R's rvest and Tidyverse.
Here we will scrape the Wikipedia data table list of countries by external debt.

1. First, we install the the required packages:

    ```R
    install.packages(c("tidyverse", "ggplot2", "rvest", "magrittr")
    library(ggplot2)
    library(magrittr)
    library(rvest)
    library(tidyverse)

    ```

2. Second, we read  the HTML code from the Wiki website:

     ```R
    url <- 'https://en.wikipedia.org/wiki/List_of_countries_by_external_debt'
    wikiforreserve <- read_html(url)
    class(wikiforreserve)
     ```
*Obs: Prerequisites: Chrome browser, Selector Gadget*

3. We then get the XPath data using Inspect element feature in Safari, Chrome or Firefox. 
At Inspect tab, look for <table class=....> tag. Leave the table closed and 
right click the table and Copy the XPath, paste at html_nodes(xpath =):

    ```R
    externaldebt <- wikiforreserve %>%
    html_nodes(xpath='//*[@id="mw-content-text"]/div/table') %>%
    html_table()
    class(externaldebt)
    fores = externaldebt[[1]]
    ```

4. Next we clean up the variables. The date and debt variables, for instance, will go from this:

![Old Variable](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/debt1.PNG?raw=true)

to this:

![New Variable](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/debt3.PNG?raw=true)

using the following code:

    ```R
    library(stringr)
    fores$newdate = str_split_fixed(fores$Date, "\\[", n = 2)[, 1]
    fores$newdebt = str_split_fixed(fores$`External debtUS dollars`, "\\×", n = 5)[, 1]
    ```
    
5. We can then, for instance, convert the "newdebt" variable to numeric and return it to its original measure:

    ```R
    debt <- as.numeric(fores$newdebt)
    fores$newdebt2 <- debt * (10^12)
    head(fores$newdebt2)
    [1] 2.541271e+12 8.475956e+12 4.974278e+12 4.510400e+12 3.586817e+12 2.785709e+12
    ```
