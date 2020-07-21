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

4. Next we clean up the variables. The date variable, for instance, will go from this:

![Old Variable](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/debt1.png)

to this:

![New Variable](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/debt2.png)

using the following code:

    ```R
    library(stringr)
    fores$newdate = str_split_fixed(fores$Date, "\\[", n = 2)[, 1]
    ```
