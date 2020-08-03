---
layout: post
title: Data Visualization and Animation with R
---

This example visualizes data with *ggplot2* in many different ways and employs animation techniques with *gganimation*.

1. The foreign direct investment (FDI) data for this example comes from the *Financial Times* and it is measured as the number of investment projects per year, at the state-level. 
The other variables employed are the population size and GDP per capita, from the BEA. I subsequently generate a dummy variable to identify right-to-work (RTW) states, meaning states that have RTW laws that prohibit union security agreements between employers and labor unions. The general hypothesis is that RTW states should, in theory, receive more FDI as they have a more "business friendly" environment. The idea is to investigate this hypothesis **visually**. 
First, I read in the data from a Stata file. Then we run a simple scatterplot to investigate the relationship between GDP per capita and FDI on RTW and non-RTW states:

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
![Scatterplot 1](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/scatter.png?raw=true)

The main problems with this data is that the cluster at the far right side of the plot is making it difficult to visualize the rest of the data. If we further investigate the cluster we see that these data points, which show a very high GDP per capita and low population size, belong to Washington-DC. We then proceed to eliminate the outlier (DC) from the data and generate the same plot again:

```R
datapd2 <- read.dta(file.choose())

scplot2 <- ggplot(datapd2, aes(x=GDPpc, y=Projects)) + 
  geom_point(aes(col=RTW, size=Pop)) + 
  labs(subtitle="GDP per capita Vs FDI Projects", 
       y="FDI Projects", 
       x="GDP per capita", 
       title="Scatterplot", 
       caption = "Source: Financial Times")

plot(scplot2)
```
![Scatterplot 2](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/scatter2.png?raw=true)

Now we can better visualize the data and investigate the relationship it illustrates. For instance, at first look, it seems non-RTW states have higher GDP per capita and receive more FDI. It also seems populous states (larger circles) receive more FDI, but not necessarily richer states (higher GDP per capita). We can also employ an *encircling* techinque from the *ggalt* package to highlight the data points with high FDI inflows:

```R
library(ggalt)

fdi_select <- datapd2[datapd2$Projects > 100 & 
                            datapd2$GDPpc > 50000 & 
                            datapd2$GDPpc < 65000,]

scplot3 <- ggplot(datapd2, aes(x=GDPpc, y=Projects)) + 
  geom_point(aes(col=RTW, size=Pop)) + 
  geom_encircle(aes(x=GDPpc, y=Projects), 
                data=fdi_select, 
                color="red", 
                size=2, 
                expand=0.08) +   
  labs(subtitle="GDP per capita Vs FDI Projects", 
       y="FDI Projects", 
       x="GDP per capita", 
       title="Scatterplot", 
       caption = "Source: Financial Times")

plot(scplot3)

```
![Scatterplot 3](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/scatter3.png?raw=true)

Another interesting plot we can explore is the **Diverging Bars**, which can handle both negative and positive values. The idea is to see which states are above or below the average number of FDI projects in all states. This means we will have to prepare the data before we run the code for the plot:
```R
datapd2$Projects_z <- round((datapd2$Projects - mean(datapd2$Projects))/sd(datapd2$Projects), 2)  # compute normalized FDI Projects
datapd2$ab_bel <- ifelse(datapd2$Projects_z< 0, "below", "above")  # above / below avg FDI
datapd2 <- datapd2[order(datapd2$Projects_z), ]  # sort
```

And then we can obtain the diverging bars as follows:

```R
  ggplot(datapd2, aes(x=State, y=Projects_z, label=Projects_z)) + 
  geom_bar(stat='identity', aes(fill=datapd2$ab_bel), width=.5)  +
  scale_fill_manual(name="FDI Projects", 
                    labels = c("Above Average", "Below Average"), 
                    values = c("above"="#00ba38", "below"="#f8766d")) + 
  labs(subtitle="Normalised FDI Projects", 
       title= "Diverging Bars") + 
  coord_flip()
```
![Scatterplot 3](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/diver_bar.PNG?raw=true)

As expected, we see that the large states of California, Texas, and New York receive, by far, the largest amount of investment in the US. In order to better see the ranking of the states that receive the most FDI, we can also order the data and plot an *ordered bar chart*:

```R
  mean_state <- aggregate(datapd2$Projects, by=list(datapd2$State), FUN=mean)  # aggregate by year
  mean_state <- mean_state[order(mean_state$x), ]  # sort
  mean_state$Group.1 <- factor(mean_state$Group.1, levels = mean_state$Group.1)  # to retain the order in plot.
  
  ggplot(mean_state, aes(x=Group.1, y=x)) + 
    geom_bar(stat='identity', width=.5, fill="tomato3")  +
    labs(title="Ordered Bar Chart", 
         subtitle="State's Avg. FDI Projects",
         y="FDI Projects", 
         x="States", 
         caption="source: Financial Times") + 
    theme(axis.text.x = element_text(angle=65, vjust=0.6))

```
![Ordered bar chart](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/ord_bar.png?raw=true)

This plot clearly illustrates the gradual increase in FDI inflows between the states and the difference between them and the three most important FDI destinations in the country; California, Texas, and New York.

Going back to our original investigation, we can generate a time-series plot differentiating between RTW and non-RTW states to see their differences over time. Since this is actually panel data, we can aggregate FDI by RTW and year before generating the plot:

```R
rtw <- aggregate(datapd2$Projects, by=list(datapd2$RTW, datapd2$Year), FUN=mean)  # aggregate by RTW and year


ggplot(rtw, aes(x=Group.2, y=x)) + 
  geom_line(aes(color = as.factor(Group.1)), size = 1) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  labs(title="Time Series Plot", 
       subtitle="FDI by RTW States",
       y="FDI Projects", 
       x="Year",
       color = "RTW",
       caption="source: Financial Times") + 
  theme_minimal()

```
![time-series plot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/time.png?raw=true)

The plot suggests non-RTW states have increased their share of FDI since 2003. Ee can also see the same information as an **area plot** instead:

```R
ggplot(rtw, aes(x=Group.2, y=x)) + 
  geom_area(aes(fill = as.factor(Group.1)), 
            alpha = 0.5) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
  labs(title="Stacked Area Chart", 
       subtitle="FDI by RTW States",
       y="FDI Projects", 
       x="Year",
       fill = "RTW",
       caption="source: Financial Times") 
```
![time-series plot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/stack.png?raw=true)
       
The area plot better illustrates the increasing difference between non-RTW and RTW states. Overall, the plots suggest non-RTW states attract more FDI, which goes against the expectation that the more *business friendly* RTW states would be more attractive to investment. 

And lastly, as the *pièce de résistance*, we can run an animation with **gganimate** to see the differences in RTW and non-RTW states evolving over time since 2003:
