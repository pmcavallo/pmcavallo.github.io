---
layout: post
title: Twitter Data
---

This example scrapes Twitter data, visualizes it, and looks at some descriptive information:

1. First, we install the *rtweet* package:

```R
install.packages("rtweet")

library(ggplot2)
library(rtweet)
library(igraph)
library(tidyverse)
library(ggraph)

```

2. Second, we create the **Twitter** token:

```R
token <- rtweet::create_token(
  app = "APPNAME",
  consumer_key <- "YOURKEY",
  consumer_secret <- "YOURSECRETKEY",
  access_token <- "...",
  access_secret <- "...")
```
*Obs: You need a Twitter developer account for this.*

3. We collect *tweets* for a specific subject or user, in this case we will collect *tweets* that mention Biden and Trump:

```R
biden <- rtweet::search_tweets("Biden", n = 5000, include_rts = FALSE)

trump <- rtweet::search_tweets("Trump", n = 5000, include_rts = FALSE)
```

4. We then geocode them, or extract latitude and longitude, and map them:

```R
coordB <- rtweet::lat_lng(biden)

coordT <- rtweet::lat_lng(trump)

par(mar = c(0, 0, 0, 0))
maps::map("state", lwd = .25)

with(coordB, points(lng, lat, pch = 20, cex = .75, col = "blue"))

with(coordT, points(lng, lat, pch = 20, cex = .75, col = "red"))
```

![Resulting Map](https://github.com/pmcavallo/pmcavallo.github.io/master/images/trump_biden.png "Resulting Map")
 
