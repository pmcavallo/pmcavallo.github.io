---
layout: post
title: Twitter Data
---

This example scrapes Twitter data, visualizes it, and looks at some descriptive information:

1. First, I install the *rtweet* package:

```R
install.packages("rtweet")

library(ggplot2)
library(rtweet)
library(igraph)
library(tidyverse)
library(ggraph)

```

2. Second, I create the token:
```R
token <- rtweet::create_token(
  app = "Cavallo",
  consumer_key <- "YOURKEY",
  consumer_secret <- "YOURSECRETKEY",
  access_token <- "...",
  access_secret <- "...")


 
