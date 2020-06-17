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

3. We collect *tweets* for a specific subject or user, in this case we will collect *tweets* that mention just the names Biden and Trump:

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

![Resulting Map](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/trump_biden.png?raw=true)

5. Now we will collect *tweets* that mentions the *users* **realDonaldTrump** and **JoeBIden**:

```R
rdt <- rtweet::search_tweets(q = "realDonaldTrump", n = 1000, lang = "en")

bid <- rtweet::search_tweets(q = "JoeBiden", n = 1000, lang = "en")
```

6. And then we check the most popular hashtags used when *tweeting* about Trump and Biden:

```R
library(stringr)
dt <- str_extract_all(rdt$text, "#(\\d|\\w)+")
dt <- unlist(dt)
head(sort(table(ht), decreasing = TRUE))

jb <- str_extract_all(bid$text, "#(\\d|\\w)+")
jb <- unlist(jb)
head(sort(table(jb), decreasing = TRUE))
```
And the results are:

| Trump                | Biden                |
| ---------------------|----------------------|
| #AIDS    (24)        | #Trump (124)         |
| #ExecutiveOrder (24) | #ExecutiveOrder (121)|  
| #HIV (24)            | #ObamaBiden (120)    |  
| #IdiotInChief (24)   | #JoeBiden (22)       |
| #Trump (24)          | #Trump2020 (20)      |


7. And now we can check  how many times their names are mentioned when *tweeting* about the other. As a bonus, we are also going to check how many times Obama's name is mentioned when *tweeting* about them:

```R
length(grep("obama", rdt$text, ignore.case=TRUE))  
[1] 78

length(grep("obama", bid$text, ignore.case=TRUE))  
[1] 302

length(grep("trump", bid$text, ignore.case=TRUE))  
[1] 491

length(grep("biden", rdt$text, ignore.case=TRUE))
[1] 112
```

8. We can then create a network plot to see based, for instance, on *retweets* to check user's *influence*:

```R
filter(rdt, retweet_count > 0 ) %>% 
  select(screen_name, mentions_screen_name) %>%
  unnest(mentions_screen_name) %>% 
  filter(!is.na(mentions_screen_name)) %>% 
  graph_from_data_frame() -> rdt_g
V(rdt_g)$node_label <- unname(ifelse(degree(rdt_g)[V(rdt_g)] > 20, names(V(rdt_g)), "")) 
V(rdt_g)$node_size <- unname(ifelse(degree(rdt_g)[V(rdt_g)] > 20, degree(rdt_g), 0)) 
ggraph(rdt_g, layout = 'kk') + 
  geom_edge_arc(edge_width=0.1, aes(alpha=..index..)) +
  geom_node_label(aes(label=node_label, size=node_size),
                  label.size=0, fill="#ffffff66", segment.colour="light blue",
                  color="red", repel=TRUE, family="Apple Garamond") +
  coord_fixed() +
  scale_size_area(trans="sqrt") +
  labs(title="Title", subtitle="Edges=volume of retweets. Screenname size=influence") +
  theme_graph(base_family="Apple Garamond") +
  theme(legend.position="none") 
  ```
  
  ![Network Plot](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/net_T.png?raw=true)

 
