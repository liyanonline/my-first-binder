#### Figure 19.1

library(igraph)

# define links in data
edges <- rbind(
  c("Dave", "Jenny"), c("Peter", "Jenny"), c("John", "Jenny"), 
  c("Dave", "Peter"), c("Dave", "John"), c("Peter", "Sam"), 
  c("Sam", "Albert"), c("Peter", "John")
)

# generate and plot graph
# set argument directed = FALSE in graph.edgelist() to plot an undirected graph.
g <- graph.edgelist(edges, directed = FALSE)
plot(g, vertex.size = 1, vertex.label.dist = 0.5)



#### Figure 19.2

# generate and plot graph
# set argument directed = TRUE in graph.edgelist() to plot a directed graph.
g <- graph.edgelist(edges, directed = TRUE)
plot(g, vertex.size = 1, vertex.label.dist = 0.5)



#### Figure 19.4

drug.df <- read.csv("Drug.csv")

# convert edges to edge list matrix
edges <- as.matrix(drug.df[, c(1,2)])
g <- graph.edgelist(edges,directed=FALSE)

# plot graph
# nodes' size is proportional to their eigenvector centrality
plot(g, vertex.label = NA, vertex.size = eigen_centrality(g)$vector * 20)



#### Figure 19.5

# Building on the code presented in Figure 19.1:
edges <- rbind(
  c("Dave", "Jenny"), c("Peter", "Jenny"), c("John", "Jenny"), 
  c("Dave", "Peter"), c("Dave", "John"), c("Peter", "Sam"), 
  c("Sam", "Albert"), c("Peter", "John")
)

# generate and plot graph
# set argument directed = FALSE in graph.edgelist() to plot an undirected graph.
g <- graph.edgelist(edges, directed = FALSE)
#

plot(g, layout = layout_in_circle, vertex.size = 1, vertex.label.dist = 0.5)
plot(g, layout = layout_on_grid, vertex.size = 1, vertex.label.dist = 0.5)



#### Table 19.3

degree(g)
betweenness(g)
closeness(g)
eigen_centrality(g)



#### Figure 19.6

# get Peter's 1-level ego network
# for a 2-level ego network set argument order = 2 in make_ego_graph().
peter.ego <- make_ego_graph(g, order = 1, nodes = "Peter")
plot(peter.ego[[1]], vertex.size = 1, vertex.label.dist = 0.5)



#### Table 19.5

degree.distribution(g) # normalized
edge_density(g)



#### Table 19.11

library(twitteR)
# replace key and secret number with those you obtained from Twitter
setup_twitter_oauth(consumer_key = "XXX", consumer_secret = "XXX")

# get recent tweets
recent.25.tweets <- searchTwitter("text mining", resultType="recent", n = 25)



#### Table 19.12

library(Rfacebook)
library(httpuv)
# replace the app id and secret number with those you obtained from Facebook
fb_oauth <- fbOAuth(app_id = "XXX", app_secret = "XXX")

# get recent posts on page "dataminingbook"
fb_page <- getPage(page = "dataminingbook", token = fb_oauth$credentials$access_token)

# a facebook page contains the following information:
t(t(names(fb_page)))
fb_page[1,]

# get information about most recent post
post <- getPost(post=fb_page$id[1], n=20, token = fb_oauth$credentials$access_token)

post$likes
post$comments

