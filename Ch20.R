#### Table 20.1

library(tm)

# define vector of sentences ("docs")
text <- c("this is the first sentence", 
          "this is a second sentence", 
          "the third sentence is here")

# convert sentences into a corpus
corp <- Corpus(VectorSource(text))

# compute term frequency
tdm <- TermDocumentMatrix(corp)
inspect(tdm)



#### Table 20.2

library(tm)

text <- c("this is the first     sentence!!", 
          "this is a second Sentence :)", 
          "the third sentence, is here", 
          "forth of all sentences")
corp <- Corpus(VectorSource(text))
tdm <- TermDocumentMatrix(corp)
inspect(tdm)



#### Table 20.3

# tokenization
corp <- tm_map(corp, stripWhitespace) 
corp <- tm_map(corp, removePunctuation) 
tdm <- TermDocumentMatrix(corp)
inspect(tdm)



#### Table 20.4

stopwords("english")



#### Table 20.5

# stopwords
library(SnowballC)
corp <- tm_map(corp, removeWords, stopwords("english"))

# stemming
corp <- tm_map(corp, stemDocument) 

tdm <- TermDocumentMatrix(corp)
inspect(tdm)



#### Table 20.6

tfidf <- weightTfIdf(tdm)
inspect(tfidf)



#### Table 20.7

library(tm)
# step 1: import and label records
# read zip file into a corpus
corp <- Corpus(ZipSource("AutoAndElectronics.zip", recursive = T))


# create an array of records labels
label <- c(rep(1, 1000), rep(0, 1000))

# step 2: text preprocessing
# tokenization
corp <- tm_map(corp, stripWhitespace) 
corp <- tm_map(corp, removePunctuation) 
corp <- tm_map(corp, removeNumbers) 

# stopwords
corp <- tm_map(corp, removeWords, stopwords("english"))

# stemming
corp <- tm_map(corp, stemDocument) 

# step 3: TF-IDF and latent semantic analysis
# compute TF-IDF
tdm <- TermDocumentMatrix(corp)
tfidf <- weightTfIdf(tdm)

# extract (20) concepts 
library(lsa)
lsa.tfidf <- lsa(tfidf, dim = 20)

# convert to data frame
words.df <- as.data.frame(as.matrix(lsa.tfidf$dk)) 



#### Table 20.8

# sample 60% training data
training <- sample(c(1:2000), 0.6*2000)

# run logistic model on training
trainData = cbind(label = label[training], words.df[training,])
reg <- glm(label ~ ., data = trainData, family = 'binomial')

# compute accuracy on validation set
validData = cbind(label = label[-training], words.df[-training,])
pred <- predict(reg, newdata = validData, type = "response")

# produce confusion matrix
library(caret)
confusionMatrix(ifelse(pred>0.5, 1, 0), label[-training])



