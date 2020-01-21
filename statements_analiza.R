
# Loading packages

# install.packages("tidytext")
# install.packages("SentimentAnalysis")
# install.packages("readtext")
# install.packages("read_excel")
# install.packages("textstem")
library(tidyverse)
library(tidytext)
library(SentimentAnalysis)
library(readtext)
library(readxl)
library(dplyr)

# Loading scrapped statements (from 2006 to 2019)

DATA_DIR <- "C:/Users/esobolewska/Documents/FOMC-text-mining/Statements"
fomc_2006 <- readtext(paste0(DATA_DIR, "/2006/*"))
fomc_2007 <- readtext(paste0(DATA_DIR, "/2007/*"))
fomc_2008 <- readtext(paste0(DATA_DIR, "/2008/*"))
fomc_2009 <- readtext(paste0(DATA_DIR, "/2009/*"))
fomc_2010 <- readtext(paste0(DATA_DIR, "/2010/*"))
fomc_2011 <- readtext(paste0(DATA_DIR, "/2011/*"))
fomc_2012 <- readtext(paste0(DATA_DIR, "/2012/*"))
fomc_2013 <- readtext(paste0(DATA_DIR, "/2013/*"))
fomc_2014 <- readtext(paste0(DATA_DIR, "/2014/*"))
fomc_2015 <- readtext(paste0(DATA_DIR, "/2015/*"))
fomc_2016 <- readtext(paste0(DATA_DIR, "/2016/*"))
fomc_2017 <- readtext(paste0(DATA_DIR, "/2017/*"))
fomc_2018 <- readtext(paste0(DATA_DIR, "/2018/*"))
fomc_2019 <- readtext(paste0(DATA_DIR, "/2019/*"))

statements <- rbind(fomc_2006,fomc_2007,fomc_2008,fomc_2009,fomc_2010,fomc_2011,
                    fomc_2012,fomc_2013,fomc_2014,fomc_2015,fomc_2016,fomc_2017,fomc_2018,fomc_2019)

remove(fomc_2006,fomc_2007,fomc_2008,fomc_2009,fomc_2010,fomc_2011,
       fomc_2012,fomc_2013,fomc_2014,fomc_2015,fomc_2016,fomc_2017,fomc_2018,fomc_2019)

# Initial preprocessing
statements <- statements %>% mutate(ID = 1:n())
colnames(statements) <- c("Date", "Text", "ID")
statements$Date <- gsub(".txt", "", statements$Date)
statements$Date <- as.Date(statements$Date, "%Y%m%d ")
statements_all <- as.vector(statements$Text)
length(statements_all) # 112 documents

# Converting documents into corpus (112 documents)
library(tm)
(corpus_all <- VCorpus(VectorSource(statements_all)))
inspect(corpus_all[[1]])
as.character(corpus_all[[1]]) 

# Preprocessing - cleaning text
stopwords <- stopwords("en")

# tm package
system.time (
  corpus_clean <- corpus_all %>% 
    tm_map(tolower) %>%
    tm_map(removeWords, stopwords) %>% 
    tm_map(removePunctuation) %>%
    tm_map(removeNumbers)  %>%
    tm_map(stripWhitespace) %>% 
    tm_map(PlainTextDocument)
)
# statement after cleaning
as.character(corpus_clean[[1]]) 


# install.packages("qdap")
# install.packages("qdap", INSTALL_opts = "--no-multiarch")
# install.packages("rJava")

# suppressPackageStartupMessages({
#   library(rJava); library(qdap)})

# Sys.setenv(JAVA_HOME='/Library/Java/JavaVirtualMachines/jdk1.8.0_221.jdk/Contents/Home/jre')


df_corpus <- data.frame(text = unlist(sapply(corpus_clean, `[`, "content")), stringsAsFactors = F)
df_corpus <- df_corpus %>% mutate(doc_id = 1:n())

statements_clean <- statements %>% 
  mutate(cleaned_text = df_corpus$text)

count_cleaned_word <- statements_clean %>%
  unnest_tokens(word_count, cleaned_text) %>%
  count(ID, word_count, sort = T) %>% 
  group_by(ID) %>% 
  summarize(word_cleaned_count = sum(n))

statements_clean_count <- left_join(statements_clean, count_cleaned_word, by = 'ID')

count_word <- statements_clean_count %>%
  unnest_tokens(word_count, Text) %>%
  count(ID, word_count, sort = T) %>% 
  group_by(ID) %>% 
  summarize (word_count = sum(n))

statements_final <- left_join(statements_clean_count, count_word, by = 'ID')

library(textstem)
statements_final$lemma_text <- lemmatize_strings(statements_final$cleaned_text)

# Tokenization
tokens <- statements_final %>%
  unnest_tokens(word, lemma_text) 





# Topic Modelling ---------------------------------------------------------

# topic modelling - do poprawy na pewno, bo słabo wychodzi
library(tm)
# install.packages("wordcloud")
library(wordcloud)
library(slam)
# install.packages("topicmodels")
library(topicmodels)

# Creating a Term document Matrix
tdm = DocumentTermMatrix(corpus_clean) 

# create tf-idf matrix
term_tfidf <- tapply(tdm$v/row_sums(tdm)[tdm$i], tdm$j, mean) * log2(nDocs(tdm)/col_sums(tdm > 0))
summary(term_tfidf)
tdm <- tdm[,term_tfidf >= 0.05]
tdm <- tdm[row_sums(tdm) > 0,]
summary(col_sums(tdm))
# finding best K 
best.model <- lapply(seq(2, 50, by = 1), function(d){LDA(tdm, d)})
best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik)))
# calculating LDA
k = 5 # number of topics
SEED = 112 # number of documents 
CSC_TM <-list(VEM = LDA(tdm, k = k, 
                        control = list(seed = SEED)), 
              VEM_fixed = LDA(tdm, k = k,
                              control = list(estimate.alpha = FALSE, seed = SEED)),
              Gibbs = LDA(tdm, k = k, method = "Gibbs",
                          control = list(seed = SEED, burnin = 1000, thin = 100, iter = 1000)),
              CTM = CTM(tdm, k = k,
                        control = list(seed = SEED,
                                       var = list(tol = 10^-4), 
                                       em = list(tol = 10^-3))))

sapply(CSC_TM[1:2], slot, "alpha")
sapply(CSC_TM, function(x) mean(apply(posterior(x)$topics, 1, function(z) sum(z*log(z)))))
Topic <- topics(CSC_TM[["VEM"]], 1)
Terms <- terms(CSC_TM[["VEM"]], 8)
Terms

# https://campus.datacamp.com/courses/topic-modeling-in-r/quick-introduction-to-the-workflow?ex=1