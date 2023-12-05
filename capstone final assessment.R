##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

head(edx, 10)
summary(edx)

#----------------------------------------------------------------------------
#loading essential modules

if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(ggplot2)
library(lubridate)
library(stringr)

# prep data make a column that has the rating deviation from mean
edx = edx %>%
  mutate(rating_dev = rating-mean(rating), movie_year = str_extract(title, "(\\(\\d{4}\\))"), reviewer_genre = paste(genres, userId))
edx$movie_year = gsub("\\(", "", edx$movie_year)
edx$movie_year = gsub("\\)", "", edx$movie_year)
edx$movie_year = as.numeric(edx$movie_year)
edx$userId = as.factor(edx$userId)
# extract the year of the movie



# convert to datetime then to date
edx = mutate(edx, review_date = as.Date(as_datetime(timestamp)))
edx = mutate(edx, review_year = year(review_date))


# First split data into test and validation sets:
test_index = createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set = edx[-test_index,]
val_set = edx[test_index,]
#----------------------------------------------------------------------------
# base case, what is the mean?
mu = mean(edx$rating, na.rm = TRUE)
rmse_base = RMSE(edx$rating, mu)
print(rmse_base)

#----------------------------------------------------------------------------
# time effects
# calculate time dependent variation in scores. Will use loess
date_mean = train_set %>% group_by(review_date) %>% 
  summarise(mean_rating = mean(rating), mean_rating_dev = mean(rating_dev), weight = n())

time_deviation = loess(mean_rating_dev ~ as.numeric(review_date), data = date_mean, span=0.3, degree = 1, weights = weight)

# plot fit
time_effect = date_mean %>% mutate(date_fit = predict(time_deviation, data = date_mean))
ggplot(data = time_effect, aes(x = review_date, y=mean_rating_dev)) + geom_point() + geom_line(aes(x = review_date, y=date_fit), color='green')

# plot correction
ggplot(data = time_effect, aes(x = review_date, y=mean_rating_dev-date_fit)) + geom_point()

# add fit to data
edx$time_bias = predict(time_deviation, data.frame(review_date = as.numeric(edx$review_date)))
print(RMSE(edx$rating, mu))
print(RMSE(edx$rating, edx$time_bias+mu))

# residual variance after review time fitted
edx$residual =  edx$rating-mu-edx$time_bias

# check release year dependency
year_rating = edx %>% group_by(movie_year) %>% summarise(residual = mean(residual), weight = n())
plot(year_rating$movie_year, year_rating$residual)


# fit another loess to predict year variance
release_year_deviation = loess(residual ~ movie_year, data = year_rating, span=0.3, degree = 1, weights = weight)
year_rating$prediction = predict(release_year_deviation, year_rating)
ggplot(data = year_rating, aes(x = movie_year, y=residual)) + geom_point() + geom_line(aes(x = movie_year, y=prediction), color='green')
edx$release_bias = predict(release_year_deviation, edx)

# plot the corrected values
ggplot(data = year_rating, aes(x = movie_year, y=residual-prediction)) + geom_point()

# error RMSE:
print(RMSE(edx$rating, edx$release_bias + mu))

# inital data
edx$mu_t = edx$time_bias + mu + edx$release_bias
print(RMSE(edx$rating, mu))
print(RMSE(edx$rating, edx$mu_t)) #a tiny improvement


# recreate the sets now with the time adjustment in place
train_set = edx[-test_index,]
val_set = edx[test_index,]
val_set_testing = edx[test_index,]

#----------------------------------------------------------------------------
# calculate genre dependent variation 
genre_bias = train_set %>% group_by(genres) %>% summarise(mean_rating_genre = mean(rating), mean_dev_genre = mean(rating-mu_t),  sum_dev_genre = sum(rating-mu_t),  entries = n())
error = left_join(edx, genre_bias, by='genres')

print('base rmse:' )
print(RMSE(error$rating, error$mu_t))
print('genre corrected rmse:' )
print(RMSE(error$rating, replace_na(error$mean_dev_genre, 0)+error$mu_t))

#----------------------------------------------------------------------------
# calculate user dependent variation
average_user_ratings = train_set %>% group_by(userId) %>% summarise(mean_rating_user = mean(rating), mean_dev_user = mean(rating-mu_t), sum_dev_genre = sum(rating-mu_t), entries = n())
error = left_join(edx, average_user_ratings, by='userId')
print('user corrected rmse:' )
print(RMSE(error$rating, replace_na(error$mean_dev_user, 0)+error$mu_t))

#----------------------------------------------------------------------------
# calculate movie dependent variation
average_movie_rating = train_set %>% group_by(title) %>% summarise(mean_rating_movie = mean(rating), mean_dev_movie = mean(rating-mu_t), sum_dev_genre = sum(rating-mu_t), entries = n())
error = left_join(edx, average_movie_rating, by='title')
print('title corrected rmse:' )
print(RMSE(error$rating, replace_na(error$mean_dev_movie, 0)+error$mu_t))

#----------------------------------------------------------------------------
# calculate reviewer, genre co-dependent variation
average_rg_rating = train_set %>% group_by(reviewer_genre) %>% summarise(mean_rating_movie = mean(rating), mean_dev_rg = mean(rating-mu_t), sum_dev_genre = sum(rating-mu_t), entries = n())
error = left_join(edx, average_rg_rating, by='reviewer_genre')
print('reviewer, genre co-dependent corrected rmse:' )
print(RMSE(error$rating, replace_na(error$mean_dev_rg, 0)+error$mu_t))

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# regularised model

# isolated genre effect
lambda_range = seq(0, 10, 0.5)

lambda_search_genre = function(lambda){
  genre_alone_reg = train_set %>% group_by(genres) %>% summarize(genre_effect_reg = sum(rating-mu_t)/(n()+lambda))
  prediction = val_set %>%
    left_join(genre_alone_reg, by = 'genres')
  rmse = RMSE(prediction$rating, replace_na(prediction$genre_effect_reg, 0) + prediction$mu_t)
  return(rmse)
}

predictions = sapply(lambda_range, lambda_search_genre)
plot(lambda_range, predictions)
lambda_t_g = lambda_range[which.min(predictions)] # returns 0, regularization does not appear to help in this case. 

# predictor table for genre effects
genre_alone_reg = edx %>% group_by(genres) %>% summarize(genre_effect_reg = sum(rating-mu_t)/(n()+lambda_t_g))
train_set = train_set %>% 
  left_join(genre_alone_reg, by='genres')
val_set = val_set %>%
  left_join(genre_alone_reg, by='genres')

# isolated user effect
lambda_search_user = function(lambda){
  user_effect_reg = train_set %>% 
    group_by(userId) %>%
    summarize(user_effect_reg = sum(rating - mu_t - coalesce(genre_effect_reg, 0))/(n()+lambda))
  
  prediction = val_set %>%
    left_join(user_effect_reg, by = 'userId')
  
  estimate = replace_na(prediction$genre_effect_reg, 0) + prediction$mu_t + replace_na(prediction$user_effect_reg, 0)
  estimate[estimate>5] = 5
  estimate[estimate<0.5] = 0.5

  rmse = RMSE(prediction$rating, estimate)
  return(rmse)

}

predictions = sapply(lambda_range, lambda_search_user)
plot(lambda_range, predictions)
lambda_t_g_u = lambda_range[which.min(predictions)]

# predictor table for user effects (use it all this time)
user_effect_reg = edx %>% 
  left_join(genre_alone_reg, by='genres') %>% 
  group_by(userId) %>%
  summarize(user_effect_reg = sum(rating - mu_t - genre_effect_reg)/(n()+lambda_t_g_u))

train_set = train_set %>% 
  left_join(user_effect_reg, by='userId')
val_set = val_set %>%
  left_join(user_effect_reg, by='userId')

# isolate title effect
lambda_search_title = function(lambda){
  title_effect_reg = train_set %>% 
    group_by(title) %>%
    summarize(title_effect_reg = sum(rating - mu_t - genre_effect_reg - user_effect_reg)/(n()+lambda))
  
  prediction = val_set %>%
    left_join(title_effect_reg, by = 'title')
  
  estimate = replace_na(prediction$genre_effect_reg, 0) + prediction$mu_t + replace_na(prediction$user_effect_reg, 0) + replace_na(prediction$title_effect_reg, 0)
  estimate[estimate>5] = 5
  estimate[estimate<0.5] = 0.5
  
  rmse = RMSE(prediction$rating, estimate)
  return(rmse)

}

predictions = sapply(lambda_range, lambda_search_title)
plot(lambda_range, predictions)
lambda_t_g_u_t = lambda_range[which.min(predictions)]

title_effect_reg = edx %>% 
  left_join(genre_alone_reg, by='genres') %>% 
  left_join(user_effect_reg, by='userId') %>% 
  group_by(title) %>%
  summarize(title_effect_reg = sum(rating - genre_effect_reg - user_effect_reg - mu_t)/(n()+lambda_t_g_u_t))

train_set = train_set %>% 
  left_join(title_effect_reg, by='title')
val_set = val_set %>%
  left_join(title_effect_reg, by='title')

# reviewer genre interaction
reviewer_genre_effect_reg = train_set %>% 
  group_by(reviewer_genre) %>%
  summarize(reviewer_genre_effect_sum = sum(rating - mu_t - genre_effect_reg - user_effect_reg - title_effect_reg), count=n())
  
lambda_search_rg = function(lambda){
  reviewer_genre_effect_reg$reviewer_genre_effect_reg = reviewer_genre_effect_reg$reviewer_genre_effect_sum/(reviewer_genre_effect_reg$count+lambda)
  print('here')
  prediction = val_set %>%
    left_join(reviewer_genre_effect_reg, by = 'reviewer_genre')
  
  estimate = replace_na(prediction$genre_effect_reg, 0) + prediction$mu_t + replace_na(prediction$user_effect_reg, 0) + replace_na(prediction$title_effect_reg, 0) + replace_na(prediction$reviewer_genre_effect_reg, 0)
  estimate[estimate>5] = 5
  estimate[estimate<0.5] = 0.5
  
  
  rmse = RMSE(prediction$rating, estimate)
  return(rmse)
  
}

predictions = sapply(lambda_range, lambda_search_rg)
plot(lambda_range, predictions)
lambda_t_g_u_t_r = lambda_range[which.min(predictions)]

reviewer_genre_effect_reg = edx %>% 
  left_join(genre_alone_reg, by='genres') %>% 
  left_join(user_effect_reg, by='userId') %>% 
  left_join(title_effect_reg, by='title') %>%
  group_by(reviewer_genre) %>%
  summarize(reviewer_genre_effect_reg = sum(rating - mu_t - genre_effect_reg - user_effect_reg - title_effect_reg)/(n()+lambda_t_g_u_t_r))

#----------------------------------------------------------------------------
# function for prediction

final_predictor = function(dataset){
  
  # to make code easier for passing in the final holdout there will be some repeat stuff
  
  dataset = dataset %>%
    mutate(rating_dev = rating-mean(rating), movie_year = str_extract(title, "(\\(\\d{4}\\))"), reviewer_genre = paste(genres, userId))
  dataset$movie_year = gsub("\\(", "", dataset$movie_year)
  dataset$movie_year = gsub("\\)", "", dataset$movie_year)
  dataset$movie_year = as.numeric(dataset$movie_year)
  
  dataset$userId = as.factor(dataset$userId)
  dataset = mutate(dataset, review_date = as.Date(as_datetime(timestamp)))
  dataset = mutate(dataset, review_year = year(review_date))
  
  
  # setting the base
  dataset$base_prediction = mu
  
  
  # setting the time bias with base
  dataset$time_bias = predict(time_deviation, data.frame(review_date = as.numeric(dataset$review_date)))
  dataset$release_bias = predict(release_year_deviation, dataset)
  dataset$mu_t = dataset$time_bias + mu + dataset$release_bias
  
  # user adjustment
  dataset = left_join(dataset, genre_alone_reg, by='genres')
  
  # genre
  dataset = left_join(dataset, user_effect_reg, by='userId')
  
  # title adjustment
  dataset = left_join(dataset, title_effect_reg, by='title')
  
  # reviewer genre effects
  dataset = left_join(dataset, reviewer_genre_effect_reg, by='reviewer_genre')
  View(dataset)

  # logic side, favor title means with user adjustment
  # if title missing from training, guess using genre prediction
  # if user missing, assume the mean title (i.e. 0 adjustment) 
  
  dataset$final_prediction = replace_na(dataset$genre_effect_reg, 0) + replace_na(dataset$user_effect_reg, 0) + replace_na(dataset$title_effect_reg, 0) + replace_na(dataset$reviewer_genre_effect_reg, 0) + dataset$mu_t
  
  # set constraints
  
  dataset$final_prediction[dataset$final_prediction > 5] = 5
  dataset$final_prediction[dataset$final_prediction < 0.5] = 0.5
  dataset$prediction_error = (dataset$rating - dataset$final_prediction)^2
  
  #calculate rmse
  rmse = RMSE(dataset$rating, dataset$final_prediction)
  print(rmse)
  return(dataset)
}

output = final_predictor(final_holdout_test)





