# Convolutional Neural Networks

# Load packages
library(keras)
library(EBImage)

# Read Images
setwd('---')
pic1 <- c('p1.jpg', 'p2.jpg', 'p3.jpg', 'p4.jpg', 'p5.jpg',
          'c1.jpg', 'c2.jpg', 'c3.jpg', 'c4.jpg', 'c5.jpg',
          'b1.jpg', 'b2.jpg', 'b3.jpg', 'b4.jpg', 'b5.jpg')
train <- list()
for (i in 1:15) {train[[i]] <- readImage(pic1[i])}

pic2 <- c('p6.jpg', 'c6.jpg', 'b6.jpg')
test <- list()
for (i in 1:3) {test[[i]] <- readImage(pic2[i])}

# Explore
print(train[[12]])
summary(train[[12]])
display(train[[12]])
plot(train[[12]])

par(mfrow = c(---, ---))
for (i in 1:15) plot(train[[i]])
par(mfrow = c(1,1))

# Resize & combine
str(train)
for (i in 1:15) {train[[i]] <- resize(train[[i]], 100, 100)}
for (i in 1:3) {test[[i]] <- ---(test[[i]], 100, 100)}

train <- combine(train)
x <- tile(train, 5)
display(x, title='Pictures')

test <- ---(test)
y <- ---(test, 3)
---(y, title = 'Pics')

# Reorder dimension
train <- ---(train, c(4, 1, 2, 3))
test <- ---(test, c(4, 1, 2, 3))
str(train)

# Response
trainy <- c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
testy <- c(0, 1, 2)

# One hot encoding
trainLabels <- ---(trainy)
testLabels <- ---(testy)

# Model
model <- keras_model_sequential()

model %>%
         layer_conv_2d(filters = 32, 
                       kernel_size = c(3,3),
                       activation = '---',
                       input_shape = c(---, ---, ---)) %>%
         ---(filters = 32,
                       kernel_size = c(3,3),
                       activation = 'relu') %>%
         ---(pool_size = c(2,2)) %>%
         ---(rate = 0.25) %>%
         layer_conv_2d(filters = 64,
                       kernel_size = c(3,3),
                       activation = 'relu') %>%
         layer_conv_2d(filters = 64,
                       kernel_size = c(3,3),
                       activation = 'relu') %>%
         layer_max_pooling_2d(pool_size = c(2,2)) %>%
         layer_dropout(rate = 0.25) %>%
         ---() %>%
         layer_dense(units = 256, activation = 'relu') %>%
         layer_dropout(rate=0.25) %>%
         layer_dense(units = 3, activation = '---') %>%
         
         compile(loss = '---',
                 optimizer = optimizer_sgd(lr = 0.01,
                                           decay = 1e-6,
                                           momentum = 0.9,
                                           nesterov = T),
                 metrics = c('---'))
summary(model)

# Fit model
history <- model %>%
         fit(train,
             trainLabels,
             --- = 60,
             --- = 32,
             validation_split = 0.2,
             validation_data = list(test, testLabels))
plot(history)

# Evaluation & Prediction - train data
model %>% ---(train, trainLabels)
pred <- model %>% ---(train)
table(Predicted = pred, Actual = trainy)

prob <- model %>% ---(train)
cbind(prob, Predicted_class = pred, Actual = trainy)

# Evaluation & Prediction - test data
model %>% ---(test, testLabels)
pred <- model %>% ---(test)
table(Predicted = pred, Actual = testy)

prob <- model %>% ---(test)
---(prob, Predicted_class = pred, Actual = testy)
