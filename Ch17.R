# Clear plots
if(!is.null(dev.list())) dev.off()
# Clean workspace
rm(list=ls())
# Clear console
cat("\014") 
# setwd("E:/SSPU/Courses/Courses/BI/Wiley.Data.Mining.for.Business.Analytics.Concepts.Techniques.and.Applications.in.R.1118879368/codes")



#### Figure 17.1
# sudo apt-get install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev
# sudo -i R
# install.packages('devtools')
# install.packages("forecast")
library(forecast) 
Amtrak.data <- read.csv("Amtrak data.csv")

# create time series
ridership.ts <- ts(Amtrak.data$Ridership, start = c(1991,1),
                   end = c(2004,3), freq = 12)

# produce linear trend model
ridership.lm <- tslm(ridership.ts ~ trend)

# plot the series
plot(ridership.ts, xlab = "Time", ylab = "Ridership", ylim = c(1300,2300),
     bty = "l")
lines(ridership.lm$fitted, lwd = 2)

# partition the data
nValid <- 36
nTrain <- length(ridership.ts) - nValid
train.ts <- window(ridership.ts, start = c(1991, 1), end = c(1991, nTrain))
valid.ts <- window(ridership.ts, start = c(1991, nTrain + 1), 
                   end = c(1991, nTrain + nValid))


#### Figure 17.2

# fit linear trend model to training set and create forecasts
train.lm <- tslm(train.ts ~ trend)
train.lm.pred <- forecast(train.lm, h = nValid, level = 0)

par(mfrow = c(2, 1))
plot(train.lm.pred, ylim = c(1300, 2600),  ylab = "Ridership", xlab = "Time", 
     bty = "l", xaxt = "n", xlim = c(1991,2006.25), main = "", flty = 2)
axis(1, at = seq(1991, 2006, 1), labels = format(seq(1991, 2006, 1)))
lines(train.lm.pred$fitted, lwd = 2, col = "blue")
lines(valid.ts)
plot(train.lm.pred$residuals, ylim = c(-420, 500),  ylab = "Forecast Errors", 
     xlab = "Time", bty = "l", xaxt = "n", xlim = c(1991,2006.25), main = "")
axis(1, at = seq(1991, 2006, 1), labels = format(seq(1991, 2006, 1)))
lines(valid.ts - train.lm.pred$mean, lwd = 1)



#### Table 17.2

summary(train.lm)



#### Figure 17.3

# fit exponential trend using tslm() with argument lambda = 0 
train.lm.expo.trend <- tslm(train.ts ~ trend, lambda = 0)
train.lm.expo.trend.pred <- forecast(train.lm.expo.trend, h = nValid, level = 0)

# fit linear trend using tslm() with argument lambda = 1 (no transform of y)
train.lm.linear.trend <- tslm(train.ts ~ trend, lambda = 1)
train.lm.linear.trend.pred <- forecast(train.lm.linear.trend, h = nValid, level = 0)

plot(train.lm.expo.trend.pred, ylim = c(1300, 2600),  ylab = "Ridership", 
     xlab = "Time", bty = "l", xaxt = "n", xlim = c(1991,2006.25), main = "", flty = 2)
axis(1, at = seq(1991, 2006, 1), labels = format(seq(1991, 2006, 1)))
lines(train.lm.expo.trend.pred$fitted, lwd = 2, col = "blue")  # Added in 6-5
lines(train.lm.linear.trend.pred$fitted, lwd = 2, col = "black", lty = 3)
lines(train.lm.linear.trend.pred$mean, lwd = 2, col = "black", lty = 3)
lines(valid.ts)



#### Figure 17.4

# fit quadratic trend using function I(), which treats an object "as is".
train.lm.poly.trend <- tslm(train.ts ~ trend + I(trend^2))
summary(train.lm.poly.trend)
train.lm.poly.trend.pred <- forecast(train.lm.poly.trend, h = nValid, level = 0)

par(mfrow = c(2,1))
plot(train.lm.poly.trend.pred, ylim = c(1300, 2600),  ylab = "Ridership", 
     xlab = "Time", bty = "l", xaxt = "n", xlim = c(1991,2006.25), main = "", flty = 2)
axis(1, at = seq(1991, 2006, 1), labels = format(seq(1991, 2006, 1))) 
lines(train.lm.poly.trend$fitted, lwd = 2)
lines(valid.ts)

plot(train.lm.poly.trend$residuals, ylim = c(-400, 550),  ylab = "Forecast Errors", 
     xlab = "Time", bty = "l", xaxt = "n", xlim = c(1991,2006.25), main = "")
axis(1, at = seq(1991, 2006, 1), labels = format(seq(1991, 2006, 1)))
lines(valid.ts - train.lm.poly.trend.pred$mean, lwd = 1)


par(mfrow = c(1,1))

#### Table 17.4

# include season as a predictor in tslm(). Here it creates 11 dummies, 
# one for each month except for the first season, January.  
train.lm.season <- tslm(train.ts ~ season)
summary(train.lm.season)

#### Table 17.5

train.lm.trend.season <- tslm(train.ts ~ trend + I(trend^2) + season)
summary(train.lm.trend.season)

#### Figure 17.7

ridership.24.ts <- window(train.ts, start = c(1991, 1), end = c(1991, 24))
Acf(ridership.24.ts, lag.max = 12, main = "")

#### Table 17.7

# fit linear regression with quadratic trend and seasonality to Ridership
train.lm.trend.season <- tslm(train.ts ~ trend + I(trend^2) + season)

# fit AR(1) model to training residuals
# use Arima() in the forecast package to fit an ARIMA model 
# (that includes AR models); order = c(1,0,0) gives an AR(1).
train.res.arima <- Arima(train.lm.trend.season$residuals, order = c(1,0,0))
train.res.arima.pred <- forecast(train.res.arima, h = 1)
valid.res.arima.pred <- forecast(train.res.arima, h = 1)
summary(train.res.arima)
valid.res.arima.pred

#### Figure 17.9

plot(train.lm.trend.season$residuals, ylim = c(-250, 250),  ylab = "Residuals", 
     xlab = "Time", bty = "l", xaxt = "n", xlim = c(1991,2006.25), main = "")
axis(1, at = seq(1991, 2006, 1), labels = format(seq(1991, 2006, 1)))
lines(train.res.arima.pred$fitted, lwd = 2, col = "blue")

#### Table 17.8

sp500.df <- read.csv("SP500.csv")

sp500.ts <- ts(sp500.df$Close, start = c(1995, 5), end = c(2003, 8), freq = 12)
sp500.arima <- Arima(sp500.ts, order = c(1,0,0))
sp500.arima

