# Clear plots
if(!is.null(dev.list())) dev.off()
# Clean workspace
rm(list=ls())
# Clear console
cat("\014") 
# setwd("E:/SSPU/Courses/Courses/BI/Wiley.Data.Mining.for.Business.Analytics.Concepts.Techniques.and.Applications.in.R.1118879368/codes")



#### Figure 16.1
# install.packages("forecast")
library(forecast)
Amtrak.data <- read.csv("Amtrak data.csv")

# create time series object using ts()
# ts() takes three arguments: start, end, and freq. 
# with monthly data, the frequency of periods per season is 12 (per year). 
# arguments start and end are (season number, period number) pairs. 
# here start is Jan 1991: start = c(1991, 1); end is Mar 2004: end = c(2004, 3).
ridership.ts <- ts(Amtrak.data$Ridership, 
                   start = c(1991, 1), end = c(2004, 3), freq = 12)

# plot the series
plot(ridership.ts, xlab = "Time", ylab = "Ridership (in 000s)", ylim = c(1300, 2300))

#### Figure 13.6

library(forecast) 

# create short time series
# use window() to create a new, shorter time series of ridership.ts
# for the new three-year series, start time is Jan 1997 and end time is Dec 1999
ridership.ts.3yrs <- window(ridership.ts, start = c(1997, 1), end = c(1999, 12))

# fit a linear regression model to the time series
ridership.lm <- tslm(ridership.ts ~ trend + I(trend^2))

# shorter and longer time series
par(mfrow = c(2, 1))
plot(ridership.ts.3yrs, xlab = "Time", ylab = "Ridership (in 000s)", 
     ylim = c(1300, 2300))
plot(ridership.ts, xlab = "Time", ylab = "Ridership (in 000s)", ylim = c(1300, 2300))

# overlay the fitted values of the linear model
lines(ridership.lm$fitted, lwd = 2)


#### Figure 16.4

nValid <- 36
nTrain <- length(ridership.ts) - nValid

# partition the data
train.ts <- window(ridership.ts, start = c(1991, 1), end = c(1991, nTrain))
valid.ts <- window(ridership.ts, start = c(1991, nTrain + 1), 
                   end = c(1991, nTrain + nValid))

#  generate the naive and seasonal naive forecasts
naive.pred <- naive(train.ts, h = nValid)
snaive.pred <- snaive(train.ts, h = nValid)

# plot forecasts and actuals in the training and validation sets
par(mfrow = c(1, 1))

plot(train.ts, ylim = c(1300, 2600),  ylab = "Ridership", xlab = "Time", bty = "l", 
     xaxt = "n", xlim = c(1991,2006.25), main = "")
axis(1, at = seq(1991, 2006, 1), labels = format(seq(1991, 2006, 1)))
lines(naive.pred$mean, lwd = 2, col = "blue", lty = 1)
lines(snaive.pred$mean, lwd = 2, col = "blue", lty = 1)
lines(valid.ts, col = "grey20", lty = 3)
lines(c(2004.25 - 3, 2004.25 - 3), c(0, 3500)) 
lines(c(2004.25, 2004.25), c(0, 3500))
text(1996.25, 2500, "Training")
text(2002.75, 2500, "Validation")
text(2005.25, 2500, "Future")
arrows(2004 - 3, 2450, 1991.25, 2450, code = 3, length = 0.1, lwd = 1,angle = 30)
arrows(2004.5 - 3, 2450, 2004, 2450, code = 3, length = 0.1, lwd = 1,angle = 30)
arrows(2004.5, 2450, 2006, 2450, code = 3, length = 0.1, lwd = 1, angle = 30)


#### Table 16.1

accuracy(naive.pred, valid.ts)
accuracy(snaive.pred, valid.ts)

