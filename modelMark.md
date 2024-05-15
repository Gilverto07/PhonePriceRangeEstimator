---
author: "Gilverto De Los Santos Rios"
output: html_document
source: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data 
---


# Importing libraries

```r
library(randomForest)
library(caret)
```


# Load data

```r
TrainSet = read.csv("train.csv", header = TRUE)
head(TrainSet)
```

```
##   battery_power blue clock_speed dual_sim fc four_g int_memory m_dep mobile_wt n_cores pc
## 1           842    0         2.2        0  1      0          7   0.6       188       2  2
## 2          1021    1         0.5        1  0      1         53   0.7       136       3  6
## 3           563    1         0.5        1  2      1         41   0.9       145       5  6
## 4           615    1         2.5        0  0      0         10   0.8       131       6  9
## 5          1821    1         1.2        0 13      1         44   0.6       141       2 14
## 6          1859    0         0.5        1  3      0         22   0.7       164       1  7
##   px_height px_width  ram sc_h sc_w talk_time three_g touch_screen wifi price_range
## 1        20      756 2549    9    7        19       0            0    1           1
## 2       905     1988 2631   17    3         7       1            1    0           2
## 3      1263     1716 2603   11    2         9       1            1    0           2
## 4      1216     1786 2769   16    8        11       1            0    0           2
## 5      1208     1212 1411    8    2        15       1            1    0           1
## 6      1004     1654 1067   17    1        10       1            0    0           1
```

```r
TestingSet = read.csv("test.csv", header = TRUE)
head(TestingSet)
```

```
##   id battery_power blue clock_speed dual_sim fc four_g int_memory m_dep mobile_wt n_cores pc
## 1  1          1043    1         1.8        1 14      0          5   0.1       193       3 16
## 2  2           841    1         0.5        1  4      1         61   0.8       191       5 12
## 3  3          1807    1         2.8        0  1      0         27   0.9       186       3  4
## 4  4          1546    0         0.5        1 18      1         25   0.5        96       8 20
## 5  5          1434    0         1.4        0 11      1         49   0.5       108       6 18
## 6  6          1464    1         2.9        1  5      1         50   0.8       198       8  9
##   px_height px_width  ram sc_h sc_w talk_time three_g touch_screen wifi
## 1       226     1412 3476   12    7         2       0            1    0
## 2       746      857 3895    6    0         7       1            0    0
## 3      1270     1366 2396   17   10        10       0            1    1
## 4       295     1752 3893   10    0         7       1            1    0
## 5       749      810 1773   15    8         7       1            0    1
## 6       569      939 3506   10    7         3       1            1    1
```

# Multiple Linear Regression Model
In order to increase the accuracy of the model, I used a multiple regression model to
determine which characteristics are the most significant against the price_range.
Using the lm() function I was able to create a linear regression model and using the summary() function I can see the significance of each column. There were six values that had a significance value of less the 5% and those will be the only columns I will use.

```r
fit = lm(price_range ~., data = TrainSet)
summary(fit)
```

```
## 
## Call:
## lm(formula = price_range ~ ., data = TrainSet)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -1.04345 -0.24705  0.00349  0.24601  0.81279 
## 
## Coefficients:
##                 Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   -1.575e+00  6.164e-02 -25.553  < 2e-16 ***
## battery_power  5.096e-04  1.640e-05  31.071  < 2e-16 ***
## blue          -2.032e-03  1.442e-02  -0.141   0.8879    
## clock_speed   -1.206e-02  8.814e-03  -1.368   0.1713    
## dual_sim      -2.371e-02  1.442e-02  -1.644   0.1004    
## fc             9.348e-04  2.166e-03   0.432   0.6660    
## four_g        -1.477e-03  1.774e-02  -0.083   0.9337    
## int_memory     8.647e-04  3.970e-04   2.178   0.0295 *  
## m_dep         -9.963e-03  2.494e-02  -0.399   0.6896    
## mobile_wt     -8.801e-04  2.030e-04  -4.335 1.53e-05 ***
## n_cores        1.821e-03  3.148e-03   0.579   0.5629    
## pc             1.287e-04  1.551e-03   0.083   0.9339    
## px_height      2.765e-04  1.891e-05  14.626  < 2e-16 ***
## px_width       2.796e-04  1.937e-05  14.437  < 2e-16 ***
## ram            9.472e-04  6.638e-06 142.688  < 2e-16 ***
## sc_h           1.140e-03  1.982e-03   0.575   0.5651    
## sc_w          -3.284e-04  1.915e-03  -0.171   0.8639    
## talk_time      3.639e-04  1.319e-03   0.276   0.7827    
## three_g        2.705e-02  2.079e-02   1.301   0.1934    
## touch_screen  -5.710e-03  1.438e-02  -0.397   0.6914    
## wifi          -2.145e-02  1.440e-02  -1.489   0.1367    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.3206 on 1979 degrees of freedom
## Multiple R-squared:  0.9186,	Adjusted R-squared:  0.9178 
## F-statistic:  1117 on 20 and 1979 DF,  p-value: < 2.2e-16
```

# Clean Data

```r
TestingSet$id = NULL 
```

# Building Random Forest Model

```r
model = randomForest(as.factor(price_range) ~ battery_power + int_memory + mobile_wt + px_height + px_width +ram, data = TrainSet, ntree = 500, importance = TRUE)
```

# Save Model
Here we can see that the error rate is 8.3% which is lower than a previous model that used all the columns and resulted in a error rate of 11.15%

```r
saveRDS(model, "models2.rds")
model
```

```
## 
## Call:
##  randomForest(formula = as.factor(price_range) ~ battery_power +      int_memory + mobile_wt + px_height + px_width + ram, data = TrainSet,      ntree = 500, importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 8.2%
## Confusion matrix:
##     0   1   2   3 class.error
## 0 476  24   0   0       0.048
## 1  28 450  22   0       0.100
## 2   0  32 440  28       0.120
## 3   0   0  30 470       0.060
```

