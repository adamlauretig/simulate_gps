Generalized Propensity Score Weighting
================
Adam Lauretig
10/29/2018

Introduction
============

I'm writing this gist to better understand both the generalized propensity score, and marginal structural models, and especially, their intersection. In this, the goal is to estimate how a real-valued treatment can be estimated and understood for a time varying effect. Here, I denote treatment *A* at time *t* as *A*<sub>*t*</sub>, and covariates *X* for individual *i* at time *t* as *X*<sub>*i*, *t*</sub>. We are interested in the average treatment effect *Y*<sub>*A* = 1</sub> − *Y*<sub>*A* = 0</sub> in normal causal inference questions, however, in this case, because *A* is continuous and we're interested in the history of treatment on our outcome *Y*, inference is a little more difficult. Drawing on work examining the Generalized Propensity Score (Hirano and Imbens 2004; Austin 2018), I simulate covariates, calculate inverse propensity score weights, and ultimately, estimate a reponse model.

Marginal structural models were originally developed in epidemiology (Robins, Hernan, and Brumback 2000), and present a way to estimate potential outcomes for an outcome whose counterfactual depends on treatment *history*, rather than simply a static treatment. Estimating the inverse probability of treatment weights is similar to other propensity score methods, however, these weights are then used to weight a second stage regression, where the outcome is regressed on the treatment history.

This post assumes a knowledge of the Rubin Causal Model ("Potential Outcomes"), and the R package `data.table()`.

Setting up parameters for simulation
====================================

Here, I assume there are 3 covariates, 50 individuals, ten time periods, and a lag length of 1. That is, *A*<sub>*t*</sub>|*X*<sub>*t*</sub>, *A*<sub>*t* − 1</sub>, *X*<sub>*t* − 1</sub> and *X*<sub>*t*</sub>|*X*<sub>*t* − 1</sub>, *A*<sub>*t* − 1</sub>. We'll generate *X* and *A* in steps, for each of our 3 time periods, along with a constant unobserved confounder *U* following Havercroft and Didelez (2012). Since my interest here is in a continuous treatment, I'll use OLS to generate *A*<sub>*t*</sub>, plus some Gaussian *ϵ* ∼ *N*(0, 1) noise. The challege here comes from incorporating the lagged treatment as well, and how it affects *X*<sub>*t*</sub> and *A*<sub>*t*</sub>, which means that we have to calculate treatment values at each time *t*.

``` r
library(data.table)
library(ggplot2)
library(mgcv)
set.seed(614L)
#number of covariates
p <- 3

# number of individuals
I <- 1000
ids <- paste0("i_", 1:I)

# number of time periods
t <- 3L

treatment_coefs <- matrix(rnorm(n = (2*p) + 1, mean = .5, sd = .25), nrow = 1)

# a time-invariant unobserved confounder
U <- matrix(runif(n = I, 0, 1), ncol= 1)
# unobserved confounder effects, 1 for each time period
U_effect <- rnorm(2, 1, 3)



# now, generating contemporaneous covariates
simulate_covariates <- function(time_period, units, columns){
  set.seed(time_period)
  X <- matrix(runif(units*columns, -1, 1), nrow = units)
  for(i in 1:columns){
    X[, i] <- X[, i] + U%*%U_effect[1]
  }
  covariates <- data.table(id = 1:units, time = time_period, X)
  return(covariates)
}
X1 <- simulate_covariates(time_period = 1, units = I, columns = p)
# now, adding our "lagged" variables. all zeroes since this is t = 1
X1[, `:=`(
  lag_1_V1 = 0,
  lag_1_V2 = 0,
  lag_1_V3 = 0,
  lag_1_treatment = 0
)]

eps <- matrix(rnorm(I, 0, 1), ncol = 1)

A1 <- as.matrix(X1[, 3:9]) %*% t(treatment_coefs) + eps
time_1_data <- data.table(X1[, 1:5], treatment = A1, X1[, 6:9])
setnames(time_1_data, c("treatment.V1"), c("treatment"))

# for generating X2 and X3
new_data_coefs <- rnorm(4, 1, 2)
```

Now, we'll generate treatments for each successive time period. Since the data are conditioned on the past treatment, and the past data, simulating this will be a little funky. I'll do each period by hand, but there's probably a way to do this which will scale more effectively.

``` r
# time 2 starts
time_2_data <- as.data.table(time_1_data)
time_2_data[, time := time + 1]
time_2_data[, 7:10] <- time_2_data[, 3:6]
Xt1_temp <- as.matrix(time_2_data[, 7:9])
Xt2 <- matrix(NA, nrow = I, ncol = p)
# generate new covariates
for( i in 1:3){
 Xt2[, i] <- (as.matrix(Xt1_temp[, i]) %*% new_data_coefs[i]) + as.matrix(time_2_data[, 10]) %*% new_data_coefs[4] 
 Xt2[, i] <- Xt2[, i] + U %*% U_effect[1]
}

for(i in 1:p){
  time_2_data[, (i + 2)] <- Xt2[, i]
}
eps <- matrix(rnorm(I, 0, 1), ncol = 1)
A2 <- as.matrix(time_2_data[, c(3:5, 7:10)])%*% t(treatment_coefs) + eps
time_2_data[, treatment := A2]

# time 3

time_3_data <- as.data.table(time_2_data)
time_3_data[, time := time + 1]
time_3_data[, 7:10] <- time_2_data[, 3:6]
Xt3 <- matrix(NA, nrow = I, ncol = p)
# generate new covariates
for( i in 1:3){
 Xt3[, i] <- (as.matrix(Xt1_temp[, i]) %*% new_data_coefs[i]) + as.matrix(time_2_data[, 10]) %*% new_data_coefs[4] 
 Xt3[, i] <- Xt3[, i] + U %*% U_effect[1]
}

for(i in 1:p){
  time_3_data[, (i + 3)] <- Xt3[, i]
}
eps <- matrix(rnorm(I, 0, 1), ncol = 1)
A3 <- as.matrix(time_3_data[, c(3:5, 7:10)])%*% t(treatment_coefs) + eps
time_3_data[, treatment := A3]

covariates <- rbindlist(list(time_1_data, time_2_data, time_3_data))
```

Now, we want to generate the outcome variable *Y*. Since we have a continuous treatment *A*, we divide the treatment into quantiles (Austin 2018), and assign a treatment effect to each quantile. We then simulate our potential outcome for each combination of quantiles over time. Here, we'll set up terciles, split at 33% and 67%, assign treatment values, and generate our outcome. We'll treat the treatment value below 33% as the default. I've chosen to hard-code the coefficients for the outcome model here, but you could replace them with `rnorm()`. For personal reasons, I'm interested in the effect on a poisson-distributed outcome, however, as I provide the code here, feel free to modify.

``` r
cutpoints <- quantile(covariates$treatment, c(.33, .67))
covariates[,`:=`(
  treatment_low = ifelse(treatment < cutpoints[1], 1, 0),
  treatment_mid = ifelse(treatment >= cutpoints[1] & 
      treatment < cutpoints[2], 1, 0),
  treatment_high = ifelse(treatment >= cutpoints[2], 1, 0)
)]

# Now, the outcomes
# create the matrix of covariates (we'll treat the low-range as the default)

treatment_history_dt <- dcast.data.table(covariates, id ~ time, value.var = c("treatment_low", "treatment_mid", "treatment_high"))
X_matrix1 <- as.matrix(treatment_history_dt[, 5:10])
X_matrix1 <- cbind(1, X_matrix1)
# treatments for intercept, time periods 1, 2, 3 for mid and high, respectively
outcome_beta <- c(seq(-5, 1, 1), U_effect[2])
X_matrix <- cbind(X_matrix1, U)
# treatment * treatment effect, and then, multiplied by weights
Y <- (X_matrix%*% outcome_beta) + rnorm(I, 0, 3)
treatment_history_dt[, Y := Y]
# what if we have a poisson outcome?
treatment_history_dt[, Y_lambda := exp(Y)]
treatment_history_dt[, Y_pois := rpois(n = I, Y_lambda)]
outcome_beta
```

    ## [1] -5.000000 -4.000000 -3.000000 -2.000000 -1.000000  0.000000  1.000000
    ## [8]  6.617358

Recovering initial parameters
=============================

Now we'll see how we do at recovering our original parameters.

``` r
observed_data <- dcast.data.table(covariates, id~time, value.var = c("V1", "V2", "V3", "treatment"))
final_data <-  merge(treatment_history_dt, observed_data, by = "id")
```

Let's start with a simple regression, including covariates, w/our continuous treatment:

``` r
m1 <- glm(Y_pois ~ treatment_1 + treatment_2 + treatment_3 + 
    V1_1 + V1_2 + V1_3 + 
    V2_1 + V2_2 + V2_3 + 
    V3_1 + V3_2 + V3_3, data = final_data, family = "poisson")
summary(m1)
```

    ## 
    ## Call:
    ## glm(formula = Y_pois ~ treatment_1 + treatment_2 + treatment_3 + 
    ##     V1_1 + V1_2 + V1_3 + V2_1 + V2_2 + V2_3 + V3_1 + V3_2 + V3_3, 
    ##     family = "poisson", data = final_data)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -17.151   -3.857   -2.241   -1.120   74.868  
    ## 
    ## Coefficients: (5 not defined because of singularities)
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  -0.61719    0.03893  -15.85   <2e-16 ***
    ## treatment_1 -10.14704    0.11127  -91.19   <2e-16 ***
    ## treatment_2  -0.35349    0.01060  -33.34   <2e-16 ***
    ## treatment_3  -0.23807    0.00923  -25.79   <2e-16 ***
    ## V1_1         11.04304    0.10240  107.84   <2e-16 ***
    ## V1_2          4.90742    0.05126   95.73   <2e-16 ***
    ## V1_3               NA         NA      NA       NA    
    ## V2_1         -0.83979    0.02042  -41.13   <2e-16 ***
    ## V2_2               NA         NA      NA       NA    
    ## V2_3               NA         NA      NA       NA    
    ## V3_1         -0.84228    0.02008  -41.94   <2e-16 ***
    ## V3_2               NA         NA      NA       NA    
    ## V3_3               NA         NA      NA       NA    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 67487  on 999  degrees of freedom
    ## Residual deviance: 49523  on 992  degrees of freedom
    ## AIC: 50341
    ## 
    ## Number of Fisher Scoring iterations: 8

Now, let's use our tercile treatments:

``` r
m2 <- glm(Y_pois ~ treatment_mid_1 + treatment_mid_2 + treatment_mid_3 + treatment_high_1 + treatment_high_2 + treatment_high_3, data = final_data, family = "poisson")
summary(m2)
```

    ## 
    ## Call:
    ## glm(formula = Y_pois ~ treatment_mid_1 + treatment_mid_2 + treatment_mid_3 + 
    ##     treatment_high_1 + treatment_high_2 + treatment_high_3, family = "poisson", 
    ##     data = final_data)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -10.166   -1.597   -1.378   -0.284   80.388  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)       0.25367    0.08275   3.065 0.002174 ** 
    ## treatment_mid_1  -3.70155    0.05651 -65.503  < 2e-16 ***
    ## treatment_mid_2  -1.24377    0.64993  -1.914 0.055658 .  
    ## treatment_mid_3  -2.22334    0.60918  -3.650 0.000263 ***
    ## treatment_high_1       NA         NA      NA       NA    
    ## treatment_high_2  2.75248    0.65672   4.191 2.77e-05 ***
    ## treatment_high_3  0.93880    0.66021   1.422 0.155032    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 67487  on 999  degrees of freedom
    ## Residual deviance: 39299  on 994  degrees of freedom
    ## AIC: 40113
    ## 
    ## Number of Fisher Scoring iterations: 7

and we see that our estimates in both cases of the treatment effects are quite different than `outcome_beta` above. I'll also point out the `NA` for `treatment_high_1`, as there are no treatment values at time 1 which are in the top 3<sup>*r**d*</sup> tercile.

Weighting
=========

However, we can weight our models with the probability of treatment, which allows us to condition on the covariates which confound selection, throughout the treatment history, a technique originally developed in (Robins, Hernan, and Brumback 2000). Here, I'm grabbing the `covariates` data.table from above, but only keeping those variables we would expect to see in the actual dataset (mostly because the format is easier initially). It's important to note that throughout the process of estimating the weights, we do not work with our outcome *Y*, right now, we're interested in *A*<sub>*t* = 1</sub>, *A*<sub>*t* = 2</sub>, *A*<sub>*t* = 3</sub> and *X*<sub>*t*</sub> and *X*<sub>*t* − 1</sub>.

``` r
observed_covariates <- covariates[,.(id, time, V1, V2, V3, treatment)]
# observed_covariates <- observed_covariates[ id %in% sample_ids]
head(observed_covariates)
```

    ##    id time        V1         V2        V3 treatment
    ## 1:  1    1 1.4772656 2.00786587 2.6898583 5.6703968
    ## 2:  2    1 2.1898973 2.81537133 3.3800436 5.6278032
    ## 3:  3    1 1.7195138 1.34037390 2.3076397 5.0996892
    ## 4:  4    1 3.1863947 3.27995514 2.2454097 5.9033757
    ## 5:  5    1 0.2455929 0.07894221 0.2261046 0.9730124
    ## 6:  6    1 2.9160902 1.19751095 1.2838997 4.0934380

``` r
observed_covariates[, normalized_treatment := (treatment - mean(treatment))/sd(treatment) ]
```

Now, to make our weights. Following (Austin 2018), we'll perform OLS regression on our treatment, as our outcome is continuous. Then, for the denominator of our inverse propensity score weights, to calculate the probability of a particular treatment *a*<sup>\*</sup>, we will calculate *P**r*(*A* = *a*|*a*<sup>\*</sup>, *σ*<sub>*a*<sup>\*</sup></sub>). The stabilizing numerator is *P**r*(*A*<sup>\*</sup> = *a*<sup>\*</sup>|0, 1) (Z. Zhang et al. 2016). Here, we know that we need a single lag of our data to estimate the propensity scores. We then take the product of our stabilized weights from each time period, in order to produce individual-specific weights.

``` r
observed_covariates_lag <- as.data.table(observed_covariates)
observed_covariates_lag[, time := time + 1]
setnames(observed_covariates_lag, 
  c("V1", "V2", "V3", "treatment", "normalized_treatment"), 
  c("lag1_V1", "lag1_V2", "lag1_V3", "lag1_treatment", "lag1_normalized_treatment"))
observed_covariates_2 <- merge(observed_covariates, observed_covariates_lag, 
  by = c("id", "time"), all.x = TRUE)
observed_covariates_2[, `:=`(
  lag1_treatment = ifelse(is.na(lag1_treatment), 0, lag1_treatment),
  lag1_normalized_treatment = ifelse(
    is.na(lag1_normalized_treatment), 0, lag1_normalized_treatment),
  lag1_V1 = ifelse(is.na(lag1_V1), 0, lag1_V1),
  lag1_V2 = ifelse(is.na(lag1_V2), 0, lag1_V2),
  lag1_V3 = ifelse(is.na(lag1_V3), 0, lag1_V3)
)]

a_star_model <- lm(normalized_treatment ~ lag1_normalized_treatment + V1 + V2 + V3 + lag1_V1 + lag1_V2 + lag1_V3, data = observed_covariates_2)
# observed_covariates[, a_star := predict(a_star_model,type =  "response")]
observed_covariates[, ipw_denominator := dnorm(normalized_treatment, mean = a_star_model$fitted.values, sd = sd(a_star_model$fitted.values)) ]
# observed_covariates[, dnorm(normalized_treatment, mean = a_star_model$fitted.values, sd = summary(a_star_model)$sigma) ]

observed_covariates[, ipw_numerator := dnorm(normalized_treatment, mean = 0, sd = 1) ]
observed_covariates[, ipw_weight := exp(log(ipw_numerator)-log(ipw_denominator))]


wt_cutpoints <- quantile(observed_covariates$ipw_weight[!(is.infinite(observed_covariates$ipw_weight))], probs = c(.01, .99), na.rm = TRUE)

observed_covariates[, ipw_weight_trunc := ifelse(ipw_weight < wt_cutpoints[1], wt_cutpoints[1], ipw_weight)]
observed_covariates[, ipw_weight_trunc := ifelse(ipw_weight > wt_cutpoints[2], wt_cutpoints[2], ipw_weight)]
# observed_covariates[, ipw_weight_trunc := ifelse(is.infinite(wt_cutpoints), wt_cutpoints[2], wt_cutpoints)]
ids_weights <- observed_covariates[, prod(ipw_weight_trunc), by = .(id)]
```

We'll then regress our observed outcome, *Y*, on our treatment history, which we'll break into terciles, following the way we generated our data above. Note here that until we regress *Y* on *A*, we still are only focused on *A*, *Y* does not enter into our estimation at all.

``` r
# getting cutpoints for terciles
cut_values <- quantile(observed_covariates$treatment, probs = c(.33, .67))

observed_covariates[,`:=`(
  treatment_low = ifelse(treatment < cut_values[1], 1, 0),
  treatment_mid = ifelse(treatment >= cut_values[1] & 
      treatment < cut_values[2], 1, 0),
  treatment_high = ifelse(treatment >= cut_values[2], 1, 0)
)]

treatment_history <- dcast.data.table(observed_covariates, id ~ time, value.var = c("treatment_low", "treatment_mid", "treatment_high"))
```

This is a **wide** dataset. We only observe *Y* at the "end" of the treatment history, however, we have an entire sequence of treatments *A* we regress *Y* on. We'll now regress *Y* on *A*, the treatment history, and include the weights in the `weights` argument of `lm()`. We'll treat the lowest tercile of *A* as the default, and include the "mid" and "high" terciles in the model. To include our inverse probability of treatment weights, we'll use the `survey` package.

``` r
treatment_history[, Y := final_data$Y]
treatment_history[, Y_pois := final_data$Y_pois]

treatment_history[, ipw_weights := ids_weights$V1]

library(survey)

design1 <- survey::svydesign(ids = ~factor(id), 
    data = treatment_history, weights = ~ (ipw_weights))

m3 <- survey::svyglm(Y_pois ~ treatment_mid_1 + treatment_mid_2 + treatment_mid_3 + treatment_high_1 + treatment_high_2 + treatment_high_3,  design = design1, family=quasipoisson())
summary(m3)
```

    ## 
    ## Call:
    ## svyglm(formula = Y_pois ~ treatment_mid_1 + treatment_mid_2 + 
    ##     treatment_mid_3 + treatment_high_1 + treatment_high_2 + treatment_high_3, 
    ##     design = design1, family = quasipoisson())
    ## 
    ## Survey design:
    ## survey::svydesign(ids = ~factor(id), data = treatment_history, 
    ##     weights = ~(ipw_weights))
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)       -1.7647     0.5092  -3.466 0.000552 ***
    ## treatment_mid_1   -3.9502     0.6039  -6.542 9.72e-11 ***
    ## treatment_mid_2    0.3124     1.2219   0.256 0.798279    
    ## treatment_mid_3   -2.3257     1.0303  -2.257 0.024199 *  
    ## treatment_high_2   3.5822     1.4122   2.537 0.011343 *  
    ## treatment_high_3   1.8727     1.4438   1.297 0.194900    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for quasipoisson family taken to be 42.8374)
    ## 
    ## Number of Fisher Scoring iterations: 7

To examine how well our models perform, I calculate the mean squared error of the difference between the true coefficients, and the estimated coefficients.

``` r
# Mean squared error on coefficients from the naive treatment regression, and
# the ipw-weighted regression

true_coef <- outcome_beta[c(1:4, 6:7)]
m2_coef <- coef(m2)[c(1:4, 6:7)]
m3_coef <- na.omit(coef(m3))
mean((true_coef - m2_coef)^2)
```

    ## [1] 6.400712

``` r
mean((true_coef - m3_coef)^2)
```

    ## [1] 5.856875

While this isn't perfect, it is much better than our naive model, which only regresses treatments on the outcome.

Conclusion
==========

In this post, I illustrated how to estimate marginal structural models with a continuous treatment, and a count-valued outcome. I simulated the treatment regime, generated an outcome, and then, estimated marginal structual models from the original data. If this method seems interesting to you, I would encourage you to check out any of the previously cited works, or Blackwell and Glynn (2018).

Bibliography
============

Austin, Peter C. 2018. “Assessing the Performance of the Generalized Propensity Score for Estimating the Effect of Quantitative or Continuous Exposures on Binary Outcomes.” *Statistics in Medicine* 37 (11). Wiley Online Library: 1874–94.

Blackwell, Matthew, and Adam Glynn. 2018. “How to Make Causal Inferences with Time-Series Cross-Sectional Data Under Selection on Observables.”

Havercroft, W.G., and V. Didelez. 2012. “Simulating from Marginal Structural Models with Time-Dependent Confounding.” *Statistics in Medicine* 31 (30). Wiley Online Library: 4190–4206.

Hirano, Keisuke, and Guido W. Imbens. 2004. “The Propensity Score with Continuous Treatments.” *Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives* 226164: 73–84.

Robins, James M., Miguel Angel Hernan, and Babette Brumback. 2000. “Marginal Structural Models and Causal Inference in Epidemiology.” *Epidemiology* 11 (5).

Zhang, Zhiwei, Jie Zhou, Weihua Cao, and Jun Zhang. 2016. “Causal Inference with a Quantitative Exposure.” *Statistical Methods in Medical Research* 25 (1). SAGE Publications Sage UK: London, England: 315–35.
