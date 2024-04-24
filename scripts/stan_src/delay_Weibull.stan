data {
    int<lower=1> N; // number of case records 
    array[N] real exposureL, exposureR, outcomeR;
    int<lower=0> Ncens; // number of left censored case records
    array[N-Ncens] real outcomeL; // observed outcomeL for non-censored records
    
    real logprior_mean, logprior_sd;
}

transformed data {
    int Nobs = N - Ncens;
}

parameters {
    vector<lower=0, upper=1>[N] exposure_raw, outcome_raw;
  
    real logmean_interval, logparam1;
}

transformed parameters {
    real<lower = 0> mean_interval = exp(logmean_interval), 
        param1 = exp(logparam1),
        param2 = mean_interval / tgamma(1.0 + 1.0 / param1),
        sd_interval = param2 * sqrt(tgamma(1.0 + 2.0 / param1) - square(tgamma(1.0 + 1.0 / param1)));
    real logsd_interval = log(sd_interval);
}

model {
    logmean_interval ~ normal(logprior_mean, logprior_sd);
    logsd_interval ~ normal(logprior_mean, logprior_sd);
    // log-transformed jacobian because of using the prior for logsd_inc
    // see https://mc-stan.org/docs/stan-users-guide/changes-of-variables.html
    target += 2*log(param2) - 2*logsd_interval - 2*logparam1 + 
            log(abs(square(tgamma(1.0 + 1.0 / param1)) * digamma(1.0 + 1.0 / param1) - tgamma(1.0 + 2.0 / param1) * digamma(1.0 + 2.0 / param1)));
  
    outcome_raw ~ beta(1, 1); 
    exposure_raw ~ beta(1, 1);

    vector[N] exposure = fma(to_vector(exposureR) - to_vector(exposureL), exposure_raw, to_vector(exposureL)),
        outcome;
    for (n in 1:N) {
        real outcomeL_ = (n > Nobs) ? exposure[n] : ((exposure[n] > outcomeL[n]) ? exposure[n] : outcomeL[n]);
        outcome[n] = fma(outcomeR[n] - outcomeL_, outcome_raw[n], outcomeL_);
    }
    vector[N] time_interval = outcome - exposure; 

    target += weibull_lpdf(time_interval | param1, param2);
}

generated quantities {
    real pred_interval = weibull_rng(param1, param2);
}