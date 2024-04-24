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
  
    real logsd_interval, logmean_interval;
}

transformed parameters {
    real<lower = 0> mean_interval = exp(logmean_interval),
        sd_interval = exp(logsd_interval),
        // parameters of gamma distribution
        param1 = square(mean_interval / sd_interval),
        param2 = mean_interval / square(sd_interval);
}

model {
    logmean_interval ~ normal(logprior_mean, logprior_sd);
    logsd_interval ~ normal(logprior_mean, logprior_sd);

    outcome_raw ~ beta(1, 1); 
    exposure_raw ~ beta(1, 1);

    vector[N] exposure = fma(to_vector(exposureR) - to_vector(exposureL), exposure_raw, to_vector(exposureL)),
        outcome;
    for (n in 1:N) {
        real outcomeL_ = (n > Nobs) ? exposure[n] : ((exposure[n] > outcomeL[n]) ? exposure[n] : outcomeL[n]);
        outcome[n] = fma(outcomeR[n] - outcomeL_, outcome_raw[n], outcomeL_);
    }
    vector[N] time_interval = outcome - exposure; 

    target += gamma_lpdf(time_interval | param1, param2);
}

generated quantities {
    real pred_interval = gamma_rng(param1, param2);
}