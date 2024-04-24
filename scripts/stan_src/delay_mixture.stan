data {
    int<lower=1> N; // number of case records 
    array[N] real exposureL, exposureR, outcomeR;
    int<lower=0> Ncens; // number of left censored case records
    array[N-Ncens] real outcomeL; // observed outcomeL for non-censored records
    
    real logprior_mean, logprior_sd;
}

transformed data {
    int D = 3;
    
    int Nobs = N - Ncens;
}

parameters {
    vector<lower=0, upper=1>[N] exposure_raw, outcome_raw;
  
    real logmean_interval, logparam1_weibull;
    
    simplex[D] weight; // mixing proportions
}

transformed parameters {
    real<lower = 0> mean_interval = exp(logmean_interval), sd_interval;
    real logsd_interval;
    
    vector[D] param1, param2,
        lps = log(weight); // internal component likelihoods 
    {
        vector[N] exposure = fma(to_vector(exposureR) - to_vector(exposureL), exposure_raw, to_vector(exposureL)),
            outcome;
        for (n in 1:N) {
            real outcomeL_ = (n > Nobs) ? exposure[n] : ((exposure[n] > outcomeL[n]) ? exposure[n] : outcomeL[n]);
            outcome[n] = fma(outcomeR[n] - outcomeL_, outcome_raw[n], outcomeL_);
        }
        vector[N] time_interval = outcome - exposure; 

        // Weibull distribution
        param1[2] = exp(logparam1_weibull);
        param2[2] = mean_interval / tgamma(1.0 + 1.0 / param1[2]); 
        sd_interval = param2[2] * sqrt(tgamma(1.0 + 2.0 / param1[2]) - square(tgamma(1.0 + 1.0 / param1[2])));
        logsd_interval = log(sd_interval);
        
        // Gamma distribution
        param1[1] = square(mean_interval / sd_interval);
        param2[1] = mean_interval / square(sd_interval);

        // Lognormal distribution
        param2[3] = sqrt(log(square(sd_interval / mean_interval) + 1.0));
        param1[3] = log(mean_interval) - square(param2[3]) / 2.0;

        lps[1] += gamma_lpdf(time_interval | param1[1], param2[1]);
        lps[2] += weibull_lpdf(time_interval | param1[2], param2[2]);
        lps[3] += lognormal_lpdf(time_interval | param1[3], param2[3]);
    }
}

model {
    logmean_interval ~ normal(logprior_mean, logprior_sd);
    logsd_interval ~ normal(logprior_mean, logprior_sd);
    // log-transformed jacobian because of using the prior for logsd_inc
    // see https://mc-stan.org/docs/stan-users-guide/changes-of-variables.html
    target += 2*log(param2[2]) - 2*logsd_interval - 2*logparam1_weibull + 
            log(abs(square(tgamma(1.0 + 1.0 / param1[2])) * digamma(1.0 + 1.0 / param1[2]) - tgamma(1.0 + 2.0 / param1[2]) * digamma(1.0 + 2.0 / param1[2])));
  
    outcome_raw ~ beta(1, 1); 
    exposure_raw ~ beta(1, 1);

    target += log_sum_exp(lps);
}

generated quantities {
    vector<lower = 0, upper = 1>[D] prob = exp(lps - log_sum_exp(lps));
    int comp = categorical_rng(prob);
    real pred_interval;

    if (comp == 2) 
        pred_interval = weibull_rng(param1[comp], param2[comp]);
    else if (comp == 1)
        pred_interval = gamma_rng(param1[comp], param2[comp]);
    else
        pred_interval = lognormal_rng(param1[comp], param2[comp]);
}