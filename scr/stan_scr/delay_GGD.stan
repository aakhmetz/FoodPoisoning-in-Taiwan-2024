functions {
    real gengammaloc(real q, real sigma, real logmean) {
        real a = inv_square(q), cinv = sigma / q,
             value = lgamma(a + cinv) - lgamma(a) - log(a) * cinv;
        
        return logmean - value;
    }
  
    real gengammacv(real q, real mu, real sigma) {
        real a = inv_square(q), cinv = sigma / q,
             value = lgamma(a) + lgamma(2*cinv + a) - 2 * lgamma(cinv + a); 

        return sqrt(expm1(value));
    }

    real gengamma_cdf(real x, real q, real mu, real sigma) {
        real logx = log(x), z = (logx - mu) / sigma, a = inv_square(q),
            value = gamma_cdf(a * exp(q * z) | a, 1);

        return value;
    }
    
    /* discretized version of GGD */
    vector dGGD(real q, real mu, real sigma, real xmax, int K) {
        vector[K+1] res; real x_halfstep = 0.5 * xmax / K; 
        for (k in 1:K+1) {
            real x = xmax * k / K;
            res[k] = gengamma_cdf(x - x_halfstep | q, mu, sigma);
        }
        
        return append_row(res[1], tail(res, K) - head(res, K));
    }
}

data {
    int<lower=1> N; // number of case records 
    array[N] real exposureL, exposureR, outcomeR;
    int<lower=0> Ncens; // number of left censored case records
    array[N-Ncens] real outcomeL; // observed outcomeL for non-censored records
    
    real logprior_mean, logprior_sd;

    real<lower = 0> xmax_plt;
    int<lower = 1> K_plt;
}

transformed data {
    int Nobs = N - Ncens;
}

parameters {
    vector<lower=0, upper=1>[N] exposure_raw, outcome_raw;
  
    real loga, logsigma, logmean_interval;
}

transformed parameters {
    real logq = - 2.0 .* loga;
}

model {
    logq ~ std_normal();
    logsigma ~ normal(logprior_mean, logprior_sd);
    logmean_interval ~ normal(logprior_mean, logprior_sd);
    target += logq; // jacobian adjustment
  
    outcome_raw ~ beta(1, 1); 
    exposure_raw ~ beta(1, 1);

    vector[N] exposure = fma(to_vector(exposureR) - to_vector(exposureL), exposure_raw, to_vector(exposureL)),
        outcome;
    for (n in 1:N) {
        real outcomeL_ = (n > Nobs) ? exposure[n] : ((exposure[n] > outcomeL[n]) ? exposure[n] : outcomeL[n]);
        outcome[n] = fma(outcomeR[n] - outcomeL_, outcome_raw[n], outcomeL_);
    }
    vector[N] time_interval = outcome - exposure; 

    real sigma = exp(logsigma);
    real a = exp(loga), q = inv_sqrt(a),
        mu = gengammaloc(q, sigma, logmean_interval);
    vector[N] logx = log(time_interval), z = (logx - mu) / sigma, y = a * exp(q * z);
    target += gamma_lpdf(y | a, 1) + 
        sum(q * z - logx) - N * (logsigma + logq); // jacobian adjustment
}

generated quantities {
    real pred_interval, 
        mean_interval = exp(logmean_interval),
        sd_interval;
  
    real sigma = exp(logsigma), q_ = exp(logq);
    vector[K_plt+1] pdf_interval;
  
    {
        real a = exp(loga), q = inv_sqrt(a), 
            mu = gengammaloc(q, sigma, logmean_interval);
        sd_interval = mean_interval * gengammacv(q, mu, sigma);
        real y = gamma_rng(a, 1);
        real logx = mu + sigma / q * log(y / a);
        pred_interval = exp(logx);
        pdf_interval = dGGD(q, mu, sigma, xmax_plt, K_plt);
    }
}