data {
    int<lower=1> N; // number of days 
    array[N] int cases, deaths;
}

parameters {
    // individual estimates
    vector[N] logit_CFR;

    real logit_CFR_pooled;
    real<lower = 0> tau;
}

model {
    logit_CFR ~ normal(0, 2);
    logit_CFR_pooled ~ normal(0, 2);
    tau ~ cauchy(0, 5);

    // individual estimates
    target += binomial_logit_lupmf(deaths | cases, logit_CFR);
    logit_CFR ~ normal(logit_CFR_pooled, tau);
}

generated quantities {
    vector[N] CFR = inv_logit(logit_CFR);
    real CFR_pooled = inv_logit(logit_CFR_pooled);
}