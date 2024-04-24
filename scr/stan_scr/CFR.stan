data {
    int<lower=1> N; // number of days 
    array[N] int cases, deaths;
}

parameters {
    // individual estimates
    vector[N] logit_CFR;
}

model {
    logit_CFR ~ normal(0, 2);

    // individual estimates
    target += binomial_logit_lupmf(deaths | cases, logit_CFR);
}

generated quantities {
    vector[N] CFR = inv_logit(logit_CFR);
}