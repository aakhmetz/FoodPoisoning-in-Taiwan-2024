data {
    int<lower=1> T; // number of days 
    array[T] int<lower=0> sickened;

    int<lower=1> M; // number of order records
    array[M] int order_day;
    array[M] int<lower=1> orders_lower;
    array[M] int<lower=orders_lower> orders_upper;
}

transformed data {
    array[T] int orders_lower_by_day = rep_array(0, T), orders_upper_by_day = rep_array(0, T);
    for (m in 1:M) {
        orders_lower_by_day[order_day[m]] += orders_lower[m]; 
        orders_upper_by_day[order_day[m]] += orders_upper[m]; 
    }
    print(orders_lower_by_day);
    print(orders_upper_by_day);

    array[T] int<lower=orders_lower_by_day, upper=orders_upper_by_day> exposed = rep_array(0, T);
    for (m in 1:M) 
        exposed[order_day[m]] += (orders_upper[m] > orders_lower[m]) ? discrete_range_rng(orders_lower[m], orders_upper[m]) : orders_lower[m];
    print(exposed);

}

parameters {
    vector[T] logit_AR_t; 
}

transformed parameters {
    vector<lower=0, upper=1>[T] AR_t = inv_logit(logit_AR_t);
    real<lower=0, upper=1> AR = mean(AR_t);
}

model {
    logit_AR_t ~ std_normal();

    target += binomial_lpmf(sickened | exposed, AR_t);
}

generated quantities {
    vector[T] llk;
    for (t in 1:T)
        llk[t] = binomial_lpmf(sickened[t] | exposed[t], AR_t[t]);
}