# Parameter recovery

| dataset | model | parameter | true | estimate | se | ci_low | ci_high | covered | z_score | expect_recovery |
|---|---|---|---|---|---|---|---|---|---|---|
| ols_data | OLS | Intercept | 2.0000 | 2.4961 | 0.4513 | 1.6060 | 3.3862 | True | 1.0991 | True |
| ols_data | OLS | X1 | 1.5000 | 1.3941 | 0.0568 | 1.2821 | 1.5060 | True | -1.8656 | True |
| ols_data | OLS | X2 | -0.7000 | -0.7101 | 0.0357 | -0.7805 | -0.6397 | True | -0.2826 | True |
| ols_data | OLS | sigma | 1.5000 | 1.4848 | 0.0748 | 1.3381 | 1.6314 | True | -0.2038 | True |
| glm_poisson | GLM Poisson (log) | Intercept | 0.5000 | 0.4608 | 0.0464 | 0.3697 | 0.5518 | True | -0.8449 | True |
| glm_poisson | GLM Poisson (log) | X | 0.9000 | 0.9097 | 0.0147 | 0.8808 | 0.9386 | True | 0.6596 | True |
| glm_logistic | Logit | Intercept | -0.5000 | -0.4919 | 0.1276 | -0.7419 | -0.2419 | True | 0.0636 | True |
| glm_logistic | Logit | X | 0.8000 | 0.8617 | 0.1518 | 0.5642 | 1.1592 | True | 0.4066 | True |
| arima_series | ARIMA(1,0,1) | ar.L1 | 0.8000 | 0.8414 | 0.0353 | 0.7722 | 0.9105 | True | 1.1733 | True |
| arima_series | ARIMA(1,0,1) | ma.L1 | 0.4000 | 0.3456 | 0.0664 | 0.2155 | 0.4758 | True | -0.8185 | True |
| arima_series | ARIMA(1,0,1) | sigma2 | 1.0000 | 0.9344 | 0.0922 | 0.7536 | 1.1151 | True | -0.7118 | True |
| manova_data | OLS group contrast | mean_diff_Y1 | 2.0000 | 1.7055 | 0.1142 | 1.4807 | 1.9303 | False | -2.5784 | True |
| manova_data | OLS group contrast | mean_diff_Y2 | 2.0000 | 1.7329 | 0.1155 | 1.5057 | 1.9601 | False | -2.3133 | True |
| heteroskedastic_data | OLS + HC3 | Intercept | 3.0000 | 2.9218 | 0.1990 | 2.5318 | 3.3118 | True | -0.3930 | True |
| heteroskedastic_data | OLS + HC3 | X | 2.0000 | 2.4171 | 0.2610 | 1.9056 | 2.9286 | True | 1.5981 | True |
| multivariate_group_data | sample mean | Num1_mean_A | 50.0000 | 50.0684 | 0.4388 | 49.2083 | 50.9285 | True | 0.1558 | True |
| multivariate_group_data | sample mean | Num1_mean_B | 60.0000 | 60.8804 | 0.6447 | 59.6168 | 62.1440 | True | 1.3656 | True |
| multivariate_group_data | sample mean | Num5_mean_A | 0.0000 | -1.4777 | 2.1818 | -5.7541 | 2.7987 | True | -0.6773 | True |
| multivariate_group_data | sample mean | Num5_mean_B | 20.0000 | 15.5862 | 2.6178 | 10.4553 | 20.7171 | True | -1.6861 | True |
| ols_diagnostics | OLS (contaminated) | Intercept | 5.0000 | 12.6811 | 5.4230 | 2.0306 | 23.3317 | True | 1.4164 | False |
| ols_diagnostics | OLS (contaminated) | X1 | 1.5000 | 1.4533 | 0.0665 | 1.3227 | 1.5839 | True | -0.7024 | False |
| ols_diagnostics | OLS (contaminated) | X2 | -2.0000 | -2.0618 | 0.1291 | -2.3154 | -1.8082 | True | -0.4789 | False |
| ols_diagnostics | OLS (contaminated) | X3 | 0.3000 | 0.2588 | 0.0249 | 0.2100 | 0.3076 | True | -1.6584 | False |
| ols_diagnostics | RLM Huber | Intercept | 5.0000 | 8.9218 | 3.8104 | 1.4536 | 16.3901 | True | 1.0292 | True |
| ols_diagnostics | RLM Huber | X1 | 1.5000 | 1.4729 | 0.0467 | 1.3813 | 1.5645 | True | -0.5801 | True |
| ols_diagnostics | RLM Huber | X2 | -2.0000 | -2.0324 | 0.0907 | -2.2102 | -1.8545 | True | -0.3567 | True |
| ols_diagnostics | RLM Huber | X3 | 0.3000 | 0.2704 | 0.0175 | 0.2362 | 0.3047 | True | -1.6937 | True |
| posthoc_dataset | group mean | mean_A | 60.0000 | 58.9615 | 0.9082 | 57.1815 | 60.7415 | True | -1.1435 | True |
| posthoc_dataset | group mean | mean_B | 70.0000 | 70.2677 | 1.1444 | 68.0246 | 72.5107 | True | 0.2339 | True |
| posthoc_dataset | group mean | mean_C | 65.0000 | 65.5192 | 0.8674 | 63.8190 | 67.2193 | True | 0.5985 | True |
| robust_regression_data | OLS (contaminated) | Intercept | 5.0000 | 4.7673 | 2.3126 | 0.2068 | 9.3278 | True | -0.1006 | False |
| robust_regression_data | OLS (contaminated) | X | 2.5000 | 2.6375 | 0.2253 | 2.1932 | 3.0819 | True | 0.6103 | False |
| robust_regression_data | RLM Huber | Intercept | 5.0000 | 4.2337 | 1.0519 | 2.1720 | 6.2954 | True | -0.7285 | True |
| robust_regression_data | RLM Huber | X | 2.5000 | 2.6307 | 0.1025 | 2.4298 | 2.8316 | True | 1.2753 | True |
| robust_regression_data | Median regression | Intercept | 5.0000 | 4.4608 | 1.2643 | 1.9676 | 6.9539 | True | -0.4265 | True |
| robust_regression_data | Median regression | X | 2.5000 | 2.6066 | 0.1232 | 2.3636 | 2.8495 | True | 0.8650 | True |
| seasonal_ts_data | Harmonic regression | level | 50.0000 | 50.1335 | 0.3804 | 49.3854 | 50.8817 | True | 0.3510 | True |
| seasonal_ts_data | Harmonic regression | trend | 0.0500 | 0.0511 | 0.0009 | 0.0493 | 0.0530 | True | 1.1841 | True |
| seasonal_ts_data | Harmonic regression | amplitude | 10.0000 | 9.9793 | 0.1400 | 9.7039 | 10.2547 | True | -0.1480 | True |
| seasonal_ts_data | Harmonic regression | exog_effect | 0.0000 | -0.0637 | 0.0649 | -0.1913 | 0.0639 | True | -0.9817 | True |
| panel_data | Fixed effects (LSDV) | X1 | 1.5000 | 1.4876 | 0.0229 | 1.4425 | 1.5326 | True | -0.5429 | True |
| panel_data | Fixed effects (LSDV) | X2 | -0.8000 | -0.8028 | 0.0305 | -0.8628 | -0.7429 | True | -0.0926 | True |
| panel_data | MixedLM random intercept | Intercept | 10.0000 | 11.3277 | 0.7935 | 9.7725 | 12.8830 | True | 1.6732 | True |
| panel_data | MixedLM random intercept | X1 | 1.5000 | 1.4876 | 0.0229 | 1.4427 | 1.5325 | True | -0.5416 | True |
| panel_data | MixedLM random intercept | X2 | -0.8000 | -0.8032 | 0.0305 | -0.8630 | -0.7434 | True | -0.1058 | True |
| panel_data | MixedLM random intercept | sd_individual | 5.0000 | 5.2488 |  |  |  | False |  | False |
| survival_data | Cox PH | age | 0.0200 | 0.0303 | 0.0073 | 0.0159 | 0.0446 | True | 1.4020 | True |
| survival_data | Cox PH | treatment | -0.5000 | -0.7835 | 0.1698 | -1.1164 | -0.4507 | True | -1.6696 | True |
| survival_data | Cox PH | biomarker | 0.0100 | 0.0103 | 0.0042 | 0.0019 | 0.0186 | True | 0.0599 | True |
| zero_inflated_count | ZIP | Intercept | 0.5000 | 0.4234 | 0.0727 | 0.2809 | 0.5660 | True | -1.0525 | True |
| zero_inflated_count | ZIP | x1 | 0.6000 | 0.6223 | 0.0251 | 0.5730 | 0.6715 | True | 0.8855 | True |
| zero_inflated_count | ZIP | x2 | 0.3000 | 0.3215 | 0.0192 | 0.2839 | 0.3592 | True | 1.1199 | True |
| zero_inflated_count | ZIP | inflate_Intercept | -0.5000 | -0.4917 | 0.1978 | -0.8793 | -0.1041 | True | 0.0419 | True |
| zero_inflated_count | ZIP | inflate_x3 | 0.7000 | 0.6867 | 0.1910 | 0.3124 | 1.0611 | True | -0.0694 | True |
| var_data | VAR(1) | L1.y1->y1 | 0.5000 | 0.4655 | 0.0565 | 0.3547 | 0.5763 | True | -0.6098 | True |
| var_data | VAR(1) | L1.y2->y1 | 0.2000 | 0.1696 | 0.0485 | 0.0747 | 0.2646 | True | -0.6265 | True |
| var_data | VAR(1) | L1.y1->y2 | 0.3000 | 0.3459 | 0.0546 | 0.2388 | 0.4530 | True | 0.8401 | True |
| var_data | VAR(1) | L1.y2->y2 | 0.6000 | 0.5680 | 0.0468 | 0.4762 | 0.6598 | True | -0.6826 | True |
| gee_data | GEE exchangeable | Intercept | -1.0000 | -0.8054 | 0.3829 | -1.5559 | -0.0550 | True | 0.5081 | False |
| gee_data | GEE exchangeable | X | 0.5000 | 0.2866 | 0.0668 | 0.1557 | 0.4174 | False | -3.1965 | False |
| gee_data | GEE exchangeable | treatment | 0.8000 | 0.5722 | 0.2136 | 0.1536 | 0.9909 | True | -1.0663 | False |
| mediation_data | Mediator model | a | 0.7000 | 0.6802 | 0.0283 | 0.6246 | 0.7359 | True | -0.6998 | True |
| mediation_data | Outcome model | b | 0.6000 | 0.5455 | 0.0962 | 0.3562 | 0.7347 | True | -0.5671 | True |
| mediation_data | Outcome model | c_prime | 0.4000 | 0.4065 | 0.0805 | 0.2480 | 0.5649 | True | 0.0802 | True |
| mediation_data | Moderation model | mod_X | 0.5000 | 0.5530 | 0.0485 | 0.4575 | 0.6486 | True | 1.0927 | True |
| mediation_data | Moderation model | mod_W | 0.3000 | 0.2768 | 0.0459 | 0.1865 | 0.3670 | True | -0.5068 | True |
| mediation_data | Moderation model | mod_XW | 0.4000 | 0.3433 | 0.0444 | 0.2559 | 0.4307 | True | -1.2766 | True |
