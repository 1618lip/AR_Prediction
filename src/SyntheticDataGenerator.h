#ifndef SYNTHETIC_DATA_GENERATOR_H
#define SYNTHETIC_DATA_GENERATOR_H

#include <vector>

class SyntheticDataGenerator {
public:
    // Generate synthetic stock prices via Geometric Brownian Motion.
    // n: number of time points
    // S0: initial price
    // mu: drift
    // sigma: volatility
    // deltaT: time increment (e.g., 1/252 for daily)
    // seed: random seed for reproducibility
    static std::vector<double> generateGBM(
        int n,
        double S0,
        double mu,
        double sigma,
        double deltaT,
        unsigned int seed = 0
    );
};

#endif


