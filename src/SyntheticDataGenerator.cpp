#include "SyntheticDataGenerator.h"
#include <random>
#include <cmath>

std::vector<double> SyntheticDataGenerator::generateGBM(
    int n,
    double S0,
    double mu,
    double sigma,
    double deltaT,
    unsigned int seed
) {
    std::vector<double> prices;
    prices.reserve(n);

    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    double currentPrice = S0;
    for (int i = 0; i < n; ++i) {
        double Z = dist(gen);
        // S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*deltaT + sigma*sqrt(deltaT)*Z)
        currentPrice *= std::exp((mu - 0.5 * sigma * sigma) * deltaT + sigma * std::sqrt(deltaT) * Z);
        prices.push_back(currentPrice);
    }
    return prices;
}

