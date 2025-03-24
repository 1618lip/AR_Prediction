#include "ARModel.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

ARModel::ARModel(const std::vector<double>& data, int order)
    : data_(data), order_(order)
{
}

void ARModel::computeAutocorrelation() {
    int n = data_.size();
    autocorrelation_.resize(order_ + 1, 0.0);

    for (int lag = 0; lag <= order_; ++lag) {
        double sum = 0.0;
        for (int i = lag; i < n; ++i) {
            sum += data_[i] * data_[i - lag];
        }
        autocorrelation_[lag] = sum / n;
    }
}

bool ARModel::computeCoefficients() {
    if (data_.size() < static_cast<size_t>(order_)) {
        std::cerr << "Not enough data to compute AR coefficients.\n";
        return false;
    }

    computeAutocorrelation();
    std::vector<double> a(order_ + 1, 0.0);
    std::vector<double> e(order_ + 1, 0.0);

    a[0] = 1.0;
    e[0] = autocorrelation_[0];
    if (autocorrelation_[0] == 0.0) {
        std::cerr << "Zero lag autocorrelation. Cannot compute coefficients.\n";
        return false;
    }

    // Levinson-Durbin
    for (int k = 1; k <= order_; ++k) {
        double lambda = 0.0;
        for (int j = 1; j < k; ++j) {
            lambda += a[j] * autocorrelation_[k - j];
        }
        lambda = (autocorrelation_[k] - lambda) / e[k - 1];

        a[k] = lambda;
        for (int j = 1; j < k; ++j) {
            a[j] -= lambda * a[k - j];
        }
        e[k] = e[k - 1] * (1.0 - lambda * lambda);
    }

    // Store coefficients (ignore a[0] which is 1.0)
    coefficients_.resize(order_);
    for (int i = 0; i < order_; ++i) {
        coefficients_[i] = a[i + 1];
    }
    return true;
}

double ARModel::forwardPredict() const {
    if (data_.size() < static_cast<size_t>(order_)) {
        std::cerr << "Insufficient data for one-step forward prediction.\n";
        return 0.0;
    }
    double prediction = 0.0;
    for (int i = 0; i < order_; ++i) {
        prediction += coefficients_[i] * data_[data_.size() - 1 - i];
    }
    return prediction;
}

std::vector<double> ARModel::forwardPredictSteps(int k) const {
    std::vector<double> predictions;
    predictions.reserve(k);

    if (data_.size() < static_cast<size_t>(order_)) {
        std::cerr << "Insufficient data for multi-step prediction.\n";
        return predictions;
    }

    // Start with the last 'order_' data points
    std::vector<double> window(data_.end() - order_, data_.end());

    for (int step = 0; step < k; ++step) {
        double pred = 0.0;
        for (int i = 0; i < order_; ++i) {
            pred += coefficients_[i] * window[order_ - 1 - i];
        }
        predictions.push_back(pred);

        // Shift window: remove oldest, add new pred
        window.erase(window.begin());
        window.push_back(pred);
    }
    return predictions;
}

