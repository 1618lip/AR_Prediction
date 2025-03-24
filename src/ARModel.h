#ifndef ARMODEL_H
#define ARMODEL_H

#include <vector>

class ARModel {
public:
    // Constructor: data should be a (stationary) series, e.g., log-returns.
    ARModel(const std::vector<double>& data, int order);

    // Compute AR coefficients using Levinson-Durbin.
    bool computeCoefficients();

    // One-step forward prediction based on the last 'order_' data points.
    double forwardPredict() const;

    // Multi-step forward prediction (recursive).
    std::vector<double> forwardPredictSteps(int k) const;

private:
    std::vector<double> data_;
    int order_;
    std::vector<double> coefficients_;
    std::vector<double> autocorrelation_;

    // Compute autocorrelation up to 'order_'.
    void computeAutocorrelation();

public:
    // Accessors
    const std::vector<double>& getCoefficients() const { return coefficients_; }
};

#endif


