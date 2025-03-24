#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include "ARModel.h"
#include "SyntheticDataGenerator.h"

// Helper: write a vector of doubles to a file.
void writeVectorToFile(const std::string &filename, const std::vector<double> &data) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening " << filename << " for writing.\n";
        return;
    }
    for (double val : data) {
        outFile << val << "\n";
    }
    outFile.close();
}

// Helper: write a single double to a file.
void writeSingleValueToFile(const std::string &filename, double value) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening " << filename << " for writing.\n";
        return;
    }
    outFile << value << "\n";
    outFile.close();
}

// Structure for error metrics.
struct ErrorMetrics {
    double mse;
    double rmse;
    double mape;
};

ErrorMetrics computeErrors(const std::vector<double>& forecast, const std::vector<double>& actual) {
    ErrorMetrics em {0.0, 0.0, 0.0};
    if (forecast.size() != actual.size() || forecast.empty()) return em;
    double sumSq = 0.0, sumAbsPct = 0.0;
    for (size_t i = 0; i < forecast.size(); ++i) {
        double diff = forecast[i] - actual[i];
        sumSq += diff * diff;
        if (actual[i] != 0.0)
            sumAbsPct += std::fabs(diff / actual[i]) * 100.0;
    }
    em.mse = sumSq / forecast.size();
    em.rmse = std::sqrt(em.mse);
    em.mape = sumAbsPct / forecast.size();
    return em;
}

int main() {
    // -------------------------------
    // 1. Generate a Synthetic Price Series via GBM
    // -------------------------------
    int totalDays = 300;      // Total data length
    int trainDays = 240;      // Use first 260 days for training
    int validDays = totalDays - trainDays; // Forecast horizon (e.g., 40 days)
    double S0 = 100.0;        // Initial stock price
    double mu = 0.01;         // Drift (adjust as needed)
    double sigma = 0.1;      // Volatility (adjust as needed)
    double deltaT = 1.0 / totalDays; // Time increment (using trainDays)

    // Generate full synthetic price series.
    std::vector<double> fullPrices = SyntheticDataGenerator::generateGBM(
        totalDays, S0, mu, sigma, deltaT, 42
    );
    std::cout << "Generated " << fullPrices.size() << " synthetic GBM prices.\n";
    writeVectorToFile("full_prices.txt", fullPrices);

    // Split into training set (days 0 .. trainDays-1) and validation set (days trainDays .. totalDays-1).
    std::vector<double> trainPrices(fullPrices.begin(), fullPrices.begin() + trainDays);
    std::vector<double> validPrices(fullPrices.begin() + trainDays, fullPrices.end());
    writeVectorToFile("train_prices.txt", trainPrices);
    writeVectorToFile("actual_future_prices.txt", validPrices);

    // -------------------------------
    // 2. Transform Training Prices by Differencing
    // -------------------------------
    // Compute first differences: diff[i] = trainPrices[i+1] - trainPrices[i]
    std::vector<double> diffData;
    for (size_t i = 0; i < trainPrices.size() - 1; ++i) {
        diffData.push_back(trainPrices[i+1] - trainPrices[i]);
    }
    // Write differenced data to "log_returns.txt" (as expected by Python scripts)
    writeVectorToFile("log_returns.txt", diffData);

    // -------------------------------
    // 3. AR Model Order Selection over Differenced Data
    // -------------------------------
    int maxOrder = 80;  // Try AR orders from 1 to 10.
    std::vector<double> orders, mses, rmses, mapes;
    double bestMse = std::numeric_limits<double>::infinity();
    int bestOrder = 20;
    std::vector<double> bestForecastedPrices;

    // Last training price (for integration)
    double lastTrainPrice = trainPrices.back();

    for (int order = 20; order <= maxOrder; ++order) {
        ARModel model(diffData, order);
        if (!model.computeCoefficients()) {
            orders.push_back(order);
            mses.push_back(std::numeric_limits<double>::infinity());
            rmses.push_back(std::numeric_limits<double>::infinity());
            mapes.push_back(std::numeric_limits<double>::infinity());
            continue;
        }
        // Forecast differenced values over the validation horizon.
        std::vector<double> forecastedDiff = model.forwardPredictSteps(validDays);
        // Reconstruct forecasted prices by integrating the forecasted differences.
        std::vector<double> forecastedPrices(validDays);
        double currentPrice = lastTrainPrice;
        for (int i = 0; i < validDays; ++i) {
            currentPrice += forecastedDiff[i];
            forecastedPrices[i] = currentPrice;
        }
        // Compute error metrics (comparing forecastedPrices with validPrices).
        ErrorMetrics em = computeErrors(forecastedPrices, validPrices);

        orders.push_back(order);
        mses.push_back(em.mse);
        rmses.push_back(em.rmse);
        mapes.push_back(em.mape);

        // Keep forecast from the best model based on MSE.
        if (em.mse < bestMse) {
            bestMse = em.mse;
            bestOrder = order;
            bestForecastedPrices = forecastedPrices;
        }
    }

    // Save AR order selection metrics for plotting.
    writeVectorToFile("ar_orders.txt", orders);
    writeVectorToFile("ar_mses.txt", mses);
    writeVectorToFile("ar_rmses.txt", rmses);
    writeVectorToFile("ar_mapes.txt", mapes);

    std::cout << "Best AR order based on MSE: " << bestOrder << "\n";
    std::cout << "MSE at best order: " << bestMse << "\n";

    // -------------------------------
    // 4. Output Forecasts using the Best AR Order
    // -------------------------------
    // For best AR order, also output forecasted differences and integrated prices.
    // (We re-run the forecast for clarity.)
    ARModel bestModel(diffData, bestOrder);
    if (!bestModel.computeCoefficients()) {
        std::cerr << "Error computing best AR model coefficients.\n";
        return -1;
    }
    std::vector<double> bestForecastedDiff = bestModel.forwardPredictSteps(validDays);
    writeVectorToFile("forecasted_diff.txt", bestForecastedDiff);

    // Reconstruct level forecasts.
    std::vector<double> forecastedPrices(bestForecastedDiff.size());
    double currentPrice = lastTrainPrice;
    for (size_t i = 0; i < bestForecastedDiff.size(); ++i) {
        currentPrice += bestForecastedDiff[i];
        forecastedPrices[i] = currentPrice;
    }
    writeVectorToFile("forecasted_prices.txt", forecastedPrices);

    // One-step forecast (optional).
    double oneStepDiff = bestModel.forwardPredict();
    double oneStepPrice = lastTrainPrice + oneStepDiff;
    writeSingleValueToFile("one_step_diff.txt", oneStepDiff);
    writeSingleValueToFile("one_step_price.txt", oneStepPrice);

    // -------------------------------
    // 5. Export Time Indices for Plotting
    // -------------------------------
    std::vector<double> trainTime;
    for (int i = 0; i < trainDays; ++i) {
        trainTime.push_back(i);
    }
    writeVectorToFile("train_time_indices.txt", trainTime);

    std::vector<double> forecastTime;
    for (int i = 0; i < validDays; ++i) {
        forecastTime.push_back(trainDays + i);
    }
    writeVectorToFile("forecast_time_indices.txt", forecastTime);

    // -------------------------------
    // 6. Compute and Save Validation Error Metrics for Best Model
    // -------------------------------
    ErrorMetrics em_best = computeErrors(forecastedPrices, validPrices);
    std::cout << "Validation Error Metrics for Best Model (AR(" << bestOrder << ")):\n";
    std::cout << "MSE: " << em_best.mse << "\nRMSE: " << em_best.rmse << "\nMAPE: " << em_best.mape << "%\n";
    writeSingleValueToFile("validation_mse.txt", em_best.mse);
    writeSingleValueToFile("validation_rmse.txt", em_best.rmse);
    writeSingleValueToFile("validation_mape.txt", em_best.mape);

    // -------------------------------
    // 7. Print Summary
    // -------------------------------
    std::cout << "Training ends at day " << (trainDays - 1)
              << " with price " << lastTrainPrice << "\n";
    std::cout << "Forecast horizon: " << validDays << " days.\n";
    std::cout << "One-step Price Forecast (via differencing): " << oneStepPrice << "\n";
    std::cout << "Multi-step forecasted prices (using best AR order " << bestOrder << "):\n";
    for (double p : forecastedPrices) {
        std::cout << p << " ";
    }
    std::cout << "\nData saved to text files for plotting.\n";

    return 0;
}





