#pragma once

#include <any>
#include <atomic>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <inference-addon-cpp/ModelInterfaces.hpp>
#include <inference-addon-cpp/RuntimeStats.hpp>

namespace classifier_addon {

class LogisticRegression : public qvac_lib_inference_addon_cpp::model::IModel {
  /// Model parameters produced by train():
  /// [bias, w1, ..., wd, mu1, ..., mud, sd1, ..., sdd]
  /// where d is the number of features, mu is per-feature mean,
  /// and sd is per-feature standard deviation used for normalization.
  std::vector<double> params_;
  std::atomic<int64_t> predict_count_{0};

  static double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

  /// Normalizes X in-place, returns per-feature {mean, stddev}.
  static std::pair<std::vector<double>, std::vector<double>>
  normalize(std::vector<std::vector<double>> &X) {
    size_t n = X.size();
    size_t d = X[0].size();
    std::vector<double> mu(d, 0.0), sd(d, 0.0);

    for (const auto &row : X)
      for (size_t j = 0; j < d; j++)
        mu[j] += row[j];
    for (size_t j = 0; j < d; j++)
      mu[j] /= static_cast<double>(n);

    for (const auto &row : X)
      for (size_t j = 0; j < d; j++)
        sd[j] += (row[j] - mu[j]) * (row[j] - mu[j]);
    for (size_t j = 0; j < d; j++) {
      sd[j] = std::sqrt(sd[j] / static_cast<double>(n));
      if (sd[j] == 0.0)
        sd[j] = 1.0;
    }

    for (auto &row : X)
      for (size_t j = 0; j < d; j++)
        row[j] = (row[j] - mu[j]) / sd[j];

    return {mu, sd};
  }

public:
  explicit LogisticRegression(std::vector<double> params)
      : params_(std::move(params)) {
    if (params_.size() < 4 || (params_.size() - 1) % 3 != 0) {
      throw std::invalid_argument(
          "params must have 1+3*d elements "
          "[bias, weights..., means..., stddevs...]");
    }
  }

  [[nodiscard]] std::string getName() const override {
    return "LogisticRegression";
  }

  /// Predict: input is std::vector<double> features (raw, unnormalized).
  /// Normalizes using stored mean/stddev, returns probability as double.
  std::any process(const std::any &input) override {
    const auto &features = std::any_cast<const std::vector<double> &>(input);
    size_t d = (params_.size() - 1) / 3;
    if (features.size() != d) {
      throw std::runtime_error(
          "expected " + std::to_string(d) + " features, got " +
          std::to_string(features.size()));
    }
    std::span<const double> mean(params_.data() + d + 1, d);
    std::span<const double> stdev(params_.data() + 2 * d + 1, d);

    double z = params_[0];
    for (size_t i = 0; i < d; i++) {
      double normalized = (features[i] - mean[i]) / stdev[i];
      z += params_[i + 1] * normalized;
    }

    ++predict_count_;
    return sigmoid(z);
  }

  [[nodiscard]] qvac_lib_inference_addon_cpp::RuntimeStats
  runtimeStats() const override {
    return {{"predict_count", predict_count_}};
  }

  /// Train logistic regression via gradient descent.
  /// Normalizes X internally. Returns [bias, w1, ..., wd, mu1, ..., mud, sd1,
  /// ..., sdd].
  static std::vector<double> train(const std::vector<std::vector<double>> &X,
                                   const std::vector<double> &y,
                                   int maxIter = 1000, double lr = 0.1) {
    if (X.empty() || y.empty()) {
      throw std::invalid_argument("X and y must not be empty");
    }
    if (X.size() != y.size()) {
      throw std::invalid_argument(
          "X and y must have the same number of rows");
    }
    size_t cols = X[0].size();
    for (size_t i = 1; i < X.size(); i++) {
      if (X[i].size() != cols) {
        throw std::invalid_argument(
            "all rows of X must have the same number of columns");
      }
    }

    auto Xn = X;
    auto [mu, sd] = normalize(Xn);

    size_t n = Xn.size();
    size_t d = Xn[0].size();
    std::vector<double> w(d + 1, 0.0);

    for (int iter = 0; iter < maxIter; iter++) {
      std::vector<double> grad(d + 1, 0.0);
      for (size_t i = 0; i < n; i++) {
        double z = w[0];
        for (size_t j = 0; j < d; j++) {
          z += w[j + 1] * Xn[i][j];
        }
        double err = sigmoid(z) - y[i];
        grad[0] += err;
        for (size_t j = 0; j < d; j++) {
          grad[j + 1] += err * Xn[i][j];
        }
      }
      for (size_t j = 0; j <= d; j++) {
        w[j] -= lr * grad[j] / static_cast<double>(n);
      }
    }

    w.insert(w.end(), mu.begin(), mu.end());
    w.insert(w.end(), sd.begin(), sd.end());
    return w;
  }
};

} // namespace classifier_addon
