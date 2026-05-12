#pragma once

#include <memory>
#include <span>
#include <vector>

#include <inference-addon-cpp/JsInterface.hpp>
#include <inference-addon-cpp/JsUtils.hpp>
#include <inference-addon-cpp/ModelInterfaces.hpp>
#include <inference-addon-cpp/addon/AddonJs.hpp>
#include <inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp>
#include <inference-addon-cpp/queue/OutputCallbackJs.hpp>

#include "LogisticRegression.hpp"

namespace classifier_addon {

struct JsDoubleOutputHandler
    : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<double> {
  JsDoubleOutputHandler()
      : JsBaseOutputHandler<double>([this](const double &out) -> js_value_t * {
          return qvac_lib_inference_addon_cpp::js::Number::create(this->env_,
                                                                  out);
        }) {}
};

/// Creates a LogisticRegression model instance from JS weights (Float64Array)
/// and wires up the output callback pipeline.
///
/// JS signature: createInstance(jsHandle, Float64Array, outputCb)
inline js_value_t *createInstance(js_env_t *env, js_callback_info_t *info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);

  auto weights = js::TypedArray<double>(env, args.get(1, "weights"))
                     .as<std::vector<double>>(env);

  auto model = std::make_unique<LogisticRegression>(std::move(weights));

  // Output Callback
  out_handl::OutputHandlers<out_handl::JsOutputHandlerInterface> outHandlers;
  outHandlers.add(std::make_shared<JsDoubleOutputHandler>());
  auto callback = std::make_unique<OutputCallBackJs>(
      env, args.get(0, "jsHandle"), args.getFunction(2, "outputCallback"),
      std::move(outHandlers));

  auto addon =
      std::make_unique<AddonJs>(env, std::move(callback), std::move(model));

  return JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

/// Runs prediction on features (Float64Array).
/// Returns JS Boolean indicating whether the job was accepted.
///
/// JS signature: runJob(handle, Float64Array)
inline js_value_t *runJob(js_env_t *env, js_callback_info_t *info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  auto &instance = JsInterface::getInstance(env, args.get(0, "instance"));

  auto features = js::TypedArray<double>(env, args.get(1, "features"))
                      .as<std::vector<double>>(env);

  return instance.runJob(std::any(std::move(features)));
}
JSCATCH

/// Trains logistic regression on provided data. Standalone function
/// (no instance needed). Returns weights as Float64Array.
///
/// JS signature: train(X: Float64Array[], y: Float64Array) -> Float64Array
inline js_value_t *train(js_env_t *env, js_callback_info_t *info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);

  auto xArray = js::Array(env, args.get(0, "X"));
  uint32_t n = xArray.size(env);
  std::vector<std::vector<double>> X;
  X.reserve(n);
  for (uint32_t i = 0; i < n; i++) {
    X.push_back(xArray.get<js::TypedArray<double>>(env, i)
                    .as<std::vector<double>>(env));
  }

  auto y = js::TypedArray<double>(env, args.get(1, "y"))
               .as<std::vector<double>>(env);

  auto weights = LogisticRegression::train(X, y);

  std::span<const double> weightsSpan(weights.data(), weights.size());
  return js::TypedArray<double>::create(env, weightsSpan);
}
JSCATCH

} // namespace classifier_addon
