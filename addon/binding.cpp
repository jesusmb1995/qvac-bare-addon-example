#include <bare.h>

#include "AddonJs.hpp"

js_value_t*
classifierAddonExports(js_env_t* env, js_value_t* exports) {

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define V(name, fn)                                                            \
  {                                                                            \
    js_value_t* val;                                                           \
    if (js_create_function(env, name, -1, fn, nullptr, &val) != 0) {           \
      return nullptr;                                                          \
    }                                                                          \
    if (js_set_named_property(env, exports, name, val) != 0) {                 \
      return nullptr;                                                          \
    }                                                                          \
  }

  V("createInstance", classifier_addon::createInstance)
  V("runJob", classifier_addon::runJob)
  V("train", classifier_addon::train)
  V("destroyInstance",
    qvac_lib_inference_addon_cpp::JsInterface::destroyInstance)

#undef V
  return exports;
}

BARE_MODULE(classifier_addon, classifierAddonExports)
