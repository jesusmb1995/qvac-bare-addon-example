/** JS interface for the binary classifier bare addon.
 *
 * Wraps the native C++ binding and provides a Promise-based predict() API.
 * Training is exposed as a standalone train() function that returns weights
 * suitable for constructing a new ClassifierInterface instance.
 */
const { exclusiveRunQueue } = require('@qvac/infer-base')

const binding = require.addon()
const DEBUG = false

class ClassifierInterface {
  constructor (params) {
    this._result = null
    this._resolve = null
    this._reject = null
    this._run = exclusiveRunQueue()

    this._handle = binding.createInstance(
      this,
      new Float64Array(params),
      (addon, event, data, error) => this._onOutput(addon, event, data, error)
    )
  }

  // Callback invoked by the native side for each output event.
  // `event` is the C++ mangled type name of the output — the framework uses it
  // to dispatch to the correct output handler. For our classifier, 'd' maps to
  // the double (prediction probability) and the stats event carries runtime metrics.
  _onOutput (addon, event, data, error) {
    if (DEBUG) console.debug('_onOutput', { event, data, error })
    if (typeof error === 'string') {
      return this._settle(null, new Error(error))
    }

    if (typeof data === 'number') {
      this._result = data
    } else if (typeof data === 'object' && data !== null) {
      this._settle(this._result)
    }
  }

  _settle (value, error) {
    const { _resolve: resolve, _reject: reject } = this
    this._resolve = null
    this._reject = null
    if (error) { if (reject) reject(error) } else { if (resolve) resolve(value) }
  }

  predict (features) {
    return this._run(() => {
      return new Promise((resolve, reject) => {
        this._resolve = resolve
        this._reject = reject
        const accepted = binding.runJob(
          this._handle,
          new Float64Array(features)
        )
        if (!accepted) {
          this._settle(null, new Error('Job not accepted: another job is in progress'))
        }
      })
    })
  }

  destroy () {
    this._settle(null, new Error('Instance destroyed'))
    if (this._handle) {
      binding.destroyInstance(this._handle)
      this._handle = null
    }
  }

  static train (X, y) {
    const xArrays = X.map(row => new Float64Array(row))
    const yArray = new Float64Array(y)
    return Array.from(binding.train(xArrays, yArray))
  }
}

module.exports = { ClassifierInterface }
