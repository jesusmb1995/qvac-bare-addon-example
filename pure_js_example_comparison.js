function sigmoid (z) {
  return 1.0 / (1.0 + Math.exp(-z))
}

function fit (X, y, maxIter = 1000, lr = 0.1) {
  const n = X.length
  const d = X[0].length

  const Xn = X.map(row => [...row])

  const mu = new Array(d).fill(0)
  const sd = new Array(d).fill(0)

  for (const row of Xn) {
    for (let j = 0; j < d; j++) mu[j] += row[j]
  }
  for (let j = 0; j < d; j++) mu[j] /= n

  for (const row of Xn) {
    for (let j = 0; j < d; j++) sd[j] += (row[j] - mu[j]) ** 2
  }
  for (let j = 0; j < d; j++) {
    sd[j] = Math.sqrt(sd[j] / n)
    if (sd[j] === 0) sd[j] = 1
  }

  for (const row of Xn) {
    for (let j = 0; j < d; j++) row[j] = (row[j] - mu[j]) / sd[j]
  }

  const w = new Array(d + 1).fill(0)

  for (let iter = 0; iter < maxIter; iter++) {
    const grad = new Array(d + 1).fill(0)
    for (let i = 0; i < n; i++) {
      let z = w[0]
      for (let j = 0; j < d; j++) z += w[j + 1] * Xn[i][j]
      const err = sigmoid(z) - y[i]
      grad[0] += err
      for (let j = 0; j < d; j++) grad[j + 1] += err * Xn[i][j]
    }
    for (let j = 0; j <= d; j++) w[j] -= lr * grad[j] / n
  }

  return [...w, ...mu, ...sd]
}

function predict (params, features) {
  const d = (params.length - 1) / 3
  const bias = params[0]
  const weights = params.slice(1, d + 1)
  const mean = params.slice(d + 1, 2 * d + 1)
  const stdev = params.slice(2 * d + 1, 3 * d + 1)

  let z = bias
  for (let i = 0; i < d; i++) {
    z += weights[i] * ((features[i] - mean[i]) / stdev[i])
  }
  return sigmoid(z)
}

const { X, y } = require('./dataset.json')
console.log(`Training on ${X.length} samples x ${X[0].length} features...\n`)

const params = fit(X, y)

const student = predict(params, [21, 12000])
console.log('student would buy?', student >= 0.5 ? 'yes' : 'no')

const senior_executive = predict(params, [55, 130000])
console.log('senior_executive would buy?', senior_executive >= 0.5 ? 'yes' : 'no')
