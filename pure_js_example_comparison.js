function sigmoid (z) {
  return 1.0 / (1.0 + Math.exp(-z))
}

function train (X, y, maxIter = 1000, lr = 0.1) {
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

console.log('=== Pure JS Classifier (comparison baseline) ===\n')

console.log(`Loading dataset: ${X.length} samples, ${X[0].length} features (age, income)`)

console.log('Training logistic regression in pure JavaScript (gradient descent, 1000 iterations)...')
const t0 = Date.now()
const params = train(X, y)
const elapsed = Date.now() - t0
console.log(`Training complete in ${elapsed}ms — learned ${params.length} parameters\n`)

console.log('Running predictions:\n')

const student = predict(params, [21, 12000])
console.log(`  Student        (age: 21, income: $12,000)  → probability: ${student.toFixed(4)} → ${student >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

const senior = predict(params, [55, 130000])
console.log(`  Senior exec    (age: 55, income: $130,000) → probability: ${senior.toFixed(4)} → ${senior >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

const midCareer = predict(params, [35, 55000])
console.log(`  Mid-career     (age: 35, income: $55,000)  → probability: ${midCareer.toFixed(4)} → ${midCareer >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

const retiree = predict(params, [65, 80000])
console.log(`  Retiree        (age: 65, income: $80,000)  → probability: ${retiree.toFixed(4)} → ${retiree >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

console.log('\nDone.')
