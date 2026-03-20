const { ClassifierInterface } = require('./addon')
const { X, y } = require('./dataset.json')

async function main () {
  console.log('=== Binary Classifier Addon Example ===\n')

  console.log(`Loading dataset: ${X.length} samples, ${X[0].length} features (age, income)`)

  console.log('Training logistic regression via native C++ addon (gradient descent, 1000 iterations)...')
  const t0 = Date.now()
  const params = ClassifierInterface.train(X, y)
  const elapsed = Date.now() - t0
  console.log(`Training complete in ${elapsed}ms — learned ${params.length} parameters\n`)

  console.log('Creating classifier instance with trained weights...')
  const classifier = new ClassifierInterface(params)

  console.log('Running predictions (non-blocking, executed on C++ background thread):\n')

  const student = await classifier.predict([21, 12000])
  console.log(`  Student        (age: 21, income: $12,000)  → probability: ${student.toFixed(4)} → ${student >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

  const senior = await classifier.predict([55, 130000])
  console.log(`  Senior exec    (age: 55, income: $130,000) → probability: ${senior.toFixed(4)} → ${senior >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

  const midCareer = await classifier.predict([35, 55000])
  console.log(`  Mid-career     (age: 35, income: $55,000)  → probability: ${midCareer.toFixed(4)} → ${midCareer >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

  const retiree = await classifier.predict([65, 80000])
  console.log(`  Retiree        (age: 65, income: $80,000)  → probability: ${retiree.toFixed(4)} → ${retiree >= 0.5 ? 'WOULD buy' : 'would NOT buy'}`)

  console.log('\nDestroying native instance (freeing C++ memory)...')
  classifier.destroy()
  console.log('Done.')
}

main().catch(console.error)
