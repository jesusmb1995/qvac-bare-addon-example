const { ClassifierInterface } = require('./addon')
const { X, y } = require('./dataset.json')

async function main () {
  console.log(`Training on ${X.length} samples x ${X[0].length} features...\n`)

  const params = ClassifierInterface.fit(X, y)
  const classifier = new ClassifierInterface(params)

  const student = await classifier.predict([21, 12000])
  console.log('student would buy?', student >= 0.5 ? 'yes' : 'no')

  const senior_executive = await classifier.predict([55, 130000])
  console.log('senior_executive would buy?', senior_executive >= 0.5 ? 'yes' : 'no')

  classifier.destroy()
}

main()
