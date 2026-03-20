const fs = require('bare-fs')
const path = require('bare-path')

// Deterministic PRNG (mulberry32) so the dataset is reproducible
function mulberry32 (seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

const n = 50000
const rng = mulberry32(42)
const X = []
const y = []

console.log('Generating synthetic purchase-prediction dataset...')
console.log(`  samples:  ${n}`)
console.log('  features: age (18-68), income ($10k-$150k)')
console.log('  label:    1 = would buy, 0 = would not buy')
console.log('  rule:     higher age + higher income → more likely to buy\n')

for (let i = 0; i < n; i++) {
  const age = 18 + rng() * 50           // 18–68
  const income = 10000 + rng() * 140000  // 10k–150k

  X.push([age, income])

  const noise = (rng() - 0.5) * 30000
  y.push(income > 1500 * age + noise ? 1 : 0)
}

const positives = y.filter(v => v === 1).length
const negatives = n - positives
console.log(`Generated ${n} samples:`)
console.log(`  positive (would buy):     ${positives} (${(100 * positives / n).toFixed(1)}%)`)
console.log(`  negative (would not buy): ${negatives} (${(100 * negatives / n).toFixed(1)}%)`)

const out = path.join(__dirname, 'dataset.json')
fs.writeFileSync(out, JSON.stringify({ X, y }))
console.log(`\nDataset written to ${out}`)
