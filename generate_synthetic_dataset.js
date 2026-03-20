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

for (let i = 0; i < n; i++) {
  const age = 18 + rng() * 50           // 18–68
  const income = 10000 + rng() * 140000  // 10k–150k

  X.push([age, income])

  // Older people with higher income are more likely to buy.
  // Label is 1 when: income > 1500 * age + noise
  // e.g. a 30-year-old needs ~45k, a 50-year-old needs ~75k,
  // but noise makes ~10 % of labels flip near the boundary.
  const noise = (rng() - 0.5) * 30000
  y.push(income > 1500 * age + noise ? 1 : 0)
}

const out = path.join(__dirname, 'dataset.json')
fs.writeFileSync(out, JSON.stringify({ X, y }))

const positives = y.filter(v => v === 1).length
console.log(`Wrote ${out}`)
console.log(`  ${n} samples, 2 features (age, income)`)
console.log(`  ${positives} positive (${(100 * positives / n).toFixed(1)}%), ${n - positives} negative`)
