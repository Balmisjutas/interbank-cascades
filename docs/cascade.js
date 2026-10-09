// The cascade rule of the page. Same rule as src/cascades/sim.py:
// a bank fails when LGD * (what it lent to failed banks) >= its capital.
// net.links holds [lender, borrower, amount].
// The value of LGD does not matter here: it cancels between capitalOf and cascade.
// Capital, loss rate and common shock act through tau only (see index.html, tauNow).
const LGD = 0.6;

function prepare(net) {
  const assets = new Array(net.n).fill(0);
  const lendersOf = Array.from({ length: net.n }, () => []);
  for (const [i, j, w] of net.links) {
    assets[i] += w;
    lendersOf[j].push([i, w]);
  }
  return { assets, lendersOf };
}

function capitalOf(net, prep, tau) {
  return prep.assets.map((a, i) => (a > 0 ? tau * net.strength[i] * LGD * a : 1));
}

// Returns round[i]: 0 for the trigger, k for a bank that fails in round k, -1 for a survivor.
function cascade(net, prep, capital, trigger) {
  const round = new Array(net.n).fill(-1);
  const lost = new Array(net.n).fill(0); // amount lent to failed banks so far
  round[trigger] = 0;
  let wave = [trigger];
  let k = 0;
  while (wave.length) {
    k += 1;
    for (const j of wave) for (const [i, w] of prep.lendersOf[j]) lost[i] += w;
    const next = [];
    for (let i = 0; i < net.n; i++) {
      if (round[i] === -1 && LGD * lost[i] >= capital[i]) {
        round[i] = k;
        next.push(i);
      }
    }
    wave = next;
  }
  return round;
}

function allCascades(net, prep, capital) {
  const rounds = [];
  for (let t = 0; t < net.n; t++) rounds.push(cascade(net, prep, capital, t));
  const sizes = rounds.map((r) => r.filter((x) => x > 0).length);
  const fragility = new Array(net.n).fill(0);
  rounds.forEach((r) => r.forEach((x, i) => { if (x > 0) fragility[i] += 1; }));
  return { rounds, sizes, fragility: fragility.map((c) => c / (net.n - 1)) };
}

if (typeof module !== "undefined") module.exports = { LGD, prepare, capitalOf, cascade, allCascades };
