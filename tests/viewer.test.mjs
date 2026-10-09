// Checks that the page (docs/cascade.js) gives the same cascades as the Python simulator.
// Run: node tests/viewer.test.mjs
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
const require = createRequire(import.meta.url);
const { NETWORKS } = require("../docs/networks.js");
const { prepare, capitalOf, allCascades } = require("../docs/cascade.js");
const fixture = JSON.parse(readFileSync(new URL("../docs/fixture.json", import.meta.url)));

let bad = 0;
for (const row of fixture) {
  const net = NETWORKS.find((n) => n.id === row.id);
  const prep = prepare(net);
  const out = allCascades(net, prep, capitalOf(net, prep, row.tau));
  const depth = out.rounds.map((r) => Math.max(...r));
  const same = out.sizes.every((s, i) => s === row.sizes[i]) && depth.every((d, i) => d === row.rounds[i]);
  if (!same) { bad += 1; console.log("MISMATCH", row.id, row.tau); }
}
console.log(bad === 0 ? `ok: ${fixture.length} cases, ${fixture.length * 100} cascades match Python` : `${bad} mismatches`);
process.exit(bad === 0 ? 0 : 1);
