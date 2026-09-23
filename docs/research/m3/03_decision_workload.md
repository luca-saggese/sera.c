# 03 — M3 decision workload

Reference: `docs/M3.md` §56 (semantic workload) and §57 (candidate order
variation). Categories covered: entity binding, unknown entity, unknown
attribute, negation, exception, transitivity, temporal, composition,
multi-hop, insurance coverage, insurance routing, Italian.

## Design

* **One shared state prompt** (59 tokens, English) — reused as the fixed
  prefix by all 32 questions. Defines entities, attributes, an owner chain,
  a temporal fact, and an insurance policy used across every category.
* **32 questions** — each has a short suffix prompt, 2–4 candidate options,
  and the expected index. Several questions keep a deliberately permuted
  option order as a branch/candidate-index confusion canary (§57).
* Token IDs not embedded: produced later by
  `tools/oracle_qwen3_forward.py` (prompt path with the GGUF `AutoTokenizer`,
  or `--token-ids` to bypass tokenization).

## State prompt

> The machine is ALPHA. The cat is gray. The horse is black. The car is red.
> The robot is a machine. The machine BETA is a machine. The cat is grey
> because the light is dim, but the cat is gray. The code is ALPHA. The robot
> is the machine. The machine is the robot. Yesterday the code was BETA;
> today the code is ALPHA. The policy covers repairs of red cars. The policy
> does not cover machines. The policy covers the cat if it is a pet.
> Alice is the owner. Alice owns the cat. The owner will be paid. The mark
> is XQZ.

## Coverage

| Category | Questions |
|---|---|
| entity binding | q01, q05, q09 |
| unknown entity | q21, q22, q23 |
| unknown attribute | q24, q25 |
| negation | q03, q14, q28 |
| exception | q10, q31 |
| transitivity | q04, q06, q27 |
| temporal | q07, q12, q13 |
| composition | q02, q17, q32 |
| multi-hop | q08, q11, q30 |
| insurance coverage | q15, q18 |
| insurance routing | q19, q20, q29 |
| Italian | q16, q26 |

Full JSON (schema of §Worker C): `workload.json` in this directory.
