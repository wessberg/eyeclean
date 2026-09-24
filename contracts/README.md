# Validation contracts

`acceptance.yaml` defines immutable engineering and provisional quality gates. Hash it before training and include that hash in each run report. `run_report.schema.json` defines the structure of the implementing agent's actual run report; it is not a sample successful result.

JSON-schema validation checks structure. `eyeclean verify-run` must additionally verify every required engineering check by name, paths and hashes on disk, current frozen thresholds, actual metric values, profile restrictions and evidence files. A correctly shaped JSON object is not proof that a test ran. A smoke run cannot pass the delivered prototype-quality gate. Missing evidence or an unavailable quality metric cannot count as a pass.

Preservation comparisons use bit representations of FP32 reference values outside the final full-resolution support. Check semantic edit safety separately against independent ground-truth eye/reflection masks on controlled fixtures. A tautological “our pixels stayed inside our own mask” test does not prove the mask was correct.

Metric values must be `null` when unavailable, with a reason. Do not use zero as a placeholder for unmeasured error, latency, memory or changed-pixel counts. Real footage without ground truth must not receive invented full-reference quality scores. Do not write a successful run report for this unimplemented specification package.
