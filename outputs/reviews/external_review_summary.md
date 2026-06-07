# External Review Summary

## Claude

Command used:

```bash
claude --dangerously-skip-permissions --model opus -p "$(cat outputs/reviews/external_reviewer_prompt.md)"
```

Result: completed.

Main points:

- The paper is weak but salvageable if framed as a controlled empirical viability/cost study.
- The previous result was not defensible because it lacked random baseline and greedy evaluation.
- The learning curves showed collapse below random, not simply "failure to solve."
- Must-fix items:
  - add random baseline;
  - add heuristic baseline as reference;
  - add greedy evaluation after training;
  - fix epsilon decay so it reaches final epsilon;
  - add a larger MLP sanity check to prove the DQN pipeline can learn;
  - be explicit that VQC timing is CPU simulation overhead, not QPU runtime.

Actions taken:

- Added greedy evaluation to `src/qrl/training/dqn.py`.
- Added `scripts/run_baselines.py`.
- Added corrected configs `configs/cartpole_mlp_l1_eval100.yaml` and `configs/cartpole_vqc_l1_eval100.yaml`.
- Added sanity config `configs/cartpole_mlp_sanity_solve.yaml`.
- Ran random, heuristic, corrected MLP, corrected VQC, and MLP sanity experiments.
- Updated article results and PDF.

## Antigravity / Gemini

Requested command:

```bash
agy --yolo -p "$(cat outputs/reviews/external_reviewer_prompt.md)"
```

Result: failed because this installed `agy` does not define `--yolo`.

Fallback command:

```bash
agy --dangerously-skip-permissions -p "$(cat outputs/reviews/external_reviewer_prompt.md)" --print-timeout 10m
```

Result: blocked by Google authentication. The CLI printed an OAuth URL and timed out while waiting for interactive login.

## Follow-up Review for Fuzzy/Quantum Scenario

Commands used:

```bash
claude --dangerously-skip-permissions --model opus -p "$(cat outputs/reviews/fuzzy_distillation_followup_prompt.md)"
agy --dangerously-skip-permissions -p "$(cat outputs/reviews/fuzzy_distillation_followup_prompt.md)" --print-timeout 10m
```

Result:

- Antigravity returned successfully.
- Claude returned successfully.

Consensus:

- Do not add a second environment under the deadline.
- Test VQC depth L=3 under fuzzy-teacher distillation.
- Test clean fuzzy membership inputs without action-recommendation leakage.
- Main defensible claim: fuzzy-teacher distillation and descriptive fuzzy memberships can enable compact VQCs to represent useful control policies, but this is not quantum advantage and not reward-only RL.

Actions taken:

- Added `fuzzy_membership_4` representation.
- Ran MLP/VQC distillation with clean membership features.
- Ran VQC L3 raw distillation.
- Built final scenario table in `outputs/tables/fuzzy_final_scenarios.csv`.
