We are continuing an undergraduate ENIAC 2026 paper project.

Current status:

- Implemented DQN-MLP and DQN-VQC on CartPole-v1.
- Parameter-matched tiny models do not beat random:
  - DQN-MLP small, 23 params: greedy eval 9.42 +/- 0.20.
  - DQN-VQC small, 22 params: greedy eval 9.45 +/- 0.18.
  - Random: 20.65 +/- 2.01.
  - Heuristic: 500.00 +/- 0.00.
- Larger MLP sanity check, 450 params, 1 seed: greedy eval 98.65.
- Current framing is honest negative result / cost study.

New objective from the project owner:

"Bring a scenario where there is gain in a quantum/fuzzy environment."

Need advice:

1. What is a defensible way to introduce fuzzy logic or fuzzy features without p-hacking?
2. Can a fuzzy state representation improve a tiny DQN under parameter constraints?
3. What should be compared?
   - raw-state DQN-MLP vs fuzzy-feature DQN-MLP?
   - raw-state DQN-VQC vs fuzzy-feature DQN-VQC?
   - fuzzy rule-based controller as upper/reference baseline?
   - VQC with fuzzy-compressed inputs?
4. What quick experiments can plausibly show a real gain within hours?
5. What would be scientifically dishonest or likely rejected?
6. How should the paper framing change if fuzzy features help, but the quantum circuit itself still does not outperform classical?

Constraints:

- Do not invent or force positive results.
- CartPole is implemented.
- We can add fuzzy preprocessing/features and run quick experiments.
- We can add a simple custom fuzzy CartPole feature map:
  - cart position: left/center/right
  - cart velocity: negative/zero/positive
  - pole angle: left/upright/right
  - pole angular velocity: negative/zero/positive
- We can also add compact heuristic-inspired fuzzy features:
  - pole angle normalized
  - pole angular velocity normalized
  - fuzzy stability score
  - fuzzy recommended action score
- Need results that are defensible for undergraduate track.

Return:

- Verdict on whether "quantum/fuzzy gain" is feasible.
- Best experiment design.
- Minimal implementation plan.
- Metrics and baselines.
- Rejection risks.
- Recommended paper wording if fuzzy helps but VQC does not.
