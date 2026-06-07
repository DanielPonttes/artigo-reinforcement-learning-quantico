We are continuing an ENIAC 2026 undergraduate paper project.

Current state:

1. Reward-trained DQN with very small models on CartPole:
   - Raw DQN-MLP, 23 params, 3 seeds: greedy eval 9.42 +/- 0.20.
   - Raw DQN-VQC, 22 params, 3 seeds: greedy eval 9.45 +/- 0.18.
   - Random: 20.65 +/- 2.01.
   - Heuristic: 500.00.
   - Conclusion: small reward-trained models fail below random.

2. Larger MLP sanity:
   - 450 params, 1 seed: greedy eval 98.65.
   - Conclusion: DQN code can learn with more capacity.

3. Fuzzy-teacher distillation:
   - Teacher: fixed CartPole fuzzy/heuristic controller.
   - MLP distilled, 23 params, 3 seeds: eval 500.00 +/- 0.00.
   - VQC distilled, 22 params, 3 seeds: eval 96.53 +/- 38.13.
   - Random: 20.65 +/- 2.01.
   - Conclusion: tiny VQC can partially represent a useful fuzzy policy under supervised distillation, but not as well as tiny MLP.

Question:

The project owner wants to continue testing other scenarios and optimize. We must not invent quantum advantage.

Please review and recommend:

1. Should we test VQC with more layers (L=3, ~46 params) under fuzzy-teacher distillation?
2. Should we test fuzzy membership inputs (4 descriptive memberships, no recommended action leakage) for distillation?
3. Should we add another environment now, or is that risky under deadline?
4. What is the strongest defensible claim if VQC L3 improves, but MLP still dominates?
5. What exact table/figure should the paper use?
6. What experiments are highest ROI in the next 1-2 hours?

Constraints:

- No quantum advantage claim unless VQC clearly beats fair classical baseline, which is unlikely.
- Keep CartPole unless another environment is very cheap.
- Distillation must be clearly labeled as supervised fuzzy-teacher imitation, not RL from reward.
- Prefer experiments that improve the paper's defensibility.

Return concise actionable recommendations.
