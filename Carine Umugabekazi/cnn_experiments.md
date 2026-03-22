# CNN Policy Hyperparameter Experiments

Environment: `ALE/Seaquest-v5`  
Policy: `CnnPolicy`  
Total Timesteps per Experiment: `1,000,000`  
Fixed Parameters: `buffer_size=100000`, `learning_starts=10000`, `target_update_interval=1000`
## 1 version training
## Hyperparameter Table

| # | lr | gamma | batch | eps_start | eps_end | eps_frac | mean_reward | best_reward | final_reward | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00001 | 0.90 | 16 | 1.0 | 0.05 | 0.10 | 5.02 | 19.0 | 9.4 | Baseline low — very slow learning, stable |
| 2 | 0.00002 | 0.91 | 16 | 1.0 | 0.05 | 0.10 | 5.87 | 24.0 | 8.2 | Slight lr bump — marginally faster convergence |
| 3 | 0.00003 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 10.99 | 29.0 | 23.6 | Bigger batch — more stable gradient updates |
| 4 | 0.00001 | 0.92 | 16 | 0.8 | 0.05 | 0.10 | 6.97 | 21.0 | 10.6 | Lower eps_start — less early exploration |
| 5 | 0.00005 | 0.91 | 32 | 1.0 | 0.01 | 0.10 | 15.44 | 30.0 | 26.6 | Lower eps_end — more exploitation late game |
| 6 | 0.00001 | 0.93 | 16 | 1.0 | 0.05 | 0.05 | 5.73 | 20.0 | 8.8 | Faster decay — exploitation kicks in early |
| 7 | 0.00002 | 0.92 | 32 | 1.0 | 0.05 | 0.20 | 10.77 | 21.0 | 17.2 | Slower decay — longer exploration phase |
| 8 | 0.00003 | 0.93 | 16 | 0.9 | 0.01 | 0.10 | 11.12 | 22.0 | 19.8 | Mixed low — observe combined effect |
| 9 | 0.00004 | 0.90 | 32 | 1.0 | 0.05 | 0.15 | 11.68 | 33.0 | 16.0 | Moderate decay — balanced explore/exploit |
| 10 | 0.00005 | 0.93 | 32 | 0.8 | 0.01 | 0.10 | 14.57 | 22.0 | 20.6 | Best-of-low — final low-range candidate |

## Key Insights

### What improved performance
- **Larger batch size (32 vs 16):** Experiments 3, 5, 7, 9, 10 all used batch=32 and consistently scored higher than batch=16 experiments. More stable gradient updates help the CNN learn better representations.
- **Lower eps_end:** Experiments 5 and 8 used eps_end=0.01, meaning the agent exploits more aggressively in later training. Both showed strong final rewards (26.6 and 19.8).
- **Higher learning rate:** Experiment 5 (lr=0.00005) combined with batch=32 and low eps_end produced the best final reward of 26.6.

### What harmed performance
- **Very low lr + small batch:** Experiment 1 (lr=0.00001, batch=16) was the worst performer with mean=5.02. The network learned too slowly to make progress in 1M steps.
- **Faster epsilon decay (eps_frac=0.05):** Experiment 6 shifted to exploitation too early before the agent had learned enough, resulting in mean=5.73.
- **Lower eps_start (0.8):** Experiment 4 reduced early exploration which hurt initial learning, giving mean=6.97.

### Best configuration
**Experiment 9** achieved the highest single reward of **33.0**:
- `lr=0.00004`, `gamma=0.90`, `batch=32`
- `eps_start=1.0`, `eps_end=0.05`, `eps_frac=0.15`
- The moderate epsilon decay (0.15) allowed a longer exploration phase before exploitation, helping the agent discover better strategies.

## Notes on Overall Performance
All experiments used 1,000,000 timesteps which is conservative for Seaquest with CNNPolicy. Scores are low range by design and increasing timesteps to 5-10 M may increase learning,the purpose of this experiment set is to establish a performance baseline at conservative hyperparameters. Higher learning rates and larger batchesare expected to yield significantly better scores.

## 2 version training 
## Extended Trainingof 10M Steps (Experiments 5–10)

Based on the baseline results in the hyperparameter table above, experiments 5–10 showed the strongest performance at 1M steps (best rewards ranging from 20–33, with experiment 5 achieving the highest final reward of 26.6 and experiment 9 achieving the highest single reward of 33.0). These six experiments were selected for extended retraining with **10,000,000 timesteps** to test whether more training time would unlock significantly better performance. Experiments 1–4 were excluded as their hyperparameters (very low lr=0.00001–0.00003, small batch=16) showed limited learning potential — mean rewards of 5.02–6.97 — suggesting the configurations themselves were too weak regardless of training length.

**Changes from baseline run:**
- Timesteps: `1,000,000` → `10,000,000`
- Buffer size: `100,000` → `50,000` (reduced to prevent OOM kill)
- `optimize_memory_usage=True` (enabled to reduce RAM usage ~40%)
- GPU used for training (`device=cuda`)
- Eval frequency: every `200,000` steps (was `50,000`)

---

### Extended Results Table

| # | Name | lr | gamma | batch | eps_end | mean_reward | best_reward | final_reward | steps |
|---|---|---|---|---|---|---|---|---|---|
| 1 | exp1_baseline_low_extended | 0.00001 | 0.90 | 16 | 0.05 | 8.40 | 21.0 | 16.60 | 1,000,000 |
| 2 | exp2_lr_bump_extended | 0.00002 | 0.91 | 16 | 0.05 | 9.96 | 21.0 | 16.60 | 1,000,000 |
| 3 | exp3_bigger_batch_extended | 0.00003 | 0.90 | 32 | 0.05 | 12.12 | 30.0 | 19.20 | 1,000,000 |
| 4 | exp4_lower_eps_start_extended | 0.00001 | 0.92 | 16 | 0.05 | 6.53 | 21.0 | 10.60 | 1,000,000 |
| **5** | **exp5_lower_eps_end_extended** | **0.00005** | **0.91** | **32** | **0.01** | **49.09** | **239.0** | **53.80** | **10,000,000** |
| 6 | exp6_faster_decay_extended | 0.00001 | 0.93 | 16 | 0.05 | 20.16 | 80.0 | 24.20 | 10,000,000 |
| 7 | exp7_slower_decay_extended | 0.00002 | 0.92 | 32 | 0.05 | 40.98 | 201.0 | 50.80 | 10,000,000 |
| 8 | exp8_mixed_low_extended | 0.00003 | 0.93 | 16 | 0.01 | 37.35 | 186.0 | 54.80 | 10,000,000 |
| 9 | exp9_moderate_decay_extended | 0.00004 | 0.90 | 32 | 0.05 | 42.36 | 192.0 | 57.00 | 10,000,000 |
| **10** | **exp10_best_of_low_extended** | **0.00005** | **0.93** | **32** | **0.01** | **53.24** | **272.0** | **61.40** | **10,000,000** |

> Experiments 1–4 extended ran only 1M steps — the 10M training was killed by OOM before completing. Their results are included for reference but are not directly comparable to experiments 5–10.

### Comparison: 1M vs 10M Steps

| # | Name | best_reward (1M) | best_reward (10M) | improvement |
|---|---|---|---|---|
| 5 | exp5_lower_eps_end | 30.0 | 239.0 | +209.0 (+697%) |
| 6 | exp6_faster_decay | 20.0 | 80.0 | +60.0 (+300%) |
| 7 | exp7_slower_decay | 21.0 | 201.0 | +180.0 (+857%) |
| 8 | exp8_mixed_low | 22.0 | 186.0 | +164.0 (+745%) |
| 9 | exp9_moderate_decay | 33.0 | 192.0 | +159.0 (+482%) |
| 10 | exp10_best_of_low | 22.0 | 272.0 | +250.0 (+1136%) |

### Key Insights from Extended Training

**10M steps made a dramatic difference.** Every experiment that completed 10M steps saw best reward increase by at least 3x. Experiment 10 went from 22 to 272 — a 1136% improvement. This confirms the 1M baseline was too short for the CNN to learn meaningful Seaquest strategies.

**Best overall configuration — Experiment 10:**
- `lr=0.00005`, `gamma=0.93`, `batch=32`
- `eps_start=0.8`, `eps_end=0.01`, `eps_frac=0.10`
- `best_reward=272.0`, `mean_reward=53.24`, `final_reward=61.40`
- The combination of high learning rate, large batch, high gamma (valuing future rewards), and aggressive final exploitation produced the strongest result.

**Second best — Experiment 5:**
- Same lr and batch as Exp 10 but `gamma=0.91` (slightly less future-focused)
- `best_reward=239.0` — very close to Exp 10, confirming `lr=0.00005 + batch=32 + eps_end=0.01` is the strongest hyperparameter combination in this range.

**What the extended training confirmed:**
- Higher gamma (0.93) outperforms lower gamma (0.90–0.91) at longer training — the agent benefits more from valuing future rewards when it has had enough time to learn them
- `eps_end=0.01` consistently beats `eps_end=0.05` — maximum late exploitation matters more with more training time
- Larger batch (32) remains superior to batch 16 — result holds across both 1M and 10M runs
- The fastest decay (`eps_frac=0.05`, Exp 6) still underperforms — committing to exploitation too early hurts even with 10x more steps

---

### Overall Best Result

**Experiment 10 extended** — `exp10_best_of_low_extended`
- `best_reward = 272.0`
- `mean_reward = 53.24`
- `final_reward = 61.40`
- `total_timesteps = 10,000,000`

To see the best model trained used to play the game, run file watch.py, after you have cloned the repo locally for you.
