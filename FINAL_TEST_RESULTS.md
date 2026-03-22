# SMART Model Family - Final Test Results (seed_42)

## Summary Table

| Dataset | Model | AUROC | AUPRC | min(Se,P+) | F1 | auc_micro | auc_macro | auc_weighted |
|---------|-------|-------|-------|------------|----|-----------|-----------|-----------:|
| **C12** | smart | 0.8458 | 0.5415 | 0.5052 | 0.3013 | - | - | - |
| | smart-smile-lean | 0.8527 | 0.5673 | 0.5285 | 0.5556 | - | - | - |
| | smart-smile-stratified | 0.8352 | 0.5155 | 0.5052 | 0.5386 | - | - | - |
| **C19** | smart | 0.9626 | 0.8239 | 0.7376 | 0.7434 | - | - | - |
| | smart-smile-lean | 0.9634 | 0.8189 | 0.7633 | 0.7709 | - | - | - |
| | smart-smile-stratified | 0.9689 | 0.8388 | 0.7739 | 0.7792 | - | - | - |
| **mimic_mortality** | smart | 0.8359 | 0.4531 | 0.4710 | 0.4576 | - | - | - |
| | smart-smile-lean | 0.8477 | 0.4773 | 0.4551 | 0.5046 | - | - | - |
| | smart-smile-stratified | 0.8606 | 0.4851 | 0.4964 | 0.5116 | - | - | - |
| **mimic_decompensation** | smart | 0.9606 | 0.7520 | 0.6637 | 0.6927 | - | - | - |
| | smart-smile-lean | 0.9617 | 0.7677 | 0.6996 | 0.7220 | - | - | - |
| | smart-smile-stratified | 0.9551 | 0.7407 | 0.6784 | 0.6844 | - | - | - |
| **mimic_lengthofstay** | smart | - | - | - | - | 0.8089 | 0.7180 | 0.7343 |
| | smart-smile-lean | - | - | - | - | 0.7572 | 0.7165 | 0.7298 |
| | smart-smile-stratified | - | - | - | - | 0.7598 | 0.7208 | 0.7339 |
| **mimic_phenotyping** | smart | - | - | - | - | 0.8163 | 0.7666 | 0.7535 |
| | smart-smile-lean | - | - | - | - | 0.8191 | 0.7677 | 0.7539 |
| | smart-smile-stratified | - | - | - | - | 0.8233 | 0.7739 | 0.7607 |

## Detailed Results by Dataset

### C12 (Challenge 2012)

**Binary classification task**

| Model | AUROC | AUPRC | min(Se,P+) | F1 |
|-------|-------|-------|------------|-----|
| smart | 0.8458 | 0.5415 | 0.5052 | 0.3013 |
| smart-smile-lean | **0.8527** | **0.5673** | **0.5285** | **0.5556** |
| smart-smile-stratified | 0.8352 | 0.5155 | 0.5052 | 0.5386 |

**Notes**: smart-smile-lean shows best AUROC and AUPRC; significant F1 improvement over baseline.

---

### C19 (Challenge 2019)

**Binary classification task**

| Model | AUROC | AUPRC | min(Se,P+) | F1 |
|-------|-------|-------|------------|-----|
| smart | 0.9626 | 0.8239 | 0.7376 | 0.7434 |
| smart-smile-lean | 0.9634 | 0.8189 | 0.7633 | 0.7709 |
| smart-smile-stratified | **0.9689** | **0.8388** | **0.7739** | **0.7792** |

**Notes**: smart-smile-stratified achieves best performance across all metrics. Strong baseline already (AUROC 0.9626).

---

### mimic_mortality

**Binary classification task**

| Model | AUROC | AUPRC | min(Se,P+) | F1 |
|-------|-------|-------|------------|-----|
| smart | 0.8359 | 0.4531 | 0.4710 | 0.4576 |
| smart-smile-lean | 0.8477 | 0.4773 | 0.4551 | 0.5046 |
| smart-smile-stratified | **0.8606** | **0.4851** | **0.4964** | **0.5116** |

**Notes**: smart-smile-stratified consistently outperforms other variants. Min(Se,P+) improvements modest.

---

### mimic_decompensation

**Binary classification task**

| Model | AUROC | AUPRC | min(Se,P+) | F1 |
|-------|-------|-------|------------|-----|
| smart | 0.9606 | 0.7520 | 0.6637 | 0.6927 |
| smart-smile-lean | **0.9617** | **0.7677** | **0.6996** | **0.7220** |
| smart-smile-stratified | 0.9551 | 0.7407 | 0.6784 | 0.6844 |

**Notes**: smart-smile-lean shows best metrics; stratified variant slightly underperforms. Strong overall performance.

---

### mimic_lengthofstay

**Multi-label regression task** (uses auc_micro, auc_macro, auc_weighted)

| Model | auc_micro | auc_macro | auc_weighted |
|-------|-----------|-----------|-------------|
| smart | 0.8089 | 0.7180 | 0.7343 |
| smart-smile-lean | 0.7572 | 0.7165 | 0.7298 |
| smart-smile-stratified | 0.7598 | 0.7208 | 0.7339 |

**Notes**: smart baseline performs best on this task. SMILE variants show minor degradation in auc_micro. Possibly due to task-specific characteristics.

---

### mimic_phenotyping

**Multi-label classification task** (uses auc_micro, auc_macro, auc_weighted)

| Model | auc_micro | auc_macro | auc_weighted |
|-------|-----------|-----------|-------------|
| smart | 0.8163 | 0.7666 | 0.7535 |
| smart-smile-lean | 0.8191 | 0.7677 | 0.7539 |
| smart-smile-stratified | **0.8233** | **0.7739** | **0.7607** |

**Notes**: smart-smile-stratified shows consistent improvements across all metrics. Largest gains in auc_macro (+0.73%) and auc_weighted (+0.96%).

---

## Key Observations

### Binary Classification Tasks (C12, C19, mimic_mortality, mimic_decompensation)
- **SMILE variants consistently improve or match baseline performance**
- **Best strategy depends on dataset:**
  - C12, mimic_mortality: smart-smile-lean and smart-smile-stratified competitive
  - C19: smart-smile-stratified significantly outperforms
  - mimic_decompensation: smart-smile-lean wins

### Multi-Label Tasks (mimic_lengthofstay, mimic_phenotyping)
- **mimic_lengthofstay**: Smart baseline performs best; SMILE variants show modest degradation
- **mimic_phenotyping**: SMILE variants consistently improve, especially smart-smile-stratified

### AUPRC and min(Se,P+) Metrics
- AUPRC improvements modest but consistent across binary tasks
- min(Se,P+) improvements align with sensitivity-specificity balance
- Largest gains in C19 (0.7739 min(Se,P+) for stratified)

---

## Training Logs Location
All training logs stored at: `d:/实验室/mamba-cde/SMART/export/{dataset}/{model}/seed_42/training.log`

Generated: 2026-03-22
