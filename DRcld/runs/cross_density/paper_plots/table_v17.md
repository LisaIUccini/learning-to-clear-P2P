### DR / GNN hyperparameters per density

| density | lambda (DR relax.) | gamma (DR scale) | rho (GNN decay) | unroll | base_seed |
|---|---|---|---|---|---|
| 0.2 | 1.0 | 4.0 | 0.9765 | 426 | 3 |
| 0.4 | 1.0 | 4.0 | 0.9686 | 318 | 3 |
| 0.6 | 1.0 | 4.0 | 0.9688 | 321 | 3 |
| 0.8 | 1.0 | 4.0 | 0.9854 | 685 | 0 |
| 1.0 | 1.0 | 4.0 | 0.9741 | 386 | 0 |

### Communications per iteration (structural, not learned)

| density | edges | network-wide msgs/iter (=2 x edges) | per-agent msgs/iter: mean | per-agent msgs/iter: min | per-agent msgs/iter: max |
|---|---|---|---|---|---|
| 0.2 | 16 | 32 | 3.2 | 1 | 5 |
| 0.4 | 23 | 46 | 4.6 | 3 | 7 |
| 0.6 | 31 | 62 | 6.2 | 4 | 8 |
| 0.8 | 38 | 76 | 7.6 | 5 | 9 |
| 1.0 | 45 | 90 | 9.0 | 9 | 9 |

*Per-agent = each prosumer's own degree (number of direct neighbors); it sends one message to, and receives one from, each neighbor every DR iteration -- this is the actual communication burden on a single node in the decentralized algorithm, as opposed to the network-wide total.*

### Iterations to convergence: plain DR vs. DR + learned perturbation

| density | condition | DR mean | DR std dev | learned mean | learned std dev |
|---|---|---|---|---|---|
| 0.2 | in-distribution (train) | 840 | ±219 | 217 | ±26 |
| 0.2 | OOD $\pm$20% | 872 | ±223 | 240 | ±45 |
| 0.2 | OOD $\pm$30% | 924 | ±254 | 306 | ±135 |
| 0.2 | OOD $\pm$40% | 960 | ±269 | 353 | ±138 |
| 0.2 | OOD $\pm$50% | 984 | ±270 | 379 | ±143 |
| 0.4 | in-distribution (train) | 640 | ±115 | 357 | ±25 |
| 0.4 | OOD $\pm$20% | 642 | ±115 | 373 | ±29 |
| 0.4 | OOD $\pm$30% | 636 | ±109 | 415 | ±95 |
| 0.4 | OOD $\pm$40% | 632 | ±120 | 547 | ±276 |
| 0.4 | OOD $\pm$50% | 647 | ±95 | 617 | ±325 |
| 0.6 | in-distribution (train) | 656 | ±82 | 454 | ±59 |
| 0.6 | OOD $\pm$20% | 617 | ±86 | 450 | ±51 |
| 0.6 | OOD $\pm$30% | 661 | ±188 | 582 | ±290 |
| 0.6 | OOD $\pm$40% | 696 | ±194 | 644 | ±358 |
| 0.6 | OOD $\pm$50% | 734 | ±299 | 642 | ±275 |
| 0.8 | in-distribution (train) | 1379 | ±270 | 428 | ±46 |
| 0.8 | OOD $\pm$20% | 1362 | ±295 | 467 | ±99 |
| 0.8 | OOD $\pm$30% | 1394 | ±316 | 569 | ±170 |
| 0.8 | OOD $\pm$40% | 1396 | ±297 | 761 | ±401 |
| 0.8 | OOD $\pm$50% | 1413 | ±338 | 917 | ±537 |
| 1.0 | in-distribution (train) | 762 | ±160 | 450 | ±30 |
| 1.0 | OOD $\pm$20% | 787 | ±160 | 490 | ±119 |
| 1.0 | OOD $\pm$30% | 808 | ±177 | 543 | ±107 |
| 1.0 | OOD $\pm$40% | 842 | ±186 | 679 | ±265 |
| 1.0 | OOD $\pm$50% | 846 | ±179 | 748 | ±310 |
