### DR / GNN hyperparameters per density

| density | lambda (DR relax.) | gamma (DR scale) | rho (GNN decay) | unroll | base_seed |
|---|---|---|---|---|---|
| 0.2 | 1.0 | 4.0 | 0.9765 | 426 | 3 |
| 0.4 | 1.0 | 4.0 | 0.9686 | 318 | 3 |
| 0.5 | 1.0 | 6.0 | 0.9747 | 395 | 1 |
| 0.6 | 1.0 | 4.0 | 0.9688 | 321 | 3 |
| 0.8 | 1.0 | 4.0 | 0.9854 | 685 | 0 |
| 1.0 | 1.0 | 4.0 | 0.9741 | 386 | 0 |

### Communications per iteration (structural, not learned)

| density | edges | network-wide msgs/iter (=2 x edges) | per-agent msgs/iter: mean | per-agent msgs/iter: min | per-agent msgs/iter: max |
|---|---|---|---|---|---|
| 0.2 | 16 | 32 | 3.2 | 1 | 5 |
| 0.4 | 23 | 46 | 4.6 | 3 | 7 |
| 0.5 | 27 | 54 | 5.4 | 3 | 7 |
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
| 0.2 | remove 1 node | 813 | ±468 | 247 | ±117 |
| 0.2 | remove 2 nodes | 533 | ±457 | 228 | ±82 |
| 0.2 | $\pm$1 node | 911 | ±412 | 333 | ±207 |
| 0.2 | rewire (same degree) | 501 | ±285 | 398 | ±230 |
| 0.4 | in-distribution (train) | 640 | ±115 | 357 | ±25 |
| 0.4 | OOD $\pm$20% | 642 | ±115 | 373 | ±29 |
| 0.4 | OOD $\pm$30% | 636 | ±109 | 415 | ±95 |
| 0.4 | OOD $\pm$40% | 632 | ±120 | 547 | ±276 |
| 0.4 | OOD $\pm$50% | 647 | ±95 | 617 | ±325 |
| 0.4 | remove 1 node | 482 | ±221 | 346 | ±85 |
| 0.4 | remove 2 nodes | 364 | ±157 | 270 | ±36 |
| 0.4 | $\pm$1 node | 571 | ±227 | 470 | ±231 |
| 0.4 | rewire (same degree) | 707 | ±423 | 758 | ±662 |
| 0.5 | in-distribution (train) | 800 | ±91 | 574 | ±66 |
| 0.5 | OOD $\pm$20% | 804 | ±95 | 577 | ±83 |
| 0.5 | OOD $\pm$30% | 819 | ±114 | 626 | ±110 |
| 0.5 | OOD $\pm$40% | 885 | ±205 | 733 | ±239 |
| 0.5 | OOD $\pm$50% | 825 | ±126 | 659 | ±150 |
| 0.5 | remove 1 node | 670 | ±272 | 616 | ±254 |
| 0.5 | remove 2 nodes | 595 | ±275 | 711 | ±436 |
| 0.5 | $\pm$1 node | 737 | ±187 | 768 | ±665 |
| 0.5 | rewire (same degree) | 762 | ±248 | 691 | ±229 |
| 0.6 | in-distribution (train) | 656 | ±82 | 454 | ±59 |
| 0.6 | OOD $\pm$20% | 617 | ±86 | 450 | ±51 |
| 0.6 | OOD $\pm$30% | 661 | ±188 | 582 | ±290 |
| 0.6 | OOD $\pm$40% | 696 | ±194 | 644 | ±358 |
| 0.6 | OOD $\pm$50% | 734 | ±299 | 642 | ±275 |
| 0.6 | remove 1 node | 635 | ±541 | 656 | ±755 |
| 0.6 | remove 2 nodes | 371 | ±72 | 348 | ±65 |
| 0.6 | $\pm$1 node | 777 | ±711 | 962 | ±1061 |
| 0.6 | rewire (same degree) | 660 | ±427 | 761 | ±464 |
| 0.8 | in-distribution (train) | 1379 | ±270 | 428 | ±46 |
| 0.8 | OOD $\pm$20% | 1362 | ±295 | 467 | ±99 |
| 0.8 | OOD $\pm$30% | 1394 | ±316 | 569 | ±170 |
| 0.8 | OOD $\pm$40% | 1396 | ±297 | 761 | ±401 |
| 0.8 | OOD $\pm$50% | 1413 | ±338 | 917 | ±537 |
| 0.8 | remove 1 node | 712 | ±173 | 528 | ±127 |
| 0.8 | remove 2 nodes | 607 | ±223 | 599 | ±198 |
| 0.8 | $\pm$1 node | 748 | ±247 | 779 | ±655 |
| 0.8 | rewire (same degree) | 947 | ±896 | 840 | ±529 |
| 1.0 | in-distribution (train) | 762 | ±160 | 450 | ±30 |
| 1.0 | OOD $\pm$20% | 787 | ±160 | 490 | ±119 |
| 1.0 | OOD $\pm$30% | 808 | ±177 | 543 | ±107 |
| 1.0 | OOD $\pm$40% | 842 | ±186 | 679 | ±265 |
| 1.0 | OOD $\pm$50% | 846 | ±179 | 748 | ±310 |
| 1.0 | remove 1 node | 941 | ±401 | 979 | ±417 |
| 1.0 | remove 2 nodes | 796 | ±394 | 1160 | ±517 |
| 1.0 | $\pm$1 node | 1274 | ±837 | 1185 | ±574 |
