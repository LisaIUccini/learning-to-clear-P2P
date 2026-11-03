# Rivalutazione DR+GNN con criterio di arresto non perturbato

Data: 2026-10-28. Nessun riaddestramento: checkpoint ufficiali `experiments/runs/official/official_N10_d*_replicate*/phi_w.pt` caricati in sola lettura, con la configurazione salvata nel checkpoint (`blob["cfg"]`, che include `scale`/`scale_slow`).

## 1. Come è calcolato il residuo di arresto

- **DR e DR+GNN, criterio ufficiale:** `algorithm.run_dr` ferma il ciclo quando `r < cfg.tol`, con `r = net.aux_residual()` ([algorithm.py:107](../../../algorithm.py), test a riga 117).
- `Network.aux_residual()` ([network.py:525](../../../network.py)) restituisce `sqrt(Σ_n residual_sq_n)`: norma **euclidea, non normalizzata**, di x^{k+1}−x^k su tutte le componenti (D_n, G_n e q_nm per ogni arco orientato).
- `residual_sq_n` è calcolato in `Prosumer.dr_step` ([network.py:265](../../../network.py)) **dopo** l'aggiunta di v^k. Il passo DR è alle righe 248–250: D,G: λ(z−x^k); q: λ(z_q − q̄), cioè λ(z^k−x̃^k).
  - Per DR puro, quindi, il residuo è esattamente ‖λ(z^k−x̃^k)‖.
  - Per DR+GNN è ‖λ(z^k−x̃^k)+v^k‖.
- Parametri: tol = `config.DR_TOLERANCE` = 1e-5 (config.py:61), cap = `config.DR_MAX_ITER` = 4000 (config.py:59), λ = `config.DR_LAMBDA` = 1.0 (config.py:58). γ viene letto dall'intestazione del `train.log` di ogni run ufficiale (4, 4, 6, 4, 4, 6 per δ = 0.2…1.0).

**Modifica al codice (default invariato):**
- `DRConfig.stop_on ∈ {"perturbed" (default), "unperturbed"}` in [algorithm.py:36](../../../algorithm.py).
- `Prosumer.dr_residual_sq`, calcolato prima di v^k ([network.py:253](../../../network.py)), e `Network.dr_residual()` ([network.py:529](../../../network.py)).
- `DRResult.dr_residual`.

Il criterio non perturbato usa quindi la stessa norma, la stessa tolleranza e lo stesso cap.

## 2. Riproduzione dei risultati ufficiali

Per **tutte e 6 le densità e tutte le 30 istanze**:
- DR e DR+GNN con criterio perturbato riproducono gli `instance_plots/*.json` ufficiali **esattamente**: stesso k, traiettoria del residuo identica bit per bit, gap e violazione uguali (rtol 1e-6). In totale 180/180 per DR e 180/180 per DR+GNN; colonna `reproduces_official` del CSV.
- x0 è identico per i tre metodi su ogni istanza (stesso schema seed di `demo.run_pair`).
- Le colonne gap/violazione di DR e DR+GNN (P) nella nuova tabella coincidono con la Tabella 5 (`tabF_stopping_quality.tex`).

## 3. Risultati per densità (mediane su 30 istanze)

P = criterio perturbato (ufficiale), U = criterio non perturbato. Δ = (k_DR−k)/k_DR, con [IQR].

| δ | k DR | k P | k U | Δ P | Δ U | gap DR | gap P | gap U | viol DR | viol P | viol U |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 1419 | 238 | 302 | 83.4% [80.7, 87.3] | 78.4% [75.6, 82.5] | 2.9e-7 | 6.2e-6 | 1.6e-6 | 2.0e-6 | 2.1e-5 | 4.5e-6 |
| 0.4 | 900 | 350 | 387 | 61.4% [51.5, 66.8] | 50.3% [43.1, 56.1] | 3.0e-7 | 6.1e-6 | 3.3e-6 | 2.8e-6 | 1.5e-6 | 7.9e-7 |
| 0.5 | 1147 | 596 | 596 | 46.4% [43.7, 53.0] | 46.3% [43.7, 52.4] | 2.6e-7 | 2.5e-6 | 2.5e-6 | 4.2e-6 | 2.7e-6 | 2.6e-6 |
| 0.6 | 838 | 428 | 428 | 46.9% [39.2, 52.9] | 46.9% [39.2, 52.9] | 1.8e-7 | 3.1e-7 | 3.1e-7 | 4.1e-6 | 3.8e-6 | 3.8e-6 |
| **0.8** | 2467 | 610 | **1015** | **75.4%** [66.4, 77.3] | **58.8%** [44.9, 62.9] | 1.2e-7 | 2.8e-5 | **1.9e-7** | 3.9e-6 | 3.0e-4 | **7.0e-6** |
| 1.0 | 2274* | 1253 | 1578 | 45.8% [−4.6, 67.6] | 29.0% [−31.6, 59.3] | 1.8e-7 | 3.9e-6 | 4.0e-7 | 5.4e-6 | 6.8e-6 | 4.6e-6 |

\* δ=1.0: 6/30 istanze DR arrivano al cap di 4000, quindi i Δ di quelle istanze sono limiti inferiori. Nessuna esecuzione DR+GNN (P o U) raggiunge il cap a nessuna densità.

**Lettura dei risultati:**
- **δ = 0.5 e 0.6** (checkpoint senza canale lento, `scale_slow=0`): i due criteri danno lo stesso k (30/30 a δ=0.6, 22/30 a δ=0.5). Qui v^k è già trascurabile quando l'algoritmo si ferma.
- **δ = 0.4, 0.8 e 1.0** (canale lento attivo, `scale_slow=0.4`) e **δ = 0.2**: il criterio perturbato si ferma prima che la parte DR del passo sia piccola. Il guadagno si riduce, soprattutto a δ=0.8 (−16.6 punti) e δ=1.0 (−16.8 punti).
- Al punto di arresto P, ‖λ(z−x̃)‖ mediano vale: δ=0.8 → **4.3e-4** (43× la tolleranza); δ=0.2 → 4.5e-5; δ=0.4 → 1.7e-5; δ=1.0 → 3.2e-5.
- Con il criterio U, gap e violazione al termine tornano al livello di DR, o al massimo a un ordine di grandezza da esso. In particolare, l'anomalia di δ=0.8 nella Tabella 5 (gap 2.8e-5, violazione 3.0e-4) **sparisce** (1.9e-7 / 7.0e-6).
- Sconfitte rispetto a DR (Δ<0): solo a δ=1.0, 8/30 (P) e 9/30 (U).

## 4. Diagnostica del gap (δ = 0.8)

Dettaglio in `gap_diagnostic_d0.8_unperturbed_stop.txt/.csv`. Mediane su 30 istanze. k=0 è x̄^0 = proj_{X^rec}(x0); per k ≥ 1 il valore coincide con `relative_gap[k-1]` dei JSON ufficiali.

| k | DR: gap con segno | DR: #J<J* | DR: viol | DR+GNN: gap con segno | DR+GNN: #J<J* | DR+GNN: viol |
|---|---|---|---|---|---|---|
| 0 | +1.74 | 0 | 0 | +1.74 | 0 | 0 |
| 1 | −0.75 | 30 | 1e-7 | +0.37 | 0 | 0.013 |
| 2 | −0.75 | 30 | 0 | −0.03 | 17 | 0.77 |
| 5 | −0.76 | 30 | 0 | −0.18 | 21 | 1.50 |
| 10 | −0.75 | 30 | 0 | −0.18 | 22 | 0.97 |

**(a) Punti iniziali:** coincidono davvero. x0 è identico su tutte le istanze, e il gap a k=0 è identico per i due metodi.

**(b) Gap in valore assoluto e prima iterazione con gap ≤ ε:** confermato.
- `diagnostics.save_instance_plots` e `fig2_gap_levels.py` usano |J(x̄)−J*|/|J*|.
- `fig2_gap_levels.first_hit` registra la **prima** k con gap ≤ ε, anche se poi il gap risale.
- DR+GNN raggiunge |gap| ≤ 0.1 alla k mediana **3** (min 1, max 69). DR ci arriva alla k mediana 84.
- Per DR+GNN, quel valore viene da un **attraversamento di J***: il gap con segno passa da +0.37 a −0.03 tra k=1 e k=2.
  - In 16/30 istanze, alla prima k con |gap| ≤ 0.1 il costo è già **sotto l'ottimo**.
  - In **12/30** istanze |gap| torna poi sopra 0.1.
  - Nello stesso momento la violazione dei box di x̄ sale a ~1. Le iterate sono quindi tutt'altro che ammissibili.
- Su un asse y di migliaia di iterazioni, k=3 appare come "zero iterazioni".

**Altro dato emerso:** anche DR ha costo **sotto** J* (gap ≈ −0.75) per decine di iterazioni, con violazione dei box ≈ 0.
- Il motivo: x̄ = proj_{X^rec}(x) impone solo la reciprocità, e `dist_to_Xloc` misura solo i box D/G/q.
- Il **bilancio nodale** di x̄ non è misurato da nessuna delle due metriche. Nel file di diagnostica, il max|bilancio| mediano di x̄ per DR vale 9.7 a k=0, 2.6 a k=5 e 1.9 a k=10.
- Quindi il gap all'inizio confronta J* con costi di allocazioni non bilanciate. La "violazione" della Tabella 5 non vede questa parte di inammissibilità.
- La figura esistente non è stata modificata.

## 5. Nota sulla colonna e^k della Tabella 5

`paper/table_ek_comparison.py` (che produce le colonne e^k di `tabF_stopping_quality.tex`) **non riproduce DR+GNN**, per due motivi:
1. ricostruisce la GNN con `scale=1.0` fisso, mentre i checkpoint di δ=0.4/0.8/1.0 hanno `scale=1.8`;
2. il suo ciclo copiato da `dr_step` non aggiorna `prev_residual`/`stall_ema`, che sono feature d'ingresso della GNN.

Per δ=0.8 la tabella riporta e^k = 9.7e-6, mentre qui, al punto di arresto ufficiale, ‖λ(z−x̃)‖ mediano = **4.3e-4**. Valori rifatti qui (mediana al punto di arresto P), da sostituire a quelli della tabella:

| δ | qui | table_ek |
|---|---|---|
| 0.2 | 4.5e-5 | 3.9e-5 |
| 0.4 | 1.7e-5 | 9.9e-6 |
| 0.5 | 9.7e-6 | 9.5e-6 |
| 0.6 | 9.7e-6 | 9.6e-6 |
| 0.8 | 4.3e-4 | 9.7e-6 |
| 1.0 | 3.2e-5 | 1.1e-5 |

Lo stesso schema di ciclo copiato è presente anche in `paper/verify_ek.py` e `paper/gen_table_t4_timing.py`, da controllare. Non ho modificato nessuno di questi file.

## 6. File creati

Codice (nuovo):
- `DRcld/experiments/reeval_unperturbed_stop.py`: rollout (`run <out_dir> <density…>`), in parallelo, rifiuta di sovrascrivere.
- `DRcld/experiments/reeval_unperturbed_stop_report.py`: CSV, figure, tabella, diagnostica (scrive con `open(…, "x")`).

Codice modificato (opzione aggiunta, default = comportamento precedente, verificato con la riproduzione esatta):
- `DRcld/algorithm.py`: `DRConfig.stop_on`, `DRResult.dr_residual`.
- `DRcld/network.py`: `Prosumer.dr_residual_sq`, `Network.dr_residual()`.

In questa cartella:
- `trajectories_d{0.2,0.4,0.5,0.6,0.8,1.0}.json`: traiettorie complete per istanza e metodo (residuo perturbato e non perturbato, gap con segno, violazione, x0, J*).
- `per_instance_metrics_unperturbed_stop.csv`: k, Δ_i, residui al termine, gap/violazione al termine, riproduzione dell'ufficiale.
- `summary_by_density_unperturbed_stop.csv`: mediana e IQR di k e Δ; mediane di gap, violazione e residui.
- `reduction_comparison_unperturbed_stop.{pdf,png}`
- `tolerance_d08_unperturbed_stop.{pdf,png}`
- `accuracy_termination_unperturbed_stop.tex`
- `gap_diagnostic_d0.8_unperturbed_stop.{txt,csv}`
- `run_other_densities.log`
- `REPORT.md`

Riprodurre: `cd DRcld/experiments && python3 reeval_unperturbed_stop.py run <dir> 0.8` (poi le altre densità) e `python3 reeval_unperturbed_stop_report.py <dir>`.
