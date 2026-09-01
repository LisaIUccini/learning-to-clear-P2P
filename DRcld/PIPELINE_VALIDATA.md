# Pipeline validata: modello, GNN, training e loss

Questo documento descrive **solo la versione attualmente validata** della pipeline in
`DRcld/` -- cioè l'architettura e la ricetta di training che hanno effettivamente prodotto
i numeri finali riportati in `runs/INDEX.md` e nella tabella "FINAL recipe" (campagna
`dr_slope`, 2026-08-31/09-01). Il codice (`gnn.py`, `train.py`) contiene anche varianti
più vecchie o scartate (loss `hinge`/`relative`/`log_ratio`, canale lento attivo, schedule
`omega_power`): sono ancora nel repository come alternative documentate, ma **non** sono
quelle descritte qui, e non sono quelle usate per i risultati finali.

> Nota: questa pipeline è più complessa di quanto descritto nella Sezione III del paper
> attuale (architettura a due canali con gating, loss relativa/asimmetrica con
> `dr_slope`). Il disallineamento è già stato segnalato e salvato in memoria
> (`project_l2p2p_paper.md`) per quando si aggiornerà il testo.

---

## 1. Il mercato (`market.py`, Sez. II del paper)

Ogni istanza è un grafo connesso non orientato $\mathcal G=(\mathcal N,\mathcal E)$ con
$N$ prosumer. Ogni nodo $n$ ha:

- $D_n\in[\underline D_n,\overline D_n]$ (domanda flessibile), $G_n\in[\underline G_n,\overline G_n]$
  (generazione controllabile), $q_{mn}$ per ogni vicino $m$ (scambio bilaterale, $\le\kappa_{mn}$);
- $\Delta G_n\ge0$ fisso (generazione rinnovabile);
- vincolo di bilancio locale $D_n=G_n+\Delta G_n+\sum_m q_{mn}$;
- costo $J_n(x_n)=\tilde a_n(D_n-D_n^\*)^2+\tfrac12 a_nG_n^2+b_nG_n+\sum_m c_{nm}q_{mn}$
  (Eq. 3 del paper, costanti additive scartate) -- implementato in
  [network.py:106-112](network.py#L106-L112).

**Come si generano le istanze (`MarketConfig`):** si parte da un "ancora" feasible
($q=0$, $G_0$ e $D_0=G_0+\Delta G$) e si costruiscono i box attorno ad essa, così ogni
istanza è feasible per costruzione (Assunzione 1 sempre soddisfatta, mai verificata a
runtime). I range di default: $\tilde a\in[0.5,1.5]$, $a\in[0.5,2.0]$, $b\in[0.2,1.0]$,
$c\in[0.05,0.30]$, $\kappa\in[1.0,4.0]$, $\overline G\in[2,5]$, $\underline G\in[-2,0]$,
$\Delta G\in[0,2]$, semi-larghezza del box di $D$ in $[0.5,2.0]$. La densità del grafo
$\in[0,1]$ interpola tra spanning tree ($N{-}1$ archi) e grafo completo.

**Dataset usato per i risultati finali (`demo_fixed_market.py`):** un **unico** mercato
di riferimento per densità (stesso grafo, stessi parametri per tutti i nodi), $N_0=10$.
L'unica sorgente di variazione, sia in training che in test, è lo stato iniziale $x^0$
campionato casualmente dentro i box, con un seed diverso per ogni coppia
(istanza, epoca). 25 istanze di training, 30 di test (seed di valutazione fisso
`TEST_SEED=100`, indipendente dal seed di training).

---

## 2. Algoritmo base: Douglas-Rachford distribuito (`network.py`, `algorithm.py`)

Ad ogni iterazione $k$, ogni prosumer esegue in parallelo ([network.py:216-247](network.py#L216-L247)):

1. **comunica** $q_{mn}$ e riceve $q_{nm}$ da ogni vicino (unico scambio informativo);
2. **proiezione sul cono di reciprocità** (shadow): $\bar q_{mn}=q_{mn}-\tfrac12\max\{0,q_{mn}+q_{nm}\}$;
3. **riflessione**: $r_{mn}=2\bar q_{mn}-q_{mn}$ (su $D,G$ la proiezione è un no-op);
4. **prox locale**: risolve $\arg\min_{\xi\in X_n^{loc}}\gamma J_n(\xi)+\tfrac12\|\xi-r\|^2$
   dualizzando il vincolo di bilancio con un moltiplicatore scalare $\nu$, trovato per
   bisezione (puramente locale, nessuna comunicazione aggiuntiva);
5. **rilassamento**: $x_n^{k+1}=x_n^k+\lambda(z_n-\bar x_n)$, $\lambda\in(0,2)$.

Questo è esattamente l'operatore Douglas-Rachford per la coppia $(A,B)=(F+N_{X^{loc}},\,N_{X^{rec}})$
(dimostrazione formale nel Teorema 1 del paper). La sequenza $\{x^k\}$ è l'iterata
ausiliaria; l'allocazione di mercato feasible è $\bar x^k=\mathrm{proj}_{X^{rec}}(x^k)$.

Criterio di arresto usato ovunque: $\|x^{k+1}-x^k\|<\text{tol}=10^{-5}$ (norma
sull'iterata ausiliaria, sommata sui contributi locali di ogni prosumer,
[network.py:471-473](network.py#L471-L473)).

---

## 3. La GNN di perturbazione -- versione validata (`gnn.py`)

La correzione appresa entra in modo additivo, subito dopo il passo DR:
$x_n^{k+1}\leftarrow x_n^{k+1}+v_n^k$.

### 3.1 Feature di nodo (9-dim, [gnn.py:246-256](gnn.py#L246-L256))

$$h_n^k=\Big[\underbrace{\tfrac{2(D_n^k-\underline D_n)}{\overline D_n-\underline D_n}-1}_{h_D},\;
\underbrace{\tfrac{2(G_n^k-\underline G_n)}{\overline G_n-\underline G_n}-1}_{h_G},\;
\underbrace{\tfrac{2(D_n^\*-\underline D_n)}{\overline D_n-\underline D_n}-1}_{h_{D^\*}},\;
\underbrace{\tfrac{D_n-G_n-\Delta G_n-Q_n}{\overline D_n-\underline D_n}}_{h_{bal}},\;
h_{prev},\; h_{stall},\;
\tilde a_n^{norm},\,a_n^{norm},\,b_n^{norm}\Big]$$

- `h_bal`: residuo di bilancio corrente (qui entra $\Delta G_n$, implicitamente, non come
  feature propria);
- `h_prev`: $\|x_n^{k}-x_n^{k-1}\|$ normalizzato -- quanto si è mosso IL NODO STESSO
  all'iterazione precedente (0 a $k=0$);
- `h_stall`: media mobile esponenziale di `h_prev` (decay 0.95, finestra effettiva
  ~20 iterazioni, [network.py:41](network.py#L41)) -- un segnale di "stallo sostenuto";
- i coefficienti di costo $\tilde a_n,a_n,b_n$ normalizzati contro range fissi
  (`GNNConfig`, uguali ai range di `MarketConfig`), non contro statistiche di batch,
  perché un nodo non può calcolare riduzioni globali.

**Nota:** questo differisce dalla feature del paper (Eq. 9), che ha $\Delta G_n/\sigma_G$
come termine esplicito e $\delta_n^k=z_n^k-\bar x_n^k$ (spostamento DR istantaneo) al
posto di `h_prev`/`h_stall`. Stessa dimensione (9), contenuto diverso.

### 3.2 Feature di arco (3-dim, [gnn.py:258-262](gnn.py#L258-L262))

Per ogni arco diretto $(m\to n)$:
$$e_{mn}^k=\Big[\,q_{mn}^k/\kappa_{mn},\;\;q_{nm}^k/\kappa_{mn},\;\;c_{nm}^{norm}\,\Big]$$

Nel paper (Eq. 10) la feature di arco è 5-dim (aggiunge il residuo di reciprocità
$(q_{mn}+q_{nm})/\kappa$ e uno spostamento DR sul trade); nel codice questi due termini
non ci sono.

### 3.3 Message passing e memoria ricorrente ([gnn.py:274-285](gnn.py#L274-L285))

$$\mathrm{msg}_{m\to n}=\phi_{msg}(e_{mn}^k)\in\mathbb R^{d_h},\qquad
\mathrm{agg}_n^k=\frac{1}{\sqrt{\max(1,|\Omega_n|)}}\sum_{m\in\Omega_n}\mathrm{msg}_{m\to n}$$
$$\mathrm{mem}_n^k=\mathrm{GRUCell}\big([h_n^k;\mathrm{agg}_n^k],\,\mathrm{mem}_n^{k-1}\big),\qquad \mathrm{mem}_n^{-1}=0$$

$\phi_{msg}$ è un MLP a 2 layer (Linear-Tanh-Linear) $3\to d_h\to d_h$, condiviso su tutti
gli archi. $d_h=32$ (`GNN_HIDDEN_DIM`), fisso, indipendente da $N$ e $|\Omega_n|$ -- è
questo che rende un unico modello addestrato deployabile su grafi di taglia/topologia
diverse.

### 3.4 Output: **un solo canale attivo**, con gating ([gnn.py:191-310](gnn.py#L191-L310))

L'architettura nel codice prevede DUE canali (fast/slow) ciascuno con MLP di
nodo/arco/gate propri, ma nella ricetta validata il canale lento è **spento per
costruzione**: `GNN_SCALE_SLOW=0.0`, quindi il suo contributo è esattamente zero
qualunque cosa impari, e il codice lo salta anche a livello di calcolo
([gnn.py:306-309](gnn.py#L306-L309), [gnn.py:394-398](gnn.py#L394-L398)). Nella pratica
attuale, quindi, resta solo il **canale fast**, ma CON un gate sigmoide che il paper non
menziona:

$$\mathrm{gate}_n^k=\sigma(\phi_{gate}(\mathrm{mem}_n^k))\in(0,1)$$
$$\Delta_{node,n}^k=\mathrm{gate}_n^k\odot\tanh(\phi_{node}(\mathrm{mem}_n^k))\in\mathbb R^2\;(D,G)$$
$$\Delta_{mn}^k=\mathrm{gate}_n^k\cdot\tanh(\phi_{edge}([\mathrm{mem}_n^k;e_{mn}^k]))\quad\text{per ogni }m$$

Il gate è un MLP $d_h\to d_h\to1$ con bias iniziale $-2.0$ (parte quasi "spento": la rete
deve imparare a "convincersi" ad agire, non il contrario). Le direzioni sono poi
riscalate in unità fisiche (metà larghezza del box di $D$/$G$, o $\kappa_{mn}$ per i
trade) e moltiplicate per l'ampiezza a decadimento geometrico:

$$v_n^k=\underbrace{\rho^k}_{\text{decay}}\cdot\underbrace{\text{scale}}_{\text{ampiezza}}\cdot(\Delta_{node,n}^k,\{\Delta_{mn}^k\}_m)\cdot(\text{scala fisica})$$

con $\rho\in(0,1)$ fisso a priori (non allenato) e **`scale` allenato indirettamente
solo tramite gate e tanh**, non come parametro libero.

### 3.5 Sommabilità (Teorema 1)

Ogni componente di $\Delta$ è limitata in $[-1,1]$ (tanh) per costruzione, e ulteriormente
ristretta dal gate in $[0,1]$: per QUALSIASI peso $w$,
$\|v^k\|\le \text{scale}\cdot\rho^k\cdot\|\text{scala fisica}\|$, quindi
$\sum_k\|v^k\|\le \text{scale}/(1-\rho)\cdot\|\text{scala fisica}\|<\infty$. Questa è
l'ipotesi che il Teorema 1 richiede, e vale indipendentemente da come viene allenato $w$
(anche pesi casuali darebbero un $v^k$ sommabile).

### 3.6 Due percorsi di calcolo, stessa funzione

- `forward_centralised`: versione vettorizzata su tutto il grafo, usata SOLO in training
  (serve lo stato completo per il backprop attraverso la traiettoria srotolata);
- `forward_local`: quello che un singolo prosumer esegue a deployment, usando solo il
  proprio stato, la propria memoria GRU privata, e i $q$ ricevuti dai vicini (già
  scambiati per il passo DR, nessuna comunicazione aggiuntiva).

`test_gnn_parity.py` verifica che le due strade calcolino esattamente la stessa funzione
-- è la garanzia concreta dietro l'affermazione "inferenza pienamente distribuita, zero
overhead di comunicazione" del paper.

---

## 4. Loss di training -- versione validata: `asymmetric_loss` + `dr_slope` (`train.py`)

`train.py` implementa 5 loss diverse (`absolute`, `relative`, `hinge`, `log_ratio`,
`asymmetric`); **`asymmetric` è l'unica usata per tutti i risultati finali** riportati in
`runs/INDEX.md` e nella tabella "FINAL recipe". Le altre restano nel codice come
tentativi documentati (compresi i motivi per cui sono stati scartati), non descrivono la
pipeline attuale.

### 4.1 Perché serve un rollout DR di riferimento

Ad ogni step di training, per la stessa istanza e lo stesso $x^0$, si eseguono DUE
rollout di $K$ iterazioni ([train.py:132-173](train.py#L132-L173)):

1. uno **con** la GNN, differenziabile (`rollout(net, gnn, ..., differentiable=True)`);
2. uno **plain DR**, senza gradiente (`rollout(net, None, ..., differentiable=False)`) --
   serve solo a fornire un termine di paragone per-istanza, mai come target/etichetta
   precalcolata, e non entra mai nell'inferenza (la garanzia di convergenza del
   Teorema 1 non dipende da questo, riguarda solo la sommabilità di $v^k$).

Da entrambi si estrae il residuo normalizzato ad ogni step:
$$\rho^k=\frac{\|x^{k+1}-x^k\|}{\|\sigma\|+\varepsilon},\qquad
\rho_{DR}^k=\frac{\|x_{DR}^{k+1}-x_{DR}^k\|}{\|\sigma\|+\varepsilon}\;\;(\text{no-grad})$$
dove $\sigma$ è la scala fisica di riferimento (stessa di `physical_scale`, Sez. 3.5).

### 4.2 La loss

$$a^k=\rho^k-\rho_{DR}^k,\qquad
\ell(a^k) = w_{behind}\cdot\mathrm{relu}(a^k) - w_{ahead}\cdot\mathrm{relu}(-a^k)$$

con $w_{behind}=2.0$, $w_{ahead}=1.0$ ([train.py:329-417](train.py#L329-L417)): restare
indietro rispetto al DR puro pesa il DOPPIO di quanto pesa il premio per stare avanti --
questa asimmetria (non i valori assoluti) è ciò che rende la loss diversa da una
differenza pesata semplice: con $w_{behind}=w_{ahead}$ collasserebbe esattamente al
gradiente di `stability_loss` (il termine $\rho_{DR}^k$, essendo costante rispetto ai
pesi, sparirebbe dal gradiente).

A differenza di una loss con margine fisso (`hinge`), qui il premio per "congelarsi"
($\rho^k\to0$) è limitato da $\rho_{DR}^k$ stesso (non da una costante arbitraria):
$a^k\to-\rho_{DR}^k$, quindi il vantaggio del congelamento è proporzionale a quanto si
muoveva DR stesso in quel punto, non un valore fisso sfruttabile.

**Riponderazione per iterazione -- `omega_mode="dr_slope"`:** invece di pesare ogni
iterazione allo stesso modo, si pesa in base a quanto LENTAMENTE sta decrescendo la
traiettoria di RIFERIMENTO (plain DR) in quel punto, in scala logaritmica
([train.py:371-406](train.py#L371-L406)):
$$\text{log\_decrease}^k=\max\{0,\log\rho_{DR}^k-\log\rho_{DR}^{k+1}\},\qquad
\omega^k=\frac{\overline{\text{log\_decrease}}}{\text{log\_decrease}^k+0.1\cdot\overline{\text{log\_decrease}}}$$
(capped a ~10x il valore tipico su un vero plateau), poi rinormalizzato a media 1. Un
plateau genuino di DR (poco calo relativo) viene sovrappesato; una tratta di decadimento
sano viene sottopesata. Uno *guard* esplicito esclude la coda "già convergiuta" (quando
$\rho_{DR}^k$ è già entro 2x del suo valore finale) dal sovrappeso, altrimenti un
srotolamento che finisce presto (comune con eterogeneità ampia in $x^0$) verrebbe scambiato
per un plateau vero.

Loss finale per istanza:
$$\mathcal L = \overline{\omega^k\cdot\ell(a^k)} + \gamma_{mono}\cdot\overline{\mathrm{relu}(\rho^{k+1}-\rho^k)}\qquad(\gamma_{mono}=2.0)$$

**Importante:** `omega_mode="dr_slope"` funziona per densità 0.3, 0.8, 1.0 (entrambi i
seed), ma **fallisce** per densità 0.5 (0.80-0.82x, 0% win-rate in tre varianti diverse
provate) -- causa: un overshoot autoindotto che crea un plateau artificiale. Per
densità 0.5 la ricetta finale usa `omega_mode="fixed"` con `omega_power=0.0`
(riponderazione uniforme, cioè nessuna riponderazione).

### 4.3 Loop di training (`Trainer.fit`, [train.py:449-483](train.py#L449-L483))

Per ogni epoca, per ogni istanza di training (25, tutte lo stesso mercato con $x^0$
diverso): un passo Adam per istanza (non full-batch), `grad_clip=1.0`. Nessun target di
equilibrio, nessun rollout precalcolato salvato -- il riferimento DR è ricalcolato fresco
ad ogni step. Validazione (facoltativa) sulle prime 5 istanze del test set, stessa loss,
senza gradiente.

---

## 5. Ricetta finale per densità (iperparametri esatti)

Tutte le run condividono: $N_0=10$, $\lambda=1.0$, tolleranza DR $10^{-5}$, 30 passi di
bisezione, `scale=1.0` (non il default `0.4` di `config.py`), canale lento spento
(`scale_slow=0`, `rho_slow=0.99` ma ininfluente), $\eta=0.15$ (vestigiale, usato solo
dalla loss `hinge` che qui non è attiva), $w_{behind}=2.0$, $w_{ahead}=1.0$,
$\gamma_{mono}=2.0$, 60 epoche, lr $3\times10^{-3}$, 25 istanze di training / 30 di test.
"seed" nella colonna sotto è il seed con cui è stato generato il MERCATO di base
(`BASE_MARKET_SEED`, grafo+parametri), non il seed di training (sempre 0).

| densità | seed mercato | $\gamma$ (DR) | unroll $K$ | $\rho$ | `omega_mode` |
|---|---|---|---|---|---|
| 0.3 | 0 | 4.0 | 200 | 0.95 | `dr_slope` |
| 0.5 | 0 | 6.0 | 305 | 0.9672 | `fixed` (uniforme -- `dr_slope` fallisce qui) |
| 0.8 | 0 | 8.0 | 750 | 0.9925 | `dr_slope` |
| 1.0 | 0 | 4.0 | 300 | 0.9667 | `dr_slope` |
| 1.0 | 1 | 4.0 | 349 | 0.9714 | `dr_slope` |

$\gamma$ e $K$/$\rho$ sono scelti per ogni mercato in base al tempo medio di
convergenza del DR puro su quel mercato specifico (non un valore universale) --
vedi [[feedback-gamma-variance-and-loss-cap]] per il motivo (una scelta di $\gamma$ che
minimizza la varianza tra istanze distrugge proprio l'eterogeneità che la correzione
appresa deve sfruttare).

---

## 6. Protocollo di valutazione e metriche (`demo.py`)

Per ogni istanza di test: si esegue DR puro e DR+GNN a partire dallo STESSO $x^0$
(stesso seed), fino a convergenza o al cap di iterazioni ([demo.py:39-53](demo.py#L39-L53)).

$$\text{speedup}_i=\frac{\text{iter}_{DR,i}}{\text{iter}_{learned,i}},\qquad
\text{win-rate}=\frac{|\{i:\text{speedup}_i\ge1\}|}{n}$$

Si riportano media, mediana, min, max dello speedup e il win-rate
([demo.py:100-117](demo.py#L100-L117)). Viene inoltre verificata la feasibility
dell'iterata shadow $\bar x^k$ (bilancio, reciprocità, box) e, separatamente, che il
guadagno non sia un artefatto della soglia di stop (controllo QP contro l'ottimo vero,
vedi `equilibrium.py`).

---

## 7. Risultati finali validati

**Sullo stesso mercato di training (2026-08-31):**

| densità | win-rate | speedup medio | speedup max |
|---|---|---|---|
| 0.3 | 100% | **2.96x** | -- |
| 0.5 | 100% | 1.81x | -- |
| 0.8 | 100% | **1.83x** | -- |
| 1.0 (seed 0) | 100% | **2.09x** | -- |
| 1.0 (seed 1) | 100% | **1.79x** | -- |

**Zero-shot (nessun riaddestramento, stessi pesi), due assi testati separatamente
(2026-09-01):**

- *D\*/G_res $\pm10\%$* (fattore condiviso per istanza, box invariati): densità 0.3 →
  2.42x/100%; densità 0.5 → 1.57x/97%; densità 0.8 → 1.34x/100%; densità 1.0 (seed 0) →
  1.85x/100%; densità 1.0 (seed 1) → 1.68x/100%.
- *topologia $\pm10\%$* (nodi/archi, parametri di mercato invariati): densità 0.3 →
  2.28x/97%; densità 0.5 → 1.31x/**63%** (unico caso debole: usa i pesi allenati con
  `omega_mode="fixed"`, non `dr_slope`); densità 0.8 e 1.0 non ancora completati per
  questo asse alla data di stesura.

---

## 8. Limiti noti (dal codice stesso, non nascosti)

- Circa il 28% delle istanze più difficili (traiettorie con migliaia di iterazioni)
  restano perse sotto ogni variante di loss provata: il training srotola solo K=200-750
  iterazioni, quindi non vede mai il plateau che decide l'esito di quelle istanze --
  limite di una policy condivisa allenata così, non un bug (vedi il docstring di
  `hinge_loss` in `train.py` per l'analisi di un caso concreto).
- `omega_mode="dr_slope"` è instabile su densità 0.5 per un motivo specifico
  (overshoot autoindotto), non generalizzabile a priori -- ogni densità ha richiesto una
  diagnosi propria, non un'unica ricetta universale.
