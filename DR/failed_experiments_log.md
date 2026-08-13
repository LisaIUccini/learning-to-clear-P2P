# Registro dei tentativi che hanno peggiorato i risultati

Questo documento raccoglie **tutti** i tentativi di miglioramento provati nel progetto L2Play che, misurati end-to-end, sono risultati peggiori (o non chiaramente migliori) della configurazione validata. Per ciascuno: la motivazione che lo ha suggerito, cosa è stato cambiato, il risultato misurato, e l'interpretazione/lezione tratta. L'obiettivo è non ripetere questi esperimenti senza sapere che sono già stati fatti.

Il codice conserva la stessa documentazione in forma più sintetica nei docstring (`training.py`, `example_l2play.py`, `example_fixedgraph.py`) — questo documento la espande e la organizza in un unico posto.

## Configurazione di riferimento (validata)

Douglas-Rachford (gamma=2.0, lambda=1.0, bisezione=25) + perturbazione appresa da una GNN (memoria GRU, gate di confidenza, rho=0.9, unroll=400 in training, nessun TBPTT), allenata 40 epoche, learning rate costante 3e-3, loss auto-supervisionata a due termini (decadimento + monotonicità, gamma_mono=2.0). Risultati: **2.12x** su dataset a grafo variabile (30 istanze test), **1.23x** su dataset a grafo fisso e denso (N=10, densità=0.7, 10 istanze test).

---

## 1. Progettazione della loss di training

### 1.1 Tentativo 1 — residuo relativo a DR, con logaritmo

**Motivazione**: la rete deve imparare a muoversi più velocemente verso l'equilibrio; il modo più diretto sembrava minimizzare il residuo tra iterazioni consecutive, r_k = ||x̄^{k+1} - x̄^k||, confrontato contro il residuo che avrebbe DR non perturbato sulla stessa istanza: L = media_k[log(r_k+eps) - log(r_k^DR+eps)].

**Risultato**: **SFRUTTABILE**. Una perturbazione che quasi "congela" la traiettoria (x^{k+1} ≈ x^k) manda r_k → 0 e log(r_k) → -infinito: una ricompensa illimitata. Il costo di quel blocco (il tratto finale non perturbato deve recuperare la distanza non percorsa) è solo un aumento ordinario e limitato del log-residuo più avanti. La loss di training sembrava eccellente, ma misurato end-to-end: **0.73x — più lento di DR puro**.

**Lezione**: qualunque termine di loss senza un limite superiore naturale (qui il logaritmo) è a rischio di essere sfruttato da un ottimo locale degenere (congelamento) invece che dal comportamento genuino desiderato.

### 1.2 Tentativo 2 — stessa idea, ma mediata solo sulla coda

**Motivazione**: se il problema è che il congelamento iniziale "costa" solo più avanti, forse pesare la loss solo sulla coda (k oltre l'orizzonte di decadimento dell'ampiezza) evita di premiare il congelamento iniziale.

**Risultato**: **PARZIALE**. Ha aiutato (0.73x → 1.23x) ma si è fermato ben lontano da un miglioramento affidabile e uniforme: i guadagni si concentravano sulle istanze già lente (fino a 2.16x), mentre circa un terzo delle istanze normali/veloci **peggiorava** (0.61x-0.99x).

**Lezione**: la causa di fondo sopravvive al finestramento — r_k misura quanto si è mosso UN passo, non quanto si è vicini alla soluzione. Le due cose sono correlate solo debolmente.

### 1.3 Tentativo 3 — ancoraggio a un equilibrio precalcolato

**Motivazione**: se il problema è che r_k non misura la distanza dalla soluzione, si può calcolare una volta (a basso costo, con DR portato a convergenza) un punto quasi-equilibrio x̄* per ogni istanza e minimizzare d_k/d_k^DR + beta·relu(d_k/d_k^DR - 1), con d_k = ||x̄^k - x̄*||.

**Risultato**: **FUNZIONAVA BENE**, ma è stato **accantonato per principio**, non per prestazioni. x̄* gioca il ruolo di un'etichetta/target — anche se non costa nulla in più da ottenere e proviene dallo stesso algoritmo che si vuole accelerare (non un oracolo esterno), viola il requisito di auto-supervisione stretta stabilito per il progetto: nessuna soluzione di riferimento, precalcolata o meno, deve entrare nella loss.

**Lezione**: un vincolo di design (qui: auto-supervisione stretta) può scartare una soluzione empiricamente valida — è stata una scelta deliberata, non un fallimento tecnico.

### 1.4 Rimozione di `late_weight`

**Motivazione**: nella ricerca della causa del "collasso a un passo generico" (vedi sezione 2.6), si è ipotizzato che il termine `late_weight` (che pesa maggiormente le iterazioni tardive nella loss, aggiunto per il problema "mai peggio di DR") diluisse il segnale specifico per istanza nelle prime iterazioni.

**Risultato**: **CONFUTATA**. Disattivarlo da solo non ha risolto il collasso — anzi è risultato leggermente peggiore (ancora più uniforme, ~4000-4300 iterazioni su tutte le istanze invece di ~3000-3200). Rimosso comunque dal codice: non era più una salvaguardia funzionante, solo complessità senza giustificazione residua.

**Lezione**: il peso della loss non era la causa del collasso — utile per escludere un'ipotesi, anche se non ha risolto il problema.

### 1.5 Tentativo 5 — residuo di punto fisso pre-perturbazione (in sostituzione)

**Motivazione**: sia r_k (termine di decadimento) sia il residuo ausiliario (termine di monotonicità) sono misurati DOPO che v^k è stato aggiunto. In teoria la rete potrebbe scegliere v^k ≈ -lambda·(z^k - x̄^k) per annullare esattamente il passo di DR e "congelare" la traiettoria, riducendo il residuo misurato senza progresso reale (stessa patologia del tentativo 1, ma non sfruttabile tramite un premio illimitato stavolta). La proposta: sostituire entrambi i termini con d^k = ||z^k - x̄^k||, il classico residuo di punto fisso DR/KM (l'uscita del prox di linea 6 meno l'ombra di linea 4), misurato PRIMA di v^k — quindi v^k non può comparire in nessuno dei due lati della differenza, chiudendo la scappatoia. Ha anche una lettura teorica pulita: d^k=0 se e solo se x^k è un punto fisso di DR; la subregolarità metrica dà dist(x^k, Fix) ≤ kappa·d^k per la struttura poliedrale/quadratica di questo problema, quindi è un surrogato di distanza legittimo, non solo un'euristica.

**Risultato**: **PEGGIORE**, nonostante la teoria solida. Testato su due dataset indipendenti:
- dataset a grafo variabile (30 istanze): 2.12x → 1.45x, regressioni 7/30 → 13/30
- dataset a grafo fisso denso (10 istanze): 1.23x → 1.10x, regressioni 2/10 → 6/10

**Interpretazione**: proprio perché d^k esclude v^k dallo stesso indice, v^k non riceve NESSUN segnale di gradiente diretto dal termine di loss del passo corrente — solo uno indiretto e diluito, attraverso l'effetto di v^k sullo stato dei passi successivi. Il segnale teoricamente più pulito e a prova di sfruttamento si è rivelato un segnale di training praticamente più debole per la variabile decisionale vera e propria, e quel costo ha superato il beneficio di chiudere una vulnerabilità che la discesa del gradiente forse non stava nemmeno sfruttando da questa inizializzazione.

### 1.6 Tentativo 6 — residuo di punto fisso aggiunto (non in sostituzione)

**Motivazione**: dato che il tentativo 5 ha mostrato che d^k da solo dà un segnale debole per v, l'idea è stata di NON sostituire i termini validati ma AGGIUNGERE d^k come terzo termine sopra di essi, così il termine di decadimento continua a dare a v^k un segnale di gradiente diretto, mentre d^k fa da guardia contro lo stesso sfruttamento per congelamento. In parallelo, è stata avanzata l'ipotesi che il termine di monotonicità (gamma_mono, che penalizza ogni aumento del residuo ausiliario) potesse scoraggiare traiettorie che fanno un passo temporaneamente aggressivo ma poi convergono più in fretta — quindi si è provato anche a toglierlo, lasciando che la monotonicità emerga come conseguenza invece di essere imposta.

**Risultato**: **PEGGIORE in entrambe le varianti testate**, sullo stesso dataset a grafo fisso N10D07, a parità di tutto il resto (rho=0.9, unroll=400, 40 epoche, lr=3e-3):
- gamma_mono=0 (tolto) + w_fix=1 (aggiunto): 1.23x → **1.13x**, regressioni 2/10 → **5/10**, con l'istanza migliore del baseline (1.79x) crollata alla peggiore (**0.45x** — più lenta di DR puro, con più del doppio delle sue iterazioni).
- gamma_mono=2 (mantenuto invariato) + w_fix=1 (aggiunto), ri-testato su un dataset di test allargato a 30 istanze per escludere che il campione da 10 fosse solo rumoroso (un'unica istanza spostava la media di 0.1-0.15x in entrambi gli esperimenti sopra): ancora peggiore, 1.23x → **1.11x**.

**Interpretazione**: il termine di monotonicità non si limitava a bloccare traiettorie aggressive ma utili (l'ipotesi originale) — stava facendo un lavoro di stabilizzazione reale. Toglierlo o diluirlo, anche aggiungendo un termine extra teoricamente ben motivato (d^k, a prova di sfruttamento per costruzione), costa più in varianza tra le istanze di quanto guadagni in velocità media.

---

## 2. Estensione dell'orizzonte di training (rho/unroll)

**Contesto**: scale·rho^k è numericamente ~0 già per k≈150-200 quando rho=0.9 (verificato: 1.3e-23 a k=500) — un limite fisico duro su quanto tardi un plateau può essere raggiunto, indipendentemente dal training. Alzare rho è l'unico modo per alzare questo limite, ma ogni tentativo di farlo funzionare è fallito finora.

### 2.1 Override di rho solo in fase di deployment

**Motivazione**: verificare rapidamente, senza ritraining, se esiste margine reale prima di investire in un retraining completo — stessi pesi allenati a rho=0.9, rho cambiato solo all'inferenza.

**Risultato**: **PARZIALE, dimostra il margine ma non è utilizzabile**. Un'istanza difficile è passata da 1.31x a 57x con rho=0.95 — prova che il margine esiste — ma le istanze facili peggiorano sempre di più all'aumentare di rho, perché il gate non è mai stato allenato ad auto-regolarsi oltre k≈150: non sa cosa fare con il guinzaglio più lungo.

### 2.2 Retraining completo a rho=0.97, unroll=800

**Motivazione**: se il gate non sa auto-regolarsi con più margine perché non è mai stato allenato in quel regime, allenarlo direttamente a orizzonte lungo dovrebbe risolvere il problema.

**Risultato**: **PEGGIORE**. 1.41x medio, 19-30 istanze regredite, contro il baseline rho=0.9 (3.42x, 9-30 regredite — nota: numeri di uno specifico pannello di verifica, non l'intero test set da 30). Il training su un orizzonte più lungo e rumoroso converge a una policy di qualità inferiore, non a una versione più paziente di quella buona.

### 2.3 Screening economico (proxy) dello stesso rho=0.97/unroll=800

**Motivazione**: prima di investire nel training completo (costoso), verificare con un proxy a basso costo (24 istanze, 10 epoche) se la configurazione è promettente.

**Risultato**: **PROXY INAFFIDABILE**. Il proxy dava un segnale opposto e ottimistico (4.88x, sembrava ottimo) rispetto al training completo che si è rivelato peggiore. Il training a scala ridotta non è un predittore affidabile per questa leva (rho/unroll).

### 2.4 Curva di checkpoint per epoca dello stesso run

**Motivazione**: forse il modello a 40 epoche del run rho=0.97/unroll=800 (risultato negativo) era già oltre il suo punto ottimale — controllare epoche intermedie (4, 8, ..., 40) per vedere se una di esse recupera la qualità del baseline.

**Risultato**: **NEGATIVO**. Curva rumorosa, nessuna singola epoca recupera la qualità del baseline rho=0.9. Esclude l'ipotesi "basta fermarsi prima".

### 2.5 Curriculum (rho 0.90 → 0.99 → 0.997 a stadi)

**Motivazione**: se il salto diretto a rho alto è troppo brusco, un'estensione graduale (ogni stadio continua il training dai pesi dello stadio precedente, invece di ripartire da zero) potrebbe evitare il collasso.

**Risultato**: **PEGGIORE A OGNI STADIO**. 1.38x → 0.51x → 0.28x su un pannello rappresentativo — l'estensione graduale non è un rimedio, anzi è peggiore di un salto diretto.

### 2.6 Diagnosi: collasso a un "passo generico"

**Osservazione** (non un tentativo di per sé, ma la scoperta che ha guidato i successivi): a orizzonti lunghi la rete converge verso UN'unica cadenza di convergenza applicata quasi identicamente a ogni istanza, indipendentemente da quanto sia diversa la velocità naturale di DR per quella istanza — verificato: rapporti di decadimento iterazione-su-iterazione quasi identici tra istanze i cui baseline DR differiscono di 5x. Smette di essere una politica adattiva per istanza e diventa una politica unica per tutte.

### 2.7 Disattivazione di `late_weight` per risolvere il collasso

**Motivazione**: vedi 2.6 — ipotesi che `late_weight` diluisse il segnale specifico per istanza nelle prime iterazioni, causando il collasso al passo generico.

**Risultato**: **CONFUTATA** (stesso esperimento della sezione 1.4, testato nel contesto dell'estensione dell'orizzonte). Leggermente peggiore, non meglio.

### 2.8 Diagnosi finale: gradiente diluito su BPTT lungo

**Conclusione** (non un tentativo, la diagnosi di fondo): il problema sembra un gradiente evanescente/diluito dalla backpropagation attraverso migliaia di passi ricorrenti in un unico passaggio, non un problema di forma della loss — l'assegnazione di credito per "fai questa cosa specifica su QUESTA istanza, presto" viene lavata via su così tanti passi di BPTT, lasciando solo il segnale generico e facile da imparare a orizzonte lungo. Si è scartata l'idea di costruire a mano una feature esplicita "sono bloccato?" (decide in anticipo cosa la rete deve cercare, invece di lasciarglielo imparare dai dati). Il rimedio standard per questa classe di problema è il **TBPTT troncato** (implementato, `TrainConfig.tbptt_segment`) — funziona correttamente come infrastruttura ma non ha risolto il collasso al passo generico a rho=0.997 (vedi nota in `example_fixedgraph.py`: run TBPTT/rho=0.997 sceso fino a 0.56x medio sul dataset a grafo fisso denso).

**Stato (prima del tentativo 2.9)**: nessuna delle otto strade ha esteso in modo affidabile la portata oltre rho=0.9/unroll=400. Questo resta il tetto pratico consolidato per l'architettura/approccio di training attuale.

### 2.9 Decadimento polinomiale invece che geometrico (famiglia diversa, non solo un'estensione di rho)

**Motivazione**: gli otto tentativi precedenti (2.1-2.8) hanno sempre allungato la portata restando nella STESSA famiglia geometrica (rho più alto). Il teorema di convergenza perturbata (Combettes) non richiede che la successione sommabile $a_k$ sia geometrica — richiede solo $\sum_k a_k < \infty$ per una successione fissa, indipendente da theta. Una successione polinomiale $a_k=(k+1)^{-p}$ con $p>1$ è sommabile (zeta di Riemann) ma decade molto più lentamente: a k=6000, rho^k (rho=0.9) è numericamente zero, mentre (k+1)^-1.1 è ancora ~7e-5. Diagnosticato prima con una misura diretta (nessun training necessario): la finestra di training (unroll=400) osserva solo una piccola frazione (10-24%) del vero orizzonte di convergenza della DR (fino a 6321-9717 iterazioni su questo dataset), e questo vale ugualmente per istanze che poi migliorano o peggiorano nel deployment reale — quindi non è un problema di ponderazione per istanza, ma di orizzonte non osservato. L'idea: un decadimento più lento lascerebbe la perturbazione rilevante fin dove la DR reale ne avrebbe davvero bisogno, cambiando la FORMA del decadimento invece che solo la sua velocità entro la stessa famiglia.

**Verifiche economiche PRIMA del training completo** (per non ripetere l'errore della sezione 2.3, dove uno screening a basso costo diede un segnale opposto a quello del training completo): (1) calcolo statico del budget totale $\sum_k a_k$ per vari $p$ — scelto $p=1.1$ (budget totale ~10.58, vicino al 10.0 della configurazione geometrica attuale, così il confronto isola la FORMA e non il budget totale); (2) controllo di salute del gradiente (3 epoche, 6 istanze, unroll 400 e 800, sia single-shot sia con TBPTT) — nessun NaN/Inf, loss e norme del gradiente comparabili al controllo geometrico. Entrambi i controlli economici sono passati puliti.

**Configurazione del training completo** (nessuna scorciatoia sugli epoch/istanze, per lo stesso motivo della sezione 2.3): decadimento polinomiale p=1.1, TBPTT con 6 finestre da 400 passi (unroll totale=2400), baseline gamma=1.0 corretto (vedi nota "CONFOUND FOUND" in `documentation.txt`), 40 epoche x 50 istanze, tutto il resto identico alla ricetta validata. Costo: ~5.2 ore di training + ~0.5 ore di valutazione distribuita (~6.2 ore totali).

**Risultato**: **PEGGIORE, in modo uniforme e marcato**. La loss di training è scesa molto più in basso del solito (0.0007 contro ~0.0035-0.0040 della configurazione geometrica) — un altro promemoria che una loss auto-referenziale più bassa non implica affatto un risultato migliore in deployment (vedi pattern ricorrente #1 sotto). Ma sulle 10 istanze di test **TUTTE E 10 sono peggiorate** (mai successo prima in questo registro — di solito il pattern è "alcune migliorano molto, altre peggiorano"), da x0.65 (la migliore) a x0.07 (la peggiore, istanza con DR=639 diventata 9308 iterazioni apprese). Mean speed-up: **0.27x** — quattro volte più lento della configurazione geometrica validata (1.25x) sullo stesso identico baseline gamma=1.0.

**Diagnosi del meccanismo** (verificata direttamente sulla traiettoria dell'istanza peggiore, non solo ipotizzata): il residuo scende regolarmente e arriva vicinissimo alla soglia (3.5e-5 a k=1200, contro DR pura che a quel punto avrebbe già finito) — quindi NON è il congelamento dell'attempt 1 (l'ipotesi iniziale, poi scartata perché i dati non la confermano). Il problema è nella FASE FINALE: da k=6000 a k=9000 il residuo resta bloccato appena sopra 1e-6 per oltre 3000 iterazioni (a un certo punto risale persino leggermente, da 1.42e-6 a 1.46e-6), mentre il modello geometrico sulla STESSA istanza completa una discesa pulita da 1.3e-5 a convergenza in appena 74 iterazioni. Interpretazione: il teorema di Combettes garantisce la convergenza EVENTUALE per qualunque perturbazione sommabile, ma non dice nulla sulla VELOCITÀ vicino al punto fisso. La DR non perturbata converge lì con una contrazione pulita, tipicamente geometrica; il decadimento geometrico validato "esce di scena" abbastanza in fretta da non disturbare quella fase; QUALSIASI decadimento polinomiale, essendo asintoticamente più lento di qualunque tasso geometrico fisso (non importa quanto grande sia $p$), prima o poi diventa il termine dominante rispetto a un residuo in contrazione geometrica, e continua a iniettare un disturbo piccolo ma persistente proprio nella fase in cui la DR stava già facendo il suo lavoro migliore — trasformando uno sprint finale pulito in un'agonia lenta e rumorosa.

**Lezione**: non è solo "p=1.1 era la scelta sbagliata" — è strutturale a TUTTA la famiglia dei decadimenti più lenti del geometrico, non solo a questo valore specifico di $p$: un $p$ più grande sposterebbe solo più in là il punto in cui l'interferenza si manifesta, senza eliminarla. Questo chiude un intero asse di attacco (cambiare la FORMA del decadimento), non solo un esperimento — in modo simmetrico a come le sezioni 2.1-2.8 hanno chiuso l'asse "stessa forma, decadimento più lento" (rho più alto). Insieme, i due risultati suggeriscono una tensione strutturale: estendere la portata della perturbazione allungandone la durata di vita sembra sempre incontrare un ostacolo diverso ma ugualmente serio — credito del gradiente diluito su BPTT lungo da un lato (2.1-2.8), interferenza con la contrazione finale della DR dall'altro (2.9) — non risolvibile scegliendo meglio un singolo iperparametro all'interno della stessa famiglia di tentativi.

**Stato attuale**: nessuna delle nove strade (otto geometriche + questa polinomiale) ha esteso in modo affidabile la portata oltre rho=0.9/unroll=400. Questo resta il tetto pratico consolidato per l'architettura/approccio di training attuale.

---

## 3. Schedule di training (epoche e learning rate)

### 3.1 Più epoche (80 invece di 40) + decadimento del learning rate

**Motivazione**: la curva di loss del modello validato (40 epoche) mostrava sia il training loss sia il validation loss ancora in discesa all'epoca 40, senza segni di overfitting (val loss segue da vicino il train loss) — a differenza dei tentativi di estensione dell'orizzonte (sezione 2), qui non c'era un problema strutturale noto, sembrava semplicemente "il training non è ancora finito". Il train loss era anche piuttosto rumoroso nelle ultime epoche, indizio che il learning rate costante (3e-3) fosse troppo alto in fase finale — da cui l'aggiunta di un decadimento cosine (3e-3 → 3e-4) invece di semplicemente allungare il training alla stessa velocità.

**Risultato**: **PEGGIORE**, nonostante il training loss sia effettivamente sceso in modo pulito fino a un valore inferiore (0.0036 contro 0.0037 del baseline a 40 epoche). Sul dataset a grafo fisso denso (10 istanze test): mean speed-up **1.23x → 1.19x**. Confronto istanza per istanza: 3 istanze migliorate, le altre peggiorate o invariate, nessuna regressione catastrofica ma un peggioramento diffuso.

**Interpretazione**: ridurre ulteriormente la loss auto-referenziale (il proxy di training) non garantisce un miglioramento della metrica reale che conta (iterazioni risparmiate in deployment). Le due cose sono correlate ma non perfettamente allineate — continuare ad allenare oltre un certo punto può spostare la policy verso un compromesso diverso, non necessariamente migliore.

---

## Pattern ricorrenti tra i tentativi falliti

1. **Una loss di training più bassa non implica uno speed-up più alto in deployment** (sezioni 1.6, 3.1) — la loss auto-referenziale è un proxy, non la metrica finale.
2. **I vincoli "restrittivi" nella loss spesso fanno un lavoro di stabilizzazione reale**, non solo bloccano comportamenti utili (sezione 1.6, monotonicità) — rimuoverli aumenta la varianza tra istanze più di quanto acceleri la media.
3. **I segnali teoricamente più puliti (a prova di sfruttamento) possono essere segnali di training praticamente più deboli** (sezioni 1.5, 1.6) — d^k è corretto e a prova di congelamento, ma esclude v^k dal gradiente diretto.
4. **Gli screening economici (proxy a basso costo) non sono affidabili per estensioni di orizzonte lungo** (sezione 2.3) — possono dare segnali opposti al risultato reale.
5. **Le estensioni graduali (curriculum) non sono automaticamente più sicure di un salto diretto** (sezione 2.5) — a volte sono peggiori.
6. **Un campione di test piccolo (10 istanze) è rumoroso**: una singola istanza può spostare la media di 0.1-0.15x (sezione 1.6) — verificare le conclusioni importanti su un campione più ampio prima di scartare o accettare un'idea.
7. **"Sommabile" garantisce la convergenza, non la velocità vicino al punto fisso** (sezione 2.9) — un decadimento più lento del geometrico resta teoricamente valido (il teorema di Combettes non ne risente) ma può interferire con la contrazione finale, tipicamente rapida, che la DR non perturbata avrebbe altrimenti — trasformando un'approssimazione veloce in un'agonia lenta proprio vicino al traguardo. Vale per l'intera famiglia (qualunque esponente polinomiale), non solo per il valore provato.
