# Methods

## Restoration and candidate screening

The state contains network connectivity, remaining damaged repairable components, prior repairs, load interruptions, prescribed anchor support, and accumulated load-level unserved energy. One crew selects a repair at each decision epoch. The current outage is integrated over that component's repair interval, then the repair changes connectivity.

The source-based screen evaluates immediate marginal interruption-damage reduction for each unrepaired candidate. Its final ranking uses damage reduction, shorter repair duration, and an edge-ID tie break, preserving the original sort order. The first K = 50 entries receive a valid candidate mask; padding cannot be selected. The same candidate IDs, mask and physical features are supplied to each learned encoder. A prescribed controller supplies eligible interrupted loads subject to the aggregate export budget.

## Learned policies

GraphRL-A2C, GraphRL-PPO and GraphBC share an edge-informed message-passing graph policy. Node/edge state and global context combine with candidate repair features. The final graph has hidden dimension 64 and two message-passing layers. GraphBC learns five-repair lookahead targets from the recorded behavior states. Its loss combines classification and a ranking term; validation regret selects the BC checkpoint.

GraphRL-A2C and GraphRL-PPO initialize from that behavior-cloned graph policy. Their copied trainers retain their advantage estimation, rewards, optimizer settings, clipping and schedules. MLP-A2C and CNN-A2C initialize from corresponding behavior-cloned encoders, then use the matched A2C protocol. The MLP hidden dimension is 192. The CNN uses a 32 Ã— 32 coordinate raster, 24 convolution channels, a 96-dimensional spatial embedding, and a 192-dimensional hidden representation.

The RL training budget is 32,768 steps. A2C collects 128-step batches; graph PPO uses 256-step batches, four update epochs and clip range 0.15. The final configuration preserves the remaining learning rates, discount/GAE settings, entropy schedule, reward weights and seed 32. Periodic development evaluations and the copied Greedy-relative checkpoint-selection rules remain part of training. Fresh validation results are outside checkpoint selection.

## Comparison methods

Greedy chooses the leading source-based candidate. Random Top-K chooses uniformly among the valid screened entries using the configured episode seed. Both use the same restoration and support logic.

Bounded GA searches repair permutations with the supplied locked cohort. It seeds the population with the source-based Greedy order, retains elites and mutates repair order through swaps or segment relocation. The final comparison uses population 12 and eight generations. Fitness values are cached for repeated orders. Cohort membership, order and seeds must be preserved; this is a budgeted search baseline without a global-optimality claim.

## Interruption damage and metrics

The economic state accumulates unserved kWh by load. Equivalent interruption duration is accumulated kWh divided by baseline kW. Noncritical residential/commercial blocks use source-based duration-dependent customer-damage functions. Up to 24 hours the implementation uses the configured dollar-per-kWh anchor; longer durations use normalized damage growth with logâ€“log interpolation and the preserved terminal extrapolation rule. Price-year conversion is explicit in the configuration. Critical loads use the cited $100/kWh critical-service assumption from the source configuration.

Both policy evaluation and GA apply this same economic formulation. The package also retains physical ENS, critical ENS, recovery-threshold timing, full-restoration timing, normalized service-loss area, load-based duration indices, and anchor resource use. Recovery service can come from grid reconnection or prescribed anchor support. Aggregate summaries should retain their cohort and denominator definitions.

The benefit-cost implementation pairs resource cases with no-anchor counterfactuals. The lifecycle export-enablement assessment counts incremental capital, export-specific fixed O&M and event resource costs, including event climate damage by default. The host-community incidence assessment counts local fiscal and resilience benefits against community-funded capital/O&M, rate and tax burdens, and local air-quality damage. Routine and event climate damages are reported separately as broader societal externalities. Recurrence intervals, payment shares and cost cases are explicit scenario coordinates. Annualization and event costs must not be counted twice. Source URLs and the original numerical parameters remain in `configs/economics/`.
