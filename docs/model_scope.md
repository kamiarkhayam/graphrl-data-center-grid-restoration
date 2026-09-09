# Model scope

## Prescribed anchor zones

The final testbed constructs three synthetic regions from transformed copies of the selected SMART-DS-derived network. Each region receives a prescribed DC overlay and a prescribed set of eligible load blocks. The construction ranks loads by distance and load, then applies the original zone-load selection limits. These sets define the assumed electrically enabled support zones; they are not recomputed through an intertie optimization at each repair decision.

Support is allocated to interrupted loads in those zones, using deterministic critical-first allocation with a fixed within-class priority (`critical_first_fixed_priority`). The fixed priority uses legacy compatibility fields, not the duration-dependent repair objective; see [controller provenance](provenance/controller_trace.md). Loads outside the zones remain ineligible. Connected loads do not consume anchor support. Grid restoration and temporary support both affect the reported service trajectory.

## Power and energy

The constructed anchor export ratings are 4, 5 and 6 MW, for a **15 MW aggregate cap**. The reported controller pools the ratings for allocation across eligible loads. It enforces the aggregate bound rather than separate intertie-flow constraints at each site. The original allocation order and partial-load treatment are preserved.

The no-anchor case has no support. The power-only reference uses the aggregate power limit without cumulative-energy depletion. E12 and E24 set initial export-energy budgets to 12 and 24 hours at the aggregate export rating: 180 and 360 MWh for the reported portfolio. Support consumes energy over each repair interval, including partial-interval depletion. After depletion, finite-energy support is zero.

DC construction tables also carry UPS, backup generation, facility-load and survival metadata. The final export sensitivities use the resource configuration's operative power and energy rules; the ancillary DC metadata are not additional independently dispatched supplies or alternative energy budgets in the reported controller.

## Network and damage representation

Restoration is a connectivity-and-repair model with component-specific repair durations. Supply requires connectivity to the synthetic source, except for prescribed eligible anchor support. The network contains benchmark-derived distribution assets and synthetic/equivalent source and connection edges.

The original hurricane module includes wind exposure and fragility for overhead-distribution proxies, transmission lines and substations, with equivalent source and schematic DC connections excluded. Its default configuration disables underground, transformer and switch damage where that legacy fragility model is absent.

The final benchmark sampler separately selects repairable hurricane-eligible benchmark edges, including overhead-line, transformer and switch classes, using the copied spatial, component and length weights. Repair times use the existing bounded stochastic rule. This sampler and its scenario parameters are preserved in `testbed/benchmark.py` and `configs/testbed/benchmark.yaml`. Historical-source labels, track/intensity augmentation and stress cases remain separate categories.

## Electrical support-path assumptions

The prescribed zones assume that suitable electrically enabled and islandable support paths and operating arrangements exist and remain available during the event. The DC assets, prescribed support paths and abstract source/backbone links are excluded from the final benchmark damage set. The model does not design detailed intertie power flow, AC voltage/reactive-power operation, protection coordination, synchronization, line thermal constraints, or site-specific reverse-power controls. Enablement cost and community-incidence calculations operate on the stated restoration abstraction and source-based accounting inputs.
