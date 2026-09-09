# Data and source provenance

The original research workspace is the authoritative archive. This is a curated research-software release derived from the archived implementation. Source files were hashed before copying, destination edits were confined to the release folder, and original source hashes were checked afterward. Original source hashes are preserved across refinements; destination hashes describe the curated files. [The manifest](provenance/source_file_manifest.csv) maps relative archive paths to public destinations and records SHA-256 hashes, roles and copy/extraction notes.

The final method lineage was identified from the matched source-based policy-training configuration, selected-policy paper refresh, locked GA implementation, final benefit-cost figure implementation, and final composite figure code. Experiment outputs were read only to resolve this lineage and validate external accounting schemas. The allowed accounting comparison wrote temporary ledgers only inside the release folder. None were copied into the repository.

The testbed source configuration identifies Texas7k transmission/mapping inputs and SMART-DS Full Texas OpenDSS/profile inputs. The selected final benchmark uses the locally parsed P49U-derived composite under three synthetic substations, with explicit coordinate/load transformations. The construction code, configuration and geographic assumptions are preserved; this does not establish real-utility geographic calibration.

The original hurricane module consumes previously filtered HURDAT2-derived tracks and mainland geometry. The final benchmark's hurricane-inspired spatial damage sampling is separately preserved and documented in [model scope](model_scope.md). The processed tracks and original preprocessing assets are external prerequisites; their source metadata must accompany any future data bundle.

Economic source URLs are copied from the source-bearing YAML configurations. They cover duration-dependent interruption valuations, price-year conversion, critical-service assumptions, capital-enablement analogs, generator resource/externality costs and host-community accounting. The package does not assign new provenance, reinterpret a published source as a local estimate, or grant redistribution permission for source material.

The six-node software fixture was created for this package. Its topology, coordinates, load IDs and damage list are invented and independent of the paper dataset. Its deliberately oversized disconnected loads exercise the published aggregate-cap contract using a very small graph.

Intentional publication refinements correct the host-community climate boundary, make benefit-cost configuration authoritative, add validated post-processing, expose the preserved allocation as an explicit compatibility mode, update figure export interfaces, and complete the supplied citation metadata. The changes are catalogued in [release refinements](provenance/release_refinements.md). They are not represented as exclusively formatting or import changes.

Inherited Texas7k, Full Texas, Gulf Coast, Houston, Harris County and Galveston labels in acquisition configurations or archived identifiers describe source lineage and preprocessing. They do not define the study case identity; the study is a benchmark-derived synthetic multi-region distribution-system testbed. Archived identifiers and acquisition coordinates remain intact for bundle compatibility.

The source-path column in the provenance manifest necessarily preserves archive filenames. These are traceability identifiers, not supported public experiment names. Public method exposure is limited to the eight entries in the method registry.
