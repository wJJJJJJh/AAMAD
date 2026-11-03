# Implementation details

These concrete rules make the manuscript's cascade and routing auditable.

## Discovery and routing

All source roles share tools/candidates.py. Repeated IDs retain their evidence
windows, and merged discovery confidence is the maximum source confidence.
Discovery roles produce no validation votes. Parent dataset namespaces with
explicit data-use evidence bypass debate. PDB, ChEMBL and other ambiguous
scientific-entity namespaces require semantic verification and validation.
Unsupported batches of subordinate run/sample records use the configured
threshold, with explicit data-use or deposition context taking precedence.

Text evidence has a conservative 900 UTF-8-byte budget. This is below 900 tokens
for byte-level tokenizers, rather than an approximate word-count claim about the
provider's exact tokenizer. Metadata and arguments are provided separately.

## Validation

A routed candidate receives one Prosecutor response, one Defender response using
the full prosecution, then one Judge response using both arguments.
The suppression gate uses the Prosecutor's confidence in its concrete red flag.
The quote must be grounded in supplied candidate-local text or metadata and the
Judge must uphold the flag after reviewing the defense. The score must be
strictly greater than 0.85. The Judge's final-answer confidence and a comparison
between the two agents' confidence scores cannot override this requirement.

## Provenance relation direction

Both relation endpoints must match the current article and candidate exactly.
The following operational mapping is used before textual evidence:

| Registered direction | Relation type | Role |
| --- | --- | --- |
| Dataset -> article | IsSupplementTo, IsDerivedFrom | Study output / Primary |
| Article -> dataset | IsSupplementedBy, IsSourceOf | Study output / Primary |
| Dataset -> article | IsCitedBy, IsReferencedBy, IsSourceOf | Study input / Secondary |
| Article -> dataset | Cites, References, IsDerivedFrom | Study input / Secondary |

This is an operational provenance convention. A generic citation or supplement
relationship is not a universal guarantee of data creation. Version, part/whole,
shared-author or repository-membership information alone gives no decision.
Opposing decisive roles fall through to candidate-local text. Text first uses
generation/deposition and acquisition/reuse language, then local authorship,
ownership/data-availability and external-citation cues for mixed signals.
Unresolved cases use Primary for dataset DOI deposits and Secondary for database
accessions. These deterministic labels are not calibrated probabilities.

DataCite relation semantics are documented in
[Connecting to Works](https://support.datacite.org/docs/connecting-to-works).

## Reproduction inputs

The 400 MDC paper identifiers are supplied. The original full texts, 719-row
annotation cohort, Data Gatherer evaluation files and model-response caches are
not bundled. Use the exact original inputs and checkpoint to assess numerical
reproduction. Synthetic tests establish program behavior, not published scores.

The runner records actual evaluated counts, the complete article cohort, the
label-file hash, model, temperature, predictions, decision traces and cost
statistics. It never inserts paper table values into computed metrics.
