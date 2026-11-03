# AAMAD: Annotation-Aligned Multi-Agent Debate for Scientific Paper-to-Dataset Linking

> [!IMPORTANT]
> **All six core agent prompts are displayed in full below.**
> The text is exported directly from the runtime Python classes, so the prompts
> shown here match those used by the implementation.

**Quick navigation:** [Agent prompts](#agent-prompts) ·
[Method correspondence](#method-correspondence) · [Setup](#setup) ·
[Data](#data) · [Run and evaluate](#run-and-evaluate) ·
[Tests and prompt synchronization](#tests-and-prompt-synchronization)

<!-- BEGIN GENERATED AGENT PROMPTS -->
## Agent prompts

The six core AAMAD roles are shown in full below. Each block is the exact
runtime `system` message, including shared instructions, exported from the
linked Python class. These blocks are not abbreviated examples.

| Stage | Agent | Task | Runtime source |
| --- | --- | --- | --- |
| Discovery | [DOIExpert](#doiexpert) | Dataset DOI references | [agents/experts.py](agents/experts.py) |
| Discovery | [AccessionExpert](#accessionexpert) | Database accession references | [agents/experts.py](agents/experts.py) |
| Discovery | [MetadataExpert](#metadataexpert) | Registered article-dataset links | [agents/experts.py](agents/experts.py) |
| Validation | [Prosecutor](#prosecutor) | Examine disqualifying red flags | [agents/emit_debate.py](agents/emit_debate.py) |
| Validation | [Defender](#defender) | Respond to the prosecution | [agents/emit_debate.py](agents/emit_debate.py) |
| Validation | [Judge](#judge) | Arbitrate while protecting recall | [agents/emit_debate.py](agents/emit_debate.py) |

Per-request `user` messages contain the article and candidate evidence.
Discovery calls receive candidate batches. The Defender also receives the
complete prosecution, and the Judge receives both arguments. The linked
source files define these JSON payloads.

### DOIExpert

```text
Discover paper-to-dataset references with high recall.
Data produced, analyzed or reused by the article counts, including database
records that also denote compounds, protein structures or other scientific entities.
Return identifiers only from the supplied candidate list. Copy supporting evidence
verbatim from the supplied context. Do not invent identifiers or quotations.
Reject only clearly established non-data objects or context-free false identifiers.
Multiplicity or being a database accession alone is not a rejection reason.
Do not assign Primary/Secondary labels or take part in validation voting.
For every candidate return a record in mentions with dataset_id, verdict (EMIT or
SUPPRESS), evidence, reason and confidence. Keep uncertain candidates for validation.

Find repository dataset DOI references. Use supplied DataCite resource types and
repository information to distinguish datasets from articles, software, images and
supplementary documents. Mixed repositories can contain valid research data.
```

### AccessionExpert

```text
Discover paper-to-dataset references with high recall.
Data produced, analyzed or reused by the article counts, including database
records that also denote compounds, protein structures or other scientific entities.
Return identifiers only from the supplied candidate list. Copy supporting evidence
verbatim from the supplied context. Do not invent identifiers or quotations.
Reject only clearly established non-data objects or context-free false identifiers.
Multiplicity or being a database accession alone is not a rejection reason.
Do not assign Primary/Secondary labels or take part in validation voting.
For every candidate return a record in mentions with dataset_id, verdict (EMIT or
SUPPRESS), evidence, reason and confidence. Keep uncertain candidates for validation.

Find database accession references. Distinguish accessions from genes, sample
labels, figure labels, primers and catalog numbers using their local context.
PDB and ChEMBL records are valid when they supply study data. Keep a batch of
runs/samples when it is explicitly deposited or reused as study data.
```

### MetadataExpert

```text
Discover paper-to-dataset references with high recall.
Data produced, analyzed or reused by the article counts, including database
records that also denote compounds, protein structures or other scientific entities.
Return identifiers only from the supplied candidate list. Copy supporting evidence
verbatim from the supplied context. Do not invent identifiers or quotations.
Reject only clearly established non-data objects or context-free false identifiers.
Multiplicity or being a database accession alone is not a rejection reason.
Do not assign Primary/Secondary labels or take part in validation voting.
For every candidate return a record in mentions with dataset_id, verdict (EMIT or
SUPPRESS), evidence, reason and confidence. Keep uncertain candidates for validation.

Find dataset DOIs registered as related to the exact current article in DataCite.
Preserve relation endpoints and direction. Dataset links absent from article text
can be valid. A shared author, a shared repository, or a version/collection link
alone does not establish citation or provenance. Use metadata and textual anchors
when the registered relationship is ambiguous.
```

### Prosecutor

```text
Annotation standard: a resource supplies valid data if the
article produced, analyzed or reused its data. Database accessions are not invalid
merely because they denote compounds, structures, proteins or other entities.
A large number of accessions alone is not a disqualifying red flag.
Disqualifying red_flag_type values: publication, software, non_data_object,
context_free_identifier. Use none when no such flag is established.
All evidence must be verbatim from the supplied local context or metadata.

Examine the candidate for a concrete disqualifying red flag. Be adversarial but
concede EMIT if no red flag is supported. confidence always measures confidence
in SUPPRESSION, including when conceding EMIT (then it should be low).
Return JSON: verdict (EMIT or SUPPRESS), red_flag_type, red_flag (description or
none), evidence (verbatim supporting quote), confidence (0 to 1).
```

### Defender

```text
Annotation standard: a resource supplies valid data if the
article produced, analyzed or reused its data. Database accessions are not invalid
merely because they denote compounds, structures, proteins or other entities.
A large number of accessions alone is not a disqualifying red flag.
Disqualifying red_flag_type values: publication, software, non_data_object,
context_free_identifier. Use none when no such flag is established.
All evidence must be verbatim from the supplied local context or metadata.

Read the same candidate/evidence AND the Prosecutor's complete argument.
Rebut unsupported exclusions using candidate-specific data-generation/reuse
evidence. Concede when an actual non-data object is established.
Return JSON: verdict (EMIT or SUPPRESS), rebuttal, evidence (verbatim), confidence.
```

### Judge

```text
Annotation standard: a resource supplies valid data if the
article produced, analyzed or reused its data. Database accessions are not invalid
merely because they denote compounds, structures, proteins or other entities.
A large number of accessions alone is not a disqualifying red flag.
Disqualifying red_flag_type values: publication, software, non_data_object,
context_free_identifier. Use none when no such flag is established.
All evidence must be verbatim from the supplied local context or metadata.

Review the candidate, pooled discovery evidence and BOTH arguments. Prioritize
verbatim evidence of study data generation or reuse. Suppression requires an
explicit, supported disqualifying red flag with suppression confidence STRICTLY
ABOVE 0.85. Otherwise EMIT to protect recall. A confidence of exactly 0.85 does
not qualify. Do not use a confidence gap or the largest confidence as a vote.
Uphold a Prosecutor red flag only if its quote really establishes an exclusion,
after considering the Defender's rebuttal.
Return JSON: decision (EMIT or SUPPRESS), red_flag_supported (boolean), reason,
confidence (confidence in the final decision, NOT the red-flag threshold).
```

Baseline and Conventional Debate prompts are also available in
[PROMPTS.md](PROMPTS.md). To change a prompt, edit its Python source and run
`python -B scripts/export_prompts.py` to refresh both documents.

<!-- END GENERATED AGENT PROMPTS -->

## Release scope

The supplied source has been revised to follow the manuscript's methods. The
original predictions, gold annotations, model-response caches and full-text corpus
are not bundled. The offline tests validate behavior using synthetic fixtures.
They do not reproduce or validate the paper's numerical results. Rerun inference
and evaluation on the original cohort before claiming numerical reproduction.

## Method correspondence

| Manuscript | Implementation |
| --- | --- |
| 3.3 multi-source discovery | DOIExpert, AccessionExpert and MetadataExpert contribute candidate IDs, evidence and confidence. Duplicate IDs merge evidence and retain maximum discovery confidence. |
| 3.4 selective admission | A distinctive dataset namespace AND explicit candidate-specific data-use evidence bypass validation. Other candidates receive one Prosecutor response, one Defender response with the prosecution, then one Judge decision. |
| 3.4 arbitration | SUPPRESS requires an explicit disqualifying red flag, a verbatim supporting quote, suppression confidence strictly above 0.85, and Judge confirmation after considering the defense. All other cases EMIT. |
| 3.5 provenance | The first decisive source wins: exact, directed article-dataset relation, candidate-local text, then repository prior. Dataset DOI defaults to Primary and database accession to Secondary only as a last resort. |
| 4.1 MDC metrics | Only supplied annotated article-identifier pairs are scored. Missing/NonReference with a resolvable identifier is a negative. Missing without a resolvable identifier is excluded. Unlabeled predictions are counted separately and excluded from MDC metrics. |
| 4.1 Data Gatherer metrics | Compute precision and recall per paper, average them arithmetically, then compute F1 from those two averages. Papers without gold references are excluded from recall averaging. |
| 4.5 cost | Each output includes measured model requests, cache hits, token usage and runtime per article, together with mean, sample SD, minimum and maximum. |

Discovery agents never vote in validation. The discovery-only ablation removes
validation but keeps provenance classification. Conventional Debate uses the same
discovery pool, routing, model and temperature, with generic adversarial prompts
in place of annotation-aligned prompts and its suppression gate.

## Setup

Python 3.10 or later is required. From this directory:

```sh
python -m pip install -r requirements.txt
export DEEPSEEK_API_KEY='your-key'
export DEEPSEEK_MODEL='deepseek-v4-pro'
export AAMAD_DATA_DIR='/path/to/data'
```

The default model name reflects the manuscript. Set DEEPSEEK_MODEL to the exact
deployment identifier serving the intended checkpoint. Model availability and
live provider responses have not been validated by the offline tests.
All model-based variants default to temperature 0.2. Credentials are read only
from environment variables. The example environment file is a reference, not
automatically loaded by Python.

## Data

The included paper_ids.txt contains 400 distinct MDC article identifiers. Supply
the original labels and full texts in the following layout:

```text
data/
  train/
    XML/<article_id>.xml
    TXT/<article_id>.txt
    PDF/<article_id>.pdf
  train_labels.csv
```

Only one full-text format is necessary. TEI or JATS XML is preferred, followed by
plain text and PDF. DOI slashes are replaced by underscores in file names, as in
the included article list. Article DOI URLs and these file-name forms normalize
to the same evaluation key. For PMC-named XML files, article DOI and surnames are
read from the article header, not from its reference list.

MDC labels must include article_id,dataset_id,type. Positive types are Primary
or Secondary. Missing or NonReference with a resolvable ID denotes an annotated
negative. Alternatively, use an explicit is_reference column (true/false), with
a provenance type for positive pairs. Duplicate conflicting labels raise an error.
Predictions are never used to decide which pairs are annotated.

For Data Gatherer, supply the actual evaluated article list and citation records
as a separate labels CSV. Positive references may omit type because only
discovery is evaluated. DataRef-EXP's published set contains 21 articles and 48
references. DataRef-REV's published evaluation subset contains 1,242 articles and
1,883 citation records. Those data files and article lists are not bundled here,
and their sizes are never hard-coded into scoring. See the
[Data Gatherer paper](https://aclanthology.org/2025.sdp-1.10.pdf).

## Run and evaluate

```sh
python run.py --labels /path/to/train_labels.csv --mode AAMAD --output outputs/AAMAD
python run_ablation.py --labels /path/to/train_labels.csv --output outputs/ablations
python run.py --articles /path/to/dataref_ids.txt --labels /path/to/dataref.csv --evaluation dataref --output outputs/dataref
```

Available modes: AAMAD, discovery_only, conventional_debate, rule_based,
metadata_only, single_agent and bare_llm. The tool-augmented single agent uses
the same identifier/registry tools and deterministic provenance classifier without
specialized discovery roles or debate. Bare LLM performs one article-text model
pass without those tools and predicts its own provenance labels.

run_ablation.py discovers candidates once per article and reuses that exact pool
for the three validation variants. It does not silently select the first ten
articles or assign all no-validation predictions to Primary. shared_pools.json
records the pool IDs. Each variant writes predictions.jsonl and report.json,
including the actual article cohort, label-file hash, traces and evaluated counts.

To score existing predictions without model or registry calls:

```sh
python run.py --articles paper_ids.txt --labels /path/to/train_labels.csv --score-only /path/to/predictions.jsonl
```

Missing full text, invalid model JSON, conflicting annotations or unmatched
evaluation cohorts raise errors rather than silently excluding failed articles.

## Important implementation details

- Identifier spelling normalizes DOI URLs/case/punctuation and accession case. Distinct DOI version suffixes and numeric suffixes remain distinct identifiers.
- Input and output relations are resolved only after matching both article and dataset endpoints and preserving direction. IsSupplementTo from dataset to article is an operational output signal. Cites/References from article to dataset and inverse citation links are input signals. Version and part/whole links alone do not decide provenance. Opposing decisive relations defer to text.
- Local generation/deposition indicates Primary, and acquisition/reuse/prior publication indicates Secondary. Candidate-bound clauses prevent a nearby dataset's role from labeling another candidate. Mixed cues use article-author surnames, ownership/data-availability markers and external citation context. Unresolved evidence falls through to the repository prior.
- Short/scientific-entity namespaces such as PDB and ChEMBL require validation. Multiplicity alone is not a rejection reason. Unsupported batches of subordinate run/sample records are filtered in discovery using the configured threshold, with explicit data-use/deposition evidence taking precedence.
- The red-flag threshold applies to the Prosecutor's suppression confidence, not the Judge's confidence in its final answer. Exactly 0.85 never qualifies. The Judge must uphold a grounded flag. A higher-confidence defense/prosecution argument alone cannot bypass this gate.
- Text evidence uses a conservative 900 UTF-8-byte cap, below a 900-token budget for byte-level tokenizers. No approximate word count is represented as an exact DeepSeek token count. Metadata evidence is preserved separately.
- DataCite lookups use cached forward and reverse links and paginate results. Failed requests are logged and are not persisted as successful empty responses. Metadata coverage can therefore affect a fresh run.

## Cost and caches

total_tokens counts actual model usage from live responses. recorded_tokens also
includes historical usage attached to cache hits and is not an additional API
charge. calls counts completed live responses, network_requests includes retries,
and logical_calls includes cache hits. Runtime is measured end to end for the
current run. A cached replay's runtime is not the runtime reported in the paper.
Shared-discovery ablations include the common discovery cost in each variant's
logical per-article accounting; it is executed only once in that comparison.

```sh
export LLM_OFFLINE=1
```

Offline mode forbids model network requests and requires matching caches.
Uncached model calls fail. Uncached DataCite metadata is unavailable in offline
mode. To disable registry lookup explicitly, set DATACITE_ENABLED=0.
Updated prompts and the model/temperature are included in cache keys, so old
responses generated with incompatible prompts are not reused silently.

## Tests and prompt synchronization

```sh
python -B -m unittest discover -s tests -v
python -B scripts/export_prompts.py --check
```

Tests run with mocks and synthetic fixtures, without paid model calls or
downloads. Optional OpenAI/PDF libraries are needed only for live inference or
PDF extraction. The six core prompts are displayed directly in this README.
[PROMPTS.md](PROMPTS.md) additionally includes the baseline and Conventional
Debate prompts. Both documents are exported from the Python classes and checked
for consistency by the command above. Regenerate both after changing a prompt:

```sh
python -B scripts/export_prompts.py
```

See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for concrete operational
rules and the inputs needed for numerical reproduction.
