# Safety risk annotation guidelines

## 1. Purpose

The annotation task identifies safety entities, evidence-grounded relations, and causal risk structures in English railway accident reports and Chinese safety-management documents. Model output is pre-annotation only. Human-reviewed annotations form the gold standard.

## 2. Annotation unit

- Preserve the source document, page or paragraph, segment identifier, and character offsets.
- Annotate the shortest span that expresses the complete entity.
- Keep repeated mentions as separate annotations. Link aliases with `same_as` only when useful.
- Do not annotate facts that are absent from the source, even when they are plausible domain knowledge.

## 3. Entity rules

Use the entity types in `configs/risk_ontology.yaml`.

The frozen ontology version is `1.0.0`; annotation records continue to use schema version `0.1.0`. Ontology and annotation-schema versions are tracked separately.

- `EVENT` is an occurrence; `HAZARD` is a potentially harmful condition.
- `ASSET` is the physical or technical object; `FAILURE` is its failed state or function.
- `BARRIER` is an existing prevention, detection, protection, or recovery control.
- `MITIGATION` is a proposed, corrective, or newly introduced action.
- `HUMAN_FACTOR` describes individual performance; `ORGANIZATIONAL_FACTOR` describes management or system conditions.
- `REGULATION` must identify a rule, standard, procedure, or provision, not a vague reference to safety.

## 4. Relation rules

- Add a relation only when at least one evidence span supports it.
- Use `causes` only for direct causal claims. Use `contributes_to` for indirect, underlying, or partial influence.
- Use `precedes` for temporal order without asserting causation.
- Use `results_in` from an event or failure to its consequence.
- Use `normative` when the relation comes from a plan, regulation, or recommendation rather than an observed accident fact.
- Mark multi-sentence deductions as `inferred`; do not present them as explicit findings.
- Mark ambiguous or conflicting claims as `uncertain` and send them to adjudication.

## 5. Pre-annotation and review

1. A teacher model generates candidate entities and relations in the project JSON format.
2. A second model or deterministic validator detects omissions, duplicate spans, invalid identifiers, and illegal relation signatures.
3. A human reviewer accepts, rejects, or modifies every candidate.
4. At least 20% of gold documents receive independent second review.
5. Disagreements are adjudicated without using the teacher model as the final judge.

Deterministic repair is limited to auditable cases: an invalid entity quote may be rebound only when its text has one occurrence in the supplied source segments, and an illegal relation may be reversed only when the swapped direction satisfies the frozen ontology signature. Missing relation claim status is rejected by default. Repairs remain pending until human review and are never promoted directly to gold.

## 6. Dataset integrity

- Split by source document before creating chunks or pseudo-labels.
- Keep duplicate and near-duplicate templates in the same split.
- Do not use teacher-generated annotations as unreviewed test labels.
- Record model identifier, prompt version, generation settings, and review status.
- Keep the final test set frozen and hidden from training and prompt development.
