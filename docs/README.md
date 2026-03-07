# Docs Index

Last reviewed: 2026-03-07 (`dev`)

This folder contains project standards, status ledgers, and migration/audit notes.

## Purpose Map

- `bad-practices.md`
  - Source of coding standards and anti-pattern rules.
  - Use this as the rule baseline for code review/refactors.

- `technical-debt.md`
  - Canonical debt/status ledger.
  - Lists current debt by area and priority.

- `technical-debt-remediation-plan.md`
  - Prioritized execution plan derived from `technical-debt.md`.
  - Tracks next implementation sequence.

- `goal-audit.md`
  - High-level project-status snapshot (pipelines, SmartDs surface, validation baseline).
  - Does not own implementation sequencing.

- `quicklook-feature-plan.md`
  - Feature migration status for non-3D quicklook workflows.

- `function-audit-notes.md`
  - Current function/class inventory with concise purpose and caller notes.
  - Regenerate after API moves/renames.

- `shim-sized-function-audit.md`
  - Short-function shim-smell audit.
  - Review list only; not an automatic delete list.

- `batplotlib-test-migration-checklist.md`
  - Legacy-to-current test migration coverage mapping.

## Maintenance Rule

No-overlap rule for the three core state docs:

1. `goal-audit.md`: snapshot + verification only
2. `technical-debt.md`: debt ledger only
3. `technical-debt-remediation-plan.md`: ordered action plan only

When architecture or API shape changes in `starwinds_analysis/`, update these docs in this order:

1. `function-audit-notes.md` and `shim-sized-function-audit.md`
2. `technical-debt.md`
3. `technical-debt-remediation-plan.md`
4. `goal-audit.md` and `quicklook-feature-plan.md`
5. `batplotlib-test-migration-checklist.md` (if test coverage mapping changed)
