---
applyTo: "**"
---

# Faceless Semantic Version Bump Rule

When asked to bump or recommend a version bump, follow this process.

## Source of truth

- Package version: `pyproject.toml` -> `[project].version`
- Change intent: `CHANGELOG.md` -> `## [Unreleased]`

## Use the version script

Use `scripts/Invoke-Bump-Faceless-Version.ps1`.

### Recommend bump only

`pwsh -NoProfile -File scripts/Invoke-Bump-Faceless-Version.ps1 -BumpVersion auto`

### Apply explicit bump

- major: `pwsh -NoProfile -File scripts/Invoke-Bump-Faceless-Version.ps1 -BumpVersion major -Apply`
- minor: `pwsh -NoProfile -File scripts/Invoke-Bump-Faceless-Version.ps1 -BumpVersion minor -Apply`
- patch: `pwsh -NoProfile -File scripts/Invoke-Bump-Faceless-Version.ps1 -BumpVersion patch -Apply`

### Apply recommended bump

`pwsh -NoProfile -File scripts/Invoke-Bump-Faceless-Version.ps1 -BumpVersion auto -Apply`

### Apply bump and create local tag

`pwsh -NoProfile -File scripts/Invoke-Bump-Faceless-Version.ps1 -BumpVersion auto -Apply -CreateTag`

The script prints push commands but does not run them.

## Bump decision policy

- `major`: use for breaking changes or explicit `BREAKING` notes.
- `minor`: use for backward-compatible feature additions/changes.
- `patch`: use for backward-compatible fixes and maintenance.

If uncertain, run `-BumpVersion auto`, report recommendation + rationale from `CHANGELOG.md`, then ask whether to apply.
