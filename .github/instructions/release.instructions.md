---
applyTo: "**"
---

# Faceless Release Cut Instructions

Use this workflow whenever asked to create a release for Faceless.

## Required checks before release

1. Confirm `pyproject.toml` has the intended `[project].version`.
2. Confirm `CHANGELOG.md` has relevant entries under `## [Unreleased]`.
3. Build locally if requested:
   - `mise run clean`
   - `mise run build`

## Release process

1. Commit release-related changes to `main`.
2. Create and push a semantic version tag in `vX.Y.Z` form:
   - `git tag v0.1.0`
   - `git push origin v0.1.0`
3. The GitHub Actions `Release` workflow will:
   - build `dist/*.whl` and `dist/*.tar.gz`
   - create/update the GitHub Release for that tag
   - attach built artifacts to the release

## Manual release trigger

Use Actions -> `Release` -> `Run workflow` and provide:
- `tag` (required, must look like `v1.2.3`)
- `title` (optional)
- `prerelease` (optional)

## Version/tag consistency rule

Tag version and `pyproject.toml` version must match:
- tag `v1.2.3` -> version `1.2.3`
