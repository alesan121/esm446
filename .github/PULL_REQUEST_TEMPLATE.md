# 📝 Pull Request Description

## 🔗 Related Issue
Fixes #

## ⚙️ Types of Changes
- [ ] 🐛 **Bug fix** (non-breaking change which fixes an issue)
- [ ] ✨ **New feature** (non-breaking change which adds functionality)
- [ ] 💥 **Breaking change** (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 **Documentation** (update to documentation, README, or code comments)
- [ ] 🔧 **Build/Chore** (updates to Poetry, dependencies, CI/CD, or tool configuration)

## 🧪 Testing & Evidence
- [ ] **Unit Tests**: Ran `poetry run pytest`.
- [ ] **Coverage**: Total coverage stays at or above 80%.
- [ ] **Manual Testing**: Verified functionality in the local environment.

**Environment Details:**
- **OS:** Windows / Linux / MacOS
- **Python Version:** 3.13.x
- **Key Libraries:** (e.g., numpy 2.2, scipy 1.15)

## 📡 Signal Processing Impact
Complete this section for any change under `esm446/core/`.

- [ ] **Real-time budget**: `poetry run esm446-bench` still meets the budget. Measured
      ratio: `___` CPU-seconds per signal second.
- [ ] **Numerical behaviour**: Any change to filter design, detection thresholds or power
      scaling is accompanied by a test asserting the new numbers, not just that it runs.
- [ ] **Calibration validity**: If receiver gain handling or power scaling changed, state
      whether existing calibration files remain valid.

## ⚠️ Breaking Changes
---
