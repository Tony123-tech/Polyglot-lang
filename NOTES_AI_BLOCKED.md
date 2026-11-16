# Blocked and Skipped Tasks

Tasks that could not be completed within this batch's constraints
(≤ 3 changed files per task, no large type-inference changes) are listed
here with the reason and a suggested follow-up.

## Task 49 — `HashMap::iter_mut`

- **Status:** skipped
- **Reason:** a mutable iterator over map entries requires returning
  borrowed mutable references into the map's slots. The Python backend
  currently emulates `get_mut` with a `_MapSlot` wrapper, and exposing an
  iterator of those wrappers would leak an internal abstraction into
  translated user code. The Rust backend would also need lifetime-aware
  lowering for `IterMut`.
- **Suggested follow-up:** introduce a first-class mutable-slot view used
  by `get_mut`, `iter_mut`, and `values_mut`, then lower iteration over
  that view in both backends in one coordinated change.

## Task 50 — `HashMap::values_mut`

- **Status:** skipped
- **Reason:** same mutable-slot/lifetime problem as `HashMap::iter_mut`.
  It also needs a mutation statement path that writes back through the
  slot wrapper, which is outside the scope of a single small task.
- **Suggested follow-up:** implement together with `HashMap::iter_mut` so
  the slot semantics stay consistent across both backends.

## Notes

- Task 51 (`HashMap::into_keys`) and Task 52 (`HashMap::into_values`)
  shared one commit (`40f1791`) because the two methods have identical
  lowering and test scaffolding.
- All other tasks in the 1–100 batch were completed with a separate
  commit per task (or per tightly coupled method pair, as noted above).
