# Commit Rules (Mandatory)

These rules are mandatory for any AI agent or tool (including OpenCode)
working in this repository. If an agent cannot follow them, it must not
be used on this repository.

## 1. One commit = one logical change

- Do not accumulate unrelated changes into a single commit.
- A commit message must be summarizable in one sentence.
- If a change touches multiple subsystems, split it by subsystem:
  frontend, Python backend, Rust backend, API/CLI, tests, docs, build/CI.

## 2. Never use `git add -A` or `git add .`

- Stage files explicitly by path or by subsystem directory.
- Do not stage unrelated working-tree changes.
- Do not stage untracked files unless they are part of the current task.

## 3. Test before every commit

- Run the relevant tests for the subsystem being committed.
- Before the final commit of a task, run the full suite:
  - `pytest -q`
  - `pytest -q -m slow` when slow tests are affected
- Do not commit with failing tests.
- If a test is expected to fail (e.g. pins a known limitation), the
  commit message must say so explicitly.

## 4. Commit message format

Use Conventional Commits:

```
type(scope): summary

- bullet details
```

Allowed types: `feat`, `fix`, `refactor`, `test`, `docs`, `build`,
`chore`, `perf`, `ci`.

Scope is the subsystem: `frontend`, `python`, `rust`, `api`, `cli`,
`tests`, `docs`, `build`, `ci`.

## 5. Report after each commit

After every commit, report:

- `git --no-pager log --oneline -1`
- `git status --short`
- test result summary for the committed scope

Do not continue to the next subsystem until the current commit is
reported and the working tree is clean for that scope.

## 6. Do not rewrite shared history

- Do not use `git rebase`, `git reset --hard`, or `git commit --amend`
  on commits already pushed to `origin/main`.
- If a commit must be split, do it before pushing.
- Force-push is forbidden unless the user explicitly requests it.

## 7. Do not mix generated or unrelated files

- Lockfiles (`uv.lock`), CI config, and packaging changes go in their
  own `build`/`ci` commit.
- Untracked scratch files (e.g. `=`, `typescript`, editor backups) must
  be removed, not committed.

## 8. If rules cannot be followed

Stop and ask the user. Do not silently fall back to a single large
commit.
