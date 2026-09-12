---
name: git-pull-main
description: Pull the latest code from the main branch of the upstream repository. Handles fetching, stash management, branch switching, and merging with conflict detection.
---

# Git Pull Main — Skill Instructions

## Purpose
This skill automates pulling the latest code from the `main` branch. It safely handles uncommitted changes, fetches the latest remote state, and merges updates into the current working branch.

## Workflow

Follow these steps **in order**. Run each git command using the `run_command` tool with the workspace root as the working directory.

### Step 1: Pre-flight Checks

1. **Check current branch and status:**
   ```bash
   git status --short --branch
   ```
   - Note the current branch name (e.g., `anil`, `main`, `feature-x`).
   - Note if there are any uncommitted changes (modified, untracked, or staged files).

2. **If there are uncommitted changes**, stash them:
   ```bash
   git stash push -m "auto-stash before pulling main"
   ```
   - Remember that a stash was created so it can be restored later.

### Step 2: Fetch Latest from Remote

```bash
git fetch origin main
```
- This updates `origin/main` without modifying the working tree.
- If this fails (e.g., network error, auth issue), **stop and report the error** to the user.

### Step 3: Merge or Rebase Main into Current Branch

**Option A — If the current branch IS `main`:**
```bash
git pull origin main
```

**Option B — If the current branch is NOT `main`:**
Merge the latest `main` into the current branch:
```bash
git merge origin/main --no-edit
```

### Step 4: Conflict Detection

After the merge/pull, check for conflicts:
```bash
git status --short
```
- If there are files prefixed with `UU`, `AA`, `DU`, etc., **merge conflicts exist**.
- **Report the conflicted files** to the user and ask how they want to resolve them.
- **Do NOT auto-resolve conflicts.**

### Step 5: Restore Stashed Changes

If changes were stashed in Step 1:
```bash
git stash pop
```
- If this causes conflicts, report them to the user.

### Step 6: Summary Report

After all steps complete, provide a summary:
- Previous branch and whether it changed
- Number of new commits pulled
- Any files that were updated
- Any conflicts or warnings
- Whether stashed changes were restored

Run this to show the latest commits that were pulled:
```bash
git log --oneline -10
```

## Error Handling

| Scenario | Action |
|---|---|
| Network error on fetch | Report error, suggest checking internet/VPN |
| Authentication failure | Report error, suggest checking credentials or SSH keys |
| Merge conflicts | List conflicted files, ask user for resolution strategy |
| Stash pop conflicts | List conflicted files, inform user their local changes conflict with new code |
| Dirty submodules | Run `git submodule update --init --recursive` if needed |

## Safety Rules

- **Never force-push** (`git push --force`) during this workflow.
- **Never hard-reset** (`git reset --hard`) without explicit user approval.
- **Always stash** uncommitted work before switching or merging.
- **Never auto-resolve** merge conflicts — always ask the user.
