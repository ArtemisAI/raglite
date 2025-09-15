---
mode: copilot-coding-agent
---

# PR Iteration and Management Instructions

## Context
When assigned to work on a pull request, you must follow these strict guidelines to avoid branch confusion and duplicate work.

## Core Rules

### 1. **ALWAYS Work on Assigned Branch**
- Check the PR number you've been assigned to work on
- Fetch and checkout the correct branch: `git fetch origin pull/{PR_NUMBER}/head:{BRANCH_NAME}`
- Never create new branches for existing PR work

### 2. **Before Starting Work**
```bash
# Check current branch and PR status
git branch -a
git status
git log --oneline -5

# Verify you're on the correct branch for the assigned PR
```

### 3. **Iteration Workflow**
1. **Read ALL existing PR comments and feedback**
2. **Address each review comment specifically**
3. **Commit changes with descriptive messages**
4. **Push to the same branch**
5. **Add comment explaining what was fixed**

### 4. **Never Create Duplicate PRs**
- If PR already exists for your task, work on that PR
- If unsure, ask in PR comments before creating new branch
- Close any accidental duplicate PRs immediately

## Required Steps for Each Iteration

### Step 1: Assessment
```markdown
## Iteration Assessment
- [ ] Read all existing review comments
- [ ] Identify specific issues to address
- [ ] Check current branch is correct
- [ ] Verify no duplicate PRs exist
```

### Step 2: Implementation
```markdown
## Changes Made
- [ ] Issue 1: [Description of fix]
- [ ] Issue 2: [Description of fix]
- [ ] Testing: [How changes were tested]
```

### Step 3: Communication
```markdown
## Review Response
@{reviewer} I've addressed the following issues:
1. **Issue 1**: Fixed by [specific change]
2. **Issue 2**: Implemented [specific solution]

Ready for re-review.
```

## Error Prevention

### ⚠️ **Common Mistakes to Avoid**
- Creating new PR when existing one should be updated
- Working on wrong branch
- Not reading existing review comments
- Not pushing changes to correct branch
- Creating multiple branches for same feature

### ✅ **Success Indicators**
- Working on branch that matches assigned PR number
- All review comments addressed
- Commit history shows clear progression
- No duplicate PRs exist
- Clear communication in PR comments

## Branch Management

### Current Active PRs (as of Sept 9, 2025):
- **PR #16**: Playwright MCP PDF Capabilities (PRIORITY)
- **Other PRs**: To be audited and potentially closed

### Cleanup Commands:
```bash
# List all branches
git branch -a

# Delete old/stale local branches
git branch -d {branch_name}

# Delete old remote tracking branches
git remote prune origin
```

## Task Completion Criteria

### Before Marking Work Complete:
- [ ] All review comments addressed
- [ ] Tests passing
- [ ] Documentation updated if required
- [ ] No merge conflicts
- [ ] PR description updated if scope changed

### Final Comment Template:
```markdown
## Iteration Complete

### ✅ Issues Addressed:
1. [Issue 1]: [Solution implemented]
2. [Issue 2]: [Solution implemented]

### 🧪 Testing Performed:
- [Test 1]: [Result]
- [Test 2]: [Result]

### 📝 Additional Changes:
- [Any additional improvements made]

Ready for final review and merge consideration.
```

---

**⚠️ CRITICAL: Always verify you're working on the correct PR branch before making any changes. When in doubt, ask in PR comments.**
