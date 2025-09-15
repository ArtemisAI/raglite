---
mode: copilot-coding-agent
---

# Branch Cleanup and PR Consolidation Instructions

## Context
Clean up stale branches and consolidate duplicate work to maintain clean repository state.

## Current Branch Audit (September 9, 2025)

### Active Branches Status:
```
ACTIVE:
- pr-16: Playwright MCP PDF Capabilities (PRIMARY)
- pr-14: Previous work (needs audit)

STALE/CLEANUP NEEDED:
- pr-16-fresh: Duplicate of pr-16 (DELETE)
- copilot/vscode*: Multiple temp branches (DELETE)
- copilot/implement-*: Old database work (AUDIT/DELETE)
```

## Cleanup Tasks

### 1. **Identify Primary PR**
For each feature area:
- Find the most recent, most complete PR
- Mark as PRIMARY
- Close all duplicates with reference to primary

### 2. **Branch Deletion Strategy**
```bash
# Delete local stale branches
git branch -d pr-16-fresh
git branch -d copilot/vscode1757364557394
git branch -d copilot/vscode1757380747436
# ... (continue for all copilot/vscode* branches)

# Delete remote stale branches (after confirming locally)
git push origin --delete pr-16-fresh
git push origin --delete copilot/vscode1757364557394
# ... (continue for confirmed stale branches)
```

### 3. **PR Consolidation Process**
For duplicate PRs:
1. **Compare the PRs** - identify which has more complete work
2. **Cherry-pick essential changes** from secondary to primary if needed
3. **Close secondary PRs** with comment referencing primary
4. **Update all task references** to point to primary PR

## Specific Actions Required

### PR #16 (Playwright MCP PDF Capabilities)
- **Status**: PRIMARY - Keep open
- **Action**: Continue iteration on this PR
- **Branch**: `pr-16` (keep)

### PR #16-fresh
- **Status**: DUPLICATE
- **Action**: Close and delete branch
- **Reason**: Created accidentally during debugging

### Copilot/* Branches
- **Status**: STALE - created by various copilot sessions
- **Action**: Audit each, close PRs, delete branches
- **Priority**: Clean up to prevent confusion

## Implementation Steps

### Step 1: Audit All Open PRs
```bash
# List all remote branches
git branch -r

# Check each PR status
# (Use GitHub web interface or CLI if available)
```

### Step 2: Close Duplicate/Stale PRs
For each stale PR:
1. Add closing comment explaining reason
2. Reference the primary PR if applicable
3. Close the PR
4. Delete the branch

### Step 3: Update Documentation
- Update task assignments to reference correct PRs
- Update any documentation mentioning old PR numbers
- Ensure .github prompts reference current active PRs

## Template for Closing Stale PRs

```markdown
## 🧹 Closing Stale PR

This PR is being closed as part of repository cleanup:

**Reason**: [Duplicate/Superseded/Abandoned]
**Primary PR**: [Link to main PR if applicable]
**Action Taken**: [Describe any work consolidated elsewhere]

All future work on this feature should continue in the primary PR.

---
*Closed as part of PR management policy enforcement*
*Date: September 9, 2025*
```

## Automation Opportunities

### GitHub Actions for Branch Cleanup
Consider implementing:
- Automatic detection of stale branches
- Notification when multiple PRs exist for same feature
- Automated closing of abandoned PRs after timeout

### VS Code Integration
- Tasks for common branch operations
- Shortcuts for PR status checking
- Integration with GitHub CLI if installed

## Success Criteria

### After Cleanup:
- [ ] Only necessary active PRs remain open
- [ ] Each feature has single primary PR
- [ ] All stale branches deleted
- [ ] Task assignments reference correct PRs
- [ ] Clear branch naming convention established

### Ongoing Maintenance:
- [ ] Weekly branch cleanup routine
- [ ] PR duplication prevention measures
- [ ] Clear guidelines for copilot agents
- [ ] Automated alerts for branch proliferation

---

**Execute this cleanup immediately to prevent further confusion in development workflow.**
