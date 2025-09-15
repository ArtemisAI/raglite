# GitHub Repository Environment Configuration for Raglite

## GitHub Copilot Environment Setup

### 1. Repository Settings Configuration

#### Branch Protection Rules
Navigate to: **Settings → Branches → Add rule**

**Branch name pattern**: `main`

**Require pull request reviews before merging**
- [x] Require approvals (1 minimum)
- [x] Dismiss stale pull request approvals when new commits are pushed

**Require status checks to pass before merging**
- [x] Require branches to be up to date before merging
- [x] Status checks: `test`, `lint`, `security`

**Include administrators**
- [x] Include administrators in these restrictions

#### Repository Features
Navigate to: **Settings → General**

**Features**
- [x] Issues
- [x] Discussions
- [x] Projects
- [x] Wiki
- [x] Sponsorships

**Pull Requests**
- [x] Allow merge commits
- [x] Allow squash merging
- [x] Allow rebase merging
- [x] Automatically delete head branches

### 2. GitHub Environments Setup

#### Create Copilot Environment
Navigate to: **Settings → Environments → New environment**

**Environment name**: `copilot`

**Environment secrets**
```
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=sqlite:///data/raglite.db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Environment variables**
```
NODE_ENV=development
PYTHONPATH=/workspace/src
LOG_LEVEL=INFO
```

**Deployment protection rules**
- [x] Required reviewers (add repository maintainers)

### 3. GitHub Actions Configuration

#### Required Permissions
Navigate to: **Settings → Actions → General**

**Actions permissions**
- [x] Allow all actions and reusable workflows

**Workflow permissions**
- [x] Read and write permissions
- [x] Allow GitHub Actions to create and approve pull requests

### 4. Security Settings

#### Code Security
Navigate to: **Settings → Code security**

**Dependabot**
- [x] Dependabot alerts
- [x] Dependabot security updates

**Code scanning**
- [x] CodeQL analysis (GitHub's default)
- [x] Third-party analysis

#### Secrets Management
Navigate to: **Settings → Secrets and variables → Actions**

**Repository secrets**
```
CODECOV_TOKEN=your_codecov_token
PYPI_API_TOKEN=your_pypi_token
DOCKER_HUB_TOKEN=your_docker_token
```

### 5. Branch Management

#### Default Branch
Navigate to: **Settings → Branches**

**Default branch**: `main`

#### Branch Naming Convention
Create `.github/branch-naming.md`:
```markdown
# Branch Naming Convention

## Format
`<type>/<description>`

## Types
- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Critical fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates
- `test/` - Testing improvements

## Examples
- `feature/add-vector-search-optimization`
- `bugfix/fix-memory-leak-in-embeddings`
- `docs/update-api-documentation`
```

### 6. Issue Templates

#### Create Issue Templates
Navigate to: **Settings → General → Features**

Create `.github/ISSUE_TEMPLATE/` directory with:

**bug_report.md**
```markdown
---
name: Bug Report
about: Report a bug in Raglite
title: "[BUG] "
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment**
- OS: [e.g., Windows 11]
- Python Version: [e.g., 3.11.0]
- Raglite Version: [e.g., 0.1.0]

**Additional context**
Add any other context about the problem here.
```

**feature_request.md**
```markdown
---
name: Feature Request
about: Suggest a new feature for Raglite
title: "[FEATURE] "
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

### 7. Pull Request Template

#### Create PR Template
Create `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Description
Brief description of the changes made in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works

## Performance Impact
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance slightly degraded (explain why)

## Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes (explain what breaks and why)

## Related Issues
Fixes # (issue number)

## Screenshots (if applicable)
Add screenshots to help explain your changes.
```

### 8. Code Owners

#### Create CODEOWNERS File
Create `.github/CODEOWNERS`:

```
# Core maintainers
* @ArtemisAI/maintainers

# Database module
src/raglite/_database.py @ArtemisAI/database-team

# Search module
src/raglite/_search.py @ArtemisAI/search-team

# Documentation
docs/ @ArtemisAI/docs-team
*.md @ArtemisAI/docs-team
```

### 9. Repository Labels

#### Standard Labels to Create
Navigate to: **Issues → Labels**

Create the following labels:
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation updates
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `question` - Further information is requested
- `wontfix` - This will not be worked on
- `duplicate` - This issue or pull request already exists

### 10. Repository Topics

#### Add Repository Topics
Navigate to: **Settings → General → Topics**

Add topics:
- `python`
- `rag`
- `retrieval-augmented-generation`
- `embeddings`
- `vector-database`
- `machine-learning`
- `nlp`
- `ai`

### 11. Repository Description

#### Update Repository Description
Navigate to: **Settings → General → Description**

```
Raglite: High-performance Python library for Retrieval-Augmented Generation (RAG) systems with efficient text processing, vector databases, and semantic search capabilities.
```

### 12. Repository Website

#### Set Repository Website
Navigate to: **Settings → Pages**

**Source**: Deploy from a branch
**Branch**: `main` / `docs`

### 13. Verification Commands

#### Environment Validation
```bash
# Check repository settings
gh repo view ArtemisAI/raglite --json name,description,topics

# Check branch protection
gh api repos/ArtemisAI/raglite/branches/main/protection

# Check environments
gh api repos/ArtemisAI/raglite/environments

# Check secrets (requires admin access)
gh secret list

# Check workflows
gh workflow list
```

#### Local Validation
```bash
# Validate GitHub CLI installation
gh --version

# Authenticate with GitHub
gh auth login

# Test repository access
gh repo clone ArtemisAI/raglite

# Check local configuration
git config --list --local
```

This comprehensive setup ensures that GitHub Copilot has all the necessary context, permissions, and automation to work effectively with the Raglite repository.
