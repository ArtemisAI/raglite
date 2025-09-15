---
mode: copilot-coding-agent
---
Task: Document and Scaffold New Feature Proposals

Context:
You’ve received a set of high-level feature ideas for this Academic Course Management System. Your goal is to prepare the project for future implementation by:

1. Creating individual GitHub issue templates (without assigning) for each feature idea.
2. Generating corresponding task plans under `.shared-resources/.tasks/` (one task folder per feature) with:
   - A clear objective summary.
   - Detailed requirements and constraints.
   - Success criteria and deliverables.
3. Drafting initial stub documentation in `.shared-resources/.instructions/` for each feature, outlining:
   - Background and motivation.
   - High-level design considerations.
   - Next steps for implementation.

Requirements:
- Do not assign issues or tasks to any user.
- Use descriptive titles and bodies in issues.
- Follow existing file and folder naming conventions.
- Preserve the DRY shared toolbox architecture.

Output:
- New issue title and body text for each feature (Markdown).
- New task plan files under `.shared-resources/.tasks/feature-name-plan.md`.
- Documentation stubs under `.shared-resources/.instructions/feature-name.md`.
- A summary comment listing all created artifacts.
