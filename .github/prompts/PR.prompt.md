# Pull Request Review & Testing

You are a senior software engineer responsible for reviewing a pull request. Your task is to thoroughly analyze the PR, test the changes locally, and provide a detailed report.

## 1. Understand the Pull Request

Use the GitHub MCP server to get the details of the pull request.
- **PR Number:** [Get PR number from user or context]
- **Title:** [Get PR title]
- **Description:** [Get PR description]
- **Branch:** [Get PR branch name]
- **Files Changed:** [List files changed in the PR]

## 2. Prepare Local Environment

1.  Fetch the latest changes from the remote repository.
2.  Check out the pull request's branch to your local environment.
3.  Install any new dependencies if `package.json`, `pyproject.toml`, `go.mod`, etc. have changed.

## 3. Test the Changes

Execute a comprehensive testing process:

1.  **Run Automated Tests:** Execute the entire test suite (unit, integration, e2e) and report the results.
2.  **Perform Manual Testing:**
    -   Based on the PR description and files changed, identify the key user flows and features affected.
    -   Manually test these flows to ensure they work as expected.
    -   Test for edge cases and potential regressions.

## 4. Create a Review Report

Summarize your findings in a detailed report using the following structure.

### PR Review Report

**PR:** #[PR Number]: [PR Title]

**Summary of Changes:**
- [Provide a high-level summary of the changes introduced in this PR.]

**Testing Results:**

*   **Automated Tests:**
    -   **Status:** [Pass/Fail]
    -   **Output:**
      ```
      [Paste relevant test runner output here]
      ```

*   **Manual Testing:**
    -   **Scenarios Tested:**
        -   [Scenario 1: Step-by-step description] -> **Result:** [Pass/Fail/Notes]
        -   [Scenario 2: Step-by-step description] -> **Result:** [Pass/Fail/Notes]

**Bugs or Issues Found:**
-   **Issue 1:** [Description of the issue, including how to reproduce it.]
-   **Issue 2:** [Description of the issue, including how to reproduce it.]

**Suggestions for Improvement:**
-   [Any suggestions for code quality, performance, or other improvements.]

**Conclusion:**
- [Provide a final recommendation: Approve, Request Changes, or Comment.]