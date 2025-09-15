---
mode: copilot-coding-agent
---
Task: Tool-Use-PDF-Conversion

Description:
Automate downloading of PDF course materials, converting them to Markdown, cleaning up formatting, and generating summaries. Ensure no duplicate processing by checking against the database.

Requirements:
- Identify and download PDFs using stored URLs or portal checks.
- Convert PDFs to Markdown with `pandoc` or similar tools.
- Clean up Markdown formatting (remove headers/footers, fix headings).
- Summarize key sections into bullet points.
- Store outputs in `.shared-resources/Materials/<CourseName>/`.
- Update SQLite DB to mark processed files.

Deliverables:
- PowerShell script `Convert-PDFsToMd.ps1` in `.shared-resources/.scripts/`.
- Generated Markdown files and summary under `.shared-resources/Materials/`.

Success Criteria:
- Running `Convert-PDFsToMd.ps1` processes all new PDFs and outputs clean Markdown and summaries.
