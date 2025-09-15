# PDF Extraction via Playwright MCP Server

## When to Use
- When you need to extract all text content from a PDF document reachable by URL or already loaded in the browser.
- Ideal for automating syllabus parsing, course plan extraction, or any academic PDF content ingestion.

## Prompt Instructions for Copilot Agent

1. **Ensure the PDF Extraction Helper is Available**
   - The project includes a helper module at `.shared-resources/.tools/pdf-extraction-utils.js` exporting `extractPdfText`.
   - The helper uses PDF.js (loaded via CDN) under the hood and requires `--caps=pdf` in `mcp.json`.

2. **Invoke `extractPdfText`**
   - If you have a Playwright `page` instance already on the PDF URL, pass it directly:

    ```js
    import { extractPdfText } from '.shared-resources/.tools/pdf-extraction-utils.js';
    const text = await extractPdfText(page);
    console.log(text);
    ```

   - If you only have a URL, open a new page and pass the URL string:

    ```js
    import { extractPdfText } from '.shared-resources/.tools/pdf-extraction-utils.js';
    const text = await extractPdfText('https://example.com/my.pdf');
    console.log(text);
    ```

3. **Fallback: Direct MCP JNI or Evaluate**
   If the helper is unavailable, you can fall back to injecting PDF.js via the MCP server’s `mcp_playwright-pd_browser_evaluate` tool:

   ```json
   {
     "tool": "mcp_playwright-pd_browser_evaluate",
     "element": "PDF page",
     "function": "async () => {\n      const script = document.createElement('script');\n      script.src = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js';\n      document.body.appendChild(script);\n      await new Promise(r => script.onload = r);\n      window['pdfjs-dist/build/pdf'].GlobalWorkerOptions.workerSrc = 
        'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js';\n      const loadingTask = window['pdfjs-dist/build/pdf'].getDocument(window.location.href);\n      const pdf = await loadingTask.promise;\n      let text = '';\n      for (let i = 1; i <= pdf.numPages; i++) {\n        const p = await pdf.getPage(i);\n        const content = await p.getTextContent();\n        text += content.items.map(item => item.str).join(' ') + '\n';\n      }\n      return text;\n    }"
   }
   ```

4. **Return Results**
   - Ensure the extracted text string is returned or logged.
   - For large PDFs, handle errors and consider streaming page by page.

---

**Use this prompt whenever you need reliable, automated text extraction from PDFs in your Playwright MCP automation workflows.**
