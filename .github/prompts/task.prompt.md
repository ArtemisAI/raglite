# Task: Course Material Aggregation and Database Indexing for the "Studies" Project

## Objective
Your task is to use the `playwright-mcp` automation server to access a specified online course from the University of Montreal's educational platform (StudiUM), systematically extract all educational materials, and then populate the `sqlite-studies` database with indexed records for each item in the `content_items` table.

## Context
The goal of this project is to create a centralized and searchable database of all academic materials for my courses at the University of Montreal. This will help with organization, assignment tracking, and efficient study preparation by having all resources indexed in one place.

## Workflow
1.  **Phase 1: Data Extraction via Playwright:** Automate the process of logging into StudiUM, navigating to the specified course, and downloading all available materials.
2.  **Phase 2: Data Processing and Cleaning:** Once the materials are downloaded, process and clean the data. This includes extracting text from PDFs, cleaning up file names, and identifying key metadata.
3.  **Phase 3: Database Indexing via SQLite:** Connect to the `sqlite-studies` MCP server and populate the `content_items` table with the processed data.
4.  **Phase 4: Export to Google Sheets:** Export the indexed data from the database to a Google Sheet for easy access and sharing.

---

## Phase 1: Data Extraction via Playwright
-   **Initiate Connection:** Connect to the `playwright-mcp` server to begin a web automation session.
-   **Authenticate and Navigate:** Log into the educational portal at [https://studium.umontreal.ca/](https://studium.umontreal.ca/) and navigate to the following course:
    -   **Course:** `SIA1010 - Économie, commerce et IA`
-   **Systematic Extraction:** Scrape and download all available materials from the course page. Be sure to look for and acquire the following types of content:
    -   Syllabus
    -   Lecture Notes / Slides (PDF, PPTX, etc.)
    -   Assignment Descriptions and Requirements
    -   Required Readings or Articles
    -   Announcements
    -   Quiz or Exam details

## Phase 2: Data Processing and Cleaning
-   **Text Extraction:** Use tools like `pdftotext` or other libraries to extract the text content from downloaded PDF files.
-   **File Naming Convention:** Rename the files to a consistent format, e.g., `[course_code]_[content_type]_[title].[extension]`.
-   **Metadata Identification:** Identify key metadata from the extracted text, such as due dates for assignments, week numbers for lectures, etc.

## Phase 3: Database Indexing via SQLite
-   **Connect to Database:** Establish a connection to the `sqlite-studies` MCP server.
-   **Process Each Item:** For each piece of content you extracted, create a new record in the `content_items` table.
-   **Populate Records:** Use your analytical abilities to populate the database fields accurately. The table schema is as follows:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `course_id` | TEXT | The identifier for the course (e.g., "SIA1010"). |
| `content_type_id` | TEXT | The category of the material. Use one of: `SYLLABUS`, `LECTURE`, `ASSIGNMENT`, `READING`, `ANNOUNCEMENT`, `EXAM`. |
| `title` | TEXT | The official title of the document or assignment. |
| `description` | TEXT | A brief, auto-generated summary of the content's key points. |
| `file_name` | TEXT | The name of the downloaded file. |
| `file_path` | TEXT | The path to the downloaded file. |
| `source_url` | TEXT | The URL or source path where the material was found. |
| `download_date` | DATETIME | The timestamp of when the data was downloaded. |
| `last_modified` | DATETIME | The last modified date of the file on the platform. |
| `week_number` | INTEGER | The week number of the course the content belongs to. |

## Phase 4: Export to Google Sheets
-   **Create a Script:** Write a script (e.g., in Python) to connect to the SQLite database, fetch the data from the `content_items` table, and export it to a Google Sheet using the Google Sheets API.
-   **Authentication:** Ensure the script handles authentication with the Google Sheets API correctly.
-   **Sheet Formatting:** Format the Google Sheet for readability, with clear headers and organized columns.

## Important Notes
-   **Comprehensive Sweep:** Ensure you navigate through all sub-folders and links within the course page to gather all available documents.
-   **Read-Only on Source:** Your interaction with the course portal should be strictly read-only. Do not modify, submit, or delete anything on the platform.
-   **Accuracy is Key:** The value of the database depends on the accuracy of the extracted information, especially the `content_type_id` and `week_number` fields.