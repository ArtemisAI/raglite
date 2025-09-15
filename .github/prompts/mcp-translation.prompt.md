# MCP Configuration Translation Between AI Agents

## Overview

This prompt provides detailed instructions for translating Model Context Protocol (MCP) server configurations between different AI agents, specifically from VS Code Copilot (`#sym:### .vscode/mcp.json`) to Roo Code (`#.roo/mcp.json`) and vice versa.

## Request Format

To request MCP configuration translation, use this format:

```
Please translate the MCP configuration from [SOURCE_AGENT] to [TARGET_AGENT] format.

Source: #sym:### [SOURCE_PATH]
Target: #[TARGET_PATH]

[Additional specific requirements or customizations]
```

### Example Request

```
Please translate the MCP configuration from VS Code Copilot to Roo Code format.

Source: #sym:### .vscode/mcp.json
Target: #.roo/mcp.json

Include all existing servers and add enhanced stealth configuration for Playwright.
```

## Key Syntactical Differences

### Primary Structure Changes

| Aspect | VS Code Copilot | Roo Code |
|--------|----------------|----------|
| **Root Object** | `"servers": { ... }` | `"mcpServers": { ... }` |
| **Input Variables** | `"inputs": [...]` | Not used (direct env vars) |
| **File Location** | `.vscode/mcp.json` | `.roo/mcp.json` |
| **Config References** | Relative to `.vscode/` | Relative to project root |

### Server Configuration Properties

| Property | VS Code Copilot | Roo Code | Notes |
|----------|----------------|----------|-------|
| **command** | Same | Same | No change |
| **args** | Same | Same | No change |
| **env** | Same | Same | No change |
| **type** | Same | Same | No change |
| **alwaysAllow** | Not typically used | **Required** | List of allowed actions |
| **timeout** | Not typically used | **Recommended** | Timeout in seconds |
| **disabled** | Same | Same | Boolean flag |

## Translation Steps

### Step 1: Change Root Object Name

**From VS Code Copilot:**
```json
{
  "servers": {
    "my-server": { ... }
  },
  "inputs": []
}
```

**To Roo Code:**
```json
{
  "mcpServers": {
    "my-server": { ... }
  }
}
```

### Step 2: Add Required Roo Code Properties

For each server in Roo Code, add:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "...",
      "args": [...],
      "env": {...},
      "type": "stdio",
      "alwaysAllow": [
        "action1",
        "action2",
        "action3"
      ],
      "timeout": 300,
      "disabled": false
    }
  }
}
```

### Step 3: Update Configuration File Paths

**VS Code Copilot paths (relative to `.vscode/`):**
```json
"args": [
  "--config",
  ".vscode/my-config.json"
]
```

**Roo Code paths (relative to project root):**
```json
"args": [
  "--config",
  "./my-config.json"
]
```

### Step 4: Handle Input Variables

**VS Code Copilot with inputs:**
```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "apiKey",
      "description": "API Key"
    }
  ],
  "servers": {
    "my-server": {
      "args": ["--api-key", "${input:apiKey}"]
    }
  }
}
```

**Roo Code (direct environment variables):**
```json
{
  "mcpServers": {
    "my-server": {
      "env": {
        "API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Common Server Types and Their Translations

### SQLite Server

**VS Code Copilot:**
```json
{
  "servers": {
    "sqlite-database": {
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "database_schema/my_database.db"
      ],
      "type": "stdio"
    }
  }
}
```

**Roo Code:**
```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "./database_schema/my_database.db"
      ],
      "env": {
        "DATABASE_URL": "sqlite:./database_schema/my_database.db"
      },
      "alwaysAllow": [
        "read_query",
        "write_query",
        "create_table",
        "list_tables",
        "describe_table",
        "append_insight"
      ],
      "timeout": 300,
      "type": "stdio"
    }
  }
}
```

### Playwright Server

**VS Code Copilot:**
```json
{
  "servers": {
    "playwright-stealth": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest",
        "--extension",
        "--config",
        ".vscode/enhanced-stealth-config.json"
      ],
      "env": {
        "PLAYWRIGHT_BROWSERS_PATH": "./playwright-browsers",
        "PLAYWRIGHT_USER_DATA_DIR": "./playwright-browsers/user-data"
      },
      "type": "stdio"
    }
  }
}
```

**Roo Code:**
```json
{
  "mcpServers": {
    "playwright-stealth": {
      "command": "npx",
      "args": [
        "-y",
        "@playwright/mcp@latest",
        "--executable-path",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "--extension",
        "--config",
        "./roo-stealth-config.json"
      ],
      "env": {
        "PLAYWRIGHT_BROWSERS_PATH": "./playwright-browsers",
        "PLAYWRIGHT_USER_DATA_DIR": "./playwright-browsers/user-data",
        "ROO_CODE_INTEGRATION": "true",
        "MCP_STEALTH_MODE": "enabled"
      },
      "type": "stdio",
      "alwaysAllow": [
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_snapshot",
        "browser_take_screenshot",
        "browser_wait_for",
        "browser_evaluate",
        "browser_fill_form",
        "browser_handle_dialog",
        "browser_close",
        "browser_tabs",
        "browser_console_messages",
        "browser_network_requests"
      ],
      "timeout": 300
    }
  }
}
```

### Filesystem Server

**VS Code Copilot:**
```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "./documents"
      ],
      "type": "stdio"
    }
  }
}
```

**Roo Code:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "./Case_Documents",
        "./.scripts",
        "./database_schema",
        "./_Indexes"
      ],
      "env": {},
      "disabled": true,
      "alwaysAllow": [
        "read_file",
        "write_file",
        "create_directory",
        "list_directory",
        "get_file_info",
        "search_files"
      ],
      "type": "stdio"
    }
  }
}
```

## Roo Code Specific Requirements

### Required Properties

1. **alwaysAllow**: Array of allowed MCP actions
   ```json
   "alwaysAllow": [
     "action1",
     "action2",
     "action3"
   ]
   ```

2. **timeout**: Timeout in seconds for server operations
   ```json
   "timeout": 300
   ```

### Common alwaysAllow Values by Server Type

**SQLite Servers:**
```json
"alwaysAllow": [
  "read_query",
  "write_query",
  "create_table",
  "list_tables",
  "describe_table",
  "append_insight"
]
```

**Playwright Servers:**
```json
"alwaysAllow": [
  "browser_navigate",
  "browser_click",
  "browser_type",
  "browser_snapshot",
  "browser_take_screenshot",
  "browser_wait_for",
  "browser_evaluate",
  "browser_fill_form",
  "browser_handle_dialog",
  "browser_close",
  "browser_tabs",
  "browser_console_messages",
  "browser_network_requests"
]
```

**Filesystem Servers:**
```json
"alwaysAllow": [
  "read_file",
  "write_file",
  "create_directory",
  "list_directory",
  "get_file_info",
  "search_files"
]
```

**MarkItDown Servers:**
```json
"alwaysAllow": [
  "convert_document",
  "extract_text",
  "process_file"
]
```

### Environment Variables

Roo Code specific environment variables to add:

```json
"env": {
  "ROO_CODE_INTEGRATION": "true",
  "MCP_STEALTH_MODE": "enabled",
  "ROO_ENVIRONMENT": "development"
}
```

## File Path Conventions

### Configuration File Locations

**VS Code Copilot:**
- Main config: `.vscode/mcp.json`
- Browser config: `.vscode/enhanced-stealth-config.json`
- Auth manager: `.vscode/gmail-auth-manager.ts`

**Roo Code:**
- Main config: `.roo/mcp.json`
- Browser config: `./roo-stealth-config.json`
- Auth manager: `./gmail-auth-manager.ts` (can be shared)

### Relative Path Updates

When translating, update these path patterns:

| VS Code Copilot | Roo Code | Example |
|-----------------|----------|---------|
| `.vscode/config.json` | `./config.json` | Browser configuration |
| `database_schema/db.sqlite` | `./database_schema/db.sqlite` | Database paths |
| `./documents` | `./Case_Documents` | Filesystem roots |

## Complete Translation Example

### Source: .vscode/mcp.json

```json
{
  "servers": {
    "sqlite-divorce": {
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "database_schema/divorce_case.db"
      ],
      "type": "stdio"
    },
    "playwright-stealth": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest",
        "--extension",
        "--config",
        ".vscode/enhanced-stealth-config.json"
      ],
      "env": {
        "PLAYWRIGHT_BROWSERS_PATH": "./playwright-browsers"
      },
      "type": "stdio"
    }
  },
  "inputs": []
}
```

### Target: .roo/mcp.json

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "./database_schema/divorce_case.db"
      ],
      "env": {
        "DATABASE_URL": "sqlite:./database_schema/divorce_case.db"
      },
      "alwaysAllow": [
        "read_query",
        "write_query",
        "create_table",
        "list_tables",
        "describe_table",
        "append_insight"
      ],
      "timeout": 300,
      "type": "stdio"
    },
    "playwright-stealth": {
      "command": "npx",
      "args": [
        "-y",
        "@playwright/mcp@latest",
        "--executable-path",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "--extension",
        "--config",
        "./roo-stealth-config.json"
      ],
      "env": {
        "PLAYWRIGHT_BROWSERS_PATH": "./playwright-browsers",
        "PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD": "false",
        "PLAYWRIGHT_HEADLESS": "false",
        "PLAYWRIGHT_SLOWMO": "150",
        "PLAYWRIGHT_TIMEOUT": "120000",
        "PLAYWRIGHT_USER_DATA_DIR": "./playwright-browsers/user-data",
        "PLAYWRIGHT_STORAGE_STATE": "./playwright-browsers/auth/gmail-auth.json",
        "ROO_CODE_INTEGRATION": "true",
        "MCP_STEALTH_MODE": "enabled"
      },
      "type": "stdio",
      "alwaysAllow": [
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_snapshot",
        "browser_take_screenshot",
        "browser_wait_for",
        "browser_evaluate",
        "browser_fill_form",
        "browser_handle_dialog",
        "browser_close",
        "browser_tabs",
        "browser_console_messages",
        "browser_network_requests"
      ],
      "timeout": 300
    }
  }
}
```

## Validation Checklist

After translation, verify:

- [ ] Root object name changed from `servers` to `mcpServers`
- [ ] Removed `inputs` array (if present)
- [ ] Added `alwaysAllow` arrays for each server
- [ ] Added `timeout` values for each server
- [ ] Updated configuration file paths from `.vscode/` to `./`
- [ ] Added Roo Code specific environment variables
- [ ] Updated database paths to include `./` prefix
- [ ] Added executable paths for browser-based servers
- [ ] Verified all command and args remain unchanged
- [ ] Added appropriate disabled flags where needed

## Troubleshooting Common Issues

### Server Not Starting
- Check `mcpServers` spelling (not `servers`)
- Verify `alwaysAllow` array is present and populated
- Ensure timeout value is set

### Permission Denied
- Add missing actions to `alwaysAllow` array
- Check file paths are accessible from project root

### Configuration Not Found
- Update config file paths from `.vscode/` to `./`
- Ensure configuration files exist in project root

### Browser Not Launching
- Add explicit `executablePath` for browser servers
- Verify Chrome installation path is correct

## Advanced Features

### Conditional Server Enabling

```json
{
  "mcpServers": {
    "development-only-server": {
      "command": "...",
      "disabled": false,
      "env": {
        "NODE_ENV": "development"
      }
    },
    "production-disabled-server": {
      "command": "...",
      "disabled": true
    }
  }
}
```

### Environment-Specific Configuration

```json
{
  "mcpServers": {
    "adaptive-server": {
      "env": {
        "ROO_ENVIRONMENT": "development",
        "DEBUG_MODE": "true",
        "STEALTH_LEVEL": "maximum"
      }
    }
  }
}
```

This prompt provides comprehensive guidance for translating MCP configurations between AI agents, ensuring all syntactical differences and requirements are properly addressed.
