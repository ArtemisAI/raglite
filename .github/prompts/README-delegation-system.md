# AI Agent Delegation System - 4-Prompt Cycle

This system optimizes AI agent collaboration by separating strategic thinking (expensive) from execution (cost-effective) while maintaining quality control.

## 🔄 **Workflow Cycle**

### **Prompt #1: Strategic Analysis & Planning** 
📁 `to-small-ai.prompt.md`
- **Agent**: Big AI (GPT-4/Claude 3.5 Sonnet)
- **Role**: Architect, Planner, Strategist
- **Output**: Detailed execution plan for smaller agent

### **Prompt #2: Task Execution**
📁 `small-ai-execution.prompt.md`  
- **Agent**: Small AI (GPT-3.5/Claude 3 Haiku)
- **Role**: Executor, Implementation
- **Output**: Documented results and completion report

### **Prompt #3: Quality Assurance**
📁 `big-ai-validation.prompt.md`
- **Agent**: Big AI (returns for validation)
- **Role**: Quality Gate, Supervisor, Tester
- **Output**: Approval or iteration guidance

### **Prompt #4: Continuous Improvement**
📁 `big-ai-iteration.prompt.md`
- **Agent**: Big AI (when iteration needed)
- **Role**: Process Optimizer, Problem Solver
- **Output**: Refined instructions for retry

## 💰 **Cost Optimization Strategy**

| Phase | Agent Type | Token Usage | Cost Impact | Git Requirement |
|-------|------------|-------------|-------------|------------------|
| Planning | Big AI | Medium | High precision planning reduces iterations | Pre-delegation commit |
| Execution | Small AI | High | 90% cost reduction for implementation | Pre + Post commits |
| Validation | Big AI | Low | Quick quality check prevents rework | Git history review |
| Iteration | Big AI | Medium | Only when needed, improves future success | Rollback planning |

## 🔄 **Git Workflow Integration**

### **Commit Strategy**
- **Pre-Delegation**: Big AI commits current state before delegating
- **Pre-Execution**: Small AI commits before starting work  
- **Post-Execution**: Small AI commits completed changes
- **Post-Validation**: Big AI commits any additional fixes

### **Version Control Benefits**
- **Rollback Safety**: Can undo changes if validation fails
- **Change Tracking**: Clear audit trail of what was modified
- **Blame Assignment**: Know which agent made which changes
- **Iteration Support**: Easy to reset to clean state for retries

## 🎯 **Usage Examples**

### **SIA1010 Workspace Issue** (Current)
1. **Big AI**: Analyzes missing workspace folders → Plans diagnostic approach
2. **Small AI**: Reads workspace files, identifies gaps, fixes configuration  
3. **Big AI**: Validates workspace loads correctly with all shared folders
4. **Big AI**: (If needed) Refines approach based on any issues found

### **Database Path Issues** (Solved)
1. **Big AI**: Identified MCP config problems → Planned comprehensive fix
2. **Small AI**: Would execute path corrections across all files
3. **Big AI**: Would validate all database references work correctly
4. **Big AI**: Would optimize detection script for future prevention

## 🚀 **Benefits**

- **Cost Reduction**: 70-90% savings on execution tasks
- **Quality Assurance**: Big AI oversight ensures robust solutions  
- **Efficiency**: Clear role separation reduces confusion
- **Scalability**: System works for any technical task
- **Learning**: Each cycle improves future performance

## 📋 **Implementation Guide**

1. **Start** with problem description to Big AI using Prompt #1
2. **Execute** plan using Small AI with Prompt #2  
3. **Validate** results using Big AI with Prompt #3
4. **Iterate** if needed using Big AI with Prompt #4
5. **Document** lessons learned for future similar issues

This system transforms expensive AI consultation into cost-effective, high-quality technical problem solving!
