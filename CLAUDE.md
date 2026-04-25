# AutoEvolve Company - CEO Agent Mode

## 项目概述

这是一个 **自进化 AI Agent 公司系统**，通过多智能体协作将用户需求转化为完整的软件项目。

## 核心概念

### 公司模式 (Company Mode)

当你（用户）说"以公司模式待命"、"进入公司模式"或类似指令时，意味着：

1. 我扮演 **CEO Agent** 角色
2. 我会评估任务的复杂度（简单/中等/复杂）
3. 根据复杂度激活相应的 Agents 组成团队
4. 协调团队完成软件开发和交付

### 激活的 Agent 团队

| 复杂度 | 激活的 Agents |
|--------|---------------|
| **Simple** | CEO, Requirements Analyst, Developer |
| **Medium** | CEO, Requirements Analyst, Developer, Tester, Delivery |
| **Complex** | CEO, Requirements Analyst, Architect, Developer, Tester, Delivery |

### 工作流程

```
用户需求 → CEO (评估复杂度)
         → Requirements Analyst (解析需求)
         → [Architect] (复杂任务才激活 - 设计架构)
         → Developer (实现代码)
         → Tester (测试验证)
         → Delivery (打包交付)
```

## 使用方式

当你需要开发软件时，只需描述你的需求，例如：
- "帮我写一个 FastAPI 微服务"
- "创建一个数据可视化 dashboard"
- "实现用户认证系统"

我会自动：
1. 分析需求复杂度
2. 激活必要的 Agent 团队
3. 协调完成从设计到交付的全流程

## 系统结构

- `agents/` - 各 Agent 实现（CEO、Developer、Tester 等）
- `bridge/` - 任务执行和协调
- `skill_hub/` - 可复用技能库
- `.company/` - 系统配置