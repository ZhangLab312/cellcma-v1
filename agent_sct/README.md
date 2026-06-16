# Agent-SCT Framework

基于AI Agent的单细胞多组学分析框架，整合三大任务：单细胞多组学数据整合、基因调控网络推断、细胞扰动预测。

## 架构设计

```
Agent-SCT/
├── modules/
│   ├── data_parsing/          # 文献检索及数据解析模块
│   │   ├── semantic_parser.py      # 数据语义解析器（含小样本展示）
│   │   └── literature_retrieval.py # 文献检索模块
│   ├── debate/                # 多专家辩论模块
│   │   └── multi_expert_debate.py  # 四角色专家辩论系统
│   ├── memory/                # 多层级记忆模块
│   │   └── hierarchical_memory.py  # 分层记忆系统
│   └── execution/             # 基于数据注入的代码执行模块
│       └── data_injection_executor.py
├── agents/
│   ├── base_agent.py          # Agent基类
│   └── expert_agents.py       # 四角色专家Agent
├── tasks/
│   ├── base_task.py           # 任务基类
│   ├── integration/           # 多组学整合任务
│   ├── grn_inference/         # GRN推断任务
│   └── perturbation/          # 扰动预测任务
└── run_agent_sct.py           # 统一入口脚本
```

## 核心特性

### 1. 数据语义解析器（小样本展示）
- 从大数据集中采样部分细胞和特征
- 展示实际的obs、var和X内容
- 让LLM能看到数据的实际结构，而不只是统计信息

### 2. 四角色专家辩论系统
- **架构师(Architect)**：设计初始蓝图
- **科学家(Scientist)**：理论校验
- **工程师(Engineer)**：工程实现评估
- **评审专家(Critic)**：批判性审查

### 3. 分层记忆系统
- **短期记忆**：单轮执行周期
- **长期记忆**：跨会话经验累积
- **情景记忆**：具体实验场景记录

### 4. 数据注入执行模块
- 数据预加载到内存
- 代码生成专注于分析逻辑
- 避免路径/格式问题

## 使用方法

### 多组学整合
```bash
python run_agent_sct.py --task integration --rna data_rna.h5ad --atac data_atac.h5ad
```

### GRN推断
```bash
python run_agent_sct.py --task grn --data data.h5ad
```

### 扰动预测
```bash
python run_agent_sct.py --task perturbation --data data.h5ad --mode drug
```

### 完整流程
```bash
python run_agent_sct.py --task full_pipeline --rna data_rna.h5ad --atac data_atac.h5ad
```

## 高级选项

```bash
# 禁用辩论
python run_agent_sct.py --task integration --rna data.h5ad --no-debate

# 禁用文献检索
python run_agent_sct.py --task grn --data data.h5ad --no-literature

# 调整采样数量
python run_agent_sct.py --task perturbation --data data.h5ad --sample-n-obs 200 --sample-n-vars 100

# 调整迭代次数
python run_agent_sct.py --task integration --rna data.h5ad --max-iterations 10
```

## 项目状态

- ✅ Phase 1: 核心基础设施（数据解析、文献检索、记忆模块）
- ✅ Phase 2: 多专家辩论系统
- ✅ Phase 3: 数据注入执行模块
- ✅ Phase 4: 三大任务适配
- ✅ Phase 5: 统一工作流与入口
- ⏳ Phase 6: 配置系统整合（低优先级）
