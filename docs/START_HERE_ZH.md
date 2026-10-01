# 从哪里开始理解 BranchLab

这是一个从文本、分词到训练、生成和性能测量的完整个人工程项目。
推荐先运行发布模型，再阅读实现，最后重跑小实验。项目的主要价值是
每项实现都有解释、每个性能数字都能回到原始记录。

## 十分钟体验

1. 按仓库 README 下载带 SHA256 校验的 inference 发布包。
2. 使用 `branchlab generate` 生成一段故事，固定 prompt、seed 和采样参数。
3. 安装 `.[demo]`，运行 `python scripts/demo.py --model artifacts/portfolio/showcase-gqa`。
4. 阅读 [实验报告](../reports/portfolio/REPORT.md)，对照生成样例和模型限制。

展示模型只训练了小规模英文故事，没有指令微调。评价它时应看测试损失、
训练是否稳定、实现是否正确，以及在明确工作负载下的推理成本。

## 建议的代码阅读顺序

| 顺序 | 文件 | 阅读时要回答的问题 |
| --- | --- | --- |
| 1 | `src/branchlab/tokenizer.py` | UTF-8 字节如何变成 token？为什么未知字符仍能还原？ |
| 2 | `src/branchlab/model.py` | 每层张量形状是什么？因果 mask 与 RoPE 如何处理已有缓存？ |
| 3 | `src/branchlab/optim.py` | AdamW 的一阶/二阶矩、偏差修正和权重衰减分别做什么？ |
| 4 | `src/branchlab/portfolio_data.py` | 分词器有没有看到测试文本？文档重叠怎样排除？ |
| 5 | `src/branchlab/fit.py` | 累积梯度如何缩放？恢复训练需要哪些状态？ |
| 6 | `src/branchlab/systems.py` | 计时是否同步 GPU？不同路径是否处理同样的 token？ |
| 7 | `scripts/cuda_graph_decode.py` | 固定形状为何能重放图？哪些状态必须重置？ |
| 8 | `scripts/verify_portfolio.py` | 一个好看的结果表被改了数字，检查器能否发现？ |

[架构导读](ARCHITECTURE.md) 给出公式、形状和边界条件。
测试文件与实现分开，包含独立参考计算、错误输入和中断恢复，而不仅检查函数能运行。

## 面试前必须能自己解释

- 对比手写 attention、PyTorch SDPA、动态缓存、静态缓存和 CUDA Graph。
- 用缓存形状推导 8→2 KV heads 为什么减少 75% 持久 K/V 存储。
- 说明这不意味着总显存减少 75%，也不意味着 GQA 的质量一定更好。
- 解释为什么 warm graph 的 decode 提速不能替代端到端延迟。
- 展示三组种子的结果，包括不利结果；区分等 token 与等 FLOPs 比较。
- 修改一个小配置并重跑测试，解释训练中断后需要恢复 RNG、数据游标和优化器状态。

项目使用 AI 辅助开发。CV 中使用与你实际理解、实现和维护程度一致的动词；
不要把已有的 GQA/CUDA Graph 方法说成原创算法。详细问答见
[面试导读](INTERVIEW_GUIDE.md)，可核验的中英文条目见
[CV 文案](../reports/portfolio/CV_ZH_EN.md)。
