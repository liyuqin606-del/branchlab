# BranchLab：从零实现 Transformer 与训练诊断程序搜索实验

以下条目只引用随附产物；使用时保留测量条件与结果边界。

- 实现 byte-level BPE、causal attention、RoPE、RMSNorm、SwiGLU、AdamW、完整训练恢复和 KV cache，并建立自动化正确性检查。
- 在报告记录的数据子集上训练 19,140,096 参数模型，共 1,228,800 tokens；开发集交叉熵 6.381173 → 2.045100，设备 mps:0。
- 在 cpu / torch.float32、batch 1、prompt 64 / decode 32 的固定测量中，uncached/cached 吞吐分别为 58.72 / 155.55 tokens/s，decode 倍率 2.649；logits 最大绝对差 0.000013351。
- 实现 18 项短干预探针的有限诊断语言、反例优先搜索及最多两个探针的继承，隔离发现/开发/测试状态，与免费日志、固定专家、随机搜索和枚举比较；查询成本与真实全分支采集成本分别记录。
- 该次 pilot 的预设 gate 为 NOGO：The counterexample program does not beat every prespecified comparator in every audit seed, or a branch failed. No diagnostic advantage or RSI claim is admitted.。保留原始结果、动作记录和失败清单，结论限定于本次训练配置。

不将工程完成写成算法领先；不将开发集成绩写成测试集成绩；不宣称复现 Marin 大模型或验证 full RSI。
