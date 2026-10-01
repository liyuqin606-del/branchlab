# BranchLab — 从零训练语言模型与高效推理系统

- 实现 byte-level BPE、decoder-only Transformer、RoPE、RMSNorm、SwiGLU、GQA 与手写 attention 参考路径；通过输出/梯度对照验证 PyTorch SDPA 路径，提供可阅读的张量形状与算法导读。
- 在 Colab NVIDIA L4 上训练 17.57M 参数模型，处理 39.32M 训练 tokens；在独立文档的 65,536 个测试 target tokens 上达到交叉熵 1.2290、困惑度 3.42，发布权重与复现配置。
- 实现动态与预分配 KV-cache，8→2 KV heads 将同容量持久 K/V 缓冲区减少 75%；公开普通缓存的有限加速收益及三组种子的等 token 质量对照。
- 实现固定形状 CUDA Graph 解码；在 L4/BF16、batch 1、prompt 256/decode 64 的七次重复中，稳态吞吐达 1512.2 tokens/s，相对动态 KV-cache 提速 10.50×；单独报告捕获开销与数值误差。
- 完成三组训练种子的等 token 预算 MHA/GQA 对照，实现梯度累积、BF16 训练与原子完整状态检查点，并以连续/中断恢复轨迹一致性测试和独立哈希/指标重算验证结果。

英文压缩版：

Built an explainable from-scratch language-model stack with BPE, RoPE, RMSNorm,
SwiGLU, GQA and verifiable attention/cache implementations; trained a
17.57M-parameter TinyStories subset model on 39.32M tokens.
Implemented resumable BF16 training, dynamic/static KV caching and three-seed
MHA/GQA comparisons, with 75% less persistent KV storage for 8→2 KV heads at equal capacity.

Implemented reusable CUDA graph decoding, reaching 1512.2 tokens/s (10.50× over dynamic KV caching) on L4/BF16 at batch 1, prompt 256/decode 64 across seven repetitions; capture and prefill excluded and separately reported.

使用边界：以上指标来自本次特定配置。交叉熵仅代表该数据子集，解码提速不含 prefill。
按你实际掌握和承担的工作描述贡献；项目使用 AI 辅助，不宣称无辅助独立完成或原创 GQA。
