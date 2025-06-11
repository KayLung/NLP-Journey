# Transformer架构学习笔记 [2024-06-11]

## 1. 核心理解
- 自注意力机制解决了______问题（附手绘图截图）
- 与RNN相比的优势：______

## 2. 代码实践
```python
# 简易Attention实现
import torch
def attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # 更多代码...
```
**测试结果**：输入序列长度=10时，输出准确率85%

## 3. 疑问
- 为什么LayerNorm比BatchNorm更适合NLP？
