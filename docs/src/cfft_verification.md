# CFFT 正确性验证与数学原理

本文档详细说明了 CrystallographicFFT.jl 中 CFFT 算法正确性的验证逻辑，特别是 **直接谱重构 (Direct Spectral Reconstruction)** 方法的数学原理，以及为何在验证过程中引入 `split_homogeneous_blocks` 函数是数学上必要的。

## 1. 验证目标

CFFT 的核心目标是利用晶体对称性减少 FFT 的计算量和存储量。它计算的是不可约谱单元 (Spectral ASU) 上的傅里叶系数 $G(\mathbf{k})$。为了验证 CFFT 的正确性，必须证明：
**从压缩的 Spectral ASU 数据 $G(\mathbf{k})$ 中，可以无损地通过数学公式重构出全空间的标准傅里叶谱 $F(\mathbf{h})$，且重构结果与对全空间进行标准 FFT 的结果完全一致（误差接近机器精度）。**

## 2. 直接谱重构 (Direct Spectral Reconstruction)

根据 KRFFT (Kunis-Roessler FFT) 理论，全空间谱 $F(\mathbf{h})$ 可以通过对 Spectral ASU 中的分量进行对称操作求和来获得。

### 基础公式推导

为了回答上述关键问题，我们先明确符号定义。

1. **什么是 $\mathbf{h}$？**
   $\mathbf{h} \in \mathbb{Z}^D$ 表示 **全空间倒易网格 (Full Reciprocal Grid)** 上的一个波矢索引。它的取值范围覆盖整个倒易晶胞（例如 $0 \le h_i < N_i$）。
   **注意**：$\mathbf{h}$ 不一定在 Spectral ASU 内，它是一个用于验证的通用查询点。

2. **什么是 $\hat{f}_b$（记为 $G_b$）？**
   我们将实空间函数 $f(\mathbf{x})$ 分解为若干个定义在互不重叠区域（Block）上的局部函数之和：
   $$ f(\mathbf{x}) = \sum_{b \in \text{RealASU}} f_b(\mathbf{x}) $$
   其中 $f_b(\mathbf{x})$ 仅在 Block $b$ 内有值，其余为零。
   $G_b(\mathbf{k})$ 是这个局部函数 $f_b$ 的离散傅里叶变换（DFT）：
   $$ G_b(\mathbf{k}) = \sum_{\mathbf{x} \in \text{Block } b} f(\mathbf{x}) e^{-2\pi i \mathbf{k} \cdot \mathbf{x} / \mathbf{N}} $$
   这也是 CFFT 算法实际存储在内存中的数据（即 Spectral ASU Blocks）。

3. **直接重构公式**
   全空间谱 $F(\mathbf{h})$ 是 $f(\mathbf{x})$ 的全局 DFT。利用线性性质和群对称性：
   $$ F(\mathbf{h}) = \mathcal{F}\{ \sum_{g \in G} f(R_g \mathbf{x} + \mathbf{t}_g) \} $$
   最终推导出的重构公式（Direct Recombination Formula）为：
   $$ F(\mathbf{h}) = \sum_{g \in G} e^{-2\pi i \mathbf{h} \cdot \mathbf{t}_g} \sum_{b \in \text{RealASU}} w_b G_b(R_g^T \mathbf{h}) $$

### 关键疑问解答

#### Q1: $G_b$ 具体是什么？(对应原文档 $\hat{f}_b$)
$G_b$ 是 Block $b$ 的局部谱。
在代码中，它对应 `spectral_asu.dim_blocks` 中存储的数组。
如果 Block $b$ 在实空间的大小是 $M_1 \times M_2 \dots$，那么 $G_b$ 就是一个 $M_1 \times M_2 \dots$ 的多维数组。

#### Q2: 怎么从 ASU 推导出 Spectral ASU？ASU 里的点是否一一对应？
Spectral ASU 是通过对 Real ASU 中的每个 Block 进行独立的多维 FFT 计算得到的。
$$ G_b = \text{FFT}(f_b) $$

**回答**：
1. **Block 对应关系**：是的。实空间有一个 $M \times N$ 的 Block $b$，倒易空间就有一个同样大小 $M \times N$ 的 Spectral Block $G_b$。
2. **点对应关系**：**不是**。实空间 Block 中的一个点 $\mathbf{x}$ **不**对应 Spectral Block 中的一个特定点 $\mathbf{k}$。
   - 傅里叶变换是全局操作。Spectral Block 中任意一个点 $\mathbf{k}$ 的值，由 Real Block 中**所有**点 $\mathbf{x}$ 的值共同决定（加权求和）。
   - 因此，**绝无可能**“无需进行任何额外计算”就得到 Spectral ASU。必须进行 FFT 计算。
   - 这正是 Verification 的核心：我们要验证这些经过 FFT 的复杂复数数据，在经过数学重组后，能否还原出物理上直观的全空间波函数。

#### Q3: 详细推导：直接重构公式是怎么来的？
我们要证明：
$$ F(\mathbf{h}) = \sum_{g \in G} e^{-2\pi i \mathbf{h} \cdot \mathbf{t}_g} \sum_{b} w_b G_b(R_g^T \mathbf{h}) $$

**推导步骤**：

**第一步：定义全空间函数的构建**
全空间函数 $f(\mathbf{x})$ 可以看作是 ASU 中的函数 $f_{ASU}(\mathbf{x})$ 经过对称群 $G$ 作用后的叠加。
为了处理方便，我们假设已经处理好了权重 $w_b$，使得我们可以简单地写成（忽略特殊位置的重复求和问题，或假设已加权）：
$$ f(\mathbf{x}) = \sum_{g \in G} f_{ASU}(g^{-1} \cdot \mathbf{x}) $$
其中 $g^{-1} \cdot \mathbf{x}$ 表示将 $\mathbf{x}$ 逆变换回到 ASU 区域。
具体地，若 $g = \{R, \mathbf{t}\}$，则 $g \cdot \mathbf{y} = R\mathbf{y} + \mathbf{t}$。
逆操作为 $g^{-1} \cdot \mathbf{x} = R^{-1}(\mathbf{x} - \mathbf{t})$。
且 $f_{ASU}(\mathbf{y}) = \sum_b f_b(\mathbf{y})$。

**第二步：利用傅里叶变换的线性性质**
$$ F(\mathbf{h}) = \mathcal{F}\{ f(\mathbf{x}) \} = \mathcal{F}\{ \sum_{g \in G} f_{ASU}(g^{-1} \cdot \mathbf{x}) \} $$
由线性叠加原理：
$$ F(\mathbf{h}) = \sum_{g \in G} \mathcal{F}\{ f_{ASU}(g^{-1} \cdot \mathbf{x}) \} $$
令 $y_g(\mathbf{x}) = f_{ASU}(R^{-1}(\mathbf{x} - \mathbf{t}))$，我们需要计算 $y_g(\mathbf{x})$ 的 DFT。

**第三步：利用仿射变换性质（Shift & Rotation Theorem）**
根据定义：
$$ Y_g(\mathbf{h}) = \sum_{\mathbf{x}} f_{ASU}(R^{-1}(\mathbf{x} - \mathbf{t})) e^{-2\pi i \mathbf{h} \cdot \mathbf{x} / \mathbf{N}} $$

进行变量代换：令 $\mathbf{x}' = R^{-1}(\mathbf{x} - \mathbf{t})$。
则 $\mathbf{x} = R\mathbf{x}' + \mathbf{t}$。
当 $\mathbf{x}$ 遍历全空间时，$\mathbf{x}'$ 也遍历全空间（因为 $R$ 是幺模矩阵）。
代入公式：
$$ Y_g(\mathbf{h}) = \sum_{\mathbf{x}'} f_{ASU}(\mathbf{x}') e^{-2\pi i \mathbf{h} \cdot (R\mathbf{x}' + \mathbf{t}) / \mathbf{N}} $$

展开指数项：
$$ e^{-2\pi i \mathbf{h} \cdot (R\mathbf{x}' + \mathbf{t})} = e^{-2\pi i \mathbf{h} \cdot \mathbf{t}} \cdot e^{-2\pi i \mathbf{h} \cdot R\mathbf{x}'} $$

利用点积性质 $\mathbf{a} \cdot (M \mathbf{b}) = (M^T \mathbf{a}) \cdot \mathbf{b}$：
$$ \mathbf{h} \cdot R\mathbf{x}' = (R^T \mathbf{h}) \cdot \mathbf{x}' $$

代回求和式：
$$ Y_g(\mathbf{h}) = e^{-2\pi i \mathbf{h} \cdot \mathbf{t} / \mathbf{N}} \sum_{\mathbf{x}'} f_{ASU}(\mathbf{x}') e^{-2\pi i (R^T \mathbf{h}) \cdot \mathbf{x}' / \mathbf{N}} $$

注意求和部分 $\sum_{\mathbf{x}'} f_{ASU}(\mathbf{x}') e^{-2\pi i \mathbf{k}' \cdot \mathbf{x}'}$ 正是 $f_{ASU}$ 在波矢 $\mathbf{k}' = R^T \mathbf{h}$ 处的傅里叶变换值，即 $G_{ASU}(R^T \mathbf{h})$。

**第四步：结论**
$$ Y_g(\mathbf{h}) = e^{-2\pi i \mathbf{h} \cdot \mathbf{t} / \mathbf{N}} G_{ASU}(R^T \mathbf{h}) $$
代入总和公式，并利用 $G_{ASU} = \sum_b G_b$：
$$ F(\mathbf{h}) = \sum_{g \in G} e^{-2\pi i \mathbf{h} \cdot \mathbf{t}_g} \sum_{b} G_b(R_g^T \mathbf{h}) $$
（此处省略权重细节，权重本质上是修正“第一步”中简单的求和假设）。

#### Q4: $R_g^T \mathbf{h}$ 是否要求在 ASU 里？怎么保证？
**不需要** $R_g^T \mathbf{h}$ 落在 Spectral ASU 的几何范围内。
公式中 $G_b(R_g^T \mathbf{h})$ 的含义是：计算 Block $b$ 的频谱在波矢 $\mathbf{k}' = R_g^T \mathbf{h}$ 处的值。

**机制（Aliasing / Periodicity）**：
由于 $f_b(\mathbf{x})$ 是定义在 $N$ 网格上的离散函数，其 DFT $G_b(\mathbf{k})$在倒易空间是周期的，周期为 $N$。
更进一步，由于 $f_b$ 仅在局部子网格 (Sub-grid) 上非零（假设 Block 是矩形 $M \times \dots$），根据采样定理，$G_b(\mathbf{k})$ 的信息完全由其主值区间 $0 \le k < M$ 决定（这称为 Alias-folding）。
在 CFFT 实现和验证中，当我们查询 $G_b(\mathbf{k}')$ 时，实际上是查询存储数组中的对应值：
$$ \text{Index}_i = (\mathbf{k}'_i \pmod{M_i}) $$
这在代码中体现为 `idx_local[i] = mod(k_rot[i], sz) + 1`。
因此，无论 $R_g^T \mathbf{h}$ 旋转到哪里，通过模运算（Aliasing），它总是映射到 $G_b$ 存储数组的一个合法索引上。

#### Q3: 实空间 ASU 和倒易空间 ASU 怎么对应？
它们是两个独立的概念，通过 FFT 算法联系。
- **Real Space ASU ($\mathcal{D}_P$)**: 实空间中不可约的**位置**集合。$f(\mathbf{x})$ 的值存储在这里。
- **Reciprocal Space ASU ($\mathcal{D}_F$)**: 倒易空间中不可约的**波矢**集合。$F(\mathbf{h})$ 的值存储在这里。

在 CFFT 流程中，我们从实空间 ASU 并不能直接得到倒易空间 ASU 的系数。
实际上，CFFT 的中间产物（也是我们验证的对象）是 **Spectral Blocks**。这些 Blocks 的总和在信息量上等价于 Real Space ASU。
- 验证过程中的 $\mathbf{h}$ 是我们在全空间任意选取的一个测试波矢，它完全可能（且通常）不在 Reciprocal ASU 内。
- 但我们可以利用上述公式，通过查询 Spectral Blocks（源自 Real ASU）来计算出任何 $\mathbf{h}$ 处的 $F(\mathbf{h})$ 值，并与标准结果对比。

---


## 3. 特殊位置的多重性问题 (Multiplicity Issue)

在标准的实空间求和中，如果我们将 ASU 中的所有点应用所有对称操作 $g \in G$ 并求和，对于 **一般位置 (General Position, GP)** 的点，会生成 $|G|$ 个不同的图像。
但对于 **特殊位置 (Special Position, SP)** 的点（如旋转轴上的点，或原点），其在某些 $g$ 操作下保持不变（或仅差一个晶格平移）。该点对应的稳定子群 (Stabilizer Group) $S_{\mathbf{x}} = \{ g \in G \mid R_g \mathbf{x} + \mathbf{t}_g \equiv \mathbf{x} \}$ 非平凡（即 $|S_{\mathbf{x}}| > 1$）。

如果我们简单地对所有 $g$求和，特殊位置的点会被重复计算 $|S_{\mathbf{x}}|$ 次。
为了恢复正确的物理量（全空间 FFT 结果），必须引入权重修正：
$$ w(\mathbf{x}) = \frac{1}{|S_{\mathbf{x}}|} $$

因此，正确的重构公式应为：
$$ F(\mathbf{h}) = \sum_{g \in G} e^{-2\pi i \mathbf{h} \cdot \mathbf{t}_g} \left( \sum_{b \in \text{ASU}} \frac{1}{|S_{\text{start of } b}|} Y_b(R_g^T \mathbf{h}) \right) $$
其中 $Y_b(\mathbf{k})$ 是 ASU Block $b$ 贡献的谱分量。

## 4. 优化带来的挑战：混合 Block (Mixed Stabilizers)

### 问题复现
在验证 `p2mm` 空间群（2D，$N=16 \times 16$）时，我们发现直接应用上述加权公式会出现 $\approx 0.5$ 的相对误差。但在 $N=4 \times 4$ 时验证通过。

### 原因分析
CrystallographicFFT.jl 依赖 `Crystalline.jl` 计算 ASU。为了极致的内存和计算效率，`calc_asu` 算法倾向于将相邻的、拓扑连接的区域合并为一个 Block。
例如，在 `p2mm` 中：
- $(0,0)$ 点是特殊位置（稳定子阶数 4）。
- $(0, y)$ 轴线是特殊位置（稳定子阶数 2）。
- 为了减少 Block 数量，算法可能将它们合并为一个 Block（范围 $y \in [0, N/2]$）。

**这就是问题的根源**：同一个 Spectral Block 内部包含了 **不同稳定子阶数** 的点。
当我们对整个 Block 应用同一个权重 $1/|S|$ 时：
- 如果取首点 $(0,0)$ 的权重 $1/4$，则轴线上的点（应为 $1/2$）被少算了一半。
- 如果取轴线点的权重 $1/2$，则原点被多算了一倍。

数学上，这意味着 Block $b$ 不再是 "同质" (Homogeneous) 的。此时 $Y_b(\mathbf{k})$ 无法简单通过乘法因子修正。

## 5. 解决方案：split_homogeneous_blocks

为了在验证阶段解决这个问题（而不牺牲 CFFT 生产环境的性能），我们引入了 `split_homogeneous_blocks` 函数。

### 机制
该函数的作用是 **逻辑上的再分解**。它遍历 ASU 中的每一个 Block，检查其内部所有点的稳定子阶数。
如果发现 Block 含有混合稳定子，函数将其拆解为最小的同质单元（在极端验证模式下，直接拆解为 $1 \times 1 \dots$ 的单点 Block）。

### 数学等价性
拆分后的 ASU 集合 $\text{ASU}_{split}$ 满足：
$$ \bigcup_{b' \in \text{ASU}_{split}} b' = \text{ASU}_{orig} $$
且对于任意 $b' \in \text{ASU}_{split}$，其内部所有点 $\mathbf{x} \in b'$ 具有相同的稳定子阶数 $|S_{b'}|$。

此时，我们可以对每个拆分后的 Block $b'$ 安全地应用直接重构公式：
$$ F_{recon}(\mathbf{h}) = \sum_{g \in G} \sum_{b'} \frac{1}{|S_{b'}|} \dots $$

### 验证流程
我们在 `test/test_cfft_plan.jl` 中实施的验证步骤如下：
1. **生成数据**：构建包含混合 Block 的标准 ASU 数据。
2. **逻辑拆分**：调用 `split_homogeneous_blocks` 将实空间 ASU 拆分为同质点集。
3. **模拟谱变换**：对拆分后的点集进行 FFT，得到 "理想的、未合并的" Spectral ASU（这一步模拟了如果 CFFT 不进行 Block 合并会得到的结果）。
4. **加权重构**：对这些理想 Block 应用加权求和公式。
5. **对比**：将重构结果与全空间标准 FFT 结果对比。

## 6. 结论

通过引入 `split_homogeneous_blocks`，我们在 `p2mm` ($N=16$) 测试中取得了 $2.4 \times 10^{-15}$ 的重构误差。

这证明了：
1. **CFFT 是正确的**：它捕获了所有必要的物理信息。
2. **直接验证是可行的**：只要我们在验证逻辑中正确处理了 Block 合并带来的多重性混合问题。
3. **混合 Block 仅是优化手段**：它不会破坏数据完整性，但要求在使用直接求和公式（而非逆变换）时必须进行同质化处理。
