# EgoRecover 推进日志

所有操作仅在 WSL `zyc111`、Linux 用户 `ld666` 下执行。代码目录：`UEM-E7/`，基线分支 `e7-codebase`，起点 `03d0202`。本文按阶段记录事实、验证结果和限制；接口通过不等于方法精度得到验证。

## 2026-09-23 04:48 +08:00：任务与验收边界

用户授权从接口开发继续推进方案，完成可独立完成的工作，并记录每阶段结果和问题。

| 阶段 | 当前状态 | 验收标准 |
|---|---|---|
| A. Conda 环境 | 已完成 | `egorecover` 可导入依赖，实际 CUDA 运算成功 |
| B. 历史主干/Flow/动作/迁移接口 | 已完成；当前 54 项测试通过 | 真实 12 层 G 反传、单帧采样、mask 隔离、迁移检查、旧 E7 回归 |
| C. 数据交接与适配 | 已完成；120 个变体、24,000 个变体帧检查通过 | READY 校验、实际 9D→18D、243D 目标/历史重编码、GT/观测隔离 |
| D. 小样本训练与连续回放 | v1/v2/v3 工程实验完成；20 条常规闭环均有有限结果 | 可复跑，训练/评估分离，明确历史及坐标来源，报告故障/恢复表现 |
| E. P/G/Q 方法验证 | 预测历史工程适配完成；正式方法效果仍未验证 | 尚需有效 E7 启动、实际 SMPL-X FK、有效 P 与正式 Q 学习 |

### 已完成的设计与代码

- 设计见 `EgoRecover_主干改造设计.md`。
- 新增 `model/history_uniegomotion.py`：复用 E7 参数名，主干拼接历史与当前，输出 `[B,1,243]`。
- 新增 `egorecover/actions.py`、`conditioning.py`、`history_flow.py`：四动作及实际缺失合并、严格张量边界、当前帧 Flow。
- 新增 `egorecover/checkpoint.py`：先 strict 加载原 E7，按原参数顺序应用 EMA，再迁移新主干；新参数归零，不恢复旧优化器。
- 新增 `run/smoke_egorecover.py`：完整网络的合成输入反传与四动作 Euler10 检查。
- 原 `model/core.py`、`model/uniegomotion.py`、`mydiffusion/flow_matching.py` 未修改。

### 环境问题与处理

- 首次 smoke 使用系统 Python 3.10，因缺少 `yacs` 在导入阶段退出，**没有通过网络测试**。
- 在用户指定 Conda 前，曾创建项目 `.venv` 并安装小型依赖；该环境不再用于后续运行。
- 已按用户要求创建 `/home/ld666/miniconda3/envs/egorecover`，Python 3.11；正在安装可复现的 CUDA PyTorch 和项目依赖。
- 早前受限环境的 `torch.cuda.is_available()` 为 False；切换可访问 GPU 的权限后，`nvidia-smi` 识别 **RTX PRO 6000 Blackwell 97887 MiB**、驱动 **596.49**。因此先前“尚未识别可用 CUDA”不是无 GPU 的结论，待新环境实际张量运算确认。
- CUDA wheel 来源：PyTorch 官方历史版本页 https://pytorch.org/get-started/previous-versions/，选定 PyTorch 2.10.0 / torchvision 0.25.0 / CUDA 12.8。

### 数据等待

已执行用户指定命令：

```bash
python3 -u /home/ld666/projects/EgoRecover/data_pipeline/wait_for_dataset.py --timeout 7200
```

当前进程仍在等待 `data/EE4D_MISMATCH_READY.json`，尚未收到完成输出。等待脚本检查每个数据集的 `audit/validation.json`、`status=passed`、`source_hashes_verified` 和 `spec.json`。不会由模型工作线自行生成/伪造 READY 信号。

### 当前研究限制

- 真实 E7 checkpoint、SMPL-X 资产尚未在本工作线定位；不能将随机初始化模型的数值通路测试当作预训练重建效果。
- 数据使用独立验证集构建的 pilot 时，必须记录开发/测试拆分，不能声称官方测试泛化或完整论文结果。
- 坐标重编码、预测历史误差累积、被屏蔽轨迹是否通过参考系泄漏，仍是下一阶段的关键验收项。

## 2026-09-23 04:55 +08:00：数据交接成功，环境安装继续

- 指定等待脚本确实输出 **“数据集构建完成”**，已取得 **exit_code=0**。完成信号记录构建结束时间为 04:52:14 +08:00。
- 已阅读 `data_pipeline/HANDOFF.md`、`README.md`；只通过 `observations()` 向在线模型传递数据，`supervision()` 和审计字段单独使用。
- smoke：`data/ee4d_mismatch_smoke_v0`，2 takes × 18 variants，共 36 × 200 帧。
- pilot：`data/ee4d_mismatch_pilot_v0`，12 takes × 7 variants，共 84 × 200 帧。
- 数据侧报告 22 passed / 0 failed，并通过源文件 SHA256 和独立审计；详细结果在 `data_pipeline/BUILD_REPORT.md` 和数据目录 `audit/validation.json`。
- Conda 内 PyTorch 2.10.0+cu128 / torchvision 0.25.0+cu128 安装成功；大型 wheel 的两次中断已由 pip 续传恢复。正在安装原仓库 requirements 及测试依赖。
- 已补上几何 codec：保存 dense joint transforms 与内部 reference，历史重新编码，当前增量以此前**预测** reference 为基准，解码接口不读取轨迹或 GT；180° heading 使用 atan2，退化 6D rotation 显式报错。
- 18D 当前轨迹的后 9 维也以此前提交的身体 reference 为基准，避免引入 G 的过去原始观测依赖。干净、对齐的历史下与 E7 相邻头部 reference 一致；有预测漂移时属于需要微调验证的条件分布变化。该选择会在使用说明与测试中明确。
- 对 `/home/ld666` 做了 Linux 文件名搜索（排除环境/缓存目录），暂未定位 E7 checkpoint / SMPLX_NEUTRAL.npz；已向用户询问已存在的 Linux 路径，其他工作继续。

## 2026-09-23 04:59 +08:00：阶段 A/B 首轮验收完成

### 环境结果

- `egorecover` Python 3.11、PyTorch 2.10.0+cu128、CUDA runtime 12.8、Lightning 2.4.0、NumPy 1.26.4。
- CUDA 可用，真实 GPU 矩阵乘法输出有限值；完整 G 的 GPU 反传和四动作 Euler10 全部通过。
- `pip check`：No broken requirements found。
- 初次 pytest 被系统 `/opt/ros/humble/lib/python3.10/site-packages` 插件污染，在收集测试前因缺 `lark` 失败。没有给项目补装无关 ROS 依赖；已为 **egorecover 环境本身** 设置 `PYTHONPATH=""`、`PYTHONNOUSERSITE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，重跑成功。未改变系统 ROS 配置。

### 代码验证

```bash
cd /home/ld666/projects/EgoRecover/UEM-E7
conda run --no-capture-output -n egorecover python -m pytest tests -q --junitxml=verification/egorecover_tests.xml
conda run --no-capture-output -n egorecover python -m run.smoke_egorecover --device cuda --output verification/egorecover_interface_smoke.json
```

- **40 passed，12.59 s**。其中包括原 E7 配置、Lightning 一步训练/EMA/采样及 repaint 回归，以及新增完整主干、权重迁移、四动作、屏蔽 NaN、历史不变、当前帧损失、坐标重编码测试。
- GPU smoke：loss=3.043029，梯度有限，四动作 `[1,1,243]`，每动作 10 次 G，历史未改变。随机初始化、合成数据，**不作为精度结论**。
- 迁移测试使用人工构造的标准 E7 checkpoint/EMA 验证完整映射、缺 key/EMA 长度错误拒绝；未取得真实预训练 E7 权重。
- codec 测试覆盖：物理状态换参考后往返、上一帧预测偏差的纠正增量、物理保持先验的 identity delta、180° heading、缺失轨迹隔离、退化旋转拒绝。

### 真实数据读取发现

- 已通过 Conda 读取首条真实 pilot。身体标签含 200×76×3 joints、21 个 body rotation6d、12D 左右手 PCA，故可进行无需 SMPL 资产的 **dense 表示编码/解码诊断**。
- 这不等于 SMPL22 FK 评估；正式 FK 仍需 SMPL-X 模型资产。
- 首条样本 `floor_height=-1.24582196 m`。原 E7 将身体和头部观测减去该 floor；新在线系统不能不声明就从监督读取它。下一阶段应将 floor 作为明确场景标定输入，工程实验若用数据提供的标定，必须单独标注其假设；不得声称已经解决无标定在线部署。
- smoke 的 2 个 take 也属于 pilot 的 12 个 take；两套目录不是独立训练/测试划分，需要按 take 重新划分工程 train/dev。

## 2026-09-23 05:10—05:22 +08:00：阶段 C/D，真实数据第一轮实验

### 实现范围

- 新增 `HistoryBuffer`：从共同初始化器的预测建立 20 帧缓存；候选解码不提交；仅选中结果按连续时间提交一次；保存固定 `beta_boot`。
- 新增 `MotionCodec`：物理保持与常速度（世界平移 + SO(3) 旋转外推）先验；重新编码首项/增量；当前目标连接此前预测 reference。
- 新增 P（256D、4 层、8 头，物理保持先验上的表示修正）；冻结后父 module 切 train 也不会重新开启 dropout/梯度。
- 新增 Q 的双编码器和动作选择接口；缺失时统一合并等价动作。离线四动作标签生成共享噪声，不修改状态缓存。尚未进行正式 Q 训练。
- 新增 `rollout.py`：只接收 `observations(as_of=t, history_frames=...)`，线上不读取 supervision；每帧 P/Q 各一次、G 只采样一个动作。解码后才提交，故障开始/恢复时不重置。
- `annotations.py` 和 `engineering.py` 专门隔离 **离线 GT 历史开发批次**，不作为在线输入生产器。

### 真实数据学习实验 v0

命令（Conda 环境名始终为 egorecover）：

```bash
conda run --no-capture-output -n egorecover python -u -m run.engineering_pilot \
  --output exp/egorecover_engineering_v0 --prior-steps 200 --flow-steps 100 --batch-size 32 \
  --contact-floor-source startup_estimate
conda run --no-capture-output -n egorecover python -m run.compare_engineering_baselines \
  --experiment exp/egorecover_engineering_v0
```

- 按 take 固定分为 8 train / 2 dev / 2 holdout；所有故障变体随原 take。holdout 未用于选模型或参数。
- **随机初始化 G，没有 E7 预训练权重；GT 历史；官方 val 工程样本；dense22 位置指标，不是 SMPL22 FK。**
- 两种源获得相同 H/μ，初始权重、训练抽样和噪声种子相同，四动作均衡随机抽样，σ=1.0、Euler10。
- P 200 步后完整 clean-dev 加权 MSE：物理保持基线 0.083238 → P 0.068356，按 dev 选择第 200 步并冻结。
- G 两种源各 100 步；固定 dev 时间/噪声的 loss：Gaussian 2.632976→0.511587；History 2.709198→0.459521。
- 总时长约 50.93 s；权重、训练曲线、split、统计/数据/权重哈希保存在 `exp/egorecover_engineering_v0/`。

同一组 64 个开发帧的单步对照（不是连续策略表现）：

| 方法 | dense22 世界位置误差 mm |
|---|---:|
| 物理末帧保持 | 25.20 |
| 物理常速度 | **15.17** |
| P | 38.04 |
| Gaussian-source G，a11 | 400.82 |
| History-source G，a11 | 288.33 |

四动作 oracle gap：Gaussian 24.66 mm，History 22.35 mm。此处仅表明这批 **GT 历史 + 随机初始化短训 G + dense 指标** 下动作有差异，不能据此证明正式 Q 可学习或有闭环收益。缓存命名为 `utility_diagnostic_*.pt`，不能混作正式 FK 标签。

### 第一轮暴露的问题

1. **P 的表示损失下降，不代表位置误差下降。** P 在上述同帧位置指标上落后于保持/常速度；需要真实 FK 指标、动作转折分析和预测历史适配，不能只看 loss。
2. **短训 G 破坏了很强的历史先验。** 即便 History 源优于 Gaussian 源，两者都远差于物理简单先验。当前证据不支持“完整方法优于基线”。
3. **地面估计误差不可忽略。** 用启动头高减训练平均头高，10 个已使用 take 的误差约 -0.488～+0.428 m。骑行/弯腰等姿态会破坏固定头高假设。在线没有读取 GT floor，但估计本身不够准。
4. v0 离线 contact 通道也使用了估计地面，导致接触监督受标定误差影响。v1 已修正：**只有离线 contact 标签** 使用 annotation floor，身体坐标、轨迹输入和在线生成仍仅使用启动估计。代码保留 `startup_estimate` 选项用于复现 v0。
5. 常速度对照只外推 22 个身体关节和内部 reference；手 PCA/接触/beta 保持末帧。缺少手部模型资产时不伪造手旋转外推。

### 预测历史闭环 v0：失败记录

```bash
conda run --no-capture-output -n egorecover python -u -m run.check_closed_loop \
  --experiment exp/egorecover_engineering_v0 --output exp/egorecover_closed_loop_v0 \
  --allow-random-initializer
```

- 使用共同 **随机 E7 初始化器** 在前 20 个干净观测上生成身体历史，没有 GT 启动或中途重置；明确是缺少预训练权重时的稳定性诊断。
- 两种源 × clean/freeze_3s/drift_0p03mps 共 6 条回放均未完成：解码遇到退化 6D rotation，进程返回失败码 1。
- 追加诊断 `exp/egorecover_closed_loop_v0_diagnosis/report.json`：History/clean 在 **物理帧 63** 提交时失败。
- 当前 decoder 对退化旋转显式报错；没有裁剪输出、重置 GT 或伪造后续帧来把检查变为通过。
- 已确认 GT-history 单步结果不能迁移成稳定闭环。随机初始化器和训练/预测历史分布差异是未排除的原因；这次失败本身不构成对“有有效启动的正式方案”的否定。

### 下一次有限迭代

正在运行 v1：σ=0.3、两种 G 各 1000 步、P 200 步；修正离线 contact 标签来源。两种源内部仍使用相同训练预算/初始化/条件。v0→v1 同时改变噪声、预算和 contact 派生方式，不能把差值归因给其中任意单项。

## 2026-09-23 05:23—05:40 +08:00：阶段 D 第二轮完成、闭环诊断与最终检查

### v1 训练与同帧对照

实际执行：

```bash
cd /home/ld666/projects/EgoRecover/UEM-E7
conda run --no-capture-output -n egorecover python -u -m run.engineering_pilot \
  --output exp/egorecover_engineering_v1 --prior-steps 200 --flow-steps 1000 \
  --batch-size 32 --sigma 0.3
conda run --no-capture-output -n egorecover python -m run.compare_engineering_baselines \
  --experiment exp/egorecover_engineering_v1
```

- P 200 步；G 每种源 1000 步；Euler10；σ=0.3；离线 contact 标签使用 annotation floor。总时长 **201.52 s**。
- P 完整 clean-dev 表示损失：保持基线 0.084704 → P 0.069373；选择第 200 步并冻结。
- G 固定 dev 帧、τ=0.5、噪声条件的 loss：Gaussian 2.590847 → 0.277512；History 2.667362 → 0.220112。
- **两种 G 均随机初始化，历史来自 GT；使用官方 val 内划出的工程开发数据。没有真实 E7 预训练、预测历史训练、完整时序评估或正式 SMPL22 FK 指标。**
- 同一组 64 个开发帧、相同 H/μ、共享采样噪声、a11 的结果如下。它们是开发集的抽样诊断，不代表 24,000 个变体帧的重建精度；24,000 帧数字仅用于坐标适配检查。

| 方法 | v0 dense22 世界位置误差 mm | v1 dense22 世界位置误差 mm |
|---|---:|---:|
| 物理末帧保持 | 25.20 | 25.20 |
| 物理常速度 | **15.17** | **15.17** |
| P | 38.04 | 38.15 |
| Gaussian-source G，a11 | 400.82 | 152.77 |
| History-source G，a11 | 288.33 | 168.24 |

结论：训练损失可下降，但 **P/G 没有超过简单先验**。History 源在 v0 的优势在 v1 反转，即使其表示损失更低也没有得到更低的位置误差。现有结果不支持“历史源必然更好”或“该方案已经有效”。v0→v1 不是单因素消融，不能把改善归因于更多步数或更小 σ 中某一个因素。

v1 四动作诊断：

| 源 | a11 | a10 | a01 | a00 | 逐帧 oracle gap |
|---|---:|---:|---:|---:|---:|
| Gaussian | 152.77 | 139.96 | 153.53 | 141.15 | 14.77 mm |
| History | 168.24 | 167.12 | 168.17 | 167.07 | 10.11 mm |

oracle gap 按每帧的最优动作计算，不是“a11 均值减去最好的列均值”。这些标签只保存在 `utility_diagnostic_*.pt`；有 oracle gap 不保证 Q 在采样前能预测收益，更不保证策略闭环获益。

### v1 闭环失败：区分数值缺陷与模型发散

```bash
conda run --no-capture-output -n egorecover python -u -m run.check_closed_loop \
  --experiment exp/egorecover_engineering_v1 --output exp/egorecover_closed_loop_v1 \
  --allow-random-initializer
```

- 共同随机 E7 启动；使用开发 take `iiith_soccer_002_2`，两种源各测 clean、freeze_3s、drift_0p03mps；没有 GT 身体启动、GT beta、GT 地面输入或中途重置。
- 初次 v1 的 Gaussian 三个变体均在物理帧索引 **85** 失败，History 三个变体均在 **93** 失败。此处和各 report 的 `last_requested_frame` 都是 **从 0 开始的索引**。
- 该 take 故障从索引 **89** 开始；Gaussian 在故障之前就失败，故不能把该失败归因于错配。clean 也失败，当前没有可报告的完整故障/恢复精度或正式延迟结果。
- 进一步检查 `exp/egorecover_closed_loop_v1_numeric_diagnosis/report.json`：History/clean 在索引 93 的有限 rotation6d 绝对值达到 **2.25217×10^19**，float32 投影的 norm 平方溢出使 determinant=0，而 float64 对同一个编码投影得到 determinant=1。
- 已修复 `MotionCodec.transform_from_9d()`：仅在原投影无效时，以 float64 重算同一投影并转换回输入 dtype；真正零轴/共线轴仍报错。新增 1e20 量级有限旋转回归测试。没有裁剪输出、替换轴或用 GT 修复身体。
- 修复后重跑 History/clean：

```bash
conda run --no-capture-output -n egorecover python -u -m run.check_closed_loop \
  --experiment exp/egorecover_engineering_v1 \
  --output exp/egorecover_closed_loop_v1_projection_fix \
  --allow-random-initializer --source-modes history --variants clean
```

- 仍在物理帧索引 **98** 失败，错误为 `prior_mu has NaN/Inf at an unmasked position.`，退出码 **1**。修复解决了投影溢出的实现问题，**没有解决实际预测状态的持续发散**。保留失败状态而没有生成剩余帧。
- 随机初始化器、GT/预测历史分布差异、参考系累积和 P 在分布外的输出均需后续分离验证。当前证据不能确定各因素贡献，也不能用这个无有效预训练启动的实验否定正式方案。

### C 阶段全量适配检查

`run.check_mismatch_adapter` 已检查两个目录的全部 **120 个变体、24,000 个变体帧**：extended 36×200，pilot 84×200。目录间共享原 take，不是 24,000 个独立原始帧。

- 9D 实际观测 → 新 18D 条件 → 世界变换往返最大绝对误差：extended **2.38419e-7**，pilot **9.53674e-7**。
- extended 中 60 个视觉缺失帧、60 个轨迹缺失帧均经过可用性处理；pilot 本身没有实际缺失帧，不能据此宣称 pilot 覆盖了传感器 dropout。
- 结果 `verification/mismatch_adapter.json`：`passed=true`。这验证坐标适配，不验证身体预测或传感器标定精度。

### 固定体型 FK 接口与最终回归

- 新增 `egorecover/fk.py::FixedShapeFK`：接受调用者提供的 SMPL-X layer；使用共同启动的固定 β_boot，从预测全局关节旋转恢复局部旋转，以预测骨盆和该体型的中性 root offset 构造平移；不读取 GT 体型、root offset 或关节。
- 已用语义测试检查：固定体型、预测骨盆、父子旋转恢复、手 PCA 矩阵、SMPL layer 冻结。**模拟 layer 测试不是实际 SMPL-X 资产验证**；目前还没有正式 FK 指标或 FK 收益标签。
- 最终 `conda run -n egorecover python -m pytest tests -q --junitxml=verification/egorecover_tests.xml`：**49 passed，0 failed，0 skipped，13.43 s**。4 条警告来自原 Lightning CPU 测试（GPU 可用但该测试未用、pytree 弃用、worker 数量），无失败。
- 已格式化新增文件；保留原 E7 训练、Flow 与模型代码。实验权重位于 git 忽略的 `exp/`；未提交 git commit。

## 2026-09-23 05:40 +08:00：本轮可完成范围与下一阶段条件

**已验证的是工程可实现性：**历史条件 G、当前帧 Flow、四动作、P/Q 接口、权重迁移、坐标缓存、实际数据读取、真实 GPU 反传/采样和两轮短训可以运行。**方法有效性尚未得到验证：**当前 P/G 输给常速度，预测历史闭环发散，正式 FK 与 Q 学习尚未完成。

| 待解决问题 | 当前证据 | 后续推进顺序与验收条件 |
|---|---|---|
| 有效共同启动 | 在已搜索的 Linux 项目/home 路径未定位真实 E7 checkpoint；目前只能显式随机启动 | 提供或训练与本分支匹配的 E7 权重，先检验原模型 clean 启动和固定体型；官方 diffusion 权重不能直接替代 |
| 正式物理指标 | 没有定位 `SMPLX_NEUTRAL.npz`；只有 dense 诊断和 FK 接口语义测试 | 接入实际模型资产，验证 root offset、关节次序、坐标与 β_boot；先完成 FK 往返及公共坐标误差检查 |
| P 损失与位置表现不一致 | v1 表示损失下降，dense 位置 38.15 mm，差于常速度 15.17 mm | 用同帧 FK/旋转/位置指标选 P，分开检查通道权重、运动转折及多步累计误差；常速度作为必须保留的对照 |
| GT 历史到预测历史的分布变化 | GT 单步可运行，纯预测 clean 回放在索引 98 发散 | 在训练 take 生成合法预测历史；以此前预测 reference 编码监督，适配 P/G；固定 P 后训练两种 G；先过 clean 全长，再比较故障/恢复 |
| 参考系与地面标定 | 当前 head-height 启动估计误差最高约 0.49 m | 确定部署时可用的场景标定或单独训练估计器；分别评估地面误差和身体误差，线上仍不读 GT floor |
| Q 正式收益学习 | 已有共享噪声/动作等价接口；仅有 GT-history dense 诊断标签 | 等 P/G 稳定并冻结后，在可达预测历史上以固定 β_boot FK 生成四动作收益，记录 G/P/统计/源/σ/NFE 身份；之后训练 Q 并在 dev 闭环校准阈值 |
| 独立评估 | 已对官方 val 的工程 take 调参 | 正式训练转用训练 split；独立测试排除已用于工程调参的 take，按 take/故障强度分层报告；本轮两个 holdout 未用于模型选择或精度评估 |

上述正式阶段没有被标成完成。没有在缺少有效启动与 FK 的条件下继续扩大随机模型训练、训练伪正式 Q 或发布正向结果；这些会混淆接口检验和方法证据。下一步的优先级是 **有效 E7 启动 + 实际 FK → clean 预测历史稳定性 → 故障/恢复对照 → Q**，而非先扩充错配数据规模。

可直接查看的产物：

- [接口、环境与复跑说明](EGORECOVER.md)
- [机器可读实验汇总](verification/egorecover_engineering_summary.json)：两轮完整 report、基线、历次闭环失败、最终测试、环境与源码 SHA256。
- [实验图 PNG](verification/egorecover_engineering_summary.png) / [PDF](verification/egorecover_engineering_summary.pdf)：训练曲线与同帧基线；图中明确注明 GT 历史、随机初始化和非 FK 指标。
- [测试报告](verification/egorecover_tests.xml)、[实际数据适配检查](verification/mismatch_adapter.json)、[完整主干 GPU smoke](verification/egorecover_interface_smoke.json)。
- `exp/egorecover_engineering_v0/`、`..._v1/`：P/G 权重、diagnostic 标签、split、曲线、配置与哈希。
- `exp/egorecover_closed_loop_*/report.json`：失败版本原样保留。最终修复后报告位于 `egorecover_closed_loop_v1_projection_fix/`。

复生成汇总图表：`conda run --no-capture-output -n egorecover python -m run.report_engineering`（在 `UEM-E7/` 下）。所有本轮实验进程均已结束。

05:41 最终文件检查：Black 检查 27 个新增 Python 文件全部通过；`git diff --check` 通过，`git status` 显示本工作线只有新增文件、无已跟踪原代码修改。汇总 JSON 已生成并记录 49/0/0/0（tests/failures/errors/skipped）；PNG 已实际打开检查，PDF 同步导出。

## 2026-09-23 08:10 起：用户要求原因排查，冻结现有权重做诊断对照

新增独立脚本 `run/diagnose_engineering.py`；不训练或修改 P/G 权重、不改数据。诊断按以下因素展开：

1. 相同 64 帧的分通道 MSE、GT codec 往返、离线 GT reference/局部位置替换，以区分表示损失和物理位置误差。
2. 相同条件/噪声的 NFE=1/10/50，及 τ=0.1/0.5/1 的训练插值路径输出。
3. 相同开发 clean take，随机启动 / **离线 GT 理想启动**、学习 P / 物理保持 μ；20 帧启动之后均只使用自己的预测历史。
4. 训练帧与开发帧的单步差异；同一开发 take 的 a11/a10/a00，以及仅将内部 reference 投影回平面变换的对照。

GT 启动、GT reference/位置替换仅用于归因，不是合法部署结果。平面投影只作用于内部参考变换，不限制身体关节或相机的完整三维姿态，也不读取 GT。下方最终记录会区分“执行到 200 帧”和“身体预测稳定”，两者不能混同。

已发现的直接证据（后续结果继续补充）：

- 随机 E7 启动 20 帧平均 dense22 误差 **11,245.25 mm**；第一步轨迹条件绝对值达约 **637.5**。这不是方案所假设的有效身体启动。
- 相同 64 个开发帧，P 对比保持：局部关节旋转 MSE **0.04196→0.05283**，局部位置 **0.02696→0.03258**；参考旋转 **0.16040→0.13994**，参考平移 **0.27033→0.21233**。参考项权重 8，故总 MSE **0.08033→0.07845** 仍下降，身体位置却 **25.20→38.15 mm**。这是 loss/指标不一致的实测分解。
- 开发帧 NFE=1/10/50：Gaussian **152.60/152.77/155.89 mm**，History **160.10/168.24/186.67 mm**。增加采样步数未解决问题。
- GT 当前表示编码/解码误差约 **0.00032 mm**，自身往返不存在可解释百毫米误差的整体单位/逆变换错误；这不等于完整 SMPL FK 或所有上游表示约定都已验证。
- 训练帧上，仅把输出 reference 投影回平面变换（不读取 GT），Gaussian **154.42→37.67 mm**，History **152.26→37.46 mm**。尚需等连续回放及开发帧对照完成，才能判断它能解决多少问题。

初轮 `exp/egorecover_diagnosis_v1/` 的随机启动对照额外经历了一次物理状态编码/解码，因此其失败索引 102 与旧回放 98 不构成数值完全相同的复现；二者都迅速发散。脚本后续版本改为直接复制随机启动物理缓存，避免额外浮点扰动。旧产物保留，不覆盖。

## 2026-09-23 08:26 +08:00：原因排查完成，12 条闭环对照与单步分解

### 1. 已定位的主要可修复问题：内部 reference 没有保持平面约束

原 v4 表示和新目标构造中的内部 reference 是 **水平位置 x/y + 绕竖直轴的 heading**。它不是身体姿态，也不是相机完整姿态。GT reference 的 roll/pitch/z 恒为 0，两个 reference 的相对变换也应在同一平面变换集合内（前提是缓存/启动 reference 本身合法）。

当前 G 输出完整 rotation6d + xyz，decoder 将其当自由三维变换逐帧累乘。短训后非零 roll/pitch/z 残差不受硬约束：在 GT 理想启动的 soccer 对照中，第一步 reference 的 tilt magnitude 为 0.093，第三步达 0.451；位置误差随帧索引 20/21/22/23 从 **129/305/524/852 mm** 增长，而不是在报 NaN 的时刻才开始坏。

参考系偏斜同时影响两处：

- 将全部局部关节变换到世界时，引入身体整体旋转/平移偏差，并在后续累积。
- `encode_observation()` 使用 `inv(previous_predicted_reference) @ current_observed_reference`。此前 reference 偏斜会使**干净传感器**产生训练中未见的条件；条件过大又推动后续预测离开正常范围。

这条反馈通路得到分量跟踪和遮蔽轨迹对照的支持，但没有独立测量各反馈支路的增益，不能声称所有误差都来自单一支路。`model/core.py` 的数值行为、历史身体与 μ 的分布变化也未做单独结构消融。

**固定权重的纠正对照：**仅在每帧采样后、提交前，将输出 reference delta 投影为平面 heading 与 x/y，z=0；身体 22 个关节的三维姿态、当前观测、P/G 参数、噪声和 Euler10 均不变。不使用 GT 做这项投影。

| 开发 clean 序列 / 当前采样种子 | 原解码，GT 启动后纯预测 | 加平面 reference 投影，GT 启动后纯预测 |
|---|---|---|
| iiith_soccer_002_2 / 62 | 索引 117 非有限 μ，早已发散 | 完成至索引 199；均值 **175.12 mm**，峰值 **279.69 mm** |
| iiith_soccer_002_2 / 63 | 索引 126 非有限 μ，索引 24 已 >1 m | 完成至索引 199；均值 **174.95 mm**，峰值 **278.62 mm** |
| georgiatech_covid_06_8 / 62 | 索引 122 非有限 μ，索引 27 已 >1 m | 完成至索引 199；均值 **206.48 mm**，峰值 **275.91 mm** |

均值/峰值只计算启动后的 **180 帧**。所有对照使用 History-source G、学习 P、a11、σ=0.3、NFE10。这里的 **GT 20 帧启动是明确的离线理想条件**，不是已完成的线上初始化。两个开发 take、少量种子支持这个实现问题可重复，尚不构成统计显著性、正式独立泛化或完整方法优势。

这修正了上一阶段过于集中在“缺预训练/预测历史适配”的排查方向：**存在不依赖额外资产就能验证的参考系结构问题**。仅补上 E7 权重不能代替对这个实现选择的检查。正式修复应统一处理启动和滚动缓存；仅约束当前增量，无法保证一个已倾斜的启动 reference 自动恢复为平面。

### 2. 位置指标为何比 loss 更差：分组损失与几何约束共同作用

开发帧上，P 的总 weighted MSE 因参考项改善而下降，但局部身体旋转/位置 MSE 都变差（上节已记录具体数值）。参考项权重为 8，**不能用总 loss 代替物理位置验收**。同时官方统计将零 std 通道替换为 1；这是原 E7 的处理，新 codec 与其一致。但对 reference 原本恒为零的分量，小的归一化误差仍可能对应有明显物理影响的倾角或竖直位移。

对 G 的单步后处理对照也支持该解释：

| 同一固定 64 帧集合 | Gaussian 原解码 | Gaussian 平面 reference | History 原解码 | History 平面 reference |
|---|---:|---:|---:|---:|
| 训练 take 抽样 | 154.42 | **37.67** | 152.26 | **37.46** |
| 开发 take 抽样 | 152.77 | **108.76** | 168.24 | **109.87** |

单位均为 dense22 mm；每行内共享相同帧/条件/噪声。训练与开发是不同样本，不能将行间差值视为纯单因素因果效应，但它显示解决参考系问题后仍有明显开发误差。开发集的同帧常速度仍为 **15.17 mm**，因此修复参考系不等于已经超过简单先验。

进一步把 reference 换成 GT（仅离线诊断），开发误差仍为 Gaussian **103.74 mm**、History **106.54 mm**，说明剩余问题包括局部身体预测，并非只需把 reference 做得更准。root-relative dense 误差也约 **120 mm**，支持这一点。

### 3. 启动与训练分布问题确实存在，但不是唯一原因

- 随机 E7 启动平均误差 **11.25 m**。第一步轨迹编码 max abs 约 **637.5**，而 GT 启动下为 **1.58**；从一开始就不满足“已有有效身体历史”的方法前提。
- GT 启动只能排除初始身体错误，不能消除每一步自身预测带来的分布变化。没有 reference 约束时，GT 启动仍发散；加入约束后的三组回放保持有限且误差有界，为后续预测历史适配提供了可运行的诊断基线。
- G 参数数 **88,820,979**。本轮训练仅 8 个工程 take，共 1,440 个启动后原始目标帧；7 个故障变体共享身体目标，不能当作 7 倍独立运动数据。P 200 步、G 各 1000 步也不是充分训练的依据。
- 平面约束后的训练误差约 37 mm，开发误差约 109 mm；剩余问题应继续分离训练预算、运动覆盖、当前 full-state x0 回归、损失物理尺度和预测历史适配，不能直接把它全部归因于某一个超参数。
- 本次失败 soccer take 的地面估计误差只有 **0.0261 m**，而错误迅速到米级乃至更大。其他 take 最高约 0.49 m 的地面误差仍是风险，但不足以解释这条 clean 回放的爆炸过程。

### 4. 已测试、不能作为主要解释或修复的方向

- **加 NFE：**1→10→50 没有改善开发误差，History 甚至 160.10→168.24→186.67 mm。当前优先级不是延长积分。
- **只去掉 P：**将 μ 替换为物理保持后，原解码可以执行到 200 帧，但状态幅值达约 **1e31**，物理上早已失效。`completed=true` 只表示循环走完，绝不等于稳定或精度通过。初版诊断中的 float32 误差 norm 也溢出，JSON 记 null；后续诊断仅对度量使用 float64，模型推理仍是 float32。
- **只屏蔽头轨迹：**GT 启动下 a10 原解码可以执行完，但后 180 帧平均误差 **3378.87 mm**，末帧 **4449.10 mm**；a00 平均 **3317.54 mm**。删除反馈输入能抑制数值爆炸，却丢失世界位置纠正，不能用此替代参考系修复。
- **只修 NaN：**float64 旋转投影已经避免一种数值溢出，但 NaN 是长期误差放大后的末端症状；应看索引 20–30 的误差增长，而不是只看首次异常索引。
- **先训练 Q：**这些失败回放并未调用 Q；当前表现不能归因于 Q，也不能期待 Q 通过删除干净轨迹替代几何修复。
- **基础坐标往返：**GT 编码/解码误差约 0.00032 mm，以及既有 24,000 个变体帧的观测适配结果，未发现足以解释当前误差的整体单位/求逆问题；仍未替代实际 SMPL-X FK 验证。

### 5. 下一阶段顺序

1. 将 reference 的平面定义显式化，统一覆盖启动、缓存、增量解码、监督构造；保留原行为作为可复现对照。不能把这个约束错误应用到身体关节或真实相机姿态。
2. 在同帧 dense 世界位置/根部相对位置/参考误差上验收 P/G，并用物理指标选择 checkpoint；针对目前损失改善与身体误差退化的冲突做分组权重或可微 dense 几何损失的单因素实验。实际 FK 资产到位后再用固定 β_boot SMPL22 指标确认。
3. 接入有效共同启动，在训练 take 上收集自身预测历史并适配；先验收 clean 长回放，再测试故障与恢复。正式测试仍禁止 GT 启动/重置。
4. 在稳定 P/G 上重新生成四动作收益并训练 Q。当前问题的首要动作不是继续扩充错配数据、增大 NFE 或用 Q 遮住结构问题。

本轮 **平面投影只在诊断脚本中作为干预实现，未悄悄改变常规 rollout 的默认解码**；原始失败产物和 checkpoint 全部保留。主干/数据/权重没有被这些实验修改。正式方法可行性的结论仍待有效启动、预测历史适配、FK 与独立评估。

### 复跑与产物

全部命令在 `UEM-E7/` 下，通过 `conda run --no-capture-output -n egorecover` 调用；输出目录必须新建，以下名称已存在时需换名：

```bash
python -u -m run.diagnose_engineering --output exp/egorecover_diagnosis_v1
python -u -m run.diagnose_engineering --output exp/egorecover_diagnosis_train_v1 --probe-set none --eval-split train
python -u -m run.diagnose_engineering --output exp/egorecover_diagnosis_dev_geometry_v1 --probe-set none
python -u -m run.diagnose_engineering --output exp/egorecover_diagnosis_geometry_v1 --probe-set geometry --skip-single-step
python -u -m run.diagnose_engineering --output exp/egorecover_diagnosis_planar_take2 --probe-set planar_pair --skip-single-step --take-index 1
python -u -m run.diagnose_engineering --output exp/egorecover_diagnosis_planar_seed63 --probe-set planar_pair --skip-single-step --rollout-noise-seed 63
python -m run.report_diagnosis
```

- [原因排查汇总 JSON](verification/egorecover_cause_analysis.json)：12 条闭环干预、单步通道分解、逐帧幅值/参考偏斜/误差、checkpoint 身份和限制。
- [原因排查图 PNG](verification/egorecover_cause_analysis.png) / [PDF](verification/egorecover_cause_analysis.pdf)。
- 每个 `exp/egorecover_diagnosis*/report.json` 保留原始逐帧记录；原有 v0/v1 训练报告不覆盖。
- 两个新增诊断/报告脚本已实际执行并通过 Black 格式检查。生产代码未变，未为文档和独立诊断重复运行已通过的 49 项回归测试。
- 汇总工具已确认收录 **12 条闭环对照**；PNG 已实际打开检查，PDF 同步导出，所有本轮诊断进程已结束。

## 2026-09-23 09:31 起：按排查结果修正并验证

用户授权将诊断结论落实到代码并测试。本轮顺序为：统一 reference 约束 → 回归与旧权重真实回放 → 几何损失/物理选模短训 → 预测历史适配 → 汇总实际结果与剩余限制。

- `MotionCodec` 默认 `reference_mode="planar"`，在初始 reference、编码、增量解码、世界 reference 累积和观测参考桥接中使用同一平面定义；身体关节与相机姿态保持完整三维。`legacy_se3` 显式保留旧行为。
- `HistoryBuffer` 在建立启动缓存时即使用统一 codec，避免只修当前增量而保留倾斜启动坐标；线上保存 raw G 表示与重新编码后的 committed 表示及 reference_mode。
- 原因排查脚本显式锁定 legacy，避免默认值变更悄悄改变历史对照；旧报告的基线重算也按记录的 reference_mode，缺字段时按 legacy 处理。
- 新增 dense 世界位置几何损失（米，固定 scale=0.1 m），与原表示 MSE 相加；P 默认按完整 clean-dev dense 位置选模，G 默认按固定开发帧/噪声的 Euler10 采样位置误差选模。仍保留原损失/选模选项供复现。
- 针对性测试 **13 passed**：包含 200 次连续提交平面不变、三维身体姿态保留、重新锚定后的世界身体往返、legacy 行为及几何损失有限梯度。已处理测试中一次 requires_grad 张量转标量的提示。
- 已启动完整回归、旧 v1 权重在新常规 rollout 下的随机启动回放，以及 v2（P200/G各1000、σ=0.3、几何损失、物理选模）工程训练。有效 E7 预训练与真实 SMPL-X 资产尚未得到，所有随机启动测试仍明确标记。

09:38 进展：完整回归 **52 passed / 13.05 s**。v1 Gaussian 在修复后的常规线上路径、随机 E7 启动、无 GT 身体或重置的 clean 回放已完成，后 180 帧 dense22 均值 **182.07 mm**（峰值 640.71 mm）；freeze_3s 均值 **182.03 mm**。这比上一轮只能验证 GT 理想启动更进一步，但仍只涉及工程开发数据，不能称为预训练模型精度。

v2 的 P 按完整 clean-dev 物理指标未超过其初始保持先验（25.94 mm），因此**正确选择第 0 步的零修正 P**，没有把表示 loss 更低、身体位置更差的权重送入 G。Gaussian G 已完成，采样开发误差最优为第 800 步 **101.98 mm**；其余结果待最终汇总。训练/回放/缓存收集曾并行占用同一 GPU，回放报告中的耗时不能用于公平速度比较。

已接入训练专用预测历史缓存：仅从 8 个 train take 收集，前 20 帧 GT 仅用于训练初始化，之后使用冻结 P/G 的实际提交结果；当前监督按此前预测 reference 重新编码。Gaussian/History 两种旧策略的历史汇入共同缓存，后续源消融共享同一批 H/μ。加载器拒绝 dev/holdout take、坐标模式不一致、统计版本不一致和非有限数据。默认以 50% 概率将完整预测历史行混入训练，不允许只替换 H 而保留不对应的监督参考。

## 2026-09-23 09:47 +08:00：公共 codec 修复验收、v2 训练完成

- 最终新增缓存隔离/配对采样测试后，全套 **54 passed，13.28 s**；仍是原 Lightning 的 4 条非失败提示。
- 全部 120 个变体 / 24,000 个变体帧通过新 planar codec 的观测适配，最大误差保持 9.53674e-7；报告 `verification/mismatch_adapter_planar.json`。
- v1 的常规 `run.check_closed_loop` 使用新公共 codec，**随机原 E7 启动、实际观测输入、全程模型预测历史，无 GT 启动或重置**；以下每条均完成启动后的 180 帧：

| v1 固定权重 + planar codec | clean | freeze_3s | drift_0p03mps |
|---|---:|---:|---:|
| Gaussian，dense22 mm | 182.07 | 182.03 | 186.23 |
| History，dense22 mm | 185.42 | 185.40 | 188.18 |

目录 `exp/egorecover_closed_loop_v1_planar_production/`。同一批旧权重在原 decoder 曾无法完成回放；这里进一步验证统一修复启动和滚动缓存的作用。此表仍是单个开发 take、随机初始化器的工程诊断，不是正式 FK 或跨序列结论。

v2：`exp/egorecover_engineering_v2/`，默认 planar + geometry_weight=1、P200/G各1000、σ=0.3。P 在完整 clean-dev 上没有超过保持先验，选第 0 步；Gaussian/History G 都按物理采样指标选中第 800 步。固定 64 个开发帧：

| 方法 | dense22 mm |
|---|---:|
| 常速度 | 15.17 |
| 保持 / 选中的零修正 P | 25.20 |
| Gaussian G | 101.98 |
| History G | 110.43 |

相较旧 v1 在平面解码下约 108.76/109.87 mm，v2 并非两种源均改善；相较 v1 原始解码 152.77/168.24 mm 则都降低。v2 同时改变几何损失、P 选模、G 选模，不能将差异单独归因给某一项。

当前四动作 oracle gap 只有 Gaussian **0.139 mm**、History **0.144 mm**（仍是 GT-history 开发帧诊断）；这些数值更不能支持直接开展正式 Q 收益学习。未来需在实际可达预测状态与故障窗口分层重新评价。

v2 Gaussian 的首条 clean 纯预测回放也已完成，均值 **177.20 mm**；新报告额外记录共同随机启动身体的误差 **598.56 mm**。参考系修复把原先的米级累积破坏显著减轻，但这仍不等于已经取得有效的预训练初始化器。

## 2026-09-23 09:55 +08:00：v2 闭环回归发现、预测历史缓存完成

- v2 clean 的 History 分支回放也完成，但后 180 帧均值 **265.00 mm**，明显差于固定 v1 + planar 的 185.42 mm；Gaussian 为 177.20 mm。因此不能把几何损失与单步物理选模组合宣称为闭环精度全面改善。
- 两种分支共享同一随机 E7 启动，其前 20 帧 dense22 均值为 **598.56 mm**。用保存的相同启动身体计算后续 180 帧纯身体基线：保持 **646.55 mm**，常速度 **28,565.57 mm**。常速度把随机初始化器最后两帧的速度噪声外推了整个窗口；这些基线没有后续观测，不能据此宣称超过有效训练的观测驱动基线。结果位于 `exp/egorecover_closed_loop_v2_clean/body_baselines.json`。
- 前述 **15.17 mm** 常速度数字来自每帧获得真实历史的单步评估，与这里完全预测的连续回放不是同一协议；两个结果均保留并标明历史来源。
- 冻结 v1 的两个 G 策略已完成 **8 train takes × 3 variants × 2 sources = 48 条**训练缓存收集，共 **8,640 行**，无失败。GT 只用于 train take 的前 20 帧初始化与离线监督；之后使用模型提交状态。没有收集 dev/holdout 用于训练。
- 缓存 `exp/egorecover_predicted_train_v1/frames.pt`，SHA256 `d7d00cc3020af8151228f1b1654f719decd3593785b95f228a39791d0015ca9d`。收集时 Gaussian 三变体平均 dense22 为 109.23/109.57/112.27 mm，History 为 111.12/112.07/114.68 mm；这些是 train + 理想启动的缓存质量诊断，不是开发性能。
- v3 已按相同种子、P200/G各1000、σ=0.3、同一损失/选模运行，唯一预定训练改动为 **50% 整行预测历史混合采样**。仍按真实历史开发指标选模，需额外检查模型启动的连续闭环；不能仅凭 teacher-forced 指标推断部署行为。

## 2026-09-23 10:01 +08:00：v3 混合预测历史训练完成，闭环测试启动

- `exp/egorecover_engineering_v3/report.json` 标记完成，训练用时 **320.59 s**。8,640 行训练专用缓存通过 take、参考模式、统计 SHA256 和有限性校验，以 50% 概率整行混入；两个 G 分支共享同一训练/开发划分和噪声设定。
- P 再次按完整 clean-dev 物理指标选择**第 0 步零修正**，其基线与选中值均为 **25.94 mm**。不能将此实验解释为学到了有效历史 P。
- 相同 64 个 GT-history 开发帧上，Gaussian 第 800 步为 **117.37 mm**，History 第 1000 步为 **116.62 mm**；高于 v2 的 **101.98/110.43 mm**。相同帧物理保持 **25.20 mm**、常速度 **15.17 mm**。v3 四动作离线 oracle gap 为 **1.85/0.83 mm**，正式 Q 的收益仍未成立。
- 两个开发 take 的常规随机初始化、模型预测历史、无 GT 重置闭环已启动，分别覆盖 clean、freeze_3s、drift_0p03mps。将依据实际闭环结果而非单步指标决定 v3 是否改进。

10:04 进展：v3 的两条 **clean** 开发回放均完成 180 帧，无发散。足球 take：Gaussian **154.58 mm**、History **184.14 mm**；另一个 Covid take：Gaussian **220.95 mm**、History **237.09 mm**。足球相同启动、同样 clean 协议下 v2 为 **177.20/265.00 mm**，说明 v3 的预测历史混训虽然使 GT-history 单步指标变差，但改善了这条实际闭环。该差异支持“训练/推理历史分布错配”是原因之一；不同训练抽样/选中步数也随混训变化，且只有一个同视频 v2 对照，不能将全部差异精确归因给缓存本身。另一视频目前只有 v3 闭环，不做跨视频优劣判断。待两条视频的故障/恢复回放结束后记录完整结论。

## 2026-09-23 10:06 +08:00：修正测试收尾，停止本轮工作

v3 两个开发 take × 2 种源 × 3 种观测变体的 **12 条**常规回放全部完成：同一随机 E7 启动、线上只见当下实际观测、20 帧以后只用自己预测的身体历史，逐帧不重置，均完成启动后 180 帧并取得有限 dense22 世界位置误差。GT 仅在推理完后离线算误差。

| v3，后 180 帧平均 dense22 (mm) | Gaussian clean | Gaussian freeze | Gaussian drift | History clean | History freeze | History drift |
|---|---:|---:|---:|---:|---:|---:|
| 足球开发 take | 154.58 | 154.56 | 156.91 | 184.14 | 184.13 | 185.15 |
| Covid 开发 take | 220.95 | 221.13 | 223.43 | 237.09 | 236.96 | 238.48 |

足球/Covid 启动的前 20 帧自身误差分别为 **598.56/647.70 mm**，同启动身体的后续 180 帧无观测保持基线为 **646.55/607.40 mm**；无观测常速度为 **28,565.57/31,535.78 mm**，因为把随机初始化器的末两帧噪声速度长时外推。后两者仅是这个随机启动下的身体位置对照，不能充当有效预训练、持续观测的技术基线。相同足球 clean 协议的 v1 固定权重 + planar codec 为 Gaussian/History **182.07/185.42 mm**，v2 为 **177.20/265.00 mm**，v3 为 **154.58/184.14 mm**。v3 在这条闭环上两种源均优于 v2，History 从 v2 回退状态恢复到与 v1 接近；不能宣称跨数据集的稳定优势。

预定修正的验收：默认平面 reference 的公共 codec 阻止了旧三维 reference 的持续污染；几何损失和物理选模避免选择单步表示损失更低、身体误差更大的 P，但当前 P 实际保持第 0 步；50% 预测历史训练改善了同视频真实闭环，GT-history 单步反而变差。故单步 GT 指标不能作为闭环选择的充分依据。轨迹故障三变体差异在此随机模型/两条开发 take 上较小，并不构成故障鲁棒性的正式统计结论。

验收材料：

- `verification/egorecover_tests.xml`：**54 passed / 0 failed / 0 errors**；本轮代码最终 Black 检查 **36 个文件全部通过**。
- `verification/mismatch_adapter_planar.json`：**120 变体 / 24,000 变体帧**适配通过，最大差异 **9.53674e-7**。
- `exp/egorecover_closed_loop_v1_planar_production/`、`exp/egorecover_closed_loop_v2_clean/`、`exp/egorecover_closed_loop_v3_take0/`、`exp/egorecover_closed_loop_v3_take1/`：总计 **20/20** 条有限闭环，其中 v3 **12/12**；每条的逐帧预测、故障区间误差和报告分别保存。并行运行过 GPU 作业，latency 字段不能比较公平速度。
- `verification/egorecover_repair_summary.json`、`egorecover_repair_summary.png`、`egorecover_repair_summary.pdf`：整合源码 SHA256、模型/缓存身份、回归/适配检查、单步与闭环不同协议；图已打开检查。

**当前可行性判断**：公共坐标和因果滚动接口在这批工程数据上已可稳定执行，预测历史适配有同视频闭环改进证据；方法精度仍未成立。没有真实训练过的 E7 启动权重、实际 SMPL-X FK 资产与独立测试集准确率；P 被选为未训练的零修正，Q 尚未正式训练，两个分支的 GT-history 单步误差均高于真实历史常速度。当前数据来自官方 val 的工程划分；holdout 未被用于选择。故本轮结束于**修复与有限工程验收**，不延伸为论文指标或方法优越性声明。按用户指令，记录完毕即停止，不再开启新训练/实验。

## 2026-09-23 11:33 +08:00：离线 SMPL-X 人体评估链路接入与资产阻断

- 用户随后要求接通完整 SMPL-X 链路并直接下载模型。已核实 [SMPL-X 官方站](https://smpl-x.is.tue.mpg.de/)明确要求注册及同意许可，匿名打开 [下载页](https://smpl-x.is.tue.mpg.de/download.php) 会跳到登录页。当前 WSL 未找到 `SMPLX_NEUTRAL.npz`，无法代表用户完成站点注册/许可同意或取得授权下载。因此**实际模型资产未下载、未生成实际 SMPL-X 指标**；没有用来路不明的镜像文件替代。
- 新增 `egorecover/smpl_evaluation.py` 和 `run/evaluate_closed_loop_smplx.py`：从已完成的预测历史回放恢复每帧身体，用模型自身启动 `beta_boot` 固定体型；运行 SMPL-X 得到 55 关节和 mesh 顶点；从干净原始序列离线生成 GT，先用前 22 关节审计模型文件与标注坐标是否匹配，再计算身体/手部 MPJPE 与 PA、头部旋转/平移、根位移、足部滑动/穿透/腾空/接触。按整体、故障前、故障中和恢复期分别报告；TMR/FID 与论文正式采样协议仍不在这个工程入口内。
- 对真实 v3 两个开发 take 的 **12 条、2160 帧**已保存预测轨迹完成无资产预检：预测状态与保存 dense 世界关节最大往返差 **9.53674e-7 m**；重算 dense22 与原闭环报告最大差 **3.05176e-5 mm**。结果 `verification/smplx_prediction_preflight.json` 明确标注 `actual_smplx_geometry_evaluated=false`。
- 新增模拟人体 layer 的正反测试，验证固定启动体型、预测 FK、身体/手部误差、GT 资产不匹配和预测轨迹篡改拒绝。完整测试 **56 passed**，Black 检查通过；缺模型 CLI 以退出码 2 明确报错且不写指标文件。模拟层不能证明真实 SMPL-X 模型与 EE4D 标注一致。
- 一旦获得官方授权文件，放入 `body_models/smplx/SMPLX_NEUTRAL.npz`，在 `egorecover` 环境从仓库根目录运行 `python -m run.evaluate_closed_loop_smplx --rollout exp/egorecover_closed_loop_v3_take0 --output exp/egorecover_closed_loop_v3_take0/smplx_geometry.json --smplx-dir body_models/smplx --device cuda`。这会先审计实际资产；若审计失败，须查版本/关节顺序/坐标与 PCA 基底，不能强行发表指标。详细说明见 `EGORECOVER.md`。

## 2026-09-23 11:42 +08:00：授权模型到位，真实 SMPL-X 几何评估完成

- 用户已将 `SMPLX_NEUTRAL.npz` 放入本地 `body_models/smplx/`。随包 `version.txt` 写明 **Version 1.0**，故更正此前代码/说明中未经资产核对的“v1.1”字样。模型 SHA256 为 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`；仓库仍忽略整个 `body_models/`，不上传许可资产。
- 先用真实模型重建原始 EE4D 的 55 个身体/手部 GT 关节。足球/Covid 开发 take 的平均差分别为 **0.00088/0.00064 mm**、最大差 **0.0485/0.0925 mm**，远低于预设 5/20 mm 审计阈值。身体和手部一起审计，防止错误手 PCA 基底静默污染手指标。
- 两个开发 take × Gaussian/History × clean/freeze/drift = **12 条、2160 帧**保存的模型预测历史全部通过真实 SMPL-X 前向、mesh 输出、完整指标计算和原 dense22 报告复核。无 GT 身体/体型/地面进入预测或 FK；GT 只用于推理结束后的审计和误差/足地面指标。

| SMPL22 MPJPE / PA-MPJPE (mm)，后 180 帧 | Gaussian clean | Gaussian freeze | Gaussian drift | History clean | History freeze | History drift |
|---|---:|---:|---:|---:|---:|---:|
| 足球 | 172.38 / 75.79 | 172.33 / 75.77 | 173.47 / 75.54 | 196.88 / 98.70 | 196.84 / 98.68 | 197.57 / 98.65 |
| Covid | 261.24 / 106.80 | 261.55 / 106.41 | 262.80 / 106.85 | 256.81 / 133.72 | 256.81 / 133.62 | 257.99 / 133.68 |

- 完整结果（55 关节/身体/手部 MPJPE 与 PA、头旋转/位移、root、足滑/穿透/腾空/接触、三阶段分段值、审计和文件哈希）见 `exp/egorecover_closed_loop_v3_take0/smplx_geometry.json`、`exp/egorecover_closed_loop_v3_take1/smplx_geometry.json` 和 `verification/smplx_geometry_summary.json`。同一足球/Covid clean Gaussian 的旧 dense22 为 **154.58/220.95 mm**，真实固定体型 FK 后是 **172.38/261.24 mm**，不能混称。
- clean Gaussian 的固定体型 FK 与直接 dense22 的平均关节差是 **70.77/89.06 mm**，但根关节差仅约 **0.00006/0 mm**。在离线诊断中保持相同预测旋转、改用每帧预测 β 后，差降至 **47.51/43.49 mm**：固定 β 带来部分差异，其余说明直接关节与旋转/骨架预测也不完全自洽。该逐帧 β 只用于问题定位，**正式指标仍严格固定启动 β_boot**。
- 环境 `egorecover` 的完整回归 **56 passed**，Black 检查通过。尚无预训练 E7 初始化器、独立测试集或论文 80 帧协议；TMR/FID 未计算。上表是授权真实模型上的工程开发数据人体指标，不能宣称论文可比精度或方法优越性。下一阶段应针对预测姿态与直接关节的不自洽训练/选模，并在有效初始化与独立测试上重评；实际 SMPL-X 链路本身已接通。
