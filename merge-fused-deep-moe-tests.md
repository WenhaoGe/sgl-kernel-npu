# 合并 A3/A5 FusedDeepMoe 测试文件

## Story名称

合并 A3/A5 FusedDeepMoe 测试入口并清理无效代码

## Story描述

当前 A3 和 A5 的 FusedDeepMoe 测试分别位于 `test_fused_deep_moe.py` 和 `test_fused_deep_moe_a5.py`。两个文件各自维护命令行入口、进程启动、权重构造、参考实现、正确性检查和性能统计，测试入口分散，且存在未被调用的历史代码。

本 Story 计划将 A5 实现合并到 `test_fused_deep_moe.py`，通过显式 `--platform a3|a5` 选择测试平台，默认使用 `a3`。合并后删除原 A5 文件，并同步修改 CI、跨文件 import 和相关文档引用。

本次只统一测试文件和入口，不统一 A3/A5 的算法实现。两个平台原有的输入形式、通信 Buffer、量化方式、权重布局、baseline、精度阈值和性能统计行为均保持不变，也不顺带调整 SiTU 的性能测试范围。
   
## 功能点分解

### 功能点名称

1. 统一 A3/A5 命令行入口
2. 合并 A3/A5 测试实现
3. 清理未调用代码
4. 兼容跨文件依赖
5. 更新 CI 与文档引用

### 功能点描述

1. **统一 A3/A5 命令行入口**：增加 `--platform a3|a5`，默认 `a3`；按平台保留原参数默认值、专属参数和校验规则，最终分别启动 A3 `test_loop` 或 A5 `run_rank`。
2. **合并 A3/A5 测试实现**：将 A5 的常量、输入与权重构造、small-op baseline、融合调用、正确性检查、profiling 和统计函数并入 A3 文件，平台专属逻辑仍分节独立维护。A5 dtype 在实际使用时解析，模块导入阶段不访问 MXFP 专属属性；增加“缺少 MXFP 属性时仍能导入 A3 公共接口”的检查。
3. **清理未调用代码**：删除已确认没有直接或间接调用的五个函数，并清理合并后不再使用的 import；条件调用、函数对象间接调用和跨文件导入的函数不在清理范围内。
4. **兼容跨文件依赖**：保留 `init_base_weights`、`init_fused_weights_int8` 和 `run_small_op_baseline` 的名称及签名；将 MegaMoe 对 A5 文件的 import 改为从合并文件导入。
5. **更新 CI 与文档引用**：A3 命令通过默认平台保持兼容；A5 命令切换到合并文件并增加 `--platform a5`，同时更新 README 和相关设计文档中的文件路径。

## 功能实现思路

1. **先识别平台，再构造参数解析器**  
   使用轻量预解析器读取 `--platform`，再调用 `build_parser(platform)` 构造对应平台的完整 parser。这样既保留两套默认值，也使 `--platform a5 --help` 只展示有效的 A5 参数。

2. **保留平台独立执行链路**  
   A3 继续执行 `test_loop → test → baseline_test/fused_deep_moe`；A5 继续执行 `run_rank → run_small_op_baseline/run_buffer_fused`。不为了代码复用而强行统一权重、baseline 或 profiler。

3. **按静态调用关系清理死代码**  
   只删除确认未被调用的函数。`make_random_profile_cases` 和 `make_fixed_profile_cases` 虽通过 `case_builder` 间接调用，但仍是有效代码，必须保留。

4. **先迁移引用，再删除旧文件**  
   保留其他测试依赖的公共函数名称和签名，先更新 `test_mega_moe.py`、CI 和文档，再删除 `test_fused_deep_moe_a5.py`，避免产生临时不可用的引用。

5. **保持现有测试行为不变**  
   不修改量化模式、数值阈值、profiler 事件匹配和输出校验。A5 SiTU 仍在普通模式完成正确性检查后返回，`--event-timing` 时仍只进行 fused 单独计时。

## 功能实现设计

### 代码设计

#### 1. 合并后的模块结构

文件：`tests/python/deepep/test_fused_deep_moe.py`

```text
公共 import 与公共配置
A3 权重、路由、baseline、test、test_loop
A5 量化配置、输入与权重、small-op baseline、fused 调用
A5 正确性检查、profiling、统计和 run_rank
build_parser(platform)
main()
```

保留以下跨文件公共接口：

- `init_base_weights(...)`
- `init_fused_weights_int8(...)`
- `run_small_op_baseline(...)`

导入兼容性检查：模拟 `torch` 或 `torch_npu` 缺少 MXFP dtype 属性的环境，确认仍能从合并文件导入 `init_base_weights` 和 `init_fused_weights_int8`，且导入过程不触发 A5 dtype 解析。实际执行 A5 量化路径时再检查所需 dtype 是否可用。

#### 2. 统一入口设计

```text
main()
  → 预解析 --platform，默认 a3
  → build_parser(platform)
      → 注册该平台的公共参数、专属参数和原默认值
  → 执行平台专属参数归一化与校验
  → a3：spawn(test_loop, ...)
  → a5：spawn(run_rank, ...)
```

参数设计约束：

- A3 的 `--debug` 继续写入 `args.debug`。
- A5 的 `--dump-profile-events/--debug` 继续写入 `args.dump_profile_events`。
- A5 的 `no_linear_beta`、SiTU 参数、shape、进程数和性能参数校验只在 A5 分支执行。
- 不接受与当前平台无关的专属参数，避免错误配置被静默忽略。

#### 3. 死代码清理设计

| 原文件 | 删除函数 | 删除依据 |
|---|---|---|
| A3 | `make_uniform_topk_idx` | 当前无直接、间接或跨文件调用 |
| A3 | `from_inclusive_prefix_sum` | 当前无直接、间接或跨文件调用 |
| A3 | `str_to_bool` | 当前无直接、间接或跨文件调用 |
| A5 | `run_buffer_fused_with_burn_in` | 当前无直接、间接或跨文件调用 |
| A5 | `format_triplet` | 当前无直接、间接或跨文件调用 |

合并后再检查并删除仅被上述函数使用的 import，不扩大为通用重构。

#### 4. 文件改造设计

| 文件 | 设计修改 |
|---|---|
| `tests/python/deepep/test_fused_deep_moe.py` | 合并 A5 实现，增加统一 parser 和 `main()`，清理死代码 |
| `tests/python/deepep/test_fused_deep_moe_a5.py` | 引用迁移完成后删除 |
| `tests/python/deepep/test_mega_moe.py` | 将 `run_small_op_baseline` 改为从 `test_fused_deep_moe` 导入 |
| `tests/python/deepep/test_normal_and_fused_deep_moe.py` | 不修改，继续依赖合并文件中的原 A3 公共函数 |
| `.github/workflows/pr-test-deepep-npu.yml` | A3 命令不变；A5 命令改用合并文件并增加 `--platform a5` |
| `scripts/generalization_test_fused_deep_moe.sh` | 不修改，利用默认 `a3` 保持兼容 |
| `python/deep_ep/README.md` | 更新 A5 测试文件和平台参数说明 |
| `docs/design_a5_megamoe_param_adaptation.md` | 更新旧 A5 文件路径及调用示例 |

#### 5. 行为边界

- 最终只保留一个物理测试文件，不增加 A5 兼容 wrapper。
- 使用显式平台参数，不增加设备自动探测。
- 不统一 A3/A5 的通信 Buffer 创建方式、权重布局、量化参数和第二返回值语义。
- 不修改任何精度阈值、性能事件模式或 SiTU 覆盖范围。
- 只修改当前工作树根目录文件，不处理嵌套镜像目录中的同名副本。
