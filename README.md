# Pangu-FSDS:pangu Fast–Slow think for Dynamic Scheduling

中文 | [English](README_EN.md)

## 1. 简介

在智能制造不断发展的过程中，作业车间调度面临动态决策问题。面对任务插入、订单取消、设备维护等不确定事件，传统方法往往响应滞后、缺乏解释性。盘古大模型的“快思考—慢思考”机制为动态调度提供了新的解决路径，快思考负责快速给出可行决策，慢思考则对调度问题进行推理与解释，并调用合适的调度算法进行优化，从而实现高效调度。
本项目基于昇腾AI算力，对盘古Embedded-7B模型进行微调，构建融合快思考与慢思考机制的智能制造调度智能体系，实现智能制造动态调度智能体原型系统。

## 2. 模型架构

|                               |   openPangu-Embedded-7B-V1.1   |
| :---------------------------: | :----------------: |
|       **Architecture**        |       Dense        |
|     **Parameters (Non-Embedding)**     |         7B         |
|     **Number of Layers**      |         34         |
|     **Hidden Dimension**      |       12800        |
|    **Attention Mechanism**    |     GQA      |
| **Number of Attention Heads** | 32 for Q，8 for KV |
|      **Vocabulary Size**      |        153k        |
|      **Context Length (Natively)**       |        32k         |
|    **Pretraining Tokens**     |        25T         |

## 3. 微调和部署

### 3.1 环境准备

##### 硬件规格

Atlas 800T A2 (64GB)，驱动与固件安装包获取请参照 [[Atlas 800T A2](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.2.RC1.alpha003&driver=Ascend+HDK+25.0.RC1)]。

##### 软件环境

- 操作系统：Linux（推荐 openEuler>=24.03）
- CANN==8.1.RC1，安装准备及流程请参照 [[CANN Install]](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)
- python==3.10
- torch==2.1.0
- torch-npu==2.1.0.post12
- transformers==4.53.2
- peft==0.17.1

以上软件配套经过验证，理论可以支持更高版本，如有疑问，可以提交 issue。

### 3.2 微调

> 在 `lora` 框架上进行微调，运行前请修改train.py，添加原始模型路径，数据集路径，Lora权重输出路径。

```bash
cd train
python train.py   --use_card_num 8   --batch_size 8    --epochs 5   --lr 5e-5   --grad_acc_steps 4   --lora_r 16   --lora_alpha 32   --lora_dropout 0.1
```

### 3.3 推理

> 在 `transformers` 框架上进行推理，运行前请修改 infer.py，添加原始模型路径，微调后的Lora权重路径，测试数据集路径。

```bash
cd infer
python infer.py
```

Pangu-FSDS 模型默认为慢思考模式，可以通过以下手段切换至快慢自适应切换/快思考模式：

- 在代码实例`infer.py`中，`auto_thinking_prompt`与`no_thinking_prompt`变量的定义展示了切换至快慢自适应或快思考模式的具体实现：通过在用户输入末尾添加`/auto_think`或`/no_think`标记，可将当前轮次切换至快慢自适应切换/快思考模式。

## 4. 模型许可证

除文件中对开源许可证另有约定外，Pangu-FSDS 模型根据 OPENPANGU MODEL LICENSE AGREEMENT VERSION 1.0 授权，旨在允许使用并促进人工智能技术的进一步发展。

## 5. 免责声明

由于 Pangu-FSDS（“模型”）所依赖的技术固有的技术限制，以及人工智能生成的内容是由盘古自动生成的，本公司无法对以下事项做出任何保证：

- 尽管该模型的输出由 AI 算法生成，但不能排除某些信息可能存在缺陷、不合理或引起不适的可能性，生成的内容不代表本公司的态度或立场；
- 无法保证该模型 100% 准确、可靠、功能齐全、及时、安全、无错误、不间断、持续稳定或无任何故障；
- 该模型的输出内容不构成任何建议或决策，也不保证生成的内容的真实性、完整性、准确性、及时性、合法性、功能性或实用性。生成的内容不能替代工业、医疗、法律等领域的专业人士回答您的问题。生成的内容仅供参考，不代本公司的任何态度、立场或观点。您需要根据实际情况做出独立判断，本公司不承担任何责任。
