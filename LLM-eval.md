# LLM 评估标准
LLM的评价指标大致可以分为两大类别：自动评价，人工评价。自动评价方法基于计算机算法和可量化的指标，能够快速且高效地评估LLM模型的性能。而人工评价则侧重于人类专家的主观判断和质量评估，能够提供更深入、细致的分析和意见。由
于人工评价成本高且不易于实施，这里我们仅介绍自动评价相关内容。关于自动评价，也会从基础评测任务，高级能力评估，公开基准以及指令跟随这四个方面介绍。
## 基础评测任务 
基础评测任务即为评测自然语言处理中的基础任务。共三大类，分别为：语言生成，知识利用与复杂推理。
### 语言生成
根据任务定义，现有语言生成的任务主要可以分为语言建模、条件文本生成和代码合成任务。需要注意的是，代码合成不是典型的自然语言处理任务，但可以直接地用（经过代码数据训练的）LLM 以类似自然语言文本生成的方法解决，因此也纳入
讨论范围。
#### 语言建模： 
语言建模是 LLM 的基本能力，旨在基于前一个token 预测下一个 token，主要关注基本的语言理解和生成能力。典型的语言建模数据集包括 Penn Treebank 、WikiText-103和 Pile，其中困惑度（perplexity）指标通常用于评估零样本情况下模型的性能。相关研究表明， LLM 在这些评估数据集上相较于之前效果最好的方法带来了实质性的性能提升。为了更好地测试文本的长程依赖的建模能力， LAMBADA 数据集要求 LLM 基于一段上下文来预测句子的最后一个单词。然后使用预测的最后一个单词的准确性和困惑度来评估 LLM 性能。量化评价指标：困惑度（perplexity）。
#### 条件文本生成： 
作为语言生成中的一个重要话题，条件文本生成旨在基于给定的条件生成满足特定任务需求的文本，通常包括机器翻译、文本摘要和问答系统 等。为了衡量生成文本的质量，通常使用自动化指标（如准确率、 BLEU 和 ROUGE）和人类评分来评估性能。人类评估本文暂且不讨论。由于 LLM 具有强大的语言生成能力，它们在现有的数据集上取得了显著的性能，甚至超过了人类（在测试数据集上的）表现。例如，当仅给出 32 个示例作为输入时， GPT-3 通过 ICL （上下文学习）能够在 SuperGLUE 的平均得分上超过使用完整数据微调的 BERT-Large；在 MMLU 指标上，一个 5-样本的Chinchilla的准确率几乎是人类平均准确率的两倍；而在5-样本的设定下， GPT-4取得了当前最优秀的性能，平均准确率比之前的最佳模型提高了超过 10%。于是，人们开始关注现有的条件文本生成任务，能否很好地评估和反映 LLM的能力。考虑到这个问题，研究人员试图通过收集目前无法解决的任务（即 LLM 无法取得良好表现的任务）或创建更具挑战性的任务（例如超长文本生成）来制定新的评估基准，
例如 BIG-bench Hard。此外，最近的研究还发现自动化指标可能会低估 LLM 的生成质量。在 OpenDialKG 中， ChatGPT 在 BLEU 和 ROUGE-L 指标上表现不如微调的 GPT-2，但在人类评分中获得了更多的好评。量化评价指标：准确率、 BLEU 和 ROUGE。
#### 代码生成：
除了生成高质量的自然语言外，现有的 LLM 还表现出强大的生成形式语言的能力，尤其是满足特定条件的计算机程序（即代码），这种能力被称为代码生成。与自然语言生成不同，由于生成的代码可以直接用相应的编译器或解释器执行，现有的工作主要通过计算测试用例的通过率（即 pass@k）来评估 LLM 生成的代码的质量。量化评价指标：pass@k，APPS。
### 知识利用
知识利用是一种智能系统基于事实证据的支撑，完成知识密集型任务的重要能力（例如常识问题回答和事实补全）。具体而言，它要求 LLM 适当地利用来自预训练语料库的丰富事实知识，或在必要的时候检索外部数据。特别地，问答和知识补全已经成为评估这一能力的两种常用任务。根据测试任务（问答或知识补全）和评估设定（有或没有外部资源），我们将现有的知识利用任务分为三种类型，即闭卷问答，开卷问答和知识补全。
#### 闭卷问答：
闭卷问答任务测试 LLM 从预训练语料库中习得的事实知识。 LLM 只能基于给定的上下文回答问题，而不能使用外部资源。为了评估这一能力，可以利用几个数据集，包括 Natural Questions、 Web Questions和TriviaQA。量化评价指标：准确率。
#### 开卷问答：
与闭卷问答不同，在开卷问答任务中， LLM 可以从外部知识库或文档集合中提取有用的证据，然后基于提取的证据回答问题。典型的开卷问答数据集（例如， NaturalQuestions、 OpenBookQA和 SQuAD）与闭卷问答数据集有所重叠，但是前者包含外部数据源，例如维基百科。在开卷问答任务中，量化评价指标：准确率和 F1-score。
### 知识补全
在知识补全任务中， LLM（在某种程度上）可以被视为一个知识库，补全或预测知识单元（例如知识三元组）的缺失部分。这种任务可以探索和评估 LLM 从预训练数据中学习到的知识的数量和种类。现有的知识补全任务可以粗略地分为知识图谱补全任务（例如 FB15k-237和WN18RR和事实补全任务（例如， WikiFact]），分别旨在补全知识图谱中的三元组和有关特定事实的句子。量化评价指标：MeanRank，Hits，MRR。MeanRank：通过评分函数 来计算三元组的得分（比如真实性），从实体集合中取出部分实体代替原有的三元组中的尾实体（或头实体），并对替换后的三元组计算得分，分越底排名越靠前，再对所有与测试集相符的排名次数求和取均值就为MeanRank，值越低表示模型性能越好。Hits：与MeanRank类似，通过相同的评分函数来计算排名，Hits的值越高说明符合排名内的三元组可能越多，模型效果越好。MRR（Mean Reciprocal Rank）：MRR是对文本的匹配顺序排名，在匹配文本中查找的n个元素里正确元素的排名记为k，其对应的积分为1/k。
### 复杂推理
复杂推理是指理解和利用相关的证据或逻辑来推导结论或做出决策的能力。根据推理过程中涉及的逻辑和证据类型，我们考虑将现有的评估任务分为三个主要类别，即知识推理、符号推理和数学推理。
#### 知识推理
知识推理任务依赖于逻辑关系和事实知识的证据来回答给定的问题。现有的工作主要使用特定的数据集来评估相应类型的知识推理能力，例如 CSQA/StrategyQA用于常识推理， ScienceQA用于科学知识推理。量化评价指标：准确率，BLEU。
#### 符号推理
符号推理任务主要关注于在形式化规则设定中操作符号以实现某些特定目标，且这些操作和规则可能在 LLM 预训练期间从未被看到过。现有的工作通常用尾字母拼接和硬币反转任务来评估 LLM，其中用于评测的数据与上下文例子有相同的推理步骤（称为领域内测试）或更多步骤（称为领域外测试）。比如一个领域外测试的例子，LLM 在上下文例子中看到的示例只有两个单词，但在测试中LLM 需要将三个或更多的单词的最后一个字母进行拼接。通常会采用生成符号的准确性来评估 LLM 在些任务上的性能。量化评价指标：准确率。
#### 数学推理
数学推理任务需要全面利用数学知识、逻辑和计算来解决问题或生成证明陈述。现有的数学推理任务主要可以分为数学问题求解和自动定理证明两大类。对于数学问题求解任务，SVAMP、GSM8k 和MATH 数据集通常用于评估，LLM需要生成准确的具体数字或方程来回答数学问题。量化评价指标：准确率。
## 高级能力评估
除了上述基本评测任务外， LLM 还展现出一些需要特殊考虑的高级能力。我们只讨论一种有代表性的高级能力及其相应的评测方法，即与人类对齐。
### 与人类对齐
与人类对齐（human alignment）指的是让 LLM 能够很好地符合人类的价值和需求，这是在现实世界应用中广泛使用 LLM的关键能力。为了评估这种能力，现有的研究考虑了多个人类对齐的标准，例如有益性、真实性和安全性。对于有益性和真实性，可以利用对抗性问答任务（例如 TruthfulQA）来检查 LLM 在检测文本中可能的虚假性方面的能力。此外，有害性也可以通过若干现有的基准测试来评估，例如CrowS-Pairs（准确率）和 Winogender（准确率）。量化评价指标：交叉熵，准确率，困惑度。TruthfulQA数据集自动化评估方式：对于真实性和信息性，分别finetune两个models(GPT-3-6B)，finetune方法为：1. 构建三元组训练数据(question, answer, label)label表示该answer是否正确。对于信息性来说label就是包含信息量的分数（informative就为1，uninformative就为0）。2. 输入question和answer，loss就是交叉熵。
## 公开基准
我们将介绍几个具有代表性并得到广泛使用的评测基准，即 MMLU、 BIG-bench,C-Eval 和 HELM。
### MMLU
MMLU是一个通用评测基准，用于大规模评测LLM 的多任务知识理解能力。其涉及到的知识涵盖数学，计算机科学以及人文和社会科学等领域，并包含从基础到进阶不同难度的任务。量化评价指标：准确率。
### BIG_bench
BIG-bench是一个由社区协作收集的评测基准，旨在从各个方面探究现有 LLM 的能力。它包含了 204 个任务，主题包括语言学、儿童发展、数学、常识推理、生物学、物理学、社会偏见、软件开发等等。量化评价指标：bleu, bleurt,rouge and exact_str_match。
### C-Eval
C-Eval由上海交通大学，清华大学，爱丁堡大学共同完成，是构造了一个覆盖人文，社科，理工，其他专业四个大方向，52 个学科（微积分，线代 …），从中学到大学研究生以及职业考试，一共 13948 道题目的中文知识和推理型测试集。量化评价指标：正确性、流畅性、信息量、逻辑性和无害性。
### HELM
HELM不专注于特定的任务和评估指标，而是提供了对LLMs的全面评估。它评估语言模型在各个方面，如语言理解、生成、连贯性、上下文敏感性、常识推理和领域特定知识。HELM的目标是全面评估语言模型在不同任务和领域的性能。量化评价指标：
准确性，校准性，鲁棒性，公平性，偏差，毒性，效率。
## 指令跟随
现有的这些评价基准，都偏向泛用性评价，评估的是整体的、通用性的能力，但这些能力，往往并不能直接转换为生产能力，并且不适合在生产链路中集成。比如 Chatbot Athena ，是通过让人来比较大语言模型的结果优劣，并对他们进行排序，进而评估出最优模型。但如果我们在生产中需要的并不是排序类的诉求，而是需要有一个准确回答的需求，那么 Chatbot Athena 并无法满足。特别是我们还需要根据客户的需要定制大模型。因此， LLM 在工业生产中真正被需要评估的特性是指令跟随。下面是一个指令微调评价数据集例子。
```python
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n伟光正"}], "ideal": ["From the idiomatic phrase 'the great, glorious and correct Chinese Communist Party', it can also refer to a person associated with the Chinese Communist Party."]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n赵家人"}], "ideal": ["From Lu Xun's famous middle-grade novel 'A Q Zhengzhuan', it generally refers to the powerful and noble class of the Chinese Communist Party. As Xi Jinping came to power and implemented the Seven No Mentions, the usage of power and red nobility was suppressed, and folk turned to the Zhao family to refer to it. Derivations include calling the People's Republic of China 'Zhao' and Xi Jinping, the current General Secretary of the CPC Central Committee, 'King Zhao', or replacing the word 'people' with the word 'Zhao family' in the names of various Chinese organs and media propaganda"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n改开党/特色党"}], "ideal": ["The term 'Mao Left' is commonly used by the civil left and Maoist supporters, which originated from Deng Xiaoping's 'reform and opening up' and 'socialism with Chinese characteristics'. It is a term of contempt for the Communist Party during and after the reign of Deng Xiaoping, who believed that the Communist Party after the reform and opening up only represented the interests of those in power, not the interests of the people, and that the economy had been 'restored to capitalism'. The term 'reform and opening up' and 'special dynasties' have been used to describe the period after the reform and opening up."]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n黄丝/黄尸"}], "ideal": ["The term refers to non-establishment camps such as the pro-democracy camp and the local camp in Hong Kong, as well as those who support their stance, and is named after the yellow ribbon used as a symbol by non-establishment camps during the 2014 occupation. Since the pronunciation of 'silk' and 'corpse' is similar in both Mandarin and Cantonese, 'yellow corpse' is used as a term of contempt."]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n蟹堡王"}], "ideal": ["The term refers to the Hong Kong pro-establishment camp, it is often accused of not having a political stance and just being in line with Beijing"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\nww"}], "ideal": ["The term refers to mainland Chinese netizens to refer to Taiwan or the Republic of China (Taiwan period) (from the superimposed style, a neutral term). In January 2022, Taiwan Affairs Office spokesperson Zhu Fenglian said that the word Wanwan is a nickname for the Taiwanese people 'Mengmeng' by the Chinese mainlanders"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n黄俄"}], "ideal": ["The term refers to pro-Russian netizens in the Russian invasion of Ukraine in 2022, named after the Yellow Russia Project."]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n乌贼"}], "ideal": ["The term refers to Pro-Ukrainian netizens make disparaging claims about Russian invasion of Ukraine in 2022"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n冻蒜"}], "ideal": ["The term refers to the hope that someone will be elected to a certain position (but in mainland China is mostly used for flirtation), such as 'Tsai Ing-wen freeze garlic'"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\nBF"}], "ideal": ["The term refers to boy friend"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n剁手党"}], "ideal": ["The term refers to people who indulge in online shopping. Also known as 'chopper', 'chopper gang', etc."]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n蛤丝"}], "ideal": ["The term refers to those who admire Jiang Zemin"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n噬元兽"}], "ideal": ["The term refers to cats, especially orange cats, from 'Captain Marvel'"]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n蓝厂"}], "ideal": ["The term refers to the Chinese communications technology company Vivo, named after its blue logo."]}
{"input": [{"role": "user", "content": "请解释下面词语的意思,请使用英文回答。\n---\n最强法务部"}], "ideal": ["The term refers to the companies and their legal departments are considered by enthusiasts to be those who are keen to initiate copyright lawsuits, etc. and often win"]}
```

## 评价数据集链接
| Benchmark   | Focus                              | Domain                   | Evaluation Criteria                           |
|-------------|------------------------------------|--------------------------|-----------------------------------------------|
| SOCKET [[paper](https://arxiv.org/abs/2305.14938)]      | Social knowledge                        | Specific downstream task | Social language understanding           |
| MME [[paper](https://arxiv.org/abs/2306.13394)]      | Multimodal LLMs                        | Multi-modal task | Ability of perception and cognition           |
| Xiezhi [[paper](https://arxiv.org/abs/2306.05783)][[GitHub](https://github.com/MikeGu721/XiezhiBenchmark)]      | Comprehensive domain knowledge   | General language task | Overall performance across multiple benchmarks   |
| Choice-75 [[paper](https://arxiv.org/abs/2309.11737)][[GitHub](https://github.com/JoeyHou/branching)]  | Script learning | Specific downstream task | Overall performance of LLMs |
| CUAD [[paper](https://arxiv.org/abs/2103.06268)] | Legal contract review | Specific downstream task | Legal contract understanding |
| TRUSTGPT [[paper](https://arxiv.org/abs/2306.11507)] | Ethic | Specific downstream task | Toxicity, bias, and value-alignment |
| MMLU [[paper](https://arxiv.org/abs/2009.03300)]      | Text models                        | General language task | Multitask accuracy           |
| MATH [[paper](https://arxiv.org/abs/2103.03874)] | Mathematical problem  | Specific downstream task | Mathematical ability |
| APPS [[paper](https://arxiv.org/abs/2105.09938)]|Coding challenge competence | Specific downstream task | Code generation ability|
| CELLO[[paper](https://arxiv.org/abs/2309.09150)][[GitHub](https://github.com/Abbey4799/CELLO)] |Complex instructions | Specific downstream task | Count limit, answer format, task-prescribed phrases and input-dependent query|
| C-Eval [[paper](https://arxiv.org/abs/2305.08322)][[GitHub](https://github.com/SJTU-LIT/ceval)]      | Chinese evaluation                 | General language task | 52 Exams in a Chinese context   |
| EmotionBench [[paper](https://arxiv.org/abs/2308.03656)]      | Empathy ability                 | Specific downstream task | Emotional changes   |
| OpenLLM [[Link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)] | Chatbots          | General language task | Leaderboard rankings   |
| DynaBench [[paper](https://arxiv.org/abs/2104.14337)]   | Dynamic evaluation                 | General language task    | NLI, QA, sentiment, and hate speech               |
| Chatbot Arena [[Link](https://lmsys.org/blog/2023-05-03-arena/)]  | Chat assistants      | General language task    | Crowdsourcing and Elo rating system              |
| AlpacaEval [[GitHub](https://github.com/tatsu-lab/alpaca_eval)]  | Automated evaluation               | General language task    | Metrics, robustness, and diversity       |
| CMMLU [[paper](https://arxiv.org/abs/2306.09212)][[GitHub](https://github.com/haonan-li/CMMLU)] | Chinese multi-tasking               | Specific downstream task    | Multi-task language understanding capabilities|
| HELM [[paper](https://arxiv.org/abs/2211.09110)][[Link](https://crfm.stanford.edu/helm/latest/)]        | Holistic evaluation           | General language task    | Multi-metric                         |
| API-Bank [[paper](https://arxiv.org/abs/2304.08244)]    | Tool-augmented                     | Specific downstream task | API call, response, and planning                                       |
| M3KE [[paper](https://arxiv.org/abs/2305.10263)]    | Multi-task  | Specific downstream task | Multi-task accuracy                                       |
| MMBench [[paper](https://arxiv.org/abs/2307.06281)][[GitHub](https://github.com/open-compass/MMBench)]    | Large vision-language models(LVLMs) |  Multi-modal task | Multifaceted capabilities of VLMs        |
| SEED-Bench [[paper](https://arxiv.org/abs/2307.16125)][[GitHub](https://github.com/AILab-CVC/SEED-Bench)]    | Multi-modal Large Language Models |  Multi-modal task | Generative understanding of MLLMs |
| ARB [[paper](https://arxiv.org/abs/2307.13692)]  | Advanced reasoning ability       | Specific downstream task | Multidomain advanced reasoning ability|
| BIG-bench [[paper](https://arxiv.org/abs/2206.04615)][[GitHub](https://github.com/google/BIG-bench)]    | Capabilities and limitations of LMs | General language task | Model performance and calibration         |
| MultiMedQA [[paper](https://arxiv.org/abs/2212.13138)]  | Medical QA       | Specific downstream task | Accuracy and human evaluation|
| CVALUES [[paper](https://arxiv.org/abs/2307.09705)] [[GitHub](https://github.com/X-PLUG/CValues)]     | Safety and responsibility | Specific downstream task | Alignment ability of LLMs|
| LVLM-eHub [[paper](https://arxiv.org/abs/2306.09265)]   |  LVLMs |  Multi-modal task |  Multimodal capabilities of LVLMs |
| ToolBench [[GitHub](https://github.com/sambanova/toolbench)]  | Software tools               | Specific downstream task | Execution success rate                  |
| FRESHQA [[paper](https://arxiv.org/abs/2310.03214)] [[GitHub](https://github.com/freshllms/freshqa)]     | Dynamic QA| Specific downstream task |Correctness and hallucination|
| CMB [[paper](https://arxiv.org/abs/2308.08833)] [[Link](https://cmedbenchmark.llmzoo.com/)]     | Chinese comprehensive medicine| Specific downstream task |Expert evaluation and automatic evaluation|
| PandaLM [[paper](https://arxiv.org/abs/2306.05087)] [[GitHub](https://github.com/WeOpenML/PandaLM)] | Instruction tuning               | General language task    | Winrate judged by PandaLM             |
| MINT [[paper](https://arxiv.org/abs/2309.10691)] [[GitHub](https://xingyaoww.github.io/mint-bench/)]  | Multi-turn interaction, tools and language feedback   | Specific downstream task | Success rate with _k_-turn budget _SR<sub>k</sub>_|
| Dialogue CoT [[paper](https://arxiv.org/abs/2305.11792)] [[GitHub](https://github.com/ruleGreen/Cue-CoT)]  | In-depth dialogue  | Specific downstream task | Helpfulness and acceptness of LLMs|
| BOSS [[paper](https://arxiv.org/abs/2306.04618)] [[GitHub](https://github.com/lifan-yuan/OOD_NLP)] | OOD robustness in NLP               | General language task    | OOD robustness            |
| MM-Vet [[paper](https://arxiv.org/abs/2308.02490)] [[GitHub](https://github.com/yuweihao/MM-Vet)]  | Complicated multi-modal tasks  |  Multi-modal task | Integrated vision-language capabilities|
| LAMM [[paper](https://arxiv.org/abs/2306.06687)] [[GitHub](https://github.com/OpenLAMM/LAMM)]  | Multi-modal point clouds  |  Multi-modal task | Task-specific metrics|
| GLUE-X [[paper](https://arxiv.org/abs/2211.08073)] [[GitHub](https://github.com/YangLinyi/GLUE-X)]     | OOD robustness for NLU tasks     | General language task    | OOD robustness                       |
| KoLA [[paper](https://arxiv.org/abs/2306.09296)]       | Knowledge-oriented evaluation      | General language task    | Self-contrast metrics |
| AGIEval [[paper](https://arxiv.org/abs/2304.06364)]     | Human-centered foundational models | General language task    | General                            |
| PromptBench [[paper](https://arxiv.org/abs/2306.04528)] [[GitHub](https://github.com/microsoft/promptbench)] | Adversarial prompt resilience      | General language task    | Adversarial robustness           |
| MT-Bench [[paper](https://arxiv.org/abs/2306.05685)]  | Multi-turn conversation      | General language task    | Winrate judged by GPT-4                |
| M3Exam [[paper](https://arxiv.org/abs/2306.05179)] [[GitHub](https://github.com/DAMO-NLP-SG/M3Exam)]     | Multilingual, multimodal and multilevel | Specific downstream task | Task-specific metrics                         |
| GAOKAO-Bench [[paper](https://arxiv.org/abs/2305.12474)]     | Chinese Gaokao examination | Specific downstream task | Accuracy and scoring rate                         |
| SafetyBench [[paper](https://arxiv.org/abs/2309.07045)] [[GitHub](https://github.com/thu-coai/SafetyBench)]      | Safety | Specific downstream task | Safety abilities of LLMs                        |
| LLMEval² [[paper](https://arxiv.org/abs/2308.01862)] [[Link](https://drive.google.com/file/d/1sRbYZ0SWqmbIlzC_eB2zjyQF5TBynSXo/view)] | LLM Evaluator | General language task | Accuracy, Macro-F1 and Kappa Correlation Coefficient                        |

![图片](https://github.com/wtjgh/LLM-notes/assets/34306488/ed787a35-fd2e-4480-ba22-24912f0b377c)


