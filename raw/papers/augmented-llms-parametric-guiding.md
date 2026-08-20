# Augmented Large Language Models with Parametric Knowledge Guiding

Ziyang Luo$^{1,*}$, Can Xu$^2$, Pu Zhao$^2$, Xiubo Geng$^2$, Chongyang Tao$^2$,
Jing Ma$^1$, Qingwei Lin$^2$, Daxin Jiang$^{2,\dagger}$

$^1$ Hong Kong Baptist University, Hong Kong SAR, China
$^2$ Microsoft Corporation

cszyluo@comp.hkbu.edu.hk, majing@hkbu.edu.hk
{caxu,pu.zhao,xigeng,chongyang.tao,qlin,djiang}@microsoft.com

### Abstract

Large Language Models (LLMs) have significantly advanced natural language processing (NLP) with their impressive language understanding and generation capabilities. However, their performance may be suboptimal for domain-specific tasks that require specialized knowledge due to limited exposure to the related data. Additionally, the lack of transparency of most state-of-the-art (SOTA) LLMs, which can only be accessed via APIs, impedes further fine-tuning with domain custom data. Moreover, providing private data to the LLMs' owner leads to data privacy problems. To address these challenges, we propose the novel **Parametric Knowledge Guiding (PKG)** framework, which equips LLMs with a knowledge-guiding module to access relevant knowledge without altering the LLMs' parameters. Our PKG is based on open-source "white-box" language models, allowing offline memory of any knowledge that LLMs require. We demonstrate that our PKG framework can enhance the performance of "black-box" LLMs on a range of domain knowledge-intensive tasks that require factual (+7.9%), tabular (+11.9%), medical (+3.0%), and multimodal (+8.1%) knowledge.

## 1 Introduction

Large Language Models (LLMs) such as GPT-family [OpenAI, 2023b] have exhibited impressive proficiency across a diverse range of NLP tasks. These models are typically trained on extensive data from the internet, thereby enabling them to assimilate an immense amount of implicit world knowledge into their parameters. As a result, LLMs have emerged as versatile tools that find numerous applications in both research and industry. For instance, they can be used for machine translation [Jiao et al., 2023], document summarization [Yang et al., 2023], and recommendation systems [Gao et al., 2023]. With their exceptional language understanding and generation capabilities, LLMs have opened up new opportunities for diverse industrial applications, such as the recently launched New Bing [Microsoft, 2023] and ChatGPT Plugins [OpenAI, 2023a].

Despite their impressive performance across various general tasks, LLMs may face challenges when applied to domain-specific tasks [Chalkidis, 2023; Kasai et al., 2023; Nascimento et al., 2023] due to their limited exposure to relevant knowledge and vocabulary. Although LLMs acquire implicit world knowledge during pre-training, such knowledge may be insufficient or inappropriate for specific tasks, resulting in less effective performance. Furthermore, many state-of-the-art LLMs are considered "black-box" models, accessible only through APIs. This lack of transparency presents significant

\*Work done during the internship at Microsoft.

†Corresponding author

![Diagram illustrating the Parametric Knowledge Guiding (PKG) framework across different domains: Fact Checking, Open-Domain QA over Tables, Medical Domain QA, and Multimodal Science QA. Each domain has a specific Knowledge Guider that generates a Background Document, which is then fed via API to a Black-Box General LLM to produce answers.](page_2_image_1_v2.jpg)

Figure 1: A brief introduction of our parametric knowledge guiding framework (PKG) for augmenting "black box" LLMs on domain-specific tasks.

challenges and high costs for most researchers and companies seeking to fine-tune these models for their specific use cases or domains. Moreover, users who can afford to fine-tune must provide their private data to the LLMs’ owner, thereby exposing it to potential risks such as misuse, breaches, or other security threats [BBC, 2023]. These limitations hinder the adaptability of LLMs to diverse scenarios and domains.

A common approach to enhance LLMs is to leverage retrieval-based methods that access domain-specific knowledge from external sources [Liu, 2022; Shi et al., 2023; Peng et al., 2023a]. While these methods have shown promise, they face several challenges. First, they heavily rely on modern dual-stream dense retrieval models [Karpukhin et al., 2020] which suffer from shallow interaction between the query and candidate documents. Second, most dense retrieval models are based on small-scale pre-trained models such as BERT [Devlin et al., 2019] and therefore cannot take advantage of the world knowledge of large-scale pre-trained models. Third, retrieval models may struggle with complex knowledge that requires the integration of information from multiple sources or modalities.

In this work, we propose the Parametric Knowledge Guiding (PKG) framework, which enables LLMs to access relevant information without modifying their parameters, by incorporating a trainable background knowledge generation module, as illustrated in Figure 1. Unlike retrieval-based methods, our PKG module utilizes open-source and free-to-use "white-box" language models, LLaMa-7B [Touvron et al., 2023], which encode implicit world knowledge from large-scale pre-training. The framework consists of two steps. First, we align the PKG module with the specific task or domain knowledge via instruction fine-tuning [Ouyang et al., 2022] to capture the necessary expertise. Second, for a given input, the PKG module generates the related knowledge, fed as extra context to the background-augmented prompting for LLMs. By supplying the necessary knowledge, our framework can enhance the performance of LLMs on domain-specific tasks.

Our experiments demonstrate that the proposed PKG framework enhances the performance of "black-box" LLMs on various downstream tasks which require domain-specific background knowledge, including factual knowledge (FM2 [Eisenschlos et al., 2021], +7.9%), tabular knowledge (NQ-Table [Herzig et al., 2021], +11.9%), medical knowledge (MedMC-QA [Pal et al., 2022], +3.0%), and multimodal knowledge (ScienceQA [Lu et al., 2022], +8.1%).

We summarize our contributions as follows:

* We propose a novel Parametric Knowledge Guiding (PKG) framework that integrates a background knowledge generation module to enhance the performance of LLMs on domain-specific tasks.

* We introduce a knowledge-guiding process by first aligning the parametric modules with specific tasks or domain knowledge and then generating related knowledge as the extra context in the background-augmented prompting.
* We conduct extensive experiments on various downstream tasks to evaluate the effectiveness of our proposed PKG framework. The experiments demonstrate that our PKG framework can improve the capability of LLMs on domain knowledge-intensive tasks.

# 2 Related Work

**Large Language Models.** LLMs, such as GPT3 [Brown et al., 2020], Codex [Chen et al., 2021], PaLM [Chowdhery et al., 2022], and GPT4 [OpenAI, 2023b], have gained widespread attention due to their remarkable language understanding and generation capabilities [Wei et al., 2022c; Shi et al., 2022]. However, their performance can be limited when it comes to domain-specific tasks, where they may lack exposure to specialized knowledge and vocabulary [Chalkidis, 2023; Kasai et al., 2023; West, 2023]. Moreover, while some SOTA LLMs such as InstructGPT3.5 and ChatGPT [Ouyang et al., 2022] exist, they are available only as "black box" APIs due to commercial considerations. This limits researchers and developers with limited resources, who may not be able to access or modify the models' parameters. While open-source LLMs such as OPT-175B [Zhang et al., 2022] and BLOOM-176B [Scao et al., 2022] are available, they lag significantly behind SOTA LLMs on most tasks. Additionally, running and fine-tuning these open LLMs locally requires significant computational resources.

**Augmented Large Language Models.** ALLMs are a recent popular topic in NLP that aim to enhance the context processing ability of LLMs by incorporating external modules [Mialon et al., 2023; Wu et al., 2023; Shen et al., 2023; Lu et al., 2023; Huang et al., 2023]. One approach to achieving this goal is through the use of retrieval-augmented large language models (RLLMs)[Guu et al., 2020; Izacard et al., 2022; Ram et al., 2023; Shi et al., 2023]. RLLMs leverage external knowledge by retrieving relevant documents or passages from knowledge sources using retrieval-based methods such as BM25[Robertson and Zaragoza, 2009] and DPR [Karpukhin et al., 2020]. These retrieved passages are then used as additional contexts to improve the LLMs' performance on the task at hand. Although RLLMs have shown promise in enhancing LLMs' performance, they have certain limitations. For instance, they rely heavily on the dual-stream dense retriever, which leads to shallow interaction between the query and the candidate information. Furthermore, they may struggle with complex queries that require integrating information from multiple sources or modalities.

**Instruction Fine-Tuning.** IFT is a technique in NLP that aims to align language models with specific user intents [Ouyang et al., 2022]. While many LLMs are trained on large datasets of internet data to predict the next word, they may not be tailored to the specific language tasks that users require, meaning that these models are not inherently aligned with their users' needs. Recent research [Wei et al., 2022a; Sanh et al., 2022; Xu et al., 2022; Xie et al., 2022] has highlighted the potential of IFT as a key technique for improving the usability of LLMs. Our proposed approach, PKG, follows the same principle of aligning the basic module with task-specific knowledge to enhance its performance.

# 3 Parametric Knowledge Guiding for LLMs

In this section, we present our **PKG** framework to guide the reasoning process of LLMs on domain-specific tasks. These tasks differ from general tasks such as document summarization due to their reliance on specific background knowledge. However, this knowledge may be absent or incomplete in the LLMs' training data. Furthermore, continuous pre-training of LLMs with domain knowledge poses several challenges: (1) limited transparency of accessing current SOTA LLMs solely through APIs, (2) the potentially high fine-tuning cost associated with APIs usage, and (3) concerns regarding data privacy when providing private data to LLMs' owners. To tackle these issues, we adhere to the *generate-then-read* paradigm [Yu et al., 2023] and leverage an offline PKG module to generate relevant background knowledge. Our method is first formulated in § 3.1. Next, we describe the background knowledge alignment of our PKG modules in § 3.2. Finally, we introduce background-augmented prompting for LLMs in § 3.3.

![Diagram showing the two-step process of knowledge alignment: Step 1 involves data reformulation from raw task/domain data to instruction data, and Step 2 involves instruction fine-tuning of an open-source white-box LM to create a Parametric Knowledge Guider.](page_4_image_1_v2.jpg)

Figure 2: The knowledge alignment example of the PKG module on the fact-checking task (FM2). The passage behind the "Response" is the background knowledge of the "Input".

# 3.1 Formulation

Given a question/input $Q$ associated with some contexts, LLMs take the input and generate a response by maximum a posteriori estimation (MAP):

$$ \hat{\mathcal{A}} := \text{argmax}_{\mathcal{A}} P(\mathcal{A} | \mathcal{Q}, \mathcal{M}^{LLM}), \quad (1) $$

where $\mathcal{M}^{LLM}$ represents the parameters of the LLMs. However, for tasks that require background knowledge beyond what is contained in the input, such as knowledge-intensive tasks, relying solely on LLMs may not be effective. This is because there may be a significant amount of additional domain-specific knowledge that remains unexploited.

To improve performance, we first introduce an auxiliary PKG module $\mathcal{M}^{PKG}$ to align specific background knowledge (§3.2). Next, we estimate the input-related background knowledge $\mathcal{K}$ using MAP estimation:

$$ \hat{\mathcal{K}} := \text{argmax}_{\mathcal{K}} P(\mathcal{K} | \mathcal{Q}, \mathcal{M}^{PKG}). \quad (2) $$

Finally, the background knowledge $\mathcal{K}$ enriches the input by incorporating background-augmented prompting for LLMs (§ 3.3) in the form:

$$ P(\mathcal{A} | \mathcal{Q}) := P(\mathcal{A} | \mathcal{K}, \mathcal{Q}, \mathcal{M}^{LLM}) P(\mathcal{K} | \mathcal{Q}, \mathcal{M}^{PKG}). \quad (3) $$

# 3.2 Background Knowledge Alignment

Given a target task or domain, our PKG framework utilizes an open-source language model to align with relevant knowledge. Figure 2 presents an example of the fact-checking task. This process is divided into two steps. First, we collect raw data about the target task/domain, which serves as our background knowledge. Second, we transform the data into a set of (instruction, input, output) triples. The instruction serves as a prompt for the input and guides the module to align with the expected knowledge.

Next, this set of triples is adopted to tune our basic PKG module with instruction fine-tuning [Ouyang et al., 2022], which optimizes its ability to provide relevant and effective background knowledge to the LLMs. This two-step process can be completed fully offline, without requiring us to provide our private data to tune the LLMs. Once aligned with the task background knowledge, the PKG module learns to generate domain-specific knowledge to assist the LLMs during runtime.

The instruction data format of the fact-checking task is:

```
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
### Instruction:
<instruction>
### Input:
<input sentence>
### Response:
<background>
```

The `<input sentence>` is a sentence within the specified task. The `<background>` is the background knowledge that the model generates based on the given `<instruction>` and `<input sentence>`. The basic PKG module is trained in a standard supervised way with an auto-regressive

manner, where the model generates the `<background>` given the previous context. More instruction data formats for different tasks are presented in Appendix F.

## 3.3 Background-Augmented Prompting

Instead of directly requesting the LLMs to generate the answer or response for the input question or sentence via APIs, we first instruct the PKG module to generate the background knowledge. In the second step, we utilize the generated background in combination with the input question to derive the final answer from the LLMs. This is similar to the "zero-shot" open-domain question-answering setting that has been widely explored in prior research [Brown et al., 2020; Lazaridou et al., 2022; Yu et al., 2023]. The background-augmented prompt of the fact-checking task is:

```
<background>
Claim: <input sentence>
Is the claim true or false?
```

Finally, the augmented prompt is fed into the LLMs to generate an answer. More prompts for different tasks are presented in Appendix G.

# 4 Experiment

In this section, we evaluate our proposed PKG framework across four distinct types of knowledge: factual, tabular, medical, and multimodal. Factual knowledge entails the model's ability to access accurate information, serving as a foundational type of knowledge crucial for numerous NLP applications (§ 4.2). Tabular knowledge necessitates the model's capability to access structured knowledge in the form of tables, which is relatively scarce in the training data of LLMs (§ 4.3). Medical knowledge, being highly specialized, exhibits limited exposure within the general data (§ 4.4). Lastly, multi-modal knowledge poses a challenge as most LLMs are unable to process non-language information, highlighting the significance of assistance from PKG modules (§ 4.5).

The experimental results depicted in Tables 1 and 2 demonstrate substantial enhancements attained through our PKG framework compared to the baseline systems. These results offer compelling evidence supporting the generalizability and effectiveness of our approach.

## 4.1 Models Steup

**Black-Box LLMs.** We adopt one of the SOTA LLM InstructGPT3.5 [Ouyang et al., 2022] as our target "black box" general LLMs, using the text-davinci-002 version. With up to 175B parameters, this model is one of the largest LLMs and is pre-trained on a vast amount of internet data, which exhibits great language understanding and generation ability. However, this model can only be accessed through an API, which limits users' interaction.

**Basic PKG Module.** Our knowledge guiding module employs the open-source and popular foundation model LLaMa-7B [Touvron et al., 2023]. It has been pre-trained on massive amounts of text data and possesses extensive world knowledge. Though its performance in many tasks may be inferior to the InstructGPTs, it can be locally fine-tuned and customized [Taori et al., 2023; Xu et al., 2023; Peng et al., 2023b; Geng et al., 2023], making it an effective starting point for developing a task-specific PKG module.

**Baselines.** Our work includes three different types of baselines: (1) *Direct generation without guiding*: We do not provide any background knowledge for a given task and ask the InstructGPT to generate the answer or response directly in a zero-shot manner, following the approach of prior works [Brown et al., 2020; Ouyang et al., 2022]. (2) *Generation with retrieval guiding*: We follow the retrieve-then-read paradigm [Chen et al., 2017; Yang et al., 2019; Karpukhin et al., 2020] to retrieve related knowledge from external knowledge sources using retrieval models such as BM25 [Robertson and Zaragoza, 2009] and DPR [Karpukhin et al., 2020]. We fine-tune the DPR on specific tasks following the REPLUG [Shi et al., 2023] method. InstructGPTs then generate responses based on the combination of the question and retrieved background documents. (3) *Generation with self-guiding*: we adopt the InstructGPTs to generate the related background knowledge by themselves with two different methods. The first method, CoT Kojima et al. [2022], adopts the prompt "Let's think

Table 1: Evaluating on three different tasks, requiring factual (FM2), tabular (NQ-Table), and medical (MedMC-QA) knowledge. $\diamond$: we fine-tune the dense retrieval models with the task data. $\dagger$: we use InstructGPT3.5 to generate the chain-of-thoughts as the background knowledge. $\ddagger$: we use InstructGPT3.5 to generate the background documents.


<table>
  <thead>
    <tr>
        <th>Models</th>
        <th>FM2</th>
        <th>NQ-Table</th>
        <th>MedMC-QA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="4">Direct generation without guiding.</td>
    </tr>
    <tr>
        <td>InstructGPT3.5 [Ouyang et al., 2022]</td>
        <td>59.4</td>
        <td>16.9</td>
        <td>44.4</td>
    </tr>
    <tr>
        <td colspan="4">Generation with retrieval guiding.</td>
    </tr>
    <tr>
        <td>BM25 + InstructGPT3.5 [Karpukhin et al., 2020]</td>
        <td>65.2</td>
        <td>17.1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>$\diamond$REPLUG + InstructGPT3.5 [Shi et al., 2023]</td>
        <td>65.9</td>
        <td>24.3</td>
        <td>-</td>
    </tr>
    <tr>
        <td colspan="4">Generation with self-guiding.</td>
    </tr>
    <tr>
        <td>$\dagger$CoT + InstructGPT3.5 [Kojima et al., 2022]</td>
        <td>60.4</td>
        <td>21.4</td>
        <td>41.5</td>
    </tr>
    <tr>
        <td>$\ddagger$GenRead + InstructGPT3.5 [Yu et al., 2023]</td>
        <td>65.5</td>
        <td>23.5</td>
        <td>44.4</td>
    </tr>
    <tr>
        <td><strong>PKG + InstructGPT3.5 (Ours)</strong></td>
        <td><strong>67.3</strong></td>
        <td><strong>28.8</strong></td>
        <td><strong>47.4</strong></td>
    </tr>
  </tbody>
</table>


Table 2: Evaluating on the ScienceQA, requiring multimodal science knowledge. $\dagger$: results from Lu et al. [2023]. gpt-3.5-turbo is much more capable than text-davinic-002.


<table>
  <thead>
    <tr>
        <th>Models</th>
        <th>NAT</th>
        <th>SOC</th>
        <th>LAN</th>
        <th>TXT</th>
        <th>IMG</th>
        <th>NO</th>
        <th>G1-6</th>
        <th>G7-12</th>
        <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td colspan="10">Base on gpt-3.5-turbo.</td>
    </tr>
    <tr>
        <td>$\dagger$ChatGPT</td>
        <td>78.82</td>
        <td>70.98</td>
        <td>83.18</td>
        <td>77.37</td>
        <td>67.92</td>
        <td>86.13</td>
        <td>80.72</td>
        <td>74.03</td>
        <td>78.31</td>
    </tr>
    <tr>
        <td>$\dagger$Chameleon</td>
        <td>81.62</td>
        <td>70.64</td>
        <td>84.00</td>
        <td>79.77</td>
        <td>70.80</td>
        <td>86.62</td>
        <td>81.86</td>
        <td>76.53</td>
        <td>79.93</td>
    </tr>
    <tr>
        <td colspan="10">Base on text-davinic-002.</td>
    </tr>
    <tr>
        <td>InstructGPT3.5</td>
        <td>72.96</td>
        <td>62.88</td>
        <td>76.09</td>
        <td>70.77</td>
        <td>62.77</td>
        <td>77.84</td>
        <td>75.04</td>
        <td>65.59</td>
        <td>71.66</td>
    </tr>
    <tr>
        <td>+CoT</td>
        <td>71.94</td>
        <td>61.19</td>
        <td>74.00</td>
        <td>69.50</td>
        <td>61.18</td>
        <td>75.75</td>
        <td>72.61</td>
        <td>65.92</td>
        <td>70.22</td>
    </tr>
    <tr>
        <td>+GenRead</td>
        <td>72.91</td>
        <td>64.68</td>
        <td>76.36</td>
        <td>72.14</td>
        <td>63.31</td>
        <td>76.66</td>
        <td>74.96</td>
        <td>66.91</td>
        <td>72.08</td>
    </tr>
    <tr>
        <td>+PKG (Ours)</td>
        <td>79.35</td>
        <td>82.90</td>
        <td>81.91</td>
        <td>79.86</td>
        <td>74.32</td>
        <td>83.41</td>
        <td>80.80</td>
        <td>80.69</td>
        <td><strong>80.76</strong></td>
    </tr>
  </tbody>
</table>


step-by-step" to generate the chain-of-thought as the background knowledge. The second method, GenRead Yu et al. [2023], directly requires the InstructGPTs to provide task-specific knowledge with the prompt "Please provide the background document from [domain] to [task]."

## 4.2 Factual Knowledge

**Datasets and Implementation Details.** We evaluate our approach on the FM2 dataset [Eisenschlos et al., 2021], which is a benchmark for fact-checking. In this task, given a factual claim, our models are required to determine whether it is true or false. We use the claim in the training set and the corresponding evidence as factual knowledge. Additionally, we sample 100k passages from English Wikipedia, each consisting of up to 256 tokens. We treat the first sentence as the input and the remaining sentences as background knowledge. Accuracy is adopted as the evaluation metric. More details can be found in Appendix A and B.

**Results.** As shown in Table 1, our PKG outperforms all the baseline systems for fact-checking. In comparison to direct generation, the results reveal that it is necessary to provide extra background knowledge for InstructGPTs with retrieval-based or generation-based methods. Specifically, our PKG outperforms InstructGPT3.5 by 7.9% (67.9% vs. 59.4%), and outperforms REPLUG, a retrieval-based method, by 1.4% (67.3% vs. 65.9%). It is noteworthy that our generation-based method does not necessitate an additional knowledge database as the retrieval-based methods. Additionally, our PKG performs better than the self-guiding method GenRead by 1.8% (67.3% vs. 65.5%), indicating that our PKG can provide more useful information than the InstructGPTs themselves.

## 4.3 Tabular Knowledge

**Datasets and Implementation Details.** We evaluate the effectiveness of our approach on the NQ-Table dataset [Herzig et al., 2021], which serves as a benchmark for open-domain question answering over tables. The dataset consists of questions whose answers can be found in a Wikipedia table. We adopted the question in the training set as input and the corresponding flattened table as background knowledge. Our PKG was trained to follow instructions and generate the relevant table. Exact matching is adopted as the evaluation metric. More details can be found in Appendix A and B.

**Results.** Table 1 demonstrates the superior performance of our PKG framework over all baseline systems on the tabular knowledge-related task. Notably, our PKG outperforms InstructGPT3.5 by a substantial margin of 11.9% (28.8% vs. 16.9%), and outperforms REPLUG, the retrieval-based method, by 4.5% (28.8% vs. 24.3%). Furthermore, our PKG significantly outperforms the self-guiding method GenRead by 5.3% (28.8% vs. 23.5%). These results demonstrate the efficacy and superiority of our approach in leveraging parametric knowledge to augment InstructGPTs for tabular knowledge-related tasks.

## 4.4 Medical Knowledge

**Datasets and Implementation Details.** We evaluate the effectiveness of our approach on the MedMC-QA dataset [Pal et al., 2022], which serves as a benchmark for multi-subject multi-choice medical question answering. Each question requires the use of relevant medical information as background knowledge to provide the correct answer. We use the questions in the training set as input and the corresponding medical explanation as background knowledge. Our PKG is trained to follow the instruction and generate the relevant medical background. Accuracy is the evaluation metric. Unlike the previous tasks with all Wikipedia passages as the knowledge database, we do not have access to an external medical knowledge database, and thus we do not evaluate the performance of retrieval-based methods on this task. More details can be found in Appendix A and B.

**Results.** Our PKG framework also outperforms all baseline systems on this medical knowledge-related task, as shown in Table 1. Specifically, our PKG outperforms InstructGPT3.5 by 3.0% (47.4% vs. 44.4%). It is worth noting that the baseline self-guiding methods, CoT and GenRead, do not improve the performance of InstructGPTs. This may be due to the fact that InstructGPTs lack sufficient medical information to effectively solve this task.

## 4.5 Multimodal Knowledge

**Datasets and Implementation Details.** Our approach is evaluated on the ScienceQA dataset [Lu et al., 2022], which presents a challenging multimodal multiple-choice question-answering task covering diverse science topics. Each question requires leveraging relevant scientific background knowledge to provide the correct answer. We use the training set's questions as input and their corresponding science lecture as background knowledge. To handle the images information, we augment our basic PKG module with the CLIP-ViT [Radford et al., 2021] to extract visual features, which are then fused with text features using a simple one-head cross-attention mechanism in each layer of LLaMa:

$$ H := H^{txt} + W^o \left( \text{softmax} \left( (W^q H^{txt})(W^k H^{img})^T \right) (W^v H^{img}) \right), \tag{4} $$

where $W^{o,q,k,v}$ are the linear projection, $H^{txt,img}$ are the hidden states of texts and images. We adopt accuracy as the evaluation metric. More details can be found in Appendix A and B.

Similarly, this task is also difficult to obtain an external multimodal science knowledge database, retrieval-based methods are not considered. To facilitate a fair comparison of our methods, we include two additional baseline systems [Lu et al., 2023] based on the gpt-3.5-turbo model. The first baseline is ChatGPT direct generation, and the second is the Chameleon model, which utilizes several external tools, such as searching, OCR, and captioning. According to OpenAI, the gpt-3.5-turbo model is more capable than text-davinci-002 [?].

**Results.** Table 2 shows that our PKG framework achieves a significant improvement in the performance of InstructGPTs on the multimodal scientific knowledge-related task. Specifically, the

(a) Accuracy on FM2.
<table>
  <tbody>
    <tr>
        <td>Model Size</td>
        <td>PKG + InstructGPT</td>
        <td>InstructGPT</td>
    </tr>
    <tr>
        <td>0.3B</td>
        <td>40.5</td>
        <td>43.5</td>
    </tr>
    <tr>
        <td>1.3B</td>
        <td>46.0</td>
        <td>41.0</td>
    </tr>
    <tr>
        <td>6.7B</td>
        <td>52.0</td>
        <td>51.0</td>
    </tr>
    <tr>
        <td>175B</td>
        <td>64.5</td>
        <td>59.0</td>
    </tr>
  </tbody>
</table>



(b) Exact Matching on NQ-Table.
<table>
  <tbody>
    <tr>
        <td>Model Size</td>
        <td>PKG + InstructGPT</td>
        <td>InstructGPT</td>
    </tr>
    <tr>
        <td>0.3B</td>
        <td>0.0</td>
        <td>0.0</td>
    </tr>
    <tr>
        <td>1.3B</td>
        <td>0.0</td>
        <td>0.0</td>
    </tr>
    <tr>
        <td>6.7B</td>
        <td>0.0</td>
        <td>0.0</td>
    </tr>
    <tr>
        <td>175B</td>
        <td>29.0</td>
        <td>17.0</td>
    </tr>
  </tbody>
</table>



(c) Accuracy on MedMC-QA.
<table>
  <tbody>
    <tr>
        <td>Model Size</td>
        <td>PKG + InstructGPT</td>
        <td>InstructGPT</td>
    </tr>
    <tr>
        <td>0.3B</td>
        <td>21.0</td>
        <td>21.0</td>
    </tr>
    <tr>
        <td>1.3B</td>
        <td>21.0</td>
        <td>21.0</td>
    </tr>
    <tr>
        <td>6.7B</td>
        <td>24.0</td>
        <td>21.0</td>
    </tr>
    <tr>
        <td>175B</td>
        <td>45.0</td>
        <td>44.0</td>
    </tr>
  </tbody>
</table>



(d) Average Accuracy on ScienceQA
<table>
  <tbody>
    <tr>
        <td>Model Size</td>
        <td>PKG + InstructGPT</td>
        <td>InstructGPT</td>
    </tr>
    <tr>
        <td>0.3B</td>
        <td>29.0</td>
        <td>27.0</td>
    </tr>
    <tr>
        <td>1.3B</td>
        <td>35.0</td>
        <td>41.0</td>
    </tr>
    <tr>
        <td>6.7B</td>
        <td>34.0</td>
        <td>37.0</td>
    </tr>
    <tr>
        <td>175B</td>
        <td>80.0</td>
        <td>72.0</td>
    </tr>
  </tbody>
</table>


Figure 3: Comparing our PKGs framework with the direct generation on various types of InstructGPT. The number indicates the number of parameters in the InstructGPT. 0.3B: text-ada-001, 1.3B: - babbage-001, 6.7B: text-curie-001, 175B: text-davinci-002.

Table 3: Comparing various sizes of language models as the basic PKG modules.


<table>
  <thead>
    <tr>
        <th>Basic PKG</th>
        <th>FM2</th>
        <th>NQ-Table</th>
        <th>MedMC-QA</th>
        <th>ScienceQA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>LLaMa-7B [Touvron et al., 2023]</td>
        <td>67.3</td>
        <td>28.8</td>
        <td>47.4</td>
        <td>80.8</td>
    </tr>
    <tr>
        <td>OPT-2.7B [Zhang et al., 2022]</td>
        <td>59.6</td>
        <td>17.9</td>
        <td>34.4</td>
        <td>79.5</td>
    </tr>
    <tr>
        <td>OPT-1.3B</td>
        <td>58.2</td>
        <td>16.5</td>
        <td>33.9</td>
        <td>77.0</td>
    </tr>
    <tr>
        <td>OPT-0.3B</td>
        <td>56.4</td>
        <td>14.6</td>
        <td>31.7</td>
        <td>68.7</td>
    </tr>
  </tbody>
</table>


average accuracy is increased by 9.1% (80.76% vs. 71.66%), demonstrating the effectiveness of our approach. In contrast, other guiding methods, CoT (-1.44%) and GenRead (+0.42%), hard to improve the performance of InstructGPTs. Moreover, our PKG framework outperforms the gpt-3.5-turbo based models on average by 2.45% (80.76% vs. 78.31%), despite using weaker InstructGPTs.

## 4.6 Analysis

**Scale of LLMs.** Figure 3 presents the impact of our PKG framework on several "black-box" LMs, including text-ada-001, text-babbage-001, text-curie-001, and text-davinci-002. The results suggest that the effectiveness of our approach is correlated with the size of the LMs, with larger LMs benefiting more from our PKGs than smaller ones. Specifically, in Figure 3b, the small LMs show negligible exact matching scores on the tabular task, with or without the background knowledge from our PKGs, while the LLMs exhibit significantly better performance. In Figure 3c, the 0.3B and 1.3B LMs perform similarly on the medical domain task, while the 6.7B LM shows improved performance with the additional knowledge. This difference can be attributed to the relatively weaker language understanding capabilities of smaller LMs, which struggle to reason over contexts and generate the correct responses even with relevant knowledge from our PKGs. These observations align with the emergent abilities of LLMs, as discussed in Wei et al. [2022b]. Therefore, the scale of LLMs is a critical factor for achieving better performance.

**Scale of PKGs.** We conducted an investigation of various sizes of language models as basic PKG modules in Table 3. Since LLaMa-7B is the smallest model in the LLaMa family, we conducted experiments on the OPT family [Zhang et al., 2022], another open-source large-scale language model with a similar structure to LLaMa. Our observations reveal that larger basic PKGs tend to exhibit

Table 4: Examples of background documents generated by our PKGs to guide different tasks. Clues to answer the input are highlighted in blue within the documents.


<table>
  <tbody>
    <tr>
        <td>Input (FM2)</td>
        <td>The Natural is a book about Roy Hobbs a natural southpaw boxer who goes on to win the heavyweight title from Boom Boom Mancini.</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>This is <mark>a story about a baseball player, Roy Hobbs</mark>, who is shot by a woman, Harriet Bird, and then recovers to become a great player.</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>Input (NQ-Table)</td>
        <td>Batman The Enemy Within episode 5 release date</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Batman: The Enemy Within<br/>Ep</td>
        <td>Title</td>
        <td>Directed by</td>
        <td>Release date<br/>1</td>
        <td>The Enigma</td>
        <td>Kent Mudle</td>
        <td>August 8, 2017<br/>2</td>
        <td>The Pact</td>
        <td>Matthew Leach</td>
        <td>October 3, 2017<br/>3</td>
        <td>Fractured Mask</td>
        <td>Sean Manning</td>
        <td>November 21, 2017<br/>4</td>
        <td>What Ails You</td>
        <td>Chris Rieser</td>
        <td>January 23, 2018<br/>5</td>
        <td>Same Stitch</td>
        <td>Kent Mudle</td>
        <td><mark>March 27, 2018</mark></td>
    </tr>
    <tr>
        <td>Input (MedMC-QA)</td>
        <td>Calcium ions triggers muscle contraction by binding to: Options: (A) Actin (B) Myosin (C) Troponin (D) Tropomyosin</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Troponin is a complex of three proteins that are attached to actin. <mark>Calcium ions bind to troponin-C</mark>, which then causes the troponin-I and troponin-T to shift and allow the tropomyosin to move off of the active site of the myosin so that the myosin head can then bind to actin and cause contraction.</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>Input (ScienceQA)<br/><img src="page_9_image_1_v2.jpg" alt="saddle-billed stork"></td>
        <td>Which animal’s neck is also adapted for hunting prey while keeping the rest of its body still? Context: Saddle-billed storks live near wetlands and lakes. They eat mostly fish. The 's neck helps it grab fish while keeping the rest of its body still. If the stork had to move its body, it might scare the fish away. Figure: saddle-billed stork. A black and white bird is standing in the grass. Options: (A) northern pintail (B) black-headed heron</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Look at the picture of the saddle-billed stork. The saddle-billed stork has a long neck. Its neck is adapted for hunting prey while keeping the rest of its body still. Now look at each animal. Figure out which animal has a similar adaptation. <mark>The black-headed heron has a long neck. Its neck is adapted for hunting prey while keeping the rest of its body still.</mark> The northern pintail has a short neck. Its neck is not adapted for hunting prey while keeping the rest of its body still.</td>
        <td colspan="18"></td>
    </tr>
  </tbody>
</table>


superior performance. For example, increasing the number of parameters from 1.3B to 2.7B leads to performance improvements of 1.4% on FM2, 1.4% on NQ-Table, 0.5% on MedMC-QA, and 2.5% on ScienceQA, which is consistent with the scaling law [Kaplan et al., 2020].

**Examples of Generated Background Documents.** Table 4 presents examples of background documents generated by our PKGs to assist LLMs in different tasks. For the factual task, our PKG can supply input-related factual information to support or refute the input, such as the example of Roy Hobbs being a baseball player and not a boxer. For the tabular task, our PKG can offer an input-related background table, like the episode table of Batman. For the medical task, our PKG can provide relevant medical knowledge, such as the background of calcium ions. For the multimodal task, our PKG can produce a document based on text information while taking into account the image context in the input, for example, noting that the bird in the image has a long neck. Additional examples can be found in Appendix D.

# 5 Conclusion

In this work, we propose the novel Parametric Knowledge Guiding (PKG) framework to enhance the performance of "black-box" LLMs on domain-specific tasks by equipping them with a knowledge-guiding module. Our approach allows for access to relevant knowledge at runtime without altering the LLM's parameters. The experiments demonstrate the effectiveness of our PKG framework for various domain knowledge-intensive tasks.

Limitation and Future Work. Although our PKGs have shown strong performance on the presented datasets, they may still suffer from hallucination errors, leading to the provision of incorrect background knowledge. We provide examples of such errors in Appendix E. Combining our approach with retrieval methods to enhance generative faithfulness is a promising direction for future research.

# References

BBC, 2023. Chatgpt banned in italy over privacy concerns. Webpage. URL: https://www.bbc.com/news/technology-65139406. accessed on May 8, 2023.

Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., Amodei, D., 2020. Language models are few-shot learners. CoRR abs/2005.14165. URL: https://arxiv.org/abs/2005.14165, arXiv:2005.14165.

Chalkidis, I., 2023. Chatgpt may pass the bar exam soon, but has a long way to go for the lexglue benchmark. arXiv:2304.12202.

Chen, D., Fisch, A., Weston, J., Bordes, A., 2017. Reading wikipedia to answer open-domain questions, in: Barzilay, R., Kan, M. (Eds.), Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, Association for Computational Linguistics. pp. 1870–1879. URL: https://doi.org/10.18653/v1/P17-1171, doi:10.18653/v1/P17-1171.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H.P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F.P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W.H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A.N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., Zaremba, W., 2021. Evaluating large language models trained on code. CoRR abs/2107.03374. URL: https://arxiv.org/abs/2107.03374, arXiv:2107.03374.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., Reif, E., Du, N., Hutchinson, B., Pope, R., Bradbury, J., Austin, J., Isard, M., Gur-Ari, G., Yin, P., Duke, T., Levskaya, A., Ghemawat, S., Dev, S., Michalewski, H., Garcia, X., Misra, V., Robinson, K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim, H., Zoph, B., Spiridonov, A., Sepassi, R., Dohan, D., Agrawal, S., Omernick, M., Dai, A.M., Pillai, T.S., Pellat, M., Lewkowycz, A., Moreira, E., Child, R., Polozov, O., Lee, K., Zhou, Z., Wang, X., Saeta, B., Diaz, M., Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck, D., Dean, J., Petrov, S., Fiedel, N., 2022. Palm: Scaling language modeling with pathways. CoRR abs/2204.02311. URL: https://doi.org/10.48550/arXiv.2204.02311, doi:10.48550/arXiv.2204.02311, arXiv:2204.02311.

Devlin, J., Chang, M., Lee, K., Toutanova, K., 2019. BERT: pre-training of deep bidirectional transformers for language understanding, in: Burstein, J., Doran, C., Solorio, T. (Eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), Association for Computational Linguistics. pp. 4171–4186. URL: https://doi.org/10.18653/v1/n19-1423, doi:10.18653/v1/n19-1423.

Eisenschlos, J., Dhingra, B., Bulian, J., Börschinger, B., Boyd-Graber, J.L., 2021. Fool me twice: Entailment from wikipedia gamification, in: Toutanova, K., Rumshisky, A., Zettlemoyer, L., Hakkani-Tür, D., Beltagy, I., Bethard, S., Cotterell, R., Chakraborty, T., Zhou, Y. (Eds.), Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, Association for Computational Linguistics. pp. 352–365. URL: https://doi.org/10.18653/v1/2021.naacl-main.32, doi:10.18653/v1/2021.naacl-main.32.

Gao, Y., Sheng, T., Xiang, Y., Xiong, Y., Wang, H., Zhang, J., 2023. Chat-rec: Towards interactive and explainable llms-augmented recommender system. CoRR abs/2303.14524. URL: https://doi.org/10.48550/arXiv.2303.14524, doi:10.48550/arXiv.2303.14524, arXiv:2303.14524.

Geng, X., Gudibande, A., Liu, H., Wallace, E., Abbeel, P., Levine, S., Song, D., 2023. Koala: A dialogue model for academic research. Blog post. URL: https://bair.berkeley.edu/blog/2023/04/03/koala/.

Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M., 2020. Retrieval augmented language model pre-training, in: Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, PMLR. pp. 3929–3938. URL: http://proceedings.mlr.press/v119/guu20a.html.

Herzig, J., Müller, T., Krichene, S., Eisenschlos, J., 2021. Open domain question answering over tables via dense retrieval, in: Toutanova, K., Rumshisky, A., Zettlemoyer, L., Hakkani-Tür, D., Beltagy, I., Bethard, S., Cotterell, R., Chakraborty, T., Zhou, Y. (Eds.), Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, Association for Computational Linguistics. pp. 512–519. URL: https://doi.org/10.18653/v1/2021.naacl-main.43, doi:10.18653/v1/2021.naacl-main.43.

Huang, R., Li, M., Yang, D., Shi, J., Chang, X., Ye, Z., Wu, Y., Hong, Z., Huang, J., Liu, J., Ren, Y., Zhao, Z., Watanabe, S., 2023. Audiogpt: Understanding and generating speech, music, sound, and talking head. CoRR abs/2304.12995. URL: https://doi.org/10.48550/arXiv.2304.12995, doi:10.48550/arXiv.2304.12995, arXiv:2304.12995.

Izacard, G., Lewis, P.S.H., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-Yu, J., Joulin, A., Riedel, S., Grave, E., 2022. Few-shot learning with retrieval augmented language models. CoRR abs/2208.03299. URL: https://doi.org/10.48550/arXiv.2208.03299, doi:10.48550/arXiv.2208.03299, arXiv:2208.03299.

Jiao, W., Wang, W., Huang, J., Wang, X., Tu, Z., 2023. Is chatgpt A good translator? A preliminary study. CoRR abs/2301.08745. URL: https://doi.org/10.48550/arXiv.2301.08745, doi:10.48550/arXiv.2301.08745, arXiv:2301.08745.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D., 2020. Scaling laws for neural language models. CoRR abs/2001.08361. URL: https://arxiv.org/abs/2001.08361, arXiv:2001.08361.

Karpukhin, V., Oguz, B., Min, S., Lewis, P.S.H., Wu, L., Edunov, S., Chen, D., Yih, W., 2020. Dense passage retrieval for open-domain question answering, in: Webber, B., Cohn, T., He, Y., Liu, Y. (Eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, Association for Computational Linguistics. pp. 6769–6781. URL: https://doi.org/10.18653/v1/2020.emnlp-main.550, doi:10.18653/v1/2020.emnlp-main.550.

Kasai, J., Kasai, Y., Sakaguchi, K., Yamada, Y., Radev, D., 2023. Evaluating GPT-4 and chatgpt on japanese medical licensing examinations. CoRR abs/2303.18027. URL: https://doi.org/10.48550/arXiv.2303.18027, doi:10.48550/arXiv.2303.18027, arXiv:2303.18027.

Kojima, T., Gu, S.S., Reid, M., Matsuo, Y., Iwasawa, Y., 2022. Large language models are zero-shot reasoners, in: Oh, A.H., Agarwal, A., Belgrave, D., Cho, K. (Eds.), Advances in Neural Information Processing Systems. URL: https://openreview.net/forum?id=e2TBb5y0yFf.

Lazaridou, A., Gribovskaya, E., Stokowiec, W., Grigorev, N., 2022. Internet-augmented language models through few-shot prompting for open-domain question answering. CoRR abs/2203.05115. URL: https://doi.org/10.48550/arXiv.2203.05115, doi:10.48550/arXiv.2203.05115, arXiv:2203.05115.

Liu, J., 2022. LlamaIndex. URL: https://github.com/jerryjliu/llama_index, doi:10.5281/zenodo.1234.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K., Zhu, S., Tafjord, O., Clark, P., Kalyan, A., 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. CoRR abs/2209.09513. URL: https://doi.org/10.48550/arXiv.2209.09513, doi:10.48550/arXiv.2209.09513, arXiv:2209.09513.

Lu, P., Peng, B., Cheng, H., Galley, M., Chang, K., Wu, Y.N., Zhu, S., Gao, J., 2023. Chameleon: Plug-and-play compositional reasoning with large language models. CoRR abs/2304.09842. URL: https://doi.org/10.48550/arXiv.2304.09842, doi:10.48550/arXiv.2304.09842, arXiv:2304.09842.

Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., Rozière, B., Schick, T., Dwivedi-Yu, J., Celikyilmaz, A., Grave, E., LeCun, Y., Scialom, T., 2023. Augmented language models: a survey. CoRR abs/2302.07842. URL: https://doi.org/10.48550/arXiv.2302.07842, doi:10.48550/arXiv.2302.07842, arXiv:2302.07842.

Microsoft, 2023. New bing. Webpage. URL: https://www.bing.com/new. accessed on May 8, 2023.

Nascimento, C., Monteiro, C., Pimentel, Silva, A., 2023. Do large language models understand chemistry? a conversation with chatgpt. Journal of Chemical Information and Modeling 63, 1649–1655. URL: https://doi.org/10.1021/acs.jcim.3c00285, doi:10.1021/acs.jcim.3c00285, arXiv:https://doi.org/10.1021/acs.jcim.3c00285. pMID: 36926868.

OpenAI, 2023a. Chatgpt plugins. Webpage. URL: https://openai.com/blog/chatgpt-plugins. accessed on May 8, 2023.

OpenAI, 2023b. GPT-4 technical report. CoRR abs/2303.08774. URL: https://doi.org/10.48550/arXiv.2303.08774, doi:10.48550/arXiv.2303.08774, arXiv:2303.08774.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P.F., Leike, J., Lowe, R., 2022. Training language models to follow instructions with human feedback. CoRR abs/2203.02155. URL: https://doi.org/10.48550/arXiv.2203.02155, doi:10.48550/arXiv.2203.02155, arXiv:2203.02155.

Pal, A., Umapathi, L.K., Sankarasubbu, M., 2022. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering, in: Flores, G., Chen, G.H., Pollard, T.J., Ho, J.C., Naumann, T. (Eds.), Conference on Health, Inference, and Learning, CHIL 2022, 7-8 April 2022, Virtual Event, PMLR. pp. 248–260. URL: https://proceedings.mlr.press/v174/pal22a.html.

Peng, B., Galley, M., He, P., Cheng, H., Xie, Y., Hu, Y., Huang, Q., Liden, L., Yu, Z., Chen, W., Gao, J., 2023a. Check your facts and try again: Improving large language models with external knowledge and automated feedback. CoRR abs/2302.12813. URL: https://doi.org/10.48550/arXiv.2302.12813, doi:10.48550/arXiv.2302.12813, arXiv:2302.12813.

Peng, B., Li, C., He, P., Galley, M., Gao, J., 2023b. Instruction tuning with GPT-4. CoRR abs/2304.03277. URL: https://doi.org/10.48550/arXiv.2304.03277, doi:10.48550/arXiv.2304.03277, arXiv:2304.03277.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I., 2021. Learning transferable visual models from natural language supervision, in: Meila, M., Zhang, T. (Eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, PMLR. pp. 8748–8763. URL: http://proceedings.mlr.press/v139/radford21a.html.

Ram, O., Levine, Y., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., Shoham, Y., 2023. In-context retrieval-augmented language models. CoRR abs/2302.00083. URL: https://doi.org/10.48550/arXiv.2302.00083, doi:10.48550/arXiv.2302.00083, arXiv:2302.00083.

Robertson, S.E., Zaragoza, H., 2009. The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr. 3, 333–389. URL: https://doi.org/10.1561/1500000019, doi:10.1561/1500000019.

Sanh, V., Webson, A., Raffel, C., Bach, S.H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Raja, A., Dey, M., Bari, M.S., Xu, C., Thakker, U., Sharma, S.S., Szczechla, E., Kim, T., Chhablani, G., Nayak, N.V., Datta, D., Chang, J., Jiang, M.T., Wang, H., Manica, M., Shen, S., Yong, Z.X., Pandey, H., Bawden, R., Wang, T., Neeraj, T., Rozen, J., Sharma, A., Santilli, A., Févry, T., Fries, J.A., Teehan, R., Scao, T.L., Biderman, S., Gao, L., Wolf, T., Rush, A.M., 2022. Multitask prompted training enables zero-shot task generalization, in: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, OpenReview.net. URL: https://openreview.net/forum?id=9Vrb9D0WI4.

Scao, T.L., Fan, A., Akiki, C., Pavlick, E., Ilic, S., Hesslow, D., Castagné, R., Luccioni, A.S., Yvon, F., Gallé, M., Tow, J., Rush, A.M., Biderman, S., Webson, A., Ammanamanchi, P.S., Wang, T., Sagot, B., Muennighoff, N., del Moral, A.V., Ruwase, O., Bawden, R., Bekman, S., McMillan-Major, A., Beltagy, I., Nguyen, H., Saulnier, L., Tan, S., Suarez, P.O., Sanh, V., Laurençon, H., Jernite, Y., Launay, J., Mitchell, M., Raffel, C., Gokaslan, A., Simhi, A., Soroa, A., Aji, A.F., Alfassy, A., Rogers, A., Nitzav, A.K., Xu, C., Mou, C., Emezue, C., Klamm, C., Leong, C., van Strien, D., Adelani, D.I., et al., 2022. BLOOM: A 176b-parameter open-access multilingual language model. CoRR abs/2211.05100. URL: https://doi.org/10.48550/arXiv.2211.05100, doi:10.48550/arXiv.2211.05100, arXiv:2211.05100.

Shen, Y., Song, K., Tan, X., Li, D., Lu, W., Zhuang, Y., 2023. Hugginggpt: Solving AI tasks with chatgpt and its friends in huggingface. CoRR abs/2303.17580. URL: https://doi.org/10.48550/arXiv.2303.17580, doi:10.48550/arXiv.2303.17580, arXiv:2303.17580.

Shi, F., Suzgun, M., Freitag, M., Wang, X., Srivats, S., Vosoughi, S., Chung, H.W., Tay, Y., Ruder, S., Zhou, D., Das, D., Wei, J., 2022. Language models are multilingual chain-of-thought reasoners. CoRR abs/2210.03057. URL: https://doi.org/10.48550/arXiv.2210.03057, doi:10.48550/arXiv.2210.03057, arXiv:2210.03057.

Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., Yih, W., 2023. REPLUG: retrieval-augmented black-box language models. CoRR abs/2301.12652. URL: https://doi.org/10.48550/arXiv.2301.12652, doi:10.48550/arXiv.2301.12652, arXiv:2301.12652.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., Hashimoto, T.B., 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., Lample, G., 2023. Llama: Open and efficient foundation language models. CoRR abs/2302.13971. URL: https://doi.org/10.48550/arXiv.2302.13971, doi:10.48550/arXiv.2302.13971, arXiv:2302.13971.

Wei, J., Bosma, M., Zhao, V.Y., Guu, K., Yu, A.W., Lester, B., Du, N., Dai, A.M., Le, Q.V., 2022a. Finetuned language models are zero-shot learners, in: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, OpenReview.net. URL: https://openreview.net/forum?id=gEZrGCozdqR.

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E.H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., Fedus, W., 2022b. Emergent abilities of large language models. CoRR abs/2206.07682. URL: https://doi.org/10.48550/arXiv.2206.07682, doi:10.48550/arXiv.2206.07682, arXiv:2206.07682.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E.H., Le, Q.V., Zhou, D., 2022c. Chain-of-thought prompting elicits reasoning in large language models, in: NeurIPS. URL: http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.

West, C.G., 2023. AI and the FCI: can chatgpt project an understanding of introductory physics? CoRR abs/2303.01067. URL: https://doi.org/10.48550/arXiv.2303.01067, doi:10.48550/arXiv.2303.01067, arXiv:2303.01067.

Wu, C., Yin, S., Qi, W., Wang, X., Tang, Z., Duan, N., 2023. Visual chatgpt: Talking, drawing and editing with visual foundation models. CoRR abs/2303.04671. URL: https://doi.org/10.48550/arXiv.2303.04671, doi:10.48550/arXiv.2303.04671, arXiv:2303.04671.

Xie, T., Wu, C.H., Shi, P., Zhong, R., Scholak, T., Yasunaga, M., Wu, C., Zhong, M., Yin, P., Wang, S.I., Zhong, V., Wang, B., Li, C., Boyle, C., Ni, A., Yao, Z., Radev, D., Xiong, C., Kong, L., Zhang, R., Smith, N.A., Zettlemoyer, L., Yu, T., 2022. Unifiedskg: Unifying and multi-tasking structured knowledge grounding with text-to-text language models, in: Goldberg, Y., Kozareva, Z., Zhang, Y. (Eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, Association for Computational Linguistics. pp. 602–631. URL: https://aclanthology.org/2022.emnlp-main.39.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Jiang, D., 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv:2304.12244.

Xu, H., Chen, Y., Du, Y., Shao, N., Wang, Y., Li, H., Yang, Z., 2022. Zeroprompt: Scaling prompt-based pretraining to 1, 000 tasks improves zero-shot generalization, in: Goldberg, Y., Kozareva, Z., Zhang, Y. (Eds.), Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, Association for Computational Linguistics. pp. 4235–4252. URL: https://aclanthology.org/2022.findings-emnlp.312.

Yang, W., Xie, Y., Lin, A., Li, X., Tan, L., Xiong, K., Li, M., Lin, J., 2019. End-to-end open-domain question answering with bertserini, in: Ammar, W., Louis, A., Mostafazadeh, N. (Eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Demonstrations, Association for Computational Linguistics. pp. 72–77. URL: https://doi.org/10.18653/v1/n19-4013, doi:10.18653/v1/n19-4013.

Yang, X., Li, Y., Zhang, X., Chen, H., Cheng, W., 2023. Exploring the limits of chatgpt for query or aspect-based text summarization. CoRR abs/2302.08081. URL: https://doi.org/10.48550/arXiv.2302.08081, doi:10.48550/arXiv.2302.08081, arXiv:2302.08081.

Yu, W., Iter, D., Wang, S., Xu, Y., Ju, M., Sanyal, S., Zhu, C., Zeng, M., Jiang, M., 2023. Generate rather than retrieve: Large language models are strong context generators, in: The Eleventh International Conference on Learning Representations. URL: https://openreview.net/forum?id=fB0hRu9GZUS.

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M.T., Li, X., Lin, X.V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P.S., Sridhar, A., Wang, T., Zettlemoyer, L., 2022. OPT: open pre-trained transformer language models. CoRR abs/2205.01068. URL: https://doi.org/10.48550/arXiv.2205.01068, doi:10.48550/arXiv.2205.01068, arXiv:2205.01068.

Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., Smola, A., 2023. Multimodal chain-of-thought reasoning in language models. CoRR abs/2302.00923. URL: https://doi.org/10.48550/arXiv.2302.00923, doi:10.48550/arXiv.2302.00923, arXiv:2302.00923.

# A Datasets and Splits

- Fool Me Twice (FM2) [Eisenschlos et al., 2021] contains a set of claims with evidence that were originally scarped from Wikipedia.

- Natural Questions Over Tables (NQ-Table) [Herzig et al., 2021] were mined from real Google search queries and the answers are spans in Wikipedia tables identified by human annotators.
- Multi-Subject Multi-Choice Dataset for Medical domain (MedMC-QA) [Pal et al., 2022] contains a set of real-world medical entrance exam questions and answers.
- Multimodal Reasoning for Science Question Answering (ScienceQA) [Lu et al., 2022] consists of multimodal multiple-choice questions with a diverse set of science topics.
Table 5 shows the dataset splits and statistics.

Table 5: Datasets splits and statistics. For MedMC-QA, labels in the test are hidden, so the model performance is evaluated on the validation set.


<table>
  <thead>
    <tr>
        <th>Datasets</th>
        <th>Domain</th>
        <th>Train</th>
        <th>Valid</th>
        <th>Test</th>
        <th>Test labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>FM2 [Eisenschlos et al., 2021]</td>
        <td>Factual</td>
        <td>10,419</td>
        <td>1,169</td>
        <td>1,380</td>
        <td>Public</td>
    </tr>
    <tr>
        <td>NQ-Table [Herzig et al., 2021]</td>
        <td>Tabular</td>
        <td>9,594</td>
        <td>1,068</td>
        <td>959</td>
        <td>Public</td>
    </tr>
    <tr>
        <td>MedMC-QA [Pal et al., 2022]</td>
        <td>Medical</td>
        <td>160,869</td>
        <td>4,183</td>
        <td>6,150</td>
        <td>Private</td>
    </tr>
    <tr>
        <td>ScienceQA [Lu et al., 2022]</td>
        <td>Multimodal</td>
        <td>12,726</td>
        <td>4,241</td>
        <td>4,241</td>
        <td>Public</td>
    </tr>
  </tbody>
</table>


## B Implementation Details

We use LLaMa-7B [Touvron et al., 2023] as our backbone models to implement the PKG modules. We use AdamW as the optimizer, with 10% warmup steps. We use 8 V100 GPUs for training PKG modules. The open-source code LLaMa-X<sup>3</sup> is widely used in our experiments. We refer to more individual implementation details in Table 6.

We implement other baseline methods based on the following repositories:

- BM25 + GPT3.5: https://github.com/castorini/pyserini
- REPLUG + GPT3.5: https://github.com/facebookresearch/DPR/tree/main
- CoT + GPT3.5: https://github.com/kojima-takeshi188/zero_shot_cot
- GenRead + GPT3.5: https://github.com/wyu97/GenRead

## C All Experiment Results of Figure 3

In Figure 3, we compare our PKGs framework with the direct generation on various types of LMs. We include all results in Table 7.

## D Case Studies

Table 8 9 10 11 presents more examples of background documents generated by our baseline methods (CoT and GenRead) and PKGs for different tasks. We can notice that our PKGs can provide clues for LLMs to answer specific questions. Table 12 also compares our PKGs with the retrieval-based

<sup>3</sup>https://github.com/AetherCortex/Llama-X

Table 6: Hyperparameters settings of our PKG modules on different tasks.


<table>
  <thead>
    <tr>
        <th>Settings</th>
        <th>FM2</th>
        <th>NQ-Table</th>
        <th>MedMC-QA</th>
        <th>ScienceQA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>Peak learning rate</td>
        <td>2e-5</td>
        <td>2e-5</td>
        <td>2e-5</td>
        <td>2e-5</td>
    </tr>
    <tr>
        <td>Total batch size</td>
        <td>64</td>
        <td>32</td>
        <td>32</td>
        <td>32</td>
    </tr>
    <tr>
        <td>Total training epochs</td>
        <td>3</td>
        <td>10</td>
        <td>3</td>
        <td>5</td>
    </tr>
    <tr>
        <td>Warmup Schedule</td>
        <td>cosine</td>
        <td>cosine</td>
        <td>cosine</td>
        <td>cosine</td>
    </tr>
    <tr>
        <td>Warmup ratio</td>
        <td>0.1</td>
        <td>0.1</td>
        <td>0.1</td>
        <td>0.1</td>
    </tr>
  </tbody>
</table>

Table 7: All experiments results of Figure 3 for different sizes of LMs.


<table>
  <thead>
    <tr>
        <th>Methods</th>
        <th>FM2</th>
        <th>NQ-Table</th>
        <th>MedMC-QA</th>
        <th>ScienceQA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>PKG-Davinci</td>
        <td><strong>67.3</strong></td>
        <td><strong>28.8</strong></td>
        <td><strong>47.4</strong></td>
        <td><strong>80.76</strong></td>
    </tr>
    <tr>
        <td>PKG-Curie</td>
        <td>52.0</td>
        <td>0.1</td>
        <td>23.8</td>
        <td>32.87</td>
    </tr>
    <tr>
        <td>PKG-Babbage</td>
        <td>45.4</td>
        <td>0.1</td>
        <td>20.0</td>
        <td>35.77</td>
    </tr>
    <tr>
        <td>PKG-Ada</td>
        <td>38.0</td>
        <td>0.0</td>
        <td>20.6</td>
        <td>29.76</td>
    </tr>
    <tr>
        <td>Direct-Davinci</td>
        <td>59.4</td>
        <td>16.9</td>
        <td>44.4</td>
        <td>71.66</td>
    </tr>
    <tr>
        <td>Direct-Curie</td>
        <td>50.8</td>
        <td>0.0</td>
        <td>20.3</td>
        <td>36.76</td>
    </tr>
    <tr>
        <td>Direct-Babbage</td>
        <td>39.5</td>
        <td>0.0</td>
        <td>20.6</td>
        <td>41.08</td>
    </tr>
    <tr>
        <td>Direct-Ada</td>
        <td>42.6</td>
        <td>0.0</td>
        <td>20.3</td>
        <td>26.79</td>
    </tr>
  </tbody>
</table>


methods. We can find that the retrieval methods cannot provide relevant background documents to answer the question.

## E Errors

Table 13 presents a hallucination error of our PKGs.

Table 8: Examples of hallucination errorsbackground documents generated by our baseline methods and PKGs for FM2. Clues to answering the input are highlighted in blue within the documents. The information that will disrupt LLMs is highlighted in red.


<table>
  <tbody>
    <tr>
        <td>Input (FM2)</td>
        <td>The Natural is a book about Roy Hobbs a natural southpaw boxer who goes on to win the heavyweight title from Boom Boom Mancini.</td>
    </tr>
    <tr>
        <td>CoT Guiding</td>
        <td>The first part of the claim is that The Natural is a book. This is true - The Natural is a novel written by Bernard Malamud. The second part of the claim is that <mark>the book is about Roy Hobbs, a natural southpaw boxer. This is also true</mark> - the novel follows the character of Roy Hobbs, a talented young baseball player who is drafted by a major league team. The third part of the claim is that Hobbs goes on to win the heavyweight title from Boom Boom Mancini. This is false - in the novel, Hobbs does not win the heavyweight title from Boom Boom Mancini.</td>
    </tr>
    <tr>
        <td>GenRead Guiding</td>
        <td><mark>The Natural is a 1984 American sports drama film based on Bernard Malamud’s 1952 baseball novel of the same name</mark>, directed by Barry Levinson and starring Robert Redford, Glenn Close, and Robert Duvall. The film, like the book, recounts the experiences of Roy Hobbs, an extraordinarily talented left-handed pitcher, who becomes a late-blooming professional baseball player with the New York Knights.</td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>This is <mark>a story about a baseball player, Roy Hobbs</mark>, who is shot by a woman, Harriet Bird, and then recovers to become a great player.</td>
    </tr>
    <tr>
        <td>Input (FM2)</td>
        <td>Goodbye, Mr. Chips’ story happens in Canada.</td>
    </tr>
    <tr>
        <td>CoT Guiding</td>
        <td>First, let’s recall what the story is about. <mark>The story is about a schoolteacher, Mr. Chips, who spends his entire life teaching at a boys’ boarding school in England.</mark> Now, let’s think about whether the story could have taken place in Canada. Could a schoolteacher in Canada spend his entire life teaching at a boys’ boarding school? It’s possible, but it seems unlikely. Could a schoolteacher in Canada spend his entire life teaching at any kind of school? Yes, that’s possible. Therefore, the claim that the story happens in Canada is false.</td>
    </tr>
    <tr>
        <td>GenRead Guiding</td>
        <td>Goodbye, Mr. Chips is a novel by James Hilton, first published in 1934. The novel tells the story of a schoolteacher, Mr. Chipping, who is nicknamed "Chips". The novel was Hilton’s best-selling book of all time, and was adapted into two films and a television series. The novel is set in the fictional public school of Brookfield, in the English county of Leicestershire. <mark>However, in the film adaptations, the school is moved to the Canadian Rockies.</mark></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Goodbye, Mr. Chips is a 1934 British drama film based on the 1934 novel of the same name by James Hilton, which tells <mark>the story of a school headmaster at an English boys’ school</mark> from 1870 to 1935.</td>
    </tr>
  </tbody>
</table>

Table 9: Examples of background documents generated by our baseline methods and PKGs for NQ-Table. Clues to answering the input are highlighted in <mark>blue</mark> within the documents. The information that will disrupt LLMs is highlighted in <mark>red</mark>.


<table>
  <tbody>
    <tr>
        <td>Input (NQ-Table)</td>
        <td>Batman The Enemy Within episode 5 release date</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>CoT Guiding</td>
        <td>1. When is episode 5 of Batman the enemy released? 2. What is the release date of episode 5 of Batman the enemy? <mark>The release date of episode 5 of Batman the enemy is October 10, 2016.</mark></td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>GenRead Guiding</td>
        <td>Batman: The Enemy Within is an episodic point-and-click graphic adventure video game developed and published by Telltale Games and distributed by Warner Bros. Interactive Entertainment under its DC Entertainment label. The game is a sequel to 2016's "Batman: The Telltale Series", and was released for Microsoft Windows, macOS, PlayStation 4, Xbox One, Android, and iOS. The first episode was released on August 8, 2017, with the fifth and <mark>final episode released on March 27, 2018.</mark></td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Batman: The Enemy Within<br/>Ep</td>
        <td>Title</td>
        <td>Directed by</td>
        <td>Release date<br/>1</td>
        <td>The Enigma</td>
        <td>Kent Mudle</td>
        <td>August 8, 2017<br/>2</td>
        <td>The Pact</td>
        <td>Matthew Leach</td>
        <td>October 3, 2017<br/>3</td>
        <td>Fractured Mask</td>
        <td>Sean Manning</td>
        <td>November 21, 2017<br/>4</td>
        <td>What Ails You</td>
        <td>Chris Rieser</td>
        <td>January 23, 2018<br/>5</td>
        <td>Same Stitch</td>
        <td>Kent Mudle</td>
        <td><mark>March 27, 2018</mark></td>
    </tr>
    <tr>
        <td>Input (NQ-Table)</td>
        <td>the old man and the sea page count</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>CoT Guiding</td>
        <td><mark>The old man is fishing and he catches a big fish. He struggles to bring the fish in, and finally does. Now, he has to get the fish home. So, he ties it to his boat and starts rowing. The old man is rowing for a long time and finally makes it back to shore. He drags the fish up onto the beach and starts to cut it open. Now, we're at the part where the old man is cutting open the fish. He's cut open the belly and is starting to pull out the guts. He's got a knife in one hand and is pulling out the guts with the other. The old man is still cutting open the fish. He's almost done now. He's got the knife in one hand and the guts in the other. He's about to pull out the last of the guts when he suddenly feels something tugging on his line. The old man has been pulled out to sea by the big fish! He's struggling to get back to shore, but the fish is too strong. The old man is fighting for his life now. The old man is still struggling to get back to shore. He's getting tired now, and the fish is still pulling him out to sea. The old man knows he's going to die soon. The old man finally dies and is pulled out to sea by the big fish.</mark></td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>GenRead Guiding</td>
        <td><mark>The Old Man and the Sea is a novel by Ernest Hemingway that was first published in 1952. The novel tells the story of an aging Cuban fisherman who struggles to catch a giant marlin. The Old Man and the Sea was published in 1952 in Life magazine. The novel was adapted into a 1958 film of the same name, starring Spencer Tracy. The Old Man and the Sea won the Pulitzer Prize for Fiction in 1953.</mark></td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>The Old Man and the Sea<br/>Author</td>
        <td>Language</td>
        <td>Genre</td>
        <td>Pages<br/>Ernest Hemingway</td>
        <td>English</td>
        <td>Literary Fiction</td>
        <td><mark>127</mark></td>
        <td colspan="12"></td>
    </tr>
  </tbody>
</table>

Table 10: Examples of background documents generated by our baseline methods and PKGs for MedMC-QA. Clues to answering the input are highlighted in <mark>blue</mark> within the documents. The information that will disrupt LLMs is highlighted in <mark>red</mark>.


<table>
  <tbody>
    <tr>
        <td>Input (MedMC-QA)</td>
        <td>Calcium ions triggers muscle contraction by binding to: Options: (A) Actin (B) Myosin (C) Troponin (D) Tropomyosin (Correct Answer: D)</td>
    </tr>
    <tr>
        <td>CoT Guiding</td>
        <td>First, we need to understand what a muscle contraction is. A muscle contraction is when the muscle fibers in the muscle shorten. This is caused by the sliding of the actin and myosin filaments past each other. Next, we need to understand what role calcium ions play in muscle contraction. Calcium ions are needed for the muscle contraction to occur because they are needed to trigger the release of the myosin head from the actin filament. <mark>So, the answer is (C) Troponin.</mark></td>
    </tr>
    <tr>
        <td>GenRead Guiding</td>
        <td>Calcium ions play an important role in muscle contraction. <mark>When calcium ions bind to troponin</mark>, they cause a conformational change in the protein that allows myosin and actin to interact. This interaction between myosin and actin is what causes muscle contraction.</td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Troponin is a complex of three proteins that are attached to actin. <mark>Calcium ions bind to troponin-C</mark>, which then causes the troponin-I and troponin-T to shift and allow the tropomyosin to move off of the active site of the myosin so that the myosin head can then bind to actin and cause contraction.</td>
    </tr>
    <tr>
        <td>Input (MedMC-QA)</td>
        <td>Which of the following statements is true regarding medical aboion?<br/>Options: (A) Ultrasound should be done in all cases (B) If the patient has an IUCD in-situ, it doesn’t need to be removed (C) Can only be done up to 72 days (D) Only a person ceified under MTP act can perform medical termination of pregnancy (Correct Answer: D)</td>
    </tr>
    <tr>
        <td>CoT Guiding</td>
        <td>First, we need to understand what medical abortion is. Medical abortion is the use of medication to end a pregnancy. Next, we need to consider the options given. Option A is not necessarily true - ultrasound may not be needed in all cases, but it can be helpful in determining how far along the pregnancy is. Option B is also not necessarily true - in some cases, the IUCD (intrauterine contraceptive device) may need to be removed before the medical abortion can be performed. <mark>Option C is generally true - medical abortions can usually be performed up to 72 days after the last menstrual period. Option D is also generally true - only a certified healthcare provider can perform a medical abortion.</mark></td>
    </tr>
    <tr>
        <td>GenRead Guiding</td>
        <td><mark>Medical abortion is a safe and effective way to end a pregnancy. It can be done up to 72 days after the first day of the last menstrual period. A medical abortion can be done with pills or with a shot.</mark></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Medical aboion can be done up to 9 weeks and 6 days. Ultrasound is not required for medical aboion. If the patient has an IUCD in-situ, it should be removed. <mark>Only a person ceified under MTP act can perform medical termination of pregnancy.</mark></td>
    </tr>
  </tbody>
</table>

Table 11: Examples of background documents generated by our baseline methods and PKGs for ScienceQA. Clues to answering the input are highlighted in <mark>blue</mark> within the documents. The information that will disrupt LLMs is highlighted in <mark>red</mark>.


<table>
  <tbody>
    <tr>
        <td>Input (ScienceQA)<br/><img src="page_20_image_1_v2.jpg" alt="saddle-billed stork"></td>
        <td>Which animal’s neck is also adapted for hunting prey while keeping the rest of its body still? Context: Saddle-billed storks live near wetlands and lakes. They eat mostly fish. The ’s neck helps it grab fish while keeping the rest of its body still. If the stork had to move its body, it might scare the fish away. Figure: saddle-billed stork. A black and white bird is standing in the grass. Options: (A) northern pintail (B) black-headed heron (Correct Answer: B)</td>
    </tr>
    <tr>
        <td>CoT Guiding</td>
        <td>Option A is incorrect because northern pintails are ducks and do not have necks adapted for hunting prey. <mark>Option B is correct because black-headed herons are wading birds that use their long necks to help them catch fish.</mark></td>
    </tr>
    <tr>
        <td>GenRead Guiding</td>
        <td><mark>The answer to the question is option (B), the painted stork.</mark> The painted stork’s neck is adapted for hunting prey while keeping the rest of its body still, which helps it grab fish while minimizing movement that could scare the fish away.</td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Look at the picture of the saddle-billed stork. The saddle-billed stork has a long neck. Its neck is adapted for hunting prey while keeping the rest of its body still. Now look at each animal. Figure out which animal has a similar adaptation. <mark>The black-headed heron has a long neck. Its neck is adapted for hunting prey while keeping the rest of its body still.</mark> The northern pintail has a short neck. Its neck is not adapted for hunting prey while keeping the rest of its body still.</td>
    </tr>
    <tr>
        <td>Input (ScienceQA)<br/></td>
        <td></td>
    </tr>
    <tr>
        <td>Average precipitation in Christchurch, New Zealand</td>
        <td></td>
    </tr>
  </tbody>
</table>
tsv
<table>
  <tbody>
    <tr>
        <td>Month</td>
        <td>Precipitation (mm)</td>
    </tr>
    <tr>
        <td>Jan</td>
        <td>45</td>
    </tr>
    <tr>
        <td>Feb</td>
        <td>42</td>
    </tr>
    <tr>
        <td>Mar</td>
        <td>58</td>
    </tr>
    <tr>
        <td>Apr</td>
        <td>52</td>
    </tr>
    <tr>
        <td>May</td>
        <td>70</td>
    </tr>
    <tr>
        <td>Jun</td>
        <td>62</td>
    </tr>
    <tr>
        <td>Jul</td>
        <td>68</td>
    </tr>
    <tr>
        <td>Aug</td>
        <td>60</td>
    </tr>
    <tr>
        <td>Sep</td>
        <td>48</td>
    </tr>
    <tr>
        <td>Oct</td>
        <td>45</td>
    </tr>
    <tr>
        <td>Nov</td>
        <td>48</td>
    </tr>
    <tr>
        <td>Dec</td>
        <td>55</td>
    </tr>
  </tbody>
</table>
	Which month is the wettest on average in Christchurch? Context: Use the graph to answer the question below. A series of photographs showing a variety of colorful objects. Options: (A) April (B) December (C) May (Correct Answer: C)
CoT Guiding	<mark>The wettest month on average in Christchurch is December.</mark>
GenRead Guiding	<mark>The graph shows that December is the wettest month on average in Christchurch.</mark>
PKG Guiding	Scientists record data from places around the world. Precipitation, or rain and snow, is one type of climate data. A bar graph can be used to show the average amount of precipitation each month. Months with taller bars have more precipitation on average. To describe the average precipitation trends in Christchurch, look at the graph. Choice "Apr" is incorrect. <mark>Choice "May" is incorrect.</mark> Choice "Dec" is incorrect. <mark>May has an average monthly precipitation of about 70 millimeters. This is higher than in any other month. So, May is the wettest month on average.</mark>
```
```

Table 12: Examples of background documents generated by our baseline methods and PKGs for NQ-Table. Clues to answering the input are highlighted in blue within the documents.


<table>
  <thead>
    <tr>
        <th>Input (NQ-Table)</th>
        <th colspan="19">Batman The Enemy Within episode 5 release date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>BM25 Retrieved</td>
        <td>is either visited by Bruce or decides to become the hero’s archenemy. However, if he was a criminal, he is shown playing with a doll version of Bruce, which he promises to see again. All episodes below were released for Windows, macOS, PlayStation, Xbox One, and mobile platforms on the dates given. The Nintendo Switch version was released as a single package on October 2, 2018. "Batman: The Enemy Within" was considered to be an improvement over its predecessor, earning praise for its story, choices, action sequences, and portrayal of the Batman mythos. However, the presence of technical issues, and</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>REPLUG Retrieved</td>
        <td>Babylon 5: The Legend of the Rangers Babylon 5: The Legend of the Rangers (subtitled: To Live and Die in Starlight) is the fifth telefilm set in the "Babylon 5" universe (not including the pilot,). Originally airing January 19, 2002 on the Sci Fi Channel, it was written by J. Michael Straczynski and directed by Mike Vejar. Though shot as a pilot for a possible new series, it aired opposite NFL playoffs and the subsequent poor ratings led to it not being picked up. As the Shadow War ended, hundreds of civilizations were devastated. It is up to the</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>Batman: The Enemy Within<br/>Ep</td>
        <td>Title</td>
        <td>Directed by</td>
        <td>Release date<br/>1</td>
        <td>The Enigma</td>
        <td>Kent Mudle</td>
        <td>August 8, 2017<br/>2</td>
        <td>The Pact</td>
        <td>Matthew Leach</td>
        <td>October 3, 2017<br/>3</td>
        <td>Fractured Mask</td>
        <td>Sean Manning</td>
        <td>November 21, 2017<br/>4</td>
        <td>What Ails You</td>
        <td>Chris Rieser</td>
        <td>January 23, 2018<br/>5</td>
        <td>Same Stitch</td>
        <td>Kent Mudle</td>
        <td><mark>March 27, 2018</mark></td>
    </tr>
    <tr>
        <td>Input (NQ-Table)</td>
        <td>the old man and the sea page count</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>BM25 Retrieved</td>
        <td>by Magneto’s Sentinels for plotting against Magneto. In "JLA/Avengers", Count Nefaria is seen in #4 among the other villains enthralled by Krona to defend his stronghold. He is shown fighting Superman in a panel spreading across two-pages. In the pages of "Old Man Logan", the elderly Logan awoke on Earth-616 and had a flashback to where Count Nefaria, Red Skull, Baron Blood, Spiral, and Whirlwind were standing over the dead bodies of the superheroes the day when the villains rose and the heroes fell. Count Nefaria appears in the "What If?" story "What If the X-Men Died on their First</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>REPLUG Retrieved</td>
        <td>The Old Man and the Sea The Old Man and the Sea is a short novel written by the American author Ernest Hemingway in 1951 in Cuba, and published in 1952. It was the last major work of fiction by Hemingway that was published during his lifetime. One of his most famous works, it tells the story of Santiago, an aging Cuban fisherman who struggles with a giant marlin far out in the Gulf Stream off the coast of Cuba. In 1953, "The Old Man and the Sea" was awarded the Pulitzer Prize for Fiction, and it was cited by</td>
        <td colspan="18"></td>
    </tr>
    <tr>
        <td>PKG Guiding</td>
        <td>The Old Man and the Sea<br/>Author</td>
        <td>Language</td>
        <td>Genre</td>
        <td>Pages<br/>Ernest Hemingway</td>
        <td>English</td>
        <td>Literary Fiction</td>
        <td><mark>127</mark></td>
        <td colspan="12"></td>
    </tr>
  </tbody>
</table>

Table 13: Examples of hallucination errors. <mark>red</mark>: indicates the errors.


<table>
  <thead>
    <tr>
        <th>Input</th>
        <th colspan="4">who won game 4 of the 2000 nba finals (Correct Answer: Los Angeles Lakers)</th>
    </tr>
    <tr>
        <th> </th>
        <th colspan="4">2000 NBA Finals</th>
    </tr>
    <tr>
        <th> </th>
        <th>Game</th>
        <th>Home Team</th>
        <th>Result</th>
        <th>Road Team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td rowspan="4">PKG Guiding</td>
        <td>Game 1</td>
        <td>Los Angeles Lakers</td>
        <td>116-86</td>
        <td>Portland Trail Blazers</td>
    </tr>
    <tr>
        <td>Game 2</td>
        <td>Los Angeles Lakers</td>
        <td>100-86</td>
        <td>Portland Trail Blazers</td>
    </tr>
    <tr>
        <td>Game 3</td>
        <td>Portland Trail Blazers</td>
        <td>86-80</td>
        <td>Los Angeles Lakers</td>
    </tr>
    <tr>
        <td>Game 4</td>
        <td>Portland Trail Blazers</td>
        <td><mark>89-78</mark></td>
        <td>Los Angeles Lakers</td>
    </tr>
  </tbody>
</table>


# F Instruction Formats

## - FM2:

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Generate a background document from Wikipedia to support or refute the statement.

### Input:

Statement: xxx

### Response:

<background fact>

## - NQ-Table:

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Generate a background table from Wikipedia to answer the given question.

### Input:

Question: xxx

### Response:

<background table>

## -MedMC-QA

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Generate a background document from the medical domain to answer the given question.

### Input:

Question: xxx

### Response:

<background medical knowledge>

-ScienceQA: We follow the "QCM-LE" format in MM-CoT [Zhang et al., 2023], where "Q" is the question, "C" is the context, "M" is the choices, "L" is the lecture and "E" is the explanation. Please refer to the paper of MM-CoT for more details.

# G Prompt

- FM2: "background \n\n claim: query \n\n Is the claim true or false?"

- NQ-Table: "Refer to the background below and answer the following question with just a few words. The answer should be less than 5 words.\n\n Background: background\n\n Question: question\n\n Answer:"

- MedMC-QA: "Refer to the medical background below and answer the following question.\n Background: background\n\nQuestion: question\nOptions: options\n\nPlease only choose the answer from options. The answer is:"

- ScienceQA: "Question: question\nBECAUSE: background\nOptions: options\nPlease only choose the answer from options. The answer is:"