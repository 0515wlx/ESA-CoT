# Unshackling Context Length: An Efficient Selective Attention Approach through Query-Key Compression

**Haoyu Wang**¹*, **Tong Teng**¹, **Tianyu Guo**¹, **An Xiao**¹, **Duyu Tang**², **Hanting Chen**¹, **Yunhe Wang**¹
¹Huawei Noah’s Ark Lab ²Huawei CBG
{wanghaoyu50, tengton1, tianyu.guo, an.xiao, tangduy, chenhanting, yunhe.wang}@huawei.com

## Abstract

Handling long-context sequences efficiently remains a significant challenge in large language models (LLMs). Existing methods for token selection in sequence extrapolation either employ a permanent eviction strategy or select tokens by chunk, which may lead to the loss of critical information. We propose Efficient Selective Attention (ESA), a novel approach that extends context length by efficiently selecting the most critical tokens at the token level to compute attention. ESA reduces the computational complexity of token selection by compressing query and key vectors into lower-dimensional representations. We evaluate ESA on long sequence benchmarks with maximum lengths up to 256k using open-source LLMs with context lengths of 8k and 32k. ESA outperforms other selective attention methods, especially in tasks requiring the retrieval of multiple pieces of information, achieving comparable performance to full-attention extrapolation methods across various tasks, with superior results in certain tasks.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across a variety of natural language processing (NLP) tasks. One of the emerging trends in this area is the increasing emphasis on extending the context length of LLMs to handle tasks like contextual learning and retrieval-augmented generation (Dong et al., 2022; Gao et al., 2023). However, this extension comes with its own set of challenges, particularly in terms of the Out-Of-Distribution (OOD) extrapolation on context length, massive memory demanded by key-value (KV) cache and the quadratic computational complexity of the attention mechanism. In this paper, we focus on mitigating the OOD issues and reducing the computational cost of LLM inference on long-context sequences, especially for models which adopt Rotary Position Embedding (RoPE) (Su et al., 2024).

Extrapolation methods based on positional embeddings address the OOD issues (Chen et al., 2023b; bloc97, 2023; Peng et al., 2023), but they are not targeted at reducing computational cost. To improve inference efficiency, methods focusing on selective attention are developed, exploiting the inherent sparsity of attention matrices (Zhang et al., 2024b; Jiang et al., 2024). These methods usually restrict the number of selected tokens to within the pre-training context length. Early approaches focus on local attention, sometimes introduce global tokens, often neglecting long-distance tokens (Belaty et al., 2022; Xiao et al., 2023; Han et al., 2023). Recent chunk-based methods, such as InFLLM, LongHeads, and Landmark Attention (Xiao et al., 2024; Lu et al., 2024; Mohsantah and Jaggi, 2023), have aimed to extend the context length by chunking input prompts into fixed-size blocks and selecting the top-ranked chunks. These approaches, while reducing computational complexity, often suffer from a lack of flexibility. In contrast, token-level selection methods provide more flexibility by selectively attending to tokens at a finer granularity. However, existing methods usually perform token selection only in the decoding stage and permanently evict certain tokens to manage memory usage as well as reduce computation, facing high latency during prefiling and significant information loss (Zhang et al., 2024b; Liu et al., 2024; Ge et al., 2023; Li et al., 2024; Cai et al., 2024). To maintain accuracy while reducing computation, it is essential to perform token selection over all preceding tokens during both prefiling and decoding. Nevertheless, selecting tokens individually could introduce significant computational overhead, raising the challenge of how to efficiently identify the most crucial tokens while minimizing
## 2 Related Work

### Position Extrapolation
Following the introduction of RoPE (Su et al., 2024), great efforts have been imposed to extend the context length by modifying the position embeddings (PE). Position interpolation (Chen et al., 2023b; kaiokende, 2023) extends the context length by interpolating positional indices within the constraints of pre-training. The NTK-aware method (bloc97, 2023; Rozière et al., 2023; Liu et al., 2023) introduces a nonlinear interpolation strategy by increasing the base parameter of RoPE. YaRN (Peng et al., 2023) proposes a method for frequency interpolation of RoPE dimensions, where higher frequency dimensions are extrapolated, and lower frequency dimensions are interpolated. Further improvements (Chen et al., 2023a; Ding et al., 2024) exploit the dynamics in position extrapolation. Another group of work redesigns the relative position matrix to overcome the OOD issue (Su, 2023; Jin et al., 2024; Erhan et al., 2024). These methods extend the context length but still compute the full attention matrix for inference thus fail to reduce the computational cost. To achieve better performance, some require fine-tuning with a certain amount of long-context data.

### Selective Attention
Selective attention mechanisms aim to mitigate the computational cost of processing long sequences by selecting only the most relevant tokens for attention computation. Approaches like Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 2020) use fixed or adaptive sparse patterns, while Han et al., 2023; Xiao et al., 2023 introduce A-shaped windows that evict middle tokens. Although these methods lower costs, they often compromise global context understanding due to restricted attention over all tokens. Some methods aim at compressing the KV cache, usually perform token selection only in the decoding stage (Zhang et al., 2022; Liu et al., 2024; Ge et al., 2023) or permanently evicting certain tokens (Xiao et al., 2023; Han et al., 2023; Li et al., 2024b). While effective, they may lose critical contextual information. As for chunk-based methods, Xiao et al., 2024 uses an efficient contextual memory mechanism to select the most relevant chunks for computing attention, and Lu et al., 2024 selects the most relevant chunks for each head separately considering the variance among different heads. Unlimiformer and its adaptation (Bertsch et al., 2024; Ahrabian et al., 2024) segment input during the pre-filling stage, using external memory blocks.
but remain computationally expensive and require additional training or architecture modifications. In contrast, our method performs efficient token-level selection in both prefiling and decoding stages, without discarding tokens permanently.

## 3 Method

Conventionally, each token attends to all preceding tokens to compute the attention output during LLM inference, leading to significant computational overhead when processing long-context sequences. We introduce ESA, which effectively identifies a subset of the most crucial tokens for attention computation, thereby improving the efficiency of the model.

### 3.1 Token Selection

Following previous work (Xiao et al., 2023; Han et al., 2023; Xiao et al., 2024) on extending the context window of LLMs, the preceding tokens (P) are segmented into three parts, as depicted in Figure 1: initial tokens (I), middle tokens (M), and local tokens (L) with respective lengths denoted as lI, lM, and lL. With a massive number of preceding tokens, lM is considerably larger than lI and lL. The initial tokens and local tokens with fixed length will always be selected due to their significance, as highlighted by Xiao et al. (2023) and Han et al. (2023). We then apply ESA to select the top-k critical tokens Mk from the middle tokens M. Specifically, LLM inference consists of two stages: prefiling and decoding. In the prefiling stage, all the input tokens are encoded, and the keys as well as values are computed and cached. We adopt the chunked-prefill method (Agrawal et al., 2023). Compared to the original prefiling method, chunked-prefill approach effectively reduces memory usage. In the decoding stage, the model generates tokens one by one autoregressively. Denote the current tokens (i.e., the current input chunk being processed or the most recently decoded token) as C, the attention output computed in each step between the current tokens and the selected tokens is defined as:

O = Attn(QC, KIMk,L,C | VI, Mk,L,C) (1)

where, QC represents the queries of the current tokens, KIMk,L,C and VIMk,L,C respectively denote the concatenated keys and values of the selected and current tokens.

### Importance Score

Successfully identifying the most crucial tokens requires a function to precisely measure the importance of each token in M w.r.t C. Let m ∈ M be a specific token, we denote the importance score of m as Fs(m; C). Given a predefined number k ≤ lM, k tokens with the highest scores will be selected, as formalized below:
Mk = arg topK(M; Fs, C, k) (2)

For an individual token c ∈ C, the degree to which it attends to a preceding token m can be determined as following:

fs(m; c) = H ∑ h=1 qh,c,kh,m ∈ d^d (3)

where, qh,c, kh,m ∈ d^d denote the query of c and the key of m, respectively, for the h-th head. All of the H d-dimensional attention heads are incorporated, and the selected tokens are shared across all the heads in a certain layer. This score can be used directly in the decoding stage for token selection as C consists of only one token (i.e., lC = 1). While at the prefiling stage, lC is the chunk size, and every token in C contributes to the score Fs(m; C). To derive a unified form of importance score, we first regularize the score w.r.t each c and then take the maximum value across all tokens in C. Eventually, the importance score is formulated as:

Fs(m; C) = maxc∈C {fs(m; c) - maxm′∈M fs(m′; c)} (4)

1. The expression fs(m; c) - maxm′∈M fs(m′; c) indicates that each token in C is constrained relative to the maximum score in M, preventing any individual token in C from exerting a dominating influence on the selection process.2 In the prefiling stage, our goal is to ensure that high-scoring tokens in M are not overlooked. Therefore, we select the highest score of each token in C to represent the score in M by applying max∈C.

### Efficient Selection through Query-Key Compression

The aforementioned scoring method of employing dot product across a considerable number tokens is computationally expensive. To achieve efficient selection, we perform dimensionality reduction on keys and queries. The right-hand side of Equation 3 is equivalent to concatenating H heads followed by performing dot product. That is, fs(m; c) = qk^T km where qc = [q1,c; q2,c; ...; qH,c] and kc = [k1,m; k2,m; ...; kH,m], and qc, km ∈ RdH = H × d. Denote the
Figure 1: (a) In long-context scenarios, the number of middle tokens occupies the majority, while the lengths of the other three parts of tokens are fixed. The importance scores between current tokens and middle tokens are utilized to select the top-k middle tokens. The selected tokens replace the middle tokens for computing attention. (b) The queries from current tokens and keys from middle tokens are compressed into smaller tensors through a linear layer respectively. The dot product of the compressed queries and keys serves as the importance scores. (c) The priority of a middle token being selected is determined by the maximum importance score among itself and several surrounding tokens.

dimensionality reduction on queries and keys as follows:
$$
q_{c} \triangleq \theta_{q}(q_{c}),
$$
$$
k_{m} \triangleq \theta_{k}(k_{m}),
$$
where, $q_{c}', k_{m}' \in \mathbb{R}^{d', d'} \subset \mathbb{R}^{d_{H}}$. The dimension-reduced representation $k_{m}'$ will be cached and reused during the subsequent computation steps. With the lower-dimensional surrogates of queries and keys, the importance score is approximated with
$$
f_{s}(m; c) \approx q_{c}'k_{m}^{T} \quad (6)
$$
The computational cost is therefore reduced compared with using the full-dimensional queries and keys. To maintain accuracy, the lower-dimensional representations should retain the token order as faithfully as possible. To this end, we perform a one-time offline procedure to learn $f_{\theta}$ and $f_{\phi}$ in each layer by minimizing the discrepancy between the importance scores before and after dimensionality reduction, formally:
$$
\min_{\theta_{\phi}} \sum_{\alpha_{j}, \theta_{k} \in \mathcal{M}} \|q_{k}^{T}m - f_{\theta}(q_{c})f_{\phi}(k_{m})\|_{2}^{2} \quad (7)
$$

Proximity Influence. Considering proximity influence, if a token in $M$ has a high importance score, it affects the scores of the surrounding tokens. We propose adaptive selection which is achieved by updating the importance score of a token based on its neighbors. Specifically, the score for the $j$-th token, where $j \in [1, l_{1} + l_{M} - 1]$, is determined as the maximum score within a window of size $\epsilon$.
surrounding it. The importance score of the j-th token, computed using the low-dimensional query and key, is denoted by \( s_j \). The updated score is given by

\[
s'_j = \min \left( j + c_{l+1} + h_{M-1} - \max (j - c_{l}) , \, \epsilon \right) \{ s_w \}
\]

where \( \epsilon \) represents the proximity influence distance, which is used to control the continuity of the selected tokens.

### 3.2 Position Encoding

We follow the extrapolated position encoding settings as described in (Su, Jianlin, 2023; Xiao et al., 2024). The attention weights in Equation 1 can be divided into (a) Long-distance attention of \( C \) with respect to \( I, M \): The positions of tokens in \( C \) are all modified to a fixed position \( w \), and the positions of tokens from \( I, M \) are set to 0. The relative position will always be \( w \); and (b) Local attention of \( C \) with respect to \( L \) and itself: The query and key positions from \( L, C \) are arranged in order of their relative positions (i.e., 0, 1, 2, ..., \( l + l_C - 1 \)). To normalize the attention weights, we fuse the two categories of attention based on their respective weights. The specific algorithm for computing attention is shown in Appendix G.

### 3.3 Complexity Analysis

#### Computational Complexity Analysis

When inferring over a long-context sequence of total length \( S \), using full-dimensional queries and keys for computing the importance score in Equation 3 incurs a time complexity of \( O(S^2 d_H) \). By utilizing low-dimensional queries and keys, the computation is reduced to \( O(S^2 d') \). The additional computation for dimensionality reduction is \( O(S d' d_f + S d') \), which scales linearly with context length and thus marginal to the quadratic savings.

In each step, ESA computes the attention matrix for a constant number of selected tokens. Considering the long sequence scenario where \( M \) occupies the majority of the tokens, this approach significantly reduces computational costs, with only a minor overhead for token selection. Compared to vanilla full attention, the complexity of computing one step of attention can be reduced by ratio \( r \) in Equation 9. The derivation can be found in Appendix A.

\[
r = \frac{2d_f + 1}{4d_H + 3H}
\]

### Cache Analysis

We introduce an additional cache alongside the traditional KV cache, which stores the dimension-reduced keys from \( M \). By incorporating a model that applies GQA (Ainslie et al., 2023), a widely used technique for reducing the KV cache, we analyze the impact of our approach on memory usage. Assuming the number of heads is denoted as \( H_G \) in GQA, the total dimensions of the kvs are given by \( d_G = H_G \times d_C \). Given that \( l_M \gg l_I, l_L, l_C \) for long sequences, we focus our analysis on the memory usage related to \( M \). The cache size increased by the dimension-reduced keys is \( \frac{2d_f}{d_c} \) of the traditional KV cache.

## 4 Experiments

### 4.1 Experimental Setup

#### Baselines and Benchmarks

ESA can be applied to all decoder-only LLMs that utilize RoPE as their position encoding. We evaluate ESA using Mistral-7B-Instruct-v0.2 and Llama-3-8B-Instruct, with context lengths of 32k and 8k, respectively. We select the following three selective attention methods as our baselines: (a) InFLLM, (b) LM-Infinite (Infinite), (c) StreamingLLM (Stream). Additionally, we also choose two methods with position extrapolation: (a) NTK-aware (NTK), (b) YaRN. We conduct extensive evaluations on LongBench, ∞BENCH, NeedleBench, and Counting-Stars.

#### Calibration Dataset

We employ a small subset of Books3 data from Pile (Gao et al., 2020) as our calibration dataset to train the networks \( f_\theta \) and \( f_\phi \). There are 50k tokens in total and therefore 50k concatenated query-key pairs for training the networks in each layer. The learning rate and global batch size is 0.0005 and 128, respectively. We trained the dimensionality reduction networks for 10 epochs.

#### Parameters

The number of attention heads \( (H) \) is 32. We compress the original size of query and key from \( d_H = 4096 \) to \( d' = 128 \). Since the number of GQA heads is 8, the additional size required for the reduced-dimensionality keys is 6.25% of the original KV cache. Compared to computing full attention, the computational complexity is reduced to up to 1.56% in each step according to Equation 9. ESA and three other baselines with selective attention select the same number of tokens. The length of initial tokens \( (l_I) \) is 128. InFLLM and ESA both choose the lengths of middle tokens and local tokens to be 2048 and 4096, respectively.
## 4.2 Results on LongBench

LongBench includes six different categories of tasks, with an average context length range from less than 10k to around 32k. We adjust the scaling factors of NTK and YaRN to accommodate the benchmark. The context length for Mistral is 32k, which does not necessitate any modification to the model’s scaling factor. Consequently, we omit the NTK and YaRN for Mistral in this section. The results of the 16 English tasks are presented in Table 1. We draw the following conclusions: (a) Our method achieves improvement over the best baselines of selective attention (including Infinite, Stream, InfLLM) for both Llama and Mistral across a variety of tasks. Particularly, our method outperforms other methods of selective attention on the PassageRetrieval-en significantly, demonstrating the benefit of token-level selection. (b) Our method is comparable to the baselines that compute full attention (including Origin, NTK, YaRN). The gap between our method and the best among these approaches is within 1 percentage point.

## 4.3 Results on ∞BENCH

We select 6 tasks from ∞BENCH with an average length up to around 200k, encompassing open-form question answering (QA), code retrieval tasks, and other domains. We set the scaling factor for NTK and YaRN to accommodate contexts of up to 200k. The results of the tasks are presented in Table 2. Firstly, our model slightly outperforms the scores of InfLLM. The performance of our method exhibits minimal differences in retrieval tasks compared to InfLLM. This may be due to the fact that retrieval tasks only require the model to focus on a single relevant piece of information. InfLLM retrieves the most relevant chunks from the context, which is sufficient for solving long-text tasks that require attention to only a small amount of local information. Our method, on the other hand, opts for retrieval at a finer granularity, resulting in performance that is close to that of InfLLM on such tasks. In other tasks, our method outperforms other selective attention methods. Secondly, our method outperforms NTK and YaRN, especially on Llama. This superiority may arise from the fact that methods with position embedding tend to suffer a significant decline when extrapolated to excessively long contexts, such as an 8-fold extension. It demonstrates that our approach can extrapolate to significantly longer contexts, even up to a ×25 extension for Llama.

## 4.4 Results on NeedleBench and Counting-Stars

NeedleBench and Counting-Stars evaluate the ability of LLMs to retrieve multiple pieces of related information. The two benchmarks place higher demands on the model’s capacity to handle long context. Sample examples from the benchmarks are provided in Appendix C. The context length for these two benchmarks ranges from 4k to 256k, assessing the model’s capability to retrieve multiple pieces of information across varying lengths. We uniformly set the scaling factor for NTK and YaRN to accommodate contexts of up to 200k tokens. We follow (Li et al., 2022a; Song et al., 2024) to use the recall accuracy as a metric to evaluate the performance of the models.

Our method exhibits great strength in extracting critical information distributed across the context. The experimental results on Counting-Stars and NeedleBench are shown in Table 3 and 4, respectively. Details of the Counting-Stars are provided in Appendix ??. Firstly, when multiple pieces of relevant information need to be retrieved, our method significantly outperforms Infinite, Stream, and InfLLM. This is attributed to our method’s flexibility in selecting middle tokens at the token level. Secondly, the performance of ESA is comparable to that of NTK and YaRN. NTK and YaRN achieve promising performance by computing full attention when extrapolation is limited. When extrapolated to tasks involving longer sequences, NTK and YaRN may experience performance degradation. Lastly, within the original training lengths, ESA does not exhibit significant performance degradation, whereas the NTK and YaRN show a noticeable decline.

## 4.5 Ablation Study

### Effectiveness of the proximity influence distance ε.

The parameter ε in Equation 8 controls the continuity of the selected tokens. As demonstrated in Table 5, we find this parameter to be crucial for the model, especially with regard to its retrieval capabilities. Furthermore, we observe that in Retrieve.KV, when ε = 0.1, even when the model’s predictions are incorrect, it is still able to retrieve parts of the correct values. For instance, the answer is "49c65968-6319-44c-1268-feb249694b07", while the model’s prediction is "49c65968-6319-44f-9021-cfa198896071".
# Table 1: Results (%) on 16 English tasks of LongBench. The term “Origin” indicates the original model baselines without any extrapolation methods. We underline the best score of all methods for a model on a particular task and bold the best score of the selective attention methods, and this holds for the tables below.

| Task | Origin | NTK(32k) | YaRN(32k) | Infinite Stream | InLML | Ours | Origin | Infinite Stream | InLML | Ours |
|------------------------|--------|----------|-----------|-----------------|-------|------|--------|-----------------|-------|------|
| NarrativeQA | 2.91 | 10.16 | 13.14 | 16.12 | 19.77 | 24.89| 19.18 | 18.21 | 21.92 | 22.72|
| Qasper | 41.44 | 44.87 | 41.5 | 42.35 | 42.47 | 43.59| 29.77 | 29.74 | 28.75 | 28.65|
| MultiFiedQA | 37.18 | 52.3 | 31.35 | 38.87 | 42.83 | 47.49| 39.89 | 39.49 | 37.62 | 40.06|
| MuSiQue | 0.91 | 27.03 | 28.86 | 19.74 | 19.89 | 22.58| 24.23 | 16.82 | 16.62 | 19.82|
| HotpotQA | 8.24 | 53.05 | 51.84 | 45.41 | 46.96 | 49.39| 31.94 | 31.37 | 39.67 | 40.06|
| 2WikMultihop | 30.96 | 37.15 | 35.85 | 39.17 | 36.32 | 37.61| 22.62 | 21.21 | 21.92 | 23.15|
| GovReport | 18.83 | 32.34 | 34.22 | 29.77 | 29.82 | 31.01| 29.52 | 29.46 | 30.97 | 31.31|
| QMSum | 9.19 | 22.53 | 23.41 | 20.82 | 21.37 | 22.49| 21.67 | 21.77 | 23.52 | 23.79|
| MultiNLI | 26.96 | 27.67 | 27.07 | 27.48 | 27.43 | 27.63| 26.32 | 26.63 | 26.63 | 26.57|
| TREC | 52 | 75.74 | 73 | 73 | 73 | 74 | 70.5 | 70.5 | 70.5 | 70.5 |
| TriviaQA | 30.3 | 79.39 | 90.54 | 90.10 | 85.01 | 85.51| 85.68 | 85.68 | 87.62 | 87.62|
| SAMSum | 20.55 | 42.36 | 43.44 | 42.12 | 42.3 | 41.99| 42.65 | 41.49 | 41.69 | 42.3 |
| PassageRetrieval-en | 2.08 | 10.67 | 9.5 | 7.40 | 8.56 | 8.76 | 4.33 | 4.23 | 6.23 | 5.34 |
| PassageCount | 2.86 | 5.22 | 8 | 7.67 | 7.67 | 7.67 | 1.93 | 0.71 | 2.84 | 3.03 |
| LCC | 59.37 | 35.43 | 53.79 | 58.63 | 58.94 | 59.34| 55.05 | 54.31 | 54.14 | 55.04|
| RepoBench-P | 33.92 | 33.74 | 35.48 | 40.61 | 43.42 | 44.5 | 51.73 | 51.14 | 51.52 | 52.56|
| Average | 23.27 | 42.46 | 45.39 | 56.21 | 29.68 | 42.49| 41.22 | 36.96 | 39.32 | 39.43|

# Table 2: Performance evaluation (%) on 6 English tasks of coBENCH.

| Task | Origin | NTK(200k) | YaRN(200k) | Infinite Stream | InLML | Ours |
|------------------------|--------|-----------|------------|-----------------|-------|------|
| Retrieve-KV | 0.0 | 0.86 | 3.85 | 1.8 | 1.58 | 3.4 |
| Main-Find | 0 | 4.56 | 3.46 | 14 | 13.66 | 16.57 |
| Retrieve-Number | 0 | 2.71 | 1.54 | 6.24 | 6.71 | 10.67 |
| Re-MC | 0.91 | 10.88 | 11.47 | 3.74 | 3.95 | 4.12 |
| Code-DB | 2.29 | 9.35 | 8.99 | 9.63 | 10.9 | 10.90 |
| Retrieve-Pick | 3.67 | 11.56 | 17.68 | 18.61 | 20.83 | 22.67 |
| Average | 3.77 | 4.58 | 22.66 | 16.93 | 17.62 | 18.47 |

# Table 3: Recall accuracy (%) evaluation on the Counting-Stars benchmark. We employ the notation (256k, 32) to represent the following benchmark setup: the context length is 256k, and within this length, we generate 32 samples at equal intervals (e.g., the samples being 8k, 16k, 32k, ..., up to 256k), with each sample containing 8 pieces of relevant information. Accuracy is defined as the average score across the 32 samples within each task.

| Task | Origin | NTK(200k) | YaRN(200k) | Infinite Stream | InLML | Ours |
|--------------------------|--------|-----------|------------|-----------------|-------|------|
| (128k, 32, 32) | 15.91 | 15.70 | 52.6 | 10.7 | 10.21 | 26.8 |
| (256k, 32, 32) | 23.63 | 24.34 | 54.9 | 8.17 | 9.67 | 27.6 |
| (256k, 32, 16) | 23.62 | 124.16 | 47.3 | 15.82 | 16.67 | 18.2 |
| (256k, 32, 12) | 27.67 | 17.57 | 7.7 | 5.1 | 6.44 | 16.3 |
| Average | 12.59 | 19.31 | 7.6 | 10.4 | 16.4 | 15.9 |
| Context Length | Origin | NTK(200k) | YARN(200k) | Infinite | Stream | InfLLM | Ours | Origin | NTK(200k) | YARN(200k) | Infinite | Stream | InfLLM | Ours |
|----------------|--------|-----------|------------|----------|--------|--------|------|--------|-----------|------------|----------|--------|--------|------|
| 4k | 96.67 | 90 | 96.67 | 96.67 | 96.67 | 76.67 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| 8k | 100 | 100 | 76.67 | 73.33 | 86.67 | 100 | 73.33 | 43.33 | 63.33 | 83.33 | 83.33 | 67.67 | 66.67 | 73.33|
| 16k | 96.67 | 96.67 | 20.00 | 23.33 | 36.67 | 73.33 | 133.33| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 48k | 0 | 0 | 3.33 | 3.33 | 33.33 | 83.33 | 50 | 53.33 | 0 | 0 | 0 | 0 | 0 | 0 |
| 80k | 0 | 0 | 0 | 0 | 36.67 | 66.67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 112k | 0 | 0 | 0 | 0 | 43.33 | 0 | 76.67| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 128k | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 46.67 | 73.33 | 63.33 | 0 | 0 | 0 | 0 |
| 144k | 0 | 0 | 0 | 20.00 | 73.33 | 66.67 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 176k | 0 | 0 | 0 | 0 | 0 | 70.67 | 76.67| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 200k | 0 | 0 | 0 | 0 | 0 | 23.33 | 0 | 66.67 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Average** | **19.67** | **32.33** | **19.33** | **21.33**| **5.3**| **54** | **53**| **62.67** | **16.33** | **16** | **17.33**| **0** | **0** | **68**|

# Table 4: Recall accuracy (%) evaluation on NeedleBench.

| Task | € | 0 | 1 | 3 | 5 |
|-----------------|---|-----|-----|-----|-----|
| Retrieve.KV | | 66.6| 82 | 91.6| 95.6|
| NeedleBench | | 69.67| 71.33| 68 | 58.67|

# Table 5: The ablation study results of ε on InfiniteBench’s Retrieve.KV and NeedleBench with Mistral. NeedleBench in the table represents the average scores across lengths ranging from 4k to 200k, with the specific scores detailed in Appendix D.

| Task | individual | uniform |
|-----------------|------------|---------|
| LongBench scores | 44.7 | 44.8 |

# Table 6: We employ Llama to validate different token selection strategies for heads. The LongBench scores represent the average scores across 16 subtasks in LongBench.

## Dimension Reduction of Queries and Keys.
We calculate the importance scores of tokens using the reduced-dimensionality queries and keys. To evaluate the impact of dimensionality reduction, we analyse experiments on LongBench with Llama using the full-dimensional query and key, as well as their reduced-dimensionality counterparts. As demonstrated in Table 6 and Table 1, their respective scores are 44.8 and 44.1. The difference between the two scores is only 0.39. Furthermore, we select samples from Books3 in Pile and employ Mistral to validate the recall rate of the top-k retrieval subsequent to dimensionality reduction. The ground truth is determined using the top-k tokens derived from the full-dimensional query and key. A total of 2,000 tokens are selected for this analysis, spanning positions from 23,000 to 25,000. In parallel, we execute comparative experiments utilizing principal component analysis (PCA) for dimensionality reduction inspired by (Singhania et al., 2024).

The experimental results are depicted in Figure 2. It can be observed that our dimensionality reduction method achieves a recall rate of over 90% for the majority of the model’s layers, whereas PCA exhibits a recall rate below 90% in approximately half of the layers.

## 5 Conclusions and Future work
In this paper, we propose an efficient token-level selective attention method for extending the context length of LLMs without incremental training of LLMs parameters. The insight is to select a constant number of the most important tokens at each step for computing attention at the token level, leveraging the sparsity of the attention matrix. When the input sequence is sufficiently long, we are able to reduce the computational complexity up to nearly 1.56% of the original by compressing the queries and keys into low-dimensional representations. Our empirical evaluations demonstrate that ESA can effectively handle sequences lengths up to 4× and even 25× the training length for various types of long sequence tasks. Future work could explore more accurate and efficient methods for selecting important tokens in extrapolation tasks.
6 Limitations
Our method has the following limitations:
1. We apply the same compression ratio to the queries and keys across different layers; employing varying compression ratios for different layers may yield better performance.
2. There may exist more effective token selection methods and compression techniques that warrant further exploration.
3. A more suitable positional encoding for our approach may exist and requires further investigation.

References
Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kawatra, Bhargav S. Gulavani, and Ramchandran Ramjee. 2023. Sarathi: Efficient ilm inference by piggybacking decodes with chunked prefills. ArXiv, abs/2308.16369.

Kian Abhari, Alon Benhaim, Barun Patra, Jay Pujara, Saksham Singhal, and Xia Song. 2024. On the adaptation of uniformformer for decoder-only transformers. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 12395–12400.

Al@Meta. 2024. Llama 3 model card.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lefbvre, and Sumit Sanghvi. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245.

Chen An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. 2024. Training-free long-context scaling of large language models. ArXiv, abs/2402.17463.

Yihsiu Bai, Xin Lv, Jiajie Zhang, Hongchao Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohon Yang, Lei Hou, et al. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508.

Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. ArXiv, abs/2004.05150.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. 2024. Unlimiformer: Long-range transformers with unlimited length input. Advances in Neural Information Processing Systems, 36.

block97. 2023. Ntk-aware scaled repo allows llama model to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and Wen Xiao. 2024. Pyramidkv:

Dynamic kv cache compression based on pyramidal information funneling. ArXiv, abs/2406.02069.

Guanzheng Chen, Xin Li, Zaiqiao Meng, Shangsong Liang, and Lidong Bing. 2023a. Clex: Continuous length extrapolation for large language models. arXiv preprint arXiv:2310.16450.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023b. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595.

T Dao, DY Fu, S Ermon, A Rudra, and C FlashAttention Ré́. fast and memory-efficient exact attention with two-awareness. arxiv; 2022. arXiv preprint arXiv:2205.14135.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahung Xu, Fan Yang, and Mao Yang. 2024. Longrope: Extending ilm context window beyond 2 million tokens. ArXiv, abs/2402.13753.

Qingqiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhi Fang Su. 2022. A survey on in-context learning. arXiv preprint arXiv:2310.00234.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace Rie, Anish Thite, Noah Nabeeshima, et al. 2020. The pile: An 800GB dataset of diverse text for language modeling. arXiv preprint arXiv:2011.00027.

Yunfan Gao, Yin Xiong, Xinyu Gao, Kanyings Jia, Jinlu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2023. Model tells you what to discard: Adaptive kv cache compression for lms. arXiv preprint arXiv:2310.01801.

Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangruklar, Marc Sun, and Benjamin Bossan. 2022. Accelerate: Training and inference at scale made simple, efficient and adaptable. https://github.com/huggingface/ accelerate.

Chi Huan, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. 2023. Lm-infinite: Simple on-the-fly length generalization for large language models. arXiv preprint arXiv:2308.16137.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lamplle, Lucie Saunier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.
# References

- Huqiang Jiang, Yucheng Li, Chengruiding Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdol, Dongsheng Li, Ching-Yew Lin, et al. 2024. Minference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. arXiv preprint arXiv:2407.02490.

- Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-yuan Chang, Huiyuan Chen, and Xia Hu. 2024. Llm maybe longlm: Self-extend llm context window without tuning. ArXiv, abs/2401.01325.

- kaiokendev. 2023. Things i’m learning while training superhot.

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Liamnin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626.

- Mo Li, Songyang Zhang, Yunxin Liu, and Kai Chen. 202A. Needlebench: Can lms do retrieval and reasoning in 1 million context window? arXiv preprint arXiv:2407.11963.

- Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Gary Locatelli, Hanchen Ye, Tianlei Cai, Patrick Lewis, and Deming Chen. 202B. Snapkvy: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469.

- Xiaoran Liu, Hang Yan, Shuo Zhang, Chen An, Xipeng Qiu, and Dahua Lin. 2023. Scaling laws of rope-based extrapolation. ArXiv, abs/2310.05209.

- Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrilidis, and Anshulmi Shrivastava. 2024. Scisorrhands: Exploiting the persistence of importance hypothesis for llm kw cache compression at test time. Advances in Neural Information Processing Systems, 36.

- Yi Lu, Xin Zhou, Wei He, Jun Zhao, Tao Gui, Qi Zhang, and Xuanjing Huang. 2024. Longheads: Multi-head attention is secretly a long context processor. arXiv preprint arXiv:2402.10685.

- Amirkeivan Mohtashami and Martin Jaggi. 2023. Landmark attention: Random-access infinite context length for transformers. arXiv preprint arXiv:2305.16300.

- Bowen Peng, Jeffrey Quenselle, Honglu Fan, and Enrico Shippole. 2023. Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071.

- Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gert, Xiaoqing Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jeremy Rapin, Artyom Kozhevnikov, I. Evtimov, Joanna Bitton, Manish P.

- Bhatt, Cristian Cantón Ferrer, Aaron Grattatori, Wenhan Xiong, Alexandre D’eosser, Jade Copet, Faisal Azhar, Hugo Touron, Louis Martel, Nicolas Viseur, Thomas Scialom, and Gabriel Synnaeve. 2023. Code llama: Open foundation models for code. ArXiv, abs/2308.12950.

- Prajwal Singhania, Siddharth Singh, Shival He, Soheil Feizi, and Abhinav Bhattele. 2024. Loki: Van-ram keys for efficient sparse attention. arXiv preprint arXiv:2406.05242.

- Mingyang Song, Mao Zheng, and Xuan Luo. 2024. Counting-stars: A simple, efficient, and reasonable strategy for evaluating long-context large language models. arXiv preprint arXiv:2403.11802.

- Jianlin Su. 2023. Rectified rotary position embeddings. https://github.com/bojenorepe.

- Jianlin Su, Murtadh Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

- Su, Jianlin. 2023. ReRoPE. https://kevu.fm/archives/9708.

- Chaojun Xiao, Pengzle Zhang, Xu Han, Guanguan Xiao, YanKai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, and Maosong Sun. 2024. Imllm: Unveiling the intrinsic capacity of llms for understanding extremely long sequences with training-free memory. arXiv preprint arXiv:2402.04617.

- Guanguan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.

- Manzil Zaheer, Guru Gururangan, Kumar Anirava Dubey, Joshua Ainslie, Chris Alberti, Santiago Onatón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amir Ahmed. 2020. Big bird: Transformers for longer sequences. ArXiv, abs/2007.14062.

- Xinrong Zhang, Yingfa Chen, Shending Hu, Zihang Xu, Junhao Chen, Moo Hoo, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, et al. 2024. Ocbenck: Extending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15262–15277.

- Zhenyu Zhang, Ying Zheng, Tianyi Zhou, Tianlong Chen, Liamnin Zheng, Ruisi Cai, Zhao Song, Yuan-dong Tian, Christopher Ré, Clark Barrett, et al. 202B. H2o: Heavy-hitter oracle for efficient generalization in reference of large language models. Advances in Neural Information Processing Systems, 36.
## A Complexity Analysis Derivation

Denote \( I_{L,M,L,C} = I_L + I_M + I_L + I_C \), the full attention in each step is computed as

\[
O = \text{Attn}(Q_C, K_{I_{L,M,L,C}}, V_{I_{L,M,L,C}})
\]

The complexity of computing full attention consists of the following parts:

1. The dot product of queries and keys:
\[
2 \cdot d_H \cdot l_C \cdot I_{L,M,L,C}
\]

2. The softmax operation (including exponential calculation, summation, and normalization):
\[
3 \cdot H \cdot l_C \cdot I_{L,M,L,C}
\]

3. The dot product of attention weights and values:
\[
2 \cdot d_H \cdot l_C \cdot I_{L,M,L,C}
\]

The overall complexity is
\[
(4 \cdot d_H + 3 \cdot H) \cdot l_C \cdot I_{L,M,L,C}
\]

The complexity of our method comprises the following components:

1. Reduction of the dimensions of the query and key in Equation (5):
\[
2 \cdot 2 \cdot l_C \cdot d_H \cdot d' + 2 \cdot l_C \cdot d'
\]

2. The complexity of token selection in Equation (4) and (8) is \( 2 \cdot l_M \cdot l_C \cdot d' + l_M \cdot l_C \cdot h_{max} \cdot l_C \). The complexity of taking the maximum operation on multi-dimensional vectors is denoted as \( h_{max} \). Since (a) the complexity is lower than that of an equivalent number of Floating Point Operations (FLOPS) for the same scale; and (b) \( 2 \cdot l_M \cdot l_A + (1 + 2e) \cdot l_M < 2 \cdot l_M \cdot l_C \cdot d' \), we neglect the impact of the max operation. Therefore, the complexity of token selection is:
\[
2 \cdot l_M \cdot l_C \cdot d' + l_M \cdot l_C
\]

3. Following the analogy of Equation (11), (12) and (13), after selecting \( k \) tokens from \( M \), the complexity of computing sparse attention is as follows:
\[
(4 \cdot d_H + 3 \cdot H) \cdot l_C \cdot (I_{L,C} + k)
\]

where, \( I_{L,L,C} \cong I_L + I_L + I_C \).

Given that \( I_M >> I_L, I_L, l_C, k, d_H, d' \), the overall reduction ratio of complexity is:
\[
r = \frac{I_L + I_L + I_L + I_C}{I_L + I_M + I_L + I_C} \cdot \frac{4d_Hd' + 2d' l_M + I_M}{(4d_H + 3H)(I_L + I_M + I_L + I_C)} \approx \frac{2d' + 1}{4d_H + 3H}
\]

## B Model Loading

All of our experiments were conducted on a device equipped with \( 4 \times A800 \) GPUs. The evaluated models were partitioned across the \( 4 \) GPUs by layer using the accelerate (Gugger et al., 2022). Our model loading approach is capable of supporting the execution of all the aforementioned experiments. Additionally, we support loading the model onto a single \( A800 \) GPU by offloading the original KV cache to the CPU, while the dimension-reduced keys are always kept on the GPU. During each attention computation, a small number of keys and values in KV cache are loaded onto the GPU. By employing this cache management approach, we are able to perform inference on long sequence tasks with lengths of 7000+ on a single \( A800 \) GPU. For NTK and YaRN, we employ vLLM (Kwon et al., 2023) for inference on a device equipped with \( 4 \times A800 \) GPUs.
| ε | Context Length | Average |
|----|-----------------|--------|
| | 4k | 8k | 16k | 48k | 80k | 112k | 128k | 144k | 176k | 200k |
|----|----|----|-----|-----|-----|------|------|------|------|------|
| 0 | 100| 100| 73.33| 80 | 70 | 66.67| 70 | 46.67| 46.67| 69.67|
| 1 | 100| 83.33| 73.33| 86.67| 63.33| 70 | 73.33| 56.67| 60 | 71.33|
| 2 | 100| 83.33| 73.33| 70 | 66.67| 63.33| 60 | 63.33| 53.33| 46.67| 68|
| 3 | 100| 83.33| 70 | 70 | 40 | 56.67| 36.67| 50 | 40 | 58.67|
| 4 | | | | | | | | | | |
| 5 | | | | | | | | | | |

Table 7: We validate the performance across all subtasks of NeedleBench with varying ε using Mistral.

## C Samples of NeedleBench and Counting-Stars

### NeedleBench
A sample from NeedleBench is shown below, where xxxxxxxx represents noise text:

You are an intelligent AI assistant skilled in answering user questions. Please keep your answers concise and clear. Do not talk about irrelevant topics or repeat your answers. The document given to you by the user is May 2001 xxxxxxxx Hidden on Hell Island is the legendary Dream Bubble. xxxxxxxx Hidden on Emerald Island is the legendary Ghost Pearl. xxxxxxxx Hidden on Sand Island is the legendary Stardust Shard. xxxxxxxx
Now, the questions are: What legendary item is hidden on Hell Island? What legendary item is hidden on Emerald Island? What legendary item is hidden on Sand Island? Before answering, please consider what in the document is most relevant to this question. Please answer in the format of 'The legendary item hidden on the Hell Island is ____. The legendary item hidden on the Emerald Island is ____. The legendary item hidden on the Sand Island is ____.'

### Counting-Stars
A sample from Counting-Stars is shown below, where xxxxxxxx represents noise text:

xxxxxxxx The little penguin counted 15 * xxxxxxxx The little penguin counted 117 *
xxxxxxxx The little penguin counted 42 * xxxxxxxx The little penguin counted 29 *
On this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting *. Please help the little penguin collect the number of *, for example: {"little_penguin": [x, x, x,...]}. The summation is not required, and the numbers in [x, x, x,...] represent the counted number of * by the little penguin. Only output the results in JSON format without any explanation.

```json
{"little_penguin": [

## D Proximity Influence Distance: Specific Experiments

We validate the ablation study results of the proximity influence distance, with the detailed findings presented in Table 7. When ε is set to 0 and 1, Mistral demonstrates superior performance across all subtasks ranging from 4k to 200k. As ε is further increased to 3 and 5, the scores on the short-sequence subtasks (4k-16k) remain comparable to the previous results. However, the model’s performance exhibits a significant decline on the long-sequence subtasks (48k-200k).

## E Token Selection for Heads: Specific Experiments

We employ Llama to investigate the impact of different token selection methods for heads. We extrapolate Llama's original context length of 8k to 32k and conduct experiments using LongBench, which has an average length of 32k. The scores for each subtask are presented in Table 8.

## F LM-Infinite with additional top-k middle tokens

LM-Infinite optionally select top-k middle tokens for some higher layers for each head. Following the settings in their paper, we evaluate the effectiveness of this method and the results are demonstrated in Table 9. The performance is improved compared to the original LM-Infinite setting. Nevertheless, it is not as competitive as our ESA method.
```
| Task | individual | uniform |
|----------------------|------------|---------|
| NarrativaQA | 24.52 | 25.1 |
| Qasper | 44.69 | 44.51 |
| MultiFieldQA-en | 49.18 | 49.18 |
| MuSiQue | 25.55 | 27.58 |
| HotpotQA | 49.57 | 49.26 |
| 2WikiMultihopQA | 38.1 | 37.44 |
| GovReport | 31.06 | 30.99 |
| QMSum | 22.91 | 22.75 |
| MultiNews | 27.41 | 27.45 |
| TREC | 73.5 | 73.5 |
| TriviaQA | 91.19 | 91.19 |
| SAMSum | 42.87 | 42.7 |
| PassageRetrieval-en | 86.5 | 87.5 |
| PassageCount | 8.17 | 7.17 |
| LCC | 58.32 | 58.32 |
| RepoBench-P | 41.7 | 42.6 |
| Average | 44.7 | 44.8 |

| Task | Llama | Mistral |
|----------------------|------------|----------|
| NarrativaQA | 20.6 | 22.02 |
| Qasper | 21.73 | 30.36 |
| MultiFieldQA-en | 40.2 | 44.52 |
| MuSiQue | 20.19 | 16.36 |
| HotpotQA | 44.66 | 32.63 |
| 2WikiMultihopQA | 38.09 | 22.64 |
| GovReport | 31.28 | 31.61 |
| QMSum | 21.73 | 22.5 |
| MultiNews | 27.54 | 26.7 |
| TREC | 73.5 | 70.5 |
| TriviaQA | 90.91 | 86.59 |
| SAMSum | 42.57 | 42.26 |
| PassageRetrieval-en | 38.5 | 49.42 |
| PassageCount | 8.5 | 2.37 |
| LCC | 60.75 | 57.4 |
| RepoBench-P | 43.83 | 53.51 |
| Average | 40.37 | 38.212 |

Table 8: Llama's specific experiments on LongBench using different token selection methods for heads. "Individual" refers to the importance score of each token being the maximum value among the scores of all heads, meaning that each head votes for the scores. This approach ensures that the selection process takes into account all heads. "Uniform" in the table denotes our method of selecting tokens without dimensionality reduction.

Table 9: Results on LongBench with Infinite-LM attending to top 5 tokens in the middle.
## G Pseudocode for Computing Attention

We support the computation of local and global attention using either Flash Attention (Dao et al.) or PyTorch operators. The pseudocode for computing a step of attention with Flash Attention is shown in Algorithm 1. It is worth noting that we omit the exp-normalize trick in Step 12 of Algorithm 1 to avoid numerical overflow. We employ the function `flash_attn_func` provided by Flash Attention, which returns the logarithm of the softmax normalization factor as its second result. In environments where Flash Attention is not supported, we can replace the function `flash_attn_func` with PyTorch operators.

### Algorithm 1 Pseudocode for Attention Computation with Flash Attention

1. Input:
2. `l_q`: Queries from C with position encoding (`l_L`, `l_L` + 1, `l_L` + 2, ..., `l_L` + `l_C` - 1)
3. `g_q`: Queries from C with position encoding (`w`, `w`, ..., `w`)
4. `l_k`: Concatenated keys from L and C with position encoding (0, 1, 2, ..., `l_L` + `l_C` - 1)
5. `g_k`: Selected keys from M and I with position encoding (0, 0, 0, 0, ...)
6. `l_v`: Concatenated values from L and C
7. `g_v`: Selected values from M and I
8.

9. Procedure:
10. `(l_attn, l_se) ← flash_attn_func(l_q, l_k, l_v, causal = True)`
11. `(g_attn, g_se) ← flash_attn_func(g_q, g_k, g_v, causal = False)`
12. `se ← exp([l_se, g_se])`
13. `fac ← se / ∑ se`
14. `attn ← [l_attn, g_attn] · fac`

15. Output:
16. `attn`