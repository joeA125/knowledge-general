A catalog of all wiki pages, organised by type.

**Scope: machine learning, LLMs, and mathematical foundations.** Established 2026-08 by migration from a sports-analytics vault; football and sports material remains there.

## Navigation

- [[overview]] — narrative map of what this vault holds and where it is thin
- Dashboards: [[health|Health]] · [[reinforcement|Reinforcement]] · [[sources|Source Tracking]]

## Concepts

### Transformer and Attention
- [[transformer]] — the architecture the LLM material rests on; **the vault's largest hub**
- [[attention-mechanism]] · [[scaled-dot-product-attention]] · [[multi-head-attention]] · [[additive-attention]] · [[positional-encoding]]
- [[encoder-decoder]] · [[encoder-decoder-bottleneck]] · [[feed-forward-network]] · [[layer-normalization]] · [[residual-connections]]

### Recurrent Architectures
- [[recurrence]] · [[lstm]] · [[gated-recurrent-unit]] · [[bidirectional-rnn]]
- [[dropout-for-rnns]] · [[recurrent-dropout]] *(archived)*

### Convolutional and Vision
- [[convolution]] · [[dilated-convolution]] · [[fully-convolutional-network]] · [[feature-pyramid-network]] · [[semantic-segmentation]]
- [[pre-activation-resnet]] · [[batch-normalization]] · [[siamese-network]] · [[conditional-gan]]
- [[camera-calibration]] · [[homography]] — geometric vision

### Training and Regularisation
- [[regularization]] · [[dropout]] · [[label-smoothing]] · [[adam-optimizer]] · [[teacher-forcing]]
- [[model-selection]] — the asserted-parameter problem, and the four kinds of parameter

### Representation
- [[representation-learning]] — three routes, and when handcrafting still wins
- [[feature-engineering]] · [[theory-based-modelling]] · [[tokenization]] · [[transfer-learning]] · [[pre-train-then-fine-tune]]
- [[graph-neural-network]] · [[message-passing]] — the same pattern computing inference and representation
- [[eigenvector]] · [[non-negative-matrix-factorization]] — spectral and additive decomposition

### Generative Models
- [[generative-model]] — the families, and why likelihood is a poor guide when the model is a means
- [[autoregressive-model]] · [[variational-autoencoder]] · [[variational-lossy-autoencoder]] · [[conditional-gan]]
- [[counterfactual-simulation]] · [[counterfactual-baseline]] — intervention on a fitted model, and what it assumes

### Language Models and Agents
- [[gpt]] · [[bert]] · [[masked-language-model]] · [[scaling-laws]] · [[zero-shot-learning]]
- [[chain-of-thought]] · [[react]] · [[retrieval-augmented-generation]] · [[constrained-decoding]]
- [[ai-agent]] · [[tool-use]] · [[agent-memory]] · [[rlhf]]

### Reinforcement Learning
- [[reinforcement-learning]] — the core objects and three solution families ⚠️ *thinly sourced*
- [[markov-game]] · [[value-iteration]] · [[temporal-difference-learning]] · [[deep-q-network]] · [[proximal-policy-optimization]]
- [[policy-modelling]] · [[imitation-learning]] · [[multi-agent-reinforcement-learning]] · [[game-theory]]
- [[agent-based-simulation]] · [[domain-adaptation]] — simulating what cannot be experimented on, and transferring across the gap

### Sequence and Event Modelling
- [[point-process]] · [[neural-temporal-point-process]] · [[event-prediction]]
- [[pointer-network]] · [[read-process-write]] · [[combinatorial-optimisation]] · [[neural-turing-machine]] · [[trajectory-prediction]]

### Probabilistic Modelling and Inference
- [[bayesian-inference]] · [[bayes-theorem]] · [[gaussian-process]] · [[expectation-propagation]]
- [[factor-graph]] · [[approximate-message-passing]] · [[gaussian-density-filtering]] · [[kl-divergence]]
- [[identifiability]] — when more data does not help

### Rating and Comparison
- [[bradley-terry-model]] — the base model the rest extend
- [[elo-rating-system]] · [[glicko-rating-system]] · [[trueskill]]

### Evaluation and Validity
- [[predictive-validity]] · [[split-half-reliability]] · [[capability-profiling]] · [[interpretability]]
- [[probabilistic-classification]] · [[probability-calibration]] · [[class-imbalance-evaluation]] · [[uncertainty-quantification]]
- [[selection-bias]] · [[rare-event-proxy-targets]] · [[structured-model-decomposition]] · [[gradient-boosting]]

## Entities

### Transformer authors
[[ashish-vaswani]] · [[noam-shazeer]] · [[niki-parmar]] · [[jakob-uszkoreit]] · [[llion-jones]] · [[aidan-gomez]] · [[lukasz-kaiser]] · [[illia-polosukhin]]

### Sequence models and attention
[[dzmitry-bahdanau]] · [[kyunghyun-cho]] · [[yoshua-bengio]] — attention and the GRU
[[oriol-vinyals]] · [[meire-fortunato]] · [[navdeep-jaitly]] — pointer networks, set modelling
[[alex-graves]] · [[greg-wayne]] · [[ivo-danihelka]] — neural Turing machines
[[wojciech-zaremba]] · [[ilya-sutskever]] — RNN regularisation

### Language models and scaling
[[alec-radford]] — GPT · [[jacob-devlin]] — BERT
[[jared-kaplan]] · [[sam-mccandlish]] · [[dario-amodei]] — scaling laws
[[diederik-kingma]] — VAE

### Vision
[[kaiming-he]] · [[xiangyu-zhang]] · [[shaoqing-ren]] · [[jian-sun]] — ResNet
[[fisher-yu]] · [[vladlen-koltun]] — dilated convolutions

### Bayesian rating
[[ralf-herbrich]] · [[tom-minka]] · [[thore-graepel]] — TrueSkill

### Organisations
[[openai]] · [[google-brain]] · [[google-research]] · [[google-deepmind]] · [[microsoft-research]]
[[university-of-toronto]] · [[jacobs-university-bremen]] · [[universite-de-montreal]]

## Source Summaries

### Transformer and sequence models
- [[attention-is-all-you-need]] — Vaswani et al. (2017); the architecture
- [[neural-machine-translation]] — Bahdanau et al. (2015); attention and the GRU
- [[pointer-networks]] — Vinyals et al. (2015); attention as selection
- [[sequence-to-sequence-sets]] — Vinyals et al. (2016); ordering effects on unordered data
- [[neural-turing-machines]] — Graves et al. (2014); differentiable external memory
- [[rnn-regularisation]] — Zaremba et al. (2015); dropout placement in recurrent nets
- [[identity-mapping-residual-networks]] — He et al. (2016); residual connections

### Language models
- [[language-understanding-gpt]] — Radford et al. (2018); generative pre-training
- [[bert-bidirectional-transformers]] — Devlin et al. (2019); masked language modelling
- [[scaling-neural-language-models]] — Kaplan et al. (2020); power-law scaling
- [[training-lm-follow-instructions-with-human-feedback]] — Ouyang et al. (2022); RLHF

### Retrieval and reasoning
- [[rag-intense-nlp-tasks]] — Lewis et al. (2020)
- [[autogressive-language-model-retrieval]] · [[autogressive-language-model-retrieval-iterative]] — retrieval-augmented pre-training and ITER-RETGEN
- [[universal-prompt-retrieval-zero-shot-eval]] — UPRISE
- [[augmented-llms-parametric-guiding]] — PKG
- [[chain-of-thought-reasoning-llms]] — Wei et al. (2022)
- [[synergising-reasoning-acting-llms]] — ReAct

### Generative and vision
- [[variational-lossy-autoencoders]] — Chen et al. (2017); the information-preference problem
- [[context-aggregation-dilated-convolutions]] — Yu & Koltun (2016); resolution without downsampling

### Evaluation and inference
- [[agi-definition]] — Hendrycks et al. (2025); CHC-grounded capability profiling
- [[llm-factcheck-consistency-certainty]] — PCC; stated certainty against sampled consistency
- [[bayesian-true-skill-rating]] — Herbrich et al. (2006); TrueSkill

### Articles
- [[ai-agent-architecture-breakdown]] · [[eigenvectors-explained]]
