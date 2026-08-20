# Tag Taxonomy

Canonical tags for this knowledge base. All tags on wiki
pages must come from this list. To add a new tag, add it
here first with a brief description.

Page type is recorded in the `type:` frontmatter field, not
as a tag. There are deliberately no page-type tags here —
two namespaces recording one fact drift apart, and did.

## Domain Tags

### Foundations
- `statistics` — statistical methods and theory
- `bayesian` — Bayesian inference and updating
- `machine-learning` — ML models, training, evaluation
- `AI` — artificial intelligence
- `deep-learning` — neural networks, backpropagation, gradient-based optimisation
- `data-engineering` — pipelines, storage, ETL
- `linear-algebra` — vectors, matrices, eigendecomposition, SVD, and related theory
- `information-theory` — entropy, divergence, and coding-theoretic measures (KL, cross-entropy, mutual information)
- `dimensionality-reduction` — compressing data into low-dimensional representations (PCA, NMF, functional bases)

### Architectures
- `architecture` — neural network architecture design patterns
- `transformer` — Transformer architecture and variants
- `attention` — attention mechanisms in neural networks
- `rnn` — recurrent neural networks (vanilla, LSTM, GRU, BiRNN, VRNN)
- `lstm` — Long Short-Term Memory architecture
- `encoder-decoder-bottleneck` — fixed-length vector bottleneck in seq2seq models
- `residual-learning` — residual connections, skip connections, and ResNet architectures
- `convolution` — convolutional layers and their variants
- `dilated-convolution` — convolutions with dilation factor for expanded receptive fields without resolution loss
- `graph-neural-network` — neural networks operating on graph-structured data via message passing
- `pointer-mechanism` — using attention as a pointer to select input elements as output
- `external-memory` — neural architectures augmented with external readable/writable memory
- `neural-computation` — neural networks that learn algorithms and program-like behaviour

### Training and Regularisation
- `training-technique` — optimiser schedules, training tricks, and procedures
- `regularization` — techniques to prevent overfitting (dropout, label smoothing, etc.)
- `dropout` — dropout regularisation technique and its variants
- `normalization` — normalisation techniques (layer norm, batch norm, etc.)
- `batch-normalization` — batch normalisation technique for training deep networks
- `encoding` — representation and encoding schemes (e.g. positional encoding)
- `teacher-forcing` — training autoregressive models on ground-truth history, and the exposure bias it creates
- `multi-task-learning` — jointly optimising several objectives so they share representation
- `auxiliary-loss` — a secondary training objective added to shape or regularise the primary one
- `sample-weighting` — reweighting a loss function to correct uneven representation in training data
- `weak-supervision` — learning from incomplete, indirect, or single-point labels
- `scaling-laws` — power-law relationships between performance and scale
- `experience-replay` — storing and resampling past transitions to decorrelate updates in off-policy RL

### Representation
- `representation-learning` — learning useful data representations for downstream tasks
- `feature-engineering` — constructing, selecting, or learning input representations
- `tokenization` — converting raw data into discrete units for sequence models
- `entity-embedding` — learned dense representations of discrete entities
- `transfer-learning` — reusing representations learned on one task/dataset for another
- `domain-adaptation` — transferring across a shift between source and target environments
- `pre-training` — unsupervised or self-supervised training before task-specific fine-tuning

### Sequence and Generative Models
- `sequence-modelling` — modelling sequential data (text, time series, etc.)
- `sequence-alignment` — matching two sequences that may differ in timing or length (DTW, edit distance)
- `set-modelling` — handling unordered sets as inputs or outputs in neural models
- `ordering` — effects of data ordering on model training and performance
- `machine-translation` — translating text between languages with ML
- `alignment` — word/phrase alignment in MT; also AI alignment
- `speech-recognition` — mapping acoustic signals to words or phonemes
- `generative-model` — models that learn to generate data (VAEs, GANs, autoregressive, diffusion)
- `vae` — variational autoencoders and related latent variable models
- `gan` — generative adversarial networks and variants
- `autoregressive-model` — models that factorise distributions via chain rule
- `density-estimation` — estimating probability distributions from data
- `combinatorial-optimisation` — solving combinatorial problems with learned models

### Language Models and Agents
- `language-modelling` — predicting next tokens in sequences of text
- `masked-language-model` — predicting randomly masked tokens from bidirectional context
- `instruction-tuning` — fine-tuning LMs on instruction-formatted data
- `prompt-engineering` — designing or retrieving prompts to guide LLM behaviour
- `chain-of-thought` — prompting LLMs to produce intermediate reasoning steps
- `zero-shot-learning` — performing tasks without task-specific training examples
- `knowledge-intensive` — tasks requiring factual, domain, or world knowledge beyond the input
- `multi-hop-reasoning` — reasoning over multiple pieces of evidence or steps
- `reasoning` — logical, abstract, or multi-step reasoning capabilities
- `RAG` — retrieval augmented generation
- `ai-agent` — autonomous systems that plan, use tools, and act iteratively toward goals
- `tool-use` — LLM function calling and interaction with external tools/APIs
- `agent-memory` — persistence and recall mechanisms in agentic systems
- `MCP` — model context protocol
- `fact-checking` — verifying factual claims against evidence sources
- `constrained-decoding` — restricting generation to outputs satisfying validity rules

### Reinforcement Learning
- `reinforcement-learning` — learning policies via reward signals (includes RLHF for LLM alignment)
- `policy-gradient` — RL methods that optimise policy parameters directly (REINFORCE, TRPO, PPO, actor-critic)
- `temporal-difference` — bootstrapped value learning (TD, SARSA, Q-learning)
- `dynamic-programming` — solving problems via recursive decomposition and value propagation
- `discounting` — geometric decay applied to rewards by temporal distance
- `policy-modelling` — modelling the behavioural policy observed, rather than solving for an optimal one
- `imitation-learning` — learning a policy by mimicking observed behaviour rather than optimising a reward
- `multi-agent` — several interacting decision-makers modelled as separate agents
- `action-space` — the set of choices available to an agent, and how it is discretised
- `simulator` — an environment model used to generate synthetic interaction data
- `agent-based-simulation` — simulating a system by giving agents rules and observing emergent behaviour
- `markov-model` — stochastic processes with the Markov property (Markov chains, MDPs, Markov games)
- `game-theory` — strategic interaction between agents; strategy profiles, payoffs, equilibrium
- `counterfactual` — reasoning about outcomes under hypothetical, unobserved conditions

### Probabilistic Modelling and Inference
- `inference` — computing posteriors or marginals from models and data
- `probabilistic-graphical-model` — graphical models (factor graphs, Bayesian networks, MRFs)
- `factor-graph` — bipartite graphical model for factorised distributions
- `message-passing` — inference algorithms that propagate messages on graphs
- `approximation` — approximate inference and computation methods
- `stochastic-process` — random processes indexed by time
- `point-process` — models of discrete event occurrences in time/space
- `gaussian-process` — distributions over functions used as nonparametric priors
- `mixture-model` — modelling a population as a weighted combination of component distributions
- `expectation-maximization` — iterative maximum-likelihood estimation with latent variables
- `identifiability` — whether model parameters are uniquely determined by the distribution they induce
- `clustering` — unsupervised partitioning of data into groups
- `hierarchical-model` — multilevel models sharing information across units via partial pooling
- `time-series` — observations indexed by time
- `smoothing` — recovering trend from noisy series

### Rating and Comparison
- `ranking-system` — skill rating and ranking algorithms (Elo, TrueSkill, etc.)
- `paired-comparison` — models of pairwise contest outcomes from which latent strengths are inferred
- `matchmaking` — pairing players/teams for fair competition
- `gaming` — online gaming, game design, esports
- `network-analysis` — graph-theoretic description of an interaction structure via centrality and path metrics

### Evaluation and Practice
- `evaluation` — benchmarking, testing, and measuring AI system capabilities
- `reliability` — consistency of a measurement across repeated or split samples
- `predictive-validity` — whether a metric forecasts future outcomes it should
- `construct-validity` — whether a metric measures the construct it claims to
- `calibration` — alignment of predicted probabilities with observed frequencies
- `uncertainty-quantification` — estimating and calibrating model confidence
- `probabilistic-classification` — classifiers that output calibrated probabilities rather than hard labels
- `class-imbalance` — training and evaluating when one label is far rarer than the other
- `proxy-target` — predicting a frequent correlate when the outcome of interest is too rare
- `selection-bias` — systematic non-representativeness of a sample
- `positive-unlabeled-learning` — learning where only positive and unlabelled instances are observed
- `model-selection` — choosing model complexity via penalised likelihood or validation
- `interpretability` — the degree to which a model's outputs can be explained in human terms
- `feature-attribution` — per-feature explanation of a model's output (SHAP, LIME, permutation importance)
- `gradient-boosting` — ensemble methods building additive trees via boosting
- `random-forest` — bagged ensembles of decorrelated decision trees
- `regression` — predicting continuous-valued targets rather than discrete classes
- `cognitive-science` — models of human cognition, psychometrics, and cognitive ability

### Vision
- `computer-vision` — image recognition, object detection, and visual tasks
- `semantic-segmentation` — per-pixel classification of images into semantic categories
- `object-detection` — locating and classifying objects in images
- `optical-flow` — estimating motion fields between consecutive frames
- `metric-learning` — learning distance functions or embeddings for similarity/retrieval

## Entity Tags

- `person` — an individual (researcher, engineer, etc.)
- `researcher` — academic or industry researcher
- `independent-researcher` — a researcher working without institutional affiliation
- `practitioner` — an industry analyst or consultant whose work influences research
- `organisation` — company, lab, institution
- `university` — academic institution
- `research-institute` — non-university research laboratory or institute
- `benchmark-environment` — a shared dataset, simulator, or task suite used as common ground
- `ai-research` — entity or work focused on AI research
- `google` — Google and its divisions
- `microsoft` — Microsoft and its divisions

## Meta Tags

- `needs-review` — flagged for human review
- `contradicted` — contains a known contradiction
- `single-source` — based on only one source (fragile)
- `stub` — placeholder page, needs expansion
- `stale-risk` — ageing and approaching stale day limit
- `imported` — transferred from the sports vault during the 2026-08 migration
