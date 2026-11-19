# UNIT V -- Open Source Models and Programming Frameworks

## Complete Structured Notes

**Unit focus:** Understand how generative AI models are trained and
fine-tuned, how pretrained and open-source models are reused, and how
frameworks such as Hugging Face and LangChain help build, program, and
deploy AI applications.

------------------------------------------------------------------------

## 1. Introduction to Open-Source Models and Programming Frameworks

### 1.1 Generative Models

A **generative model** is an AI model that learns patterns from training
data and creates new content.

It can generate: - **Text:** answers, summaries, stories, and code. -
**Images:** illustrations, realistic images, and designs. - **Audio:**
speech, music, and sound. - **Video:** clips and sequences. -
**Structured content:** tables and JSON.

Examples include GPT-style language models, Llama, Stable Diffusion, and
music-generation models.

### 1.2 Open-Source Models

An **open-source model** is a model whose code, weights, or other
development materials are made available for people to inspect, use,
adapt, or build upon, subject to its licence.

**Important:** "Open source" does not mean exactly the same thing for
every model. Some publishers release model weights but restrict certain
uses or do not publish the complete training data and process. Always
check the model licence and terms.

**Advantages** 1. Can be run locally or on a private server. 2. Can be
customized for a particular task. 3. Gives developers more control over
deployment. 4. Can reduce dependence on a single hosted AI provider. 5.
Supports research and experimentation.

**Limitations** 1. Requires hardware, memory, and technical knowledge.
2. The user is responsible for security and maintenance. 3. Model
quality may be lower than some hosted models. 4. Licence restrictions
may apply. 5. Fine-tuning and inference can be computationally
expensive.

### 1.3 Programming Frameworks

A **programming framework** provides reusable libraries, tools, and
standard methods to develop software applications.

In generative AI, frameworks help developers: - Load pretrained
models. - Prepare datasets and tokenizers. - Train and fine-tune
models. - Connect models to external tools and data. - Evaluate model
performance. - Build APIs and deploy applications.

Examples: Hugging Face Transformers, PyTorch, TensorFlow, and LangChain.

------------------------------------------------------------------------

## 2. Training and Fine-Tuning of Generative Models

### 2.1 Model Training

**Training** is the process through which a model learns patterns from
data by adjusting its internal parameters (weights).

For example, a language model is trained on a large collection of text.
It learns grammar, relationships between words, and patterns of
language.

### 2.2 Basic Training Workflow

``` text
Raw Data
   |
   v
Data Collection
   |
   v
Data Cleaning and Preprocessing
   |
   v
Tokenization / Data Encoding
   |
   v
Model Initialization
   |
   v
Forward Pass (Model Prediction)
   |
   v
Loss Calculation
   |
   v
Backpropagation and Weight Update
   |
   v
Evaluation
   |
   v
Trained Model
```

### 2.3 Main Steps in Training

1.  **Data collection:** Gather relevant text, images, audio, or video.
2.  **Data cleaning:** Remove duplicates, corrupted samples, and
    irrelevant content.
3.  **Preprocessing:** Convert data into a format the model can process.
4.  **Tokenization or encoding:** Convert text into tokens or media into
    numerical representations.
5.  **Model selection:** Choose an architecture such as a Transformer,
    CNN, or diffusion model.
6.  **Forward pass:** The model produces a prediction.
7.  **Loss calculation:** A loss function measures the difference
    between prediction and target.
8.  **Backpropagation:** Gradients are calculated to determine how
    parameters should change.
9.  **Optimization:** An optimizer such as Adam updates model weights.
10. **Evaluation:** Test the model on data not used for training.

### 2.4 Fine-Tuning

**Fine-tuning** means taking an already pretrained model and training it
further on a smaller, task-specific dataset.

Example: A general language model can be fine-tuned using
customer-support conversations so it becomes more suitable for
customer-support responses.

### 2.5 Training vs Fine-Tuning

  -----------------------------------------------------------------------
  Feature                 Training from scratch   Fine-tuning
  ----------------------- ----------------------- -----------------------
  Starting point          New model with          Pretrained model
                          initialized weights     

  Data requirement        Usually very large      Usually smaller and
                                                  task-specific

  Compute requirement     Very high               Often lower

  Time                    Usually longer          Usually shorter

  Purpose                 Learn general patterns  Adapt existing
                                                  knowledge

  Cost                    Usually expensive       Often more affordable
  -----------------------------------------------------------------------

### 2.6 Types of Fine-Tuning

**1. Full fine-tuning** - Updates most or all model parameters. - Offers
extensive adaptation. - Requires more memory and compute.

**2. Parameter-Efficient Fine-Tuning (PEFT)** - Updates only a small
number of additional or selected parameters. - Reduces memory and
training cost.

**3. LoRA (Low-Rank Adaptation)** - A popular PEFT technique. - Adds
small trainable matrices to selected model layers. - Keeps most original
model weights frozen. - The learned adapter can be stored separately.

**4. Instruction fine-tuning** - Trains a model using
instruction-and-response examples. - Helps the model follow user
instructions.

**5. Domain-specific fine-tuning** - Adapts a model to a field such as
education, finance, or law. - Requires reliable, appropriately licensed
data and careful evaluation.

### 2.7 Fine-Tuning Workflow

``` text
Pretrained Model
       |
       v
Prepare Task-Specific Dataset
       |
       v
Format Examples (Input, Expected Output)
       |
       v
Choose Fine-Tuning Method
       |
       v
Train / Update Parameters
       |
       v
Validate and Evaluate
       |
       v
Save Model or Adapter
       |
       v
Deploy for the Target Task
```

### 2.8 Example: College FAQ Model

``` text
Input: What is the library opening time?
Output: The library opens at 9:00 AM.

Input: How can I apply for a bonafide certificate?
Output: Submit the request through the college office portal.
```

After training on a sufficiently varied and verified dataset, the model
may respond in a style suited to college-related questions.

**Benefits:** task-specific responses, consistent output format, and
better adaptation to a domain.

**Risks:** overfitting, biased or incorrect data, forgetting some
general capabilities, and memorizing sensitive information.

------------------------------------------------------------------------

## 3. GPT4All

### 3.1 Introduction

**GPT4All** is an ecosystem of tools that makes it easier to download
and run supported language models locally on a personal computer. It
includes a desktop application and developer interfaces.

It is useful for experimenting with local LLMs without sending every
prompt to a hosted model service. Available models and hardware
requirements depend on the selected model.

### 3.2 Main Features

1.  Runs supported language models locally.
2.  Provides a desktop chat interface.
3.  Offers developer interfaces for application integration.
4.  Supports local experimentation and offline use for compatible models
    after downloading them.
5.  Can help keep prompts and documents on the local machine, depending
    on configuration.

### 3.3 Basic Working

``` text
User Prompt
    |
    v
GPT4All Application / API
    |
    v
Locally Downloaded Language Model
    |
    v
Model Inference
    |
    v
Generated Response
```

### 3.4 Applications

-   Personal AI assistant.
-   Local document question-answering experiments.
-   Text summarization.
-   Drafting and brainstorming.
-   Prototyping with local LLMs.

### 3.5 Advantages and Limitations

**Advantages** - Simple way to experiment with local models. - Can
reduce dependence on cloud inference. - Useful for privacy-conscious
local workflows. - Provides desktop and developer integration options.

**Limitations** - Performance depends on CPU, GPU, RAM, and model
size. - Local models may be slower or less capable than larger hosted
models. - Downloading models requires storage and internet access. -
Local execution does not automatically guarantee privacy or security;
check settings and model sources.

### 3.6 Example Use Case

A student downloads a compatible model in GPT4All and asks it to
summarize a text file of class notes. The application passes the prompt
and relevant text to the local model, which generates a summary.

------------------------------------------------------------------------

## 4. Transfer Learning and Pretrained Models

### 4.1 Transfer Learning

**Transfer learning** is a machine-learning technique in which knowledge
learned by a model on one task or dataset is reused for another related
task.

Example: - A vision model is trained on a large image dataset to
recognize many object categories. - Its learned visual features are
reused to classify plant diseases. - The final classification layer is
adapted to the new plant-disease categories.

### 4.2 Pretrained Model

A **pretrained model** is a model that has already been trained on a
dataset before being used for a new task.

Examples: - BERT pretrained on text. - Llama pretrained on large text
collections. - ResNet pretrained on ImageNet. - TimeSformer pretrained
on video datasets.

### 4.3 Transfer Learning Workflow

``` text
Large General Dataset
        |
        v
Train Base Model
        |
        v
Pretrained Model
        |
        v
Replace / Adapt Task-Specific Layer
        |
        v
Train on Smaller Target Dataset
        |
        v
Evaluate
        |
        v
Final Task-Specific Model
```

### 4.4 Feature Extraction vs Fine-Tuning

**Feature extraction** - Keep the pretrained backbone frozen. - Use it
to extract features. - Train a new classifier or task-specific head.

**Fine-tuning** - Start with pretrained weights. - Unfreeze some or all
layers. - Continue training on the target dataset.

### 4.5 Advantages

1.  Reduces training time.
2.  Requires less labelled data in many cases.
3.  Reuses useful learned features.
4.  Can improve performance on related tasks.
5.  Makes advanced models more accessible.

### 4.6 Limitations

1.  Source and target tasks should be sufficiently related.
2.  Dataset mismatch can cause negative transfer.
3.  Pretrained models may contain bias.
4.  Large models may still require significant compute.
5.  Licence and dataset restrictions must be checked.

### 4.7 Example: Image Classification

Suppose a dataset contains 1,000 images of healthy and diseased leaves.

Instead of training a CNN from the beginning: 1. Load a pretrained
ResNet model. 2. Remove or replace its original classification layer. 3.
Add a new output layer with two classes. 4. Freeze the backbone
initially. 5. Train the new classifier. 6. Optionally unfreeze selected
layers and fine-tune with a small learning rate. 7. Evaluate on separate
test images.

------------------------------------------------------------------------

## 5. Training Vision Models

### 5.1 Vision Model

A **vision model** is an AI model designed to process visual data such
as images or video.

Common tasks: - Image classification. - Object detection. - Image
segmentation. - Pattern recognition. - Video classification. - Visual
question answering.

### 5.2 Common Vision Architectures

**1. CNN (Convolutional Neural Network)** - Uses convolution filters to
learn visual features. - Early layers learn edges and textures. - Deeper
layers learn shapes and object-level patterns. - Commonly used for image
classification and detection.

**2. Vision Transformer (ViT)** - Splits an image into patches. -
Converts patches into embeddings. - Uses Transformer attention to learn
relationships between patches.

**3. ResNet** - A CNN architecture with residual/skip connections. -
Helps train deeper networks by allowing information and gradients to
pass through shortcut paths.

### 5.3 Vision Model Training Pipeline

``` text
Collect Images
     |
     v
Label Images
     |
     v
Split Dataset
(Train / Validation / Test)
     |
     v
Resize, Normalize, Augment
     |
     v
Load / Build Vision Model
     |
     v
Train Model
     |
     v
Validate and Tune
     |
     v
Test Final Model
     |
     v
Deploy
```

### 5.4 Preprocessing Steps

1.  **Resizing:** Convert images to a consistent size.
2.  **Normalization:** Scale pixel values to a suitable range.
3.  **Augmentation:** Apply rotation, crop, flip, or brightness changes
    to improve robustness.
4.  **Label encoding:** Convert class labels into numerical form.
5.  **Dataset splitting:** Keep separate training, validation, and test
    sets.

### 5.5 Important Training Concepts

-   **Epoch:** One complete pass through the training dataset.
-   **Batch size:** Number of samples processed before a parameter
    update.
-   **Learning rate:** Controls the size of weight updates.
-   **Loss function:** Measures prediction error.
-   **Optimizer:** Updates model parameters.
-   **Overfitting:** Model performs well on training data but poorly on
    unseen data.
-   **Regularization:** Techniques such as dropout, weight decay, and
    augmentation that can reduce overfitting.

### 5.6 Evaluation Metrics

-   **Accuracy:** Fraction of predictions that are correct.
-   **Precision:** Of the samples predicted positive, how many are truly
    positive?
-   **Recall:** Of the truly positive samples, how many were found?
-   **F1-score:** Harmonic mean of precision and recall.
-   **Confusion matrix:** Shows correct and incorrect predictions for
    each class.

For imbalanced datasets, accuracy alone may be misleading; precision,
recall, F1-score, and per-class results are important.

### 5.7 Example: Plant Disease Classification

Input: Photograph of a leaf.

Process: 1. Resize the image to the model's expected input size. 2.
Normalize pixel values. 3. Pass the image through a CNN or pretrained
ResNet. 4. Obtain class scores. 5. Select the predicted class. 6.
Display the predicted disease category and confidence score.

------------------------------------------------------------------------

## 6. Google Copilot / AI Coding Assistants

### 6.1 Clarifying the Name

The phrase **"Google Copilot"** is not the standard product name of
Google's main coding assistant. Google's coding assistant is commonly
known as **Gemini Code Assist**. **GitHub Copilot** is a separate
product from GitHub/Microsoft.

If the syllabus or lecturer specifically uses "Google Copilot," follow
that classroom terminology, but understand the distinction.

### 6.2 AI Coding Assistant

An AI coding assistant uses a large language model to help developers
write, understand, debug, and improve code.

### 6.3 Gemini Code Assist

Gemini Code Assist is Google's AI coding assistant. Depending on the
edition and supported environment, it can help with: - Code
completion. - Generating code from natural-language instructions. -
Explaining code. - Finding and fixing errors. - Writing tests. -
Suggesting code improvements.

### 6.4 GitHub Copilot

GitHub Copilot is an AI coding assistant that can provide: - Inline code
suggestions. - Code generation from comments or prompts. - Chat-based
programming help. - Test generation and code explanations. - Assistance
in supported IDEs and workflows.

### 6.5 Example

Prompt: "Write a Java method to check whether a number is prime."

An AI coding assistant may generate a method, explain its logic, and
suggest test cases.

### 6.6 Benefits and Precautions

**Benefits** 1. Speeds up repetitive coding. 2. Helps explain unfamiliar
code. 3. Supports learning and debugging. 4. Can generate tests and
documentation.

**Precautions** 1. Generated code can be incorrect or insecure. 2.
Developers must test and review suggestions. 3. Do not paste secrets,
passwords, or confidential code into tools without authorization. 4.
Check licensing and organizational policies. 5. The developer remains
responsible for the final code.

------------------------------------------------------------------------

## 7. Programming LLMs (Large Language Models)

### 7.1 LLM

A **Large Language Model (LLM)** is a neural network trained on large
amounts of text to understand and generate language.

Most modern LLMs are based on the Transformer architecture.

### 7.2 Basic Transformer / LLM Flow

``` text
Input Text
    |
    v
Tokenizer
    |
    v
Token IDs
    |
    v
Embedding Layer
    |
    v
Transformer Blocks
(Attention + Feed-Forward)
    |
    v
Output / Vocabulary Scores
    |
    v
Next Token Selection
    |
    v
Generated Text
```

### 7.3 Important Concepts

**1. Tokenization** - Splits text into tokens. - A token can be a word,
part of a word, or punctuation. - Each token is mapped to a numerical
ID.

**2. Embeddings** - Convert token IDs into numerical vectors. - These
vectors represent learned information about tokens.

**3. Attention** - Allows the model to weigh relationships between
tokens. - Self-attention helps the model use relevant context from the
input sequence.

**4. Transformer** - A neural-network architecture based on attention
mechanisms. - Commonly used in modern language models.

**5. Context window** - The amount of input and conversation context a
model can consider at one time. - Exact limits depend on the model.

**6. Inference** - Using a trained model to generate a prediction or
response.

**7. Prompt engineering** - Designing clear instructions and context to
guide model output.

### 7.4 Basic LLM Programming Workflow

1.  Select a model.
2.  Load the model and tokenizer or connect to its API.
3.  Prepare a prompt.
4.  Generate output.
5.  Apply output validation or post-processing.
6.  Evaluate quality and safety.

### 7.5 Example: Prompt-Based Programming

``` python
prompt = """
Explain the difference between a list and a tuple in Python.
Give one example of each.
"""

# Send the prompt to a selected LLM using its supported API or library.
# Read and display the generated response.
```

The exact code used to call the model depends on whether it is accessed
through a hosted API or loaded locally.

### 7.6 Generation Parameters

-   **Temperature:** Controls randomness. Lower values usually produce
    more predictable output.
-   **Top-k:** Restricts token selection to the k most likely
    candidates.
-   **Top-p:** Selects from the smallest group of tokens whose
    cumulative probability reaches p.
-   **Max new tokens:** Limits the number of generated tokens.
-   **Stop sequence:** Tells generation when to stop at a specified
    sequence.

### 7.7 Applications

-   Chatbots.
-   Code generation.
-   Text summarization.
-   Translation.
-   Question answering.
-   Document analysis.
-   Retrieval-augmented generation (RAG).

------------------------------------------------------------------------

## 8. LangChain

### 8.1 Introduction

**LangChain** is an open-source framework for developing applications
powered by language models. It provides components for connecting models
with prompts, documents, retrievers, tools, and application logic.

It is useful when an application needs more than a single prompt sent to
an LLM.

### 8.2 Main Components

**1. Models** - Connect to chat models or language models. - The model
may be hosted through an API or run locally, depending on integration.

**2. Prompts** - Reusable templates for constructing model
instructions. - Allow application data to be inserted into a prompt.

**3. Output Parsers** - Convert model output into a desired format, such
as a string or structured data.

**4. Document Loaders** - Load data from PDF, text, web pages, and other
supported sources.

**5. Text Splitters** - Divide long documents into smaller chunks. -
Makes retrieval and context management easier.

**6. Embeddings** - Convert text into vectors that represent semantic
meaning.

**7. Vector Stores** - Store embeddings and associated text for
similarity search. - Examples include FAISS and Chroma.

**8. Retrievers** - Find relevant documents or chunks for a user query.

**9. Tools** - Allow an LLM-based application or agent to call functions
or external services.

**10. Chains** - Connect multiple processing steps into an application
workflow.

### 8.3 Basic LangChain Architecture

``` text
User Question
      |
      v
Prompt Template
      |
      v
Language Model
      |
      v
Output Parser
      |
      v
Final Answer
```

### 8.4 LangChain with RAG

**RAG (Retrieval-Augmented Generation)** combines document retrieval
with LLM generation. It supplies relevant external information to the
model when answering a question.

``` text
DOCUMENT INGESTION

PDF / TXT / DOCS
       |
       v
Document Loader
       |
       v
Text Splitter
       |
       v
Embedding Model
       |
       v
Vector Database
       |
       |  (stored chunks and vectors)
       |
USER QUESTION
       |
       v
Question Embedding
       |
       v
Retriever / Similarity Search
       |
       v
Relevant Document Chunks
       |
       v
Prompt + Retrieved Context
       |
       v
LLM
       |
       v
Grounded Answer
```

### 8.5 RAG Example: College Information Assistant

Suppose a college uploads attendance rules, exam guidelines, department
information, and hostel rules.

When a student asks, "What is the minimum attendance requirement?", the
system retrieves the relevant section from the documents and gives it to
the LLM as context. The model then generates an answer based on that
retrieved information.

### 8.6 Advantages

1.  Modular development.
2.  Supports multiple model providers and integrations.
3.  Simplifies document-based question answering.
4.  Supports retrieval and tool integration.
5.  Useful for prototypes and production workflows.

### 8.7 Limitations

1.  Framework abstractions can make debugging harder.
2.  Integrations and APIs can change between versions.
3.  Retrieval quality depends on document quality, chunking, and
    embeddings.
4.  An LLM can still produce unsupported statements.
5.  Developers must handle security, access control, and evaluation.

------------------------------------------------------------------------

## 9. Open-Source Models

### 9.1 Meaning

Open-source or openly released models allow developers to access and use
model artifacts under the terms set by the publisher. Available
artifacts may include model weights, configuration files, tokenizer
files, inference code, or training code.

The exact degree of openness varies. A model with downloadable weights
is not necessarily fully open source.

### 9.2 Examples of Model Families

  -----------------------------------------------------------------------
  Model family            Main area               Typical uses
  ----------------------- ----------------------- -----------------------
  Llama                   Language models         Chat, summarization,
                                                  coding, question
                                                  answering

  Mistral                 Language models         Text generation and
                                                  assistant applications

  Qwen                    Language and multimodal Chat, coding,
                          models                  reasoning, multimodal
                                                  tasks

  Gemma                   Language models         Text and developer
                                                  experiments

  Stable Diffusion        Image generation        Text-to-image
                                                  generation

  Whisper                 Speech recognition      Speech-to-text
                                                  transcription
  -----------------------------------------------------------------------

Availability, capabilities, and licence terms differ by model and
version.

### 9.3 Selecting an Open Model

1.  Identify the task: chat, coding, vision, audio, or other.
2.  Compare quality on relevant benchmarks and real examples.
3.  Check model size and hardware requirements.
4.  Check supported languages and context length.
5.  Read the licence and usage restrictions.
6.  Evaluate latency, cost, privacy, and deployment needs.
7.  Test the model on representative data.

### 9.4 Benefits

-   Customization.
-   Local or private deployment options.
-   Greater infrastructure control.
-   Community contributions.
-   Useful for research and learning.

### 9.5 Challenges

-   Hardware and maintenance requirements.
-   Quality and safety evaluation.
-   Licence compliance.
-   Model updates and security.
-   Responsible handling of training and user data.

------------------------------------------------------------------------

## 10. Llama

### 10.1 Introduction

**Llama** is a family of large language models released by Meta.
Different generations and sizes are available, with capabilities and
licence terms that vary by release.

Llama models can be used for: - Text generation. - Conversational
assistants. - Summarization. - Question answering. - Coding
assistance. - Applications built with retrieval systems.

### 10.2 General Llama Workflow

``` text
User Input
    |
    v
Application / Chat Interface
    |
    v
Llama Tokenizer
    |
    v
Llama Model
    |
    v
Generated Tokens
    |
    v
Decoded Response
```

### 10.3 Important Characteristics

1.  Transformer-based architecture.
2.  Available in different model sizes and versions.
3.  Can be used through supported local inference tools or hosted
    services.
4.  Can be adapted for specific tasks using fine-tuning methods.
5.  Can be connected to RAG pipelines and external tools.

### 10.4 Llama Use Case

A company wants an internal assistant that answers questions from its
technical documentation.

Possible approach: 1. Select a Llama model whose licence and hardware
requirements are suitable. 2. Prepare internal documentation. 3. Build a
RAG pipeline to retrieve relevant passages. 4. Provide retrieved
passages and the question to Llama. 5. Display an answer with document
references. 6. Evaluate answer accuracy and access permissions.

### 10.5 Advantages and Limitations

**Advantages** - Flexible model choices. - Supports local and hosted
deployment options. - Can be adapted and integrated into applications. -
Strong ecosystem of developer tools.

**Limitations** - Larger models need more memory and compute. - Output
quality depends on model version and task. - Licence terms must be
followed. - Generated answers require validation.

------------------------------------------------------------------------

## 11. Programming for TimeSformer

### 11.1 What is TimeSformer?

**TimeSformer (Time-Space Transformer)** is a Transformer-based
architecture designed for video understanding. It applies attention
mechanisms to video frames to learn spatial information within frames
and temporal relationships across frames.

A video model must understand: - **Spatial information:** What objects
and patterns appear in each frame? - **Temporal information:** How do
objects and actions change over time?

### 11.2 Why TimeSformer?

A video is a sequence of frames. Recognizing an action often requires
observing how frames change.

Example: - One frame may show a person with a raised hand. - Several
consecutive frames may show the person waving. - The temporal sequence
helps identify the action as waving.

### 11.3 Basic Architecture

``` text
Input Video
    |
    v
Sample Video Frames
    |
    v
Divide Frames into Patches
    |
    v
Patch Embeddings + Position Information
    |
    v
Transformer Blocks
    |
    +----------------------+
    |                      |
    v                      v
Spatial Attention    Temporal Attention
    |                      |
    +----------+-----------+
               |
               v
        Feature Representation
               |
               v
        Classification Head
               |
               v
        Predicted Action
```

### 11.4 Spatial and Temporal Attention

**Spatial attention** - Learns relationships between patches within a
frame. - Helps identify objects, shapes, and scene information.

**Temporal attention** - Learns relationships across different frames. -
Helps recognize motion and action over time.

**Divided space-time attention** - Applies spatial and temporal
attention in separate stages rather than performing one large joint
attention operation. - This can reduce computational cost compared with
full joint space-time attention.

### 11.5 TimeSformer Workflow

1.  Collect labelled video clips.
2.  Sample a fixed number of frames from each clip.
3.  Resize and normalize frames as required by the model.
4.  Convert frames into patches and embeddings.
5.  Pass embeddings through the Transformer.
6.  Use the classification head to predict an action.
7.  Train or fine-tune using labelled video data.
8.  Evaluate on unseen video clips.

### 11.6 Programming Outline

The following is a **conceptual PyTorch/Hugging Face-style outline**.
Exact class names, checkpoint availability, input shapes, and processor
arguments depend on the installed Transformers version and selected
checkpoint.

``` python
# Conceptual outline: video classification with a pretrained model

# 1. Install compatible packages:
# pip install torch transformers

# 2. Select a compatible pretrained video model/checkpoint.
# 3. Load its video processor and model.
# 4. Read a video and sample the required number of frames.
# 5. Preprocess frames using the processor.
# 6. Pass the processed frames to the model.
# 7. Convert output scores into predicted class labels.

# Pseudocode:
#
# frames = sample_frames(video_path)
# inputs = video_processor(frames, return_tensors="pt")
# outputs = model(**inputs)
# predicted_class = outputs.logits.argmax(dim=-1)
```

### 11.7 Applications

-   Human action recognition.
-   Sports-video analysis.
-   Surveillance-event understanding, subject to privacy and legal
    requirements.
-   Gesture recognition.
-   Video content classification.
-   Activity recognition in research.

### 11.8 Advantages and Challenges

**Advantages** - Learns spatial and temporal video information. - Uses
Transformer attention. - Useful for action-recognition tasks.

**Challenges** - Video data requires substantial storage. - Training can
be computationally expensive. - Frame sampling can miss important
motion. - Dataset quality and class balance affect performance. -
Real-time inference may require optimization.

------------------------------------------------------------------------

## 12. Deployment of Generative AI Models

### 12.1 Deployment

**Deployment** is the process of making a trained model available so
that users or other software systems can access it.

A model may be deployed: - Locally on a computer. - On a private
server. - On a cloud platform. - Through an API. - Inside a mobile or
desktop application, if supported.

### 12.2 Deployment Workflow

``` text
Trained Model
     |
     v
Save Model + Tokenizer / Processor
     |
     v
Prepare Inference Code
     |
     v
Build API or Application
     |
     v
Package Dependencies
     |
     v
Deploy to Server / Cloud / Local Device
     |
     v
User Sends Request
     |
     v
Model Generates Prediction
     |
     v
Return Response
     |
     v
Monitor and Maintain
```

### 12.3 Deployment Options

**1. Local deployment** - Model runs on the user's own machine. - Useful
for learning, experimentation, or selected privacy-sensitive
workflows. - Limited by local hardware.

**2. Cloud deployment** - Model runs on cloud infrastructure. - Can
provide scalable compute resources. - Requires attention to cost,
latency, security, and data handling.

**3. API deployment** - Model is exposed through an endpoint. - Other
applications send requests and receive responses. - Requires
authentication, input validation, and rate limiting.

**4. Edge deployment** - Model runs on or near the device where data is
generated. - Can reduce network latency. - Model size and device
resources are important constraints.

### 12.4 Common Tools

-   Hugging Face for model hosting, inference, and application demos.
-   FastAPI or Flask for Python APIs.
-   Docker for packaging applications.
-   Cloud platforms for hosting and scaling.
-   Monitoring tools for performance and reliability.

### 12.5 Deployment Considerations

1.  **Latency:** Time taken to return a response.
2.  **Throughput:** Number of requests handled in a given time.
3.  **Scalability:** Ability to handle more users or requests.
4.  **Security:** Authentication, authorization, and data protection.
5.  **Cost:** Compute, storage, bandwidth, and API expenses.
6.  **Monitoring:** Track errors, latency, usage, and model quality.
7.  **Versioning:** Track model and application versions.
8.  **Responsible AI:** Test for harmful, biased, or unreliable outputs.

### 12.6 Example: Text Generation API

A developer loads a pretrained language model in a Python application,
creates a `/generate` endpoint, accepts a user's prompt, runs model
inference, and returns generated text as JSON.

A production implementation should also validate input, restrict request
size, protect credentials, handle errors, and monitor resource usage.

------------------------------------------------------------------------

## 13. Hugging Face

### 13.1 Introduction

**Hugging Face** is an AI platform and open-source ecosystem that
provides tools, libraries, and a community hub for sharing and using
machine-learning models and datasets.

It is widely used for Natural Language Processing (NLP), computer
vision, audio, and multimodal AI.

### 13.2 Important Hugging Face Tools

**1. Model Hub** - Hosts model repositories. - Provides model cards,
configuration, weights, and usage information. - Lets developers search
for models by task and capability.

**2. Datasets** - Provides access to datasets for machine-learning
tasks. - Supports loading, processing, and sharing datasets.

**3. Transformers** - A library for using pretrained Transformer
models. - Supports tasks such as text classification, generation,
summarization, translation, and more.

**4. Tokenizers** - Provides fast tokenization tools. - Converts text
into model-compatible tokens.

**5. Diffusers** - Provides tools for diffusion models, especially
image, video, and audio generation workflows supported by the library.

**6. Accelerate** - Helps run training and inference across different
hardware configurations.

**7. Spaces** - Hosts interactive machine-learning demos and
applications.

### 13.3 Hugging Face Workflow

``` text
Search Model Hub
      |
      v
Select Pretrained Model
      |
      v
Read Model Card and Licence
      |
      v
Load Tokenizer / Processor
      |
      v
Load Model
      |
      v
Prepare Input
      |
      v
Run Inference or Fine-Tune
      |
      v
Evaluate
      |
      v
Deploy / Share Application
```

### 13.4 Transformers Example

A simple text-generation example using a supported checkpoint:

``` python
# Install:
# pip install transformers torch

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

result = generator(
    "Artificial intelligence is",
    max_new_tokens=40,
    num_return_sequences=1
)

print(result[0]["generated_text"])
```

**Explanation** 1. `pipeline` provides a convenient high-level
interface. 2. `"text-generation"` selects the task. 3. `model` specifies
the pretrained model checkpoint. 4. The input string is the prompt. 5.
`max_new_tokens` limits the number of newly generated tokens. 6. The
result contains the generated text.

The first run may download model files. Hardware and library
compatibility affect execution.

### 13.5 Hugging Face Advantages

1.  Large collection of pretrained models.
2.  Reusable libraries and tools.
3.  Supports multiple AI tasks.
4.  Helps developers compare and experiment with models.
5.  Model cards provide information about intended use and limitations.
6.  Supports sharing models, datasets, and demos.

### 13.6 Limitations and Precautions

-   Model quality varies.
-   Some models need powerful hardware.
-   Checkpoint licences and usage restrictions differ.
-   Downloaded models should come from trusted sources.
-   Review model cards, limitations, and security considerations before
    deployment.

------------------------------------------------------------------------

## 14. Comparison of Important Tools and Technologies

  -----------------------------------------------------------------------
  Technology              Main purpose            Typical use
  ----------------------- ----------------------- -----------------------
  GPT4All                 Run supported LLMs      Local AI assistant
                          locally                 

  Hugging Face            Model and dataset       Load, train, fine-tune,
                          ecosystem               and share models

  Transformers            Use Transformer         NLP, vision, audio, and
                          architectures           multimodal tasks

  LangChain               Build LLM-powered       RAG, tools, and
                          applications            multi-step workflows

  Llama                   Family of language      Chat, text generation,
                          models                  coding

  TimeSformer             Video understanding     Video/action
                          architecture            classification

  Gemini Code Assist      AI coding assistant     Code completion and
                          from Google             programming help

  GitHub Copilot          AI coding assistant     Code generation and
                                                  developer productivity

  PyTorch                 Deep-learning framework Build and train neural
                                                  networks

  FastAPI                 Python web framework    Expose model inference
                                                  through APIs

  Docker                  Application             Package and deploy
                          containerization        model services
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 15. Important Differences for Exams

### 15.1 Training vs Fine-Tuning

-   Training starts with a newly initialized model.
-   Fine-tuning starts with a pretrained model and adapts it to a target
    task.
-   Training usually needs more data and computation.
-   Fine-tuning can be faster and cheaper.

### 15.2 Pretrained Model vs Transfer Learning

-   A pretrained model is the model already trained on earlier data.
-   Transfer learning is the method of reusing that model's learned
    knowledge for a new task.

### 15.3 Feature Extraction vs Fine-Tuning

-   Feature extraction freezes the pretrained backbone and trains a new
    task-specific head.
-   Fine-tuning updates some or all pretrained model parameters.

### 15.4 CNN vs Vision Transformer

-   CNNs use convolution operations to learn local visual patterns.
-   Vision Transformers divide images into patches and use attention to
    learn relationships.
-   Both can be used for image-recognition tasks.

### 15.5 Image Model vs Video Model

-   An image model processes one image at a time.
-   A video model processes a sequence of frames and learns temporal
    relationships.
-   TimeSformer is designed for video understanding.

### 15.6 LangChain vs Hugging Face

-   Hugging Face provides models, datasets, and libraries for machine
    learning.
-   LangChain helps connect language models with prompts, retrievers,
    tools, and application workflows.
-   They can be used together in one application.

### 15.7 Local vs Cloud Deployment

-   Local deployment runs the model on a local device or private
    machine.
-   Cloud deployment runs the model on remote cloud infrastructure.
-   Local execution offers control but is hardware-limited; cloud
    execution can scale but introduces hosting costs and data-transfer
    considerations.

------------------------------------------------------------------------

## 16. Important Exam Questions

### Short-Answer Questions (2--3 Marks)

1.  Define a generative model.
2.  What is an open-source model?
3.  Define fine-tuning.
4.  What is transfer learning?
5.  What is a pretrained model?
6.  What is GPT4All?
7.  Define an LLM.
8.  What is tokenization?
9.  What is an embedding?
10. What is LangChain?
11. What is RAG?
12. What is Llama?
13. What is TimeSformer?
14. Define model deployment.
15. What is Hugging Face?
16. What is a vector database?
17. What is LoRA?
18. What is a context window?

### Long-Answer Questions (5--10 Marks)

1.  Explain the training and fine-tuning process of generative models
    with a diagram.
2.  Differentiate training from scratch and fine-tuning.
3.  Explain GPT4All, its features, working, advantages, and limitations.
4.  Explain transfer learning and pretrained models with a suitable
    example.
5.  Describe the complete workflow for training a vision model.
6.  Explain AI coding assistants and their applications.
7.  Explain the architecture and working of an LLM.
8.  Explain LangChain components and its architecture.
9.  Explain RAG using LangChain with a neat diagram and example.
10. Discuss open-source model families and their advantages and
    limitations.
11. Explain the Llama model family and its applications.
12. Explain TimeSformer architecture, spatial attention, and temporal
    attention.
13. Explain the steps involved in deploying a generative AI model.
14. Explain Hugging Face and its important libraries.
15. Compare LangChain, Hugging Face, GPT4All, and Llama.

### Very Important Topics to Prepare First

1.  Training and fine-tuning of generative models.
2.  Transfer learning and pretrained models.
3.  Training vision models.
4.  LLM architecture and programming concepts.
5.  LangChain architecture and RAG diagram.
6.  Open-source models and Llama.
7.  TimeSformer architecture.
8.  Deployment of generative models.
9.  Hugging Face tools and a basic Transformers example.
10. GPT4All and AI coding assistants.

------------------------------------------------------------------------

## 17. Quick Revision -- One-Line Definitions

-   **Generative model:** A model that learns data patterns and
    generates new content.
-   **Training:** Learning model parameters from data.
-   **Fine-tuning:** Further training a pretrained model for a specific
    task.
-   **Transfer learning:** Reusing knowledge learned on one task for
    another task.
-   **Pretrained model:** A model already trained on an earlier dataset.
-   **GPT4All:** A toolkit and application ecosystem for running
    supported LLMs locally.
-   **Vision model:** A model that processes images or videos.
-   **LLM:** A large neural language model that processes and generates
    text.
-   **Open-source model:** A model released with accessible artifacts
    under specified licence terms.
-   **Llama:** Meta's family of large language models.
-   **LangChain:** A framework for building applications powered by
    language models.
-   **RAG:** A method that retrieves external information and supplies
    it to an LLM when generating an answer.
-   **TimeSformer:** A Transformer architecture for video understanding
    using spatial and temporal information.
-   **Deployment:** Making a trained model available for real-world use.
-   **Hugging Face:** A platform and ecosystem for models, datasets, and
    machine-learning tools.
-   **Gemini Code Assist:** Google's AI coding assistant.
-   **GitHub Copilot:** GitHub's AI coding assistant.

------------------------------------------------------------------------

## 18. Suggested Study Order

For efficient exam preparation, study the unit in this order:

1.  **Foundations:** Open-source models, generative models, LLM basics.
2.  **Model adaptation:** Training, fine-tuning, transfer learning,
    pretrained models.
3.  **Vision:** Vision-model training and TimeSformer.
4.  **Tools:** GPT4All, Llama, Hugging Face.
5.  **Application development:** LangChain, RAG, programming LLMs.
6.  **Production:** Deployment, evaluation, security, and monitoring.
7.  **Revision:** Practise diagrams, comparison tables, definitions, and
    long-answer questions.

**Exam tip:** For a 10-mark answer, write the definition, explain the
components or steps in numbered points, draw the relevant
workflow/architecture diagram, give one practical example, and finish
with advantages and limitations.

------------------------------------------------------------------------

## 19. Further Reading and Official Documentation

Use official documentation for version-specific instructions and model
licences:

-   Hugging Face: https://huggingface.co/docs
-   Transformers: https://huggingface.co/docs/transformers
-   LangChain: https://python.langchain.com/docs/
-   GPT4All: https://docs.gpt4all.io/
-   Meta Llama: https://www.llama.com/
-   TimeSformer paper: https://arxiv.org/abs/2102.05095
-   Google Gemini Code Assist: https://codeassist.google/
-   GitHub Copilot documentation: https://docs.github.com/en/copilot

> **Note:** These notes cover every topic listed in the supplied Unit V
> syllabus. Product features, model availability, API syntax, and
> licences can change; use official documentation for implementation
> details.
