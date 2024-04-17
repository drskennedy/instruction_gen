# Automating Instruction Generation For Internal Documents Using Mistral 7B for LLM Fine-Tuning

**Step-by-step guide on Medium**: [Automating Instruction Generation for Documents for LLM Fine-Tuning](https://medium.com/@heelara/automating-instruction-generation-for-documents-for-llm-fine-tuning-5180d7288ccc)
___
## Context
Curating instruction and training datasets for fine-tuning a language model is a painstaking exercise, especially if you are considering to use your domain knowledge stored in your internal documents.
In this project, we will develop a script to automate the generation of instruction as well as training datasets. Even though the instructions are automatically generated, it still needs to be checked and verified manually to ensure a good list of instructions are available to ensure an acceptable quality training dataset can be produced.
<br><br>
![System Design](/assets/architecture.png)
___
## How to Install
- Create and activate the environment:
```
$ python3.10 -m venv llm_tuning
$ source llm_tuning/bin/activate
```
- Install libraries:
```
$ pip install -r requirements.txt
```
- Download mistral-7b-instruct-v0.2.Q4_K_M.gguf from [TheBloke HF report](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) to directory `models`.
- Run script `main.py` with option -i to start the instruction generation:
```
$ python main.py -i
```
- Run script `main.py` with option -t to start the training dataset generation:
```
$ python main.py -t
```
___
## Quickstart
- To start the app, launch terminal from the project directory and run the following command:
```
$ source llm_tuning/bin/activate
$ python main.py -i
```
- Here is a sample instruction generation run:
```
$ python main.py -i
--------------------------------------------------
Q #0: 
1. Where do I find the QoS settings in SteelHead?
2. Is there a difference between MX-TCP and TCP in terms of handling packet loss?
Time: 57.88847145799991
--------------------------------------------------

--------------------------------------------------
Q #1: 
1. Where do I find information about configuring SSL on SteelHead?
2. Is there a requirement for a trust relationship between Client Accelerators for SSL configuration on SteelHead?
Time: 47.30005858300001
--------------------------------------------------
...
Q #199:
1. What is correct addressing in the context of SteelHead WAN acceleration?
2. How does correct addressing enable connection pooling acceleration in SteelHead?
Time: 63.51242004099913
--------------------------------------------------

Total generation time => 4:20:10.565294
```
- Here is a sample training dataset generation run:
```
$ python main.py -t
Handling (1/150):
--------------------------------------------------
Q #1: Is there a difference between MX-TCP and TCP in terms of handling packet loss?
A:
Yes, there is a difference between MX-TCP and TCP in terms of handling packet loss. MX-TCP is designed to handle packet loss without a decrease in throughput, while TCP typically experiences a decrease in throughput when there is packet loss. MX-TCP effectively handles packet loss without error correction packets over the WAN through forward error correction (FEC).
Time: 152.08327770899996

Handling (2/150):
--------------------------------------------------
Q #2: Where is the configuration for enabling Connection Forwarding between SteelHeads in a cluster?
A:
To enable Connection Forwarding between SteelHeads in a cluster, you need to configure the in-path0_0 IP address of the two SteelHeads as neighbors in the CLI of each SteelHead. Then, you can enter the following commands in each SteelHead's CLI:

enable
configure terminal
SteelHead communication enable
SteelHead communication multi-interface enable
SteelHead name <SteelHead name> main-ip <SteelHead IP address>

Once you have enabled Connection Forwarding, you can configure fail-to-block and allow-failure commands to provide greater resiliency and redundancy in your ITD deployment.
Time: 215.70585895799923
...
Handling (150/150):
--------------------------------------------------
Q #150: How does correct addressing enable connection pooling acceleration in SteelHead?

A:

Correct addressing enables connection pooling acceleration in SteelHead by allowing it to create several TCP connections between each other before they are needed. This is because correct addressing uses specific values in the TCP/IP packet headers, which allows SteelHeads to detect what types of client and server IP addresses and ports are needed. When transparent addressing is enabled, SteelHeads cannot create the TCP connections in advance because they cannot detect what types of client and server IP addresses and ports are needed. If the number of connections that you want to accelerate exceeds the limit of the SteelHead model, the excess connections are passed through unaccelerated by the SteelHead.
Time: 159.28865737500018
--------------------------------------------------

Total training generation time => 9:20:06.321521
```
___
## Key Libraries
- **LangChain**: Framework for developing applications powered by language models
- **FAISS**: Open-source library for efficient similarity search and clustering of dense vectors.
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Open-source pre-trained transformer model for embedding text to a dense vector space for tasks like cosine similarity calculation.

___
## Files and Content
- `models`: Directory hosting the downloaded LLM in GGUF format
- `opdf915_index`: Directory for FAISS index and vectorstore
- `main.py`: Main Python script to launch the application
- `LoadVectorize.py`: Python script to load the pdf document, split and vectorize
- `requirements.txt`: List of Python dependencies (and version)
___

## References
- https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
- https://github.com/ml-explore/mlx
