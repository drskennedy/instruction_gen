from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import argparse
import random
import timeit
import json
from datetime import timedelta
import LoadVectorize

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def generate_instructions(db,QA_PROMPT,llm) -> None:
    output_parser = StrOutputParser()
    # Custom QA Chain
    chain = (
        {"context": RunnablePassthrough() , "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | output_parser
        )

    # access vector for k-doc chunks
    vs = db.__dict__.get("docstore")
    docstore_id_list = list(db.__dict__.get("index_to_docstore_id").values())
    rand_doc_id_list = random.choices(docstore_id_list, k=100)

    query = '''
    Please generate two questions about SteelHead based on the provided context. The question should be around SteelHead WAN acceleration and its related concepts only. The questions should start with any of the following: "What", "How', "Is there a", "What are the", "How do I", "When is it", "Does SteelHead have", "How to", "What is the difference", "Which", "List". You do not need to provide an answer or category to each question.
    '''
    qfile = open("instructions.txt", "w")
    start_gen = timeit.default_timer()
    for i,doc_id in enumerate(rand_doc_id_list):
        start = timeit.default_timer()
        a_doc = vs.search(doc_id)
        #print(f'CHOSEN DOC => {a_doc.page_content}\n_________________\n')
        result = chain.invoke({"question": query, "context": a_doc.page_content})
        resp_time = timeit.default_timer() - start # seconds
        print(f'{"-"*50}\nQ #{i}: {result}\nTime: {resp_time}\n{"-"*50}\n')
        qfile.write(result[3:])
    qfile.close()
    gen_time = timeit.default_timer() - start_gen # seconds
    print(f'Total generation time => {timedelta(seconds=gen_time)}')
    
def generate_training(db,bm25_r,QA_PROMPT,llm) -> None:
    faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}, max_tokens_limit=1000)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_r,faiss_retriever],weights=[0.3,0.7])
    output_parser = StrOutputParser()
    # Custom QA Chain
    chain = (
        {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | output_parser
        )
    with open('instructions.txt') as tfile:
        instructions = tfile.readlines()
    start_t_gen = timeit.default_timer()
    train_lines = list()
    for i, instruction in enumerate(instructions, start=1):
        print(f"Handling ({i}/{len(instructions)}):")
        start = timeit.default_timer()
        try:
            answer = chain.invoke(instruction)
        except Exception as e:
            print(f'FAILED for => {e}')
            continue
        resp_time = timeit.default_timer() - start # seconds
        print(f'{"-"*50}\nQ #{i}: {instruction}\nA:{answer}\nTime: {resp_time}\n{"-"*50}\n')
        result = json.dumps({
            'text': f'<s>[INST] {instruction}[/INST] {answer}</s>'
        }) + "\n"
        with open('train_valid.jsonl', 'a') as file:
            file.write(result)
        train_lines.append(result)
    gen_time = timeit.default_timer() - start_t_gen # seconds
    with open('valid.jsonl', 'w') as file:
        file.writelines(train_lines[:int(len(train_lines) * 0.2)])
    with open('train.jsonl', 'w') as file:
        file.writelines(train_lines[int(len(train_lines) * 0.2):])
    print(f'Total training generation time => {timedelta(seconds=gen_time)}')


def main(is_gen_instruct=False,is_gen_training=False):
    # Prompt template 
    qa_template = """<s>[INST] You are a helpful assistant.
    Use the following context to answer the question below accurately and concisely:
    {context}
    [/INST] </s>{question}
    """

    # Create a prompt instance 
    QA_PROMPT = PromptTemplate.from_template(qa_template)

    llm = LlamaCpp(
        model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.01,
        max_tokens=2000,
        top_p=1,
        verbose=False,
        n_ctx=3000
    )
    db,bm25_r = LoadVectorize.load_db()
    if is_gen_instruct:
        generate_instructions(db,QA_PROMPT,llm) 
    elif is_gen_training:
        generate_training(db,bm25_r,QA_PROMPT,llm) 

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser("Script to generate instructions or training data for LLM fine-tuning")
    group = parser.add_mutually_exclusive_group()

    # Adding optional argument
    group.add_argument("-i", "--instructions", action='store_true', help = "Generate Instructions")
    group.add_argument("-t", "--training", action='store_true', help = "Generate Training and Validation Data")

    # Read arguments from command line
    args = parser.parse_args()
    if args.instructions:
        main(is_gen_instruct=args.instructions)
    elif args.training:
        main(is_gen_training=args.training)
