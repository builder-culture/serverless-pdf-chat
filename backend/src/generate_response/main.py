import os, json
import boto3
from aws_lambda_powertools import Logger
from langchain.llms.bedrock import Bedrock
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]


s3 = boto3.client("s3")
logger = Logger()


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    event_body = json.loads(event["body"])
    file_name = event_body["fileName"]
    human_input = event_body["prompt"]
    conversation_id = event["pathParameters"]["conversationid"]

    user = event["requestContext"]["authorizer"]["claims"]["sub"]

    s3.download_file(BUCKET, f"{user}/{file_name}/index.faiss", "/tmp/index.faiss")
    s3.download_file(BUCKET, f"{user}/{file_name}/index.pkl", "/tmp/index.pkl")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    ##'max_tokens_to_sample':8191

    inference_modifier = {'max_tokens_to_sample':4096, 
                      "temperature":0.2,
                      "top_k":500,
                      "top_p":1,
                      "stop_sequences": ["\n\nHuman"]
                     }


    bedrock_model_id = "anthropic.claude-v2:1"
    ##bedrock_model_id = "anthropic.claude-v1"

    embeddings, llm = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime,
        region_name="us-east-1",
    ), Bedrock(
        model_id=bedrock_model_id, 
        client=bedrock_runtime, 
        region_name="us-east-1",
        model_kwargs=inference_modifier
    )

    faiss_index = FAISS.load_local("/tmp", embeddings)

    message_history = DynamoDBChatMessageHistory(
        table_name=MEMORY_TABLE, session_id=conversation_id
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        input_key="question",
        output_key="answer",
        return_messages=True,
        
    )

    # Retrieve more documents with higher diversity
    # Useful if your dataset has many similar documents
    faiss_retriever = faiss_index.as_retriever(search_kwargs={'k': 7})
    #faiss_retriever = faiss_index.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=faiss_retriever,
        memory=memory,
        return_source_documents=True,
    )

    qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
                                                                   
        Human: Eres un asistente para buscar información contenida en documentos y responder de forma concisa. 
        Utiliza el siguiente contexto del documento o fragmentos de documentos para dar una respuesta concisa y analítica.
        No generarás una respuesta fuera del contexto del documento o fragmentos de documentos proporcionados, 
        tampoco elaborarás inferencias o entendidos sin un soporte.  
        
        Assistant: Entendido

        {chat_history}
                                                                                                                                           
        <context>                                                   
        {context}                                                                
        </context>
                                                                          
        Human: Utiliza el contexto del documento proporcionado para dar una respuesta concisa y analítica a la pregunta proporcionada, sin hacer inferencias o entendidos.
        Proporciona primero la respuesta y después cita al menos uno de los apartados o fragmentos  que fundamentan la respuesta,
        Si es muy extenso el fundamento de la respuesta basta con citar en la respuesta secciones o apartados en los que se soporta la respuesta.
        Muy importante, si no conoces la respuesta, solamente di que no sabes, no trates de generar una respuesta.

        Assistant: Entendido, te proporcionare una respuesta concisa y analítica mediante el contexto del documento o fragmentos de documentos proporcionados, 
        y lo hare de forma breve y concisa. 
        No generaré respuestas, o inferencias, o entendidos sin un fundamento.                                                                   
                                                                                                                                         
        Human: {question}
                                                  
        Assistant: """)
    
    res = qa({"question": human_input})

    logger.info(res)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps(res["answer"]),
    }
