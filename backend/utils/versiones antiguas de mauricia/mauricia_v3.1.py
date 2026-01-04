import os
import sys
from dotenv import load_dotenv

# --- LIBRERÍAS DE LANGCHAIN ---
from langchain_ollama import ChatOllama  # Para el cerebro local
from langchain_openai import OpenAIEmbeddings # Para leer tu DB actual
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Cargar variables de entorno (.env)
load_dotenv()

# --- CONFIGURACIÓN ---
CARPETA_DB = "chroma_db_prod" # Usamos la carpeta que ya creaste
MODELO_OLLAMA = "llama3.1"    # Asegúrate de tenerlo instalado: ollama pull llama3.1
MODELO_EMBEDDINGS = "text-embedding-3-small"

def iniciar_mauricia_local():
    print("🏠 Iniciando MauricIA en MODO LOCAL...")

    # --- A) CARGAR MODELO DE EMBEDDINGS ---
    # Necesitamos esto porque tu base de datos 'chroma_db_prod' fue creada con OpenAI.
    # Si usáramos HuggingFace aquí, daría error de dimensiones.
    try:
        embedding_function = OpenAIEmbeddings(
            model=MODELO_EMBEDDINGS,
            api_key=os.getenv("GITHUB_TOKEN"),
            base_url="https://models.inference.ai.azure.com"
        )
        print("✅ Embeddings configurados (Azure/OpenAI).")
    except Exception as e:
        print(f"❌ Error configurando Embeddings: {e}")
        sys.exit(1)

    # --- B) CONECTAR A LA BASE DE DATOS VECTORIAL ---
    if not os.path.exists(CARPETA_DB):
        print(f"❌ ERROR: No encuentro la carpeta '{CARPETA_DB}'.")
        print("   Ejecuta primero el script 'crear_cerebro_cloud.py' para generarla.")
        sys.exit(1)

    vector_db = Chroma(
        persist_directory=CARPETA_DB,
        embedding_function=embedding_function
    )
    print(f"✅ Base de datos '{CARPETA_DB}' cargada correctamente.")

    # --- C) CONFIGURAR EL LLM LOCAL (OLLAMA) ---
    print(f"🦙 Conectando con Ollama ({MODELO_OLLAMA})...")
    try:
        llm = ChatOllama(
            model=MODELO_OLLAMA,
            temperature=0.0, # Creatividad baja para ser precisos
            base_url="http://localhost:11434"
        )
    except Exception as e:
        print(f"❌ Error conectando con Ollama. ¿Está corriendo la app?: {e}")
        sys.exit(1)

    # --- D) CREAR EL PROMPT (PERSONALIDAD) ---
    system_prompt = (
        "Eres MauricIA, una asistente experta en los programas de postgrado del Departamento de Ingeniería Informática de la USACH. "
        "Usa los siguientes fragmentos de contexto recuperado para responder la pregunta del usuario. "
        "Si no sabes la respuesta, di que no tienes esa información. No inventes datos. "
        "Responde de manera amable, formal y concisa.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # --- E) ARMAR LA CADENA RAG (CEREBRO + MEMORIA) ---
    try:
        # 1. Cadena que procesa los documentos
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # 2. Cadena que busca y luego responde
        rag_chain = create_retrieval_chain(vector_db.as_retriever(), question_answer_chain)
        print("✅ ¡Sistema MauricIA Local LISTO! 🚀\n")
        return rag_chain
    except Exception as e:
        print(f"❌ Error armando la cadena RAG: {e}")
        sys.exit(1)

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    app_rag = iniciar_mauricia_local()
    
    print("💬 Escribe 'salir' para terminar.")
    print("--------------------------------------------------")

    while True:
        pregunta = input("\n👤 Tú: ")
        if pregunta.lower() in ["salir", "exit", "chau"]:
            print("👋 ¡Hasta luego!")
            break
        
        # Invocar al agente
        print("🤖 Pensando...", end="\r")
        try:
            respuesta = app_rag.invoke({"input": pregunta})
            print(f"🤖 MauricIA: {respuesta['answer']}")
            
            # (Opcional) Ver qué documentos leyó:
            # for i, doc in enumerate(respuesta["context"]):
            #     print(f"   [Fuente {i+1}]: {doc.metadata.get('source', 'Desconocido')}")

        except Exception as e:
            print(f"❌ Error generando respuesta: {e}")