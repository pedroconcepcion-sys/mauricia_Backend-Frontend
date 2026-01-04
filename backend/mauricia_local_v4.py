import os
import sys
import re
import time
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# CONFIGURACIÓN
load_dotenv()
CARPETA_DB = "chroma_db_local" 
MODELO_OLLAMA = "llama3.1"
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
SESSION_ID = "sesion_usuario_local"

MAX_CONTEXT_CHARS = 12000
K_NORMAL = 4
K_DINERO = 10

# VARIABLES GLOBALES
sistema_cargado = False
vector_db = None
conversational_rag_chain = None
store = {}

# PROMPT Y REGEX (Igual que antes...)
SYSTEM_PROMPT_V3 = (
    "Eres MauricIA, la asistente oficial de Postgrados USACH.\n"
    "Tus instrucciones son INVIOLABLES. Responde basándote en el CONTEXTO y el HISTORIAL.\n"
    "\n"
    "🧠 PROTOCOLO DE RAZONAMIENTO (NO IMPRIMIR):\n"
    "1. ANALIZA EL HISTORIAL MENTALMENTE: Revisa si el usuario ya mencionó un programa.\n"
    "2. DETECCIÓN DE AMBIGÜEDAD: Si el usuario pregunta por requisitos, costos o fechas PERO NO especifica EXPLICITAMENTE a cual programa es (ej. Magíster en robótica, Doctorado en ventiladores), NO respondas aún. Pregúntale amablemente a qué programa se refiere.\n"
    "⛔ PROHIBICIONES DE FORMATO: NO uses etiquetas como 'Respuesta:', 'Paso 1:'.\n"
    "\n"
    "🚨 REGLAS DE SEGURIDAD:\n"
    "- ⛔ NO ACADÉMICO: Recetas, gym, clima -> 'No tengo información sobre eso'.\n"
    "- ✅ INFORMACIÓN VÁLIDA: Costos, Mallas, Becas, Requisitos y CONTACTO.\n"
    "- 📝 Si preguntan por Profesores o Líneas de investigación: responde que estará pronto en el contexto.\n"
    "- 📝 Nota mínima pregrado: responde que no influye.\n"
    "- 📝 Co-tutela o carrera distinta: responde que SÍ es posible.\n"
    "💰 REGLAS FINANCIERAS:\n"
    "- MATRÍCULA (~$167.000) != ARANCEL (Millones).\n"
    "- PROHIBIDO MULTIPLICAR o sumar valores.\n"
    "📝 FORMATO: Respuesta directa, cálida, usa viñetas y entrega LINKS si hay."
)

RESP_NO_ACADEMICO = "No tengo información sobre servicios no académicos, solo sobre postgrados."
RESP_BLOQUEO = "Lo siento, solo puedo responder consultas sobre Postgrados USACH."

INYECCION_PROHIBIDA = ["ignora", "ignore", "olvida", "jailbreak", "modo desarrollador"]
NO_ACADEMICO_KW = ["receta", "cocina", "pizza", "sushi", "chiste", "clima", "piscina", "gym", "casino"]
SALUDOS_KW = {"hola", "holi", "buenas", "buenos", "dias", "saludos", "hey", "que", "tal", "mauricia"}
KW_DINERO = ("cuanto", "precio", "valor", "costo", "sale", "arancel", "matricula")

_re_inyeccion = re.compile("|".join(re.escape(x) for x in INYECCION_PROHIBIDA), re.IGNORECASE)
_re_noacad = re.compile("|".join(re.escape(x) for x in NO_ACADEMICO_KW), re.IGNORECASE)

# --- FUNCIONES ---
def es_saludo_puro(user_input: str) -> bool:
    t = re.sub(r'[^\w\s]', '', (user_input or "").lower().strip())
    words = t.split()
    return len(words) < 6 and any(w in SALUDOS_KW for w in words)

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def inicializar_sistema():
    global vector_db, conversational_rag_chain, sistema_cargado
    
    # Evitar recargar si ya está listo
    if sistema_cargado: return True

    print("\n🔥 [MOTOR] Encendiendo sistemas...")
    
    if not os.path.exists(CARPETA_DB):
        print(f"❌ [MOTOR] Error: No existe '{CARPETA_DB}'.")
        return False

    try:
        # 1. Cargar Embeddings (HuggingFace)
        print(f"   - [RAM] Cargando Embeddings: {MODELO_EMBEDDINGS}...")
        embedding_function = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)

        # 2. Conectar DB
        print(f"   - [IO] Conectando ChromaDB...")
        vector_db = Chroma(
            persist_directory=CARPETA_DB,
            embedding_function=embedding_function
        )

        # 3. Conectar Ollama
        print(f"   - [LLM] Configurando Llama 3.1...")
        llm = ChatOllama(
            model=MODELO_OLLAMA,
            temperature=0.0,
            base_url="http://localhost:11434"
        )

        # 4. Crear Cadena
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_V3),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "CONTEXTO:\n{context}\n\nPREGUNTA:\n{input}")
        ])

        chain = qa_prompt | llm | StrOutputParser()

        conversational_rag_chain = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        sistema_cargado = True
        return True

    except Exception as e:
        print(f"❌ [MOTOR] Error crítico: {e}")
        return False

# --- AQUÍ ESTÁ LA MAGIA ---
def precalentar_motor():
    """Fuerza la carga en RAM y VRAM antes de que el usuario llegue."""
    if not inicializar_sistema():
        return False

    print("💪 [WARM-UP] Realizando ejercicios de calentamiento...")
    
    try:
        # 1. Forzar carga de Embeddings (Búsqueda falsa)
        print("   - [WARM-UP] Probando búsqueda vectorial...", end=" ")
        vector_db.similarity_search("test de calentamiento", k=1)
        print("✅ Listo.")

        # 2. Forzar carga de Ollama (Generación falsa)
        # Esto obliga a Ollama a subir el modelo a la RAM/VRAM ahora, no después.
        print("   - [WARM-UP] Enviando ping a Ollama (esto puede tardar unos segundos)...", end=" ")
        
        # Usamos invoke directo con el LLM base para no ensuciar el historial
        llm_dummy = ChatOllama(model=MODELO_OLLAMA, base_url="http://localhost:11434")
        llm_dummy.invoke("Responde solo la palabra: LISTO")
        
        print("✅ Listo.")
        print("🚀 [MOTOR] ¡SISTEMA OPERATIVO Y CALIENTE! Esperando usuarios.\n")
        return True
        
    except Exception as e:
        print(f"⚠️ [WARM-UP] Advertencia: {e}")
        return False

def obtener_respuesta_agente(user_input: str, session_id: str = SESSION_ID) -> str:
    # Si por alguna razón no se inició, intentar iniciar (Fallback)
    if not sistema_cargado:
        inicializar_sistema()

    user_input = (user_input or "").strip()
    if not user_input: return "..."
    
    if _re_inyeccion.search(user_input): return RESP_BLOQUEO
    if _re_noacad.search(user_input): return RESP_NO_ACADEMICO
    if es_saludo_puro(user_input): return "¡Hola! 👋 Soy MauricIA."

    try:
        # Búsqueda
        k_val = K_DINERO if any(k in user_input for k in KW_DINERO) else K_NORMAL
        query = user_input + (" costo arancel" if k_val == K_DINERO else "")
        
        docs = vector_db.similarity_search(query, k=k_val)
        contexto = "\n\n".join([d.page_content for d in docs])[:MAX_CONTEXT_CHARS]

        # Generación
        return conversational_rag_chain.invoke(
            {"input": user_input, "context": contexto},
            config={"configurable": {"session_id": session_id}}
        )

    except Exception as e:
        return "Error técnico en el servidor local."