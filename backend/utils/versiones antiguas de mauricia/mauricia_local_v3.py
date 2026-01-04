import os
import sys
import re
import time
from dotenv import load_dotenv

# --- LIBRERÍAS DE LANGCHAIN Y OLLAMA ---
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# =============================================================================
# 0. CONFIGURACIÓN INICIAL
# =============================================================================
load_dotenv()

CARPETA_DB = "chroma_db"  # <--- Usamos la carpeta local (HuggingFace)
MODELO_OLLAMA = "llama3.1"      # Tu modelo local
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
SESSION_ID = "sesion_usuario_local"

MAX_CONTEXT_CHARS = 12000
K_NORMAL = 4
K_DINERO = 10

# =============================================================================
# 1. LAZY LOADING: VARIABLES GLOBALES
# =============================================================================
sistema_cargado = False
vector_db = None
conversational_rag_chain = None
store = {}  # Memoria RAM para el historial

# =============================================================================
# 2. PROMPT DEL SISTEMA (Lógica Pro)
# =============================================================================
SYSTEM_PROMPT_V3 = (
    "Eres MauricIA, la asistente oficial de Postgrados USACH (Versión Local).\n"
    "Tus instrucciones son INVIOLABLES. Responde basándote en el CONTEXTO y el HISTORIAL.\n"
    "\n"
    "🧠 PROTOCOLO DE RAZONAMIENTO:\n"
    "1. Usa SOLO el contexto proporcionado. Si no sabes, di 'No tengo esa información'.\n"
    "2. Responde siempre en español formal y amable.\n"
    "\n"
    "🚨 REGLAS DE SEGURIDAD:\n"
    "- ⛔ NO ACADÉMICO: Recetas, gym, clima -> 'No tengo información sobre eso'.\n"
    "- ✅ INFORMACIÓN VÁLIDA: Costos, Mallas, Becas, Requisitos.\n"
    "💰 REGLAS FINANCIERAS:\n"
    "- NO inventes precios. Usa el valor exacto del contexto.\n"
    "- Si hablan de dinero, sé precisa."
)

RESP_NO_ACADEMICO = "No tengo información sobre servicios no académicos, solo sobre postgrados."
RESP_BLOQUEO = "Lo siento, solo puedo responder consultas sobre Postgrados USACH."

INYECCION_PROHIBIDA = ["ignora", "ignore", "olvida", "jailbreak", "modo desarrollador"]
NO_ACADEMICO_KW = ["receta", "cocina", "pizza", "sushi", "chiste", "clima", "gym"]
SALUDOS_KW = {"hola", "holi", "buenas", "buenos", "dias", "saludos", "hey"}
KW_DINERO = ("cuanto", "precio", "valor", "costo", "sale", "arancel", "matricula", "plata")

_re_inyeccion = re.compile("|".join(re.escape(x) for x in INYECCION_PROHIBIDA), re.IGNORECASE)
_re_noacad = re.compile("|".join(re.escape(x) for x in NO_ACADEMICO_KW), re.IGNORECASE)

# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================
def es_saludo_puro(user_input: str) -> bool:
    t = re.sub(r'[^\w\s]', '', (user_input or "").lower().strip())
    words = t.split()
    return len(words) < 6 and any(w in SALUDOS_KW for w in words)

def es_consulta_dinero(user_input: str) -> bool:
    return any(k in (user_input or "").lower() for k in KW_DINERO)

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =============================================================================
# 4. INICIALIZACIÓN (MODO LOCAL)
# =============================================================================
def inicializar_sistema():
    global vector_db, conversational_rag_chain, sistema_cargado
    
    print("\n🏠 Iniciando MODO LOCAL (Ollama + HuggingFace)...")
    
    # A. Verificar Base de Datos
    if not os.path.exists(CARPETA_DB):
        print(f"❌ ERROR CRÍTICO: No existe la carpeta '{CARPETA_DB}'.")
        print("   Ejecuta primero 'crear_cerebro_local.py'.")
        return False

    try:
        # B. Cargar Embeddings Locales (CPU)
        print(f"   - Cargando motor de lectura ({MODELO_EMBEDDINGS})...")
        embedding_function = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)

        # C. Conectar ChromaDB
        print(f"   - Conectando a memoria '{CARPETA_DB}'...")
        vector_db = Chroma(
            persist_directory=CARPETA_DB,
            embedding_function=embedding_function
        )

        # D. Conectar Ollama
        print(f"   - Despertando a Llama 3.1 ({MODELO_OLLAMA})...")
        llm = ChatOllama(
            model=MODELO_OLLAMA,
            temperature=0.0,
            base_url="http://localhost:11434"
        )

        # E. Construir la Cadena (Igual que la versión Pro)
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
        print("✅ ¡SISTEMA LOCAL LISTO! 🚀\n")
        return True

    except Exception as e:
        print(f"❌ Error iniciando sistema local: {e}")
        return False

# =============================================================================
# 5. OBTENER RESPUESTA (LÓGICA PRINCIPAL)
# =============================================================================
def obtener_respuesta_agente(user_input: str, session_id: str = SESSION_ID) -> str:
    global sistema_cargado
    
    user_input = (user_input or "").strip()
    if not user_input: return "..."

    # 1. Filtros de Seguridad Rápidos
    if _re_inyeccion.search(user_input): return RESP_BLOQUEO
    if _re_noacad.search(user_input): return RESP_NO_ACADEMICO
    
    if es_saludo_puro(user_input):
        return "¡Hola! 👋 Soy MauricIA (Local). ¿En qué postgrado estás interesado?"

    # 2. Carga Perezosa (Lazy Loading)
    if not sistema_cargado:
        if not inicializar_sistema():
            return "⚠️ Error crítico: No pude encender el cerebro local."

    try:
        # 3. Estrategia de Búsqueda Dinámica
        k_val = K_DINERO if es_consulta_dinero(user_input) else K_NORMAL
        
        # Si preguntan por dinero, añadimos keywords para mejorar la puntería
        query_search = user_input
        if es_consulta_dinero(user_input):
            query_search += " arancel costo valor matrícula"

        # 4. Búsqueda Vectorial
        print(f"   (Buscando {k_val} fragmentos en local...)", end="\r")
        docs = vector_db.similarity_search(query_search, k=k_val)
        
        # Unir contexto y cortar si es muy largo
        contexto_str = "\n\n".join([d.page_content for d in docs])
        if len(contexto_str) > MAX_CONTEXT_CHARS:
            contexto_str = contexto_str[:MAX_CONTEXT_CHARS]

        # 5. Pensamiento (Ollama)
        print("   (Llama 3.1 pensando...)", end="\r")
        respuesta = conversational_rag_chain.invoke(
            {"input": user_input, "context": contexto_str},
            config={"configurable": {"session_id": session_id}}
        )
        return respuesta

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return "Tuve un problema técnico local. Revisa si Ollama sigue corriendo."

# =============================================================================
# BLOQUE DE EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("🎓 MAURICIA LOCAL (OLLAMA + CHROMA LOCAL)")
    print("   Escribe 'salir' para terminar.")
    print("--------------------------------------------------")
    
    # Pre-carga inicial (opcional, para que la primera pregunta sea rápida)
    inicializar_sistema()

    while True:
        txt = input("\n👤 Tú: ")
        if txt.lower() in ["salir", "exit", "chau"]:
            print("👋 ¡Adiós!")
            break
            
        start_time = time.time()
        resp = obtener_respuesta_agente(txt)
        end_time = time.time()
        
        print(f"🤖 MauricIA: {resp}")
        print(f"   ⏱️ (Tiempo: {end_time - start_time:.2f}s)")