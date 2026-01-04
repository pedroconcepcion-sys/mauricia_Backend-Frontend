import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma

# Cargar variables (.env)
load_dotenv()

# --- CONFIGURACIÓN ---
CARPETA_DATA = "data"
RUTA_DB = "chroma_db_prod" 

URLS_POR_PROGRAMA = {
    "doctorado": "https://www.postgradosudesantiago.cl/wp-content/uploads/2023/Malla_Doctorado_Informatica.pdf",
    "magister": "https://www.postgradosudesantiago.cl/wp-content/uploads/2023/Malla_Magister_Informatica.pdf",
    "diplomado": "https://www.postgradosudesantiago.cl/diplomados"
}

def limpiar_texto_maestro(texto):
    patron_basura = r'(?i)(events=%5B%5B%22pageview|tw_document_href=|integration=advertiser|p_id=Twitter).*'
    texto = re.sub(patron_basura, '', texto)
    patron_precio = r'(?i)(.*(?:arancel|matr[ií]cula|costo|valor).*?)\n\s*(\$\s*.*)'
    texto = re.sub(patron_precio, r'\1: \2', texto)
    return texto

def main():
    if not os.path.exists(CARPETA_DATA):
        print(f"❌ Error: La carpeta '{CARPETA_DATA}' no existe.")
        return

    print(f"📚 Escaneando carpeta: {CARPETA_DATA}...")
    documentos_totales = []

    # --- A) CARGAR ARCHIVOS MARKDOWN ---
    try:
        loader_md = DirectoryLoader(CARPETA_DATA, glob="*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        docs_md = loader_md.load()
        for doc in docs_md:
            doc.page_content = limpiar_texto_maestro(doc.page_content)
            doc.page_content = f"CONTEXTO WEB USACH (CRAWLER):\n{doc.page_content}"
        documentos_totales.extend(docs_md)
    except Exception as e:
        print(f"  ⚠️ Alerta MDs: {e}")

    # --- B) CARGAR ARCHIVOS PDF ---
    try:
        loader_pdf = DirectoryLoader(CARPETA_DATA, glob="*.pdf", loader_cls=PyPDFLoader)
        docs_pdf = loader_pdf.load()
        for doc in docs_pdf:
            nombre_archivo = doc.metadata.get('source', '').lower()
            etiqueta_programa = "PROGRAMA DESCONOCIDO"
            url_descarga = "No disponible"

            if "doctorado" in nombre_archivo:
                etiqueta_programa = "DOCTORADO EN CIENCIAS DE LA INGENIERÍA, MENCIÓN INFORMÁTICA"
                url_descarga = URLS_POR_PROGRAMA["doctorado"]
            elif "magister" in nombre_archivo or "magíster" in nombre_archivo:
                etiqueta_programa = "MAGÍSTER EN INGENIERÍA INFORMÁTICA"
                url_descarga = URLS_POR_PROGRAMA["magister"]
            
            doc.page_content = limpiar_texto_maestro(doc.page_content)
            header = (
                f"DOCUMENTO OFICIAL/MALLA (PDF) DEL: {etiqueta_programa}.\n"
                f"📥 PUEDES DESCARGAR EL PDF AQUÍ: {url_descarga}\n"
                f"--------------------------------------------------\n"
            )
            doc.page_content = header + doc.page_content
        documentos_totales.extend(docs_pdf)
    except Exception as e:
        print(f"  ⚠️ Alerta PDFs: {e}")

    # --- C) CHUNKING ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", "####", " ", ""]
    )
    chunks = text_splitter.split_documents(documentos_totales)

    # --- D) GUARDAR EN CHROMA (CON OPENAI / AZURE) ---
    print("🧠 Generando cerebro vectorial con OpenAI via Azure/GitHub...")
    
    # El base_url va aquí adentro del modelo de embeddings
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com"
    )

    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)
        print("   (Base de datos anterior eliminada)")

    # Ahora le pasamos el objeto 'embedding_model' ya configurado
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=RUTA_DB
    )
    
    print(f"✅ ¡Cerebro de NUBE actualizado! {len(chunks)} fragmentos guardados en {RUTA_DB}.")

if __name__ == "__main__":
    main()