import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#
# print("--- Dostępne modele dla Twojego klucza ---")
# for m in genai.list_models():
#     if 'embedContent' in m.supported_generation_methods:
#         print(f"MODEL DO EMBEDDINGÓW: {m.name}")
#     if 'generateContent' in m.supported_generation_methods:
#         print(f"MODEL CZATOWY (LLM): {m.name}")

class MedicalAdvisor:
    def __init__(self, db_path="../data/vector_db", docs_path="../data/knowledge_base"):
        self.db_path = db_path
        self.docs_path = docs_path
        # embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # AI studio model llm
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self):
        # Jeśli baza już istnieje na dysku, załaduj ją
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

        # Jeśli nie, wczytaj PDFy i stwórz nową bazę
        loader = DirectoryLoader(self.docs_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        return vectorstore

    def get_interpretation(self, probability, user_data):

        prob_value = round(probability * 100, 2)

        # Słownik definiujący jednostki dla każdego klucza
        units_map = {
            "Age": "Wiek (lata)",
            "BMI": "BMI (kg/m²)",
            "Chol": "Cholesterol całkowity (mmol/L)",
            "TG": "Triglicerydy (mmol/L)",
            "HDL": "Cholesterol HDL ('dobry') (mmol/L)",
            "LDL": "Cholesterol LDL ('dobry') (mmol/L)",
            "Cr": "Kreatynina (µmol/L)",
            "BUN": "Mocznik/BUN (mmol/L)"
        }

        # Tworzymy nowy słownik z opisami
        formatted_data = {}
        for key, value in user_data.items():
            unit = units_map.get(key, "")  # pobierz jednostkę, jeśli istnieje
            formatted_data[key] = f"{value} {unit}".strip()

        # Teraz zamieniamy na JSON do promptu
        user_data_str = json.dumps(formatted_data, indent=2, ensure_ascii=False)
        #user_data_str = json.dumps(user_data, indent=2)
        print(user_data_str)

        # 1. Retrieval
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(
            "cukrzyca diagnostyka interpretacja wyników BMI Age/Wiek Cholesterol Triglicerydy Cholesterol HDL Cholesterol LDL Kreatynina Mocznik ryzyko"
        )
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Prompt
        prompt_template = """
        Jesteś profesjonalnym asystentem medycznym wspierającym diagnostykę cukrzycy. 

        DANE PACJENTA:
        - Prawdopodobieństwo: {prob}%
        - Parametry:
        {user_data}

        KONTEKST:
        {context}

        INSTRUKCJA:
        1. Oceń ryzyko. Zastosuj ścisłą klasyfikację: 
           - poniżej 33%: "niskie", 
           - 33% - 66%: "umiarkowane", 
           - powyżej 66%: "wysokie".
        2. Dokonaj analizy parametrów pacjenta w oparciu o dostarczony KONTEKST. Wyjaśnij, które wyniki są niepokojące i dlaczego.
        3. Używaj prostego, empatycznego języka, unikaj nadmiernego żargonu medycznego.
        4. Odpowiedź sformatuj czytelnie przy użyciu list punktowych i pogrubień (Markdown), podsumowanie ma być widoczne i wyróżniać się za tle całej interpretacji.
        5. Na końcu dodaj wyraźny disclaimer medyczny.

        ODPOWIEDŹ:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "prob", "user_data"]
        )

        # 3. LCEL chain (NOWY STANDARD)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "prob": prob_value,
            "user_data": user_data_str
        })

        return response

# TESTY
if __name__ == "__main__":
    # 1. Inicjalizacja doradcy
    # Upewnij się, że w folderze ../data/knowledge_base masz przynajmniej jeden plik PDF!
    try:
        print("--- Inicjalizacja bazy wiedzy... ---")
        my_advisor = MedicalAdvisor()

        print("--- Generowanie interpretacji (to może chwilę potrwać)... ---")

        # 3. Wywołanie głównej funkcji
        result = my_advisor.get_interpretation(
            probability=0.72,
            user_data={
            "Glucose": 135,
            "BMI": 28.5,
            "Age": 45,
            "BloodPressure": 85
        }
        )

        print("\n=== ODPOWIEDŹ SYSTEMU RAG ===\n")
        print(result)
        print("\n==============================")

    except Exception as e:
        print(f"Wystąpił błąd podczas testu: {e}")