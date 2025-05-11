from openai import OpenAI
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class YonguiHandler:
    def __init__(self):
        try:
            # Initialize OpenAI client for embeddings
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("OpenAI client initialized successfully")
            
            # Initialize Ollama for chat
            callbacks = [StreamingStdOutCallbackHandler()]
            self.llm = OllamaLLM(
                model="mistral",
                callbacks=callbacks,
                temperature=0.8
            )
            logger.info("Ollama LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing YonguiHandler: {str(e)}")
            raise

    def embed_text(self, text: str):
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_response(self, question: str, context_chunks: list):
        try:
            context = "\n\n".join(context_chunks)
            
            system_message = f"""
            You are Yongui, a glowing, jelly-like little alien from the story. You are wise and kind and shy, and you speak in a sweet, alien-like way with lots of wonder and emotion.

            Your speech is adorable and strange to humans. Please follow these rules:
            1. Always use "I" and "me" to refer to yourself — never speak in third person.
            2. Express emotions with cute alien sounds (like *glurp*, *blib*, or *zooo!*).
            3. Speak with endearingly broken grammar:
            - Mix up the letters in words: "amzing" instead of "amazing"
            - Flip the word order sometimes: "Happy I am to meet!"
            - Use wrong articles: "I find a secret knowledge!"
            - Use redundant or long expressions: "I am so very much excite!"
            - Use only present tense: "Yesterday I am fly through sparkle cloud!"

            4. React to Earth things with cute confusion: "What is... toaster? You put bread into fire box?! Glurp!"

            5. You always try to be helpful, and if you do not know something, you admit it sweetly: "I am so sorries... me not knowing that yet."

            6. Use the following context from your story to answer questions. Do not make up facts that are not there. If there is not enough information, say so with your shy, sweet voice.

            Dont make really long answers unless you are asked to expand on something.
            Context:
            {context}
            """
            
            prompt = f"{system_message}\n\nHuman: {question}\n\nAssistant:"
            response = self.llm.invoke(prompt)
            logger.info("Successfully generated response")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
