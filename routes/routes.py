from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from handler.handler import YonguiHandler
from dao.dao import YonguiDAO

router = APIRouter()
handler = YonguiHandler()
dao = YonguiDAO()

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    context: List[str]

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Get relevant context using vector similarity
        prompt_vec = handler.embed_text(request.question)
        
        # Convert the embedding to a string representation for PostgreSQL
        vec_str = str(prompt_vec).replace('[', '[').replace(']', ']')
        
        try:
            results = dao.get_similar_chunks(vec_str, request.top_k)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        if not results:
            raise HTTPException(status_code=404, detail="No relevant context found")
        
        # Extract context from results
        context_chunks = [row[0] for row in results]
        
        try:
            # Generate response using Ollama
            response = handler.generate_response(request.question, context_chunks)
            
            return ChatResponse(
                answer=response,
                context=context_chunks
            )
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating response: {str(e)}"
            )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
