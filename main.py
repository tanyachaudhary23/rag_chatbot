# main.py - Main Application with Material Delete Function
import tempfile
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

load_dotenv()

# Import database models and auth
from database import User, Course, Material, Chat, get_db, init_db
from auth import (
    get_password_hash, verify_password, create_access_token, 
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)

app = FastAPI(title="Multi-Course RAG Educational Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG components
embedding_model = None
chroma_client = None
openai_client = None

# Pydantic models for API
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    type: str  # "student" or "teacher"

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    type: str
    
    class Config:
        from_attributes = True

class CourseCreate(BaseModel):
    name: str
    description: Optional[str] = None

class CourseResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    teacher_id: int
    
    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    question: str
    course_id: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class Token(BaseModel):
    access_token: str
    token_type: str

class MaterialResponse(BaseModel):
    id: int
    name: str
    course_id: int
    chunks_count: int
    uploaded_at: datetime
    
    class Config:
        from_attributes = True

# Initialize system
@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client, openai_client
    
    print("🚀 Initializing Multi-Course RAG system...")
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Initialize embedding model
    print("📊 Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB
    print("🗄 Setting up ChromaDB...")
    chroma_client = chromadb.Client()
    
    # Initialize OpenAI
    if os.getenv("OPENAI_API_KEY"):
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✅ OpenAI client initialized")
    else:
        print("⚠ No OpenAI API key found")
    
    print("✅ Multi-Course RAG system initialized!")

# Helper functions for ChromaDB collections
def get_or_create_collection(course_id: int):
    """Get or create a ChromaDB collection for a specific course"""
    collection_name = f"course_{course_id}_materials"
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"course_id": course_id}
        )
    return collection

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """Extract text from PDF"""
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def store_document_chunks(chunks: List[str], material_id: int, collection):
    """Store document chunks in course-specific collection"""
    if not chunks:
        return
    
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"material_{material_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"material_id": material_id, "chunk_id": i} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

def delete_material_chunks(material_id: int, collection):
    """Delete all chunks associated with a material from the collection"""
    try:
        # Get all chunk IDs for this material
        results = collection.get(
            where={"material_id": material_id}
        )
        
        if results and results['ids']:
            # Delete all chunks
            collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0
    except Exception as e:
        print(f"Error deleting chunks: {e}")
        return 0

def retrieve_relevant_chunks(query: str, collection, n_results: int = 3):
    """Retrieve relevant chunks from course collection"""
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    distances = results['distances'][0] if results['distances'] else []
    
    return documents, metadatas, distances

def generate_response(query: str, context_chunks: List[str], course_name: str) -> str:
    """Generate response using OpenAI or fallback"""
    context = "\n\n".join(context_chunks)
    
    if openai_client:
        try:
            prompt = f"""You are an educational assistant for the course "{course_name}". 
Answer the student's question based only on the provided course material.

Context from course materials:
{context}

Student's question: {query}

Instructions:
- Answer based only on the provided context
- If the answer isn't in the context, say so clearly
- Be educational, clear, and helpful
- Keep responses concise but complete

Answer:"""

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
    
    # Fallback response
    if context_chunks:
        return f"""Based on the course materials for {course_name}:

{context_chunks[0][:400]}{'...' if len(context_chunks[0]) > 400 else ''}

This information comes from the uploaded course materials."""
    return "I couldn't find relevant information in the course materials."

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate user type
    if user.type not in ["student", "teacher"]:
        raise HTTPException(status_code=400, detail="Type must be 'student' or 'teacher'")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        name=user.name,
        email=user.email,
        password_hash=hashed_password,
        type=user.type
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/api/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# ==================== COURSE ENDPOINTS ====================

@app.post("/api/courses", response_model=CourseResponse)
def create_course(
    course: CourseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new course (teachers only)"""
    if current_user.type != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can create courses")
    
    db_course = Course(
        name=course.name,
        description=course.description,
        teacher_id=current_user.id
    )
    db.add(db_course)
    db.commit()
    db.refresh(db_course)
    
    return db_course

@app.get("/api/courses", response_model=List[CourseResponse])
def get_my_courses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get courses for current user (enrolled if student, teaching if teacher)"""
    # Re-query the user in this session to avoid DetachedInstanceError
    user = db.query(User).filter(User.id == current_user.id).first()
    
    if user.type == "student":
        return user.enrolled_courses
    else:
        return user.teaching_courses

@app.post("/api/courses/{course_id}/enroll")
def enroll_in_course(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enroll in a course (students only)"""
    if current_user.type != "student":
        raise HTTPException(status_code=403, detail="Only students can enroll in courses")
    
    # Re-query the user in this session to avoid DetachedInstanceError
    user = db.query(User).filter(User.id == current_user.id).first()
    
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course in user.enrolled_courses:
        raise HTTPException(status_code=400, detail="Already enrolled in this course")
    
    user.enrolled_courses.append(course)
    db.commit()
    
    return {"message": "Successfully enrolled in course"}

@app.get("/api/courses/{course_id}/materials", response_model=List[MaterialResponse])
def get_course_materials(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all materials for a course"""
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Re-query the user in this session to avoid DetachedInstanceError
    # Check if user has access to course
    if current_user.type == "student":
        user = db.query(User).filter(User.id == current_user.id).first()
        if course not in user.enrolled_courses:
            raise HTTPException(status_code=403, detail="Not enrolled in this course")
    elif current_user.type == "teacher" and course.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your course")
    
    materials = db.query(Material).filter(Material.course_id == course_id).all()
    return materials

# ==================== MATERIAL UPLOAD ENDPOINTS ====================

@app.post("/api/courses/{course_id}/upload")
def upload_material(
    course_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload course material (teachers only)"""
    # Verify course exists and user is the teacher
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if current_user.type != "teacher" or course.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the course teacher can upload materials")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        content = file.file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Extract and process text
        text = extract_text_from_pdf(temp_file_path)
        chunks = chunk_text(text)
        
        # Create material record
        material = Material(
            name=file.filename,
            file_path=temp_file_path,
            course_id=course_id,
            chunks_count=len(chunks)
        )
        db.add(material)
        db.commit()
        db.refresh(material)
        
        # Store in course-specific collection
        collection = get_or_create_collection(course_id)
        store_document_chunks(chunks, material.id, collection)
        
        return {
            "message": "Material uploaded successfully",
            "material_id": material.id,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.delete("/api/materials/{material_id}")
def delete_material(
    material_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a material (teachers only)"""
    # Get the material
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    # Get the course
    course = db.query(Course).filter(Course.id == material.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Check if user is the teacher of this course
    if current_user.type != "teacher" or course.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the course teacher can delete materials")
    
    try:
        # Delete chunks from ChromaDB
        collection = get_or_create_collection(material.course_id)
        chunks_deleted = delete_material_chunks(material_id, collection)
        
        # Delete the physical file
        if os.path.exists(material.file_path):
            os.unlink(material.file_path)
        
        # Delete from database
        db.delete(material)
        db.commit()
        
        return {
            "message": "Material deleted successfully",
            "material_id": material_id,
            "chunks_deleted": chunks_deleted
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting material: {str(e)}")

# ==================== CHAT ENDPOINTS ====================

@app.post("/api/courses/{course_id}/chat", response_model=QueryResponse)
def chat_with_course(
    course_id: int,
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ask a question about course materials"""
    # Verify course exists and user has access
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Re-query the user in this session to avoid DetachedInstanceError
    if current_user.type == "student":
        user = db.query(User).filter(User.id == current_user.id).first()
        if course not in user.enrolled_courses:
            raise HTTPException(status_code=403, detail="Not enrolled in this course")
    
    # Get course collection
    collection = get_or_create_collection(course_id)
    
    # Retrieve relevant chunks
    documents, metadatas, distances = retrieve_relevant_chunks(request.question, collection)
    
    if not documents:
        answer = "No course materials have been uploaded yet for this course."
        sources = []
        confidence = 0.0
    else:
        # Generate response
        answer = generate_response(request.question, documents, course.name)
        
        # Get source materials
        material_ids = [meta['material_id'] for meta in metadatas]
        materials = db.query(Material).filter(Material.id.in_(material_ids)).all()
        sources = [mat.name for mat in materials]
        
        # Calculate confidence
        avg_distance = sum(distances) / len(distances)
        confidence = max(0.0, 1.0 - avg_distance)
    
    # Save chat history
    chat = Chat(
        user_id=current_user.id,
        course_id=course_id,
        question=request.question,
        answer=answer,
        sources=json.dumps(sources),
        confidence=str(confidence)
    )
    db.add(chat)
    db.commit()
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )

@app.get("/api/courses/{course_id}/chat/history")
def get_chat_history(
    course_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chat history for a course"""
    chats = db.query(Chat).filter(
        Chat.course_id == course_id,
        Chat.user_id == current_user.id
    ).order_by(Chat.created_at.desc()).limit(50).all()
    
    return [
        {
            "id": chat.id,
            "question": chat.question,
            "answer": chat.answer,
            "sources": json.loads(chat.sources),
            "confidence": chat.confidence,
            "created_at": chat.created_at.isoformat()
        }
        for chat in chats
    ]

# ==================== ADMIN ENDPOINTS ====================

@app.get("/api/admin/stats")
def get_platform_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get platform statistics"""
    total_users = db.query(User).count()
    total_courses = db.query(Course).count()
    total_materials = db.query(Material).count()
    total_chats = db.query(Chat).count()
    
    return {
        "total_users": total_users,
        "total_courses": total_courses,
        "total_materials": total_materials,
        "total_chats": total_chats
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model_loaded": embedding_model is not None,
        "openai_configured": openai_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)