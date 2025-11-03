from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database URL - Update this with your PostgreSQL credentials
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Association table for User-Course enrollment (Many-to-Many)
enrollment_table = Table(
    'enrollment',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('course_id', Integer, ForeignKey('courses.id'), primary_key=True),
    Column('enrolled_at', DateTime, default=datetime.utcnow)
)

# Models based on your schema
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    type = Column(String, nullable=False)  # "student" or "teacher"
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    enrolled_courses = relationship("Course", secondary=enrollment_table, back_populates="students")
    teaching_courses = relationship("Course", back_populates="teacher")
    chats = relationship("Chat", back_populates="user")

class Course(Base):
    __tablename__ = "courses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    teacher_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    teacher = relationship("User", back_populates="teaching_courses")
    students = relationship("User", secondary=enrollment_table, back_populates="enrolled_courses")
    materials = relationship("Material", back_populates="course")
    chats = relationship("Chat", back_populates="course")

class Material(Base):
    __tablename__ = "materials"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    course_id = Column(Integer, ForeignKey('courses.id'))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    chunks_count = Column(Integer, default=0)
    
    # Relationships
    course = relationship("Course", back_populates="materials")

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    course_id = Column(Integer, ForeignKey('courses.id'))
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    sources = Column(String)  # Store as JSON string
    confidence = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chats")
    course = relationship("Course", back_populates="chats")

# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()