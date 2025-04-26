# database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres.xfvqaitclnpmzaiimdst: jejbeh-toqzoR-2xaxwi @aws-0-eu-west-2.pooler.supabase.com:6543/postgres"

engine = create_engine(DATABASE_URL, connect_args={}, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
