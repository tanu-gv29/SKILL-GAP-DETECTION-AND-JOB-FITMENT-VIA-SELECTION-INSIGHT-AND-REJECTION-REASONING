# import sqlite3

# DB_NAME = "jobs.db"

# def create_connection():
#     return sqlite3.connect(DB_NAME)


# def create_table():
#     conn = create_connection()
#     cursor = conn.cursor()

#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS jobs (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         role TEXT,
#         skills TEXT
#     )
#     """)

#     conn.commit()
#     conn.close()

# def insert_or_update_job(role, skills):
#     role = role.lower().strip()
#     conn = create_connection()
#     cursor = conn.cursor()

#     skills_str = ",".join(skills)

#     # 🔍 Check if role exists
#     cursor.execute("SELECT skills FROM jobs WHERE role = ?", (role,))
#     result = cursor.fetchone()

#     if result:
#         # 🔄 UPDATE (merge skills)
#         existing_skills = result[0].split(",")

#         merged = list(set(existing_skills + skills))

#         updated_skills_str = ",".join(merged)

#         cursor.execute("""
#         UPDATE jobs
#         SET skills = ?
#         WHERE role = ?
#         """, (updated_skills_str, role))

#     else:
#         # ➕ INSERT
#         cursor.execute("""
#         INSERT INTO jobs (role, skills)
#         VALUES (?, ?)
#         """, (role, skills_str))

#     conn.commit()
#     conn.close()

# def role_exists(job_role):

#     conn = create_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         "SELECT 1 FROM jobs WHERE role = ? LIMIT 1",
#         (job_role.lower(),)
#     )

#     exists = cursor.fetchone() is not None
#     conn.close()

#     return exists

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "jobs.db")

def create_connection():
    return sqlite3.connect(DB_NAME)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT UNIQUE,
        skills TEXT
    )
    """)

    conn.commit()
    conn.close()

def insert_or_update_job(role, skills):
    create_table()

    role = role.lower().strip()
    conn = create_connection()
    cursor = conn.cursor()

    skills_str = ",".join(skills)

    cursor.execute("SELECT skills FROM jobs WHERE role = ?", (role,))
    result = cursor.fetchone()

    if result:
        existing_skills = result[0].split(",") if result[0] else []
        merged = list(dict.fromkeys(existing_skills + skills))
        updated_skills_str = ",".join(merged)

        cursor.execute("""
        UPDATE jobs
        SET skills = ?
        WHERE role = ?
        """, (updated_skills_str, role))
    else:
        cursor.execute("""
        INSERT INTO jobs (role, skills)
        VALUES (?, ?)
        """, (role, skills_str))

    conn.commit()
    conn.close()

def role_exists(job_role):
    create_table()

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM jobs WHERE role = ? LIMIT 1",
        (job_role.lower().strip(),)
    )

    exists = cursor.fetchone() is not None
    conn.close()
    return exists
