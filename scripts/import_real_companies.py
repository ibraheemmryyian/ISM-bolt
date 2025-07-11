import os
import json
import psycopg2
from psycopg2.extras import execute_values

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'symbioflows')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', 'password')

COMPANY_FILE = 'data/50_gulf_companies_fixed.json'

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def company_exists(cur, name, location):
    cur.execute("SELECT id FROM companies WHERE name=%s AND location=%s", (name, location))
    return cur.fetchone() is not None

def insert_company(cur, company):
    cur.execute(
        """
        INSERT INTO companies (name, industry, location, employee_count, sustainability_score, carbon_footprint, onboarding_completed)
        VALUES (%s, %s, %s, %s, %s, %s, TRUE)
        RETURNING id
        """,
        (
            company['name'],
            company.get('industry', ''),
            company.get('location', ''),
            company.get('employee_count', 0),
            company.get('sustainability_score', 0.0),
            company.get('carbon_footprint', 0.0)
        )
    )
    return cur.fetchone()[0]

def insert_materials(cur, company_id, materials, type_hint=None):
    if not materials:
        return
    for m in materials:
        cur.execute(
            """
            INSERT INTO materials (company_id, name, type, ai_generated)
            VALUES (%s, %s, %s, TRUE)
            ON CONFLICT DO NOTHING
            """,
            (company_id, m, type_hint or 'material')
        )

def insert_products(cur, company_id, products):
    if not products:
        return
    for p in products:
        cur.execute(
            """
            INSERT INTO materials (company_id, name, type, ai_generated)
            VALUES (%s, %s, 'product', TRUE)
            ON CONFLICT DO NOTHING
            """,
            (company_id, p)
        )

def insert_waste_streams(cur, company_id, wastes):
    if not wastes:
        return
    for w in wastes:
        cur.execute(
            """
            INSERT INTO materials (company_id, name, type, ai_generated)
            VALUES (%s, %s, 'waste', TRUE)
            ON CONFLICT DO NOTHING
            """,
            (company_id, w)
        )

def insert_energy_needs(cur, company_id, energy_needs):
    if not energy_needs:
        return
    for e in energy_needs:
        cur.execute(
            """
            INSERT INTO requirements (company_id, name, category, ai_generated)
            VALUES (%s, %s, 'energy', TRUE)
            ON CONFLICT DO NOTHING
            """,
            (company_id, e)
        )

def insert_matching_preferences(cur, company_id, prefs):
    if not prefs:
        return
    for k, v in prefs.items():
        cur.execute(
            """
            INSERT INTO ai_insights (company_id, insight_type, title, confidence_score, status)
            VALUES (%s, 'matching_preference', %s, %s, 'active')
            ON CONFLICT DO NOTHING
            """,
            (company_id, k, v)
        )

def main():
    with open(COMPANY_FILE, 'r', encoding='utf-8') as f:
        companies = json.load(f)
    conn = get_db_connection()
    cur = conn.cursor()
    imported = 0
    skipped = 0
    for company in companies:
        name = company['name']
        location = company.get('location', '')
        if company_exists(cur, name, location):
            print(f"Skipping existing: {name} ({location})")
            skipped += 1
            continue
        company_id = insert_company(cur, company)
        insert_materials(cur, company_id, company.get('materials', []), 'material')
        insert_products(cur, company_id, company.get('products', []))
        insert_waste_streams(cur, company_id, company.get('waste_streams', []))
        insert_energy_needs(cur, company_id, company.get('energy_needs', []))
        insert_matching_preferences(cur, company_id, company.get('matching_preferences', {}))
        print(f"Imported: {name} ({location})")
        imported += 1
    conn.commit()
    cur.close()
    conn.close()
    print(f"Import complete. Imported: {imported}, Skipped: {skipped}")

if __name__ == '__main__':
    main() 