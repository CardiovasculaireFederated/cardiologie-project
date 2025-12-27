import os
import psycopg2

def init_db():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "database_card"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "cardio111")
    )
    cur = conn.cursor()
    # Table 1 : Informations sur les PDFs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS models_info (
        id SERIAL PRIMARY KEY,
        model_name VARCHAR(255),
        weights BYTEA,                     -- POIDS MOYENNÉS
        global_accuracy FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print(" Tables créées avec succès !")

if __name__ == "__main__":
    init_db()
