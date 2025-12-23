import psycopg2

def init_db():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="database_card",
        user="admin",
        password="cardio111"
    )
    cur = conn.cursor()
    # Table 1 : Informations sur les PDFs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pdf_info (
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
