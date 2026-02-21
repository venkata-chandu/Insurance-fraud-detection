import sqlite3

def init_db():
    conn = sqlite3.connect("claims.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        months_as_customer INTEGER,
        total_claim_amount REAL,
        policy_deductable REAL,
        policy_annual_premium REAL,
        capital_gains REAL,
        capital_loss REAL,
        bodily_injuries INTEGER,
        witnesses INTEGER,
        number_of_vehicles_involved INTEGER,
        incident_hour INTEGER,
        auto_year INTEGER,
        probability REAL,
        prediction INTEGER,
        risk_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()