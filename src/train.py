import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from utils.logger import logger

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from unified root ENV
root_dir = Path(__file__).resolve().parents[3]
env_path = root_dir / "ENV" / ".env"
load_dotenv(dotenv_path=env_path)

# Load credentials from env
PG_HOST = os.environ.get("PG_HOST")
PG_PORT = int(os.environ.get("PG_PORT", 5432)) if os.environ.get("PG_PORT") else None
PG_USER = os.environ.get("PG_USER")
PG_PASSWORD = os.environ.get("PG_PASSWORD")
PG_DATABASE = os.environ.get("PG_DATABASE")
PG_SSL = os.environ.get("PG_SSL", "false").lower() == "true"

def connect_postgres():
    kwargs = {
        "host": PG_HOST,
        "port": PG_PORT,
        "user": PG_USER,
        "password": PG_PASSWORD,
        "dbname": PG_DATABASE
    }
    if PG_SSL:
        kwargs["sslmode"] = "require"
    return psycopg2.connect(**kwargs)

def train_correlation_model():
    logger.info("Starting correlation model training...")
    conn = None
    try:
        conn = connect_postgres()
        
        # 1. Fetch user cognitive states
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_id, node_id, expected_mastery 
                FROM user_cognitive_states;
                """
            )
            records = cur.fetchall()

        if len(records) < 5:
            # Seed mock data for training in case database is empty/small for demo
            logger.info("Database too small. Injecting mock records for training...")
            mock_records = []
            users = [f"user_{i}" for i in range(10)]
            concepts = [
                'CS_PROG_SYNTAX', 'CS_PROG_VARIABLES', 'CS_PROG_CONDITIONALS', 'CS_PROG_LOOPS',
                'CS_DS_ARRAYS', 'CS_DS_LINKED_LISTS', 'CS_DS_STACKS_QUEUES', 'CS_DS_TREES'
            ]
            for u in users:
                base = np.random.rand() * 0.4 + 0.3 # Base ability
                for idx, c in enumerate(concepts):
                    # Prereq mastery affects next nodes
                    val = base + np.random.randn() * 0.05 - (idx * 0.03)
                    val = max(0.1, min(0.99, val))
                    mock_records.append({"user_id": u, "node_id": c, "expected_mastery": val})
            records = mock_records

        # 2. Reshape into Student-Concept Matrix
        user_list = list(set([r["user_id"] for r in records]))
        concept_list = list(set([r["node_id"] for r in records]))
        
        user_idx = {u: i for i, u in enumerate(user_list)}
        concept_idx = {c: i for i, c in enumerate(concept_list)}
        
        matrix = np.zeros((len(user_list), len(concept_list)))
        # Fill defaults
        matrix.fill(0.50)
        
        for r in records:
            u_id = r["user_id"]
            c_id = r["node_id"]
            if u_id in user_idx and c_id in concept_idx:
                matrix[user_idx[u_id], concept_idx[c_id]] = float(r["expected_mastery"])

        print(f"[ML Train] Created Student-Concept Matrix of shape {matrix.shape}")

        # 3. Compute Pearson correlation coefficients
        # np.corrcoef calculates correlation between rows, so we transpose to get concept correlations
        corr_matrix = np.corrcoef(matrix.T)
        # Replace NaNs (if constant features) with 0.0
        corr_matrix = np.nan_to_num(corr_matrix)
        
        print(f"[ML Train] Pearson correlation matrix calculated.")

        # 4. Save model weights to JSON
        model_data = {
            "concepts": concept_list,
            "correlations": corr_matrix.tolist()
        }
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/correlation_weights.json"
        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"[ML Train] Model weights saved locally to: {model_path}")

        # 5. Write Global Correlations to PostgreSQL
        with conn.cursor() as cur:
            # Clear old coefficients
            cur.execute("DELETE FROM concept_correlations;")
            
            for i, c_src in enumerate(concept_list):
                for j, c_tgt in enumerate(concept_list):
                    if i == j:
                        continue # Skip self-correlation
                    coefficient = float(corr_matrix[i, j])
                    cur.execute(
                        """
                        INSERT INTO concept_correlations (source_node, target_node, correlation_coefficient, sample_size)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (source_node, target_node) DO UPDATE SET
                            correlation_coefficient = EXCLUDED.correlation_coefficient,
                            sample_size = EXCLUDED.sample_size,
                            updated_at = NOW();
                        """,
                        (c_src, c_tgt, coefficient, len(user_list))
                    )
            
            # Write individual student-specific correlations
            # For each user, we calculate a customized correlation based on their deviation from cohort
            cur.execute("DELETE FROM user_concept_correlations;")
            cohort_means = np.mean(matrix, axis=0)
            
            for u_idx, u_id in enumerate(user_list):
                # Only insert for real UUID users, skip string mock user IDs
                if len(str(u_id)) < 30: 
                    continue
                for i, c_src in enumerate(concept_list):
                    for j, c_tgt in enumerate(concept_list):
                        if i == j:
                            continue
                        # Student deviation correlation: base model weight adjusted by user performance ratio
                        dev_weight = float(corr_matrix[i, j] * (1.0 - abs(matrix[u_idx, i] - matrix[u_idx, j])))
                        cur.execute(
                            """
                            INSERT INTO user_concept_correlations (user_id, source_node, target_node, correlation_weight)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (user_id, source_node, target_node) DO NOTHING;
                            """,
                            (u_id, c_src, c_tgt, dev_weight)
                        )
                        
            conn.commit()
            logger.info("All correlations committed to PostgreSQL successfully.")
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    train_correlation_model()
