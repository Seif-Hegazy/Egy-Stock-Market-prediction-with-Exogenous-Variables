from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import our scraping logic
# Since we're in /opt/airflow/dags/economic/, we need to import from the utils subdirectory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.scraper_logic import (
    scrape_usd_egp,
    scrape_inflation,
    scrape_interest_rate,
    scrape_gasoline,
    scrape_gold_24k,
    scrape_gdp
)

# --- Configuration ---
DATA_PATH = "/opt/airflow/data/raw/economic/egypt_economic_data.csv"
EMAIL_RECIPIENT = "seifhegazy2003@gmail.com"

# --- Callbacks ---
def send_failure_email(context):
    """Sends an email on task failure using credentials from Airflow Variable."""
    try:
        # Retrieve SMTP password from Airflow Variable
        # WARNING: This bypasses standard Airflow SMTP config as requested by user.
        smtp_password = Variable.get("smtp_password", default_var=None)
        
        if not smtp_password:
            print("SMTP Password not found in Airflow Variables. Skipping email.")
            return

        # Email content
        subject = f"Airflow Task Failed: {context['task_instance_key_str']}"
        body = f"""
        Task Failed!
        
        DAG: {context['dag'].dag_id}
        Task: {context['task'].task_id}
        Execution Time: {context['execution_date']}
        Log URL: {context['task_instance'].log_url}
        """

        msg = MIMEMultipart()
        msg['From'] = "airflow@example.com" # Replace with actual sender if known
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # SMTP Setup (Assuming Gmail for now based on user email, but this should be generic)
        # If user didn't specify host/port, we default to Gmail settings or need more Variables.
        # For safety, I'll assume standard Gmail port 587.
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        # We need the sender email too. I'll assume the recipient is the sender for self-alerting
        # or look for another variable. Let's use the recipient as sender for simplicity
        # unless 'smtp_user' variable exists.
        smtp_user = Variable.get("smtp_user", default_var=EMAIL_RECIPIENT)
        
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print(f"Failure email sent to {EMAIL_RECIPIENT}")
        
    except Exception as e:
        print(f"Failed to send failure email: {e}")

# --- DAG Definition ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False, # We use custom callback
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': send_failure_email,
    'execution_timeout': timedelta(hours=1),  # Kill task if it exceeds 1 hour
}

with DAG(
    'egypt_economic_data_v1',
    default_args=default_args,
    description='Scrapes Egyptian economic indicators daily',
    schedule_interval='0 4 * * *', # 04:00 UTC = 06:00 Cairo (Winter) / 07:00 Cairo (Summer)
    start_date=days_ago(1),
    catchup=False,
    tags=['egypt', 'economics', 'scraping'],
) as dag:

    # --- Tasks ---

    t1_usd = PythonOperator(
        task_id='scrape_exchange_rate',
        python_callable=scrape_usd_egp,
    )

    t2_inflation = PythonOperator(
        task_id='scrape_inflation',
        python_callable=scrape_inflation,
    )

    t3_interest = PythonOperator(
        task_id='scrape_interest_rate',
        python_callable=scrape_interest_rate,
    )

    t4_gasoline = PythonOperator(
        task_id='scrape_gasoline',
        python_callable=scrape_gasoline,
    )

    t6_gold = PythonOperator(
        task_id='scrape_gold',
        python_callable=scrape_gold_24k,
    )

    t7_gdp = PythonOperator(
        task_id='scrape_gdp',
        python_callable=scrape_gdp,
    )

    def save_data_to_csv(**context):
        """Collects XCom results and appends to CSV."""
        ti = context['ti']
        
        # Pull data from XComs
        usd_rate = ti.xcom_pull(task_ids='scrape_exchange_rate')
        inflation = ti.xcom_pull(task_ids='scrape_inflation')
        cbe_rates = ti.xcom_pull(task_ids='scrape_interest_rate')  # Returns dict with deposit & lending
        gasoline = ti.xcom_pull(task_ids='scrape_gasoline')
        gold = ti.xcom_pull(task_ids='scrape_gold')
        gdp = ti.xcom_pull(task_ids='scrape_gdp')

        # Handle Gasoline dict
        gas_80 = gasoline.get('gas_80') if gasoline else None
        gas_92 = gasoline.get('gas_92') if gasoline else None
        gas_95 = gasoline.get('gas_95') if gasoline else None
        
        # Handle CBE rates dict
        cbe_deposit = cbe_rates.get('deposit_rate') if cbe_rates else None
        cbe_lending = cbe_rates.get('lending_rate') if cbe_rates else None

        # Create Record (NEW columns: cbe_deposit_rate, cbe_lending_rate)
        record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'usd_sell_rate': usd_rate,
            'gold_24k': gold,
            'gasoline_80': gas_80,
            'gasoline_92': gas_92,
            'gasoline_95': gas_95,
            'cbe_deposit_rate': cbe_deposit,
            'cbe_lending_rate': cbe_lending,
            'headline_inflation': inflation,
            'gdp_usd_billion': gdp
        }
        
        df = pd.DataFrame([record])
        
        # Check if file exists to determine header
        header = not os.path.exists(DATA_PATH)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        if os.path.exists(DATA_PATH):
            existing_df = pd.read_csv(DATA_PATH)
            # Ensure new columns exist in old data (fill NaN)
            for col in record.keys():
                if col not in existing_df.columns:
                    existing_df[col] = None
            
            # Combine
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates if run multiple times today
            combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            # Sort Newest to Oldest
            combined_df.sort_values(by='date', ascending=False, inplace=True)
            combined_df.to_csv(DATA_PATH, index=False)
        else:
            df.to_csv(DATA_PATH, index=False)
            
        print(f"Data saved to {DATA_PATH}: {record}")

    t8_save = PythonOperator(
        task_id='save_data',
        python_callable=save_data_to_csv,
        provide_context=True,
    )

    # --- Dependencies ---
    [t1_usd, t2_inflation, t3_interest, t4_gasoline, t6_gold, t7_gdp] >> t8_save
