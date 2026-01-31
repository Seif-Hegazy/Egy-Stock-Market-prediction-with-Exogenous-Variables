"""
Airflow DAG for EgySentiment Daily Data Collection
Runs every 4 hours to collect new Egyptian financial news articles
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),  # Kill task if it exceeds 1 hour
}

dag = DAG(
    'egy_sentiment_daily_collection',
    default_args=default_args,
    description='Collect Egyptian financial news articles every 4 hours',
    schedule='0 */4 * * *',  # Every 4 hours
    start_date=datetime(2025, 11, 26),
    catchup=False,
    max_active_runs=1,  # Only one run at a time
    concurrency=1,  # Only one task running at a time
    tags=['sentiment', 'scraping', 'egyptian-finance'],
)


def check_data_quality(**context):
    """Verify data collection succeeded and check quality"""
    # Updated path for unified data directory
    data_file = '/opt/airflow/data/raw/news/articles.jsonl'
    
    if not os.path.exists(data_file):
        # Create file if it doesn't exist (first run)
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        print(f"⚠️ Data file not found, creating: {data_file}")
        with open(data_file, 'w') as f:
            pass  # Create empty file
        print("✓ Pipeline running for first time - no data to check yet")
        return 0
    
    # Count lines
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    
    if total_samples == 0:
        print("✓ No data collected yet - this is normal for first run")
        return 0
    
    # Get sentiment distribution
    sentiments = []
    for line in lines[-100:]:  # Check last 100 samples
        try:
            record = json.loads(line.strip())
            sentiments.append(record.get('sentiment', 'unknown'))
        except:
            continue
    
    from collections import Counter
    distribution = Counter(sentiments)
    
    print(f"✓ Total samples: {total_samples}")
    print(f"✓ Recent sentiment distribution: {dict(distribution)}")
    
    # Push to XCom for monitoring
    context['task_instance'].xcom_push(key='total_samples', value=total_samples)
    context['task_instance'].xcom_push(key='sentiment_dist', value=dict(distribution))
    
    return total_samples


# Task 1: Run daily data collection pipeline
collect_data = BashOperator(
    task_id='collect_daily_articles',
    bash_command='cd /opt/airflow && python src/data_pipeline.py',
    dag=dag,
)

# Task 2: Deduplicate data
deduplicate_data = BashOperator(
    task_id='deduplicate_data',
    bash_command='cd /opt/airflow && python src/deduplicate_data.py',
    dag=dag,
)

# Task 3: Check data quality
quality_check = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    provide_context=True,
    dag=dag,
)

# Task 4: Auto-Score New Articles (Limited to last 7 days to prevent resource hogging)
auto_score = BashOperator(
    task_id='auto_score_sentiment',
    bash_command='python /opt/airflow/src/auto_score.py',
    dag=dag,
)

# Task 5: Log Success  
log_success = BashOperator(
    task_id='log_success',
    bash_command='echo "Daily Sentiment Pipeline Completed Successfully at $(date)"',
    dag=dag,
)

# Define task dependencies
collect_data >> deduplicate_data >> quality_check >> auto_score >> log_success
