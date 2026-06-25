from job_fetcher import fetch_and_store_jobs
import time

roles = [
    "AI-Powered Healthcare Data Engineer",
    "Smart Traffic Management Engineer",
]

for i, role in enumerate(roles, start=1):
    print(f"\n[{i}/{len(roles)}] Processing: {role}")
    try:
        fetch_and_store_jobs(role)
        time.sleep(2)
    except Exception as e:
        print(f"Failed for {role}: {e}")

print("\nDone.")