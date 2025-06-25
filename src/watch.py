"""
watch.py
Poll Numerai until live data for today's round is available.
Exit codes:
  0  — live data ready
  78 — timed out (GitHub 'neutral' code, job stops)
"""
import sys, time, datetime as dt, numerapi

POLL_SEC  = 600        # 10-min sleep
MAX_HOURS = 5          # stop before Actions' 6-h cap

napi   = numerapi.NumerAPI()
start  = dt.datetime.utcnow()

while (dt.datetime.utcnow() - start).total_seconds() < MAX_HOURS * 3600:
    if napi.check_round_open():          # ✅  confirmed existing in numerapi 0.6+
        print("Live data ready – continue workflow.")
        sys.exit(0)
    print(dt.datetime.utcnow(), "Live data not ready; sleep", POLL_SEC)
    time.sleep(POLL_SEC)

print("Timeout reached — no live round today.")
sys.exit(78)                             # special exit so Actions marks step failed
