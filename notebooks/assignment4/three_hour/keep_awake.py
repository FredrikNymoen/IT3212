"""Prevent system sleep only during the approved work window; restore on exit."""
import ctypes
from datetime import datetime,timezone
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]
FLAG=ROOT/'results/assignment4/three_hour_20261122/stop_keep_awake'
DEADLINE=datetime(2026,11,22,22,58,42,tzinfo=timezone.utc)

if __name__=='__main__':
    function=ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes=[ctypes.c_uint];function.restype=ctypes.c_uint
    assert function(0x80000001),'Could not request system wakefulness.'
    print('System-sleep prevention active until work ends or 22:58:42 UTC. Display may sleep.',flush=True)
    try:
        while datetime.now(timezone.utc)<DEADLINE and not FLAG.exists():time.sleep(5)
    finally:
        function(0x80000000);print('System-sleep prevention released.',flush=True)
