import os, time, glob
from datetime import datetime, timedelta

base = "data/outputs"
days_to_keep = 3
cutoff = time.time() - days_to_keep * 86400

def clean_dir(path, exts):
    for ext in exts:
        for f in glob.glob(os.path.join(path, f"*.{ext}")):
            if os.path.getmtime(f) < cutoff:
                os.remove(f)
                print(f"🧹 Deleted old {ext}: {f}")

clean_dir(os.path.join(base, "equity"), ["csv"])
clean_dir(os.path.join(base, "trades"), ["csv"])
print("✅ Cleanup complete.")