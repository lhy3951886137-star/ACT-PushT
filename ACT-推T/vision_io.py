"""Stage 2 I/O: unique temporary files and bounded Windows sharing retries."""
import json, os, time, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
S2=ROOT/'stage2'

def atomic(path, writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        writer(tmp)
        for attempt in range(12):
            try:os.replace(tmp,path);return
            except OSError as e:
                if getattr(e,'winerror',None) not in (5,32,33) or attempt==11:raise
                time.sleep(min(.02*2**attempt,.5))
    finally:
        if tmp.exists():tmp.unlink()

def save_json(path,value):
    atomic(path,lambda p:p.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8'))

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def stopped():
    if (ROOT/'STOP_REQUESTED').exists():raise InterruptedError('Stop requested')
