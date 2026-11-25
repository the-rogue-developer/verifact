# Demo setup on Colab: fresh session → running Streamlit app

Run these cells in order in a new Colab notebook. Each code block is one cell.

**Runtime:** Runtime → Change runtime type → **T4 GPU**. It also works on CPU,
but building the dense index and answering queries is several times slower.

**Before you start**, Drive must already contain the trained Hindi verifier at
`MyDrive/verifact/models/nli_hindi/best` (written by `scripts/run_colab_hindi.sh`).

Time: about 10 minutes on the first run (pip, Wikipedia fetch, model downloads,
index). Later sessions reuse the Wikipedia files saved to Drive.

---

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/verifact'
MODEL_DIR = f'{DRIVE}/models/nli_hindi/best'
WIKI_DIR = f'{DRIVE}/corpus'           # Wikipedia files are cached here between sessions
!mkdir -p "$WIKI_DIR"
```

### 2. Clone the repo (token entered with getpass)

The token is typed into a hidden prompt, so it never appears in the notebook.
It is only passed to the one git command (never saved in `.git/config`),
masked in any error message, and deleted from memory afterwards. Use a GitHub
fine-grained token with read-only "Contents" access to this repo. If the repo
is public, just press Enter. Rerunning the cell pulls the latest code.

```python
import getpass, os, subprocess

REPO = 'github.com/the-rogue-developer/verifact.git'
DEST = '/content/verifact'

token = getpass.getpass('GitHub token (Enter if the repo is public): ').strip()
url = f'https://{token}@{REPO}' if token else f'https://{REPO}'
if os.path.isdir(f'{DEST}/.git'):   # rerun: update the existing clone
    cmd = ['git', '-C', DEST, 'pull', '-q', url, 'main']
else:
    cmd = ['git', 'clone', '-q', url, DEST]
r = subprocess.run(cmd, capture_output=True, text=True)
msg = r.stdout + r.stderr
print((msg.replace(token, '***') if token else msg) or 'done')
del token, url
assert r.returncode == 0, 'git failed (see message above)'
# The token was only passed on the command line, never stored: .git/config keeps the plain URL.
subprocess.run(['git', '-C', DEST, 'remote', 'set-url', 'origin', f'https://{REPO}'], check=True)
```

### 3. Install dependencies

```python
%cd /content/verifact
!pip install -q -r requirements.txt
!python -m pytest -q    # optional sanity check: offline, ~15 s
```

### 4. Fetch Wikipedia lead sections (cached on Drive)

About 20 + 4 API requests. Skipped if the files are already on Drive. Wikipedia's
policy asks for contact details in the User-Agent: replace `<your email>`.

```python
UA = 'VeriFact-student-project/0.1 (BIT Mesra minor project; <your email>)'

for lang in ('en', 'hi'):
    out = f'{WIKI_DIR}/wiki_{lang}.jsonl'
    if os.path.exists(out):
        print('cached:', out)
    else:
        !python scripts/fetch_wikipedia.py --titles data/wiki_topics_{lang}.txt --lang {lang} --out "{out}" --user-agent "{UA}"
!wc -l "$WIKI_DIR"/wiki_*.jsonl
```

### 5. Build the corpus

Wikipedia articles plus the sample demo documents, chunked into passages.

```python
!python scripts/prepare_data.py corpus \
    --docs "$WIKI_DIR/wiki_en.jsonl" "$WIKI_DIR/wiki_hi.jsonl" data/sample/corpus_docs.jsonl \
    --out data/corpus/passages.jsonl
```

### 6. Point the pipeline at the Hindi verifier

Writes `configs/pipeline_colab.yaml` (a copy of `configs/pipeline.yaml` with
`verifier.model_dir` changed), and sets `VERIFACT_CONFIG` so the app opens with it.

```python
import yaml

assert os.path.exists(f'{MODEL_DIR}/verifact_task.json'), f'no trained model at {MODEL_DIR}'

cfg = yaml.safe_load(open('configs/pipeline.yaml'))
cfg['verifier']['model_dir'] = MODEL_DIR
cfg['retrieval']['corpus'] = 'data/corpus/passages.jsonl'
cfg['retrieval']['index_dir'] = 'artifacts/index'
yaml.safe_dump(cfg, open('configs/pipeline_colab.yaml', 'w'), allow_unicode=True, sort_keys=False)

os.environ['VERIFACT_CONFIG'] = '/content/verifact/configs/pipeline_colab.yaml'
print(open('configs/pipeline_colab.yaml').read())
```

### 7. Build the BM25 and FAISS indexes

Downloads the multilingual-e5 embedding model on the first run.

```python
!python scripts/build_index.py --config configs/pipeline_colab.yaml
```

### 8. Check the pipeline from the command line

If this prints a verdict, the app will work too.

```python
!python scripts/verify.py "सरकार 1 दिसंबर से सभी 500 के नोट बंद कर रही है" --config configs/pipeline_colab.yaml | head -20
```

### 9. Start Streamlit in the background

The two flags are needed because Colab shows the app through a proxy on a
different origin. Otherwise Streamlit's CORS and XSRF checks block the
browser's connection and the page stays blank or keeps reconnecting.

```python
!pkill -f "[s]treamlit run" || true   # [s] stops pkill matching its own shell
!nohup streamlit run app/streamlit_app.py --server.port 8501 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false > streamlit.log 2>&1 &

import time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=2)
        print('Streamlit is up'); break
    except Exception:
        time.sleep(1)
else:
    print(open('streamlit.log').read()[-3000:])
```

### 10. Show the app in the notebook

```python
from google.colab import output
output.serve_kernel_port_as_iframe(8501, height=1000)
```

The first "Check" takes up to a minute while the models load; later checks are
fast. For a full browser tab instead of the iframe:
`output.serve_kernel_port_as_window(8501)`.

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| Blank iframe or "connecting" forever | Rerun cell 9 and check that both `--server.enable*` flags are there, then rerun cell 10 |
| Error in the app | `!tail -50 streamlit.log` |
| `no trained model at ...` in cell 6 | Run `scripts/run_colab_hindi.sh` first, or set `MODEL_DIR` to `.../models/nli_fever/best` |
| Wikipedia fetch skipped titles | Normal for a few titles; the log lists them. Delete the Drive file to refetch |
| Changed code on GitHub | Rerun cell 2 (it pulls), then cells 9–10 |
| Stop the app | `!pkill -f "[s]treamlit run"` |
