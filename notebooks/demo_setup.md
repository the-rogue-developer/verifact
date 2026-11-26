# Demo setup on Colab: fresh session → running Streamlit app

Run these cells in order in a new Colab notebook. Each code block is one cell.

**Runtime:** Runtime → Change runtime type → **T4 GPU**. It also works on CPU,
but building the dense index and answering queries is several times slower.

**Before you start**, Drive must already contain the trained Hindi verifier at
`MyDrive/verifact/models/nli_hindi/best` (written by `scripts/run_colab_hindi.sh`).

**Recommended, once:** upload the complete Wikipedia files from the laptop
(`data/raw/wiki_en.jsonl`: 390 articles, `data/raw/wiki_hi.jsonl`: 70 articles)
to `MyDrive/verifact/corpus/`. Cell 4 then uses them and never contacts
Wikipedia, so rate limiting (HTTP 429, which lost 70 articles in the first
Colab run) cannot happen during the demo. The corpus is then 493 documents →
1781 passages.

Time: about 10 minutes on the first run (pip, model downloads, index).

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

Uses the files on Drive if they are there. Otherwise it fetches (about 20 + 4 API
requests; on HTTP 429 the fetcher waits and retries) and saves to Drive **only
if every title was fetched**, so an incomplete corpus is never cached for later
sessions. Wikipedia's policy asks for contact details in the User-Agent: replace
`<your email>`.

```python
import shutil

UA = 'VeriFact-student-project/0.1 (BIT Mesra minor project; <your email>)'

WIKI_FILES = []
for lang in ('en', 'hi'):
    cached = f'{WIKI_DIR}/wiki_{lang}.jsonl'
    if os.path.exists(cached):
        print('cached:', cached)
        WIKI_FILES.append(cached)
        continue
    tmp = f'/content/wiki_{lang}.jsonl'
    r = subprocess.run(['python', 'scripts/fetch_wikipedia.py', '--titles', f'data/wiki_topics_{lang}.txt',
                        '--lang', lang, '--out', tmp, '--user-agent', UA], capture_output=True, text=True)
    print(r.stderr[-1500:])   # the fetcher logs progress to stderr
    if r.returncode == 0:
        shutil.copy(tmp, cached)
        print('complete, saved to Drive:', cached)
    else:
        print(f'WARNING: some {lang} titles failed (listed above). Using the incomplete file for this '
              'session only; rerun this cell later to try again.')
    WIKI_FILES.append(tmp)
WIKI_ARGS = ' '.join(f'"{f}"' for f in WIKI_FILES)   # quoted paths for the shell commands below
!wc -l {WIKI_ARGS}
```

Expected: 390 English and 70 Hindi lines.

### 5. Build the corpus

Wikipedia articles plus the sample demo documents, chunked into passages.

```python
!python scripts/prepare_data.py corpus \
    --docs {WIKI_ARGS} data/sample/corpus_docs.jsonl \
    --out data/corpus/passages.jsonl
```

Expected: `corpus: 493 documents -> 1781 passages`.

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

### 9. Rehearse: run all 15 sample forwards

Runs the 5 English, 5 Hindi and 5 Hinglish messages from
`data/sample/demo_forwards.jsonl` through the same pipeline and corpus the app
uses, and saves the table to Drive. Use it to choose which examples to show
live: pick the ones where the verdict matches `expected`.

```python
!python scripts/run_demo.py --config configs/pipeline_colab.yaml \
    --set retrieval.corpus=data/corpus/passages.jsonl retrieval.index_dir=artifacts/index \
    --out "{DRIVE}/results/demo_rehearsal.json"
```

The `evidence` score only counts the exact sample document the message was
written against. With the full corpus a Wikipedia article on the same topic
(for example the PM-KISAN or Aadhaar article) sometimes ranks first. That is
still correct evidence but counts as a miss, so read the evidence column
rather than relying on the score alone.

### 10. Start Streamlit in the background

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

### 11. Show the app in the notebook

```python
from google.colab import output
output.serve_kernel_port_as_iframe(8501, height=1000)
```

The first "Check" takes up to a minute while the models load; later checks are
fast. For a full browser tab instead of the iframe:
`output.serve_kernel_port_as_window(8501)`.

The **"Try an example forward"** dropdown fills the text box with one of the 15
sample messages, so you don't have to type Hindi during the demo. You can still
edit the text or paste your own.

---

### Demo-day checklist

**The day before**
- Run cells 1–11 once in a fresh session. This checks the token, the Drive
  model, the GPU quota and the downloads.
- Look at the rehearsal table (cell 9) and note 2–3 messages per language that
  come out right. Prepare one message of your own that is not in the sample
  files, to show it isn't limited to prepared examples.
- Keep `demo_rehearsal.json` and a few screenshots of the app as a backup in
  case Colab or the network fails during the demo.

**15 minutes before**
- Fresh session: run cells 1–8 and 10–11 (skip the rehearsal to save time).
- Press **Check** once on any example so the models are loaded before the
  audience is watching.

**Things to be ready to explain**
- "UNVERIFIED" is an honest answer: nothing in the corpus is close enough to
  the claim. It is not a crash. The knowledge base is 493 documents, not the whole web.
- The verdict comes from retrieved evidence plus the NLI model, not from the
  writing style. Expand a claim to show the key evidence and its source link.
- Demo documents have no link and are labelled "Demo evidence (written for this
  prototype)".

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| Blank iframe or "connecting" forever | Rerun cell 10 and check that both `--server.enable*` flags are there, then rerun cell 11 |
| Error in the app | `!tail -50 streamlit.log` |
| `no trained model at ...` in cell 6 | Run `scripts/run_colab_hindi.sh` first, or set `MODEL_DIR` to `.../models/nli_fever/best` |
| Cell 4 warns that titles failed | Wikipedia rate limiting. Rerun cell 4 a few minutes later, or upload the laptop's `data/raw/wiki_*.jsonl` to `MyDrive/verifact/corpus/` |
| Corpus count is not 493 → 1781 | A Drive file is incomplete (e.g. from an older run). Delete it, or replace it with the laptop copy, then rerun cells 4, 5 and 7 |
| Changed code on GitHub | Rerun cell 2 (it pulls), then cells 10–11 |
| Stop the app | `!pkill -f "[s]treamlit run"` |
