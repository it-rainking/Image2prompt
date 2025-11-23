# Image2Prompt - Project Documentation

## Overview

**Image2Prompt** è uno strumento web standalone per convertire immagini in prompt Stable Diffusion dettagliati usando Claude Vision API.

- **File principale**: `image2prompt.py` (single file, ~1400 righe)
- **Versione**: 1.0
- **Autore**: Nick
- **Licenza**: Proprietario

## Use Cases

- Rigenerare immagini personali con Stable Diffusion
- Processare contenuti per creare varianti
- Monetizzare come servizio SaaS
- Catalogazione librerie visive
- Reverse engineering di stili artistici
- A/B testing di prompt
- Creazione dataset per training

## Tech Stack

| Componente | Tecnologia |
|------------|------------|
| Backend | Flask + Claude Vision API |
| Frontend | HTML/CSS/JS vanilla (dark mode, split screen) |
| Database | SQLite locale |
| Backup | Dropbox automatico |
| CLI | Rich (progress bars, tables, panels) |
| Deploy | Locale / Pod GPU 4090 |

## Project Structure

```
Image2prompt/
├── image2prompt.py          # App standalone (NUOVO)
├── claude.md                # Questa documentazione
├── clip_interrogator/       # Legacy CLIP Interrogator
├── run_cli.py               # Legacy CLI
├── run_gradio.py            # Legacy Gradio UI
└── ...
```

## Funzionalità

### Core
- [x] Upload batch immagini (drag & drop + file picker)
- [x] Analisi automatica con Claude Vision API
- [x] Split screen: anteprima sinistra, prompt destra
- [x] Editor prompt real-time
- [x] Thumbnail preview

### Persistenza
- [x] Salvataggio modifiche in SQLite
- [x] Storico sessioni
- [x] Backup Dropbox automatico

### Export
- [x] CSV (filename, prompt originale, prompt modificato, data)
- [x] TXT (formato leggibile con separatori)
- [x] HTML (con thumbnail inline, dark theme)

### Setup
- [x] Auto-installazione dipendenze
- [x] CLI Rich per setup interattivo
- [x] Apertura browser automatica

## Architettura

### Classi Principali

#### `Config`
Configurazione dell'applicazione:
- `API_KEY`: Anthropic API key (da env)
- `DB_PATH`: `~/.image2prompt/database.db`
- `BACKUP_PATH`: `~/Dropbox`
- `UPLOAD_FOLDER`: `~/.image2prompt/uploads`
- `SESSION_FOLDER`: `~/.image2prompt/sessions`
- `MAX_IMAGE_SIZE`: 5MB

#### `Database`
Gestione SQLite:
- `init_db()`: Crea tabelle sessions e analyses
- `create_session(session_id)`: Nuova sessione
- `save_analysis(...)`: Salva analisi immagine
- `update_prompt(...)`: Aggiorna prompt modificato
- `get_session_analyses(session_id)`: Recupera analisi
- `backup_to_dropbox()`: Backup automatico

#### `ImageAnalyzer`
Analisi con Claude Vision:
- `image_to_base64(path)`: Converte immagine
- `get_image_media_type(path)`: Determina MIME type
- `analyze(path)`: Chiama Claude API e restituisce JSON strutturato

#### `Exporter`
Export in vari formati:
- `export_csv(analyses, path)`
- `export_txt(analyses, path)`
- `export_html(analyses, path)` - con thumbnail inline

### API Endpoints

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/` | GET | Pagina principale |
| `/api/init_session` | POST | Inizializza sessione |
| `/api/analyze` | POST | Analizza immagine |
| `/api/update_prompt` | POST | Salva modifica prompt |
| `/api/export` | POST | Esporta dati |

### Schema Database

```sql
-- Tabella sessioni
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabella analisi
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    image_filename TEXT,
    image_base64 TEXT,
    original_prompt TEXT,
    modified_prompt TEXT,
    analysis_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
```

## Dipendenze

Auto-installate all'avvio:
- `flask` - Web framework
- `anthropic` - Claude API client
- `Pillow` - Image processing
- `python-dotenv` - Environment variables
- `requests` - HTTP client
- `rich` - CLI styling

## Installazione e Lancio

```bash
# 1. Configura API key
export ANTHROPIC_API_KEY='sk-ant-...'

# 2. Lancia l'applicazione
python3 image2prompt.py

# 3. Si apre automaticamente http://localhost:5000
```

## Claude Vision Prompt

Il sistema usa questo prompt per analizzare le immagini:

```
Analizza l'immagine e genera un prompt SD con:
1. Descrizione visiva principale
2. Stile artistico
3. Lighting e atmosfera
4. Composizione e proporzioni
5. Qualità tecnica
6. Dettagli aggiuntivi

Output JSON:
- descrizione_visiva
- stile_artistico
- lighting_atmosfera
- composizione
- qualita_tecnica
- dettagli_aggiuntivi
- prompt_completo
- aspect_ratio
```

## Storage Paths

| Path | Contenuto |
|------|-----------|
| `~/.image2prompt/` | Cartella principale |
| `~/.image2prompt/database.db` | Database SQLite |
| `~/.image2prompt/uploads/` | Immagini caricate |
| `~/.image2prompt/sessions/` | File export |
| `~/Dropbox/image2prompt_backup.db` | Backup database |

## Analisi Integrazione con CLIP Interrogator

### Risorse Disponibili nel Repository

Il repository contiene dataset preziosi in `clip_interrogator/data/`:
- `artists.txt` - Lista di ~5000 artisti (es. "Aaron Jasinski", "Abbott Handerson Thayer")
- `flavors.txt` - Termini qualitativi SD (es. "highly detailed", "cinematic lighting", "8k")
- `mediums.txt` - Tipi di medium (es. "a digital painting", "oil on canvas", "3D render")
- `movements.txt` - Movimenti artistici
- `negative.txt` - Termini per negative prompts

### Ottimizzazioni Possibili

#### 1. Arricchimento Prompt con Dataset Esistenti
Il nuovo `image2prompt.py` usa solo Claude Vision, ma potrebbe integrare i dataset esistenti:

```python
# Aggiungere a ImageAnalyzer
def enhance_prompt_with_vocabulary(self, base_prompt: str) -> str:
    """Arricchisce il prompt Claude con termini SD ottimizzati"""
    flavors = load_list('clip_interrogator/data/flavors.txt')
    # Aggiungi termini qualitativi mancanti
    quality_terms = ['highly detailed', '8k', 'sharp focus']
    for term in quality_terms:
        if term not in base_prompt.lower():
            base_prompt += f", {term}"
    return base_prompt
```

#### 2. Validazione Artisti
Usare `artists.txt` per validare/suggerire nomi artisti nel prompt:

```python
def validate_artist_names(self, prompt: str, artists_list: List[str]) -> List[str]:
    """Suggerisce artisti simili dalla lista ufficiale"""
    # Fuzzy matching per correggere typo
    pass
```

#### 3. Negative Prompt Automatico
Integrare `negative.txt` per generare automaticamente negative prompts:

```python
def generate_negative_prompt(self) -> str:
    """Genera negative prompt standard"""
    negatives = load_list('clip_interrogator/data/negative.txt')
    return ", ".join(negatives[:20])
```

### Errori e Problemi nel Codice

#### 1. Mancanza Error Handling per JSON Parsing
```python
# PROBLEMA in ImageAnalyzer.analyze():
json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
if json_match:
    analysis_data = json.loads(json_match.group())  # Può fallire!
```
**Fix**: Aggiungere try/except per JSON malformato.

#### 2. Nessuna Validazione Dimensione Immagine
```python
# PROBLEMA: MAX_IMAGE_SIZE definito ma mai usato
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
# Ma in /api/analyze non c'è controllo!
```
**Fix**: Aggiungere validazione prima del processing.

#### 3. Memory Leak Potenziale
```python
# PROBLEMA: Immagini salvate in temp_path ma mai eliminate
temp_path = Config.UPLOAD_FOLDER / image_file.filename
image_file.save(temp_path)
# ... analisi ...
# MAI: os.remove(temp_path)
```
**Fix**: Pulire file temporanei dopo analisi o usare tempfile.

#### 4. Race Condition su Session ID
```python
# PROBLEMA: Session ID generato lato client
const sessionId = 'session_' + Date.now();
# Due tab aperti contemporaneamente = stesso ID potenziale
```
**Fix**: Generare UUID lato server.

#### 5. SQL Injection Potenziale
```python
# PROBLEMA: Filename non sanitizzato
cursor.execute('''
    UPDATE analyses
    SET modified_prompt = ?
    WHERE session_id = ? AND image_filename = ?
''', (new_prompt, session_id, filename))
# filename viene dal client senza validazione
```
**Fix**: Sanitizzare filename o usare solo ID interni.

#### 6. Mancanza Timeout API Claude
```python
# PROBLEMA: Nessun timeout sulla chiamata API
message = self.client.messages.create(...)
# Può bloccare indefinitamente
```
**Fix**: Aggiungere `timeout` parameter.

### Migliorie Suggerite

#### 1. Modalità Ibrida Claude + CLIP
Combinare i due approcci:
- Claude Vision per descrizione semantica
- CLIP per matching con termini SD ottimizzati

```python
class HybridAnalyzer:
    def __init__(self):
        self.claude = ImageAnalyzer(api_key)
        self.clip = Interrogator(Config())

    def analyze(self, image_path):
        # 1. Analisi Claude per semantica
        claude_result = self.claude.analyze(image_path)

        # 2. CLIP per termini tecnici SD
        image = Image.open(image_path)
        clip_flavors = self.clip.flavors.rank(
            self.clip.image_to_features(image), 10
        )

        # 3. Merge intelligente
        return self.merge_results(claude_result, clip_flavors)
```

#### 2. Caching Embeddings
Riutilizzare il sistema di cache di CLIP Interrogator:
- Precomputed embeddings su HuggingFace
- Cache locale in `cache/` folder

#### 3. UI: Analyze Tab dal Gradio
Importare il tab "Analyze" da `run_gradio.py` che mostra:
- Top 5 Medium
- Top 5 Artists
- Top 5 Movements
- Top 5 Trending
- Top 5 Flavors

#### 4. Export Migliorato
Aggiungere formato JSON con tutti i metadati:
```python
def export_json(analyses, output_path):
    data = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "analyses": analyses
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
```

### Confronto Approcci

| Aspetto | CLIP Interrogator | Image2Prompt (Claude) |
|---------|-------------------|----------------------|
| **Velocità** | Fast mode: 1-2s | 3-5s per immagine |
| **Costo** | Gratuito (locale) | ~$0.01 per immagine |
| **Qualità** | Termini SD ottimizzati | Descrizione semantica ricca |
| **Offline** | Sì | No |
| **GPU** | Richiesta (o CPU lento) | Non richiesta |
| **Personalizzazione** | Dataset modificabili | Prompt system modificabile |

### Raccomandazione (con API)

**Approccio ibrido con Claude:**
1. Usare Claude Vision per la descrizione base (soggetto, scena, mood)
2. Arricchire con termini da `flavors.txt` e `mediums.txt`
3. Validare artisti con `artists.txt`
4. Generare negative prompt da `negative.txt`
5. Fallback a CLIP puro se API non disponibile

---

## Approccio Ibrido Offline (Gratuito, No GPU)

### Obiettivo
Creare un sistema che funzioni:
- **Gratis** (no API a pagamento)
- **Offline** (nessuna connessione richiesta)
- **Senza GPU** (solo CPU)
- **Velocità accettabile** (anche 30-60s va bene)

### Modelli Vision Consigliati per CPU

| Modello | RAM | Tempo CPU | Qualità | Miglior uso |
|---------|-----|-----------|---------|-------------|
| **Florence-2 Base** | 1-2GB | 3-8s | Buona | OCR, multi-task, più leggero |
| **Moondream 2** | 4-6GB | 5-10s | Buona | VQA generale, edge |
| **BLIP Base** | 2-3GB | 10-30s | Buona | Captioning (già nel repo) |
| **SmolVLM2** | 4-5GB | 8-15s | Buona | Multilingue |

### Architettura Proposta

```
┌─────────────────────────────────────────────────────────────┐
│                    IMAGE INPUT                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Caption Base (Moondream2 o Florence-2)             │
│  → "A woman in a red dress standing in a garden"            │
│  Tempo: 5-10s su CPU                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Arricchimento con Dataset Locali                   │
│  → flavors.txt: "highly detailed, 8k, sharp focus"          │
│  → mediums.txt: "digital painting"                          │
│  → movements.txt: "art nouveau"                             │
│  Tempo: <1s                                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Negative Prompt                                    │
│  → negative.txt: "blurry, bad anatomy, worst quality..."    │
│  Tempo: <1s                                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT FINALE                                              │
│  Prompt: "A woman in a red dress standing in a garden,      │
│           digital painting, art nouveau, highly detailed,   │
│           8k, sharp focus, cinematic lighting"              │
│  Negative: "blurry, bad anatomy, worst quality, low res"    │
│  Tempo totale: ~10-15s su CPU                               │
└─────────────────────────────────────────────────────────────┘
```

### Implementazione Consigliata

#### Opzione 1: Moondream 2 (Migliore qualità/velocità)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import os

class OfflineImageAnalyzer:
    """Analizzatore immagini offline con Moondream 2"""

    def __init__(self, data_path="clip_interrogator/data"):
        # Carica modello Moondream 2 (4-6GB RAM, 5-10s/img)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            revision="2024-08-01",
            device_map='cpu'
        )

        # Carica dataset locali
        self.flavors = self._load_list(data_path, 'flavors.txt')
        self.mediums = self._load_list(data_path, 'mediums.txt')
        self.movements = self._load_list(data_path, 'movements.txt')
        self.negatives = self._load_list(data_path, 'negative.txt')
        self.artists = self._load_list(data_path, 'artists.txt')

    def _load_list(self, path, filename):
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return []

    def generate_caption(self, image_path: str) -> str:
        """Genera caption base con Moondream 2"""
        image = Image.open(image_path).convert('RGB')
        encoded = self.model.encode_image(image)

        # Prompt ottimizzato per Stable Diffusion
        question = """Describe this image in detail for Stable Diffusion.
        Include: subject, setting, lighting, colors, mood, art style."""

        return self.model.answer_question(encoded, question, self.tokenizer)

    def enhance_prompt(self, base_caption: str) -> str:
        """Arricchisce il prompt con termini SD ottimizzati"""
        # Aggiungi quality terms
        quality_terms = ['highly detailed', '8k', 'sharp focus']
        enhanced = base_caption

        for term in quality_terms:
            if term.lower() not in enhanced.lower():
                enhanced += f", {term}"

        return enhanced

    def generate_negative(self) -> str:
        """Genera negative prompt standard"""
        # Prendi i primi 15 termini negativi
        return ", ".join(self.negatives[:15])

    def analyze(self, image_path: str) -> dict:
        """Analisi completa dell'immagine"""
        # Step 1: Caption base
        caption = self.generate_caption(image_path)

        # Step 2: Arricchimento
        prompt = self.enhance_prompt(caption)

        # Step 3: Negative
        negative = self.generate_negative()

        return {
            "prompt": prompt,
            "negative_prompt": negative,
            "base_caption": caption
        }
```

#### Opzione 2: Florence-2 (Più leggero, 1-2GB RAM)

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

class Florence2Analyzer:
    """Analizzatore ultra-leggero con Florence-2"""

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

    def analyze(self, image_path: str) -> dict:
        image = Image.open(image_path).convert('RGB')

        # Detailed caption
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return {
            "prompt": caption,
            "negative_prompt": "blurry, bad quality, low resolution"
        }
```

#### Opzione 3: BLIP Base (Già nel repository)

```python
# Usa il BLIP già presente in clip_interrogator
from clip_interrogator import Config, Interrogator

class BLIPOnlyAnalyzer:
    """Usa solo BLIP senza CLIP (più veloce su CPU)"""

    def __init__(self):
        config = Config(
            caption_model_name='blip-base',  # 990MB, più veloce
            clip_model_name=None,  # Disabilita CLIP
            device='cpu'
        )
        self.ci = Interrogator(config)

    def analyze(self, image_path: str) -> str:
        image = Image.open(image_path).convert('RGB')
        return self.ci.generate_caption(image)
```

### Confronto Finale

| Approccio | Costo | Offline | GPU | RAM | Tempo | Qualità |
|-----------|-------|---------|-----|-----|-------|---------|
| Claude Vision API | ~$0.01/img | No | No | - | 3-5s | Ottima |
| Moondream 2 + Dataset | Gratis | Sì | No | 4-6GB | 10-15s | Buona |
| Florence-2 + Dataset | Gratis | Sì | No | 1-2GB | 5-10s | Buona |
| BLIP Base + Dataset | Gratis | Sì | No | 2-3GB | 15-30s | Discreta |
| CLIP Interrogator Full | Gratis | Sì | Sì* | 6GB+ | 10-20s | Ottima |

*CLIP su CPU è molto lento (minuti per immagine)

### Dipendenze Aggiuntive

```txt
# requirements-offline.txt
transformers>=4.40.0
torch>=2.0.0
Pillow
accelerate
```

### Note Implementazione

1. **Prima esecuzione**: Download modello (~2-4GB una tantum)
2. **Cache locale**: Salvare modello in `~/.cache/huggingface/`
3. **Quantizzazione**: Per RAM limitata, usare `load_in_8bit=True`
4. **Batch processing**: Processare più immagini in sequenza per ammortizzare load time

---

## Benchmark e Test di Qualità

### File Implementati

- `offline_analyzers.py` - Implementazione dei 3 analyzer (Moondream2, Florence-2, BLIP Base)
- `benchmark.py` - Script di benchmark automatizzato

### Come Eseguire i Benchmark

```bash
# 1. Installa dipendenze
pip install transformers torch accelerate Pillow

# 2. Benchmark su singola immagine (tutti i modelli)
python benchmark.py -i test_image.jpg

# 3. Benchmark su cartella di immagini
python benchmark.py -d ./test_images/

# 4. Benchmark solo modelli specifici
python benchmark.py -i test.jpg -m moondream2 florence2

# 5. Test rapido singolo modello
python offline_analyzers.py -i test.jpg -m moondream2
```

### Output Benchmark

I risultati vengono salvati in `benchmark_results/`:
- `benchmark_YYYYMMDD_HHMMSS.json` - Dati completi
- `benchmark_YYYYMMDD_HHMMSS.txt` - Report leggibile

### Test di Qualità Consigliati

#### 1. Test Velocità (Quantitativo)

| Metrica | Come Misurarla |
|---------|----------------|
| Tempo prima inferenza | Include caricamento modello |
| Tempo medio inferenza | Media su 3+ run (modello già caricato) |
| Tempo caricamento modello | Una tantum all'avvio |
| Memoria RAM utilizzata | `htop` o `psutil` durante esecuzione |

#### 2. Test Qualità Prompt (Qualitativo)

Usa queste **10 immagini di test** per valutare la qualità:

| # | Tipo Immagine | Cosa Valutare |
|---|---------------|---------------|
| 1 | **Ritratto persona** | Descrizione volto, espressione, abbigliamento |
| 2 | **Paesaggio naturale** | Atmosfera, luce, elementi naturali |
| 3 | **Arte digitale/fantasy** | Stile artistico, elementi fantastici |
| 4 | **Foto prodotto** | Dettagli oggetto, materiali, sfondo |
| 5 | **Illustrazione anime/manga** | Stile specifico, personaggio |
| 6 | **Architettura/interni** | Spazio, prospettiva, materiali |
| 7 | **Cibo/food photography** | Colori, presentazione, appetibilità |
| 8 | **Abstract art** | Forme, colori, composizione |
| 9 | **Vintage/retro photo** | Epoca, atmosfera, grain |
| 10 | **Low quality/blurry** | Come gestisce immagini difficili |

#### 3. Criteri di Valutazione Prompt (1-5)

Per ogni immagine, valuta il prompt generato:

| Criterio | Descrizione | Punteggio 1-5 |
|----------|-------------|---------------|
| **Accuratezza** | Descrive correttamente il contenuto? | |
| **Dettaglio** | Include dettagli rilevanti? | |
| **Stile SD** | Usa termini efficaci per Stable Diffusion? | |
| **Lunghezza** | Né troppo corto né troppo lungo? | |
| **Coerenza** | Il prompt è logico e ben strutturato? | |

#### 4. Test End-to-End (Opzionale)

Se hai accesso a Stable Diffusion:

1. Genera immagine con il prompt prodotto
2. Confronta con l'originale
3. Valuta similarità (CLIP score o visivamente)

### Template Scheda Valutazione

```markdown
## Test Image: [nome_file.jpg]

### Moondream 2
- Tempo: ___s
- Caption: "..."
- Prompt finale: "..."
- Accuratezza: _/5
- Dettaglio: _/5
- Stile SD: _/5
- Note: ...

### Florence-2
- Tempo: ___s
- Caption: "..."
- Prompt finale: "..."
- Accuratezza: _/5
- Dettaglio: _/5
- Stile SD: _/5
- Note: ...

### BLIP Base
- Tempo: ___s
- Caption: "..."
- Prompt finale: "..."
- Accuratezza: _/5
- Dettaglio: _/5
- Stile SD: _/5
- Note: ...

### Vincitore per questa immagine: ___
```

### Metriche Automatiche (nel benchmark.py)

Il benchmark calcola automaticamente:

- `avg_inference_time` - Tempo medio per immagine
- `min_inference_time` - Tempo minimo
- `max_inference_time` - Tempo massimo
- `load_time` - Tempo caricamento modello
- `word_count` - Parole nel prompt
- `prompt_length` - Caratteri nel prompt

### Risultati Attesi (Stime)

| Modello | Tempo CPU | RAM | Qualità Attesa |
|---------|-----------|-----|----------------|
| Florence-2 | 3-8s | 1-2GB | Buona, concisa |
| Moondream 2 | 5-10s | 4-6GB | Buona, dettagliata |
| BLIP Base | 10-30s | 2-3GB | Discreta, generica |

### Raccomandazioni Post-Benchmark

Dopo i test, scegli in base a:

1. **Priorità velocità** → Florence-2
2. **Priorità qualità** → Moondream 2
3. **Priorità RAM bassa** → Florence-2 o BLIP Base
4. **Già installato** → BLIP Base (nel repo)

## TODO / Roadmap

### Priorità Alta
- [ ] Fix: Error handling JSON parsing
- [ ] Fix: Validazione dimensione immagine
- [ ] Fix: Pulizia file temporanei
- [ ] Fix: Session ID lato server (UUID)
- [ ] Fix: Sanitizzazione filename

### Priorità Media
- [ ] Integrazione dataset CLIP (flavors, artists, mediums)
- [ ] Generazione negative prompt automatica
- [ ] Modalità ibrida Claude + CLIP
- [ ] Export JSON
- [ ] Timeout API Claude

### Priorità Bassa
- [ ] Supporto folder picker nativo
- [ ] Batch processing con progress bar
- [ ] Integrazione diretta con Automatic1111/ComfyUI
- [ ] Multi-language support
- [ ] API rate limiting
- [ ] User authentication per versione SaaS
- [ ] Storico versioni prompt

## Note di Sviluppo

- Il file è completamente standalone, nessuna dipendenza esterna oltre ai pip packages
- Le dipendenze vengono installate automaticamente all'avvio
- Il frontend è inline nel file Python (template string)
- Supporta immagini: JPG, JPEG, PNG, GIF, WEBP
- Limite dimensione immagine: 5MB
- Modello Claude: `claude-3-5-sonnet-20241022`

## Legacy Code

Questo repository contiene anche il vecchio CLIP Interrogator:
- `clip_interrogator/` - Package CLIP+BLIP
- `run_cli.py` - CLI legacy
- `run_gradio.py` - Gradio UI legacy
- `setup.py` - Package setup legacy

Il nuovo `image2prompt.py` è indipendente e non utilizza il codice legacy.

---
*Ultimo aggiornamento: 2024-11-23*
