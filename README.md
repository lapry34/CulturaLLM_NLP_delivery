# ✅ Modelli supportati

I modelli da noi testati sono:

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Unbabel/M-Prometheus-3B`
- `Unbabel/M-Prometheus-7B`
- `sapienzanlp/Minerva-7B-instruct-v1.0`
- `google/gemma-3-4b-it`
- `google/gemma-3-12b-it`

⚠️ **Attenzione:** si sconsiglia l'uso di modelli *Gemma* di piccole dimensioni, in quanto potrebbero:
- ignorare il *system prompt*
- generare risposte allucinate

Nel caso in cui il modello fallisca (es. allucinazioni tali da rendere impossibile l'estrazione di un output valido, come ad esempio in `red`), verrà restituito un errore **HTTP 500**. Assicurarsi di gestire questo caso nel frontend o nel client.

---

# 🔐 Configurazione ambiente
Nel file `.env`, specificare le seguenti variabili:

```env
HF_TOKEN=your_huggingface_token
MODEL_ID=nome_del_modello
QUANT=quantization_type
PORT=port_number
```

Il campo `PORT` se non specificato, usa di default la 8071.

Dove `QUANT` può essere:
- `"4bit"` - Quantizzazione a 4 bit
- `"8bit"` - Quantizzazione a 8 bit  
- `"gptq"` - Quantizzazione GPTQ
- `None` - Nessuna quantizzazione

Esempio:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxx
MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
QUANT=4bit
PORT=8080
```

In caso di esecuzione su CPU usare `"gptq"` e i modelli relativi per usufruire della quantizzazione:
- `shuyuej/Llama-3.2-1B-Instruct-GPTQ`
- `shuyuej/Llama-3.2-3B-Instruct-GPTQ`
- `ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g`

⚠️ **Nota per esecuzione SOLO su CPU:** nel file `docker-compose.yml` rimuovere le seguenti righe da tutti i servizi: (Feedback da Lab Ing. Inf.)

```yaml
runtime: nvidia
environment:
   - NVIDIA_VISIBLE_DEVICES=all
```

Esempio:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxx
MODEL_ID=shuyuej/Llama-3.2-1B-Instruct-GPTQ
QUANT=gptq
```
---
# ⚙️ Requisiti GPU (CUDA) su WSL 2

Per usare CUDA con Docker all'interno di WSL 2 (Ubuntu 22.04 LTS), installare:

1. **CUDA per WSL 2:**  
   https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network

2. **NVIDIA Container Toolkit:**  
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

---

# 🚣 Avvio dei container

Per avviare l'ambiente (con build dei container):

```bash
docker compose up --build
```

Una volta avviati, tutti i servizi saranno accessibili su **http://localhost:8071** con i seguenti endpoint:

- **Orange (Tagging):** http://localhost:8071/orange
- **Yellow (Question Generation):** http://localhost:8071/yellow
- **Red (Evaluation):** http://localhost:8071/red
- **Green Validity:** http://localhost:8071/green_validity
- **Green Cultural:** http://localhost:8071/green_cultural
- **Cyan (Answer Generation):** http://localhost:8071/cyan
- **Magenta (Humanization):** http://localhost:8071/magenta
- **Green Coherence QT:** http://localhost:8071/green_coherence_QT
- **Green Coherence QA:** http://localhost:8071/green_coherence_QA

---

# 🌐 Endpoint REST (POST)

## Orange - Tagging
- **Endpoint:** `http://localhost:8071/orange`
- **Input JSON:** `{"question": "string"}`
- **Output JSON:** `{"tags": "string", "raw": "string"}`

## Yellow - Question Generation
- **Endpoint:** `http://localhost:8071/yellow`
- **Input JSON:** `{"argument": "string"}`
- **Output JSON:** `{"question_generated": "string", "raw": "string"}`

## Red - Evaluation
- **Endpoint:** `http://localhost:8071/red`
- **Input JSON:** `{"question": "string", "answer": "string", "feedback": "string"}`
- **Output JSON:** `{"score": integer, "raw": "string"}`


## Cyan - Answer Generation
- **Endpoint:** `http://localhost:8071/cyan`
- **Input JSON:** `{"argomento": "string", "livello": "string"}`
- **Output JSON:** `{"risposta": "string", "raw": "string"}`

## Magenta - Humanization
- **Endpoint:** `http://localhost:8071/magenta`
- **Input JSON:** `{"llm_response": "string", "level": "string"}`
- **Output JSON:** `{"humanized_response": "string", "raw": "string"}`

## Green Coherence QT - Question-Theme Evaluation
- **Endpoint:** `http://localhost:8071/green_coherence_QT`
- **Input JSON:** `{"question": "string", "theme": "string"}`
- **Output JSON:** `{"bool": "string", "raw": "string"}`

## Green Coherence QA - Question-Answer Evaluation
- **Endpoint:** `http://localhost:8071/green_coherence_QA`
- **Input JSON:** `{"question": "string", "answer": "string"}`
- **Output JSON:** `{"bool": "string", "raw": "string"}`

## Green Validity - Evaluation
- **Endpoint:** `http://localhost:8071/green_validity`
- **Input JSON:** `{"question": "string", "answer": "string"}`
- **Output JSON:** `{"score": integer, "feedback": "string", "raw": "string"}`

## Green Cultural - Evaluation
- **Endpoint:** `http://localhost:8071/green_cultural`
- **Input JSON:** `{"question": "string"}`
- **Output JSON:** `{"score": integer, "feedback": "string", "raw": "string"}`