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
```

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
```

In caso di esecuzione su CPU usare `"gptq"` e i modelli relativi per usufruire della quantizzazione:
- `shuyuej/Llama-3.2-1B-Instruct-GPTQ`
- `shuyuej/Llama-3.2-3B-Instruct-GPTQ`
- `ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g`

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

Per avviare l’ambiente (con build dei container):

```bash
docker compose up --build
```

---

# ❗ Nota su Red

Il servizio `red` parte **senza uso di feedback**.  
Se hai bisogno di log o output a schermo, **rimuovi l'opzione `--no-feedback`** nel rispettivo `Dockerfile`.

---

# 🌐 Endpoint REST (POST)

## Orange - Tagging
- **Endpoint:** `ip:8068/tag`
- **Input JSON:** `"question"`
- **Output JSON:** `"tags"`, `"raw_output"`, `"mode_used"`

## Yellow - Question Generation
- **Endpoint:** `ip:8069/generate_question`
- **Input JSON:** `"argument"`
- **Output JSON:** `"question_generated"`, `"raw_llm_output"`

## Red - Evaluation
- **Endpoint:** `ip:8070/evaluate`
- **Input JSON:** `"question"`, `"answer"`, `"feedback"` (opzionale)
- **Output JSON:** `"score"` (int), `"raw"`

## Green Validity - Evaluation
- **Endpoint:** `ip:8071/evaluate`
- **Input JSON:** `"question"`, `"answer"`
- **Output JSON:** `"score"` (int), `"feedback"`, `"raw"`

## Green Cultural - Evaluation
- **Endpoint:** `ip:8072/evaluate`
- **Input JSON:** `"question"`
- **Output JSON:** `"score"` (int), `"raw"`

## Cyan - Answer Generation
- **Endpoint:** `ip:8073/answer`
- **Input JSON:** `"argomento"`, `"livello"`
- **Output JSON:** `"risposta"`, `"raw"`

## Magenta - Humanization
- **Endpoint:** `ip:8074/humanize`
- **Input JSON:** `"llm_response"`, `"level"`
- **Output JSON:** `"humanized_response"`, `"raw_model_output"`

## Green Coherence QT - Question-Theme Evaluation
- **Endpoint:** `ip:8075/evaluate`
- **Input JSON:** `"question"`, `"theme"`
- **Output JSON:** `"bool"` (string "Vero" o "Falso"), `"raw"`

## Green Coherence QA - Question-Answer Evaluation
- **Endpoint:** `ip:8076/evaluate`
- **Input JSON:** `"question"`, `"answer"`
- **Output JSON:** `"bool"` (string "Vero" o "Falso"), `"raw"`
