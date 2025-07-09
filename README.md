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
- generare risposte allucininate

Nel caso in cui il modello fallisca (es. allucinazioni tali da rendere impossibile l'estrazione di un output valido, come ad esempio in `red`), verrà restituito un errore **HTTP 500**. Assicurarsi di gestire questo caso nel frontend o nel client.

---

# 🔐 Configurazione ambiente

Nel file `.env`, specificare le seguenti variabili:

```env
HF_TOKEN=your_huggingface_token
MODEL_ID=nome_del_modello
```

Esempio:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxx
MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
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

Il servizio `red` parte **senza output di feedback**.  
Se hai bisogno di log o output a schermo, **rimuovi l’opzione `--no-feedback`** nel rispettivo `Dockerfile`.
