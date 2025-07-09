#!/usr/bin/env python3
"""
Gemma Few‑Shot Agent Server – Generatore di Domande sulla Cultura Italiana via HTTP (Role-Based Enhanced)
=======================================================================================================
Espone un endpoint POST /generate_question (porta 8069) che, dato un ARGOMENTO (keyword),
restituisce una DOMANDA pertinente sulla cultura italiana.

Supporta due modalità di prompting:
1. Role-based prompting: per modelli moderni che supportano conversazioni.
2. Legacy prompting: per modelli più vecchi usando LangChain.

Il system-prompt è stato adattato per la generazione di domande.
La logica GPU/CPU è mantenuta, e TorchInductor/Triton sono disabilitati.

Prerequisiti:
1. Installare le dipendenze:
   pip install fastapi uvicorn "python-dotenv" transformers accelerate torch langchain langchain-core langchain-community
2. Creare un file `examples.json` con il seguente formato:
   [
     {
       "argument": "Cibo",
       "question_generated": "Quali sono i segreti per preparare un autentico ragù alla bolognese...?"
     },
     // ... altri esempi (argument -> question_generated)
   ]
3. (Opzionale) Creare un file `.env` per specificare il modello:
   MODEL_ID="google/gemma-1.1-2b-it" # Or any other compatible model

Utilizzo:
python NOME_SCRIPT_MODIFICATO.py
"""

# ---------------- PATCH anti‑Triton ------------------
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# ----------------------------------------------------

import json, pathlib, sys, warnings
from typing import Tuple, List, Dict, Optional # Added Optional

import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ============================================================================
# FUNZIONE 1: CARICAMENTO MODELLO E TOKENIZER (Adattata da script 1)
# ============================================================================
def load_model_and_tokenizer(
    model_id: str,
    quant: Optional[str] = "4bit"  # "4bit" | "8bit" | "gptq" | None
) -> Tuple[transformers.PreTrainedModel,
           transformers.PreTrainedTokenizer,
           str]:
    """
    Carica tokenizer + modello, provando la quantizzazione richiesta.
    Se la quantizzazione non è disponibile o fallisce, usa il modello
    non quantizzato e stampa l’esito.
    """
    # ───────────────────────── Device ─────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"💻 Device selezionato: {device}")

    # ─────────────────────── Tokenizer ────────────────────────
    print(f"📝 Caricamento tokenizer: {model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # ─────────────────── Config quantizzazione ─────────────────
    quant_cfg = None
    did_quantize = False

    # GPTQ: i pesi sono già quantizzati → nessuna cfg extra
    if quant == "gptq" or ".gptq" in model_id.lower():
        did_quantize = True
        print("✨ GPTQ rilevato/chiesto – il modello è già weight-only 4 bit")

    # BitsAndBytes 4/8-bit
    elif quant in {"4bit", "8bit"}:
        try:
            from transformers import BitsAndBytesConfig
            if quant == "4bit":
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:  # "8bit"
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            did_quantize = True
            print(f"✨ Quantizzazione {quant} attivata con bitsandbytes")
        except (ImportError, Exception) as e:
            warnings.warn(
                f"⚠️ BitsAndBytes non disponibile o errore: {e}\n"
                "   Procedo con modello non quantizzato."
            )
            quant_cfg = None
            did_quantize = False

    # ───────────────────── Caricamento modello ─────────────────
    print(f"🤖 Caricamento modello: {model_id}")
    common_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        quantization_config=quant_cfg,
    )

    # Se siamo su mps/CPU e quant_cfg è settata da bitsandbytes,
    # la ignoriamo perché bitsandbytes richiede CUDA.
    if device != "cuda":
        common_kwargs["quantization_config"] = None
        if quant_cfg is not None:
            print("⚠️  Quantizzazione richiesta ma non supportata su questo device; caricamento FP32.")
            did_quantize = False

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)

    if hasattr(model, "hf_device_map"):
        pass  # già posizionato con device_map="auto"
    else:
        model.to(device)

    model.eval()

    # ────────────────────── Messaggio finale ───────────────────
    if did_quantize:
        print("✅ Modello quantizzato caricato correttamente.")
    else:
        print("ℹ️  Modello caricato in precisione standard (no quantizzazione).")

    return model, tokenizer, device

# ============================================================================
# FUNZIONI DI PROMPTING (da script 1, adattate)
# ============================================================================

SYSTEM_PROMPT_QUESTION_GENERATION = """Sei un assistente AI specializzato nella cultura italiana, abile nel formulare domande intelligenti e pertinenti.
Il tuo compito è prendere un ARGOMENTO (una parola chiave o una breve frase) fornito dall'utente e generare una DOMANDA specifica e stimolante su quell'argomento, focalizzata sulla cultura italiana.
La domanda dovrebbe incoraggiare una riflessione o una spiegazione dettagliata, andando oltre la superficie. Evita domande banali o a cui si può rispondere con un sì/no.
""" # NOTA: Rimossi "ESEMPI:" qui, verranno aggiunti dinamicamente nel formato chat

def create_few_shot_messages_for_question_gen(examples: List[Dict]) -> List[Dict[str, str]]:
    """
    Pre-costruisce i messaggi few-shot per la generazione di domande.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT_QUESTION_GENERATION}]
    for ex in examples[:8]: # Max 8 examples
        user_content = f"Argomento: {ex['argument']}"
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(ex['question_generated'])})
    return messages

def format_messages_for_model(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Formatta i messaggi secondo il template del modello specifico.
    (Da script 1)
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback generico
        formatted = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system": formatted += f"Sistema: {content}\n\n"
            elif role == "user": formatted += f"Utente: {content}\n"
            elif role == "assistant": formatted += f"Assistente: {content}\n\n"
        return formatted.strip() # Ensure last newlines are stripped before generation prompt is added implicitly by some models

def is_text_generation_model(model_id: str, tokenizer) -> bool:
    """
    Determina se il modello supporta role-based prompting.
    (Da script 1)
    """
    role_based_models = [
        "gemma", "llama", "mistral", "mixtral", "qwen", "phi", "minerva",
        "chatglm", "baichuan", "internlm", "yi", "deepseek"
    ]
    model_id_lower = model_id.lower()
    if any(name in model_id_lower for name in role_based_models): return True
    if hasattr(tokenizer, 'apply_chat_template'): return True
    return False

# ============================================================================
# FUNZIONE 2: CREAZIONE PIPELINE (per entrambe le modalità)
# ============================================================================
def create_direct_transformers_pipeline(model, tokenizer, device: str, max_new_tokens: int = 275):
    """Crea una pipeline HuggingFace diretta per text generation (per role-based)."""
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True, "temperature": 0.75, "top_p": 0.9, # Creatività per domande
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
    }
    pipe_kwargs = {"task": "text-generation", "model": model, "tokenizer": tokenizer, "return_full_text": False, **gen_cfg}
    
    if not hasattr(model, "hf_device_map") or \
       (hasattr(model, "hf_device_map") and model.hf_device_map.get("") == "cpu"): # Check if model is explicitly on CPU
        pipe_kwargs["device"] = 0 if device == "cuda" else -1 # 0 for cuda:0, -1 for cpu
    
    print(f"🔧 Pipeline diretta generaz. domande: device hint {pipe_kwargs.get('device', 'accelerate')}, max_tokens={max_new_tokens}")
    return transformers.pipeline(**pipe_kwargs)

def create_legacy_langchain_pipeline_for_questions(model, tokenizer, device: str, max_new_tokens: int = 75):
    """Crea una HuggingFacePipeline di LangChain per la generazione di domande (per legacy mode)."""
    # Internamente usa una transformers.pipeline, quindi la configurazione è simile
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True, "temperature": 0.75, "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
    }
    tf_pipe_kwargs = {"task": "text-generation", "model": model, "tokenizer": tokenizer, "return_full_text": False, **gen_cfg}
    
    if not hasattr(model, "hf_device_map") or \
       (hasattr(model, "hf_device_map") and model.hf_device_map.get("") == "cpu"):
        tf_pipe_kwargs["device"] = 0 if device == "cuda" else -1
        
    tf_pipeline = transformers.pipeline(**tf_pipe_kwargs)
    print(f"🔧 Pipeline LangChain (via Transformers) generaz. domande: device hint {tf_pipe_kwargs.get('device', 'accelerate')}, max_tokens={max_new_tokens}")
    return HuggingFacePipeline(pipeline=tf_pipeline)

# ============================================================================
# FUNZIONE 3: COSTRUZIONE CATENA FEW‑SHOT (SOLO PER LEGACY MODE)
# ============================================================================
def build_legacy_question_generation_chain(llm, examples: List[Dict[str, str]]) -> LLMChain:
    """Costruisce una LLMChain di LangChain per la generazione di domande (legacy)."""
    # System prompt per legacy chain include "ESEMPI:"
    legacy_system_prompt = SYSTEM_PROMPT_QUESTION_GENERATION + "\n\nESEMPI:"
    parts = [legacy_system_prompt]
    for ex in examples[:8]: # Max 8 examples
        parts.append(
            f"Argomento: {ex['argument']}\n"
            f"Domanda Generata: {ex['question_generated']}"
        )
    
    base_prompt = "\n\n".join(parts)
    template = (
        base_prompt + 
        "\n\nNUOVA DOMANDA DA GENERARE:"
        "\nArgomento: {argument}\n"
        "Domanda Generata:" # Il modello completerà da qui
    )
    input_vars = ["argument"]
    prompt_template = PromptTemplate(input_variables=input_vars, template=template)
    return LLMChain(llm=llm, prompt=prompt_template)

# ============================================================================
# FASTAPI APP
# ============================================================================
print("🚀 Avvio server Generatore di Domande sulla Cultura Italiana...")

EXAMPLES_PATH = pathlib.Path(os.getenv("EXAMPLES_FILE", "examples.json"))
if not EXAMPLES_PATH.exists():
    sys.exit(f"❌ File esempi '{EXAMPLES_PATH}' non trovato. Formato atteso: [{'argument': '...', 'question_generated': '...'}].")

try:
    examples_data = json.loads(EXAMPLES_PATH.read_text("utf8"))
    valid_examples = [ex for ex in examples_data if isinstance(ex, dict) and "argument" in ex and "question_generated" in ex]
    if not valid_examples: sys.exit(f"❌ Nessun esempio valido (con 'argument' e 'question_generated') in '{EXAMPLES_PATH}'.")
    examples = valid_examples
    print(f"✅ Caricati {len(examples)} esempi validi da '{EXAMPLES_PATH}'")
except Exception as e: sys.exit(f"❌ Errore caricamento/parsing esempi da '{EXAMPLES_PATH}': {e}")

DEFAULT_MODEL_ID = "google/gemma-1.1-2b-it" # Mantieni un default sensato
model_id_to_load = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
if model_id_to_load == DEFAULT_MODEL_ID and not os.getenv("MODEL_ID"): # Se stiamo usando il default perché MODEL_ID non era settato
    print(f"⚠️  MODEL_ID non specificato, usando default: {DEFAULT_MODEL_ID}")

# ---- Caricamento e configurazione differenziata ----
model, tokenizer, device = load_model_and_tokenizer(model_id_to_load)

USE_ROLE_BASED = is_text_generation_model(model_id_to_load, tokenizer)
print(f"🎯 Modalità prompting: {'Role-based' if USE_ROLE_BASED else 'Legacy (LangChain)'}")

max_tokens_q_gen = int(os.getenv("MAX_NEW_TOKENS_QUESTION", 75))

# Dichiarazione variabili per entrambi i rami
direct_pipeline_question_gen = None
base_messages_question_gen = None
legacy_question_chain = None

if USE_ROLE_BASED:
    direct_pipeline_question_gen = create_direct_transformers_pipeline(model, tokenizer, device, max_new_tokens=max_tokens_q_gen)
    base_messages_question_gen = create_few_shot_messages_for_question_gen(examples)
    print(f"📚 Pre-costruiti {len(base_messages_question_gen)} messaggi (role-based) per la generazione di domande.")
else:
    legacy_llm = create_legacy_langchain_pipeline_for_questions(model, tokenizer, device, max_new_tokens=max_tokens_q_gen)
    legacy_question_chain = build_legacy_question_generation_chain(legacy_llm, examples)
    print("🧱 Configurato LLMChain LangChain (legacy) per la generazione di domande.")

# --- App FastAPI ---
app = FastAPI(title="API Generatore Domande Cultura Italiana (Role-Based Enhanced)")

class QuestionGenerationRequest(BaseModel):
    argument: str

class QuestionGenerationResponse(BaseModel):
    question_generated: str
    raw_llm_output: str

@app.post("/generate_question", response_model=QuestionGenerationResponse)
def generate_italian_culture_question(req: QuestionGenerationRequest):
    if not req.argument or not req.argument.strip():
        raise HTTPException(status_code=400, detail="Il campo 'argument' è obbligatorio.")
    
    argument_clean = req.argument.strip()
    print(f"▶️ Argomento ricevuto: \"{argument_clean[:100]}\"")
    raw_output = ""

    try:
        if USE_ROLE_BASED:
            current_messages = base_messages_question_gen.copy()
            # Aggiungi l'input dell'utente corrente
            current_messages.append({"role": "user", "content": f"Argomento: {argument_clean}"})
            
            formatted_prompt = format_messages_for_model(current_messages, tokenizer)
            
            # Genera la domanda
            # Assicurati che direct_pipeline_question_gen sia inizializzato
            if direct_pipeline_question_gen is None:
                 raise RuntimeError("Direct pipeline non inizializzata per la modalità role-based.")
            outputs = direct_pipeline_question_gen(formatted_prompt)
            raw_output = outputs[0]['generated_text'].strip()
        else:
            # Modalità Legacy con LangChain
            inputs = {"argument": argument_clean}
            if legacy_question_chain is None:
                raise RuntimeError("Legacy chain non inizializzata per la modalità legacy.")
            
            # .invoke() è preferito per le nuove versioni di LangChain e restituisce un dict
            chain_output = legacy_question_chain.invoke(inputs) 
            if isinstance(chain_output, dict):
                # Trova la chiave contenente il testo generato (comunemente "text")
                raw_output = chain_output.get("text", next(iter(chain_output.values()), "")).strip()
            else: # Fallback per output stringa diretta (vecchie versioni o custom chain)
                raw_output = str(chain_output).strip()

        generated_question = raw_output # Già strip()pata
        print(f"💬 Domanda generata (primi 100 char): \"{generated_question[:100]}\"")
        return QuestionGenerationResponse(question_generated=generated_question, raw_llm_output=raw_output)

    except Exception as e:
        print(f"❌ Errore elaborazione: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore server durante la generazione: {str(e)}")

@app.get("/", summary="Endpoint di benvenuto")
def read_root():
    return {"message": "API Generatore Domande Cultura Italiana. Usa POST /generate_question."}

if __name__ == "__main__":
    import uvicorn
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "8069")) # Porta originale
    print(f"📡 Server in ascolto su http://{server_host}:{server_port}")
    uvicorn.run(app, host=server_host, port=server_port, log_level="info")