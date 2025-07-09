#!/usr/bin/env python3
"""
Few‑Shot Agent Server – Valutazione di Coerenza delle Domande rispetto a un Tema
================================================================
Questo server FastAPI valuta la coerenza di una domanda in relazione ad un tema usando modelli LLM.
Supporta due modalità:
1. Role-based prompting: per modelli moderni che supportano conversazioni (Gemma, LLaMA, etc.)
2. Legacy prompting: per modelli più vecchi usando LangChain

Il server carica esempi da un file JSON e li usa per guidare il modello
nella valutazione della coerenza. Restituisce Vero o Falso a seconda della coerenza
della domanda rispetto al tema specificato.
"""

# ---------------- PATCH anti‑Triton ------------------
# Disabilita Torch Inductor per evitare problemi con Triton su alcune configurazioni
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"        # blocca Torch Inductor
# os.environ["TORCHINDUCTOR_DISABLED"] = "1"     # alias legacy (torch<=2.3)
# ----------------------------------------------------

import json, pathlib, re, sys, warnings
from typing import Optional, Tuple, List, Dict


import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
# Carica le variabili d'ambiente da .env (es. MODEL_ID)

load_dotenv()

# (facoltativo) disattiva kernel Triton di SDPA; decommenta se necessario
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(False)

# ---------------------------------------------------------------------------
# Log meno verboso - nasconde avvisi deprecati e futuri

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# LangChain import (compatibilità per versioni <0.2)

try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------------------------------------------------------------------
# MODEL + TOKENIZER ---------------------------------------------------------

def load_model_and_tokenizer(model_id: str) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer, str]:
    """
    Carica il modello e il tokenizer da HuggingFace.
    
    Args:
        model_id: Identificatore del modello su HuggingFace (es. "google/gemma-3-1b-it")
    
    Returns:
        Tupla (model, tokenizer, device) dove device è "cuda", "mps" o "cpu"
    """
    # Rileva automaticamente il device disponibile
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    elif torch.cuda.is_available():
        device = "cuda"  # GPU NVIDIA
    else:
        device = "cpu"  # Fallback CPU
    print(f"💻 Device selezionato: {device}")

    print(f"📝 Caricamento tokenizer: {model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,  # Permette codice custom dal repo
        use_fast=True,           # Usa tokenizer Rust se disponibile
        padding_side="left"      # Padding a sinistra per generazione    
    )
    # Aggiunge pad_token se mancante (alcuni modelli non ce l'hanno)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    # Carica il modello con configurazioni ottimizzate per device
    print(f"🤖 Caricamento modello: {model_id}")

    if device == "cuda":
        # Configurazione ottimizzata per GPU NVIDIA
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",          # Distribuisce automaticamente su GPU multiple
            torch_dtype=torch.bfloat16, # Precisione ridotta per minore VRAM
            low_cpu_mem_usage=True,     # Riduce uso RAM durante caricamento
            ignore_mismatched_sizes=True # Ignora mismatch dimensioni tensori

        )
        # NON chiamare .to() con device_map="auto" → errore
    elif device == "mps":
        # Configurazione per Apple Silicon
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # MPS non supporta bfloat16
        )
        model.to("mps")
        
    else:
        # Configurazione per CPU
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,

        )
        model.to("cpu")
    # Imposta modalità evaluation (disabilita dropout etc.)
    model.eval()
    return model, tokenizer, device

# ---------------------------------------------------------------------------
# PIPELINE ------------------------------------------------------------------

def create_pipeline(model, tokenizer, device: str, max_new_tokens: int = 100):
    """
    Crea una pipeline HuggingFace per text generation.
    
    Args:
        model: Il modello caricato
        tokenizer: Il tokenizer associato
        device: Device su cui eseguire ("cuda", "mps", "cpu")
        max_new_tokens: Numero massimo di token da generare (default 25 per risposte brevi)
    
    Returns:
        Pipeline configurata per text generation
    """
    # Configurazione generazione: deterministica (no sampling) per coerenza
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False, # Generazione deterministica
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, # Restituisce solo il testo generato, non il prompt
        **gen_cfg,
    )
    # Specifica device solo se il modello non ha già device_map
    if not hasattr(model, "hf_device_map"):
        pipe_kwargs["device"] = 0 if device == "cuda" else -1
    print(f"🔧 Pipeline pronta – device={device}, max_new_tokens={max_new_tokens}")
    return transformers.pipeline(**pipe_kwargs)

# ---------------------------------------------------------------------------
# PROMPTING FUNCTIONS -------------------------------------------------------
SYSTEM_PROMPT = """ Sei un giudice che deve valutare se una domanda in linguaggio naturale è coerente con il tema fornito ed è semanticamente corretta.

CRITERI DI VALUTAZIONE

COERENZA TEMATICA
- La domanda deve essere direttamente pertinente al tema specificato
- Il contenuto della domanda deve rientrare nell'ambito del tema
- Non sono accettabili collegamenti forzati o troppo vaghi

CORRETTEZZA SEMANTICA
- La domanda deve essere grammaticalmente corretta
- Deve avere un senso logico e essere comprensibile
- Le parole devono essere utilizzate in modo appropriato al contesto

CHIAREZZA E SENSATEZZA
- La domanda deve essere formulata in modo chiaro
- Deve essere possibile comprendere cosa si sta chiedendo
- Non deve contenere contraddizioni interne

FORMATO OUTPUT
"Bool: <Vero/Falso>"
Rispondi solo "Vero" o "Falso". RISPETTA IL FORMATO. Non aggiungere altro testo.

Rispondi con questa logica:
- Vero se la domanda è coerente con il tema E semanticamente corretta
- Falso se la domanda NON è coerente con il tema OPPURE NON è semanticamente corretta
"""

def create_few_shot_messages(examples: List[Dict]) -> List[Dict[str, str]]:
    """
    Pre-costruisce i messaggi few-shot all'avvio del server per modelli role-based.
    Questo evita di ricreare gli esempi ad ogni richiesta.
    
    Args:
        examples: Lista di esempi dal file JSON con campi 'question', 'answer', 'score', 'feedback' (opzionale)
    
    Returns:
        Lista di messaggi con ruoli alternati system → user → assistant → user → assistant...
    """
    # Inizia con il system prompt che definisce il compito
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Aggiungi fino a 8 esempi come conversazioni simulate
    # Ogni esempio diventa una coppia user/assistant
    for ex in examples[:8]:
        # Costruisce il messaggio utente con domanda e tema
        user_content = f"Domanda: {ex['question']}\nTema: {ex['theme']}"
        
        # Aggiunge la "domanda" dell'utente e la "risposta" del modello (il punteggio)
        messages.append({"role": "user", "content": user_content})
        assistant_content = f"Bool: {ex['bool']}"
        messages.append({"role": "assistant", "content": assistant_content})  
    
    return messages

def format_messages_for_model(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Formatta i messaggi secondo il template del modello specifico.
    Ogni modello ha un formato diverso per i ruoli (es. Gemma usa un formato, LLaMA un altro).
    
    Args:
        messages: Lista di messaggi con ruoli
        tokenizer: Tokenizer che potrebbe avere un metodo apply_chat_template
    
    Returns:
        Stringa formattata pronta per il modello
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        # Usa il template nativo del tokenizer (metodo preferito)
        # add_generation_prompt=True aggiunge il prompt per far continuare il modello
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback generico se il tokenizer non ha template
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"Sistema: {content}\n\n"
            elif role == "user":
                formatted += f"Utente: {content}\n"
            elif role == "assistant":
                formatted += f"Assistente: {content}\n\n"
        return formatted

def build_chain(llm, examples: list[dict]) -> LLMChain:
    """
    Crea una chain LangChain per modelli legacy che non supportano role-based prompting.
    Costruisce un unico prompt template con tutti gli esempi inline.
    
    Args:
        llm: Oggetto LLM di LangChain
        examples: Lista esempi
        use_feedback: Se includere il feedback negli esempi
    
    Returns:
        LLMChain configurata
    """
    # Costruisce il prompt con system prompt + esempi
    parts = [SYSTEM_PROMPT + "\n\nESEMPI:"]
    
    # Aggiunge ogni esempio come testo formattato    
    for ex in examples[:8]:  # Limita a 8 esempi per evitare prompt troppo lunghi
        parts.append(
            f"Domanda: {ex['question']}\n"
            f"Tema: {ex['theme']}\n"
            f"Bool: {ex['bool']}\n"
        )
    # Unisce tutto in un unico prompt
    base_prompt = "\n\n".join(parts)
    # Crea template con placeholder per nuova valutazione
    template = base_prompt + "\n\nNUOVA VALUTAZIONE:\nDomanda: {question}\nTema: {theme}\nBool: ?"
    vars = ["question", "theme", "bool"]

    return LLMChain(llm=llm, prompt=PromptTemplate(input_variables=vars, template=template))

# ---------------------------------------------------------------------------
# FASTAPI APP ---------------------------------------------------------------
def is_text_generation_model(model_id: str, tokenizer) -> bool:
    """
    Determina se il modello supporta role-based prompting (conversazioni).Add commentMore actions
    I modelli moderni hanno template di chat, quelli vecchi no.
    
    Args:
        model_id: Nome del modello
        tokenizer: Tokenizer caricato
    
    Returns:
        True se supporta role-based, False per usare legacy
    """
    # Lista di famiglie di modelli che sicuramente supportano chat
    role_based_models = [
        "gemma", "llama", "mistral", "mixtral", "qwen", "phi", 
        "chatglm", "baichuan", "internlm", "yi", "deepseek", "minerva"
    ]
    
    # Controlla se il nome contiene una famiglia nota
    model_id_lower = model_id.lower()
    for model_name in role_based_models:
        if model_name in model_id_lower:
            return True
    
    # Controlla se il tokenizer ha il metodo apply_chat_template
    # Questo è il metodo più affidabile
    if hasattr(tokenizer, 'apply_chat_template'):
        return True
    
    # Default: usa il metodo legacy per sicurezza
    return False
# Carica gli esempi da file JSON
EXAMPLES_PATH = pathlib.Path("examples.json")
if not EXAMPLES_PATH.exists():
    print("❌ examples.json non trovato – crealo prima di avviare il server.")
    sys.exit(1)
examples = json.loads(EXAMPLES_PATH.read_text("utf8"))
# Carica modello e tokenizer dall'ID specificato in .env
model_id = os.getenv("MODEL_ID")

model, tokenizer, device = load_model_and_tokenizer(model_id)

# Determina quale metodo di prompting usare
USE_ROLE_BASED = is_text_generation_model(model_id, tokenizer)
print(f"🎯 Modalità prompting: {'Role-based' if USE_ROLE_BASED else 'Legacy'}")

# Configura il sistema in base al tipo di modello
if USE_ROLE_BASED:
    # Modelli moderni: usa pipeline diretta e pre-costruisce messaggi
    pipeline = create_pipeline(model, tokenizer, device)
    
    # Pre-costruisce i messaggi con system prompt ed esempi
    # Questi verranno riutilizzati per ogni richiesta
    base_messages = create_few_shot_messages(examples)
    print(f"📚 Pre-costruiti {len(base_messages)} messaggi con system prompt ed esempi")
    
    llm = None  # Non serve LangChain
    chain = None
else:
    # Modelli legacy: usa LangChain
    llm = HuggingFacePipeline(pipeline=create_pipeline(model, tokenizer, device))
    chain = build_chain(llm, examples, use_feedback=True)
    base_messages = None

# Crea app FastAPI
app = FastAPI(title="Coherence_Question_Theme Evaluator")
# Modelli Pydantic per request/response
class EvalRequest(BaseModel):
    question: str # Domanda da valutare
    theme: str # Tema di riferimento

class EvalResponse(BaseModel):
    bool: str # Risultato della valutazione: "Vero" o "Falso"
    raw: str # Risposta grezza del modello, utile per debug

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    """Add commentMore actions
    Endpoint principale che valuta la coerenza di una domanda rispetto ad un tema.
    
    Riceve domanda e un tema, usa il modello per assegnare Vero o Falso
    in base alla coerenza della domanda rispetto al tema.
    """
    # Validazione input
    if not req.question or not req.theme:
        raise HTTPException(status_code=400, detail="question e theme sono obbligatori")

    if USE_ROLE_BASED:
        # === MODALITÀ ROLE-BASED ===
        # Copia i messaggi pre-costruiti (system + esempi)
        messages = base_messages.copy()
        
        # Aggiunge SOLO la nuova richiesta da valutare
        user_content = f"Domanda: {req.question}\nTema: {req.theme}"
        messages.append({"role": "user", "content": user_content})
        
        # Formatta secondo il template del modello
        formatted_prompt = format_messages_for_model(messages, tokenizer)
        # Genera la risposta (dovrebbe essere solo un numero)
        outputs = pipeline(formatted_prompt)
        
        # Handle different output formats
        if isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], dict):
                out = outputs[0]['generated_text'].strip()
            else:
                out = str(outputs[0]).strip()
        elif isinstance(outputs, str):
            out = outputs.strip()
        else:
            out = str(outputs).strip()
            
        # Estrae il risultato booleano dalla risposta del modello
        bool_match = re.search(r"\b(Vero|Falso)\b", out, re.IGNORECASE)
        
        if not bool_match:
            raise HTTPException(
                status_code=422, 
                detail=f"Output non valido: '{out}'. Deve essere 'Vero' o 'Falso'"
            )
            
        bool_result = bool_match.group(1)
        return EvalResponse(bool=bool_result, raw=out)
        
    else:
        # === MODALITÀ LEGACY ===
        inputs = {
            "question": req.question,
            "theme": req.theme
        }   
        # chain.invoke() restituisce un dict {"text": "..."}
        out_raw = chain.invoke(inputs)
        if isinstance(out_raw, dict):
            out_text = out_raw.get("text", next(iter(out_raw.values())))
        else:
            out_text = out_raw
        out = out_text.strip()
        #Estrae il risultato booleano dalla risposta del modello
        bool_match = re.search(r"\b(Vero|Falso)\b", out, re.IGNORECASE)
        
        if not bool_match:
            raise HTTPException(
                status_code=422, 
                detail=f"Output non valido: '{out}'. Deve essere 'Vero' o 'Falso'"
            )
            
        bool_result = bool_match.group(1)
        return EvalResponse(bool = bool_result, raw = out)
# Entry point quando eseguito direttamente
if __name__ == "__main__":
    import uvicorn
    # Avvia il server FastAPI
    # Passa l'oggetto app direttamente per evitare doppio caricamento del modello
    uvicorn.run(app, host="0.0.0.0", port=8075, log_level="info")
