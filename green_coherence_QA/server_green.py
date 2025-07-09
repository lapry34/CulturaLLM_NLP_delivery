#!/usr/bin/env python3
"""
Questo server FastAPI valuta la coerenza di domande e risposte usando modelli LLM.
Supporta due modalità:
1. Role-based prompting: per modelli moderni che supportano conversazioni (Gemma, LLaMA, etc.)
2. Legacy prompting: per modelli più vecchi usando LangChain

Il server carica esempi da un file JSON e li usa per guidare il modello
nella valutazione della coerenza. L'output è Vero/Falso.
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

def load_model_and_tokenizer(
    model_id: str
) -> Tuple[transformers.PreTrainedModel,
           transformers.PreTrainedTokenizer,
           str]:
    
    quant = os.getenv("QUANT", None)

    # Check if this is a GPTQ model
    is_gptq_model = "gptq" in model_id.lower()
    
    if is_gptq_model:
        print("🔍 Rilevato modello GPTQ – saltando quantizzazione aggiuntiva")
    
    """
    Carica tokenizer + modello, provando la quantizzazione richiesta.
    Se la quantizzazione non è disponibile o fallisce, usa il modello
    non quantizzato e stampa l'esito.
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

    # Per i modelli GPTQ, non usare quantizzazione aggiuntiva
    if is_gptq_model:
        did_quantize = True
        print("✨ Modello GPTQ già quantizzato – no quantizzazione aggiuntiva")
    
    # BitsAndBytes 4/8-bit (solo su CUDA e solo per modelli non-GPTQ)
    elif quant in {"4bit", "8bit"} and device == "cuda":
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
    
    # Setup kwargs base
    common_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Configurazione device-specific
    if device == "cuda":
        common_kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        })
        # Aggiungi quantization_config solo se non è GPTQ
        if not is_gptq_model and quant_cfg is not None:
            common_kwargs["quantization_config"] = quant_cfg
    else:
        # CPU/MPS: configurazione semplificata
        common_kwargs.update({
            "torch_dtype": torch.float32,
        })

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

# ---------------------------------------------------------------------------
# PIPELINE ------------------------------------------------------------------

def create_pipeline(model, tokenizer, device: str, max_new_tokens: int = 20):
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
    # parametri pipeline
    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, # Restituisce solo il testo generato, non l'intero prompt
        **gen_cfg,
    )
    # Specifica device solo se il modello non ha già device_map
    if not hasattr(model, "hf_device_map"):
        pipe_kwargs["device"] = 0 if device == "cuda" else -1
    print(f"🔧 Pipeline pronta – device={device}, max_new_tokens={max_new_tokens}")
    return transformers.pipeline(**pipe_kwargs)


# ---------------------------------------------------------------------------
# PROMPTING FUNCTIONS -------------------------------------------------------
SYSTEM_PROMPT = """ Sei un giudice che valuta la coerenza tra domande e risposte. Data una domanda e una risposta, devi determinare se la risposta è coerente con la domanda posta e semanticamente sensata.

CRITERI DI VALUTAZIONE

COERENZA TEMATICA: La risposta deve essere pertinente al tema e al contesto della domanda posta.
CORRETTEZZA SEMANTICA: La risposta deve essere logicamente sensata e non contenere contraddizioni interne o affermazioni palesemente false.
COMPLETEZZA DELLA RISPOSTA: La risposta deve affrontare effettivamente ciò che viene chiesto nella domanda, non deve essere evasiva o fuori tema.
COMPRENSIONE DELLA DOMANDA: La risposta deve dimostrare una corretta interpretazione di ciò che viene richiesto.

FORMATO OUTPUT
"Bool: <Vero/Falso>"
Rispondi SOLO "Vero" o "Falso". RISPETTA IL FORMATO. NON AGGIUNGERE ALTRO TESTO.

Rispondi con questa logica:
- Vero se la risposta è coerente con la domanda e semanticamente sensata
- Falso se la risposta NON è coerente con la domanda OPPURE NON è semanticamente sensata
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
        # Costruisce il messaggio utente con domanda e risposta
        user_content = f"Domanda: {ex['question']}\nRisposta: {ex['answer']}"
        
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
    for ex in examples[:8]:
        parts.append(
            f"Domanda: {ex['question']}\n"
            f"Tema: {ex['answer']}\n"
            f"Bool: {ex['bool']}\n"
        )
    # Unisce tutto in un unico prompt
    base_prompt = "\n\n".join(parts)
    # Crea template con placeholder per nuova valutazione
    template = base_prompt + "\n\nNUOVA VALUTAZIONE:\nDomanda: {question}\nTema: {answer}\nBool: ?"
    vars = ["question", "answer", "bool"]

    return LLMChain(llm=llm, prompt=PromptTemplate(input_variables=vars, template=template))

def is_text_generation_model(model_id: str, tokenizer) -> bool:
    """
    Determina se il modello supporta role-based prompting (conversazioni).
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

# ---------------------------------------------------------------------------
# FASTAPI APP ---------------------------------------------------------------
# Carica gli esempi da file JSON
EXAMPLES_PATH = pathlib.Path("examples.json")
if not EXAMPLES_PATH.exists():
    print("❌ examples.json non trovato – crealo prima di avviare il server.")
    sys.exit(1)
examples = json.loads(EXAMPLES_PATH.read_text("utf8"))
# Carica modello e tokenizer dall'ID specificato in .env
model_id = os.getenv("MODEL_ID")
if not model_id:
    print("❌ MODEL_ID non trovato in .env")
    sys.exit(1)
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
app = FastAPI(title="Gemma Coherence_QA Evaluator")

class EvalRequest(BaseModel):
    question: str # Domanda da valutare
    answer: str # Risposta da valutare

# Modelli Pydantic per request/response
class EvalResponse(BaseModel):
    bool: str # Risultato della valutazione ("Vero" o "Falso")
    raw: str # Risposta grezza del modello, utile per debug

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    """
    Endpoint principale che valuta la coerenza di una domanda e risposta.
    
    Riceve domanda e risposta, usa il modello per assegnare Vero o Falso
    in base alla coerenza semantica.
    """
    # Validazione input
    if not req.question or not req.answer:
        raise HTTPException(status_code=400, detail="question e answer sono obbligatori")
    '''inputs = {
        "question": req.question,
        "answer": req.answer,
    }'''
    if USE_ROLE_BASED:
        # === MODALITÀ ROLE-BASED ===
        # Copia i messaggi pre-costruiti (system + esempi)
        messages = base_messages.copy()
        
        # Aggiunge SOLO la nuova richiesta da valutare
        user_content = f"Domanda: {req.question}\nRisposta: {req.answer}"
        messages.append({"role": "user", "content": user_content})
        
        # Formatta secondo il template del modello
        formatted_prompt = format_messages_for_model(messages, tokenizer)
        # Genera la risposta (dovrebbe essere solo un numero)
        outputs = pipeline(formatted_prompt)
        out = outputs[0]['generated_text'].strip()
    else:
        # === MODALITÀ LEGACY CON LANGCHAIN ===Add commentMore actions
        inputs = {
            "question": req.question,
            "answer": req.answer
        }
        
        # LangChain restituisce un dict {"text": "..."}
        out_raw = chain.invoke(inputs)
        if isinstance(out_raw, dict):
            out_text = out_raw.get("text", next(iter(out_raw.values())))
        else:
            out_text = out_raw
        out = out_text.strip()
    # Estrae il risultato booleano dalla risposta del modello
    bool_match = re.search(r"\b(Vero|Falso)\b", out, re.IGNORECASE)
    
    if not bool_match:
        raise HTTPException(
            status_code=422, 
            detail=f"Output non valido: '{out}'. Deve essere 'Vero' o 'Falso'"
        )
        
    bool_result = bool_match.group(1)
    return EvalResponse(bool = bool_result, raw = out)

if __name__ == "__main__":
    import uvicorn
    # Avvia il server FastAPI
    # Passa l'oggetto app direttamente per evitare doppio caricamento del modello
    uvicorn.run(app, host="0.0.0.0", port=8076, log_level="info")
