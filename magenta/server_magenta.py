#!/usr/bin/env python3

# ---------------- PATCH anti‑Triton ------------------
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"        # blocca Torch Inductor
# os.environ["TORCHINDUCTOR_DISABLED"] = "1"     # alias legacy (torch<=2.3)
# ----------------------------------------------------

import json, pathlib, re, sys, warnings
from typing import Optional, Tuple, List, Dict

import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Carica le variabili d'ambiente da .env
load_dotenv()

# (facoltativo) disattiva kernel Triton di SDPA; decommenta se necessario
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(False)

# ---------------------------------------------------------------------------
# Log meno verboso
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# LangChain import (compat per <0.2)
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

def create_pipeline(model, tokenizer, device: str, max_new_tokens: int = 1024):
    """
    Crea una pipeline HuggingFace per text generation.
    
    Args:
        model: Il modello caricato
        tokenizer: Il tokenizer associato
        device: Device su cui eseguire ("cuda", "mps", "cpu")
        max_new_tokens: Numero massimo di token da generare
    
    Returns:
        Pipeline configurata per text generation
    """
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,  # Abilitato per generazione più varia nell'umanizzazione
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, # Vogliamo solo il testo generato
        **gen_cfg,
    )
    
    # Se il modello NON ha hf_device_map, possiamo indicare il device esplicitamente
    if not hasattr(model, "hf_device_map"):
        if device == "cuda":
            pipe_kwargs["device"] = 0 # Prima GPU disponibile
        elif device == "mps":
             # Per MPS, la pipeline di solito rileva il device del modello se è già su MPS
             pass
        else: # CPU
            pipe_kwargs["device"] = -1 # Indica CPU
            
    print(f"🔧 Pipeline pronta – device={device}, max_new_tokens={max_new_tokens}")
    return transformers.pipeline(**pipe_kwargs)

# ---------------------------------------------------------------------------
# PROMPTING FUNCTIONS -------------------------------------------------------

# System prompt per l'umanizzazione
SYSTEM_PROMPT_HUMANIZE = """Sei un esperto redattore con il compito di "umanizzare" il testo generato da un modello linguistico (LLM).
Il tuo obiettivo è trasformare una risposta tipica di un LLM in un testo che sembri scritto da un umano, rendendolo più naturale, meno formale e meno ripetitivo.

LIVELLI DI UMANIZZAZIONE (da 1 a 4):
- 1: Minima. Correzioni lievi, rimozione di frasi palesemente da AI (es. "Come modello linguistico..."). Mantenimento del tono formale se presente.
- 2: Moderata. Introduzione di lievi variazioni stilistiche, uso di contrazioni, linguaggio leggermente più colloquiale ma ancora professionale.
- 3: Significativa. Riformulazione più profonda, uso di un tono più personale (ma senza aggiungere opinioni non richieste), maggiore fluidità, frasi più dirette e meno verbose.
- 4: Massima. Trasformazione completa per assomigliare a un testo scritto da un umano. Evita cliché da AI, usa un linguaggio più vivace e diretto, cerca di usare il meno possibile elenchi. Cerca di abbreviare le frasi lunghe.

COSA FARE (in base al livello richiesto):
- Varia la struttura delle frasi.
- Usa contrazioni appropriate al contesto e al livello.
- Rimuovi frasi introduttive/conclusive tipiche degli LLM (es. "Certamente, ecco...", "Spero questo sia d'aiuto.").
- Semplifica il linguaggio se eccessivamente accademico o prolisso, senza perdere informazioni chiave.
- Rendi il tono più diretto e meno passivo.
- NON inventare fatti nuovi o cambiare il significato principale della risposta.
- NON inserire opinioni personali se non esplicitamente richieste dal contesto di umanizzazione o presenti nell'originale.
- NON usare emoji.
Fornisci SOLO la risposta umanizzata. NON aggiungere spiegazioni o commenti sulla tua risposta."""

def create_few_shot_messages(examples: List[Dict]) -> List[Dict[str, str]]:
    """
    Pre-costruisce i messaggi few-shot per modelli role-based.
    
    Args:
        examples: Lista di esempi dal file JSON con campi 'llm_response', 'level', 'humanized_response'
    
    Returns:
        Lista di messaggi con ruoli alternati system → user → assistant → user → assistant...
    """
    # Inizia con il system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT_HUMANIZE}]
    
    # Aggiungi fino a 8 esempi come conversazioni simulate
    for ex in examples[:8]:
        # Costruisce il messaggio utente con risposta LLM e livello
        user_content = f"Risposta LLM Originale:\n{ex['llm_response']}\nLivello di Umanizzazione: {ex['level']}"
        
        # Aggiunge la "richiesta" dell'utente e la "risposta" del modello (testo umanizzato)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ex['humanized_response']})
    
    return messages

def format_messages_for_model(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Formatta i messaggi secondo il template del modello specifico.
    
    Args:
        messages: Lista di messaggi con ruoli
        tokenizer: Tokenizer che potrebbe avere un metodo apply_chat_template
    
    Returns:
        Stringa formattata pronta per il modello
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        # Usa il template nativo del tokenizer (metodo preferito)
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

def build_chain_humanize(llm, examples: List[Dict]) -> LLMChain:
    """
    Crea una chain LangChain per modelli legacy che non supportano role-based prompting.
    
    Args:
        llm: Oggetto LLM di LangChain
        examples: Lista esempi
    
    Returns:
        LLMChain configurata per umanizzazione
    """
    parts = [SYSTEM_PROMPT_HUMANIZE + "\n\nESEMPI:"]
    
    # Usa massimo 8 esempi per non sovraccaricare il prompt
    for ex in examples[:8]:
        parts.append(
            f"Risposta LLM Originale:\n{ex['llm_response']}\n"
            f"Livello di Umanizzazione: {ex['level']}\n"
            f"Risposta Umanizzata:\n{ex['humanized_response']}"
        )
    
    base_prompt = "\n\n".join(parts)
    template = (
        base_prompt + 
        "\n\nNUOVA UMANIZZAZIONE:"
        "\nRisposta LLM Originale:\n{llm_response}\n"
        "Livello di Umanizzazione: {level}\n"
        "Risposta Umanizzata:" # Il modello completerà da qui
    )
    input_vars = ["llm_response", "level"]
    
    prompt = PromptTemplate(input_variables=input_vars, template=template)
    print(f"📋 Prompt per umanizzazione costruito con {len(examples[:8])} esempi.")
    return LLMChain(llm=llm, prompt=prompt)

def is_text_generation_model(model_id: str, tokenizer) -> bool:
    """
    Determina se il modello supporta role-based prompting (conversazioni).
    
    Args:
        model_id: Nome del modello
        tokenizer: Tokenizer caricato
    
    Returns:
        True se supporta role-based, False per usare legacy
    """
    # Lista di famiglie di modelli che sicuramente supportano chat
    role_based_models = [
        "gemma", "llama", "mistral", "mixtral", "qwen", "phi", "minerva",
        "chatglm", "baichuan", "internlm", "yi", "deepseek"
    ]
    
    # Controlla se il nome contiene una famiglia nota
    model_id_lower = model_id.lower()
    for model_name in role_based_models:
        if model_name in model_id_lower:
            return True
    
    # Controlla se il tokenizer ha il metodo apply_chat_template
    if hasattr(tokenizer, 'apply_chat_template'):
        return True
    
    # Default: usa il metodo legacy per sicurezza
    return False

# ---------------------------------------------------------------------------
# FASTAPI APP ---------------------------------------------------------------

# Carica gli esempi da file JSON
EXAMPLES_PATH = pathlib.Path("examples.json")
if not EXAMPLES_PATH.exists():
    print("❌ examples.json non trovato – crealo prima di avviare il server con esempi per l'umanizzazione.")
    print("""Formato atteso per examples.json:
[
  {
    "llm_response": "Testo originale dell'LLM...",
    "level": 3,
    "humanized_response": "Testo umanizzato corrispondente..."
  },
  ...
]""")
    sys.exit(1)

try:
    examples_data = json.loads(EXAMPLES_PATH.read_text("utf8"))
    # Validazione base degli esempi
    if not isinstance(examples_data, list) or not all(
        isinstance(ex, dict) and
        "llm_response" in ex and
        "level" in ex and
        "humanized_response" in ex and
        1 <= ex.get("level", 0) <= 4
        for ex in examples_data
    ):
        print("❌ Formato examples.json non valido. Deve essere una lista di dizionari con 'llm_response', 'level' (1-4), 'humanized_response'.")
        sys.exit(1)
    print(f"✅ Caricati {len(examples_data)} esempi di umanizzazione da {EXAMPLES_PATH}")
except json.JSONDecodeError as e:
    print(f"❌ Errore di parsing JSON in {EXAMPLES_PATH}: {e}")
    sys.exit(1)

# Carica modello e tokenizer
model_id_env = os.getenv("MODEL_ID", "google/gemma-3-4b-it") # Default se non specificato
print(f"🔧 Utilizzo del modello: {model_id_env}")

model, tokenizer, device = load_model_and_tokenizer(model_id_env)

# Determina quale metodo di prompting usare
USE_ROLE_BASED = is_text_generation_model(model_id_env, tokenizer)
print(f"🎯 Modalità prompting: {'Role-based' if USE_ROLE_BASED else 'Legacy'}")

# Configura il sistema in base al tipo di modello
if USE_ROLE_BASED:
    # Modelli moderni: usa pipeline diretta e pre-costruisce messaggi
    pipeline = create_pipeline(model, tokenizer, device)
    
    # Pre-costruisce i messaggi con system prompt ed esempi
    base_messages = create_few_shot_messages(examples_data)
    print(f"📚 Pre-costruiti {len(base_messages)} messaggi con system prompt ed esempi")
    
    llm_pipeline = None  # Non serve LangChain
    humanize_chain = None
else:
    # Modelli legacy: usa LangChain
    llm_pipeline = HuggingFacePipeline(pipeline=create_pipeline(model, tokenizer, device))
    humanize_chain = build_chain_humanize(llm_pipeline, examples_data)
    base_messages = None
    pipeline = None

# Crea app FastAPI
app = FastAPI(title="Gemma LLM Response Humanizer")

class HumanizeRequest(BaseModel):
    llm_response: str = Field(..., description="La risposta originale dell'LLM da umanizzare.")
    level: int = Field(..., ge=1, le=4, description="Livello di umanizzazione (1=min, 4=max).")

class HumanizeResponse(BaseModel):
    humanized_response: str
    raw_model_output: str # Per debug o analisi dell'output grezzo

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_endpoint(req: HumanizeRequest):
    """
    Endpoint principale che umanizza una risposta LLM.
    
    Riceve una risposta LLM e un livello di umanizzazione (1-4),
    restituisce il testo trasformato per sembrare più umano.
    """
    if not req.llm_response:
        raise HTTPException(status_code=400, detail="Il campo 'llm_response' è obbligatorio.")
    
    print(f"🔄 Richiesta di umanizzazione (livello {req.level}): {req.llm_response[:100]}...")

    try:
        if USE_ROLE_BASED:
            # === MODALITÀ ROLE-BASED ===
            # Copia i messaggi pre-costruiti (system + esempi)
            messages = base_messages.copy()
            
            # Aggiunge SOLO la nuova richiesta da umanizzare
            user_content = f"Risposta LLM Originale:\n{req.llm_response}\nLivello di Umanizzazione: {req.level}"
            messages.append({"role": "user", "content": user_content})
            
            # Formatta secondo il template del modello
            formatted_prompt = format_messages_for_model(messages, tokenizer)
            
            # Genera la risposta umanizzata
            outputs = pipeline(formatted_prompt)
            humanized_text = outputs[0]['generated_text'].strip()
            
        else:
            # === MODALITÀ LEGACY CON LANGCHAIN ===
            inputs = {
                "llm_response": req.llm_response,
                "level": req.level,
            }

            out_raw = humanize_chain.invoke(inputs) 
            
            if isinstance(out_raw, dict):
                # Cerca la chiave 'text' o la prima stringa nel dizionario
                humanized_text = out_raw.get("text", next((v for v in out_raw.values() if isinstance(v, str)), str(out_raw)))
            elif isinstance(out_raw, str):
                humanized_text = out_raw
            else:
                print(f"⚠️  Output del modello non gestito: {type(out_raw)}, {out_raw!r}")
                raise HTTPException(status_code=500, detail=f"Output del modello in formato inatteso: {type(out_raw)}")
            
            humanized_text = humanized_text.strip()

        # Salva output grezzo per debug
        raw_output = humanized_text
        
        # Pulizia finale dell'output
        humanized_text_cleaned = humanized_text.strip()

    except Exception as e:
        print(f"❌ Errore durante l'umanizzazione: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore del server durante l'umanizzazione: {str(e)}")

    print(f"✅ Risposta umanizzata: {humanized_text_cleaned[:100]}...")
    return HumanizeResponse(
        humanized_response=humanized_text_cleaned, 
        raw_model_output=raw_output
    )

if __name__ == "__main__":
    import uvicorn
    # Usa la porta specificata nel docstring del file (8070) come default
    port = int(os.getenv("PORT", 8074))
    # Passiamo direttamente l'oggetto `app` per evitare una seconda importazione
    # del modulo che causava un doppio caricamento del modello.
    print(f"🚀 Avvio server su http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")