#!/usr/bin/env python3
"""
Few‑Shot Agent Server – Valutazione di Validità delle Risposte 
================================================================

Questo server FastAPI Supporta due modalità:
1. Role-based prompting: per modelli moderni che supportano conversazioni (Gemma, LLaMA, etc.)
2. Legacy prompting: per modelli più vecchi usando LangChain

Il server carica esempi da un file JSON e li usa per guidare il modello
nella scrittura mediante un livello.
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

# ---------------------------------------------------------------------------
# PIPELINE ------------------------------------------------------------------

def create_pipeline(model, tokenizer, device: str, max_new_tokens: int = 300):
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
    # Configurazione generazione: deterministica (no sampling) per coerenza
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Generazione deterministica
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Parametri pipeline
    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,  # Ritorna solo il testo generato, non il prompt
        **gen_cfg,
    )
    
    # Specifica device solo se il modello non ha già device_map
    if not hasattr(model, "hf_device_map"):
        pipe_kwargs["device"] = 0 if device == "cuda" else -1
    
    print(f"🔧 Pipeline pronta – device={device}, max_new_tokens={max_new_tokens}")
    return transformers.pipeline(**pipe_kwargs)

# ---------------------------------------------------------------------------
# PROMPTING FUNCTIONS -------------------------------------------------------

# Prompt di sistema che definisce il compito e i criteri di valutazione
SYSTEM_PROMPT = """
Sei un giudice esperto incaricato di valutare se le domande poste siano culturalmente rilevanti e, in tal caso, attribuire un punteggio e una spiegazione basato sui seguenti criteri:

CRITERI DI VALUTAZIONE
	•	Chiarezza: La domanda è formulata in modo comprensibile e diretto.
	•	Formulazione: La domanda è espressa chiaramente, senza ambiguità e con una struttura ben definita.
	•	Complessità: La domanda richiede una risposta specifica, dettagliata e articolata.

ISTRUZIONI DI VALUTAZIONE
	•	Se la domanda è personale (es. “Qual è il mio colore preferito?”) restituisci automaticamente il punteggio 0.
	•	Se la domanda è culturalmente rilevante, assegna un punteggio da 0 a 10 in base alla qualità dei criteri sopra indicati:

SCALA DI VALUTAZIONE
	•	0 – Domanda personale/non culturalmente rilevante. Es.: “Qual è il mio colore preferito?”
	•	1–2 – Inadeguata: Domanda mal posta, estremamente generica, poco comprensibile e senza valore informativo rilevante.
	•	3–4 – Insufficiente: Domanda poco chiara o ambigua, con bassa specificità. Formulazione o contesto poco curati.
	•	5–6 – Sufficiente: Domanda comprensibile e culturalmente pertinente, ma migliorabile in chiarezza o profondità. Complessità moderata.
	•	7–8 – Buona: Domanda chiara, ben strutturata e con un buon grado di complessità. Richiede risposta specifica.
	•	9–10 – Eccellente: Domanda molto ben formulata, chiara, culturalmente rilevante e con alta complessità. Richiede risposta dettagliata, precisa e articolata.

FORMATO OUTPUT
Utilizza SEMPRE E SOLTANTO CON IL PUNTEGGIO E il seguente formato per rispondere:
 
"Feedback: <spiegazione breve della valutazione, 2-3 frasi>\n
Punteggio: <numero da 1 a 5>"

CI DEVE STARE IL PUNTEGGIO E IL FEEDBACK! SEMPRE!!!!
ESEMPI:
"""

def create_few_shot_messages(examples: List[Dict]) -> List[Dict[str, str]]:
    """
    Pre-costruisce i messaggi few-shot all'avvio del server per modelli role-based.
    Questo evita di ricreare gli esempi ad ogni richiesta.
    
    Returns:
        Lista di messaggi con ruoli alternati system → user → assistant → user → assistant...
    """
    # Inizia con il system prompt che definisce il compito
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Aggiungi fino a 8 esempi come conversazioni simulate
    # Ogni esempio diventa una coppia user/assistant
    for ex in examples[:8]:  # Limita a 8 esempi per evitare overflow
        # Costruisce il messaggio utente con argomento e livello di dettaglio
        user_content = f"NUOVA VALUTAZIONE: \nDomanda: {ex['question']}"
        # Aggiunge la "domanda" dell'utente e la "risposta" del modello (il punteggio)
        assistant_content = (f"Feedback: {ex.get('feedback')}\n"
                        f"Punteggio: {str(ex['score'])}")
        messages.append({"role": "user", "content": user_content})
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
    for ex in examples[:8]:  # Limita a 8 esempi per evitare overflow
        parts.append(
            f"Domanda: {ex['question']}\n"
            f"Feedback: {ex['feedback']}\n"
            f"Punteggio: {ex['score']}"
        )

    
    # Unisce tutto in un unico prompt
    base_prompt = "\n\n".join(parts)
    
    # Crea template con placeholder per nuova valutazione
    template = base_prompt + "\n\nNUOVA VALUTAZIONE:\Domanda: {question}\nPunteggio: ?\nFeedback: ? "
    vars = ["question"]
    
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
        "gemma", "llama", "mistral", "mixtral", "qwen", "phi", "minerva",
        "chatglm", "baichuan", "internlm", "yi", "deepseek"
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
app = FastAPI(title="Objectivity Evaluator")

# Modelli Pydantic per request/response
class EvalRequest(BaseModel):
    question: str  # Argomento da valutare

class EvalResponse(BaseModel):
    score: int  # Risposta del modello 
    feedback: str # Spiegazione della valutazione
    raw: str    # Output grezzo del modello

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="question obbligatorio")


    if USE_ROLE_BASED:

        messages = base_messages.copy()  # Copia dei messaggi pre-costruiti

        user_content = f"NUOVA VALUTAZIONE: \nDomanda: {req.question}"
        messages.append({"role": "user", "content": user_content})

        formatted_prompt = format_messages_for_model(messages, tokenizer)

        outputs = pipeline(formatted_prompt)
        out = outputs[0]["generated_text"].strip()  # Prende il testo generato
    else:

        inputs = {
            "question": req.question
        }
        # chain.invoke() restituisce un dict {"text": "..."}
        out_raw = chain.invoke(inputs)
        if isinstance(out_raw, dict):
            out_text = out_raw.get("text", next(iter(out_raw.values())))
        else:
            out_text = out_raw
        out = out_text.strip()
    feedback_match = re.search(r"Feedback:\s*(.*)", out, re.IGNORECASE)
    feedback = feedback_match.group(1) if feedback_match else ""
    nums = re.findall(r"\d+", out)
    if not nums:
        raise HTTPException(status_code=500, detail=f"Nessun numero trovato nell'output: {out}")
    score = int(nums[-1])
    if score < 0 or score > 10:
        raise HTTPException(status_code=500, detail=f"Punteggio fuori range: {score}")
    print(f"✅ {req.question!r} -> {score}")
    return EvalResponse(score=score, feedback=feedback, raw=out)

if __name__ == "__main__":
    import uvicorn
    # Passiamo direttamente l'oggetto `app` per evitare una seconda importazione
    # del modulo che causava un doppio caricamento del modello.
    uvicorn.run(app, host="0.0.0.0", port=8072, log_level="info")
