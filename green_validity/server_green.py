#!/usr/bin/env python3
"""
Few‑Shot Agent Server – Valutazione di Validità delle Risposte via HTTP
================================================================
Questo server FastAPI valuta la validità delle di domande e risposte usando modelli LLM.
Supporta due modalità:
1. Role-based prompting: per modelli moderni che supportano conversazioni (Gemma, LLaMA, etc.)
2. Legacy prompting: per modelli più vecchi usando LangChain

Il server carica esempi da un file JSON e li usa per guidare il modello
nella valutazione della validità su una scala 1-5.
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
# Carica le variabili d'ambiente da .env
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
        return_full_text=False, # Ritorna solo il testo generato, non il prompt
        **gen_cfg,
    )
    # Specifica device solo se il modello non ha già device_map
    if not hasattr(model, "hf_device_map"):
        pipe_kwargs["device"] = 0 if device == "cuda" else -1
    print(f"🔧 Pipeline pronta – device={device}, max_new_tokens={max_new_tokens}")
    return transformers.pipeline(**pipe_kwargs)


# ---------------------------------------------------------------------------
# PROMPTING FUNCTIONS -------------------------------------------------------
SYSTEM_PROMPT = """Sei un giudice esperto incaricato di valutare la qualità delle risposte fornite a domande specifiche. Il tuo compito è analizzare ogni risposta secondo criteri ben definiti e assegnare un punteggio oggettivo con una spiegazione chiara.

TASK
Valuta le risposte alle domande secondo i criteri specificati e fornisci:
- Feedback: Una breve spiegazione della valutazione (2-3 frasi)
- Punteggio: Un punteggio da 1 a 5

SCALA DI VALUTAZIONE

5 - Eccellente: Risposta completamente corretta, molto pertinente, ricca di dettagli utili e esposta in modo chiaro e ben strutturato.

4 - Buona: Risposta corretta e pertinente con buon livello di dettaglio e chiarezza. Presenta solo minori imprecisioni o aree di miglioramento.

3 - Sufficiente: Risposta generalmente corretta e pertinente ma con lacune nei dettagli o nella chiarezza espositiva.

2 - Insufficiente: Risposta parzialmente corretta o con errori significativi. Problemi evidenti di pertinenza, dettaglio o chiarezza.

1 - Inadeguata: Risposta completamente errata, non pertinente, priva di dettagli utili o incomprensibile.

CRITERI DI VALUTAZIONE

CORRETTEZZA
Accuratezza fattuale e concettuale della risposta. Assenza di errori o informazioni fuorvianti.

PERTINENZA  
Quanto la risposta è rilevante e risponde effettivamente alla domanda posta, senza divagazioni inutili.

DETTAGLIO
Livello appropriato di approfondimento e completezza delle informazioni fornite rispetto alla complessità della domanda.

CHIAREZZA
Comprensibilità, organizzazione logica e qualità espositiva della risposta. Uso di linguaggio appropriato al contesto.

FORMATO OUTPUT
Il tuo output deve essere strutturato come segue:
"Feedback: <spiegazione breve della valutazione>\n
Punteggio: <numero da 1 a 5>"
(Rispondi usando SEMPRE il FORMATO OUTPUT indicato)

ISTRUZIONI OPERATIVE
- Analizza ogni criterio separatamente prima di assegnare il punteggio finale
- Il punteggio deve riflettere una valutazione complessiva bilanciata di tutti i criteri
- Sii obiettivo e coerente nelle valutazioni
- Nel feedback, menziona i punti di forza e le aree di miglioramento principali
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
        assistant_content = (f"Feedback: {ex.get('feedback')}\n"
                             f"Punteggio: {str(ex['score'])}")
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
    
    Returns:
        LLMChain configurata
    """
    # Costruisce il prompt con system prompt + esempi
    parts = [SYSTEM_PROMPT + "\n\nESEMPI:"]
    
    # Aggiunge ogni esempio come testo formattato    for ex in examples[:8]:
    parts.append(
        f"Domanda: {examples['question']}\n"
        f"Risposta: {examples['answer']}\n"
        f"Feedback: {examples['feedback']}\n"
        f"Punteggio: {examples['score']}"
    )
    # Unisce tutto in un unico prompt
    base_prompt = "\n\n".join(parts)
    # Crea template con placeholder per nuova valutazione
    template = base_prompt + "\n\nNUOVA VALUTAZIONE:\nDomanda: {question}\nRisposta: {answer}\nFeedback: ? \nPunteggio: ?"
    vars = ["question", "answer"]

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

model_id = os.getenv("MODEL_ID")
if not model_id:
    print("❌ MODEL_ID non trovato in .env")
    sys.exit(1)
hf_token = os.getenv("HF_TOKEN")

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
app = FastAPI(title="Validity Evaluator")

# Modelli Pydantic per request/response
class EvalRequest(BaseModel):
    question: str
    answer: str

class EvalResponse(BaseModel):
    score: int # punteggio da 1 a 5
    feedback: str # spiegazione della valutazione
    raw: str # output grezzo del modello

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    """
    Endpoint principale che valuta la validità di una risposta.
    
    Riceve domanda e risposta, usa il modello per assegnare un punteggio 1-5 e un feedback.
    """
    # Validazione input
    if not req.question or not req.answer:
        raise HTTPException(status_code=400, detail="question e answer sono obbligatori")


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
        # === MODALITÀ LEGACY CON LANGCHAIN ===
        inputs = {
            "question": req.question,
            "answer": req.answer,
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
    if score < 1 or score > 5:
        raise HTTPException(status_code=500, detail=f"Punteggio fuori range: {score}")
    print(f"✅ {req.question!r} -> {score}")
    return EvalResponse(score=score, feedback=feedback, raw=out)

if __name__ == "__main__":
    import uvicorn
     # Avvia il server FastAPI
    # Passa l'oggetto app direttamente per evitare doppio caricamento del modello
    uvicorn.run(app, host="0.0.0.0", port=8071, log_level="info")