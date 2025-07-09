#!/usr/bin/env python3
"""
Few‑Shot Agent Server – valutazione di oggettività via HTTP
================================================================

Questo server FastAPI valuta l'oggettività delle risposte usando modelli LLM.
Supporta due modalità:
1. Role-based prompting: per modelli moderni che supportano conversazioni (Gemma, LLaMA, etc.)
2. Legacy prompting: per modelli più vecchi usando LangChain

Il server carica esempi da un file JSON e li usa per guidare il modello
nella valutazione dell'oggettività su una scala 0-10.

Uso: python script.py [--no-feedback] per disabilitare l'uso dei feedback
"""

# ---------------- PATCH anti‑Triton ------------------
# Disabilita Torch Inductor per evitare problemi con Triton su alcune configurazioni
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"        # blocca Torch Inductor
# os.environ["TORCHINDUCTOR_DISABLED"] = "1"     # alias legacy (torch<=2.3)
# ----------------------------------------------------

import argparse
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
# GLOBAL CONFIG -------------------------------------------------------------
# Variabile globale per il flag no-feedback
USE_FEEDBACK = True

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

    # Carica il tokenizer
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
    else:  # device == "cpu"
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

def create_pipeline(model, tokenizer, device: str, max_new_tokens: int = 25):
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
SYSTEM_PROMPT_FEEDBACK = """Sei un esperto valutatore di oggettività delle risposte. 

Il tuo compito è valutare quanto sia oggettiva una risposta su una scala da 0 a 10. 
Per aiutarti riceverai anche un feedback umano sulla risposta in questione. Tieni a 
mente che POTREBBE ESSERE ERRATO, L'ULTIMA PAROLA DEVE ESSERE LA TUA.
Di seguito la scala che devi prendere come riferimento per le tue valutazioni:

SCALA DI VALUTAZIONE:
- 0-2: Completamente soggettiva (opinioni personali, giudizi di valore, preferenze)
- 3-4: Prevalentemente soggettiva (alcune opinioni con pochi fatti)
- 5-6: Mista (equilibrio tra fatti oggettivi e elementi soggettivi)
- 7-8: Prevalentemente oggettiva (fatti verificabili con alcune interpretazioni)
- 9-10: Completamente oggettiva (solo fatti verificabili, dati scientifici)

CRITERI DI VALUTAZIONE:
✓ Fatti verificabili = più oggettivo
✓ Dati scientifici/statistiche = più oggettivo
✓ Definizioni standard = più oggettivo
✗ Opinioni personali = più soggettivo
✗ Giudizi di valore = più soggettivo
✗ Preferenze individuali = più soggettivo
LA RISPOSTA DEVE RIFERIRSI ALLA DOMANDA POSTA!
SE LA RISPOSTA NON È COERENTE CON LA DOMANDA OPPURE NON SI RIFERISCE AD ESSA, IL PUNTEGGIO È 0.
Analizza ogni risposta e fornisci SOLO il numero del punteggio (0-10) NON RISPONDERE ALTRO."""

SYSTEM_PROMPT_NO_FEEDBACK = """Sei un esperto valutatore di oggettività delle risposte. 

Il tuo compito è valutare quanto sia oggettiva una risposta su una scala da 0 a 10. 
Di seguito la scala che devi prendere come riferimento per le tue valutazioni:

SCALA DI VALUTAZIONE:
- 0-2: Completamente soggettiva (opinioni personali, giudizi di valore, preferenze)
- 3-4: Prevalentemente soggettiva (alcune opinioni con pochi fatti)
- 5-6: Mista (equilibrio tra fatti oggettivi e elementi soggettivi)
- 7-8: Prevalentemente oggettiva (fatti verificabili con alcune interpretazioni)
- 9-10: Completamente oggettiva (solo fatti verificabili, dati scientifici)

CRITERI DI VALUTAZIONE:
✓ Fatti verificabili = più oggettivo
✓ Dati scientifici/statistiche = più oggettivo
✓ Definizioni standard = più oggettivo
✗ Opinioni personali = più soggettivo
✗ Giudizi di valore = più soggettivo
✗ Preferenze individuali = più soggettivo
LA RISPOSTA DEVE RIFERIRSI ALLA DOMANDA POSTA!
SE LA RISPOSTA NON È COERENTE CON LA DOMANDA OPPURE NON SI RIFERISCE AD ESSA, IL PUNTEGGIO È 0.
Analizza ogni risposta e fornisci SOLO il numero del punteggio (0-10) NON RISPONDERE ALTRO."""

def create_few_shot_messages(examples: List[Dict], use_feedback: bool = True) -> List[Dict[str, str]]:
    """
    Pre-costruisce i messaggi few-shot all'avvio del server per modelli role-based.
    Questo evita di ricreare gli esempi ad ogni richiesta.
    
    Args:
        examples: Lista di esempi dal file JSON con campi 'question', 'answer', 'score', 'feedback' (opzionale)
        use_feedback: Se True, usa i feedback quando disponibili
    
    Returns:
        Lista di messaggi con ruoli alternati system → user → assistant → user → assistant...
    """
    # Seleziona il system prompt appropriato
    SYSTEM_PROMPT = SYSTEM_PROMPT_FEEDBACK if use_feedback else SYSTEM_PROMPT_NO_FEEDBACK
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Aggiungi fino a 8 esempi come conversazioni simulate
    # Ogni esempio diventa una coppia user/assistant
    for ex in examples[:8]:
        # Costruisce il messaggio utente con domanda e risposta
        user_content = f"Domanda: {ex['question']}\nRisposta: {ex['answer']}"
        
        # Aggiunge feedback solo se use_feedback è True e il feedback esiste
        if use_feedback and ex.get("feedback"):
            user_content += f"\nFeedback: {ex['feedback']}"
        
        # Aggiunge la "domanda" dell'utente e la "risposta" del modello (il punteggio)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": str(ex['score'])})
    
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

def build_chain(llm, examples: list[dict], use_feedback: bool = True) -> LLMChain:
    """
    Crea una chain LangChain per modelli legacy che non supportano role-based prompting.
    Costruisce un unico prompt template con tutti gli esempi inline.
    
    Args:
        llm: Oggetto LLM di LangChain
        examples: Lista esempi
        use_feedback: Se includere il feedback negli esempi e nel template
    
    Returns:
        LLMChain configurata
    """
    # Seleziona il system prompt appropriato
    SYSTEM_PROMPT = SYSTEM_PROMPT_FEEDBACK if use_feedback else SYSTEM_PROMPT_NO_FEEDBACK
    parts = [SYSTEM_PROMPT + "\n\nESEMPI:"]
    
    # Aggiunge ogni esempio come testo formattato
    for ex in examples[:8]:
        if use_feedback and ex.get("feedback"):
            parts.append(
                f"Domanda: {ex['question']}\n"
                f"Risposta: {ex['answer']}\n"
                f"Feedback: {ex['feedback']}\n"
                f"Punteggio: {ex['score']}"
            )
        else:
            parts.append(
                f"Domanda: {ex['question']}\n"
                f"Risposta: {ex['answer']}\n"
                f"Punteggio: {ex['score']}"
            )
    
    # Unisce tutto in un unico prompt
    base_prompt = "\n\n".join(parts)
    
    # Crea template con placeholder per nuova valutazione
    if use_feedback:
        template = base_prompt + "\n\nNUOVA VALUTAZIONE:\nDomanda: {question}\nRisposta: {answer}\nFeedback: {feedback}\nPunteggio: "
        vars = ["question", "answer", "feedback"]
    else:
        template = base_prompt + "\n\nNUOVA VALUTAZIONE:\nDomanda: {question}\nRisposta: {answer}\nPunteggio: "
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
# COMMAND LINE ARGUMENTS ---------------------------------------------------

def parse_arguments():
    """
    Gestisce gli argomenti da command line.
    
    Returns:
        Namespace con gli argomenti parsati
    """
    parser = argparse.ArgumentParser(
        description="Few-Shot Agent Server per valutazione oggettività",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  python script.py                    # Modalità con feedback (default)
  python script.py --no-feedback      # Modalità senza feedback
        """
    )
    
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disabilita l'uso dei feedback negli esempi e nelle valutazioni"
    )
    
    return parser.parse_args()

# ---------------------------------------------------------------------------
# FASTAPI APP ---------------------------------------------------------------

def setup_app():
    """
    Configura e inizializza l'applicazione FastAPI.
    """
    global USE_FEEDBACK
    
    # Parse command line arguments
    args = parse_arguments()
    USE_FEEDBACK = not args.no_feedback
    
    print(f"🎯 Modalità feedback: {'ABILITATA' if USE_FEEDBACK else 'DISABILITATA'}")
    
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
        base_messages = create_few_shot_messages(examples, use_feedback=USE_FEEDBACK)
        print(f"📚 Pre-costruiti {len(base_messages)} messaggi con system prompt ed esempi")
        
        llm = None  # Non serve LangChain
        chain = None
    else:
        # Modelli legacy: usa LangChain
        llm = HuggingFacePipeline(pipeline=create_pipeline(model, tokenizer, device))
        chain = build_chain(llm, examples, use_feedback=USE_FEEDBACK)
        base_messages = None
    
    return examples, model, tokenizer, device, USE_ROLE_BASED, pipeline if USE_ROLE_BASED else None, llm, chain, base_messages

# Inizializza l'app
examples, model, tokenizer, device, USE_ROLE_BASED, pipeline, llm, chain, base_messages = setup_app()

# Crea app FastAPI
app = FastAPI(title="Objectivity Evaluator")

# Modelli Pydantic per request/response
class EvalRequest(BaseModel):
    question: str
    answer: str
    feedback: Optional[str] = None

class EvalResponse(BaseModel):
    score: int  # Punteggio 0-10
    raw: str    # Output grezzo del modello

@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    """
    Endpoint principale che valuta l'oggettività di una risposta.
    
    Riceve domanda e risposta, usa il modello per assegnare un punteggio 0-10.
    """
    global USE_FEEDBACK, USE_ROLE_BASED, pipeline, chain, base_messages
    
    # Validazione input
    if not req.question or not req.answer:
        raise HTTPException(status_code=400, detail="question e answer sono obbligatori")
    
    print(f"🔍 Valutazione richiesta: {req.question!r} - {req.answer!r}")
    if USE_FEEDBACK and req.feedback:
        print(f"🔍 feedback: {req.feedback!r}")

    if USE_ROLE_BASED:
        # === MODALITÀ ROLE-BASED ===
        # Copia i messaggi pre-costruiti (system + esempi)
        messages = base_messages.copy()
        
        # Aggiunge SOLO la nuova richiesta da valutare
        user_content = f"Domanda: {req.question}\nRisposta: {req.answer}"
        
        # Aggiunge feedback solo se abilitato e fornito
        if USE_FEEDBACK and req.feedback:
            user_content += f"\nFeedback: {req.feedback}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Formatta secondo il template del modello
        formatted_prompt = format_messages_for_model(messages, tokenizer)
        
        # Genera la risposta (dovrebbe essere solo un numero)
        outputs = pipeline(formatted_prompt)
        out = outputs[0]['generated_text'].strip()
    else:
        # === MODALITÀ LEGACY CON LANGCHAIN ===
        if USE_FEEDBACK:
            inputs = {
                "question": req.question,
                "answer": req.answer,
                "feedback": req.feedback or " ",
            }
        else:
            inputs = {
                "question": req.question,
                "answer": req.answer,
            }
        
        # LangChain restituisce un dict {"text": "..."}
        out_raw = chain.invoke(inputs)
        if isinstance(out_raw, dict):
            out_text = out_raw.get("text", next(iter(out_raw.values())))
        else:
            out_text = out_raw
        out = out_text.strip()

    print(f"🔍 Risposta del modello: {out!r}")

    # Estrae il punteggio numerico dalla risposta
    nums = re.findall(r"\d+", out)
    if not nums:
        raise HTTPException(status_code=500, detail=f"Nessun numero trovato nell'output: {out}")
    
    score = int(nums[0])  # Prende il primo numero trovato
    
    # Validazione range punteggio
    if score < 0 or score > 10:
        raise HTTPException(status_code=500, detail=f"Punteggio fuori range: {score}")
    
    print(f"✅ {req.question!r} -> {score}")
    return EvalResponse(score=score, raw=out)

# Entry point quando eseguito direttamente
if __name__ == "__main__":
    import uvicorn
    # Avvia il server FastAPI
    # Passa l'oggetto app direttamente per evitare doppio caricamento del modello
    uvicorn.run(app, host="0.0.0.0", port=8070, log_level="info")