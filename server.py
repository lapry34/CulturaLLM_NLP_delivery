#!/usr/bin/env python3
"""
Unified LLM Server - Server unificato per tutti i task
======================================================

Un unico server che gestisce diversi compiti LLM basandosi su:
- System prompts caricati da file
- Esempi few-shot caricati da JSON
- Endpoint dinamici per ogni task
- Supporto per output multipli con regex dedicate

Struttura delle cartelle:
    tasks/
    ├── red/
    │   ├── system_prompt.txt
    │   ├── examples.json
    │   └── config.json (opzionale)
    ├── magenta/
    │   ├── system_prompt.txt
    │   ├── examples.json
    │   └── config.json (opzionale)
    └── ...

Esempio config.json con output multipli:
{
    "input_fields": ["question", "answer"],
    "outputs": {
        "score": {
            "extract_pattern": "Punteggio:\\s*(\\d+)",
            "type": "int"
        },
        "feedback": {
            "extract_pattern": "Feedback:\\s*(.+?)(?:\\n|$)",
            "type": "str"
        }
    },
    "max_new_tokens": 512
}

Uso: python unified_llm_server.py
"""

# ---------------- PATCH anti‑Triton ------------------
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# ----------------------------------------------------

import json
import pathlib
import re
import sys
import warnings
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
import traceback

import torch
import transformers
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv
import uvicorn

# Carica variabili d'ambiente
load_dotenv()

# Silenzia warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# LangChain imports per compatibilità legacy
try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------------------------------------------------------------------
# CONFIGURAZIONE GLOBALE
# ---------------------------------------------------------------------------

TASKS_DIR = pathlib.Path("tasks")  # Directory con le configurazioni dei task
MODEL_ID = os.getenv("MODEL_ID", "sapienzanlp/Minerva-7B-instruct-v1.0")
QUANT = os.getenv("QUANT", None)
PORT = int(os.getenv("PORT", 8071))

# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class OutputConfig:
    """Configurazione per un singolo output"""
    name: str
    extract_pattern: Optional[str] = None
    type: str = "str"  # "str", "int", "float", "bool"

@dataclass
class TaskConfig:
    """Configurazione per un singolo task"""
    name: str
    system_prompt: str
    examples: List[Dict[str, Any]]
    input_fields: List[str]  # Campi richiesti in input
    outputs: Dict[str, OutputConfig] = field(default_factory=dict)  # Output multipli
    max_new_tokens: int = 512
    # Legacy fields per retrocompatibilità
    output_field: Optional[str] = None
    extract_pattern: Optional[str] = None

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str) -> Tuple[Any, Any, str]:
    """Carica modello e tokenizer una sola volta"""
    
    is_gptq_model = "gptq" in model_id.lower()
    
    if is_gptq_model:
        print("🔍 Rilevato modello GPTQ – saltando quantizzazione aggiuntiva")
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"💻 Device selezionato: {device}")
    
    # Tokenizer
    print(f"📝 Caricamento tokenizer: {model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    
    # Quantization config
    quant_cfg = None
    did_quantize = False
    
    if is_gptq_model:
        did_quantize = True
        print("✨ Modello GPTQ già quantizzato")
    elif QUANT in {"4bit", "8bit"} and device == "cuda" and not is_gptq_model:
        try:
            from transformers import BitsAndBytesConfig
            if QUANT == "4bit":
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            did_quantize = True
            print(f"✨ Quantizzazione {QUANT} attivata")
        except Exception as e:
            print(f"⚠️ BitsAndBytes non disponibile: {e}")
    
    # Model loading
    print(f"🤖 Caricamento modello: {model_id}")
    
    common_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    if device == "cuda":
        common_kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        })
        if not is_gptq_model and quant_cfg:
            common_kwargs["quantization_config"] = quant_cfg
    else:
        common_kwargs["torch_dtype"] = torch.float32
    
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
    
    if not hasattr(model, "hf_device_map"):
        model.to(device)
    
    model.eval()
    
    print("✅ Modello caricato" + (" e quantizzato" if did_quantize else ""))
    
    return model, tokenizer, device

# ---------------------------------------------------------------------------
# PIPELINE CREATION
# ---------------------------------------------------------------------------

def create_pipeline(model, tokenizer, device: str, max_new_tokens: int = 512):
    """Crea pipeline per generazione testo"""
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    pipe_kwargs = {
        "task": "text-generation",
        "model": model,
        "tokenizer": tokenizer,
        "return_full_text": False,
        **gen_cfg,
    }
    
    if not hasattr(model, "hf_device_map"):
        pipe_kwargs["device"] = 0 if device == "cuda" else -1
    
    return transformers.pipeline(**pipe_kwargs)

# ---------------------------------------------------------------------------
# TASK LOADING
# ---------------------------------------------------------------------------

def load_task_configs() -> Dict[str, TaskConfig]:
    """Carica tutte le configurazioni dei task dalle cartelle"""
    configs = {}
    
    print(f"📁 Cercando task in: {TASKS_DIR.absolute()}")
    
    if not TASKS_DIR.exists():
        print(f"⚠️ Directory '{TASKS_DIR}' non trovata. Creala e aggiungi i task.")
        return configs
    
    task_dirs = list(TASKS_DIR.iterdir())
    print(f"📂 Directory trovate: {[d.name for d in task_dirs if d.is_dir()]}")
    
    for task_dir in TASKS_DIR.iterdir():
        if not task_dir.is_dir():
            continue
            
        task_name = task_dir.name
        system_prompt_file = task_dir / "system_prompt.txt"
        examples_file = task_dir / "examples.json"
        config_file = task_dir / "config.json"  # Opzionale per configurazioni extra
        
        if not system_prompt_file.exists() or not examples_file.exists():
            print(f"⚠️ Task '{task_name}' manca di file richiesti, saltato.")
            continue
        
        try:
            # Carica system prompt
            system_prompt = system_prompt_file.read_text("utf-8").strip()
            
            # Carica esempi
            examples = json.loads(examples_file.read_text("utf-8"))
            
            # Carica configurazione extra se presente
            extra_config = {}
            if config_file.exists():
                extra_config = json.loads(config_file.read_text("utf-8"))
            
            # Determina campi input/output dagli esempi
            if not examples:
                print(f"⚠️ Task '{task_name}' non ha esempi, saltato.")
                continue
                
            first_example = examples[0]
            all_fields = list(first_example.keys())
            
            # Campi di input
            input_fields = extra_config.get("input_fields", [])
            if not input_fields:
                # Prova a indovinare dai nomi comuni
                possible_inputs = ["question", "answer", "feedback", "llm_response", 
                                 "text", "input", "prompt", "query", "content",
                                 "argomento", "livello", "argument", "level", "theme"]
                input_fields = [f for f in all_fields if f in possible_inputs]
            
            # Gestisci outputs
            outputs = {}
            
            if "outputs" in extra_config:
                # Nuova modalità: output multipli definiti esplicitamente
                for output_name, output_cfg in extra_config["outputs"].items():
                    outputs[output_name] = OutputConfig(
                        name=output_name,
                        extract_pattern=output_cfg.get("extract_pattern"),
                        type=output_cfg.get("type", "str")
                    )
            else:
                # Modalità legacy: singolo output
                output_field = extra_config.get("output_field", "")
                
                if not output_field:
                    # Prova a indovinare l'output
                    possible_outputs = ["score", "humanized_response", "result", 
                                      "output", "response", "tag", "tags", "summary",
                                      "risposta", "bool", "question_generated"]
                    for f in all_fields:
                        if f in possible_outputs:
                            output_field = f
                            break
                    if not output_field:
                        output_field = "result"  # Default
                
                # Crea configurazione output legacy
                outputs[output_field] = OutputConfig(
                    name=output_field,
                    extract_pattern=extra_config.get("extract_pattern"),
                    type="int" if "score" in output_field else "str"
                )
            
            # Crea configurazione
            config = TaskConfig(
                name=task_name,
                system_prompt=system_prompt,
                examples=examples,
                input_fields=input_fields,
                outputs=outputs,
                max_new_tokens=extra_config.get("max_new_tokens", 512),
                # Legacy fields per retrocompatibilità
                output_field=list(outputs.keys())[0] if len(outputs) == 1 else None,
                extract_pattern=list(outputs.values())[0].extract_pattern if len(outputs) == 1 else None
            )
            
            output_names = list(outputs.keys())
            configs[task_name] = config
            print(f"✅ Task '{task_name}' caricato - Input: {input_fields}, Output: {output_names}")
                
        except Exception as e:
            print(f"❌ Errore caricamento task '{task_name}': {e}")
            traceback.print_exc()
    
    return configs

# ---------------------------------------------------------------------------
# PROMPT FORMATTING
# ---------------------------------------------------------------------------

def is_role_based_model(model_id: str, tokenizer) -> bool:
    """Determina se il modello supporta role-based prompting"""
    role_based_models = [
        "gemma", "llama", "mistral", "mixtral", "qwen", "phi",
        "chatglm", "baichuan", "internlm", "yi", "deepseek"
    ]
    
    model_lower = model_id.lower()
    for model_name in role_based_models:
        if model_name in model_lower:
            return True
    
    return hasattr(tokenizer, 'apply_chat_template')

def create_messages_for_task(task_config: TaskConfig, input_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Crea messaggi per modelli role-based"""
    messages = [{"role": "system", "content": task_config.system_prompt}]
    
    # Aggiungi esempi
    for ex in task_config.examples[:8]:  # Max 8 esempi
        # Costruisci contenuto user dai campi input
        user_parts = []
        for field in task_config.input_fields:
            if field in ex:
                # Capitalizza il nome del campo per renderlo più leggibile
                field_name = field.replace("_", " ").title()
                user_parts.append(f"{field_name}: {ex[field]}")
        
        user_content = "\n".join(user_parts)
        
        # Risposta assistant con tutti gli output
        assistant_parts = []
        for output_name in task_config.outputs:
            if output_name in ex:
                output_label = output_name.replace("_", " ").title()
                assistant_parts.append(f"{output_label}: {ex[output_name]}")
        
        assistant_content = "\n".join(assistant_parts) if assistant_parts else ""
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
    
    # Aggiungi input corrente
    current_parts = []
    for field in task_config.input_fields:
        if field in input_data:
            field_name = field.replace("_", " ").title()
            current_parts.append(f"{field_name}: {input_data[field]}")
    
    current_content = "\n".join(current_parts)
    messages.append({"role": "user", "content": current_content})
    
    return messages

def format_messages(messages: List[Dict[str, str]], tokenizer) -> str:
    """Formatta messaggi per il modello"""
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback generico
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

def build_legacy_chain(llm, task_config: TaskConfig) -> LLMChain:
    """Crea chain per modelli legacy"""
    parts = [task_config.system_prompt + "\n\nESEMPI:"]
    
    for ex in task_config.examples[:8]:
        example_parts = []
        
        # Input fields
        for field in task_config.input_fields:
            if field in ex:
                field_name = field.replace("_", " ").title()
                example_parts.append(f"{field_name}: {ex[field]}")
        
        # Outputs
        for output_name in task_config.outputs:
            if output_name in ex:
                output_label = output_name.replace("_", " ").title()
                example_parts.append(f"{output_label}: {ex[output_name]}")
        
        parts.append("\n".join(example_parts))
    
    base_prompt = "\n\n".join(parts)
    
    # Template con placeholders
    template_parts = [base_prompt, "\n\nNUOVA VALUTAZIONE:"]
    for field in task_config.input_fields:
        field_name = field.replace("_", " ").title()
        template_parts.append(f"{field_name}: {{{field}}}")
    
    # Aggiungi prompt per ogni output atteso
    for output_name in task_config.outputs:
        output_label = output_name.replace("_", " ").title()
        template_parts.append(f"{output_label}: ")
    
    template = "\n".join(template_parts)
    
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(input_variables=task_config.input_fields, template=template)
    )

# ---------------------------------------------------------------------------
# RESULT EXTRACTION
# ---------------------------------------------------------------------------

def convert_value(value: str, target_type: str) -> Any:
    """Converte un valore stringa nel tipo specificato"""
    if target_type == "int":
        try:
            return int(value)
        except ValueError:
            return value
    elif target_type == "float":
        try:
            return float(value)
        except ValueError:
            return value
    elif target_type == "bool":
        if value.lower() in ["true", "vero", "si", "yes", "1"]:
            return True
        elif value.lower() in ["false", "falso", "no", "0"]:
            return False
        return value
    else:
        return value

def extract_single_result(raw_output: str, output_config: OutputConfig, task_name: str) -> Any:
    """Estrae un singolo risultato dall'output del modello"""
    output = raw_output.strip()
    
    # Se c'è un pattern di estrazione specifico
    if output_config.extract_pattern:
        match = re.search(output_config.extract_pattern, output, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1) if match.groups() else match.group(0)
            return convert_value(result, output_config.type)
    
    # Euristiche di fallback basate sul nome del campo
    output_name = output_config.name.lower()
    
    # Campi che restituiscono punteggi numerici
    if "score" in output_name or output_name in ["punteggio", "voto", "rating"]:
        # Cerca specificamente dopo "Punteggio:" o simili
        score_patterns = [
            rf"{output_config.name}:\s*(\d+)",
            r"Punteggio:\s*(\d+)",
            r"Score:\s*(\d+)",
            r"Voto:\s*(\d+)"
        ]
        for pattern in score_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return convert_value(match.group(1), output_config.type)
        
        # Altrimenti prendi l'ultimo numero trovato
        nums = re.findall(r"\d+", output)
        if nums:
            return convert_value(nums[-1], output_config.type)
    
    # Campi che restituiscono booleani
    elif output_config.type == "bool" or output_name in ["bool", "boolean", "vero_falso"]:
        bool_match = re.search(r"\b(Vero|Falso|True|False)\b", output, re.IGNORECASE)
        if bool_match:
            return convert_value(bool_match.group(1), "bool")
    
    # Campi che restituiscono tag o etichette
    elif output_name in ["tag", "tags", "label", "etichetta"]:
        # Cerca dopo "Tags:", "Tag:", etc.
        tag_patterns = [
            rf"{output_config.name}:\s*(.+?)(?:\n|$)",
            r"Tags?:\s*(.+?)(?:\n|$)",
            r"Etichett[ae]:\s*(.+?)(?:\n|$)"
        ]
        for pattern in tag_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')
    
    # Feedback o commenti
    elif output_name in ["feedback", "commento", "comment", "spiegazione"]:
        feedback_patterns = [
            rf"{output_config.name}:\s*(.+?)(?:\n|$)",
            r"Feedback:\s*(.+?)(?:\n|$)",
            r"Commento:\s*(.+?)(?:\n|$)"
        ]
        for pattern in feedback_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
    
    # Default: ritorna None se non trovato
    return None

def extract_results(raw_output: str, task_config: TaskConfig) -> Dict[str, Any]:
    """Estrae tutti i risultati dall'output del modello"""
    results = {}
    
    # Se c'è un solo output (modalità legacy)
    if len(task_config.outputs) == 1 and task_config.output_field:
        output_name = task_config.output_field
        output_config = task_config.outputs[output_name]
        result = extract_single_result(raw_output, output_config, task_config.name)
        
        # Se non trovato, prova euristica legacy
        if result is None:
            result = extract_result_legacy(raw_output, task_config)
        
        results[output_name] = result
    else:
        # Estrai ogni output definito
        for output_name, output_config in task_config.outputs.items():
            result = extract_single_result(raw_output, output_config, task_config.name)
            if result is not None:
                results[output_name] = result
    
    return results

def extract_result_legacy(raw_output: str, task_config: TaskConfig) -> Any:
    """Estrae il risultato usando le euristiche legacy (per retrocompatibilità)"""
    output = raw_output.strip()
    task_name = task_config.name.lower()
    
    # Task che restituiscono punteggi numerici
    if task_name in ["red", "green_validity", "green_cultural", "green_coherence_qa"]:
        score_match = re.search(r"Punteggio:\s*(\d+)", output, re.IGNORECASE)
        if score_match:
            return int(score_match.group(1))
        nums = re.findall(r"\d+", output)
        if nums:
            return int(nums[-1])
    
    # Task che restituiscono booleani
    elif task_name == "green_coherence_qt":
        bool_match = re.search(r"\b(Vero|Falso)\b", output, re.IGNORECASE)
        if bool_match:
            return bool_match.group(1).capitalize()
    
    # Task che restituiscono tag
    elif task_name == "orange":
        clean_output = output
        if clean_output.lower().startswith("tags:"):
            clean_output = clean_output[5:].strip()
        return clean_output.strip('"\'')
    
    # Default
    return output

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

def create_app():
    """Crea e configura l'app FastAPI"""
    
    # Carica modello
    print("🚀 Inizializzazione server unificato...")
    model, tokenizer, device = load_model_and_tokenizer(MODEL_ID)
    
    # Determina tipo di modello
    use_role_based = is_role_based_model(MODEL_ID, tokenizer)
    print(f"🎯 Modalità: {'Role-based' if use_role_based else 'Legacy'}")
    
    # Carica configurazioni task
    task_configs = load_task_configs()
    
    print(f"\n📊 Task caricati: {len(task_configs)}")
    if not task_configs:
        print("❌ Nessun task trovato! Crea la directory 'tasks' con le configurazioni.")
        sys.exit(1)
    
    # Prepara pipeline/chains per ogni task
    task_processors = {}
    
    for task_name, config in task_configs.items():
        if use_role_based:
            # Pipeline diretta per modelli moderni
            pipeline = create_pipeline(model, tokenizer, device, config.max_new_tokens)
            task_processors[task_name] = {
                "type": "role_based",
                "pipeline": pipeline,
                "config": config
            }
        else:
            # LangChain per modelli legacy
            llm = HuggingFacePipeline(
                pipeline=create_pipeline(model, tokenizer, device, config.max_new_tokens)
            )
            chain = build_legacy_chain(llm, config)
            task_processors[task_name] = {
                "type": "legacy",
                "chain": chain,
                "config": config
            }
    
    # Crea app
    app = FastAPI(
        title="Unified LLM Server",
        description="Server unificato per tutti i task LLM con supporto output multipli",
        version="2.0.0"
    )
    
    # Endpoint di health check
    @app.get("/")
    async def root():
        return {
            "status": "online",
            "model": MODEL_ID,
            "tasks": list(task_configs.keys()),
            "mode": "role_based" if use_role_based else "legacy"
        }
    
    # Crea endpoint dinamicamente per ogni task
    for task_name, processor in task_processors.items():
        config = processor["config"]
        
        # Crea modello Pydantic dinamico per request
        request_fields = {}
        for field in config.input_fields:
            # Rendi tutti i campi opzionali per flessibilità
            request_fields[field] = (Optional[str], Field(None, description=f"Campo {field}"))
        
        RequestModel = create_model(
            f"{task_name.capitalize()}Request",
            **request_fields
        )
        
        # Crea modello response con tutti gli output definiti
        response_fields = {
            "raw": (str, Field(..., description="Output grezzo del modello"))
        }
        
        # Aggiungi tutti gli output definiti nella configurazione
        for output_name, output_config in config.outputs.items():
            if output_config.type == "int":
                field_type = int
            elif output_config.type == "float":
                field_type = float
            elif output_config.type == "bool":
                field_type = bool
            else:
                field_type = str
            
            response_fields[output_name] = (Optional[field_type], Field(None, description=f"Output: {output_name}"))
        
        ResponseModel = create_model(
            f"{task_name.capitalize()}Response",
            **response_fields
        )
        
        # Usa una lambda per catturare correttamente le variabili
        def create_endpoint_handler(t_name: str, t_processor: dict):
            async def endpoint_handler(request: Request):
                try:
                    # Ottieni dati JSON direttamente
                    data = await request.json()
                    
                    # Valida che ci siano i campi richiesti
                    missing_fields = []
                    for field in t_processor["config"].input_fields:
                        if field not in data or not data[field]:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Campi mancanti o vuoti: {missing_fields}"
                        )
                    
                    print(f"📨 Task '{t_name}' - Input: {data}")
                    
                    if t_processor["type"] == "role_based":
                        # Modalità role-based
                        messages = create_messages_for_task(t_processor["config"], data)
                        formatted = format_messages(messages, tokenizer)
                        
                        outputs = t_processor["pipeline"](formatted)
                        raw_output = outputs[0]['generated_text'].strip()
                    else:
                        # Modalità legacy
                        chain_inputs = {
                            field: data.get(field, "") 
                            for field in t_processor["config"].input_fields
                        }
                        
                        result = t_processor["chain"].invoke(chain_inputs)
                        if isinstance(result, dict):
                            raw_output = result.get("text", str(result))
                        else:
                            raw_output = str(result)
                    
                    # Estrai tutti i risultati
                    extracted_results = extract_results(raw_output, t_processor["config"])
                    
                    print(f"✅ Task '{t_name}' - Outputs: {extracted_results}")
                    
                    # Costruisci risposta
                    response = {"raw": raw_output}
                    response.update(extracted_results)
                    
                    return response
                    
                except HTTPException:
                    raise
                except Exception as e:
                    print(f"❌ Errore in task '{t_name}': {e}")
                    traceback.print_exc()
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Imposta il nome della funzione per FastAPI
            endpoint_handler.__name__ = f"handle_{t_name}"
            return endpoint_handler
        
        # Crea e registra l'handler
        handler = create_endpoint_handler(task_name, processor)
        
        # Registra l'endpoint
        app.post(
            f"/{task_name}",
            response_model=ResponseModel,
            summary=f"Esegui task {task_name}",
            tags=[task_name]
        )(handler)
        
        print(f"📌 Endpoint '/{task_name}' registrato")
    
    return app

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    print(f"\n🚀 Server avviato su http://0.0.0.0:{PORT}")
    print(f"📋 Documentazione API: http://0.0.0.0:{PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")