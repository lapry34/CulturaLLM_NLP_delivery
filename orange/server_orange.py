#!/usr/bin/env python3
"""
Gemma Few‑Shot Agent Server – generazione di tag via HTTP
================================================================
Espone un endpoint POST /tag (porta 8069) che restituisce i tag generati.
Supporta due modalità di prompting:
1. Role-based: Per modelli moderni che supportano conversazioni (es. Gemma >=2, LLaMA 2/3).
2. Legacy Completion-Style: Per modelli più vecchi, usando LangChain.
"""

# ---------------- PATCH anti‑Triton ------------------
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# ----------------------------------------------------

import json, pathlib, re, sys, warnings
from typing import Optional, Tuple, List, Dict

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

# ---------------------------------------------------------------------------
# MODEL + TOKENIZER ---------------------------------------------------------
def load_model_and_tokenizer(model_id: str) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer, str]:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"💻 Device selezionato: {device}")

    print(f"📝 Caricamento tokenizer: {model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    print(f"🤖 Caricamento modello: {model_id}")
    model_kwargs = {"trust_remote_code": True}

    if device == "cuda":
        model_kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        })
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    else: 
        model_kwargs["torch_dtype"] = torch.float32
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not hasattr(model, "hf_device_map"):
            print(f"Moving model to {device}")
            model.to(device)
        elif device == "cpu" and hasattr(model, "hf_device_map"):
             print("Model loaded with device_map, not moving to CPU explicitly.")

    model.eval()
    return model, tokenizer, device

# ---------------------------------------------------------------------------
# PIPELINE (Returns raw transformers.pipeline) -----------------------------
def create_raw_pipeline(model, tokenizer, device: str, max_new_tokens: int = 50):
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False, 
        **gen_cfg,
    )
    if not hasattr(model, "hf_device_map"):
        if device == "cuda":
            pipe_kwargs["device"] = 0
        elif device == "cpu":
             pipe_kwargs["device"] = -1

    print(f"🔧 Raw Transformers Pipeline pronta – device={device}, max_new_tokens={max_new_tokens}")
    return transformers.pipeline(**pipe_kwargs)

# ---------------------------------------------------------------------------
# PROMPTING FUNCTIONS (ROLE-BASED and LEGACY) -------------------------------
SYSTEM_PROMPT_TAGGING = """
Sei un esperto classificatore di domande. 
Per ogni domanda inviata, genera 3 tag in italiano.
Le domande saranno tutte relative a storia, cultura e costumi italiani evita quindi di inserire Italia o Italiana tra i tag.
I tag dovrebbero essere una o due parole ciascuno e dovrebbero riassumere brevemente la domanda posta.
I tag devono essere completamente rilevanti rispetto alla domanda e includere solo elementi atti a riassumerla.
I tag devono essere brevi e non essere in numero diverso da 3. Non devi indicare il numero di ogni tag, semplicemente generarli in sequenze.
Non generare altro rispetto ai tag, generali separati da virgole ed escludi ogni altra formattazione.
Ogni altro output verrà considerato errato.
"""

def is_text_generation_model(model_id: str, tokenizer) -> bool:
    model_id_lower = model_id.lower()
    chat_model_families = [
        "gemma", "llama", "mistral", "mixtral", "phi-3", "phi-2", "qwen", "yi",
        "command-r", "cohere", "claude", "chatglm", "baichuan", "internlm", "deepseek"
    ]
    if any(family in model_id_lower for family in chat_model_families):
        if hasattr(tokenizer, 'apply_chat_template'):
            print(f"Model family '{model_id_lower}' and tokenizer has 'apply_chat_template'. Assuming role-based support.")
            return True
        else:
            print(f"Model family '{model_id_lower}' suggests chat, but tokenizer lacks 'apply_chat_template'. Falling back to legacy.")
            return False

    if hasattr(tokenizer, 'apply_chat_template'):
        print("Tokenizer has 'apply_chat_template'. Assuming role-based support.")
        return True
    
    print("No clear indication of role-based support. Falling back to legacy completion-style prompting.")
    return False

def create_few_shot_messages_for_tagging(examples: List[Dict], system_prompt_content: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt_content}]
    for ex in examples[:3]: 
        if "question" in ex and "tags" in ex:
            messages.append({"role": "user", "content": f"Domanda: {ex['question']}"})
            messages.append({"role": "assistant", "content": ex['tags']})
        else:
            print(f"Skipping example due to missing 'question' or 'tags': {ex}")
    return messages

def format_messages_for_model(messages: List[Dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt_str = ""
        for msg in messages:
            formatted_prompt_str += f"{msg['role']}: {msg['content']}\n"
        if messages and messages[-1]['role'] == 'user': 
            formatted_prompt_str += "assistant: "
        return formatted_prompt_str.strip()


def build_legacy_chain_for_tagging(llm, examples: list[dict], system_prompt_content: str) -> LLMChain:
    parts = [system_prompt_content.strip()]
    parts.append("\nEcco una lista di esempi DI RIFERIMENTO (NON devi generarli tu):")

    for ex in examples[:5]: 
        if "question" in ex and "tags" in ex:
            parts.append(
                f"Domanda: {ex['question']}\n"
                f"Tags: {ex['tags']}\n" 
            )
    base_prompt = "\n\n".join(parts)
    template = base_prompt + "\n\nNUOVA VALUTAZIONE:\nDomanda: {question}\nTags: "
    input_vars = ["question"]
    return LLMChain(llm=llm, prompt=PromptTemplate(input_variables=input_vars, template=template))

# ---------------------------------------------------------------------------
# FASTAPI APP SETUP ---------------------------------------------------------
EXAMPLES_PATH = pathlib.Path("examples.json")
if not EXAMPLES_PATH.exists():
    print("❌ examples.json non trovato.")
    sys.exit(1)
examples_text = EXAMPLES_PATH.read_text("utf8")
try:
    examples_data = json.loads(examples_text)
    if not isinstance(examples_data, list) or not all(isinstance(ex, dict) and "question" in ex and "tags" in ex for ex in examples_data):
        raise ValueError("examples.json deve essere una lista di dizionari con chiavi 'question' e 'tags'.")
except Exception as e:
    print(f"❌ Errore in examples.json: {e}")
    sys.exit(1)

model_id_env = os.getenv("MODEL_ID")
if not model_id_env:
    print("❌ Variabile d'ambiente MODEL_ID non impostata.")
    sys.exit(1)

model, tokenizer, device = load_model_and_tokenizer(model_id_env)

if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    print(f"ℹ️  Tokenizer Chat Template:\n{tokenizer.chat_template}")
else:
    print("ℹ️  Tokenizer does not have a 'chat_template' attribute or it's empty.")

raw_hf_pipeline = create_raw_pipeline(model, tokenizer, device) 

USE_ROLE_BASED = is_text_generation_model(model_id_env, tokenizer)
print(f"🎯 Modalità prompting selezionata: {'Role-based (Chat)' if USE_ROLE_BASED else 'Legacy (Completion)'}")

legacy_chain = None
base_chat_messages = None

if USE_ROLE_BASED:
    base_chat_messages = create_few_shot_messages_for_tagging(examples_data, SYSTEM_PROMPT_TAGGING)
    print(f"📚 Pre-costruiti {len(base_chat_messages)} messaggi per role-based prompting.")
else: 
    stop_strings_for_legacy = ["\nDomanda:", "\nTags:", "\nNUOVA VALUTAZIONE:", "\n\n", "```"]
    langchain_llm_wrapper = HuggingFacePipeline(
        pipeline=raw_hf_pipeline,
        model_kwargs={"stop_sequences": stop_strings_for_legacy}
    )
    legacy_chain = build_legacy_chain_for_tagging(langchain_llm_wrapper, examples_data, SYSTEM_PROMPT_TAGGING)
    print("🔗 LangChain (legacy) chain creata con stop sequences.")


app = FastAPI(title="Gemma Tag Generator - Role/Legacy")

class EvalRequest(BaseModel):
    question: str

class EvalResponse(BaseModel):
    tags: str
    raw_output: str 
    mode_used: str  

@app.post("/tag", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Il campo 'question' è obbligatorio.")
    
    generated_text_raw = ""
    mode = ""

    try:
        if USE_ROLE_BASED:
            mode = "role_based"
            current_messages = base_chat_messages.copy()
            current_messages.append({"role": "user", "content": f"Domanda: {req.question}"})
            
            formatted_prompt_for_chat = format_messages_for_model(current_messages, tokenizer)
            
            pipeline_output = raw_hf_pipeline(formatted_prompt_for_chat) 
            if pipeline_output and isinstance(pipeline_output, list) and "generated_text" in pipeline_output[0]:
                generated_text_raw = pipeline_output[0]["generated_text"].strip()
            else:
                print(f"⚠️ Output inatteso dalla pipeline (role-based): {pipeline_output}")
                generated_text_raw = ""

        else: 
            mode = "legacy_completion"
            inputs = {"question": req.question}
            chain_output = legacy_chain.invoke(inputs)
            if isinstance(chain_output, dict):
                generated_text_raw = chain_output.get("text", "").strip()
            elif isinstance(chain_output, str):
                generated_text_raw = chain_output.strip()
            else:
                generated_text_raw = str(chain_output).strip()
        
        # --- ROBUST POST-PROCESSING ---
        current_processing_text = generated_text_raw
        
        # Stage 1: Primary Truncation - Cut off "new turn" or "new question" hallucinations
        # Ordered by likeliness or specificity. More specific patterns first if they overlap.
        stop_patterns_for_cleanup = [
            "\n<|start_header_id|>user<|end_header_id|>", # Llama 3 (very specific)
            "\n<|user|>",             # Common chat
            "\n<|end_of_turn|>",      # Gemma common
            "<|end_of_turn|>",       # Gemma (no preceding newline)
            "\n\nNUOVA VALUTAZIONE:", # Legacy continuation
            "\nDomanda:",             # Legacy or chat hallucination
            # "\nTags:",              # Keep commented, could be too aggressive
            "\nUser:",                # Common English user marker (often with colon)
            "\nUtente:",              # Italian "user" (often with colon)
            # More general user markers (use with caution, could be part of a valid tag if not careful)
            # These are more likely to be artifacts if they are followed by a newline or are at the end.
            "\nuser\n",               # "user" on its own line
            "\nユーザ\n",              # Japanese "user" on its own line
            # "\nuser",               # Too broad, might catch "username" if not careful
            # "\nユーザ",               # Too broad
        ]
        min_index_stage1 = len(current_processing_text)
        for pattern in stop_patterns_for_cleanup:
            idx = current_processing_text.find(pattern) 
            if idx != -1 and idx < min_index_stage1:
                min_index_stage1 = idx
        current_processing_text = current_processing_text[:min_index_stage1].strip()

        # Stage 2: Section Truncation - If a double newline exists, take only text before the first one.
        double_newline_index = current_processing_text.find("\n\n")
        if double_newline_index != -1:
            current_processing_text = current_processing_text[:double_newline_index].strip()
        
        # Stage 3: Clean up common prefixes and wrappers from the isolated block
        if current_processing_text.startswith("```") and current_processing_text.endswith("```"):
            current_processing_text = current_processing_text[3:-3].strip()
        elif current_processing_text.startswith("```"): 
            current_processing_text = current_processing_text[3:].strip()
        elif current_processing_text.endswith("```"): 
            current_processing_text = current_processing_text[:-3].strip()

        if current_processing_text.lower().startswith("tags:"):
            current_processing_text = current_processing_text[len("tags:") :].strip()
        
        # Stage 4: Parse individual tags from the cleaned block
        individual_tags = []
        normalized_for_split = current_processing_text.replace('\n', ',') 
        potential_tags_parts = [p.strip() for p in normalized_for_split.split(',') if p.strip()]

        for part in potential_tags_parts:
            cleaned_part = part
            while True:
                original_len = len(cleaned_part)
                if cleaned_part.startswith("* "): cleaned_part = cleaned_part[2:].strip()
                elif cleaned_part.startswith("- "): cleaned_part = cleaned_part[2:].strip()
                elif cleaned_part.startswith("*") and len(cleaned_part)>1 and not cleaned_part[1].isspace(): 
                    cleaned_part = cleaned_part[1:].strip()
                elif cleaned_part.startswith("-") and len(cleaned_part)>1 and not cleaned_part[1].isspace(): 
                    cleaned_part = cleaned_part[1:].strip()
                if len(cleaned_part) == original_len: break
            
            cleaned_part = cleaned_part.strip("[]'\" ")
            if cleaned_part: 
                individual_tags.append(cleaned_part)
        
        # Stage 5: Final Regex Cleanup for trailing user artifacts
        # This targets ", user" or ", ユーザ" etc., at the very end of the string.
        # It also handles cases where "user" might be the only thing left after splitting if it was a "tag".
        # And cases where "user" is directly after the last valid tag without a comma.
        
        # First, join to a preliminary string
        temp_tags_str = ", ".join(individual_tags)

        # Regex patterns to remove trailing user-like artifacts
        # \s* matches zero or more whitespace characters
        # (?: ... ) is a non-capturing group
        # $ asserts position at the end of the string
        # This will remove ", user", " user", or just "user" if it's the last "tag"
        # and also handles Japanese "ユーザ" and Italian "utente"
        # We process the list `individual_tags` again to avoid issues if "user" was a standalone "tag"
        
        cleaned_individual_tags = []
        user_artifact_patterns = [
            re.compile(r"^\s*user\s*$", re.IGNORECASE),
            re.compile(r"^\s*ユーザ\s*$", re.IGNORECASE), # Katakana 'user'
            re.compile(r"^\s*utente\s*$", re.IGNORECASE) # Italian 'user'
        ]

        for tag in individual_tags:
            is_artifact = False
            for pattern in user_artifact_patterns:
                if pattern.match(tag):
                    is_artifact = True
                    break
            if not is_artifact:
                cleaned_individual_tags.append(tag)
        
        final_tags_str = ", ".join(cleaned_individual_tags)
        # Final strip and remove any trailing commas or periods (again, just in case)
        final_tags_str = final_tags_str.strip().rstrip(',.')
        # --- END OF ROBUST POST-PROCESSING ---

        if not final_tags_str:
            print(f"⚠️  Domanda: {req.question!r} -> Raw Output ({mode}): {generated_text_raw!r} -> Nessun tag generato dopo la pulizia. Processed text before final parsing: '{current_processing_text}'")
        
        print(f"✅ Domanda: {req.question!r} ({mode}) -> Raw: {generated_text_raw!r} -> Tags Puliti: {final_tags_str!r}")
        return EvalResponse(tags=final_tags_str, raw_output=generated_text_raw, mode_used=mode)

    except Exception as e:
        print(f"❌ Errore ({mode}) per la domanda {req.question!r}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore interno del server ({mode}): {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print(f"ℹ️  Avvio del server Uvicorn su http://0.0.0.0:8068")
    uvicorn.run(app, host="0.0.0.0", port=8068, log_level="info")