#!/usr/bin/env python3
"""
Gemma Few-Shot Agent - Generatore di Domande sulla Cultura Italiana
====================================================================
Sistema basato su modelli Gemma e few-shot learning per generare
DOMANDE pertinenti a un ARGOMENTO (keyword) fornito, riguardante la cultura italiana.

Modelli Supportati
------------------
Serie Gemma 3 (consigliati):
- google/gemma-3-1b-it
- google/gemma-3-4b-it

Installazione
-------------
pip install --upgrade transformers accelerate torch langchain langchain-core langchain-community

Utilizzo Base
-------------
# Generazione domande standard
python NOME_SCRIPT_GENERATORE_DOMANDE.py

# Uso con CPU
python NOME_SCRIPT_GENERATORE_DOMANDE.py --cpu

# Modalità debug
python NOME_SCRIPT_GENERATORE_DOMANDE.py --debug

Formato Examples.json (per generare domande)
--------------------------------------------
Il file `examples.json` dovrebbe contenere una lista di dizionari, ognuno
rappresentante un argomento (keyword) e la domanda che il modello dovrebbe generare.

[
  {
    "argument": "Cibo",
    "question_generated": "Quali sono i segreti per preparare un autentico ragù alla bolognese e quali variazioni regionali esistono?"
  },
  {
    "argument": "Rinascimento",
    "question_generated": "Oltre ai celebri artisti, quali figure meno note ma cruciali hanno contribuito allo sviluppo del pensiero rinascimentale italiano?"
  },
  {
    "argument": "Musica lirica",
    "question_generated": "Come si è evoluta la struttura dell'opera lirica italiana dal Barocco al Verismo, e quali compositori ne sono stati i principali innovatori?"
  },
  {
    "argument": "Dialetti",
    "question_generated": "Qual è l'importanza storica e culturale dei dialetti in Italia e come convivono con la lingua italiana standard oggi?"
  },
  {
    "argument": "Cinema Neorealista",
    "question_generated": "Quali tecniche cinematografiche e tematiche distintive caratterizzano il Neorealismo italiano e quale impatto ha avuto sul cinema mondiale?"
  },
  {
    "argument": "Moda",
    "question_generated": "Come è nata l'alta moda italiana nel dopoguerra e quali sono le città chiave che ne hanno definito lo stile e il prestigio internazionale?"
  }
]

Author: Gabriele Onorato (Adattato per generazione domande su cultura italiana)
"""

# ============================================================================
# IMPORT E CONFIGURAZIONE INIZIALE
# ============================================================================

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import os
import torch
import transformers
import warnings
from typing import Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if "--debug" in sys.argv:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    class HfHubHTTPError(Exception): pass

try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    try:
        from langchain.llms import HuggingFacePipeline
    except ImportError:
        print("Errore: installare langchain-community o langchain")
        sys.exit(1)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ============================================================================
# FUNZIONE 1: CARICAMENTO MODELLO E TOKENIZER (Invariata)
# ============================================================================
def load_model_and_tokenizer(
    model_id: str,
    force_cpu: bool = False,
    debug: bool = False
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer, str]:
    print(f"🔄 Caricamento modello: {model_id}")
    if force_cpu:
        device = "cpu"
        use_accelerate = False
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_accelerate = (device == "cuda")
    
    print(f"💻 Device selezionato: {device}{' (con accelerate)' if use_accelerate else ''}")
    
    try:
        print("📝 Caricamento tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=True, padding_side="left"
        )
        if tokenizer.pad_token is None: tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        assert tokenizer.pad_token_id is not None and tokenizer.eos_token_id is not None
        if debug: print(f"   pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id}), eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

        print("🤖 Caricamento modello...")
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if use_accelerate: model_kwargs["device_map"] = "auto"
        else: model_kwargs["device_map"] = None
        
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = "sdpa"
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            model.resize_token_embeddings(len(tokenizer))
            model.eval()
            if device == "cuda" and not use_accelerate: model = model.to(device)
            if device == "cuda":
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "reset_peak_memory_stats"): torch.cuda.reset_peak_memory_stats()
        except RuntimeError as e:
            if "CUDA" in str(e) and not force_cpu:
                print("⚠️  Errore CUDA rilevato, passo a CPU...")
                return load_model_and_tokenizer(model_id, force_cpu=True, debug=debug)
            raise
        print("✅ Modello caricato con successo!")
        return model, tokenizer, device
    except (OSError, HfHubHTTPError) as e:
        if "gated" in str(e).lower() or "403" in str(e):
            sys.exit(f"\n[‼] Errore accesso: modello protetto. Vai su https://huggingface.co/{model_id}, accetta, crea token e login via CLI.\n")
        raise

# ============================================================================
# FUNZIONE 2: CREAZIONE PIPELINE DI GENERAZIONE
# ============================================================================
def create_question_pipeline(
    model, tokenizer, device: str, max_tokens: int = 75, debug: bool = False # Max tokens per una domanda
):
    generation_config = {
        "max_new_tokens": max_tokens,
        "do_sample": True, "temperature": 0.75, "top_p": 0.9, # Per domande più creative
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
    }
    if debug: print(f"🔧 Configurazione generazione domande: max_tokens={max_tokens}, T=0.75, top_p=0.9")
    
    pipeline_kwargs = {"task": "text-generation", "model": model, "tokenizer": tokenizer, "return_full_text": False, **generation_config}
    if not hasattr(model, 'hf_device_map'):
        pipeline_kwargs["device"] = 0 if device == "cuda" else -1
    try:
        pipe = transformers.pipeline(**pipeline_kwargs)
    except Exception as e:
        if "device" in str(e).lower():
            pipeline_kwargs.pop("device", None)
            pipe = transformers.pipeline(**pipeline_kwargs)
        else: raise
    return HuggingFacePipeline(pipeline=pipe)

# ============================================================================
# FUNZIONE 3: COSTRUZIONE CATENA FEW-SHOT PER GENERARE DOMANDE
# ============================================================================
def build_question_generation_chain(llm, examples: list[dict], debug: bool = False):
    system_prompt = """Sei un assistente AI specializzato nella cultura italiana, abile nel formulare domande intelligenti e pertinenti.
Il tuo compito è prendere un ARGOMENTO (una parola chiave o una breve frase) fornito dall'utente e generare una DOMANDA specifica e stimolante su quell'argomento, focalizzata sulla cultura italiana.
La domanda dovrebbe incoraggiare una riflessione o una spiegazione dettagliata, andando oltre la superficie. Evita domande banali o a cui si può rispondere con un sì/no.

ESEMPI:"""
    
    prompt_parts = [system_prompt]
    for ex in examples[:8]: # Usa massimo 8 esempi
        prompt_parts.append(
            f"Argomento: {ex['argument']}\n"
            f"Domanda Generata: {ex['question_generated']}"
        )
    
    examples_text = "\n\n".join(prompt_parts)
    
    template = (
        examples_text + 
        "\n\nNUOVA DOMANDA DA GENERARE:"
        "\nArgomento: {argument}\n"
        "Domanda Generata:" # L'AI continuerà da qui
    )
    input_vars = ["argument"]
    
    prompt = PromptTemplate(input_variables=input_vars, template=template)
    
    if debug:
        print(f"📋 Prompt per generazione domande: {len(examples[:8])} esempi usati.")
        print(f"📋 Primi 500 caratteri del prompt:\n{template[:500]}...\n")
    
    return LLMChain(llm=llm, prompt=prompt, verbose=debug)

# ============================================================================
# FUNZIONE 4: INVOCAZIONE SICURA DELLA CATENA (Invariata)
# ============================================================================
def safe_invoke_chain(chain, chain_input: dict, debug: bool = False) -> str:
    try:
        result = chain.run(**chain_input)
        return result.strip()
    except Exception as e:
        if debug: print(f"❌ Errore durante l'invocazione della catena: {e}")
        raise

# ============================================================================
# FUNZIONE 5: SESSIONE INTERATTIVA PER GENERARE DOMANDE
# ============================================================================
def interactive_question_session(chain, debug: bool = False):
    print("\n" + "="*60)
    print("SESSIONE INTERATTIVA - Generatore di Domande sulla Cultura Italiana")
    print("="*60)
    print("Inserisci un ARGOMENTO (keyword) sulla cultura italiana (es: 'Pizza', 'Arte Romana', 'Moda').")
    print("Il sistema genererà una domanda pertinente basata sul tuo argomento.")
    print("Comandi: 'exit' per uscire, 'help' per aiuto\n")
    
    while True:
        try:
            user_input = input("Argomento >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Sessione terminata.")
            break
        if not user_input: continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("👋 Arrivederci!")
            break
        if user_input.lower() == "help":
            print("\n📖 AIUTO:")
            print("- Inserisci un argomento (parola chiave) sulla cultura italiana.")
            print("- Esempio: 'Vino', 'Fellini', 'Design'")
            print("- Il sistema genererà una domanda specifica su quell'argomento.\n")
            continue
            
        argument = user_input
        
        try:
            print("🤔 Generazione domanda in corso...")
            chain_input = {"argument": argument}
            generated_question = safe_invoke_chain(chain, chain_input, debug)

            print("\nDomanda Generata:")
            print("-" * 20)
            print(generated_question)
            print("-" * 20)
            print()

        except Exception as e:
            print(f"❌ Errore durante la generazione della domanda: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            if "CUDA" in str(e): print("💡 Suggerimento: riavvia con --cpu flag")
            print()
            continue

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Gemma Few-Shot Agent - Generatore di Domande sulla Cultura Italiana",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="ID modello HuggingFace")
    parser.add_argument("--examples", default="examples.json", type=pathlib.Path, help="File JSON con esempi (argument -> question_generated)")
    parser.add_argument("--cpu", action="store_true", help="Forza uso CPU")
    parser.add_argument("--debug", action="store_true", help="Abilita modalità debug")
    args = parser.parse_args()
    
    if args.debug:
        print("🐛 Modalità debug attiva")
        # ... (altre info debug se necessario)

    if not args.examples.exists():
        print(f"❌ File non trovato: {args.examples}")
        print("📝 Crea 'examples.json' con il formato: [{'argument': 'Keyword', 'question_generated': 'Domanda generata...'}]. Vedi docstring per dettagli.")
        sys.exit(1)
    
    try:
        with open(args.examples, 'r', encoding='utf-8') as f: examples_data = json.load(f)
        required_fields = ["argument", "question_generated"]
        valid_examples = [ex for ex in examples_data if all(field in ex for field in required_fields)]
        
        if not valid_examples: sys.exit(f"❌ Nessun esempio valido (con campi '{', '.join(required_fields)}') nel JSON.")
        print(f"✅ Caricati {len(valid_examples)} esempi validi da {args.examples}")
        examples = valid_examples
    except json.JSONDecodeError as e: sys.exit(f"❌ Errore JSON: {e}")
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(args.model, args.cpu, args.debug)
    except Exception as e: sys.exit(f"❌ Errore caricamento modello: {e}")
    
    try:
        llm = create_question_pipeline(model, tokenizer, device, max_tokens=75, debug=args.debug) # Max 75 tokens per una domanda
    except Exception as e: sys.exit(f"❌ Errore creazione pipeline: {e}")
    
    chain = build_question_generation_chain(llm, examples, args.debug)
    interactive_question_session(chain, args.debug)

if __name__ == "__main__":
    main()