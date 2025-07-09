#!/usr/bin/env python3
"""
Gemma Few-Shot Agent - Valutatore di Oggettività
=================================================
Sistema di valutazione automatica dell'oggettività delle risposte basato su modelli
Gemma e few-shot learning. Assegna punteggi da 0 (soggettivo) a 10 (oggettivo).

Modelli Supportati
------------------
Serie Gemma 3 (consigliati):
- google/gemma-3-1b-it   (1B parametri, più veloce)
- google/gemma-3-4b-it   (4B parametri, bilanciato)

Serie Gemma precedenti:
- google/gemma-2b-it     (2B parametri)
- google/gemma-7b-it     (7B parametri, più accurato)

Altri modelli compatibili:
- Qualsiasi modello HuggingFace che supporti text-generation

Installazione
-------------
pip install --upgrade transformers accelerate torch langchain langchain-core langchain-community

Utilizzo Base
-------------
# Valutazione standard con feedback
python red.py

# Valutazione senza considerare il feedback
python red.py --no-feedback

# Uso con CPU (se problemi GPU/CUDA)
python red.py --cpu

# Modalità debug per diagnostica
python red.py --debug

Formato Examples.json
---------------------
[
  {
    "question": "Domanda esempio",
    "answer": "Risposta esempio",
    "feedback": "Commento opzionale",
    "score": 7
  }
]

Author: Gabriele Onorato
"""


# ============================================================================
# IMPORT E CONFIGURAZIONE INIZIALE
# ============================================================================

from __future__ import annotations  # Permette type hints avanzati
import argparse  # Per gestire parametri da riga di comando
import json      # Per leggere il file di esempi
import pathlib   # Per gestire percorsi file
import sys       # Per exit e argv
import os        # Per variabili ambiente
import torch     # PyTorch per gestione tensori e GPU
import transformers  # HuggingFace Transformers per modelli AI
import warnings  # Per sopprimere warning
from typing import Optional, Tuple  # Type hints

# Sopprimi warning fastidiosi durante l'esecuzione
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Se modalità debug è attiva, abilita diagnostica CUDA dettagliata
if "--debug" in sys.argv:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Debug sincrono CUDA
    os.environ["TORCH_USE_CUDA_DSA"] = "1"    # Debug errori memoria GPU

# Gestione errori per modelli che richiedono autorizzazione
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    # Fallback se la libreria non è disponibile
    class HfHubHTTPError(Exception):
        """Fallback se HfHubHTTPError non è disponibile."""
        pass

# Import LangChain con gestione per diverse versioni
try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    try:
        from langchain.llms import HuggingFacePipeline
    except ImportError:
        print("Errore: installare langchain-community o langchain")
        sys.exit(1)

from langchain.prompts import PromptTemplate  # Per creare template di prompt
from langchain.chains import LLMChain         # Per catene di processamento


# ============================================================================
# FUNZIONE 1: CARICAMENTO MODELLO E TOKENIZER
# ============================================================================

def load_model_and_tokenizer(
    model_id: str,           # ID del modello su HuggingFace
    force_cpu: bool = False, # Forza uso CPU invece di GPU
    debug: bool = False      # Modalità debug per diagnostica
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer, str]:
    """
    Carica il modello AI e il tokenizer con gestione robusta degli errori.
    
    COSA FA:
    1. Determina se usare CPU o GPU
    2. Carica il tokenizer (converte testo in numeri per l'AI)
    3. Carica il modello AI vero e proprio
    4. Configura tutto per l'uso ottimale
    
    Returns: (modello, tokenizer, device_usato)
    """
    print(f"🔄 Caricamento modello: {model_id}")
    
    # STEP 1: Determina quale dispositivo usare (CPU vs GPU)
    if force_cpu:
        device = "cpu"
        use_accelerate = False  # Accelerate è solo per GPU
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_accelerate = (device == "cuda")  # Usa accelerate solo su GPU
    
    print(f"💻 Device selezionato: {device}")
    if use_accelerate:
        print("🚀 Usando accelerate per distribuzione automatica")
    
    try:
        # STEP 2: Carica il tokenizer
        # Il tokenizer converte testo in numeri che l'AI può capire
        print("📝 Caricamento tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,  # Permetti codice custom del modello
            use_fast=True,           # Usa versione veloce se disponibile
            padding_side="left",     # Padding a sinistra per generazione
        )
        
        # STEP 3: Configura token speciali
        # Alcuni modelli non hanno tutti i token necessari configurati
        special_tokens = {}
        if tokenizer.pad_token is None:
            # Se manca il token di padding, usa quello di fine sequenza
            special_tokens["pad_token"] = tokenizer.eos_token
        
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
            
        # Verifica che tutto sia configurato correttamente
        assert tokenizer.pad_token_id is not None, "pad_token_id non configurato"
        assert tokenizer.eos_token_id is not None, "eos_token_id non configurato"
        
        if debug:
            print(f"   pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
            print(f"   eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        
        # STEP 4: Carica il modello AI vero e proprio
        print("🤖 Caricamento modello...")
        model_kwargs = {
            "trust_remote_code": True,    # Permetti codice custom
            "low_cpu_mem_usage": True,    # Ottimizza uso memoria RAM
        }
        
        # Configurazione specifica per device
        if use_accelerate:
            # Su GPU, usa accelerate per distribuzione automatica
            model_kwargs["device_map"] = "auto"
        else:
            # Su CPU, nessuna distribuzione automatica
            model_kwargs["device_map"] = None
        
        # Configurazione precisione numerica
        if device == "cuda":
            # Su GPU usa precisione ridotta per velocità
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = "sdpa"  # Attention ottimizzata
        else:
            # Su CPU usa precisione completa
            model_kwargs["torch_dtype"] = torch.float32
        
        # Carica effettivamente il modello
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # STEP 5: Configurazioni finali
            # Adatta dimensione embedding se abbiamo aggiunto token
            model.resize_token_embeddings(len(tokenizer))
            
            # Metti in modalità valutazione (non training)
            model.eval()
            
            # Se su GPU senza accelerate, sposta manualmente il modello
            if device == "cuda" and not use_accelerate:
                model = model.to(device)
            
            # Pulisci cache GPU per liberare memoria
            if device == "cuda":
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "reset_peak_memory_stats"):
                    torch.cuda.reset_peak_memory_stats()
            
        except RuntimeError as e:
            # Se errore CUDA e non stiamo già usando CPU, riprova con CPU
            if "CUDA" in str(e) and not force_cpu:
                print("⚠️  Errore CUDA rilevato, passo a CPU...")
                return load_model_and_tokenizer(model_id, force_cpu=True, debug=debug)
            else:
                raise
        
        print("✅ Modello caricato con successo!")
        return model, tokenizer, device
        
    except (OSError, HfHubHTTPError) as e:
        # Gestione errori di accesso (modelli con licenza)
        if "gated" in str(e).lower() or "403" in str(e):
            sys.exit(
                "\n[‼] Errore di accesso: modello protetto da licenza.\n"
                f"1. Vai su https://huggingface.co/{model_id} e clicca 'Agree and access'.\n"
                "2. Crea un Access Token: https://huggingface.co/settings/tokens\n"
                "3. Esegui 'huggingface-cli login' o esporta HF_TOKEN.\n"
                "4. Rilancia lo script.\n"
            )
        else:
            raise


# ============================================================================
# FUNZIONE 2: CREAZIONE PIPELINE DI GENERAZIONE
# ============================================================================

def create_safe_pipeline(
    model,                    # Modello AI caricato
    tokenizer,               # Tokenizer caricato  
    device: str,             # Device in uso (cpu/cuda)
    max_tokens: int = 50,    # Max token da generare
    debug: bool = False      # Debug mode
):
    """
    Crea una pipeline semplice e sicura per generazione testo.
    
    COSA FA:
    1. Configura parametri di generazione conservativi
    2. Crea la pipeline HuggingFace
    3. Wrappa tutto in LangChain per uso facile
    
    La pipeline è configurata per generare solo pochi token (numeri 0-10)
    """
    # STEP 1: Configurazione MINIMA per stabilità
    # Vogliamo solo un numero da 0 a 10, quindi configurazione semplice
    generation_config = {
        "max_new_tokens": 5,                    # Solo 5 token massimo
        "do_sample": False,                     # Generazione deterministica
        "pad_token_id": tokenizer.pad_token_id, # Token per padding
        "eos_token_id": tokenizer.eos_token_id, # Token fine sequenza
    }
    
    if debug:
        print(f"🔧 Config semplice: max_tokens=5, greedy decoding")
    
    # STEP 2: Crea pipeline HuggingFace
    pipeline_kwargs = {
        "task": "text-generation",  # Tipo di task
        "model": model,             # Modello da usare
        "tokenizer": tokenizer,     # Tokenizer da usare
        "return_full_text": False,  # Ritorna solo il testo generato
        **generation_config        # Applica configurazione
    }
    
    # STEP 3: Gestione device per la pipeline
    # Solo se il modello non usa già accelerate
    if not hasattr(model, 'hf_device_map'):
        if device == "cpu":
            pipeline_kwargs["device"] = -1  # -1 = CPU in HuggingFace
        elif device == "cuda":
            pipeline_kwargs["device"] = 0   # 0 = prima GPU
    
    # STEP 4: Crea la pipeline con gestione errori
    try:
        pipe = transformers.pipeline(**pipeline_kwargs)
    except Exception as e:
        # Se errore di device, riprova senza specificarlo
        if "device" in str(e).lower():
            pipeline_kwargs.pop("device", None)
            pipe = transformers.pipeline(**pipeline_kwargs)
        else:
            raise
    
    # STEP 5: Wrappa in LangChain per interfaccia uniforme
    return HuggingFacePipeline(pipeline=pipe)


# ============================================================================
# FUNZIONE 3: COSTRUZIONE CATENA FEW-SHOT
# ============================================================================


def build_few_shot_chain(
    llm,                      # Modello di linguaggio
    examples: list[dict],     # Lista esempi dal JSON
    use_feedback: bool = True, # Se usare campo feedback
    debug: bool = False       # Debug mode
):
    """
    Costruisce una catena LangChain con few-shot prompting.
    
    COSA FA:
    1. Aggiunge prompt di guida iniziale
    2. Prende esempi dal file JSON
    3. Li formatta come prompt di esempio
    4. Crea un template che l'AI può seguire
    5. Ritorna una catena pronta per valutare nuove risposte
    
    Few-shot learning = dare esempi all'AI per farle capire cosa vogliamo
    """
    
    # STEP 1: PROMPT DI GUIDA INIZIALE
    # Questo spiega all'AI esattamente cosa deve fare
    system_prompt = """Sei un esperto valutatore di oggettività delle risposte. 

    Il tuo compito è valutare quanto sia oggettiva una risposta su una scala da 0 a 10:

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

    Analizza ogni risposta e fornisci SOLO il numero del punteggio (0-10).

    ESEMPI:"""
    
    # STEP 2: Costruisci esempi formattatati dal JSON
    prompt_parts = [system_prompt]
    
    # Usa massimo 8 esempi per non sovraccaricare il prompt
    for ex in examples[:8]:
        if use_feedback and ex.get("feedback"):
            # Formato con feedback: mostra domanda, risposta, feedback, punteggio
            prompt_parts.append(
                f"Domanda: {ex['question']}\n"
                f"Risposta: {ex['answer']}\n"
                f"Feedback: {ex['feedback']}\n"
                f"Punteggio: {ex['score']}"
            )
        else:
            # Formato senza feedback: solo domanda, risposta, punteggio
            prompt_parts.append(
                f"Domanda: {ex['question']}\n"
                f"Risposta: {ex['answer']}\n"
                f"Punteggio: {ex['score']}"
            )
    
    # STEP 3: Unisci prompt di guida + esempi
    examples_text = "\n\n".join(prompt_parts)
    
    # STEP 4: Crea template finale
    # Il template finisce con la nuova domanda/risposta da valutare
    if use_feedback:
        template = (
            examples_text + 
            "\n\nNUOVA VALUTAZIONE:"
            "\nDomanda: {question}\n"
            "Risposta: {answer}\n"
            "Feedback: {feedback}\n"
            "Punteggio: "
        )
        input_vars = ["question", "answer", "feedback"]
    else:
        template = (
            examples_text + 
            "\n\nNUOVA VALUTAZIONE:"
            "\nDomanda: {question}\n"
            "Risposta: {answer}\n"
            "Punteggio: "
        )
        input_vars = ["question", "answer"]
    
    # STEP 5: Crea PromptTemplate di LangChain
    prompt = PromptTemplate(
        input_variables=input_vars,
        template=template
    )
    
    if debug:
        print(f"📋 Usando prompt di guida + {len(examples[:8])} esempi dal JSON")
        print(f"📋 Feedback: {'ABILITATO' if use_feedback else 'DISABILITATO'}")
        print(f"📋 Primi 500 caratteri del prompt:\n{template[:500]}...\n")
    
    # STEP 6: Crea catena LangChain finale
    return LLMChain(llm=llm, prompt=prompt, verbose=debug)

# ============================================================================
# FUNZIONE 4: INVOCAZIONE SICURA DELLA CATENA
# ============================================================================

def safe_invoke_chain(
    chain,                    # Catena LangChain
    chain_input: dict,        # Input per la catena
    debug: bool = False       # Debug mode
) -> str:
    """
    Invoca la catena in modo sicuro con gestione errori.
    
    COSA FA:
    1. Chiama la catena con i parametri forniti
    2. Gestisce eventuali errori
    3. Ritorna il risultato pulito
    """
    try:
        # Chiamata diretta con dizionario di parametri
        result = chain.run(**chain_input)
        return result.strip()  # Rimuovi spazi extra
    except Exception as e:
        if debug:
            print(f"❌ Errore: {e}")
        raise


# ============================================================================
# FUNZIONE 5: SESSIONE INTERATTIVA
# ============================================================================

def interactive_session(
    chain,                    # Catena di valutazione
    use_feedback: bool = True, # Se accettare feedback
    debug: bool = False       # Debug mode
):
    """
    Avvia sessione interattiva dove l'utente può valutare risposte.
    
    COSA FA:
    1. Mostra istruzioni all'utente
    2. Loop infinito per input utente
    3. Processa ogni input e mostra valutazione
    4. Gestisce comandi speciali (help, exit)
    """
    # STEP 1: Mostra intestazione e istruzioni
    print("\n" + "="*60)
    print("SESSIONE INTERATTIVA - Valutatore di Oggettività")
    print("="*60)
    
    if use_feedback:
        print("Formato: domanda|risposta|feedback")
        print("Esempio: Cos'è la pizza?|Un piatto italiano|Risposta corretta")
    else:
        print("Formato: domanda|risposta")
        print("Esempio: Cos'è la pizza?|Un piatto italiano")
    
    print("Comandi: 'exit' per uscire, 'help' per aiuto\n")
    
    # STEP 2: Loop principale di interazione
    while True:
        try:
            # Leggi input utente
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Gestisci Ctrl+C o Ctrl+D
            print("\n\n👋 Sessione terminata.")
            break
            
        # Ignora input vuoti
        if not user_input:
            continue
            
        # STEP 3: Gestisci comandi speciali
        if user_input.lower() in {"exit", "quit", "q"}:
            print("👋 Arrivederci!")
            break
            
        if user_input.lower() == "help":
            print("\n📖 AIUTO:")
            if use_feedback:
                print("- Formato: domanda|risposta|feedback")
            else:
                print("- Formato: domanda|risposta")
            print("- Il sistema valuta l'oggettività della risposta (0-10)")
            print("- 0 = soggettivo, 10 = oggettivo\n")
            continue
            
        # STEP 4: Processa input utente
        # Aspettiamo formato: domanda|risposta o domanda|risposta|feedback
        parts = user_input.split("|")
        if len(parts) < 2:
            print(f"⚠️  Usa: domanda|risposta{' oppure domanda|risposta|feedback' if use_feedback else ''}\n")
            continue
            
        # Estrai componenti
        question = parts[0].strip()
        answer = parts[1].strip()
        
        if use_feedback:
            feedback = parts[2].strip() if len(parts) > 2 else "Nessun feedback fornito"
        
        # Verifica che non siano vuoti
        if not question or not answer:
            print("⚠️  Domanda e risposta non possono essere vuote\n")
            continue
            
        # STEP 5: Esegui valutazione
        try:
            print("🤔 Valutazione in corso...")
            
            # Prepara input per la catena
            if use_feedback:
                chain_input = {
                    "question": question,
                    "answer": answer,
                    "feedback": feedback
                }
            else:
                chain_input = {
                    "question": question,
                    "answer": answer
                }

            # Invoca catena di valutazione
            result = safe_invoke_chain(chain, chain_input, debug)

            # STEP 6: Estrai e mostra punteggio
            try:
                import re
                numbers = re.findall(r'\d+', result)
                if not numbers:
                    raise ValueError("Nessun numero presente nell'output.")
                score = int(numbers[0])
                if 0 <= score <= 10:
                    print(f"📊 Punteggio oggettività: {score}/10")
                else:
                    print(f"⚠️  Valore fuori range (0-10). Output grezzo: {result}")
            except ValueError:
                print(f"⚠️  Impossibile convertire in intero o nessun numero trovato. Output grezzo: {result}")
            except Exception as e:
                print(f"❌ Errore sconosciuto: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                if "CUDA" in str(e):
                    print("💡 Suggerimento: riavvia con --cpu flag")

            print()
        except Exception as e:
            print(f"❌ Errore durante la valutazione: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            print("💡 Suggerimento: verifica il formato dell'input o riavvia con --cpu flag\n")
            continue


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """
    Funzione principale che coordina tutto il programma.
    
    FLUSSO:
    1. Legge parametri riga di comando
    2. Carica esempi dal JSON
    3. Inizializza modello AI
    4. Crea catena di valutazione
    5. Avvia sessione interattiva
    """
    # STEP 1: Configura parser per parametri riga di comando
    parser = argparse.ArgumentParser(
        description="Gemma Few-Shot Agent - Valutatore di Oggettività",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Definisci tutti i parametri possibili
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        help="ID modello HuggingFace (default: google/gemma-3-4b-it)"
    )
    
    parser.add_argument(
        "--examples",
        default="examples.json",
        type=pathlib.Path,
        help="File JSON con esempi (default: examples.json)"
    )
    
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Non utilizzare il feedback negli esempi e nell'input"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forza uso CPU invece di GPU"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Abilita modalità debug"
    )
    
    # Processa argomenti
    args = parser.parse_args()
    
    # Flag per uso feedback (inverso di no-feedback)
    use_feedback = not args.no_feedback
    
    # STEP 2: Mostra informazioni debug se richiesto
    if args.debug:
        print("🐛 Modalità debug attiva")
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
        print(f"Feedback: {'DISABILITATO' if args.no_feedback else 'ABILITATO'}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # STEP 3: Carica esempi dal file JSON
    if not args.examples.exists():
        print(f"❌ File non trovato: {args.examples}")
        print("📝 Crea un file JSON con esempi nel formato:")
        print("""[
  {
    "question": "Cosa è la pizza?",
    "answer": "Un piatto italiano",
    "feedback": "risposta generica",
    "score": 5
  }
]""")
        sys.exit(1)
    
    try:
        # Leggi file JSON
        with open(args.examples, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        
        # STEP 4: Valida esempi
        # Verifica che abbiano tutti i campi necessari
        required_fields = ["question", "answer", "score"]
        valid_examples = []
        
        for i, ex in enumerate(examples):
            if all(field in ex for field in required_fields):
                # Assicurati che score sia numerico
                try:
                    ex["score"] = int(ex["score"])
                    valid_examples.append(ex)
                except ValueError:
                    print(f"⚠️  Esempio {i}: score non valido, ignorato")
            else:
                print(f"⚠️  Esempio {i}: campi mancanti, ignorato")
        
        if not valid_examples:
            sys.exit("❌ Nessun esempio valido nel JSON")
            
        print(f"✅ Caricati {len(valid_examples)} esempi validi dal JSON")
        examples = valid_examples
        
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Errore JSON: {e}")
    
    # STEP 5: Carica modello AI
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            args.model, 
            force_cpu=args.cpu,
            debug=args.debug
        )
    except Exception as e:
        sys.exit(f"❌ Errore caricamento modello: {e}")
    
    # STEP 6: Crea pipeline di generazione
    try:
        llm = create_safe_pipeline(
            model, 
            tokenizer, 
            device,
            max_tokens=5,  # Solo 5 token per un numero
            debug=args.debug
        )
    except Exception as e:
        sys.exit(f"❌ Errore creazione pipeline: {e}")
    
    # STEP 7: Costruisci catena di valutazione con esempi
    chain = build_few_shot_chain(llm, examples, use_feedback, args.debug)
    
    # STEP 8: Avvia sessione interattiva
    interactive_session(chain, use_feedback, args.debug)


# ============================================================================
# PUNTO DI INGRESSO
# ============================================================================

if __name__ == "__main__":
    main()