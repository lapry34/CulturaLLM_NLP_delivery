#!/usr/bin/env python3
"""
Script interattivo per inviare richieste POST a un server specificato.

- Chiede IP/localhost, porta ed endpoint.
- Consente di definire una lista di parametri (nome e tipo), terminando con "STOP".
- Per ogni invio, chiede i valori dei parametri e invia la richiesta in JSON.
- Stampa la risposta JSON (o il testo se non è JSON).
"""
import requests
import time

def get_parameters():
    """Legge da input nome e tipo di ciascun parametro."""
    params = []
    while True:
        entry = input("Inserisci nome parametro e tipo (string,int,float) separati da spazio (o STOP per terminare): ")
        if entry.strip().upper() == "STOP":
            break
        parts = entry.strip().split()
        if len(parts) != 2:
            print("Formato non valido. Riprovare.")
            continue
        name, type_str = parts
        type_str = type_str.lower()
        if type_str not in ("string", "int", "float"):
            print("Tipo non valido. Usa 'string', 'int' o 'float'.")
            continue
        params.append((name, type_str))
    return params


def cast_value(value_str, type_str):
    """Converte la stringa di input nel tipo richiesto."""
    if type_str == "int":
        return int(value_str)
    if type_str == "float":
        return float(value_str)
    # string
    return value_str


def main():
    # Configurazione server
    ip = input("Inserisci indirizzo IP o localhost: ").strip()
    port = input("Inserisci porta: ").strip()
    endpoint = input("Inserisci endpoint (senza slash iniziale): ").strip()

    # Definizione parametri
    params = get_parameters()
    if not params:
        print("Nessun parametro specificato. Uscita.")
        return

    # Ciclo di invio richieste
    while True:
        data = {}
        # Raccolta valori
        for name, type_str in params:
            val = input(f"Inserisci valore per '{name}' ({type_str}): ")
            try:
                data[name] = cast_value(val, type_str)
            except ValueError:
                print(f"Valore non valido per tipo {type_str}. Riprovare.")
                break
        else:
            # Invio richiesta
            print("Invio richiesta...")
            time_start = time.time()
            url = f"http://{ip}:{port}/{endpoint}"
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                try:
                    print("Risposta JSON:", response.json())
                except ValueError:
                    print("Risposta non in formato JSON:", response.text)
            except requests.RequestException as e:
                print("Errore nella richiesta:", e)

            time_end = time.time()
            print(f"Tempo di risposta: {time_end - time_start:.2f} secondi")

        # Proseguire o terminare?
        again = input("Vuoi inviare un'altra richiesta? (s/n): ").strip().lower()
        if again != 's':
            print("Terminato.")
            break


if __name__ == "__main__":
    main()

