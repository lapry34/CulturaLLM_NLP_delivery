#!/bin/bash

# Esempio per il task yellow
echo "[YELLOW TASK] Sending curl request..."
echo "[Input]: 'argument': 'Cibo'"
curl -X POST http://localhost:8071/yellow -H "accept: application/json" -H "Content-Type: application/json" -d '{"argument": "Cibo"}'
echo ""
echo "[YELLOW TASK] Response received."
echo ""

# Esempio per il task red
echo "[RED TASK] Sending curl request..."
echo "[Input] 'question': 'Cosa è la bistecca?', 'answer': 'Cibo', 'feedback': 'no'"
curl -X POST http://localhost:8071/red -H "accept: application/json" -H "Content-Type: application/json" -d '{"question": "Cosa è la bistecca?","answer": "Cibo","feedback": "no"}'

echo ""
echo "[RED TASK] Response received."
echo ""

# Esempio per il task magenta
echo "[MAGENTA TASK] Sending curl request..."
echo "[Input] 'llm_response': 'In qualità di modello linguistico avanzato...', 'level': '1'"
curl -X POST http://localhost:8071/magenta -H "accept: application/json" -H "Content-Type: application/json" -d '{"llm_response": "In qualità di modello linguistico avanzato, desidero informarla che la sua richiesta è stata elaborata con successo e i risultati sono pronti per la sua disamina", "level": "1"}'
echo ""
echo "[MAGENTA TASK] Response received."
echo ""

# Esempio per il task orange
echo "[ORANGE TASK] Sending curl request..."
echo "[Input] 'question': 'Che cos è il Barocco?'"
curl -X POST http://localhost:8071/orange -H "accept: application/json" -H "Content-Type: application/json" -d '{"question": "Che cos è il Barocco?"}'
echo ""
echo "[ORANGE TASK] Response received."
echo ""

# Esempio per il task cyan
echo "[CYAN TASK] Sending curl request..."
echo "[Input] 'argomento': 'cucina romana', 'livello': '1'"
curl -X POST http://localhost:8071/cyan -H "accept: application/json" -H "Content-Type: application/json" -d '{"argomento": "cucina romana", "livello": "1"}'
echo ""
echo "[CYAN TASK] Response received."
echo ""

# Esempio per il task green_coherence_QA
echo "[GREEN QA TASK] Sending curl request..."
echo "[Input] 'question': 'Qual è la capitale d Italia?', 'answer': 'La capitale d Italia è Roma.'"
curl -X POST http://localhost:8071/green_coherence_QA -H "accept: application/json" -H "Content-Type: application/json" -d '{"question": "Qual è la capitale d Italia?", "answer": "La capitale d Italia è Roma."}'
echo ""
echo "[GREEN QA TASK] Response received."
echo ""

# Esempio per il task green_coherence_QT
echo "[GREEN QT TASK] Sending curl request..."
echo "[Input] 'question': 'Quali sono gli ingredienti principali...', 'theme': 'Cucina italiana'"
curl -X POST http://localhost:8071/green_coherence_QT -H "accept: application/json" -H "Content-Type: application/json" -d '{"question": "Quali sono gli ingredienti principali per preparare una pizza margherita?", "theme": "Cucina italiana"}'
echo ""
echo "[GREEN QT TASK] Response received."
echo ""

# Esempio per il task green cultural
echo "[GREEN CULTURAL TASK] Sending curl request..."
echo "[Input] 'question': 'Spiega il processo di fotosintesi nelle piante'"
curl -X POST http://localhost:8071/green_cultural -H "accept: application/json" -H "Content-Type: application/json" -d '{"question": "Spiega il processo di fotosintesi nelle piante"}'
echo ""
echo "[GREEN CULTURAL TASK] Response received."
echo ""

# Esempio per il task green validity
echo "[GREEN VALIDITY TASK] Sending curl request..."
echo "[Input] 'question': 'Quali sono i principali fattori...', 'answer': 'I principali fattori del cambiamento climatico...'"
curl -X POST http://localhost:8071/green_validity -H "accept: application/json" -H "Content-Type: application/json" -d '{"question": "Quali sono i principali fattori che influenzano il cambiamento climatico?", "answer": "I principali fattori del cambiamento climatico includono le emissioni di gas serra (CO2, metano, ossidi di azoto) derivanti da combustibili fossili, deforestazione, agricoltura intensiva e processi industriali. L effetto serra naturale viene amplificato dall attività umana, causando aumento delle temperature globali, scioglimento dei ghiacci, innalzamento del livello dei mari e alterazioni dei pattern meteorologici. Le concentrazioni di CO2 sono aumentate del 40% dall era preindustriale."}'
echo ""
echo "[GREEN VALIDITY TASK] Response received."
echo ""