# Come appizzà er docker

## Per la prima volta

```bash
docker compose up --build
```
## Poi

```bash
docker compose up 
```

## per spegnerlo
```bash
docker compose down
```

## Cosa inviargli in POST JSON su porta 8069
```json
{
        "question": "cosa è la peperonata?",
        "answer": "un piatto tipico italiano del sud italia, in particolare della campania molto buono e pesante da digerire"
}
```

## se ci sta il feedback attivato
```json
{
        "question": "cosa è la peperonata?",
        "answer": "un piatto tipico italiano del sud italia, in particolare della campania molto buono e pesante da digerire",
        "feedback": "trallalero trallala"
}
```
## lui risponderà qualcosa del tipo
```json
{
    "score": 4,
    "raw": "4\n```\nP"
}
```
# MI RACCOMANDO il file .env PRIMA de far partì er docker

nel .env mettere
```
HF_TOKEN=xxx
MODEL_ID=google/gemma-3-4b-it
```
o comunque il modello che uno vuole