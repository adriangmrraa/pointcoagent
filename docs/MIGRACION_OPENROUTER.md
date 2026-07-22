# Migración a OpenRouter / Groq (y cómo volver a OpenAI)

> Fecha: 2026-07. Motivo: OpenAI no deja cargar saldo. OpenRouter revende los
> mismos modelos de OpenAI con API compatible; Groq corre Whisper para el audio.
> **Todo se activa por variables de entorno. El código por defecto sigue en OpenAI.**

---

## 1. Idea central (no confundirse)

- **NO se reemplaza `OPENAI_API_KEY`.** Las keys conviven.
- **La migración se activa por el NOMBRE DEL MODELO / la URL base, no borrando keys.**
- Rollback = volver el modelo/URL al valor de OpenAI. Sin tocar código.

---

## 2. Qué migra y qué no

| Función | Endpoint | Proveedor posible | Archivo |
| :-- | :-- | :-- | :-- |
| Agente / chat (lo más caro) | `/chat/completions` | OpenAI **o** OpenRouter | `orchestrator_service/main.py` + `llm_provider.py` |
| Transcripción de audio | `/audio/transcriptions` (Whisper) **o** `/chat/completions` (multimodal) | OpenAI, Groq **o** OpenRouter | `whatsapp_service/main.py` + `transcription_provider.py` |

**El proyecto HOY solo usa chat + transcripción.** NO usa embeddings, NO usa
visión (recibe imágenes de WhatsApp pero no las manda a ningún modelo), NO usa
realtime. Si en el futuro se agrega alguna, se migra con el mismo principio
(nombre de modelo con "/" → OpenRouter). Realtime es lo único que OpenRouter no
revende: quedaría en OpenAI o se rediseña aparte.

---

## 3. Chat → OpenRouter

Selector: si `LLM_MODEL` tiene `/` → OpenRouter; si no → OpenAI directo.

**Activar** (EasyPanel → `orchestrator_service` → Medio ambiente):
```
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=openai/gpt-4.1-mini
```
Deploy. Verificar en troncos: `llm_provider_selected ... provider=openrouter`.

> ⚠️ Usar modelos `openai/*` en OpenRouter. Se mantiene function-calling y
> `response_format=json_object` (que el agente necesita). Un modelo no-OpenAI
> podría no soportar eso y romper el agente.
> ⚠️ Si ponés `LLM_MODEL` con `/` pero olvidás `OPENROUTER_API_KEY`, el bot
> tira error a propósito (no arranca el agente). Poné SIEMPRE las dos juntas.

---

## 4. Audio → OpenRouter (o Groq)

OpenRouter tiene Whisper REAL vía el MISMO endpoint que OpenAI
(`/audio/transcriptions`, multipart, soporta `ogg`) — verificado 2026-07. Los tres
proveedores usan el mismo código; solo cambian base_url + key + modelo.

**Opción A (recomendada) — Todo en OpenRouter, un solo proveedor:**
```
TRANSCRIPTION_BASE_URL=https://openrouter.ai/api/v1
TRANSCRIPTION_MODEL=openai/whisper-1
TRANSCRIPTION_API_KEY=sk-or-v1-...     # la MISMA key de OpenRouter del chat
```
Con esto el chat y el audio quedan en OpenRouter: una cuenta, una key, un solo
lugar. Mismo Whisper que hoy, misma calidad.

**Opción B — Groq (Whisper large-v3, aún más barato):**
```
TRANSCRIPTION_BASE_URL=https://api.groq.com/openai/v1
TRANSCRIPTION_MODEL=whisper-large-v3-turbo
TRANSCRIPTION_API_KEY=gsk_...
```
Deploy `whatsapp_service`. Verificar en troncos: `transcription_provider_selected ... provider=...`.

---

## 5. ROLLBACK total a OpenAI (cuando se arregle el saldo)

Sin tocar código. En EasyPanel, en cada servicio, borrar/ajustar variables y reiniciar:

**orchestrator_service:**
```
LLM_MODEL=gpt-4.1-mini          # sin "/" -> vuelve a OpenAI
# (OPENROUTER_API_KEY se puede dejar o borrar; ya no se usa)
```
**whatsapp_service:**
```
TRANSCRIPTION_BASE_URL=https://api.openai.com/v1
TRANSCRIPTION_MODEL=whisper-1
TRANSCRIPTION_API_KEY=           # vacío -> usa OPENAI_API_KEY
```
Reiniciar ambos. Verificar en troncos que digan `provider=openai`. Listo, todo
vuelve a OpenAI. (Alternativa aún más simple: no setear ninguna de estas
variables nunca = comportamiento OpenAI por defecto.)

---

## 6. Tabla de variables (resumen)

| Variable | Servicio | Default (= OpenAI) | Para migrar |
| :-- | :-- | :-- | :-- |
| `OPENAI_API_KEY` | ambos | (tu key) — **NO se toca nunca** | igual |
| `OPENROUTER_API_KEY` | orchestrator | (vacío) | `sk-or-v1-...` |
| `LLM_MODEL` | orchestrator | `gpt-4.1-mini` | `openai/gpt-4.1-mini` |
| `TRANSCRIPTION_BASE_URL` | whatsapp | `https://api.openai.com/v1` | groq.com o openrouter.ai |
| `TRANSCRIPTION_MODEL` | whatsapp | `whisper-1` | `openai/whisper-1` (OpenRouter) / `whisper-large-v3-turbo` (Groq) |
| `TRANSCRIPTION_API_KEY` | whatsapp | (vacío → usa OPENAI_API_KEY) | `gsk_...` / `sk-or-v1-...` |

---

## 7. Notas de mantenimiento

- **`orchestrator_service_backendv2` NO fue migrado** (es código muerto, no está
  en docker-compose). Si algún día se despliega, hay que aplicarle el mismo cambio.
- `whatsapp_service - copia/` tampoco (copia muerta).
- Tests: `tests/test_llm_provider.py` y `tests/test_transcription_provider.py`
  cubren el ruteo y el rollback (corren sin dependencias: `python tests/<archivo>.py`).
- Qué es Groq: proveedor de inferencia con API compatible con OpenAI. Se usa
  igual que OpenAI (misma forma de request), solo cambia base_url + key. Corre
  Whisper large-v3 para audio, muy rápido y barato.
