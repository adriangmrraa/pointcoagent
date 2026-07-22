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

No hay embeddings ni realtime en el proyecto: estos dos son los ÚNICOS consumos de IA.

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

## 4. Audio → Groq u OpenRouter

Selector por `TRANSCRIPTION_BASE_URL`:
- `...openai.com...` o `...groq.com...` → Whisper multipart (`/audio/transcriptions`).
- `...openrouter.ai...` → modelo multimodal con audio en base64 (`/chat/completions`).

**Opción A — Todo en OpenRouter (un solo proveedor):**
```
TRANSCRIPTION_BASE_URL=https://openrouter.ai/api/v1
TRANSCRIPTION_MODEL=google/gemini-2.0-flash-001    # debe aceptar audio
TRANSCRIPTION_API_KEY=sk-or-v1-...
```

**Opción B — Groq (Whisper dedicado, mejor transcripción, muy barato):**
```
TRANSCRIPTION_BASE_URL=https://api.groq.com/openai/v1
TRANSCRIPTION_MODEL=whisper-large-v3-turbo
TRANSCRIPTION_API_KEY=gsk_...
```
Deploy `whatsapp_service`. Verificar en troncos: `transcription_provider_selected ... provider=... mode=...`.

> ⚠️ **Probar con notas de voz reales antes de confiar.** El modo OpenRouter usa
> un LLM multimodal para transcribir: es más flexible pero puede resumir o
> equivocarse más que Whisper. Groq (Whisper large-v3) es lo más fiel para
> español. Recomendación: si el audio importa mucho para entender el pedido,
> usar Groq (Opción B); si se prioriza "un solo proveedor", OpenRouter (Opción A)
> validando calidad primero.
> ⚠️ OpenRouter: no todos los modelos aceptan `ogg` (formato de WhatsApp).
> `google/gemini-2.0-flash-001` sí. Si cambiás de modelo, verificar.

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
| `TRANSCRIPTION_MODEL` | whatsapp | `whisper-1` | `whisper-large-v3-turbo` / `google/gemini-2.0-flash-001` |
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
