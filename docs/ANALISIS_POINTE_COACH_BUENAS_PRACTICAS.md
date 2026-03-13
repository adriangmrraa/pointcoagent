# 📊 ANÁLISIS DE BUENAS PRÁCTICAS - SYSTEM PROMPT DE POINTE COACH

## 🎯 **OBJETIVO**
Extraer de manera agnóstica las buenas prácticas, reglas, configuraciones y "edges" que hacen que el sistema Pointe Coach funcione excepcionalmente bien, para poder aplicarlas a otros agentes de IA.

---

## 🔍 **METODOLOGÍA DE ANÁLISIS**
Analizados: 7 versiones de system prompts + documentación técnica + código fuente del orchestrator.

---

## 🏆 **BUENAS PRÁCTICAS IDENTIFICADAS**

### **1. 🎭 PERSONALIDAD Y TONO BIEN DEFINIDOS**

#### **Características clave:**
- **Persona específica:** "Compañera de danza experta" - no un asistente genérico
- **Tono regional:** Español de Argentina con voseo ("vos", "te cuento", "fijate")
- **Naturalidad:** Frases puente como "Mirá", "Dale", "Genial", "Bárbaro"
- **Empatía:** Valida sentimientos del usuario ("Te re entiendo, es difícil dar con el talle online")

#### **Reglas estrictas de estilo:**
- ✅ **Usar:** "vos", frases coloquiales argentinas
- ❌ **Prohibido:** "usted", "su", "has", "podéis", frases de telemarketing
- ✅ **Puntuación:** Solo signo de pregunta al final (`?`), nunca apertura (`¿`)
- ❌ **Prohibido:** Exceso de signos de admiración

#### **Aplicación agnóstica:**
```markdown
DEFINIR PERSONA ESPECÍFICA:
- Rol: [Ej: Asesor técnico, Vendedor especializado, etc.]
- Tono: [Ej: Formal pero cercano, Técnico pero accesible]
- Regionalismo: [Opcional: dialecto/expresiones locales]
- Prohibiciones: [Lista de lo que NO debe usar]
```

### **2. 📚 **DICCIONARIO DE SINÓNIMOS Y ROUTER DE CATEGORÍAS****

#### **Sistema implementado:**
```
USUARIO DICE → MAPEO → CATEGORÍA BASE
"cancán"      →       "Medias"
"malla"       →       "Leotardos" 
"pointe"      →       "Zapatillas de punta"
"slippers"    →       "Media punta"
```

#### **Router de categorías completo:**
```markdown
- ZAPATILLAS DE PUNTA: "puntas", "pointe", "zapatillas de pointe"
- MEDIA PUNTA: "media punta", "ballet", "slippers", "zapatillas de tela"
- MEDIAS / CANCÁN: "medias", "panty", "socks", "convertibles", "cancán", "cancanes"
- ACCESORIOS: "punteras", "cintas", "elásticos", "protector", "separadores", "metatarsianas"
- BOLSOS: "bolso", "mochila", "bag", "bolsa"
- LEOTARDOS / MALLAS: "leotardo", "maillot", "malla", "body"
```

#### **Reglas críticas:**
1. **VALIDATION FIRST:** Antes de buscar, identificar si el usuario pide una categoría del diccionario
2. **RELEVANCIA ESTRICTA:** Si pide "Medias", prohibido mostrar productos de otra categoría
3. **DICCIONARIO OBLIGATORIO:** Mapear CUALQUIER sinónimo a categoría base antes de llamar a tool

#### **Aplicación agnóstica:**
```markdown
IMPLEMENTAR DICCIONARIO DE DOMINIO:
1. Identificar términos clave del dominio
2. Crear mapeo sinónimos → categorías base
3. Incluir en system prompt como "ROUTER DE [DOMINIO]"
4. Forzar uso obligatorio antes de cualquier búsqueda
```

### **3. 🛡️ **SISTEMA ANTI-ALUCINACIÓN (CRÍTICO)****

#### **Reglas de veracidad absoluta:**
- ❌ **Prohibido inventar:** precios, stock, variantes, links, imágenes, estados de pedidos
- ✅ **Solo datos de tools:** Link e imageUrl solo valores exactos devueltos por tools
- ❌ **Prohibido "completar":** No inventar productos para llenar espacios
- ❌ **Prohibido construir URLs:** Nunca "arreglar" dominios/rutas

#### **Gate absoluto de catálogo:**
```markdown
PARA CUALQUIER CONSULTA DE CATÁLOGO:
1. Ejecutar tool de catálogo (search_specific_products / search_by_category)
2. Si tool falla o devuelve vacío → NO listar productos inventados
3. Si no hay tool ejecutada → PROHIBIDO mencionar productos
```

#### **Parche crítico - Anti "respuesta sin tool":**
```markdown
CONDICIÓN: Si usuario pide catálogo y NO se ejecutó tool
ACCIÓN:   Prohibido listar productos, precios, links o imágenes
RAZÓN:    Se considera invención cualquier URL o imagen aunque parezca plausible
```

#### **Aplicación agnóstica:**
```markdown
IMPLEMENTAR GATE DE VERACIDAD:
1. Identificar fuentes de datos confiables (tools/APIs)
2. Establecer regla: "Sin tool → sin datos"
3. Prohibir construcción manual de URLs/valores
4. Incluir fallback honesto: "No tengo esa información disponible"
```

### **4. 🔄 **SISTEMA ANTI-BUCLE Y ANTI-REPETICIÓN****

#### **Reglas de flujo:**
- **ANTI-BUCLE:** Si ya hiciste 1 pregunta y usuario respondió, próximo turno debe avanzar
- **ANTI-REPETICIÓN:** Revisar historial, si usuario pide "más" y tool devuelve mismos productos → NO repetir
- **PROHIBIDO ENCADENAR PREGUNTAS:** No hacer múltiples preguntas seguidas

#### **Estrategia de fallback inteligente (v7):**
```markdown
SI tool devuelve 0 resultados para búsqueda específica:
1. NO rendirse
2. Ejecutar búsqueda más amplia (misma categoría)
3. Usar browse_general_storefront como último recurso
4. Responder: "No encontré X exacto, pero mirá estas opciones similares"
```

#### **Limpieza de palabras:**
- **Eliminar:** adjetivos subjetivos ("lindas", "baratas", "mejores")
- **Buscar solo por:** sustantivos, categorías, marcas/modelos
- **Consultas vagas:** Ejecutar `browse_general_storefront` inmediatamente

#### **Aplicación agnóstica:**
```markdown
IMPLEMENTAR CONTROL DE FLUJO:
1. Revisar historial antes de responder
2. Detectar repeticiones y evitarlas
3. Establecer límite de preguntas por turno
4. Crear estrategia de fallback para búsquedas sin resultados
```

### **5. 🎯 **CALL TO ACTION (CTA) OBLIGATORIO Y ESTRUCTURADO****

#### **Sistema de CTAs contextuales:**
```markdown
CASO 1 (Producto específico - Zapatillas de punta):
  CTA: "Para puntas es clave probarse bien. ¿Te gustaría agendar un fitting?"

CASO 2 (Muchos productos - 3 o más):
  CTA: "Si querés ver más opciones, entrá a nuestra web: {store_website}"

CASO 3 (Pocos productos - 1 o 2 totales):
  CTA: "¿Te puedo ayudar con algo más?" o "Cualquier duda avisame"
```

#### **Reglas:**
- ✅ **Último mensaje SIEMPRE es CTA**
- ✅ **CTA coherente y natural** (no forzado)
- ❌ **No decir "ver más opciones"** si solo hay 1-2 productos
- ✅ **Oferta de servicio** cuando productos son limitados

#### **Formato de presentación estructurado:**
```markdown
SECUENCIA OBLIGATORIA:
1. Intro (saludo/contexto)
2. Producto 1 (con formato específico)
3. Producto 2 (si existe)
4. Producto 3 (si existe)  
5. CTA (contextual)

FORMATO PRODUCTO (TODO EN UNO):
[NOMBRE DEL PRODUCTO]
Precio: $[PRECIO]
Variantes: [LISTA]
[DESCRIPCIÓN RESUMIDA - MÁX 2 LÍNEAS]
[URL SIN ADORNOS]
```

#### **Aplicación agnóstica:**
```markdown
IMPLEMENTAR SISTEMA DE CTAS:
1. Definir CTAs por tipo de interacción
2. Establecer secuencia obligatoria de mensajes
3. Crear formato estructurado para presentación de datos
4. Asegurar que último mensaje siempre sea acción/conclusión
```

### **6. 🔧 **DISTINCIÓN CHISTE VS TÉCNICO (HUMANO VS IA)****

#### **Límites claros de competencia:**
- ✅ **IA PUEDE:** Precio, stock, variantes, links, imágenes básicas
- ❌ **IA NO PUEDE:** Comparativas técnicas profundas, biomecánica, guías complejas
- 🔄 **DERIVACIÓN OBLIGATORIA:** Para preguntas técnicas → tool `derivhumano`

#### **Reglas de interacción:**
1. **PROHIBIDO SER TÉCNICO:** No actuar como especialista en dominio complejo
2. **DERIVACIÓN INMEDIATA:** Para preguntas técnicas/comparativas complejas
3. **BREVEDAD EN PROCESOS:** Al informar estados, ser ultra breve
4. **LÍMITES DE CONSEJO:** Solo argumentos breves del "por qué"

#### **Ejemplo de derivación:**
```markdown
USUARIO: "¿Qué diferencia hay entre Grishko 2007 y Sansha Etoile?"
IA: [Usa tool derivhumano] + "Te derivamos con una asesora especializada..."
```

#### **Aplicación agnóstica:**
```markdown
DEFINIR LÍMITES DE COMPETENCIA:
1. Listar lo que la IA SÍ puede hacer (datos concretos)
2. Listar lo que la IA NO puede hacer (análisis complejos)
3. Establecer triggers para derivación a humano
4. Crear protocolo de derivación con despedida cálida
```

### **7. 📱 **OPTIMIZACIÓN PARA PLATAFORMA (WHATSAPP)****

#### **Reglas de contenido para WhatsApp:**
1. ❌ **PROHIBIDO MARKDOWN:** No usar `###`, `**bold**`, `*italics*`
2. ❌ **PROHIBIDO ETIQUETA "DESCRIPCIÓN":** No escribir "Descripción:"
3. ✅ **ETIQUETAS PERMITIDAS:** "Precio: $" y "Variantes: "
4. ❌ **PROHIBIDO INCLUIR IMAGEN EN TEXTO:** JAMÁS `![...](...)`
5. ✅ **URLS LIMPIAS:** Sin adornos, paréntesis o markdown
6. 📏 **LONGITUD MÁXIMA:** Resumir para evitar corte de WhatsApp

#### **Formato JSON estructurado:**
```json
{
  "messages": [
    { "text": "Intro...", "imageUrl": null },
    { 
      "text": "Producto 1\nPrecio: $...\nVariantes: ...\nDescripción breve\nhttps://url.com",
      "imageUrl": "https://image.url" 
    },
    { "text": "CTA final...", "imageUrl": null }
  ]
}
```

#### **Aplicación agnóstica:**
```markdown
OPTIMIZAR PARA PLATAFORMA:
1. Identificar limitaciones técnicas de la plataforma
2. Crear reglas de formato específicas
3. Establecer longitud máxima por mensaje
4. Diseñar estructura JSON que aproveche características nativas
```

### **8. 🎪 **PRIMERA INTERACCIÓN Y SALUDO ESTRUCTURADO****

#### **Protocolo de saludo:**
```markdown
SI hay intención de búsqueda:
  SALUDO + TOOL + RESULTADOS (mismo turno)

SI es SOLO saludo:
  1. "Hola! ¿Cómo estás? Soy del equipo de [Nombre]."
  2. Si preguntan cómo estás: responder con calidez
  3. Cerrar SIEMPRE con: "¿En qué te ayudo?"
```

#### **Reglas de naturalidad:**
- ✅ **Usar nombre del usuario** de forma natural y esporádica
- ✅ **Principalmente al saludar o derivar**
- ❌ **Evitar repetir** en cada respuesta
- ✅ **Preguntar de vuelta** si usuario pregunta "¿Cómo estás?"

#### **Aplicación agnóstica:**
```markdown
CREAR PROTOCOLO DE SALUDO:
1. Saludo personalizado pero no excesivo
2. Pregunta abierta para iniciar conversación
3. Uso moderado del nombre del usuario
4. Transición suave a propósito de la conversación
```

---

## 🏗️ **ARQUITECTURA DE SYSTEM PROMPT**

### **Estructura jerárquica identificada:**
```
1. PRIORIDADES (Orden absoluto)
2. OBJETIVO
3. REGLA DE VERACIDAD (Crítica)
4. GATE ABSOLUTO DE CATÁLOGO
5. PARCHE CRÍTICO - Anti "respuesta sin tool"
6. TONO Y PERSONALIDAD
7. REGLAS DE INTERACCIÓN (Chiste vs Técnico)
8. PRIMERA INTERACCIÓN
9. REGLAS DE FLUJO (Anti-bucle)
10. TOOLS DISPONIBLES
11. ROUTER DE CATEGORÍA (Diccionario de sinónimos)
12. REGLA DE RESULTADOS (Cantidad)
13. REGLA DE CALL TO ACTION
14. FORMATO DE PRESENTACIÓN
15. GUÍA DE USO DE DATOS
16. REGLAS DE CONTENIDO
17. CONOCIMIENTO DE TIENDA
18. EJEMPLO DE OUTPUT
```

### **Características de escritura:**
- ✅ **Lenguaje directo e imperativo:** "Debés", "Está prohibido", "SIEMPRE"
- ✅ **Énfasis visual:** **NEGRITA**, `código`, CAPS para crítico
- ✅ **Ejemplos concretos:** Casos reales con respuestas esperadas
- ✅ **Estructura clara:** Secciones numeradas, jerarquía visual
- ✅ **Reglas innegociables:** Marcadas como "CRÍTICO", "ABSOLUTO"

---

## 🧠 **PATRONES COGNITIVOS IMPLEMENTADOS**

### **1. Validación antes de acción:**
```markdown
PASO 1: Identificar categoría (Router)
PASO 2: Validar contra diccionario
PASO 3: Ejecutar tool apropiada
PASO 4: Presentar resultados estructurados
```

### **2. Fallback inteligente:**
```markdown
SI búsqueda específica falla → búsqueda amplia
SI búsqueda amplia falla → catálogo general
SI todo falla → admitir límites honestamente
```

### **3. Control de calidad por capas:**
```markdown
CAPA 1: Anti-alucinación (sin tool → sin datos)
CAPA 2: Anti-repetición (revisar historial)
CAPA 3: Anti-bucle (limitar preguntas)
CAPA 4: Formato consistente (estructura fija)
```

---

## 🚀 **PLANTILLA PARA APLICAR A OTROS AGENTES**

### **System Prompt Template:**
```markdown
# System Prompt - [Nombre del Agente]

Eres [descripción de persona/rol específico].

## PRIORIDADES (ORDEN ABSOLUTO)
1. [Prioridad 1 - Ej: Veracidad de datos]
2. [Prioridad 2 - Ej: Formato de salida]
3. [Prioridad 3 - Ej: Experiencia de usuario]

## OBJETIVO
* [Objetivo principal 1]
* [Objetivo principal 2]

## REGLA DE VERACIDAD (CRÍTICA)
* Prohibido inventar: [lista de datos]
* Solo usar valores de: [fuentes confiables]
* Nunca construir: [URLs, valores calculados]

## [DOMINIO] GATE ABSOLUTO
* VALIDATION FIRST: Antes de responder, validar contra [diccionario/reglas]
* RELEVANCIA ESTRICTA: Solo mostrar lo que se pidió
* CONSULTAS VAGAS: Ejecutar [tool general] inmediatamente

## PARCHE CRÍTICO - ANTI "RESPUESTA SIN TOOL"
* Para CUALQUIER consulta de [dominio], ejecutar [tool específica]
* Si tool falla o devuelve vacío: prohibido inventar datos
* Admitir límites honestamente

## TONO Y PERSONALIDAD
* Estilo: [Descripción de tono]
* Prohibido: [Lista de lo que NO usar]
* Naturalidad: [Frases puente/conectores]
* Empatía: [Cómo validar sentimientos]

## REGLAS DE INTERACCIÓN ([DOMINIO] vs TÉCNICO)
1. PROHIBIDO SER TÉCNICO: No actuar como especialista en [dominio complejo]
2. DERIVACIÓN OBLIGATORIA: Para [tipo de preguntas] → usar [tool de derivación]
3. LÍMITES DE CONSEJO: Solo [argumentos breves permitidos]

## PRIMERA INTERACCIÓN
* Si hay intención de [acción]: SALUDO + TOOL + RESULTADOS
* Si es solo saludo: [Protocolo de saludo]
* Cerrar siempre con: [Pregunta abierta]

## REGLAS DE FLUJO (ANTI-BUCLE)
* Si [condición]: NO repreguntar, ejecutar tool
* Revisar historial para evitar repeticiones
* Prohibido encadenar preguntas

## TOOLS DISPONIBLES
1. [tool1]: [descripción y cuándo usar]
2. [tool2]: [descripción y cuándo usar]

## ROUTER DE [DOMINIO] (Diccionario de Sinónimos)
* [CATEGORÍA 1]: [sinónimo1], [sinónimo2], [sinónimo3]
* [CATEGORÍA 2]: [sinónimo1], [sinónimo2]

## REGLA DE RESULTADOS
* OBJETIVO: Mostrar [número] opciones si hay suficientes
* ESCASEZ: Si hay menos, mostrar solo los que hay
* Prohibido inventar para completar

## REGLA DE CALL TO ACTION
* Último mensaje SIEMPRE debe ser CTA
* CTA contextual según [criterios]:
  - CASO 1: [CTA para caso 1]
  - CASO 2: [CTA para caso 2]

## FORMATO DE PRESENTACIÓN ([PLATAFORMA])
* Secuencia obligatoria: [paso1] → [paso2] → [paso3]
* Estructura para [tipo de dato]:
  [FORMATO ESPECÍFICO]

## GUÍA DE USO DE DATOS
* Tool [campo1] → [cómo presentar]
* Tool [campo2] → [cómo presentar]

## REGLAS DE CONTENIDO ([PLATAFORMA])
1. PROHIBIDO: [formato1], [formato2]
2. PERMITIDO: [formato3], [formato4]
3. LONGITUD MÁXIMA: [límite]

## CONOCIMIENTO DE [DOMINIO]
[Información específica del dominio]

## EJEMPLO DE OUTPUT
```json
[Ejemplo completo de salida esperada]
```
```

---

## 🎯 **CASOS DE ÉXITO ESPECÍFICOS DE POINTE COACH**

### **1. Edge Case: "No sé qué elegir"**
```markdown
USUARIO: "No sé qué elegir"
IA ANTES: "¿Qué tipo de producto buscas? ¿Para qué nivel?"
IA DESPUÉS: [Ejecuta browse_general_storefront] + muestra 3 opciones reales
```

**Aprendizaje:** Para consultas vagas, acción inmediata > preguntas adicionales.

### **2. Edge Case: "Mostrame más" (sin más productos)**
```markdown
USUARIO: "¿Tenés más opciones?" (después de ver 3 productos)
IA: "Esos son todos los modelos que tenemos por ahora en esa categoría."
```

**Aprendizaje:** Revisar historial, admitir límites honestamente, no inventar.

### **3. Edge Case: Término coloquial "cancán"**
```markdown
USUARIO: "Buscá cancán"
IA: [Mapea "cancán" → "Medias"] + busca en categoría Medias
```

**Aprendizaje:** Diccionario de sinónimos obligatorio antes de cualquier búsqueda.

### **4. Edge Case: Pregunta técnica sobre biomecánica**
```markdown
USUARIO: "¿Qué zapatilla es mejor para pie egipcio?"
IA: [Usa tool derivhumano] + "Te derivamos con una asesora especializada..."
```

**Aprendizaje:** Conocer límites de competencia y derivar apropiadamente.

---

## 🔧 **IMPLEMENTACIÓN TÉCNICA OBSERVADA**

### **1. Inyección de variables dinámicas:**
```python
# En el código se observa:
sys_template = sys_template.replace("{store_name}", store_name)
sys_template = sys_template.replace("{store_description}", store_description)
```

**Buena práctica:** Templates con placeholders para personalización por tenant.

### **2. Validación de queries:**
```python
# Limpieza de palabras antes de búsqueda
query = clean_query(user_input)  # Elimina adjetivos subjetivos
```

**Buena práctica:** Preprocesamiento de queries para mejor matching.

### **3. Sistema de fallback:**
```python
if specific_search_empty:
    return broader_search()
elif broader_search_empty:
    return general_catalog()
else:
    return honest_no_results()
```

**Buena práctica:** Jerarquía de intentos antes de rendirse.

---

## 📊 **MÉTRICAS DE ÉXITO IMPLÍCITAS**

### **1. Tasa de finalización de conversación:**
- CTAs obligatorios aumentan conversión
- Derivación apropiada mejora satisfacción

### **2. Tasa de error por alucinación:**
- Gate de veracidad reduce inventos
- Sistema anti "respuesta sin tool" elimina datos falsos

### **3. Satisfacción del usuario:**
- Tono cálido y natural mejora percepción
- Límites honestos generan confianza
- Derivación oportuna evita frustración

---

## 🧪 **TESTING Y VALIDACIÓN SUGERIDOS**

### **Casos de prueba críticos:**
1. **Consulta vaga:** "¿Qué tienen?" → Debe mostrar productos reales
2. **Sinónimo coloquial:** "cancán" → Debe mapear a "Medias"
3. **Búsqueda sin resultados:** "Zapatillas rojas talla 50" → Debe hacer fallback honesto
4. **Pregunta técnica:** "¿Cuál es mejor?" → Debe derivar
5. **Repetición:** "Mostrame más" después de ver todo → No debe inventar

### **Validación de output:**
1. **Estructura JSON:** Validar schema exacto
2. **Formato WhatsApp:** Sin markdown, URLs limpias
3. **CTA presente:** Último mensaje siempre es acción
4. **Datos reales:** Todos los valores deben venir de tools

---

## 🚀 **RECOMENDACIONES PARA IMPLEMENTACIÓN**

### **Fase 1: Diseño**
1. **Definir persona y tono** específicos para el dominio
2. **Crear diccionario de sinónimos** completo
3. **Establecer límites de competencia** claros
4. **Diseñar sistema de CTAs** contextuales

### **Fase 2: Implementación**
1. **Implementar gate de veracidad** anti-alucinación
2. **Crear router de categorías** con mapeo de sinónimos
3. **Establecer reglas de flujo** anti-bucle
4. **Optimizar formato** para plataforma objetivo

### **Fase 3: Testing**
1. **Probar casos edge** identificados
2. **Validar output** contra schema esperado
3. **Medir métricas** de éxito
4. **Iterar** basado en feedback

---

## 🎓 **LECCIONES CLAVE PARA CUALQUIER AGENTE**

### **1. Especificidad > Generalidad**
- ❌ Asistente genérico
- ✅ Rol específico con personalidad definida

### **2. Veracidad > Completitud**
- ❌ Inventar para dar respuesta completa
- ✅ Admitir límites honestamente

### **3. Acción > Preguntas**
- ❌ Encadenar preguntas indefinidamente
- ✅ Acción inmediata para consultas vagas

### **4. Estructura > Libertad**
- ❌ Respuestas libres sin formato
- ✅ Estructura fija con CTAs obligatorios

### **5. Dominio > IA general**
- ❌ Conocimiento general de IA
- ✅ Conocimiento específico del dominio + diccionario

---

## 📈 **EVOLUCIÓN OBSERVADA EN LAS VERSIONES**

### **v1 → v7 Mejoras clave:**
1. **Diccionario de sinónimos** más completo
2. **Sistema de fallback** inteligente agregado
3. **Reglas de limpieza** de queries mejoradas
4. **CTAs** más contextuales y naturales
5. **Protocolos de derivación** más claros

**Patrón:** Cada versión añade más "edges" específicos y reglas para casos particulares.

---

## 🏁 **CONCLUSIÓN**

El éxito de Pointe Coach radica en **sistemas específicos y bien definidos**, no en IA general poderosa:

### **Pilares del éxito:**
1. **🎭 Personalidad específica y consistente**
2. **📚 Diccionario de sinónimos obligatorio**
3. **🛡️ Sistema anti-alucinación robusto**
4. **🔄 Flujos controlados anti-bucle**
5. **🎯 CTAs contextuales obligatorios**
6. **🔧 Límites de competencia claros**
7. **📱 Optimización para plataforma**

### **Aplicación a otros agentes:**
Cualquier agente puede mejorar significativamente adoptando:
1. **Especificidad** en persona y dominio
2. **Sistemas** (no solo reglas) para edge cases
3. **Estructura** fija en outputs
4. **Honestidad** sobre límites de conocimiento

**La clave:** Diseñar sistemas que guíen a la IA, no confiar en que "sepa" qué hacer.

---

## 📚 **RECURSOS ADICIONALES EN EL REPOSITORIO**

### **Archivos analizados:**
1. `docs/system_prompt_final.md` - Versión de producción
2. `docs/system_prompt_v1.md` a `v7.md` - Evolución
3. `docs/04_agent_logic_and_persona.md` - Diseño de persona
4. `orchestrator_service/main.py` - Implementación técnica

### **Patrones identificables:**
- Iteración constante basada en casos reales
- Adición progresiva de "edges" específicos
- Balance entre flexibilidad y control
- Enfoque en experiencia de usuario final

---

**✨ Este análisis proporciona un framework completo para diseñar agentes de IA efectivos basados en las buenas prácticas probadas de Pointe Coach.**