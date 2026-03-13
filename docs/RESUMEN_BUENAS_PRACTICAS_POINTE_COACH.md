# 🎯 RESUMEN EJECUTIVO - BUENAS PRÁCTICAS POINTE COACH

## 🏆 **LAS 7 BUENAS PRÁCTICAS CLAVE**

### **1. 🎭 PERSONALIDAD ESPECÍFICA Y CONSISTENTE**
- **No es un asistente genérico**, es una "compañera de danza experta"
- **Tono regional definido:** Español argentino con voseo
- **Reglas estrictas de estilo:** Prohibido telemarketing, frases acartonadas
- **Empatía integrada:** Valida sentimientos, pregunta de vuelta

### **2. 📚 **DICCIONARIO DE SINÓNIMOS OBLIGATORIO****
- **Router de categorías:** "cancán" → "Medias", "malla" → "Leotardos"
- **Validación primero:** Antes de buscar, mapear sinónimos
- **Relevancia estricta:** Si pide "Medias", solo mostrar medias
- **Consultas vagas:** Acción inmediata (`browse_general_storefront`)

### **3. 🛡️ **SISTEMA ANTI-ALUCINACIÓN ROBUSTO****
- **Gate absoluto:** Sin tool ejecutada → sin datos mencionados
- **Prohibido inventar:** Precios, stock, URLs, imágenes
- **Parche crítico:** Anti "respuesta sin tool"
- **Honestidad sobre límites:** "Es lo único que tenemos por ahora"

### **4. 🔄 **FLUJOS CONTROLADOS ANTI-BUCLE****
- **Anti-repetición:** Revisar historial, no mostrar mismos productos
- **Anti-bucle:** No encadenar preguntas, avanzar después de respuesta
- **Fallback inteligente:** Si búsqueda específica falla → búsqueda amplia
- **Limpieza de queries:** Eliminar adjetivos subjetivos antes de buscar

### **5. 🎯 **CTAS CONTEXTUALES OBLIGATORIOS****
- **Último mensaje SIEMPRE es CTA**
- **Contextual por caso:**
  - Puntas de danza → Ofrecer fitting
  - Muchos productos → Link a web
  - Pocos productos → Oferta de servicio
- **Estructura fija:** Intro → Productos → CTA

### **6. 🔧 **LÍMITES DE COMPETENCIA CLAROS****
- **IA PUEDE:** Datos concretos (precio, stock, variantes)
- **IA NO PUEDE:** Análisis técnicos, comparativas complejas
- **Derivación obligatoria:** Para preguntas técnicas → `derivhumano`
- **Protocolo de derivación:** Despedida cálida + explicación

### **7. 📱 **OPTIMIZACIÓN PARA PLATAFORMA****
- **Formato WhatsApp:** Sin markdown, URLs limpias
- **Longitud controlada:** Resumir para evitar corte
- **Estructura JSON específica:** Campos `text` e `imageUrl`
- **Imágenes separadas:** Nunca incluir `![...](...)` en texto

---

## 🎪 **EDGE CASES BIEN MANEJADOS**

### **Consulta vaga:**
```
Usuario: "No sé qué elegir"
IA: [Ejecuta browse_general_storefront] + muestra 3 opciones reales
```

### **Sinónimo coloquial:**
```
Usuario: "Buscá cancán"
IA: [Mapea "cancán" → "Medias"] + busca en categoría Medias
```

### **Sin resultados:**
```
Usuario: "Zapatillas rojas talla 50"
IA: "No encontré ese color/talle exacto, pero mirá estas opciones similares"
```

### **Pregunta técnica:**
```
Usuario: "¿Qué zapatilla es mejor para pie egipcio?"
IA: [Deriva a humano] + "Te derivamos con una asesora especializada"
```

---

## 🏗️ **ARQUITECTURA DEL SYSTEM PROMPT**

### **Estructura jerárquica:**
1. **PRIORIDADES** (orden absoluto)
2. **OBJETIVO**
3. **REGLA DE VERACIDAD** (crítica)
4. **GATE DE CATÁLOGO** (inegociable)
5. **TONO Y PERSONALIDAD**
6. **REGLAS DE INTERACCIÓN**
7. **ROUTER DE CATEGORÍAS** (diccionario)
8. **CTAS OBLIGATORIOS**
9. **FORMATO DE PRESENTACIÓN**
10. **EJEMPLO DE OUTPUT**

### **Características de escritura:**
- **Lenguaje imperativo:** "Debés", "Está prohibido", "SIEMPRE"
- **Énfasis visual:** Negritas, CAPS para crítico
- **Ejemplos concretos:** Casos reales con respuestas
- **Estructura clara:** Secciones numeradas

---

## 🚀 **CÓMO APLICAR A OTROS AGENTES**

### **Paso 1: Definir identidad**
- Rol específico (no genérico)
- Tono y personalidad definidos
- Reglas de estilo estrictas

### **Paso 2: Crear diccionario de dominio**
- Listar términos clave
- Mapear sinónimos → categorías base
- Incluir en prompt como "ROUTER"

### **Paso 3: Implementar gate de veracidad**
- "Sin tool → sin datos"
- Prohibir invención de valores
- Admitir límites honestamente

### **Paso 4: Diseñar flujos controlados**
- Anti-repetición (revisar historial)
- Anti-bucle (limitar preguntas)
- Fallback inteligente

### **Paso 5: Crear CTAs contextuales**
- Último mensaje siempre es acción
- CTAs específicos por tipo de interacción
- Estructura fija de presentación

### **Paso 6: Establecer límites de competencia**
- Listar lo que SÍ puede hacer
- Listar lo que NO puede hacer
- Protocolo de derivación claro

### **Paso 7: Optimizar para plataforma**
- Formato específico de la plataforma
- Longitudes máximas
- Estructura de output

---

## 📊 **MÉTRICAS IMPLÍCITAS DE ÉXITO**

1. **✅ Tasa de finalización:** CTAs aumentan conversión
2. **✅ Satisfacción:** Tono cálido mejora percepción
3. **✅ Precisión:** Gate anti-alucinación reduce errores
4. **✅ Eficiencia:** Flujos controlados reducen bucles
5. **✅ Confianza:** Límites honestos generan credibilidad

---

## 🎓 **LECCIÓN PRINCIPAL**

**El éxito no está en la IA general, sino en los sistemas específicos que la guían:**

- **Diccionarios** > Conocimiento general
- **Reglas estrictas** > Flexibilidad abierta
- **Estructuras fijas** > Libertad creativa
- **Límites claros** > Competencia ilimitada
- **Edge cases anticipados** > Reacción improvisada

---

## 🔗 **ARCHIVOS DE REFERENCIA**

1. `system_prompt_final.md` - Versión de producción
2. `system_prompt_v1.md` a `v7.md` - Evolución del diseño
3. `04_agent_logic_and_persona.md` - Diseño de identidad
4. `orchestrator_service/main.py` - Implementación técnica

---

**✨ Este framework puede aplicarse a cualquier agente de IA para mejorar significativamente su efectividad, precisión y experiencia de usuario.**