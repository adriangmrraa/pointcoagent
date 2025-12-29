# Plan de Arreglo: Platform UI (Service Yellow/Error)

El servicio `platform_ui` falla en el despliegue de EasyPanel principalmente por una URL "hardcoded" (estática) en la configuración de Nginx que no coincide con tu entorno actual.

## Cambios Propuestos

### 1. Eliminar URL estática en Nginx
Se modificará `platform_ui/nginx.conf` para que sea genérico. La aplicación ya detecta la URL del orquestador dinámicamente en el navegador, por lo que el proxy interno de Nginx es redundante y causa errores si el dominio no existe.

### 2. Asegurar Variables de Entorno en EasyPanel
Para que la UI sepa dónde está el "cerebro" (Orquestador), debes configurar estas variables en el servicio `platform_ui` dentro de EasyPanel:
- `API_BASE_URL`: La URL pública de tu `orchestrator_service` (ej: `https://tu-orquestador.easypanel.host`)
- `ADMIN_TOKEN`: Debe coincidir con el token del orquestador.

## Instrucciones de Aplicación

1. **Modificar Archivo**: Aplicaré un cambio a [nginx.conf](file:///c:/Users/Asus/Downloads/Pointe%20Coach%20Agent%202025/platform_ui/nginx.conf) para quitar la dependencia del dominio antiguo.
2. **Re-Deploy**: Deberás hacer un nuevo "Deploy" desde EasyPanel una vez que yo suba los cambios.

> [!IMPORTANT]
> Si el servicio sigue en amarillo tras el cambio, verifica en los logs de EasyPanel si hay un error de "Port Mapping". La UI usa el puerto **80** internamente.
