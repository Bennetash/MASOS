# Proyecto: MASOS/MAAC en Crazyflie

## Objetivos
1. Reproducir el modelo MASOS con MAAC en entornos 2D.
2. Integrar el modelo MASOS en Webots/Isaac Lab para simular 4 Crazyflie.
3. Validación y adaptación para transferencia a hardware real (Crazyflie).

## Estructura del Proyecto
- `src/` - Código fuente para la política MASOS.
- `simulator/` - Scripts para la integración con Webots/Isaac Lab.
- `drones/` - Configuración específica de Crazyflie, decks y comunicaciones.
- `tests/` - Test de unidades, integración y simulación.

## Reglas del Proyecto
- Mantener el modelo de 2D inalterado en su rama.
- No mezclar lógica de control de vuelo y aprendizaje automático directamente.
- Validar por etapas: simulación básica → 1 dron → 2 drones → 4 drones.
- Los cambios en física deben ser testados primero en entornos simples.