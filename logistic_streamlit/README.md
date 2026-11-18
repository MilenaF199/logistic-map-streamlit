# Logistic Map AI Lab

Aplicación interactiva de Streamlit que compara la dinámica del mapa logístico con las predicciones de una red neuronal MLP.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar

Sube `app.py` y `requirements.txt` a un repositorio de GitHub. En https://share.streamlit.io inicia sesión con GitHub, selecciona el repositorio y establece `app.py` como archivo principal. Pulsa Deploy y comparte el enlace público.

## Nota

Este es un modelo nuevo entrenado con datos sintéticos, no el modelo guardado del proyecto académico anterior. El modo recursivo recorta predicciones al intervalo [0, 1].
