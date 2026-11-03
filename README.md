# SAM2 Video Annotator

Полуавтоматическая разметка видео в графическом интерфейсе.

## Установка зависимостей

```bash
pip install -r requirements-sam2.txt
```

## Запуск

Из папки проекта:

```bash
python sam2_video_annotator.py --device cuda
```

## Флаги

| Флаг | По умолчанию | Назначение |
| --- | --- | --- |
| `--device` | `auto` | Устройство для SAM2. Варианты `auto`, `cuda`, `cpu`, `mps`. |
| `--checkpoint` | `models/sam2/sam2.1_hiera_tiny.pt` | Путь к весам модели. |
| `--config` | `configs/sam2.1/sam2.1_hiera_t.yaml` | Конфигурация, соответствующая весам. |
| `--object-id` | `1` | Начальный идентификатор объекта. |
| `--max-display-width` | `980` | Максимальная ширина кадра в интерфейсе в пикселях. |
| `--max-sam-width` | `0` | Максимальная ширина кадра для SAM2. При `0` сохраняется исходный размер. |
| `--jpeg-quality` | `85` | Качество временных JPEG-кадров от 1 до 100. |
| `--frame-step` | `1` | Обрабатывать каждый N-й кадр. Для пропущенных кадров рамки интерполируются. |
| `--max-frames` | `0` | Взять первые N кадров до применения шага. При `0` используется всё видео. |
| `--vos-optimized` | выключен | Включить оптимизацию через `torch.compile`. Первый запуск может быть дольше. |

```bash
python sam2_video_annotator.py "/mnt/c/Videos/input.mp4" --device cuda --frame-step 2 --max-frames 300
```
