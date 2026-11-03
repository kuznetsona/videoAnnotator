# SAM2 Video Annotator

Полуавтоматическая разметка видео в графическом интерфейсе.

## Установка зависимостей

В WSL из папки проекта создай Conda-окружение и установи PyTorch для CUDA 12.9.

```bash
conda create -n video-annotator-cu129 python=3.12 pip tk -y
conda activate video-annotator-cu129
python -m pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-sam2.txt \
  -c <(printf '%s\n' 'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129')
python -m pip check
```

Параметр `-c` фиксирует CUDA-сборки при установке остальных зависимостей.

## Запуск

Из папки проекта:

```bash
python sam2_video_annotator.py --device cuda
```

## Управление разметкой

1. Нажми **Open video**. Перейди к нужному кадру ползунком или кнопками **Previous frame** и **Next frame**.
2. Выбери **Active object ID**. Щёлкни левой кнопкой по объекту, правой — по фону. Чтобы уточнить разметку, добавь точки на других кадрах с тем же ID. Для другого объекта выбери новый ID.
3. Нажми **Run annotation**. Результаты можно открыть кнопкой **Open output folder**. Ползунок **Mask opacity** меняет прозрачность маски в окне.

**Undo active-object prompt** отменяет последнюю точку выбранного объекта. **Clear active-object prompts** удаляет все его точки, а **Clear all prompts** очищает точки всех объектов.

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
