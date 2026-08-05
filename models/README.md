# Model MediaPipe

Letakkan dua berkas model MediaPipe Tasks API di folder ini (tidak
disertakan di repo karena ukurannya besar):

```
models/hand_landmarker.task
models/blaze_face_short_range.tflite
```

## Cara mengunduh

```bash
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

curl -L -o models/blaze_face_short_range.tflite \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
```

Kalau link di atas berubah, cari model terbaru di halaman resmi:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
dan
https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

`spider_filter.py` akan memeriksa keberadaan kedua berkas ini saat
dijalankan dan berhenti dengan pesan jelas kalau belum ada.
