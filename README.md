# Spider Filter

Filter kamera interaktif satu-sesi: **cari wajah → gesture bingkai foto →
pasang topeng → tembak jaring laba-laba → fade ke hitam**. Dibuat untuk
difilmkan dari layar (kamera HP mengarah ke layar laptop), bukan untuk
merekam langsung dari program.

Semua deteksi (wajah & tangan) memakai **MediaPipe 1.0.0 Tasks API**
(`mediapipe.tasks.python.vision`) — bukan `mp.solutions` lama, yang sudah
dihapus total di versi ini.

## Instalasi

Butuh Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Berkas yang perlu kamu siapkan

Program **tidak akan berjalan** tanpa empat berkas ini (masing-masing ada
`README.md` di foldernya dengan instruksi lebih detail):

```
models/hand_landmarker.task              (lihat models/README.md)
models/blaze_face_short_range.tflite     (lihat models/README.md)
assets/mask.png                          (punya kamu sendiri, PNG + alpha)
assets/song.mp3                          (punya kamu sendiri, ~60 detik)
```

Kalau ada yang hilang, program berhenti dengan pesan jelas yang
menyebutkan berkas mana saja yang belum ada — bukan traceback mentah.

## Menjalankan

```bash
python spider_filter.py
```

Jendela kamera akan tampil **layar penuh, mendatar**, dengan lapisan
diagnostik neon (landmark tangan, kotak wajah, teks status) menumpuk di
atas gambar kamera. Status setiap perpindahan fase & gesture juga
dicetak ke terminal — taruh jendela terminal di samping jendela kamera
kalau ingin keduanya terlihat saat difilmkan.

## Alur fase

1. **SEARCHING** — cari wajah; begitu terkunci ~0,4 detik, lagu mulai
   diputar dari awal & masuk Fase 2. Wajah masih polos, belum ada topeng.
2. **FRAME GESTURE** — bentuk bingkai foto dari kedua tangan (huruf L
   berseberangan diagonal, tidak bersentuhan) dan tahan ~0,4 detik.
3. **PULL** — rentangkan kedua tangan sambil naik supaya bingkai
   memekar (>1.6× ukuran awal) hingga melingkupi wajah → topeng terpasang
   dengan animasi menyusut-menyatu.
4. **MASK ON** — topeng mengikuti posisi/skala/rotasi wajah. Ulangi
   gesture bingkai kecil→memekar untuk **melepas** topeng (toggle).
5. **WEB SHOT** — hanya aktif saat topeng terpasang. Kunci pose
   jempol+telunjuk+kelingking terbuka (~0,3 detik), lalu hentakkan tangan
   ke luar layar untuk menembakkan jaring. Menumpuk terus sampai layar
   ~75% tertutup jaring.
6. **ENDING** — layar menghitam (~1,5 detik) bersamaan lagu fade out
   (~2 detik, sengaja lebih lama), tahan hitam ~1,5 detik, lalu program
   berhenti sendiri dengan rapi.

## Kontrol keyboard

| Tombol | Fungsi |
|---|---|
| `q` | Keluar paksa |
| `r` | Reset ke Fase 1 (retake tanpa restart program) |
| `m` | Mute / unmute lagu |
| `h` | Sembunyikan / tampilkan lapisan diagnostik |
| `f` | Keluar / masuk mode layar penuh |

## Menyetel perilaku

Semua nilai yang bisa disetel (skala topeng, durasi hold gesture, ambang
kecepatan hentakan, radius jaring, ambang cakupan layar, durasi fade,
resolusi kamera, warna diagnostik, dll) dikumpulkan dalam satu class `K`
di bagian atas `spider_filter.py`, masing-masing dengan komentar
penjelas berbahasa Indonesia.

## Performa

Default resolusi kamera **960×540** dipilih supaya deteksi MediaPipe +
render topeng/jaring tetap di atas target 20 FPS di CPU tanpa GPU. Kalau
laptopmu cukup kencang, naikkan `K.KAMERA_LEBAR` / `K.KAMERA_TINGGI` ke
1280×720 di blok konstanta.

## Struktur kode

- `K` — blok konstanta.
- `klasifikasi_bentuk_tangan()`, `evaluasi_pasangan_bingkai()` — fungsi
  pengenalan gesture per-pose, terpisah dari alur utama.
- `PelacakBingkai` — state machine gesture bingkai (MENUNGGU → MENGUNCI →
  MEMEKAR → trigger/batal).
- `PelacakWebShooter` / `SlotTanganWeb` — state machine web-shooter
  per-tangan (NETRAL → TERKUNCI → menembak → NETRAL).
- `LapisanJaring` / `AnimasiJaring` — kanvas jaring persisten (pola yang
  sudah selesai "dipanggang" ke kanvas statis, tidak digambar ulang).
- `Topeng` — pembungkus rotasi/skala/blending topeng (`cv2.warpAffine`).
- `PelacakWajah` / `PelacakTangan` — pembungkus MediaPipe Tasks API.
- `SpiderFilterApp` — orkestrasi fase & loop utama.
