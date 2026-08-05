#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spider_filter.py
=================

Filter kamera interaktif bertema "Spider" — satu sesi berurutan dari
pencarian wajah, gesture bingkai foto, pemasangan topeng, tembakan jaring
laba-laba, sampai transisi hitam di akhir. Dibuat untuk difilmkan dari
layar (bukan merekam langsung dari program ini).

-----------------------------------------------------------------------
CARA INSTALL
-----------------------------------------------------------------------
Butuh Python 3.10+. Disarankan pakai virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt

Atau install manual:

    pip install opencv-python "mediapipe==1.0.0" pygame numpy

-----------------------------------------------------------------------
BERKAS YANG DIBUTUHKAN (lihat README.md untuk detail & link unduhan)
-----------------------------------------------------------------------
    models/hand_landmarker.task
    models/blaze_face_short_range.tflite
    assets/mask.png     (PNG dengan alpha channel)
    assets/song.mp3     (satu track lagu, ~60 detik)

Program akan berhenti dengan pesan yang jelas kalau salah satu berkas ini
tidak ditemukan — bukan traceback mentah.

-----------------------------------------------------------------------
CARA MENJALANKAN
-----------------------------------------------------------------------
    python spider_filter.py

Kontrol keyboard:
    q  = keluar paksa
    r  = reset ke Fase 1 (retake dari awal, tanpa restart program)
    m  = mute / unmute lagu
    h  = sembunyikan / tampilkan lapisan diagnostik
    f  = keluar / masuk mode layar penuh

-----------------------------------------------------------------------
PERINGATAN API MEDIAPIPE (penting!)
-----------------------------------------------------------------------
Program ini memakai MediaPipe 1.0.0. Di versi ini `mediapipe.solutions`
SUDAH DIHAPUS. Semua deteksi wajah & tangan di sini memakai Tasks API
(`mediapipe.tasks.python.vision`), bukan API lama `mp.solutions.*`.
"""

import os
import sys
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple

import numpy as np
import cv2

# pygame dipakai khusus untuk audio (mixer), bukan untuk jendela/tampilan.
import pygame

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# =========================================================================
# LOKASI BERKAS (relatif terhadap file ini — tidak ada path absolut)
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TANGAN = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
MODEL_WAJAH = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")
ASET_TOPENG = os.path.join(BASE_DIR, "assets", "mask.png")
ASET_LAGU = os.path.join(BASE_DIR, "assets", "song.mp3")


# =========================================================================
# KONSTANTA — semua nilai yang bisa disetel dikumpulkan di sini
# =========================================================================
class K:
    """Namespace konstanta. Ubah nilai di sini untuk menyetel perilaku
    filter tanpa perlu mengubah logika utama."""

    # --- Kamera ---
    # 960x540 dipilih sebagai default supaya deteksi MediaPipe + blending
    # topeng/jaring tetap di atas 20 FPS di laptop tanpa GPU. Naikkan ke
    # 1280x720 kalau laptop kamu cukup kencang (sudah diuji tetap lancar
    # di CPU modern, hanya sedikit lebih berat).
    KAMERA_INDEX = 0
    KAMERA_LEBAR = 960
    KAMERA_TINGGI = 540

    # --- Topeng: skala & posisi ---
    TOPENG_SKALA = 2.6          # lebar topeng = jarak antar mata * nilai ini
    TOPENG_OFFSET_Y = 0.55      # offset vertikal topeng (rasio thd jarak mata, + = ke bawah)
    TOPENG_ROTASI_FAKTOR = 1.0  # balik ke -1.0 kalau arah miring topeng terbalik

    # --- Durasi tahan gesture (hold time) ---
    DURASI_KUNCI_WAJAH = 0.4        # Fase 1: wajah harus stabil berapa detik sebelum lagu mulai
    DURASI_KUNCI_BINGKAI = 0.4      # Fase 2: pose bingkai harus tertahan berapa detik
    DURASI_KUNCI_WEB_SHOOTER = 0.3  # Fase 5 tahap 1: pose web-shooter harus tertahan berapa detik

    # --- Pemekaran bingkai (Fase 3 / PULL) ---
    RASIO_PEMEKARAN_BINGKAI = 1.6       # diagonal harus tumbuh berapa kali lipat agar terpicu
    AMBANG_TUMPANG_TINDIH_WAJAH = 0.55  # seberapa besar bingkai harus menutup kotak wajah (0..1)
    FAKTOR_CADANGAN_PEMEKARAN = 0.8     # toleransi cadangan kalau tangan hilang saat memekar
    TOLERANSI_FRAME_GAGAL = 6           # berapa frame gagal beruntun sebelum pose dianggap lepas
    TOLERANSI_SUDUT_L_KETAT = (60.0, 120.0)   # toleransi sudut ibu jari-telunjuk saat mengunci
    TOLERANSI_SUDUT_L_LONGGAR = (35.0, 145.0)  # toleransi lebih longgar saat memekar
    AMBANG_ARAH_BERLAWANAN = -0.15      # dot product arah buka L kedua tangan (harus negatif)

    # --- Animasi pasang / lepas topeng ---
    DURASI_ANIMASI_PASANG = 0.4   # detik, topeng menyusut dari ukuran bingkai ke wajah
    DURASI_ANIMASI_LEPAS = 0.4    # detik, topeng membesar dari wajah ke ukuran bingkai lalu pudar
    COOLDOWN_TOGGLE_TOPENG = 1.0  # jeda kebal setelah pasang/lepas supaya tidak berkedip

    # --- Anti-kedip topeng saat wajah hilang sesaat ---
    TOLERANSI_WAJAH_HILANG_FRAME = 10

    # --- Web shooter (Fase 5) ---
    AMBANG_KECEPATAN_HENTAKAN = 900.0   # px/detik, ambang lonjakan kecepatan telapak
    COOLDOWN_TEMBAK_PER_TANGAN = 0.3    # jeda kebal per tangan setelah menembak
    KEKUATAN_BIAS_ARAH = 1.35           # seberapa jauh pola jaring dibiaskan ke arah hentakan
    RADIUS_MAKS_JARING = 480            # radius maksimal (px) satu pola jaring
    DURASI_ANIMASI_JARING = 0.55        # detik, kecepatan menyebar satu pola jaring
    JUMLAH_GARIS_RADIAL = 10            # jumlah garis lurus memancar per pola
    JUMLAH_BUSUR = 4                    # jumlah cincin busur melengkung per pola
    RIWAYAT_POSISI_TELAPAK = 6          # jumlah sampel posisi telapak yang disimpan (kecepatan)
    TOLERANSI_TANGAN_HILANG_SLOT = 1.2  # detik sebelum slot tangan dianggap benar-benar kosong

    # --- Ambang penutupan layar -> pemicu ending ---
    AMBANG_CAKUPAN_LAYAR = 0.75  # 75% piksel layar tertutup jaring -> mulai Fase 6

    # --- Ending ---
    DURASI_FADE_LAYAR = 1.5      # detik, layar menghitam
    DURASI_FADE_AUDIO = 2.0      # detik, fade out lagu (sengaja lebih panjang dari fade layar)
    DURASI_TAHAN_HITAM = 1.5     # detik, layar hitam ditahan setelah lagu habis
    DURASI_ANIMASI_PEMADAT_AKHIR = 0.5  # detik, animasi jaring "merapat" penuh sebelum fade

    # --- Audio ---
    VOLUME_LAGU = 0.85

    # --- Lapisan diagnostik (warna neon, BGR untuk OpenCV) ---
    WARNA_LANDMARK_TANGAN = (255, 0, 255)   # magenta
    WARNA_GARIS_TANGAN = (255, 255, 0)      # sian
    WARNA_KOTAK_WAJAH = (0, 255, 0)         # hijau terang
    WARNA_TEKS = (0, 255, 180)              # hijau-sian
    WARNA_GARIS_BINGKAI = (0, 255, 255)     # kuning
    WARNA_PROGRES_BAR = (255, 0, 255)
    TEBAL_LANDMARK = 3
    TEBAL_GARIS = 2
    TEBAL_KOTAK = 2

    # --- Jaring: warna & transparansi ---
    WARNA_JARING = (255, 255, 255)  # putih
    ALPHA_JARING = 175              # 0-255, sedikit transparan
    TEBAL_JARING = 2


# =========================================================================
# PEMERIKSAAN BERKAS — gagal dengan pesan jelas, bukan traceback mentah
# =========================================================================
def periksa_berkas_wajib():
    berkas_wajib = [
        (MODEL_TANGAN, "Model MediaPipe HandLandmarker (deteksi tangan)"),
        (MODEL_WAJAH, "Model MediaPipe FaceDetector (deteksi wajah)"),
        (ASET_TOPENG, "Gambar topeng (PNG dengan alpha channel)"),
        (ASET_LAGU, "Berkas lagu (mp3)"),
    ]
    hilang = [(p, ket) for p, ket in berkas_wajib if not os.path.isfile(p)]
    if hilang:
        print("=" * 70)
        print("BERKAS YANG DIBUTUHKAN TIDAK DITEMUKAN")
        print("=" * 70)
        for path, keterangan in hilang:
            rel = os.path.relpath(path, BASE_DIR)
            print(f"  [X] {rel}")
            print(f"      -> {keterangan}")
        print()
        print("Letakkan berkas yang hilang di lokasi di atas (relatif terhadap")
        print("folder program ini), lalu jalankan ulang. Lihat README.md untuk")
        print("link unduhan model MediaPipe.")
        print("=" * 70)
        sys.exit(1)


# =========================================================================
# INDEKS LANDMARK TANGAN (21 titik, urutan standar MediaPipe Hands)
# =========================================================================
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Pasangan titik untuk menggambar rangka tangan
SAMBUNGAN_TANGAN = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (WRIST, PINKY_MCP),
]

# Enam keypoint BlazeFace (FaceDetector Tasks API):
# 0=mata kanan, 1=mata kiri, 2=ujung hidung, 3=pusat mulut, 4=telinga kanan, 5=telinga kiri
KP_MATA_KANAN, KP_MATA_KIRI = 0, 1


# =========================================================================
# UTILITAS GEOMETRI DASAR
# =========================================================================
def ke_piksel(landmark, lebar, tinggi):
    """Ubah landmark ternormalisasi (0..1) jadi koordinat piksel (float)."""
    return (landmark.x * lebar, landmark.y * tinggi)


def semua_ke_piksel(daftar_landmark, lebar, tinggi):
    return [ke_piksel(lm, lebar, tinggi) for lm in daftar_landmark]


def jarak(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def titik_tengah(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def kurangi(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])


def besar_vektor(v):
    return math.hypot(v[0], v[1])


def normalisasi(v):
    m = besar_vektor(v)
    if m < 1e-6:
        return (0.0, 0.0)
    return (v[0] / m, v[1] / m)


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def sudut_antar_vektor_derajat(v1, v2):
    """Sudut antara dua vektor, dalam derajat, tanpa tanda (0..180)."""
    m1, m2 = besar_vektor(v1), besar_vektor(v2)
    if m1 < 1e-6 or m2 < 1e-6:
        return 0.0
    kos = max(-1.0, min(1.0, dot(v1, v2) / (m1 * m2)))
    return math.degrees(math.acos(kos))


def clamp(nilai, lo, hi):
    return max(lo, min(hi, nilai))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_titik(p1, p2, t):
    return (lerp(p1[0], p2[0], t), lerp(p1[1], p2[1], t))


def ease_out_cubic(t):
    t = clamp(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t):
    t = clamp(t, 0.0, 1.0)
    if t < 0.5:
        return 4 * t * t * t
    return 1 - ((-2 * t + 2) ** 3) / 2


def bbox_dari_dua_titik(p1, p2):
    """Kotak (x1,y1,x2,y2) dengan p1,p2 sebagai dua sudut diagonal berlawanan."""
    x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
    y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
    return (x1, y1, x2, y2)


def luas_bbox(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def rasio_tumpang_tindih_thd_wajah(bingkai_bbox, wajah_bbox):
    """Luas irisan dibagi luas kotak wajah — mengukur seberapa besar
    bingkai sudah 'melingkupi' wajah."""
    x1 = max(bingkai_bbox[0], wajah_bbox[0])
    y1 = max(bingkai_bbox[1], wajah_bbox[1])
    x2 = min(bingkai_bbox[2], wajah_bbox[2])
    y2 = min(bingkai_bbox[3], wajah_bbox[3])
    irisan = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    luas_wajah = luas_bbox(wajah_bbox)
    if luas_wajah < 1e-6:
        return 0.0
    return irisan / luas_wajah


# =========================================================================
# PENGENALAN GESTURE — satu fungsi per pose, terpisah dari alur utama
# =========================================================================
def _jarak_dari_pergelangan(lm_px, idx):
    return jarak(lm_px[WRIST], lm_px[idx])


def jari_lurus(lm_px, tip, pip, mcp):
    """Jari dianggap terentang kalau jaraknya dari pergelangan tangan
    bertambah secara berurutan dari mcp -> pip -> tip (jari lurus keluar)."""
    d_tip = _jarak_dari_pergelangan(lm_px, tip)
    d_pip = _jarak_dari_pergelangan(lm_px, pip)
    d_mcp = _jarak_dari_pergelangan(lm_px, mcp)
    return d_tip > d_pip * 1.05 and d_pip > d_mcp * 0.95


def jari_tekuk(lm_px, tip, pip, mcp):
    """Jari dianggap menekuk ke telapak kalau ujungnya lebih dekat ke
    pergelangan tangan dibanding pangkalnya sendiri (mcp)."""
    d_tip = _jarak_dari_pergelangan(lm_px, tip)
    d_mcp = _jarak_dari_pergelangan(lm_px, mcp)
    return d_tip < d_mcp * 0.92


def klasifikasi_bentuk_tangan(lm_px):
    """Klasifikasikan SATU tangan berdasar pola jari terentang/menekuk.

    Return ('L', info) untuk pose bingkai (jempol+telunjuk lurus, sisanya
    menekuk), ('WEB', info) untuk pose web-shooter (jempol+telunjuk+
    kelingking lurus, tengah+manis menekuk), atau (None, None) kalau tidak
    cocok keduanya.

    'info' berisi titik acuan & vektor arah yang dipakai fase-fase lain,
    supaya perhitungan jari tidak perlu diulang.
    """
    ibu_jari = jari_lurus(lm_px, THUMB_TIP, THUMB_IP, THUMB_MCP)
    telunjuk = jari_lurus(lm_px, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    tengah_lurus = jari_lurus(lm_px, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    manis_lurus = jari_lurus(lm_px, RING_TIP, RING_PIP, RING_MCP)
    kelingking_lurus = jari_lurus(lm_px, PINKY_TIP, PINKY_PIP, PINKY_MCP)

    tengah_tekuk = jari_tekuk(lm_px, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    manis_tekuk = jari_tekuk(lm_px, RING_TIP, RING_PIP, RING_MCP)
    kelingking_tekuk = jari_tekuk(lm_px, PINKY_TIP, PINKY_PIP, PINKY_MCP)

    # --- Pose bingkai (huruf L): jempol + telunjuk lurus, 3 jari lain menekuk ---
    if ibu_jari and telunjuk and tengah_tekuk and manis_tekuk and kelingking_tekuk:
        titik_sudut = titik_tengah(lm_px[THUMB_CMC], lm_px[INDEX_MCP])
        v_ibu = normalisasi(kurangi(lm_px[THUMB_TIP], titik_sudut))
        v_telunjuk = normalisasi(kurangi(lm_px[INDEX_TIP], titik_sudut))
        sudut_derajat = sudut_antar_vektor_derajat(v_ibu, v_telunjuk)
        arah_buka = normalisasi((v_ibu[0] + v_telunjuk[0], v_ibu[1] + v_telunjuk[1]))
        return "L", {
            "titik_sudut": titik_sudut,
            "arah_buka": arah_buka,
            "sudut_derajat": sudut_derajat,
        }

    # --- Pose web-shooter: jempol + telunjuk + kelingking lurus, tengah+manis menekuk ---
    if ibu_jari and telunjuk and kelingking_lurus and tengah_tekuk and manis_tekuk:
        telapak = titik_tengah(lm_px[WRIST], lm_px[MIDDLE_MCP])
        return "WEB", {"telapak": telapak}

    return None, None


def evaluasi_pasangan_bingkai(info_a, info_b, toleransi_sudut):
    """Evaluasi apakah dua hasil klasifikasi 'L' membentuk pose bingkai foto
    yang valid: sudut jempol-telunjuk masing-masing dalam toleransi, dan
    arah bukaan kedua tangan berlawanan (diagonal)."""
    for info in (info_a, info_b):
        if not (toleransi_sudut[0] <= info["sudut_derajat"] <= toleransi_sudut[1]):
            return None
    kesamaan_arah = dot(info_a["arah_buka"], info_b["arah_buka"])
    if kesamaan_arah > K.AMBANG_ARAH_BERLAWANAN:
        return None
    return info_a["titik_sudut"], info_b["titik_sudut"]


# =========================================================================
# HELPER: hasil deteksi dibungkus jadi struktur sederhana
# =========================================================================
@dataclass
class HasilTangan:
    lm_px: list           # 21 titik (x,y) piksel
    bentuk: Optional[str]  # 'L' | 'WEB' | None
    info: Optional[dict]


@dataclass
class HasilWajah:
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2 piksel
    mata_kanan: Tuple[float, float]
    mata_kiri: Tuple[float, float]
    sudut_kemiringan: float  # derajat


# =========================================================================
# PEMBUNGKUS MEDIAPIPE TASKS API
# =========================================================================
class PelacakWajah:
    def __init__(self, path_model):
        opsi = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path_model),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3,
        )
        self._detektor = vision.FaceDetector.create_from_options(opsi)

    def deteksi(self, mp_image, timestamp_ms, lebar, tinggi):
        """mp_image: mp.Image (SRGB) yang sudah dibangun sekali per frame
        oleh pemanggil (lihat bangun_mp_image) — dipakai bersama dengan
        PelacakTangan supaya konversi BGR->RGB tidak dilakukan dua kali."""
        hasil = self._detektor.detect_for_video(mp_image, timestamp_ms)
        if not hasil.detections:
            return None
        # Ambil deteksi dengan kotak terbesar (wajah paling dominan di frame)
        deteksi = max(
            hasil.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
        )
        bb = deteksi.bounding_box
        bbox = (bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height)
        kp = deteksi.keypoints
        if kp is None or len(kp) < 2:
            return None
        w, h = lebar, tinggi
        titik_a = (kp[KP_MATA_KANAN].x * w, kp[KP_MATA_KANAN].y * h)
        titik_b = (kp[KP_MATA_KIRI].x * w, kp[KP_MATA_KIRI].y * h)
        # PENTING: jangan percaya label 'kanan'/'kiri' dari model untuk arah
        # sudut — label itu anatomis (mata kanan wajah biasanya tampil di
        # sisi KIRI layar), dan frame kita juga sudah di-flip cermin.
        # Supaya sudut 0 derajat selalu berarti "kepala tegak" apa pun
        # urutan labelnya, urutkan dua titik mata berdasar posisi x di layar:
        # dari yang paling kiri ke yang paling kanan di layar.
        if titik_a[0] <= titik_b[0]:
            mata_kiri_layar, mata_kanan_layar = titik_a, titik_b
        else:
            mata_kiri_layar, mata_kanan_layar = titik_b, titik_a
        dx = mata_kanan_layar[0] - mata_kiri_layar[0]
        dy = mata_kanan_layar[1] - mata_kiri_layar[1]
        sudut = math.degrees(math.atan2(dy, dx))
        return HasilWajah(
            bbox=bbox, mata_kanan=mata_kanan_layar, mata_kiri=mata_kiri_layar,
            sudut_kemiringan=sudut,
        )

    def tutup(self):
        self._detektor.close()


class PelacakTangan:
    def __init__(self, path_model):
        opsi = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detektor = vision.HandLandmarker.create_from_options(opsi)

    def deteksi(self, mp_image, timestamp_ms, lebar, tinggi):
        """mp_image: mp.Image (SRGB) yang sama dengan yang dipakai
        PelacakWajah pada frame ini (lihat bangun_mp_image)."""
        hasil = self._detektor.detect_for_video(mp_image, timestamp_ms)
        daftar = []
        if not hasil.hand_landmarks:
            return daftar
        w, h = lebar, tinggi
        for lm_list in hasil.hand_landmarks:
            lm_px = semua_ke_piksel(lm_list, w, h)
            bentuk, info = klasifikasi_bentuk_tangan(lm_px)
            daftar.append(HasilTangan(lm_px=lm_px, bentuk=bentuk, info=info))
        return daftar

    def tutup(self):
        self._detektor.close()


def bangun_mp_image(frame_bgr):
    """Konversi BGR->RGB dan bungkus jadi mp.Image SEKALI per frame, supaya
    PelacakWajah & PelacakTangan bisa berbagi hasil yang sama tanpa
    masing-masing melakukan cv2.cvtColor sendiri-sendiri (perf)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class Timestamp:
    """Penghasil timestamp milidetik yang selalu naik (syarat wajib Tasks API)."""

    def __init__(self):
        self._mulai = time.perf_counter()
        self._terakhir = -1

    def sekarang_ms(self):
        ms = int((time.perf_counter() - self._mulai) * 1000)
        if ms <= self._terakhir:
            ms = self._terakhir + 1
        self._terakhir = ms
        return ms


# =========================================================================
# PELACAK GESTURE BINGKAI (Fase 2 -> Fase 3): MENUNGGU -> MENGUNCI -> MEMEKAR
# =========================================================================
class StatusBingkai(Enum):
    MENUNGGU = auto()
    MENGUNCI = auto()
    MEMEKAR = auto()


class PelacakBingkai:
    def __init__(self):
        self.reset()

    def reset(self):
        self.status = StatusBingkai.MENUNGGU
        self._mulai_kunci = 0.0
        self._diagonal_awal = None
        self._puncak_rasio = 1.0
        self._riwayat_diagonal = deque(maxlen=6)
        self._gagal_beruntun = 0
        self.sudut_a = None
        self.sudut_b = None
        self.progres_kunci = 0.0  # 0..1, untuk bar diagnostik

    def update(self, tangan_l, wajah, now):
        """tangan_l: list HasilTangan yang berbentuk 'L' pada frame ini.
        Return event: None | ('TERKUNCI', None) | ('BATAL', None)
                     | ('TRIGGER', (sudut_a, sudut_b, rasio))
        """
        toleransi = (
            K.TOLERANSI_SUDUT_L_KETAT
            if self.status != StatusBingkai.MEMEKAR
            else K.TOLERANSI_SUDUT_L_LONGGAR
        )
        pasangan = None
        if len(tangan_l) == 2:
            pasangan = evaluasi_pasangan_bingkai(tangan_l[0].info, tangan_l[1].info, toleransi)

        if self.status == StatusBingkai.MENUNGGU:
            if pasangan:
                self.status = StatusBingkai.MENGUNCI
                self._mulai_kunci = now
                self.progres_kunci = 0.0
            return None

        if self.status == StatusBingkai.MENGUNCI:
            if not pasangan:
                self.reset()
                return None
            self.progres_kunci = clamp((now - self._mulai_kunci) / K.DURASI_KUNCI_BINGKAI, 0.0, 1.0)
            if now - self._mulai_kunci >= K.DURASI_KUNCI_BINGKAI:
                self.status = StatusBingkai.MEMEKAR
                self.sudut_a, self.sudut_b = pasangan
                self._diagonal_awal = jarak(self.sudut_a, self.sudut_b)
                self._puncak_rasio = 1.0
                self._riwayat_diagonal.clear()
                return ("TERKUNCI", None)
            return None

        # StatusBingkai.MEMEKAR
        if pasangan:
            self._gagal_beruntun = 0
            self.sudut_a, self.sudut_b = pasangan
            diagonal = jarak(self.sudut_a, self.sudut_b)
            self._riwayat_diagonal.append(diagonal)
            rasio = diagonal / self._diagonal_awal if self._diagonal_awal else 1.0
            self._puncak_rasio = max(self._puncak_rasio, rasio)

            overlap = 0.0
            if wajah is not None:
                bbox_bingkai = bbox_dari_dua_titik(self.sudut_a, self.sudut_b)
                overlap = rasio_tumpang_tindih_thd_wajah(bbox_bingkai, wajah.bbox)

            if rasio >= K.RASIO_PEMEKARAN_BINGKAI and (
                wajah is None or overlap >= K.AMBANG_TUMPANG_TINDIH_WAJAH
            ):
                data = (self.sudut_a, self.sudut_b, rasio)
                self.reset()
                return ("TRIGGER", data)

            if len(self._riwayat_diagonal) >= 4:
                menurun = all(
                    self._riwayat_diagonal[i] >= self._riwayat_diagonal[i + 1] - 1e-6
                    for i in range(len(self._riwayat_diagonal) - 1)
                )
                if menurun and rasio < self._puncak_rasio * 0.85:
                    self.reset()
                    return ("BATAL", None)
            return None

        # Pasangan tidak valid frame ini: salah satu/kedua tangan mungkin hilang
        self._gagal_beruntun += 1
        if len(tangan_l) < 2 and self._diagonal_awal:
            # --- CADANGAN: pemekaran sudah jelas berlangsung, tangan hilang di tepi layar ---
            if self._puncak_rasio >= K.RASIO_PEMEKARAN_BINGKAI * K.FAKTOR_CADANGAN_PEMEKARAN:
                overlap = 0.0
                if wajah is not None and self.sudut_a is not None:
                    bbox_bingkai = bbox_dari_dua_titik(self.sudut_a, self.sudut_b)
                    overlap = rasio_tumpang_tindih_thd_wajah(bbox_bingkai, wajah.bbox)
                cukup_overlap = (
                    wajah is None or overlap >= K.AMBANG_TUMPANG_TINDIH_WAJAH * K.FAKTOR_CADANGAN_PEMEKARAN
                )
                if cukup_overlap:
                    data = (self.sudut_a, self.sudut_b, self._puncak_rasio)
                    self.reset()
                    return ("TRIGGER", data)

        if self._gagal_beruntun > K.TOLERANSI_FRAME_GAGAL:
            self.reset()
            return ("BATAL", None)
        return None


# =========================================================================
# PELACAK WEB SHOOTER (per-tangan): NETRAL -> TERKUNCI -> (menembak) -> NETRAL
# =========================================================================
class StatusTangan(Enum):
    NETRAL = auto()
    TERKUNCI = auto()
    TUNGGU_LEPAS = auto()  # sudah menembak, menunggu pose lepas / tangan hilang / cooldown


class SlotTanganWeb:
    def __init__(self, nama):
        self.nama = nama  # label diagnostik saja ("A"/"B"), bukan kiri/kanan sungguhan
        self.status = StatusTangan.NETRAL
        self._mulai_kunci = 0.0
        self.siap_tembak = False
        self.riwayat_posisi = deque(maxlen=K.RIWAYAT_POSISI_TELAPAK)
        self.posisi_terakhir = None
        self.waktu_terlihat_terakhir = -999.0
        self.cooldown_sampai = 0.0
        self.progres_kunci = 0.0

    def _catat_posisi(self, posisi, now):
        self.riwayat_posisi.append((now, posisi))
        self.posisi_terakhir = posisi
        self.waktu_terlihat_terakhir = now

    def _kecepatan_dan_arah(self):
        if len(self.riwayat_posisi) < 2:
            return 0.0, (0.0, 0.0)
        t0, p0 = self.riwayat_posisi[0]
        t1, p1 = self.riwayat_posisi[-1]
        dt = max(1e-3, t1 - t0)
        vektor = kurangi(p1, p0)
        kecepatan = besar_vektor(vektor) / dt
        return kecepatan, normalisasi(vektor)

    def _reset_ke_netral(self, now, dengan_cooldown=True):
        self.status = StatusTangan.NETRAL
        self.siap_tembak = False
        self.progres_kunci = 0.0
        if dengan_cooldown:
            self.cooldown_sampai = now + K.COOLDOWN_TEMBAK_PER_TANGAN

    def update(self, hasil_tangan_web, now):
        """hasil_tangan_web: HasilTangan berbentuk 'WEB' yang sudah ditandai
        milik slot ini pada frame ini, atau None kalau slot ini tidak
        mendapat tangan pada frame ini.

        Return: None atau tuple (asal_tembakan, arah_tembakan) kalau
        tembakan baru saja terpicu frame ini.
        """
        if hasil_tangan_web is not None:
            posisi = hasil_tangan_web.info["telapak"]
            self._catat_posisi(posisi, now)

            if self.status == StatusTangan.NETRAL:
                if now >= self.cooldown_sampai:
                    self.status = StatusTangan.TERKUNCI
                    self._mulai_kunci = now
                    self.riwayat_posisi.clear()
                    self.riwayat_posisi.append((now, posisi))
                return None

            if self.status == StatusTangan.TERKUNCI:
                self.progres_kunci = clamp(
                    (now - self._mulai_kunci) / K.DURASI_KUNCI_WEB_SHOOTER, 0.0, 1.0
                )
                if now - self._mulai_kunci >= K.DURASI_KUNCI_WEB_SHOOTER:
                    self.siap_tembak = True
                    kecepatan, arah = self._kecepatan_dan_arah()
                    if kecepatan >= K.AMBANG_KECEPATAN_HENTAKAN:
                        asal = self.posisi_terakhir
                        self._reset_ke_netral(now)
                        self.status = StatusTangan.TUNGGU_LEPAS
                        self.cooldown_sampai = now + K.COOLDOWN_TEMBAK_PER_TANGAN
                        return (asal, arah)
                return None

            if self.status == StatusTangan.TUNGGU_LEPAS:
                # Masih memegang pose WEB terus -> belum boleh menembak lagi,
                # tunggu sampai pose benar-benar lepas.
                return None

        else:
            # Tidak ada tangan berbentuk WEB untuk slot ini frame ini.
            if self.status == StatusTangan.TERKUNCI:
                # --- CADANGAN: tangan hilang mendadak saat terkunci = hentakan ---
                kecepatan, arah = self._kecepatan_dan_arah()
                sudah_siap = (now - self._mulai_kunci) >= K.DURASI_KUNCI_WEB_SHOOTER
                if sudah_siap and self.posisi_terakhir is not None:
                    asal = self.posisi_terakhir
                    if besar_vektor(arah) < 1e-6:
                        arah = (0.0, -1.0)  # fallback: tembak ke atas kalau arah tak jelas
                    self._reset_ke_netral(now)
                    return (asal, arah)
                self._reset_ke_netral(now)
                return None

            if self.status == StatusTangan.TUNGGU_LEPAS:
                # Tangan keluar layar juga dihitung sebagai "pose lepas"
                self._reset_ke_netral(now, dengan_cooldown=False)
                self.cooldown_sampai = max(self.cooldown_sampai, now)
                return None

            # NETRAL & tidak terlihat: tidak ada apa-apa
            return None


class PelacakWebShooter:
    """Mengelola dua slot tangan independen dan menugaskan tangan yang
    terdeteksi 'WEB' pada tiap frame ke slot terdekat (bukan berdasar
    label kiri/kanan MediaPipe yang tidak bisa diandalkan setelah flip)."""

    def __init__(self):
        self.slot = [SlotTanganWeb("A"), SlotTanganWeb("B")]

    def _tugaskan(self, tangan_web):
        tugas = [None, None]
        sisa = list(tangan_web)

        # Tahap 1: cocokkan ke slot yang punya posisi terakhir (nearest neighbour)
        for i, slot in enumerate(self.slot):
            if slot.posisi_terakhir is None or not sisa:
                continue
            terdekat = min(sisa, key=lambda t: jarak(t.info["telapak"], slot.posisi_terakhir))
            if jarak(terdekat.info["telapak"], slot.posisi_terakhir) < 400:  # px, batas wajar
                tugas[i] = terdekat
                sisa.remove(terdekat)

        # Tahap 2: sisa tangan (atau slot baru) -> urutkan dari kiri ke kanan
        slot_kosong = [i for i in range(2) if tugas[i] is None]
        sisa.sort(key=lambda t: t.info["telapak"][0])
        for i, tangan in zip(slot_kosong, sisa):
            tugas[i] = tangan

        return tugas

    def update(self, daftar_tangan, now):
        """daftar_tangan: semua HasilTangan pada frame ini (semua bentuk).
        Return list of (asal, arah) untuk tiap tembakan baru frame ini."""
        tangan_web = [t for t in daftar_tangan if t.bentuk == "WEB"]
        tugas = self._tugaskan(tangan_web)

        # Bersihkan slot yang sudah lama sekali tidak terlihat supaya tidak
        # "mewarisi" posisi basi saat tangan baru muncul.
        for slot in self.slot:
            if now - slot.waktu_terlihat_terakhir > K.TOLERANSI_TANGAN_HILANG_SLOT:
                slot.posisi_terakhir = None

        tembakan = []
        for slot, tangan in zip(self.slot, tugas):
            hasil = slot.update(tangan, now)
            if hasil is not None:
                tembakan.append(hasil)
        return tembakan


# =========================================================================
# ANIMASI JARING LABA-LABA
# =========================================================================
class AnimasiJaring:
    """Satu pola jaring dari satu ayunan/tembakan. Memancar dari titik asal
    dengan garis radial + busur melengkung, menyebar dengan easing, dan
    dibiaskan ke arah hentakan tangan."""

    def __init__(self, asal, arah_bias, mulai_waktu):
        self.asal = asal
        self.arah_bias = arah_bias if besar_vektor(arah_bias) > 1e-6 else (0.0, 0.0)
        self.mulai = mulai_waktu
        self.selesai = False

    def _progres(self, now):
        t = (now - self.mulai) / K.DURASI_ANIMASI_JARING
        return clamp(t, 0.0, 1.0)

    def gambar(self, kanvas, now):
        """Gambar pola ini ke kanvas BGRA. Kembalikan True kalau animasi
        sudah selesai menyebar penuh (boleh 'dipanggang' ke kanvas statis)."""
        t = self._progres(now)
        te = ease_out_cubic(t)
        radius_env = K.RADIUS_MAKS_JARING * te

        titik_spoke = []
        n = K.JUMLAH_GARIS_RADIAL
        for i in range(n):
            sudut = 2 * math.pi * i / n
            arah_garis = (math.cos(sudut), math.sin(sudut))
            bias = 1.0
            if besar_vektor(self.arah_bias) > 1e-6:
                kesamaan = max(0.0, dot(arah_garis, self.arah_bias))
                bias = 1.0 + K.KEKUATAN_BIAS_ARAH * kesamaan
            r = radius_env * bias
            ujung = (self.asal[0] + arah_garis[0] * r, self.asal[1] + arah_garis[1] * r)
            titik_spoke.append((sudut, ujung, r))
            cv2.line(
                kanvas,
                (int(self.asal[0]), int(self.asal[1])),
                (int(ujung[0]), int(ujung[1])),
                (*K.WARNA_JARING, K.ALPHA_JARING),
                K.TEBAL_JARING,
                cv2.LINE_AA,
            )

        fraksi_list = np.linspace(0.35, 1.0, K.JUMLAH_BUSUR)
        for frak in fraksi_list:
            for i in range(n):
                sudut1, _, r1 = titik_spoke[i]
                sudut2, _, r2 = titik_spoke[(i + 1) % n]
                r_busur = min(r1, r2) * frak
                if r_busur < 3:
                    continue
                a1 = math.degrees(sudut1)
                a2 = math.degrees(sudut2)
                if a2 < a1:
                    a2 += 360
                cv2.ellipse(
                    kanvas,
                    (int(self.asal[0]), int(self.asal[1])),
                    (int(r_busur), int(r_busur)),
                    0, a1, a2,
                    (*K.WARNA_JARING, K.ALPHA_JARING),
                    K.TEBAL_JARING,
                    cv2.LINE_AA,
                )

        if t >= 1.0:
            self.selesai = True
        return self.selesai


class LapisanJaring:
    """Kanvas persisten untuk seluruh pola jaring. Pola yang sudah selesai
    menyebar 'dipanggang' ke kanvas statis (tidak digambar ulang tiap
    frame) supaya performa tetap terjaga walau jaring makin padat."""

    def __init__(self, lebar, tinggi):
        self.lebar = lebar
        self.tinggi = tinggi
        self.kanvas_statis = np.zeros((tinggi, lebar, 4), dtype=np.uint8)
        self.aktif: List[AnimasiJaring] = []
        self._hitung_cakupan_cache = 0.0
        self._frame_sejak_hitung = 999
        # Kotak pembatas (union) area yang pernah disentuh jaring, dipakai
        # supaya blending frame tidak perlu memproses seluruh layar tiap
        # frame kalau jaring baru menutupi sebagian kecil (performa).
        self._bbox = None  # (x1, y1, x2, y2)

    def tambah(self, asal, arah, now):
        anim = AnimasiJaring(asal, arah, now)
        self.aktif.append(anim)
        # Jangkauan terjauh satu pola (dengan bias arah) diperkirakan
        # secara konservatif supaya kotak pembatas mencakup seluruh pola.
        jangkauan = K.RADIUS_MAKS_JARING * (1.0 + K.KEKUATAN_BIAS_ARAH)
        x1 = int(clamp(asal[0] - jangkauan, 0, self.lebar))
        y1 = int(clamp(asal[1] - jangkauan, 0, self.tinggi))
        x2 = int(clamp(asal[0] + jangkauan, 0, self.lebar))
        y2 = int(clamp(asal[1] + jangkauan, 0, self.tinggi))
        if self._bbox is None:
            self._bbox = (x1, y1, x2, y2)
        else:
            bx1, by1, bx2, by2 = self._bbox
            self._bbox = (min(bx1, x1), min(by1, y1), max(bx2, x2), max(by2, y2))

    def reset(self):
        self.kanvas_statis[:] = 0
        self.aktif.clear()
        self._hitung_cakupan_cache = 0.0
        self._bbox = None

    def render(self, now):
        """Kembalikan kanvas BGRA gabungan (statis + animasi yang masih
        berjalan) untuk ditumpuk ke frame kamera frame ini."""
        if not self.aktif:
            return self.kanvas_statis
        tampil = self.kanvas_statis.copy()
        masih_aktif = []
        for anim in self.aktif:
            selesai = anim.gambar(tampil, now)
            if selesai:
                # Panggang permanen ke kanvas statis, tidak perlu dihitung lagi.
                anim.gambar(self.kanvas_statis, now)
            else:
                masih_aktif.append(anim)
        self.aktif = masih_aktif
        return tampil

    def cakupan_layar(self):
        """Perkiraan persentase piksel layar yang sudah tertutup jaring.
        Dihitung dari kanvas statis + downsampling supaya murah."""
        alpha = self.kanvas_statis[::4, ::4, 3]
        if alpha.size == 0:
            return 0.0
        return float(np.count_nonzero(alpha)) / float(alpha.size)

    def tempel_ke_frame(self, frame_bgr, kanvas_bgra):
        """Tumpuk kanvas jaring ke frame kamera. Hanya memproses kotak
        pembatas area yang pernah disentuh jaring (bukan seluruh layar),
        dan memakai operasi cv2 ber-tipe uint8 (SIMD, jauh lebih cepat
        dari numpy float32) supaya frame rate tetap terjaga walau jaring
        sudah padat."""
        if self._bbox is None:
            return  # belum ada jaring sama sekali
        x1, y1, x2, y2 = self._bbox
        if x2 <= x1 or y2 <= y1:
            return
        sub_frame = frame_bgr[y1:y2, x1:x2]
        sub_kanvas = kanvas_bgra[y1:y2, x1:x2]
        alpha = sub_kanvas[:, :, 3]
        alpha_3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        inv_alpha_3 = cv2.bitwise_not(alpha_3)
        warna = sub_kanvas[:, :, :3]
        term1 = cv2.multiply(sub_frame, inv_alpha_3, scale=1.0 / 255.0)
        term2 = cv2.multiply(warna, alpha_3, scale=1.0 / 255.0)
        frame_bgr[y1:y2, x1:x2] = cv2.add(term1, term2)


# =========================================================================
# RENDER TOPENG
# =========================================================================
class Topeng:
    def __init__(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Gagal membaca gambar topeng: {path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            img = np.dstack([img, alpha])
        self.img = img
        self.h0, self.w0 = img.shape[:2]

    def _lapisan(self, lebar_target, pusat_xy, sudut_deg, ukuran_frame):
        lebar_target = max(1.0, lebar_target)
        skala = lebar_target / self.w0
        M = cv2.getRotationMatrix2D(
            (self.w0 / 2.0, self.h0 / 2.0),
            K.TOPENG_ROTASI_FAKTOR * sudut_deg,
            skala,
        )
        M[0, 2] += pusat_xy[0] - self.w0 / 2.0
        M[1, 2] += pusat_xy[1] - self.h0 / 2.0
        out = cv2.warpAffine(
            self.img, M, ukuran_frame,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return out

    def tempel(self, frame, lebar_target, pusat_xy, sudut_deg, alpha_kali=1.0):
        if alpha_kali <= 0.001:
            return
        h, w = frame.shape[:2]
        lapisan = self._lapisan(lebar_target, pusat_xy, sudut_deg, (w, h))
        # Blending pakai operasi cv2 uint8 (SIMD) — jauh lebih cepat dari
        # numpy float32 penuh-frame, penting karena ini dipanggil tiap
        # frame selama topeng terpasang.
        alpha = lapisan[:, :, 3]
        if alpha_kali < 1.0:
            alpha = cv2.convertScaleAbs(alpha, alpha=clamp(alpha_kali, 0.0, 1.0))
        alpha_3 = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        inv_alpha_3 = cv2.bitwise_not(alpha_3)
        warna = lapisan[:, :, :3]
        term1 = cv2.multiply(frame, inv_alpha_3, scale=1.0 / 255.0)
        term2 = cv2.multiply(warna, alpha_3, scale=1.0 / 255.0)
        frame[:] = cv2.add(term1, term2)


def parameter_topeng_dari_wajah(wajah: HasilWajah):
    """Hitung lebar target, pusat, dan sudut rotasi topeng dari hasil
    deteksi wajah (posisi kedua mata)."""
    jarak_mata = jarak(wajah.mata_kanan, wajah.mata_kiri)
    lebar_target = jarak_mata * K.TOPENG_SKALA
    tengah_mata = titik_tengah(wajah.mata_kanan, wajah.mata_kiri)
    garis_mata = kurangi(wajah.mata_kanan, wajah.mata_kiri)
    tegak_lurus = normalisasi((-garis_mata[1], garis_mata[0]))
    # Pastikan arah tegak lurus mengarah ke bawah (menuju dagu), bukan ke atas.
    if tegak_lurus[1] < 0:
        tegak_lurus = (-tegak_lurus[0], -tegak_lurus[1])
    pusat = (
        tengah_mata[0] + tegak_lurus[0] * K.TOPENG_OFFSET_Y * jarak_mata,
        tengah_mata[1] + tegak_lurus[1] * K.TOPENG_OFFSET_Y * jarak_mata,
    )
    return lebar_target, pusat, wajah.sudut_kemiringan


# =========================================================================
# FASE PROGRAM
# =========================================================================
class Fase(Enum):
    SEARCHING = auto()
    GESTURE_TANGAN = auto()   # topeng lepas, menunggu / melacak gesture bingkai
    TRANSISI_PASANG = auto()  # animasi topeng masuk
    TOPENG_AKTIF = auto()     # topeng terpasang, web-shooter aktif
    TRANSISI_LEPAS = auto()   # animasi topeng keluar
    ENDING = auto()


# =========================================================================
# AUDIO
# =========================================================================
class ManajerAudio:
    def __init__(self, path_lagu):
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(path_lagu)
        except Exception as e:
            print(f"[ERROR] Gagal memuat lagu '{path_lagu}': {e}")
            sys.exit(1)
        self._volume_tersimpan = K.VOLUME_LAGU
        self._mute = False
        pygame.mixer.music.set_volume(K.VOLUME_LAGU)
        self._sedang_main = False

    def mulai(self):
        pygame.mixer.music.play(loops=0, start=0.0)
        self._sedang_main = True

    def fadeout(self, durasi_detik):
        pygame.mixer.music.fadeout(int(durasi_detik * 1000))
        self._sedang_main = False

    def stop(self):
        pygame.mixer.music.stop()
        self._sedang_main = False

    def toggle_mute(self):
        self._mute = not self._mute
        pygame.mixer.music.set_volume(0.0 if self._mute else self._volume_tersimpan)
        return self._mute

    def reset(self):
        self.stop()
        pygame.mixer.music.set_volume(0.0 if self._mute else self._volume_tersimpan)


# =========================================================================
# DIAGNOSTIK (lapisan visual neon)
# =========================================================================
FONT = cv2.FONT_HERSHEY_DUPLEX


def gambar_rangka_tangan(frame, lm_px):
    for a, b in SAMBUNGAN_TANGAN:
        p1 = (int(lm_px[a][0]), int(lm_px[a][1]))
        p2 = (int(lm_px[b][0]), int(lm_px[b][1]))
        cv2.line(frame, p1, p2, K.WARNA_GARIS_TANGAN, K.TEBAL_GARIS, cv2.LINE_AA)
    for p in lm_px:
        cv2.circle(frame, (int(p[0]), int(p[1])), K.TEBAL_LANDMARK, K.WARNA_LANDMARK_TANGAN, -1, cv2.LINE_AA)


def gambar_kotak_wajah(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), K.WARNA_KOTAK_WAJAH, K.TEBAL_KOTAK, cv2.LINE_AA)


def gambar_teks(frame, teks, pos, skala=0.8, warna=None, tebal=2):
    warna = warna or K.WARNA_TEKS
    cv2.putText(frame, teks, pos, FONT, skala, (0, 0, 0), tebal + 3, cv2.LINE_AA)
    cv2.putText(frame, teks, pos, FONT, skala, warna, tebal, cv2.LINE_AA)


def gambar_bar_progres(frame, pos, lebar, tinggi, progres, warna=None):
    warna = warna or K.WARNA_PROGRES_BAR
    x, y = pos
    cv2.rectangle(frame, (x, y), (x + lebar, y + tinggi), (60, 60, 60), 1, cv2.LINE_AA)
    isi = int(lebar * clamp(progres, 0.0, 1.0))
    if isi > 0:
        cv2.rectangle(frame, (x, y), (x + isi, y + tinggi), warna, -1, cv2.LINE_AA)


def gambar_garis_bingkai(frame, sudut_a, sudut_b):
    bbox = bbox_dari_dua_titik(sudut_a, sudut_b)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), K.WARNA_GARIS_BINGKAI, 2, cv2.LINE_AA)
    cv2.line(frame, (int(sudut_a[0]), int(sudut_a[1])), (int(sudut_b[0]), int(sudut_b[1])),
              K.WARNA_GARIS_BINGKAI, 1, cv2.LINE_AA)


def gambar_glow_telapak(frame, posisi, kekuatan):
    pusat = (int(posisi[0]), int(posisi[1]))
    radius = int(18 + 14 * kekuatan)
    overlay = frame.copy()
    cv2.circle(overlay, pusat, radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25 * kekuatan, frame, 1 - 0.25 * kekuatan, 0, frame)
    cv2.circle(frame, pusat, radius, K.WARNA_GARIS_BINGKAI, 2, cv2.LINE_AA)


# =========================================================================
# APLIKASI UTAMA
# =========================================================================
class SpiderFilterApp:
    def __init__(self):
        periksa_berkas_wajib()

        print("[INIT] Memuat model MediaPipe...")
        self.pelacak_wajah = PelacakWajah(MODEL_WAJAH)
        self.pelacak_tangan = PelacakTangan(MODEL_TANGAN)
        self.ts = Timestamp()

        print("[INIT] Memuat aset...")
        self.topeng = Topeng(ASET_TOPENG)
        self.audio = ManajerAudio(ASET_LAGU)

        print("[INIT] Membuka kamera...")
        self.cap = cv2.VideoCapture(K.KAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, K.KAMERA_LEBAR)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, K.KAMERA_TINGGI)
        if not self.cap.isOpened():
            print(f"[ERROR] Tidak bisa membuka kamera index {K.KAMERA_INDEX}.")
            sys.exit(1)

        lebar = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or K.KAMERA_LEBAR
        tinggi = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or K.KAMERA_TINGGI
        self.lebar, self.tinggi = lebar, tinggi

        self.nama_jendela = "Spider Filter"
        cv2.namedWindow(self.nama_jendela, cv2.WINDOW_NORMAL)
        self._fullscreen = True
        cv2.setWindowProperty(self.nama_jendela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.jaring = LapisanJaring(lebar, tinggi)
        self.pelacak_bingkai = PelacakBingkai()
        self.pelacak_web = PelacakWebShooter()

        self._reset_status_sesi()

        self._riwayat_fps = deque(maxlen=30)
        self._tampilkan_diagnostik = True

        print("[SIAP] Program dimulai. Cari wajah di depan kamera...")

    # -- reset state sesi (dipakai saat start & saat tombol 'r') --------
    def _reset_status_sesi(self):
        self.fase = Fase.SEARCHING
        self._mulai_kunci_wajah = None
        self.wajah_terakhir = None
        self._frame_wajah_hilang = 0
        self._boleh_toggle_setelah = 0.0

        self._data_transisi = None  # dict berisi titik awal/akhir animasi topeng
        self._mulai_transisi = 0.0

        self.jaring.reset()
        self.pelacak_bingkai.reset()
        self.pelacak_web = PelacakWebShooter()

        self._mulai_ending = None
        self._audio_di_fadeout = False
        self._pemadat_ditambahkan = False
        self._siap_berhenti = False

    def reset_ke_fase_1(self):
        print("[RESET] Kembali ke Fase 1 — siap retake.")
        self.audio.reset()
        self._reset_status_sesi()

    # -- loop utama -------------------------------------------------------
    def _gambar_fps(self, frame, now):
        self._riwayat_fps.append(now)
        if len(self._riwayat_fps) >= 2 and self._tampilkan_diagnostik and self.fase != Fase.ENDING:
            dt = self._riwayat_fps[-1] - self._riwayat_fps[0]
            fps = (len(self._riwayat_fps) - 1) / dt if dt > 0 else 0.0
            gambar_teks(frame, f"FPS: {fps:0.1f}", (self.lebar - 200, 40), 0.8, (0, 255, 0))

    def _tangani_keyboard(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[KELUAR] Dihentikan oleh pengguna.")
            return False
        elif key == ord("r"):
            self.reset_ke_fase_1()
        elif key == ord("m"):
            mute = self.audio.toggle_mute()
            print(f"[AUDIO] {'Mute' if mute else 'Unmute'}")
        elif key == ord("h"):
            self._tampilkan_diagnostik = not self._tampilkan_diagnostik
            print(f"[TAMPILAN] Lapisan diagnostik {'disembunyikan' if not self._tampilkan_diagnostik else 'ditampilkan'}")
        elif key == ord("f"):
            self._fullscreen = not self._fullscreen
            mode = cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.nama_jendela, cv2.WND_PROP_FULLSCREEN, mode)
        return True

    # -- FASE 1: SEARCHING --------------------------------------------------
    def _proses_searching(self, frame, now, ts_ms):
        mp_image = bangun_mp_image(frame)
        wajah = self.pelacak_wajah.deteksi(mp_image, ts_ms, self.lebar, self.tinggi)

        if wajah is not None:
            if self._mulai_kunci_wajah is None:
                self._mulai_kunci_wajah = now
            progres = clamp((now - self._mulai_kunci_wajah) / K.DURASI_KUNCI_WAJAH, 0.0, 1.0)
        else:
            self._mulai_kunci_wajah = None
            progres = 0.0

        if self._tampilkan_diagnostik:
            if wajah is not None:
                gambar_kotak_wajah(frame, wajah.bbox)
                gambar_bar_progres(frame, (40, 90), 260, 18, progres)
            gambar_teks(frame, "MENCARI WAJAH...", (40, 60), 1.0)
            gambar_teks(frame, "Posisikan wajah di depan kamera", (40, self.tinggi - 40), 0.7)

        if wajah is not None and progres >= 1.0:
            print("[FASE 1] Wajah terkunci. Memulai lagu & Fase 2.")
            self.audio.mulai()
            self.fase = Fase.GESTURE_TANGAN
            self.wajah_terakhir = wajah
            self._frame_wajah_hilang = 0

    # -- Wajah dengan anti-kedip (dipakai fase 2 dst) ------------------------
    def _perbarui_wajah(self, wajah):
        if wajah is not None:
            self.wajah_terakhir = wajah
            self._frame_wajah_hilang = 0
            return wajah
        self._frame_wajah_hilang += 1
        if self._frame_wajah_hilang <= K.TOLERANSI_WAJAH_HILANG_FRAME:
            return self.wajah_terakhir
        return None

    # -- FASE 2/3/4/5 gabung: semua yang butuh deteksi tangan ---------------
    def _proses_utama(self, frame, now, ts_ms):
        mp_image = bangun_mp_image(frame)
        wajah_mentah = self.pelacak_wajah.deteksi(mp_image, ts_ms, self.lebar, self.tinggi)
        wajah = self._perbarui_wajah(wajah_mentah)
        tangan = self.pelacak_tangan.deteksi(mp_image, ts_ms, self.lebar, self.tinggi)
        tangan_l = [t for t in tangan if t.bentuk == "L"]

        if self._tampilkan_diagnostik:
            if wajah is not None:
                gambar_kotak_wajah(frame, wajah.bbox)
            for t in tangan:
                gambar_rangka_tangan(frame, t.lm_px)

        if self.fase == Fase.TRANSISI_PASANG:
            self._render_transisi_pasang(frame, now)
        elif self.fase == Fase.TRANSISI_LEPAS:
            self._render_transisi_lepas(frame, now)
        elif self.fase == Fase.TOPENG_AKTIF and wajah is not None:
            lebar_t, pusat, sudut = parameter_topeng_dari_wajah(wajah)
            self.topeng.tempel(frame, lebar_t, pusat, sudut, alpha_kali=1.0)
        elif self.fase == Fase.TOPENG_AKTIF and self.wajah_terakhir is not None:
            lebar_t, pusat, sudut = parameter_topeng_dari_wajah(self.wajah_terakhir)
            self.topeng.tempel(frame, lebar_t, pusat, sudut, alpha_kali=1.0)

        # Gesture bingkai jalan terus kecuali sedang cooldown / animasi transisi
        boleh_bingkai = (
            self.fase in (Fase.GESTURE_TANGAN, Fase.TOPENG_AKTIF)
            and now >= self._boleh_toggle_setelah
        )
        if boleh_bingkai:
            event = self.pelacak_bingkai.update(tangan_l, wajah, now)
            self._tangani_event_bingkai(event, now)

            if self._tampilkan_diagnostik:
                if self.pelacak_bingkai.status == StatusBingkai.MENGUNCI:
                    gambar_teks(frame, "POSE BINGKAI TERDETEKSI...", (40, 130), 0.8)
                    gambar_bar_progres(frame, (40, 150), 260, 16, self.pelacak_bingkai.progres_kunci)
                elif self.pelacak_bingkai.status == StatusBingkai.MEMEKAR:
                    label = "LEPAS TOPENG" if self.fase == Fase.TOPENG_AKTIF else "PASANG TOPENG"
                    gambar_teks(frame, f"BINGKAI TERKUNCI — mekarkan untuk {label}", (40, 130), 0.8)
                    if self.pelacak_bingkai.sudut_a is not None:
                        gambar_garis_bingkai(frame, self.pelacak_bingkai.sudut_a, self.pelacak_bingkai.sudut_b)

        # Web shooter hanya aktif kalau topeng sedang terpasang penuh
        if self.fase == Fase.TOPENG_AKTIF:
            tembakan = self.pelacak_web.update(tangan, now)
            for asal, arah in tembakan:
                print(f"[FASE 5] Web shot! asal=({asal[0]:.0f},{asal[1]:.0f}) arah=({arah[0]:.2f},{arah[1]:.2f})")
                self.jaring.tambah(asal, arah, now)

            if self._tampilkan_diagnostik:
                for slot in self.pelacak_web.slot:
                    if slot.status == StatusTangan.TERKUNCI and slot.posisi_terakhir is not None:
                        gambar_glow_telapak(frame, slot.posisi_terakhir, slot.progres_kunci)
                        if slot.siap_tembak:
                            gambar_teks(frame, "SIAP MENEMBAK",
                                        (int(slot.posisi_terakhir[0]) - 60, int(slot.posisi_terakhir[1]) - 30),
                                        0.6, K.WARNA_GARIS_BINGKAI)

        # Jaring selalu digambar & dicek cakupannya selama topeng aktif
        if self.fase in (Fase.TOPENG_AKTIF, Fase.TRANSISI_LEPAS):
            kanvas = self.jaring.render(now)
            self.jaring.tempel_ke_frame(frame, kanvas)

            if self.fase == Fase.TOPENG_AKTIF:
                cakupan = self.jaring.cakupan_layar()
                if self._tampilkan_diagnostik:
                    gambar_teks(frame, f"Cakupan jaring: {cakupan*100:0.0f}%", (40, self.tinggi - 40), 0.7)
                if cakupan >= K.AMBANG_CAKUPAN_LAYAR:
                    print(f"[FASE 6] Cakupan layar {cakupan*100:.0f}% tercapai — mulai ending.")
                    self._mulai_fase_ending(now)

        if self._tampilkan_diagnostik:
            nama_fase = {
                Fase.GESTURE_TANGAN: "FASE 2 — MENCARI GESTURE BINGKAI",
                Fase.TRANSISI_PASANG: "FASE 3 — MEMASANG TOPENG",
                Fase.TOPENG_AKTIF: "FASE 4/5 — TOPENG AKTIF",
                Fase.TRANSISI_LEPAS: "MELEPAS TOPENG",
            }.get(self.fase, "")
            gambar_teks(frame, nama_fase, (40, 40), 0.9)

    def _tangani_event_bingkai(self, event, now):
        if event is None:
            return
        jenis = event[0]
        if jenis == "TERKUNCI":
            print("[FASE 2] Gesture bingkai terkunci.")
        elif jenis == "BATAL":
            print("[FASE 2] Gesture bingkai batal / lepas.")
        elif jenis == "TRIGGER":
            sudut_a, sudut_b, rasio = event[1]
            if self.fase == Fase.GESTURE_TANGAN:
                print(f"[FASE 3] Bingkai memekar {rasio:.2f}x — topeng dipasang.")
                self._mulai_animasi_pasang(sudut_a, sudut_b, now)
            elif self.fase == Fase.TOPENG_AKTIF:
                print(f"[FASE 4] Bingkai memekar {rasio:.2f}x — topeng dilepas.")
                self._mulai_animasi_lepas(sudut_a, sudut_b, now)

    # -- Transisi pasang / lepas topeng --------------------------------------
    def _mulai_animasi_pasang(self, sudut_a, sudut_b, now):
        bbox = bbox_dari_dua_titik(sudut_a, sudut_b)
        lebar_bingkai = max(20.0, bbox[2] - bbox[0])
        pusat_bingkai = titik_tengah(sudut_a, sudut_b)

        wajah_acuan = self.wajah_terakhir
        if wajah_acuan is not None:
            lebar_akhir, pusat_akhir, sudut_akhir = parameter_topeng_dari_wajah(wajah_acuan)
        else:
            lebar_akhir, pusat_akhir, sudut_akhir = lebar_bingkai, pusat_bingkai, 0.0

        self._data_transisi = {
            "lebar_awal": lebar_bingkai, "pusat_awal": pusat_bingkai, "sudut_awal": 0.0,
            "lebar_akhir": lebar_akhir, "pusat_akhir": pusat_akhir, "sudut_akhir": sudut_akhir,
        }
        self._mulai_transisi = now
        self.fase = Fase.TRANSISI_PASANG

    def _render_transisi_pasang(self, frame, now):
        t = clamp((now - self._mulai_transisi) / K.DURASI_ANIMASI_PASANG, 0.0, 1.0)
        te = ease_out_cubic(t)
        d = self._data_transisi
        lebar_t = lerp(d["lebar_awal"], d["lebar_akhir"], te)
        pusat = lerp_titik(d["pusat_awal"], d["pusat_akhir"], te)
        sudut = lerp(d["sudut_awal"], d["sudut_akhir"], te)
        alpha = clamp(t / 0.4, 0.0, 1.0)  # topeng muncul cepat di awal transisi
        self.topeng.tempel(frame, lebar_t, pusat, sudut, alpha_kali=alpha)

        if t >= 1.0:
            self.fase = Fase.TOPENG_AKTIF
            self._boleh_toggle_setelah = now + K.COOLDOWN_TOGGLE_TOPENG
            self.pelacak_bingkai.reset()
            print("[FASE 4] Topeng terpasang penuh.")

    def _mulai_animasi_lepas(self, sudut_a, sudut_b, now):
        bbox = bbox_dari_dua_titik(sudut_a, sudut_b)
        lebar_bingkai = max(20.0, bbox[2] - bbox[0])
        pusat_bingkai = titik_tengah(sudut_a, sudut_b)

        wajah_acuan = self.wajah_terakhir
        if wajah_acuan is not None:
            lebar_awal, pusat_awal, sudut_awal = parameter_topeng_dari_wajah(wajah_acuan)
        else:
            lebar_awal, pusat_awal, sudut_awal = lebar_bingkai, pusat_bingkai, 0.0

        self._data_transisi = {
            "lebar_awal": lebar_awal, "pusat_awal": pusat_awal, "sudut_awal": sudut_awal,
            "lebar_akhir": lebar_bingkai, "pusat_akhir": pusat_bingkai, "sudut_akhir": 0.0,
        }
        self._mulai_transisi = now
        self.fase = Fase.TRANSISI_LEPAS

    def _render_transisi_lepas(self, frame, now):
        t = clamp((now - self._mulai_transisi) / K.DURASI_ANIMASI_LEPAS, 0.0, 1.0)
        te = ease_out_cubic(t)
        d = self._data_transisi
        lebar_t = lerp(d["lebar_awal"], d["lebar_akhir"], te)
        pusat = lerp_titik(d["pusat_awal"], d["pusat_akhir"], te)
        sudut = lerp(d["sudut_awal"], d["sudut_akhir"], te)
        alpha = 1.0 - clamp((t - 0.6) / 0.4, 0.0, 1.0)  # memudar di akhir transisi
        self.topeng.tempel(frame, lebar_t, pusat, sudut, alpha_kali=alpha)

        if t >= 1.0:
            self.fase = Fase.GESTURE_TANGAN
            self._boleh_toggle_setelah = now + K.COOLDOWN_TOGGLE_TOPENG
            self.pelacak_bingkai.reset()
            print("[FASE 2] Topeng terlepas — kembali menunggu gesture bingkai.")

    # -- FASE 6: ENDING -------------------------------------------------------
    def _mulai_fase_ending(self, now):
        self.fase = Fase.ENDING
        self._mulai_ending = now
        self._audio_di_fadeout = False

    def _proses_ending(self, frame, now):
        t_sejak = now - self._mulai_ending

        # Animasi pemadat singkat: satu pola besar dari tengah supaya
        # cakupan terasa "penuh" sebelum layar menghitam.
        if t_sejak < K.DURASI_ANIMASI_PEMADAT_AKHIR and not self._pemadat_ditambahkan:
            pusat_layar = (self.lebar / 2.0, self.tinggi / 2.0)
            self.jaring.tambah(pusat_layar, (0.0, 0.0), now)
            self._pemadat_ditambahkan = True

        kanvas = self.jaring.render(now)
        self.jaring.tempel_ke_frame(frame, kanvas)

        if not self._audio_di_fadeout:
            self.audio.fadeout(K.DURASI_FADE_AUDIO)
            self._audio_di_fadeout = True
            print("[FASE 6] Audio fade out dimulai.")

        # Fade ke hitam pekat = tinggal menyusutkan kecerahan frame menuju
        # 0, jadi cukup satu scalar-multiply cv2 (bukan blend numpy
        # float32 penuh-frame dgn kanvas hitam terpisah).
        alpha_hitam = clamp(t_sejak / K.DURASI_FADE_LAYAR, 0.0, 1.0)
        frame = cv2.convertScaleAbs(frame, alpha=1.0 - alpha_hitam)

        batas_selesai = K.DURASI_FADE_LAYAR + K.DURASI_TAHAN_HITAM
        if t_sejak >= batas_selesai:
            print("[SELESAI] Program berhenti dengan rapi.")
            self._siap_berhenti = True
        else:
            self._siap_berhenti = False

        return frame

    def _bersihkan(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.pelacak_wajah.tutup()
        self.pelacak_tangan.tutup()
        pygame.mixer.quit()

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[ERROR] Gagal membaca frame dari kamera.")
                    break
                frame = cv2.flip(frame, 1)
                now = time.perf_counter()
                ts_ms = self.ts.sekarang_ms()

                if self.fase == Fase.SEARCHING:
                    self._proses_searching(frame, now, ts_ms)
                elif self.fase == Fase.ENDING:
                    frame = self._proses_ending(frame, now)
                else:
                    self._proses_utama(frame, now, ts_ms)

                self._gambar_fps(frame, now)
                cv2.imshow(self.nama_jendela, frame)

                lanjut = self._tangani_keyboard()
                if not lanjut:
                    break
                if self.fase == Fase.ENDING and self._siap_berhenti:
                    break
        finally:
            self._bersihkan()


def main():
    app = SpiderFilterApp()
    app.run()


if __name__ == "__main__":
    main()
