# Changelog & Untested Features

Dokumen ini melacak perubahan yang dibuat pada codebase dan fitur-fitur yang memerlukan pengujian manual karena ketidakmampuan untuk menjalankan test suite otomatis.

---

## **Perubahan 5: Desain Ulang Skema Database untuk Emotional Memory System**

- **Tanggal**: 2025-08-21
- **File yang Diubah**: `sql/schema.sql`
- **Modul Terkait**: `memory/` (implementasi di masa depan), `agent/` (penggunaan di masa depan)

### Deskripsi Perubahan
- **Desain Ulang Komprehensif**: Berdasarkan masukan dan diskusi, skema database untuk *Emotional Memory System* telah dirancang ulang secara fundamental agar lebih kuat, terukur, dan siap produksi.
- **Tabel yang Ditambahkan/Disempurnakan**:
  - `scenes`: Tabel baru untuk memberikan konteks adegan pada setiap data emosi.
  - `emotion_analysis_runs`: Tabel baru untuk melacak setiap proses analisis, menjamin idempoten dan keamanan saat analisis dijalankan ulang.
  - `emotion_labels`: Tabel baru untuk menyimpan data "gold standard" dari anotator manusia, memungkinkan evaluasi kualitas model di masa depan.
  - `character_emotions`: Diperkaya secara masif dengan kolom untuk *provenance* (model, versi, metode), granularitas (posisi kalimat/span), atribusi (siapa pemicu emosi), dan kualitas (skor kepercayaan, kalibrasi).
  - `emotion_causal_links`: Tabel baru untuk melacak hubungan sebab-akibat antar status emosi.
  - `emotional_arcs`: Tabel yang sudah ada, namun sekarang didukung oleh data yang jauh lebih kaya.
  - `character_latest_emotion`: Sebuah `MATERIALIZED VIEW` baru untuk query cepat ke status emosi terakhir karakter, lengkap dengan indeks yang dioptimalkan.
- **Fitur yang Didukung**: Perubahan ini meletakkan fondasi data untuk analisis naratif yang sangat canggih, termasuk analisis time-series, pelacakan kausalitas emosi, evaluasi model, dan *grounding* LLM yang jauh lebih presisi.

### Fitur yang Belum Teruji (Memerlukan Perhatian)
- **Validitas Skema & Migrasi**:
  - **Lokasi**: `sql/schema.sql`.
  - **Alasan**: Skema baru yang kompleks ini belum diterapkan atau diuji pada database PostgreSQL yang aktif. Perlu dipastikan skema ini valid dan proses migrasi data (jika ada) dapat berjalan lancar.
  - **Poin Risiko**: Kesalahan sintaks SQL, relasi *foreign key* yang keliru, atau masalah performa pada *view* dan *index* yang kompleks perlu divalidasi saat pertama kali diterapkan.

---

## **Perubahan 4: Pembersihan Kode (Code Cleanup)**

- **Tanggal**: 2025-08-21
- **Tindakan**: Penghapusan file-file yang tidak terpakai.

### Deskripsi Perubahan
- Menghapus 6 file modul yang sudah tidak relevan lagi setelah proses refactoring untuk mengintegrasikan arsitektur yang lebih canggih.
- **File yang Dihapus**:
  - `agent/consistency_validators.py`
  - `ingestion/chunker.py`
  - `ingestion/enhanced_scene_chunker.py`
  - `agent/enhanced_context_builder.py`
  - `agent/advanced_context_builder.py`
  - `memory/integrated_memory_controller.py`

### Fitur yang Belum Teruji (Memerlukan Perhatian)
- **Regresi Sistem**:
  - **Lokasi**: Keseluruhan sistem.
  - **Alasan**: Meskipun file-file ini diidentifikasi sebagai tidak terpakai, penghapusannya dapat secara tidak sengaja menimbulkan masalah jika ada ketergantungan tersembunyi yang tidak terdeteksi.
  - **Poin Risiko**: Perlu dilakukan verifikasi manual bahwa aplikasi (terutama proses injeksi dan generasi) masih berjalan seperti yang diharapkan setelah penghapusan file.

---

## **Perubahan 3: Refactoring Arsitektur Generation Pipeline**

- **Tanggal**: 2025-08-21
- **File yang Diubah**: `agent/generation_pipeline.py`
- **Modul Terkait**: `memory/integrated_memory_system.py`, `agent/enhanced_context_builder.py` (sekarang tidak digunakan oleh pipeline)

### Deskripsi Perubahan
- **Perubahan Arsitektur Fundamental**: Kelas `AdvancedGenerationPipeline` telah ditulis ulang sepenuhnya untuk mendelegasikan tugas utamanya ke `IntegratedNovelMemorySystem`.
- **Penyederhanaan Logika**: Pipeline sekarang berfungsi sebagai lapisan orkestrasi tipis, sementara logika kompleks untuk membangun konteks, memanggil LLM, dan validasi kini ditangani oleh sistem memori terpadu.
- **Mengganti Komponen Lama**: Ketergantungan pada `EnhancedContextBuilder` dan `IntegratedMemoryController` telah dihapus dari pipeline, digantikan oleh `IntegratedNovelMemorySystem`.

### Fitur yang Belum Teruji (Memerlukan Perhatian)
- **Seluruh Alur Kerja Generasi Konten**:
  - **Lokasi**: Semua fungsi di `agent/generation_pipeline.py`.
  - **Alasan**: Ini adalah perubahan arsitektur inti. Cara konten dibuat telah diubah secara fundamental.
  - **Poin Risiko**: Kualitas konteks, kualitas generasi, dan penanganan error.

---

## **Perubahan 2: Refactoring Pipeline Injeksi (Ingestion)**

- **Tanggal**: 2025-08-21
- **File yang Diubah**: `ingestion/ingest.py`
- **Modul Terkait**: `memory/chunking_strategies.py`

### Deskripsi Perubahan
- **Mengganti Chunker Lama**: Logika `DocumentIngestionPipeline` telah diubah untuk menggunakan `NovelChunker` yang lebih canggih.
- **Adaptasi Tipe Data**: Menambahkan metode konversi untuk output dari `NovelChunker`.

### Fitur yang Belum Teruji (Memerlukan Perhatian)
- **Seluruh Pipeline Injeksi Data**:
  - **Lokasi**: Fungsionalitas yang dijalankan oleh `ingestion/ingest.py`.
  - **Alasan**: Perubahan besar pada komponen inti chunking.
  - **Poin Risiko**: Kualitas chunking, konversi data, dan performa.

---

## **Perubahan 1: Perbaikan Bug Kritis pada Validator Konsistensi**

- **Tanggal**: 2025-08-21
- **File yang Diubah**: `agent/approval_api.py`

### Deskripsi Perubahan
- Mengubah impor `run_all_validators` ke modul `consistency_validators_fixed`.

### Fitur yang Belum Teruji (Memerlukan Perhatian)
- **Alur Kerja Persetujuan (Approval Workflow)**:
  - **Lokasi**: Semua endpoint di bawah `/approval`.
  - **Alasan**: Menggunakan logika validasi yang diperbarui.