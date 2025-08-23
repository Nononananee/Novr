PRIORITAS TINGGI (Wajib Edit)
1. graph_utils.py

Mengapa: graph_builder.py mengimpor GraphitiClient dari sini
Yang perlu diubah:

Interface untuk novel-specific operations
Method signatures yang mendukung narrative metadata
Handling untuk character relationships dan scene connections



2. generation_pipeline.py

Mengapa: Pipeline utama yang mengkoordinasi chunking → graph building
Yang perlu diubah:

Flow logic untuk novel processing
Error handling untuk narrative-aware chunking
Integration dengan novel-optimized graph builder



3. models.py

Mengapa: Kemungkinan berisi data models untuk chunks dan graph entities
Yang perlu diubah:

Extend models untuk narrative metadata
Character, Location, Scene models
Relationship models untuk character interactions



PRIORITAS MENENGAH (Sebaiknya Edit)
4. context_optimizer.py

Mengapa: Context optimization untuk novel berbeda dengan dokumen teknis
Yang perlu diubah:

Narrative context preservation logic
Character context tracking across chunks
Scene continuity optimization



5. memory_optimizer.py

Mengapa: Memory usage pattern untuk novel processing berbeda
Yang perlu diubah:

Optimize untuk long-form narrative content
Character relationship caching
Scene metadata memory management



6. consistency_validators_fixed.py

Mengapa: Validation rules untuk novel content berbeda
Yang perlu diubah:

Character name consistency validation
Timeline consistency checks
Narrative flow validation rules



PRIORITAS RENDAH (Opsional)
7. prompts.py

Yang perlu diubah:

Prompts untuk novel analysis tasks
Character extraction prompts
Theme identification prompts



8. performance_monitor.py

Yang perlu diubah:

Metrics untuk novel processing performance
Narrative quality metrics



Rekomendasi Urutan Editing:
1. graph_utils.py       ← Start here (breaking changes)
2. models.py           ← Update data structures
3. generation_pipeline.py ← Update main flow
4. context_optimizer.py   ← Optimize for narrative
5. memory_optimizer.py    ← Memory efficiency
6. consistency_validators_fixed.py ← Validation rules