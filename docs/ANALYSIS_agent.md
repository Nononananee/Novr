**Note**: This analysis was conducted before the major novel-aware enhancements. Many issues listed below have been addressed in the current implementation. For current status, see [PROGRESS_REPORT.md](PROGRESS_REPORT.md) and [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

1. agent.py
Kekurangan dan Rekomendasi Perbaikan:

Kekurangan (Status: LARGELY ADDRESSED):
Tidak ada mekanisme khusus untuk novel: Agent dirancang untuk analisis perusahaan teknologi, bukan untuk generasi novel ✅ FIXED: Novel-specific tools and prompts added

Kurangnya emotional context: Tidak ada pertimbangan untuk elemen emosional dalam narasi ✅ FIXED: Emotional intelligence system implemented

Tidak ada karakter-specific tools: Tidak ada tools khusus untuk pengembangan karakter atau alur cerita ✅ FIXED: Character management tools added

Prompt system umum: SYSTEM_PROMPT tidak dioptimalkan untuk penulisan kreatif ✅ FIXED: Novel-specific prompts implemented

Rekomendasi Perbaikan:
python
# Tambahkan tools khusus novel
@rag_agent.tool
async def generate_character_development(
    ctx: RunContext[AgentDependencies],
    character_name: str,
    current_traits: Dict[str, Any],
    target_development: str
) -> Dict[str, Any]:
    """
    Generate character development arc for a novel character.
    """
    # Implementasi khusus untuk pengembangan karakter
    pass

@rag_agent.tool  
async def plot_consistency_check(
    ctx: RunContext[AgentDependencies],
    plot_points: List[Dict[str, Any]],
    current_chapter: int
) -> Dict[str, Any]:
    """
    Check plot consistency across chapters.
    """
    # Implementasi pengecekan konsistensi alur
    pass
2. generation_pipeline.py
Kekurangan dan Rekomendasi Perbaikan:

Kekurangan:
Tergantung pada memory system eksternal: Terlalu bergantung pada IntegratedNovelMemorySystem

Tidak ada fallback mechanism: Jika memory system gagal, tidak ada alternatif

Kurangnya kontrol kualitas: Tidak ada mekanisme untuk memastikan kualitas output

Tidak ada emotional arc management: Tidak mempertimbangkan perkembangan emosional

Rekomendasi Perbaikan:
python
# Tambahkan fallback mechanism
async def generate_content(self, request: GenerationRequest) -> GenerationResult:
    try:
        # Coba dengan memory system utama
        novel_context = self._map_request_to_novel_context(request)
        novel_result = await self.memory_system.generate_with_full_context(
            generation_context=novel_context,
            prompt=request.user_prompt
        )
    except Exception as e:
        logger.warning(f"Memory system failed, using fallback: {e}")
        # Fallback ke generator sederhana
        novel_result = await self._fallback_generation(request)
    
    return self._map_novel_result_to_pipeline_result(novel_result, request)

# Tambahkan emotional arc tracking
def _map_request_to_novel_context(self, request: GenerationRequest) -> NovelGenerationContext:
    context = NovelGenerationContext(
        # ... existing fields
        emotional_arc_requirements=request.get('emotional_arc'),
        character_emotional_states=request.get('character_emotions')
    )
    return context
3. models.py
Kekurangan dan Rekomendasi Perbaikan:

Kekurangan:
Tidak ada model untuk elemen novel: Tidak ada model khusus untuk karakter, setting, plot points

Kurangnya emotional context: Tidak ada model untuk keadaan emosional

Tidak ada metadata untuk novel: Tidak ada field untuk genre, tone, style

Terlalu umum: Model didesain untuk RAG umum bukan khusus novel

Rekomendasi Perbaikan:
python
# Tambahkan model khusus novel
class Character(BaseModel):
    """Character model for novels."""
    id: str
    name: str
    personality_traits: List[str]
    background: str
    motivations: List[str]
    relationships: Dict[str, str]  # character_id -> relationship_type
    emotional_state: Optional[Dict[str, float]] = None  # emotion -> intensity
    development_arc: Optional[str] = None

class Scene(BaseModel):
    """Scene model for novels."""
    id: str
    chapter_id: str
    setting: str
    characters_present: List[str]
    plot_points: List[str]
    emotional_tone: str
    conflict_level: float = Field(ge=0.0, le=1.0)
    purpose: str  # E.g., character development, plot advancement

# Tambahkan emotional context ke request models
class NovelGenerationRequest(ChatRequest):
    """Extended request for novel generation."""
    genre: str = "fantasy"
    tone: str = "serious"
    target_emotional_arc: Optional[Dict[str, Any]] = None
    character_states: Optional[Dict[str, Dict[str, Any]]] = None
4. prompts.py
Kekurangan dan Rekomendasi Perbaikan:

Kekurangan:
Tidak spesifik untuk novel: Prompt dirancang untuk analisis perusahaan, bukan kreativitas

Tidak ada guidance untuk elemen sastra: Tidak ada panduan untuk karakter, plot, setting

Tidak mempertimbangkan genre: Tidak ada adaptasi untuk genre yang berbeda

Tidak ada emotional depth: Tidak mempertimbangkan kedalaman emosional

Rekomendasi Perbaikan:
python
# Ganti dengan prompt khusus novel
NOVEL_SYSTEM_PROMPT = """You are an AI assistant specialized in novel writing and creative storytelling. You have access to a comprehensive knowledge base of literary elements, character development techniques, and plot structures.

Your primary capabilities include:
1. Character Development: Creating compelling characters with depth and growth arcs
2. Plot Construction: Building engaging narratives with proper pacing and structure
3. World Building: Developing immersive settings and environments
4. Emotional Depth: Infusing stories with appropriate emotional resonance
5. Genre Adaptation: Tailoring content to specific genres and styles

When assisting with novel writing:
- Focus on character consistency and development
- Maintain plot coherence across chapters
- Ensure setting descriptions enhance the narrative
- Create emotional resonance appropriate to scenes
- Adapt your style to the specified genre and tone

Your responses should be:
- Creative and engaging while maintaining consistency
- Emotionally appropriate for the context
- Genre-appropriate in style and content
- Focused on advancing the narrative purpose
- Mindful of character voices and perspectives

Special considerations:
- Balance show vs. tell appropriately
- Maintain consistent point of view
- Use sensory details to enhance immersion
- Create dialogue that reflects character personalities
"""
5. tools.py
Kekurangan dan Rekomendasi Perbaikan:

Kekurangan:
Tools tidak spesifik untuk novel: Semua tools didesain untuk pencarian informasi umum

Tidak ada emotional search: Tidak bisa mencari berdasarkan keadaan emosional

Tidak ada character-centric search: Tidak bisa mencari perkembangan karakter

Tidak ada plot analysis: Tidak bisa menganalisis struktur plot

Rekomendasi Perbaikan:
python
# Tambahkan tools khusus novel
@rag_agent.tool
async def search_emotional_content(
    ctx: RunContext[AgentDependencies],
    emotion_type: str,
    intensity_threshold: float = 0.5,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for content with specific emotional qualities.
    """
    # Implementasi pencarian konten emosional
    pass

@rag_agent.tool
async def get_character_arc(
    ctx: RunContext[AgentDependencies],
    character_name: str,
    from_chapter: int = 1,
    to_chapter: Optional[int] = None
) -> Dict[str, Any]:
    """
    Retrieve the development arc of a character across chapters.
    """
    # Implementasi pengambilan arc karakter
    pass

@rag_agent.tool
async def find_plot_holes(
    ctx: RunContext[AgentDependencies],
    plot_points: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Identify potential plot inconsistencies or holes.
    """
    # Implementasi pencarian plot holes
    pass

@rag_agent.tool
async def suggest_plot_twists(
    ctx: RunContext[AgentDependencies],
    current_plot: Dict[str, Any],
    genre: str = "general"
) -> List[Dict[str, Any]]:
    """
    Suggest potential plot twists based on current narrative.
    """
    # Implementasi saran plot twists
    pass
Rekomendasi Umum untuk Sistem Novel RAG:
Tambahkan Emotional Intelligence: Integrasikan analisis emosi dalam semua komponen

Buat Novel-Specific Models: Kembangkan model khusus untuk elemen sastra

Implement Consistency Checks: Tambahkan mekanisme pengecekan konsistensi

Develop Genre Adaptation: Buat sistem yang dapat beradaptasi dengan genre berbeda

Add Creative Constraints: Implementasikan batasan kreatif untuk menjaga kualitas

Enhance Character Management: Kembangkan tools manajemen karakter yang komprehensif

Implement Plot Tracking: Buat sistem pelacakan dan analisis plot


## 6. context_optimizer.py
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak spesifik untuk konten naratif**: Algoritma optimasi dirancang untuk konten umum, bukan untuk cerita
2. **Tidak mempertimbangkan alur cerita**: Bisa memotong bagian penting yang menghubungkan plot
3. **Tidak ada preservasi dialog**: Dialog penting bisa terpotong atau terkompresi
4. **Tidak mempertimbangkan emotional beats**: Bagian emosional penting mungkin dihilangkan
### Rekomendasi Perbaikan:
```python
# Tambahkan metode khusus untuk konten naratif
def _compress_narrative(self, content: str) -> str:
    """Compress narrative content while preserving story flow."""
    
    # Pertahankan dialog dan action beats
    lines = content.split('\n')
    compressed_lines = []
    
    for line in lines:
        # Selalu pertahankan dialog
        if line.strip().startswith(('"', "'")) or ' said ' in line or ' asked ' in line:
            compressed_lines.append(line)
        # Pertahankan action penting
        elif any(action_word in line.lower() for action_word in ['suddenly', 'began', 'started', 'turned', 'looked']):
            compressed_lines.append(line)
        # Kompres deskripsi yang berlebihan
        elif len(line.split()) > 15:  # Deskripsi panjang
            compressed_line = self._compress_description(line)
            compressed_lines.append(compressed_line)
        else:
            compressed_lines.append(line)
    
    return '\n'.join(compressed_lines)
def _preserve_emotional_beats(self, content: str) -> str:
    """Preserve emotional beats in the content."""
    emotional_keywords = [
        'cried', 'laughed', 'smiled', 'frowned', 'sighed', 
        'gasped', 'shouted', 'whispered', 'tears', 'heart'
    ]
    
    sentences = re.split(r'[.!?]+', content)
    important_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Pertahankan kalimat dengan emotional keywords
        if any(keyword in sentence.lower() for keyword in emotional_keywords):
            important_sentences.append(sentence)
        # Pertahankan kalimat pendek yang impactful
        elif len(sentence.split()) <= 8:
            important_sentences.append(sentence)
    
    return '. '.join(important_sentences) + '.'
```
## 7. database_optimizer.py
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak ada caching untuk query novel**: Query yang sering digunakan tidak di-cache
2. **Tidak optimasi untuk pattern query naratif**: Pattern query untuk cerita berbeda dengan data umum
3. **Tidak ada prioritization untuk creative content**: Query kreatif tidak diprioritaskan
4. **Tidak ada bulk operations untuk emotional data**: Operasi emotional analysis tidak dioptimasi
### Rekomendasi Perbaikan:
```python
# Tambahkan caching untuk query novel
class NovelAwareDatabasePool(OptimizedDatabasePool):
    def __init__(self, database_url: str, config: PoolConfiguration = None):
        super().__init__(database_url, config)
        self.query_cache = {}
        self.novel_patterns = {
            'character_query': r'.*(character|personality|trait).*',
            'plot_query': r'.*(plot|story|narrative).*', 
            'setting_query': r'.*(setting|location|place).*'
        }
    
    async def execute_novel_query(self, query: str, *args, priority: int = 5) -> Any:
        """Execute query with novel-specific optimizations."""
        
        # Cache frequently used novel queries
        cache_key = f"{query}_{args}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Prioritize creative content queries
        if self._is_creative_query(query):
            priority = 8  # Higher priority for creative queries
        
        result = await self.execute_with_retry(query, *args)
        
        # Cache novel queries
        if self._is_novel_related_query(query):
            self.query_cache[cache_key] = result
            # Set cache expiration
            asyncio.create_task(self._expire_cache(cache_key, 300))  # 5 minutes
        
        return result
    
    def _is_creative_query(self, query: str) -> bool:
        """Check if query is related to creative content generation."""
        creative_keywords = ['generate', 'create', 'develop', 'story', 'plot', 'character']
        return any(keyword in query.lower() for keyword in creative_keywords)
    
    def _is_novel_related_query(self, query: str) -> bool:
        """Check if query is novel-related for caching."""
        novel_keywords = ['chapter', 'scene', 'character', 'plot', 'setting', 'dialogue']
        return any(keyword in query.lower() for keyword in novel_keywords)
```
## 8. memory_optimizer.py
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak ada memory management untuk emotional data**: Data emosional membutuhkan handling khusus
2. **Tidak optimasi untuk sequential story processing**: Cerita diproses secara sequential
3. **Tidak ada prioritization untuk critical story elements**: Elemen cerita penting tidak diprioritaskan
4. **Tidak ada garbage collection untuk temporal data**: Data temporal tidak dibersihkan dengan baik
### Rekomendasi Perbaikan:
```python
# Tambahkan optimasi memori khusus novel
class NovelMemoryOptimizer(MemoryOptimizer):
    def __init__(self, config: ProcessingConfig = None):
        super().__init__(config)
        self.story_chunks = {}
        self.character_memory = {}
        self.plot_memory = {}
    
    async def process_story_sequence(
        self,
        story_chunks: List[Dict[str, Any]],
        processor: callable,
        preserve_continuity: bool = True
    ) -> List[Any]:
        """Process story sequence with continuity preservation."""
        
        results = []
        previous_context = None
        
        for i, chunk in enumerate(story_chunks):
            # Maintain story continuity
            if preserve_continuity and previous_context:
                chunk['previous_context'] = previous_context
            
            # Process with memory management
            result = await self.process_large_document_streaming(
                chunk['content'],
                lambda content, idx: processor(content, idx, chunk.get('metadata', {}))
            )
            
            results.extend(result)
            
            # Update context for next chunk
            if preserve_continuity:
                previous_context = self._extract_context_for_continuity(result)
            
            # Cleanup temporal data
            if i % 10 == 0:  # Every 10 chunks
                self._cleanup_temporal_data()
        
        return results
    
    def _extract_context_for_continuity(self, result: List[Any]) -> Dict[str, Any]:
        """Extract context important for story continuity."""
        context = {
            'characters': [],
            'location': None,
            'time_period': None,
            'ongoing_plot_points': []
        }
        
        # Extract continuity elements from result
        for item in result:
            if 'characters' in item:
                context['characters'].extend(item['characters'])
            if 'location' in item:
                context['location'] = item['location']
            if 'time_period' in item:
                context['time_period'] = item['time_period']
            if 'plot_points' in item:
                context['ongoing_plot_points'].extend(item['plot_points'])
        
        return context
    
    def _cleanup_temporal_data(self):
        """Cleanup temporary data while preserving important story elements."""
        # Keep character and plot data, cleanup temporary processing data
        current_memory = self.get_memory_report()['current_memory_mb']
        if current_memory > self.config.max_memory_mb * 0.8:
            # Prioritize keeping character and plot data
            temporary_keys = [k for k in self.story_chunks.keys() 
                            if not k.startswith(('character_', 'plot_'))]
            for key in temporary_keys[:5]:  # Remove 5 temporary items
                self.story_chunks.pop(key, None)
            
            self._force_garbage_collection()
```
## 9. performance_monitor.py
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak ada metrics untuk creative processes**: Tidak memonitor proses kreatif
2. **Tidak track quality of creative output**: Hanya monitor performance, bukan kualitas
3. **Tidak ada alert untuk creative blocks**: Tidak alert ketika proses kreatif bermasalah
4. **Tidak monitor emotional consistency**: Tidak track konsistensi emosional
### Rekomendasi Perbaikan:
```python
# Tambahkan monitoring khusus untuk proses kreatif
class CreativePerformanceMonitor(PerformanceMonitor):
    def __init__(self, max_history: int = 10000):
        super().__init__(max_history)
        self.creative_metrics = {
            'character_consistency': [],
            'plot_coherence': [],
            'emotional_consistency': [],
            'style_consistency': [],
            'creativity_score': []
        }
    
    @asynccontextmanager
    async def monitor_creative_operation(self, operation_name: str, 
                                       creative_context: Dict[str, Any] = None):
        """Monitor creative operations with quality metrics."""
        
        operation_id = f"creative_{operation_name}_{time.time()}"
        start_time = time.time()
        
        with self.operation_lock:
            self.active_operations[operation_id] = {
                "name": operation_name,
                "start_time": start_time,
                "creative_context": creative_context or {},
                "type": "creative"
            }
        
        try:
            yield operation_id
            
            # Success - measure creative quality
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Simulate creative quality assessment (would be real in production)
            quality_metrics = await self._assess_creative_quality(
                operation_name, creative_context
            )
            
            metric = CreativeMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=True,
                quality_metrics=quality_metrics,
                creative_context=creative_context
            )
            
            self._record_creative_metric(metric)
            
        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            metric = CreativeMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                creative_context=creative_context
            )
            
            self._record_creative_metric(metric)
            raise
    
    async def _assess_creative_quality(self, operation_name: str, 
                                     context: Dict[str, Any]) -> Dict[str, float]:
        """Assess quality of creative output."""
        # Ini akan diintegrasikan dengan validator konsistensi
        return {
            'character_consistency': random.uniform(0.7, 0.95),
            'plot_coherence': random.uniform(0.6, 0.9),
            'emotional_consistency': random.uniform(0.65, 0.85),
            'style_consistency': random.uniform(0.7, 0.9),
            'creativity_score': random.uniform(0.6, 0.85)
        }
    
    def _record_creative_metric(self, metric: CreativeMetrics):
        """Record creative performance metric."""
        self.metrics_history.append(metric)
        
        # Track creative quality metrics
        if metric.success and hasattr(metric, 'quality_metrics'):
            for metric_name, score in metric.quality_metrics.items():
                self.creative_metrics[metric_name].append(score)
                
                # Alert on low quality
                if score < 0.6:
                    self._send_creative_alert(
                        f"Low {metric_name} in {metric.operation_name}: {score:.2f}"
                    )
```
## 10. consistency_validators_fixed.py
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Validasi terlalu umum**: Tidak spesifik untuk elemen novel
2. **Tidak ada character consistency validation**: Tidak validasi konsistensi karakter
3. **Tidak ada plot continuity checks**: Tidak cek kelanjutan plot
4. **Tidak ada style consistency validation**: Tidak validasi konsistensi gaya penulisan
### Rekomendasi Perbaikan:
```python
# Tambahkan validator khusus novel
async def character_consistency_validator(
    content: str,
    character_data: Dict[str, Any],
    established_characters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate character consistency across the story.
    """
    violations = []
    suggestions = []
    score = 1.0
    
    # Check character personality consistency
    for char_name, char_traits in established_characters.items():
        if char_name in content:
            # Check if current portrayal matches established traits
            current_portrayal = content.lower()
            for trait, value in char_traits.items():
                if trait in ['brave', 'cowardly', 'kind', 'cruel']:
                    # Check for contradictory behavior
                    if value and any(opposite in current_portrayal 
                                   for opposite in self._get_opposite_traits(trait)):
                        violations.append({
                            "type": "character_inconsistency",
                            "character": char_name,
                            "trait": trait,
                            "context": content[:100] + "..."
                        })
                        score -= 0.2
    
    return {
        "score": max(0.0, score),
        "violations": violations,
        "suggestions": suggestions,
        "validator_type": "character_consistency"
    }
async def plot_continuity_validator(
    content: str,
    current_plot: Dict[str, Any],
    previous_plots: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate plot continuity and consistency.
    """
    violations = []
    suggestions = []
    score = 1.0
    
    # Check for plot holes or inconsistencies
    for plot_point in current_plot.get('points', []):
        if not self._validate_plot_point(plot_point, previous_plots):
            violations.append({
                "type": "plot_hole",
                "plot_point": plot_point,
                "context": content[:100] + "..."
            })
            score -= 0.3
    
    # Check timeline consistency
    if not self._validate_timeline(current_plot, previous_plots):
        violations.append({
            "type": "timeline_inconsistency",
            "context": "Overall timeline check failed"
        })
        score -= 0.4
    
    return {
        "score": max(0.0, score),
        "violations": violations,
        "suggestions": suggestions,
        "validator_type": "plot_continuity"
    }
async def writing_style_validator(
    content: str,
    established_style: Dict[str, Any],
    genre: str = "general"
) -> Dict[str, Any]:
    """
    Validate consistency of writing style.
    """
    violations = []
    suggestions = []
    score = 1.0
    
    # Check point of view consistency
    pov_consistency = self._check_point_of_view(content, established_style.get('point_of_view'))
    if not pov_consistency['consistent']:
        violations.append({
            "type": "point_of_view_inconsistency",
            "expected": established_style.get('point_of_view'),
            "found": pov_consistency['found'],
            "context": content[:100] + "..."
        })
        score -= 0.3
    
    # Check tense consistency
    tense_consistency = self._check_tense_consistency(content, established_style.get('tense'))
    if not tense_consistency['consistent']:
        violations.append({
            "type": "tense_inconsistency",
            "expected": established_style.get('tense'),
            "found": tense_consistency['found'],
            "context": content[:100] + "..."
        })
        score -= 0.2
    
    return {
        "score": max(0.0, score),
        "violations": violations,
        "suggestions": suggestions,
        "validator_type": "writing_style"
    }
```
## Rekomendasi Integrasi untuk Sistem Novel RAG:
1. **Buat Novel-Specific Pipeline**: Integrasikan semua komponen khusus novel
2. **Develop Emotional Intelligence Layer**: Tambahkan layer analisis emosional
3. **Implement Cross-Chapter Consistency**: Pastikan konsistensi antar chapter
4. **Create Genre-Specific Modules**: Kembangkan modul untuk genre berbeda
5. **Build Character Management System**: Sistem komprehensif untuk manajemen karakter
6. **Add Creative Quality Assurance**: Mekanisme QA untuk output kreatif
7. **Develop Story Arc Tracking**: Pelacakan dan analisis arc cerita

Kita telah menganalisis 5 file pertama. Sekarang, saya akan menganalisis 6 file sisanya untuk sistem Novel RAG.
## 11. **api.py**
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Endpoint tidak spesifik untuk novel**: Endpoint yang ada didesain untuk chat umum, bukan untuk penulisan novel.
2. **Tidak ada endpoint untuk manajemen novel**: Seperti mengelola chapter, karakter, plot, dll.
3. **Tidak mendukung operasi batch untuk novel**: Seperti generating multiple chapters.
4. **Tidak ada endpoint untuk mendapatkan laporan novel**: Seperti consistency report, emotional arc, dll.
### Rekomendasi Perbaikan:
```python
# Tambahkan endpoint khusus novel
@app.post("/novel/generate-chapter")
async def generate_novel_chapter(request: NovelChapterRequest):
    """
    Generate a new chapter for a novel.
    """
    try:
        # Validasi dan proses generasi chapter
        chapter_content = await generate_novel_chapter_content(request)
        return {
            "chapter_id": request.chapter_id,
            "content": chapter_content,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/novel/{novel_id}/consistency-report")
async def get_novel_consistency_report(novel_id: str):
    """
    Get consistency report for a novel.
    """
    try:
        report = await generate_consistency_report(novel_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/novel/analyze-emotional-arc")
async def analyze_emotional_arc(request: EmotionalArcAnalysisRequest):
    """
    Analyze emotional arc of a novel or chapter.
    """
    try:
        analysis = await perform_emotional_arc_analysis(request)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
## 12. **approval_api.py**
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak ada validasi khusus untuk konten novel**: Validasi yang ada terlalu umum.
2. **Tidak ada mekanisme approval untuk elemen novel**: Seperti approval untuk karakter, plot twist, dll.
3. **Tidak ada integrasi dengan tools validasi novel**: Seperti consistency validators.
### Rekomendasi Perbaikan:
```python
# Tambahkan validasi dan approval khusus novel
@router.post("/propose/character")
async def propose_character(character: CharacterProposal):
    """
    Propose a new character for approval.
    """
    try:
        # Validasi karakter
        validation_results = await validate_character(character)
        
        # Buat proposal
        proposal_id = await create_proposal(
            kind="character",
            payload=character.dict(),
            validation_results=validation_results
        )
        
        return {"proposal_id": proposal_id, "validation": validation_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/propose/plot-twist")
async def propose_plot_twist(plot_twist: PlotTwistProposal):
    """
    Propose a plot twist for approval.
    """
    try:
        # Validasi plot twist
        validation_results = await validate_plot_twist(plot_twist)
        
        proposal_id = await create_proposal(
            kind="plot_twist",
            payload=plot_twist.dict(),
            validation_results=validation_results
        )
        
        return {"proposal_id": proposal_id, "validation": validation_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
## 13. **db_utils.py**
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak ada schema untuk data novel**: Seperti tabel untuk chapter, karakter, dll.
2. **Tidak ada query khusus untuk novel**: Query untuk mengambil data novel yang efisien.
3. **Tidak ada indeks untuk data naratif**: Seperti indeks untuk emotional content.
### Rekomendasi Perbaikan:
```python
# Tambahkan fungsi untuk novel
async def create_novel_tables():
    """
    Create tables specific for novel data.
    """
    async with db_pool.acquire() as conn:
        # Table untuk novel
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS novels (
                id UUID PRIMARY KEY,
                title TEXT NOT NULL,
                genre TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Table untuk chapter
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id UUID PRIMARY KEY,
                novel_id UUID REFERENCES novels(id),
                chapter_number INTEGER,
                title TEXT,
                content TEXT,
                emotional_analysis JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Table untuk karakter
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id UUID PRIMARY KEY,
                novel_id UUID REFERENCES novels(id),
                name TEXT NOT NULL,
                description TEXT,
                traits JSONB,
                development_arc JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
# Tambahkan fungsi untuk mengambil data novel
async def get_novel_chapters(novel_id: str, start_chapter: int = 1, end_chapter: Optional[int] = None):
    """
    Get chapters of a novel with optional range.
    """
    query = """
        SELECT * FROM chapters 
        WHERE novel_id = $1 AND chapter_number BETWEEN $2 AND $3
        ORDER BY chapter_number
    """
    params = [novel_id, start_chapter, end_chapter or 9999]
    
    async with db_pool.acquire() as conn:
        results = await conn.fetch(query, *params)
        return [dict(row) for row in results]
async def get_character_arc(character_id: str):
    """
    Get the development arc of a character.
    """
    query = """
        SELECT c.*, ch.chapter_number, ch.content
        FROM characters c
        JOIN chapters ch ON c.novel_id = ch.novel_id
        WHERE c.id = $1
        ORDER BY ch.chapter_number
    """
    
    async with db_pool.acquire() as conn:
        results = await conn.fetch(query, character_id)
        return [dict(row) for row in results]
```
## 14. **graph_utils.py**
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak ada entity untuk elemen novel**: Seperti node untuk karakter, plot points, dll.
2. **Tidak ada relationship khusus novel**: Seperti relationship antara karakter dan plot.
3. **Tidak ada query untuk analisis novel**: Seperti query untuk menemukan hubungan antara karakter.
### Rekomendasi Perbaikan:
```python
# Tambahkan fungsi untuk novel knowledge graph
async def create_novel_graph_nodes(novel_data: Dict[str, Any]):
    """
    Create knowledge graph nodes for novel elements.
    """
    # Buat node untuk novel
    await graph_client.add_episode(
        episode_id=f"novel_{novel_data['id']}",
        content=novel_data['summary'],
        source="novel_creation",
        metadata={
            "type": "novel",
            "title": novel_data['title'],
            "genre": novel_data['genre']
        }
    )
    
    # Buat node untuk setiap karakter
    for character in novel_data.get('characters', []):
        await graph_client.add_episode(
            episode_id=f"character_{character['id']}",
            content=character['description'],
            source="character_creation",
            metadata={
                "type": "character",
                "name": character['name'],
                "novel_id": novel_data['id']
            }
        )
async def get_character_relationships(character_id: str, depth: int = 2):
    """
    Get relationships of a character in the novel graph.
    """
    query = f"""
    MATCH (c:Character {{id: $1}})-[r*1..{depth}]-(related)
    RETURN c, r, related
    """
    
    # Implementasi query ke graph database
    results = await graph_client.execute_query(query, character_id)
    return results
async def find_plot_connections(plot_point_id: str):
    """
    Find connections between plot points and characters.
    """
    query = """
    MATCH (p:PlotPoint {id: $1})-[r]-(connected)
    RETURN p, r, connected
    """
    
    results = await graph_client.execute_query(query, plot_point_id)
    return results
```
## 15. **providers.py**
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tidak ada provider khusus untuk model kreatif**: Model untuk generasi novel mungkin berbeda.
2. **Tidak ada konfigurasi untuk creative parameters**: Seperti temperature untuk kreativitas.
3. **Tidak ada fallback untuk creative models**: Jika model kreatif utama gagal.
### Rekomendasi Perbaikan:
```python
# Tambahkan provider khusus untuk creative writing
def get_creative_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration for creative writing.
    """
    creative_choice = model_choice or os.getenv('CREATIVE_LLM_CHOICE', 'gpt-4-creative')
    base_url = os.getenv('CREATIVE_LLM_BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('CREATIVE_LLM_API_KEY')
    
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(creative_choice, provider=provider, temperature=0.8)  # Higher temperature for creativity
def get_creative_embedding_model() -> str:
    """
    Get embedding model for creative content.
    """
    return os.getenv('CREATIVE_EMBEDDING_MODEL', 'text-embedding-creative')
# Tambahkan fallback creative model
async def get_fallback_creative_model() -> OpenAIModel:
    """
    Get fallback creative model in case primary fails.
    """
    fallback_model = os.getenv('FALLBACK_CREATIVE_MODEL', 'gpt-3.5-turbo-creative')
    return get_creative_llm_model(fallback_model)
```
## 16. **tools.py**
**Kekurangan dan Rekomendasi Perbaikan:**
### Kekurangan:
1. **Tools tidak spesifik untuk novel**: Seperti yang telah disebutkan sebelumnya.
2. **Tidak ada tools untuk emotional analysis**: Tools untuk menganalisis dan menghasilkan konten emosional.
3. **Tidak ada tools untuk plot development**: Tools untuk mengembangkan plot.
### Rekomendasi Perbaikan (tambahan):
```python
# Tambahkan tools untuk emotional analysis
@rag_agent.tool
async def analyze_emotional_content(
    ctx: RunContext[AgentDependencies],
    content: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze emotional content of a story segment.
    """
    # Implementasi analisis emosional
    emotional_analysis = await emotional_analyzer.analyze(content, context)
    return emotional_analysis
@rag_agent.tool
async def generate_emotional_scene(
    ctx: RunContext[AgentDependencies],
    emotional_tone: str,
    intensity: float = 0.5,
    characters: List[str] = None
) -> Dict[str, Any]:
    """
    Generate a scene with specific emotional tone.
    """
    # Implementasi generasi adegan emosional
    scene = await emotional_scene_generator.generate(
        emotional_tone, intensity, characters
    )
    return scene
@rag_agent.tool
async def develop_plot_arc(
    ctx: RunContext[AgentDependencies],
    current_plot: Dict[str, Any],
    target_outcome: str,
    genre: str = "fantasy"
) -> Dict[str, Any]:
    """
    Develop a plot arc towards a target outcome.
    """
    # Implementasi pengembangan plot
    plot_arc = await plot_developer.develop(current_plot, target_outcome, genre)
    return plot_arc
```
## **Rekomendasi Integrasi Tambahan:**
1. **Bangun Emotional Analysis Service**: Layanan khusus untuk analisis emosional konten.
2. **Develop Plot Management System**: Sistem untuk melacak dan mengembangkan plot.
3. **Create Character Database**: Database terpusat untuk karakter dan perkembangannya.
4. **Implement Chapter Versioning**: Versi berbeda untuk chapter dengan approval process.
5. **Build Reader Feedback Integration**: Integrasi dengan umpan balik pembaca untuk perbaikan.