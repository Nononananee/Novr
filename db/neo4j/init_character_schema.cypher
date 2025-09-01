// Neo4j Character Schema Initialization Script
// Run this script to set up the character relationship graph schema

// Create constraints for data integrity
CREATE CONSTRAINT character_id_unique IF NOT EXISTS FOR (c:Character) REQUIRE c.character_id IS UNIQUE;
CREATE CONSTRAINT character_project_id IF NOT EXISTS FOR (c:Character) REQUIRE c.project_id IS NOT NULL;
CREATE CONSTRAINT location_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE;
CREATE CONSTRAINT plot_arc_id_unique IF NOT EXISTS FOR (p:PlotArc) REQUIRE p.arc_id IS UNIQUE;

// Create indexes for performance
CREATE INDEX character_name_index IF NOT EXISTS FOR (c:Character) ON (c.name);
CREATE INDEX character_project_index IF NOT EXISTS FOR (c:Character) ON (c.project_id);
CREATE INDEX location_name_index IF NOT EXISTS FOR (l:Location) ON (l.name);
CREATE INDEX plot_arc_name_index IF NOT EXISTS FOR (p:PlotArc) ON (p.name);

// Create sample character nodes for demo project
MERGE (lyra:Character {
    character_id: "demo_lyra_stormwind",
    project_id: "demo",
    name: "Lyra Stormwind",
    description: "19-year-old protagonist with newly awakened Storm Crystal abilities",
    traits: ["determined", "quick-tempered", "loyal", "inexperienced"],
    role: "protagonist",
    crystal_type: "Storm Crystal",
    created_at: datetime()
});

MERGE (theron:Character {
    character_id: "demo_theron_brightforge",
    project_id: "demo",
    name: "Master Theron Brightforge",
    description: "45-year-old Master Shaper mentor haunted by past failures",
    traits: ["patient", "perfectionist", "cautious", "wise"],
    role: "mentor",
    crystal_type: "Golden Forge Crystal",
    created_at: datetime()
});

MERGE (kael:Character {
    character_id: "demo_kael_shadowmere",
    project_id: "demo",
    name: "Kael Shadowmere",
    description: "23-year-old Void Seeker operative with a dark past",
    traits: ["brooding", "protective", "secretive", "conflicted"],
    role: "deuteragonist",
    crystal_type: "Void Shard",
    created_at: datetime()
});

MERGE (mira:Character {
    character_id: "demo_mira_dawnbringer",
    project_id: "demo",
    name: "Captain Mira Dawnbringer",
    description: "35-year-old leader of Lumina Guard with crystal prosthetic arm",
    traits: ["tactical", "protective", "pragmatic", "fair"],
    role: "ally",
    crystal_type: "Shield Crystal",
    created_at: datetime()
});

MERGE (zara:Character {
    character_id: "demo_zara_resonator",
    project_id: "demo",
    name: "Zara the Resonator",
    description: "16-year-old prodigy from wealthy merchant family",
    traits: ["brilliant", "naive", "competitive", "kind-hearted"],
    role: "friend_rival",
    crystal_type: "Harmony Crystal",
    created_at: datetime()
});

// Create character relationships
MERGE (lyra)-[:MENTORED_BY {
    strength: "strong",
    duration: "recent",
    description: "Theron reluctantly trains Lyra in crystal wielding"
}]->(theron);

MERGE (lyra)-[:ALLIES_WITH {
    strength: "growing",
    trust_level: "developing",
    description: "Initial distrust evolving into deep friendship"
}]->(kael);

MERGE (lyra)-[:FRIENDS_WITH {
    strength: "strong",
    rivalry_type: "friendly",
    description: "Competitive friendship that pushes both to improve"
}]->(zara);

MERGE (lyra)-[:PROTECTED_BY {
    strength: "moderate",
    official: true,
    description: "Mira provides official protection and guidance"
}]->(mira);

MERGE (kael)-[:DISTRUSTS {
    strength: "moderate",
    reason: "different_methods",
    description: "Kael questions Theron's cautious approach"
}]->(theron);

MERGE (theron)-[:CONCERNED_ABOUT {
    strength: "high",
    reason: "dangerous_path",
    description: "Theron worries about Kael's use of Void magic"
}]->(kael);

MERGE (zara)-[:ADMIRES {
    strength: "moderate",
    aspect: "experience",
    description: "Zara looks up to Mira's practical leadership"
}]->(mira);

// Create location nodes
MERGE (crystal_peaks:Location {
    location_id: "demo_crystal_peaks",
    project_id: "demo",
    name: "The Crystal Peaks",
    description: "Northern mountains with largest Aether Crystal formations",
    location_type: "mountain_range",
    danger_level: "high",
    created_at: datetime()
});

MERGE (port_lumina:Location {
    location_id: "demo_port_lumina",
    project_id: "demo",
    name: "Port Lumina",
    description: "Largest city and trading hub with crystal-powered ships",
    location_type: "city",
    danger_level: "low",
    created_at: datetime()
});

MERGE (whispering_woods:Location {
    location_id: "demo_whispering_woods",
    project_id: "demo",
    name: "The Whispering Woods",
    description: "Forest where trees are infused with crystal fragments",
    location_type: "forest",
    danger_level: "medium",
    created_at: datetime()
});

// Connect characters to locations
MERGE (lyra)-[:GREW_UP_IN]->(whispering_woods);
MERGE (lyra)-[:CURRENTLY_IN]->(port_lumina);
MERGE (theron)-[:LIVES_IN]->(port_lumina);
MERGE (mira)-[:COMMANDS_IN]->(port_lumina);
MERGE (kael)-[:OPERATES_FROM]->(crystal_peaks);
MERGE (zara)-[:LIVES_IN]->(port_lumina);

// Create plot arc nodes
MERGE (hero_journey:PlotArc {
    arc_id: "demo_hero_journey",
    project_id: "demo",
    name: "Hero's Journey",
    description: "Lyra's transformation from farm girl to crystal wielder",
    arc_type: "character_development",
    progress: 0.2,
    key_events: ["power_discovery", "mentor_meeting", "first_challenge"],
    created_at: datetime()
});

MERGE (mystery_arc:PlotArc {
    arc_id: "demo_family_mystery",
    project_id: "demo",
    name: "Family Mystery",
    description: "Uncovering the truth about Lyra's family connection to the Shattering",
    arc_type: "mystery",
    progress: 0.1,
    key_events: ["strange_crystal", "family_secrets"],
    created_at: datetime()
});

// Connect characters to plot arcs
MERGE (lyra)-[:CENTRAL_TO]->(hero_journey);
MERGE (lyra)-[:INVOLVED_IN]->(mystery_arc);
MERGE (theron)-[:GUIDES_IN]->(hero_journey);
MERGE (kael)-[:CONNECTED_TO]->(mystery_arc);

// Create useful queries as stored procedures (if APOC is available)
// These would be custom procedures for common character context queries

// Example query patterns for reference:

// Get character with all relationships:
// MATCH (c:Character {character_id: $character_id, project_id: $project_id})
// OPTIONAL MATCH (c)-[r]-(related:Character)
// RETURN c, collect({relationship: type(r), character: related.name, properties: properties(r)}) as relationships

// Get characters in location:
// MATCH (c:Character {project_id: $project_id})-[:CURRENTLY_IN|LIVES_IN|OPERATES_FROM]->(l:Location {location_id: $location_id})
// RETURN c

// Get plot arc participants:
// MATCH (c:Character {project_id: $project_id})-[r:CENTRAL_TO|INVOLVED_IN|GUIDES_IN|CONNECTED_TO]->(p:PlotArc {arc_id: $arc_id})
// RETURN c, type(r) as involvement_type

// Find relationship path between characters:
// MATCH path = shortestPath((c1:Character {character_id: $char1_id})-[*1..3]-(c2:Character {character_id: $char2_id}))
// WHERE c1.project_id = $project_id AND c2.project_id = $project_id
// RETURN path

RETURN "Character schema initialized successfully" as result;