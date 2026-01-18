use std::collections::{HashMap, HashSet};
use super::embeddings::{Embedding, cosine_similarity};
use super::embedding_service::EmbeddingService;

/// Represents a node in the graph-based file system
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub id: u32,
    pub name: String,
    pub path: String,
    pub file_type: String,
    pub size: u64,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub embedding: Option<Embedding>,
    pub metadata: HashMap<String, String>,
    pub content: String, // File content
}

impl GraphNode {
    pub fn new(id: u32, name: String, path: String) -> Self {
        let file_type = Self::infer_file_type(&name);
        Self {
            id,
            name,
            path,
            file_type,
            size: 0,
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            embedding: None,
            metadata: HashMap::new(),
            content: String::new(),
        }
    }
    
    /// Get file content
    pub fn get_content(&self) -> &str {
        &self.content
    }
    
    /// Set file content
    pub fn set_content(&mut self, content: String) {
        self.content = content.clone();
        self.size = content.len() as u64;
    }

    fn infer_file_type(name: &str) -> String {
        if let Some(ext) = name.split('.').last() {
            ext.to_lowercase()
        } else {
            "unknown".to_string()
        }
    }

    pub fn get_text_for_embedding(&self) -> String {
        // Include name, path, and content for better semantic matching
        let content_preview = if self.content.len() > 200 {
            &self.content[..200]
        } else {
            &self.content
        };
        format!("{} {} {}", self.name, self.path, content_preview)
    }
}

/// Graph-based file system with semantic indexing
pub struct GraphFileSystem {
    pub(crate) nodes: HashMap<u32, GraphNode>,
    pub(crate) edges: HashMap<u32, HashSet<u32>>, // node_id -> set of connected node_ids
    pub(crate) next_id: u32,
    pub(crate) similarity_threshold: f32, // cosine distance threshold for edge creation
    pub(crate) embedding_service: Option<EmbeddingService>,
}

impl GraphFileSystem {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
            similarity_threshold: 0.85, // Very strict: only connect truly similar files (85%+)
            embedding_service: None,
        }
    }

    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
            similarity_threshold: threshold,
            embedding_service: None,
        }
    }

    /// Add a new file node to the graph
    pub fn add_file(&mut self, name: String, path: String) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        
        let mut node = GraphNode::new(id, name, path);
        // Initialize position randomly
        use rand::Rng;
        let mut rng = rand::rng();
        node.x = rng.random_range(200.0..600.0);
        node.y = rng.random_range(200.0..400.0);
        
        self.nodes.insert(id, node);
        self.edges.insert(id, HashSet::new());
        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: u32) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node
    pub fn get_node_mut(&mut self, id: u32) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes
    pub fn get_all_nodes(&self) -> Vec<&GraphNode> {
        self.nodes.values().collect()
    }

    /// Get all nodes as mutable (for physics updates)
    pub fn get_all_nodes_mut(&mut self) -> Vec<&mut GraphNode> {
        // This is a workaround since we can't return mutable references from HashMap
        // We'll need to update nodes individually
        vec![]
    }

    /// Get nodes for rendering (returns clones for now)
    pub fn get_nodes_for_rendering(&self) -> Vec<GraphNode> {
        self.nodes.values().cloned().collect()
    }

    /// Update node positions (physics simulation)
    pub fn update_physics(&mut self) {
        let node_ids: Vec<u32> = self.nodes.keys().cloned().collect();
        let len = node_ids.len();
        
        // First pass: collect positions for force calculation
        let positions: Vec<(f32, f32)> = node_ids.iter()
            .filter_map(|id| self.nodes.get(id).map(|n| (n.x, n.y)))
            .collect();
        
        // Second pass: calculate forces and update
        for i in 0..len {
            if let Some(node_i) = self.nodes.get_mut(&node_ids[i]) {
                for j in 0..len {
                    if i == j { continue; }
                    if let Some(&(x_j, y_j)) = positions.get(j) {
                        let dx = node_i.x - x_j;
                        let dy = node_i.y - y_j;
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq > 0.1 && dist_sq < 10000.0 {
                            let force = 40.0 / dist_sq;
                            node_i.vx += dx * force;
                            node_i.vy += dy * force;
                        }
                    }
                }
                
                // Center attraction
                let dx = 400.0 - node_i.x;
                let dy = 300.0 - node_i.y;
                node_i.vx += dx * 0.005;
                node_i.vy += dy * 0.005;
                
                // Update position
                node_i.x += node_i.vx;
                node_i.y += node_i.vy;
                node_i.vx *= 0.90;
                node_i.vy *= 0.90;
            }
        }
    }

    /// Get connections for a node
    pub fn get_connections(&self, node_id: u32) -> Vec<u32> {
        self.edges.get(&node_id)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, from: u32, to: u32) {
        self.edges.entry(from).or_insert_with(HashSet::new).insert(to);
        self.edges.entry(to).or_insert_with(HashSet::new).insert(from);
    }

    /// Remove an edge between two nodes
    pub fn remove_edge(&mut self, from: u32, to: u32) {
        if let Some(neighbors) = self.edges.get_mut(&from) {
            neighbors.remove(&to);
        }
        if let Some(neighbors) = self.edges.get_mut(&to) {
            neighbors.remove(&from);
        }
    }
    
    /// Remove a node and all its edges
    pub fn remove_node(&mut self, id: u32) -> bool {
        if self.nodes.remove(&id).is_some() {
            // Remove all edges to/from this node
            self.edges.remove(&id);
            for neighbors in self.edges.values_mut() {
                neighbors.remove(&id);
            }
            true
        } else {
            false
        }
    }

    /// Update edges based on semantic similarity
    pub fn update_edges_by_similarity(&mut self) {
        let node_ids: Vec<u32> = self.nodes.keys().cloned().collect();
        let len = node_ids.len();
        
        // Clear existing edges
        for neighbors in self.edges.values_mut() {
            neighbors.clear();
        }
        
        // Compare all pairs
        for i in 0..len {
            for j in (i + 1)..len {
                if let (Some(node_i), Some(node_j)) = (self.nodes.get(&node_ids[i]), self.nodes.get(&node_ids[j])) {
                    if let (Some(emb_i), Some(emb_j)) = (&node_i.embedding, &node_j.embedding) {
                        let similarity = cosine_similarity(emb_i, emb_j);
                        
                        // Only connect files with high similarity (threshold is a similarity value, not distance)
                        if similarity >= self.similarity_threshold {
                            self.add_edge(node_ids[i], node_ids[j]);
                        }
                    }
                }
            }
        }
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Set similarity threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold;
    }

    /// Get similarity threshold
    pub fn get_threshold(&self) -> f32 {
        self.similarity_threshold
    }
    
    /// Read file content
    pub fn read_file(&self, node_id: u32) -> Option<String> {
        self.nodes.get(&node_id).map(|n| n.content.clone())
    }
    
    /// Write file content
    pub fn write_file(&mut self, node_id: u32, content: String) -> bool {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.set_content(content);
            // Regenerate embedding after editing
            if self.embedding_service.is_some() {
                let _ = self.generate_embedding_for_node(node_id);
                // Update edges after the embedding changes
                self.update_edges_by_similarity();
            }
            true
        } else {
            false
        }
    }
    
    /// Create a new file with content
    pub fn create_file(&mut self, name: String, path: String, content: String) -> u32 {
        let id = self.add_file(name.clone(), path.clone());
        if let Some(node) = self.nodes.get_mut(&id) {
            node.set_content(content);
        }
        // Generate embedding for the new file
        if self.embedding_service.is_some() {
            let _ = self.generate_embedding_for_node(id);
        }
        id
    }
    
    /// Initialize the embedding service
    pub fn init_embedding_service(&mut self, model_path: &str, tokenizer_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("[GraphFileSystem] Initializing embedding service...");
        self.embedding_service = Some(EmbeddingService::new(model_path, tokenizer_path)?);
        println!("[GraphFileSystem] Embedding service initialized successfully");
        Ok(())
    }
    
    /// Generate embedding for a specific node
    pub fn generate_embedding_for_node(&mut self, node_id: u32) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(service) = &mut self.embedding_service {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                let text = node.get_text_for_embedding();
                match service.generate_embedding(&text) {
                    Ok(embedding) => {
                        node.embedding = Some(embedding);
                        Ok(())
                    }
                    Err(e) => {
                        eprintln!("[GraphFileSystem] Failed to generate embedding for node {}: {}", node_id, e);
                        Err(e)
                    }
                }
            } else {
                Err("Node not found".into())
            }
        } else {
            Err("Embedding service not initialized".into())
        }
    }
    
    /// Generate embeddings for all nodes
    pub fn generate_all_embeddings(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        if self.embedding_service.is_none() {
            return Err("Embedding service not initialized".into());
        }
        
        let node_ids: Vec<u32> = self.nodes.keys().cloned().collect();
        let mut successful = 0;
        
        println!("[GraphFileSystem] Generating embeddings for {} files...", node_ids.len());
        for (i, node_id) in node_ids.iter().enumerate() {
            if (i + 1) % 5 == 0 || i == node_ids.len() - 1 {
                println!("[GraphFileSystem] Progress: {}/{}", i + 1, node_ids.len());
            }
            
            if self.generate_embedding_for_node(*node_id).is_ok() {
                successful += 1;
            }
        }
        
        println!("[GraphFileSystem] Generated {} embeddings successfully", successful);
        Ok(successful)
    }
    
    /// Update edges based on embeddings and then update them by similarity
    pub fn update_graph_with_embeddings(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // First generate embeddings for all nodes
        self.generate_all_embeddings()?;
        
        // Then update edges based on similarity
        println!("[GraphFileSystem] Updating graph edges based on similarity...");
        self.update_edges_by_similarity();
        
        let total_edges: usize = self.edges.values().map(|set| set.len()).sum();
        println!("[GraphFileSystem] Graph updated with {} edges", total_edges / 2); // Divide by 2 because edges are bidirectional
        
        Ok(())
    }
}
