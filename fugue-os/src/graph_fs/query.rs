use super::graph::{GraphFileSystem, GraphNode};
use super::embeddings::cosine_similarity;

/// Result of a query
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub node_id: u32,
    pub node: GraphNode,
    pub score: f32,
}

impl GraphFileSystem {
    /// Query by filename keywords
    pub fn query_by_keywords(&self, keywords: &str) -> Vec<QueryResult> {
        let keywords_lower = keywords.to_lowercase();
        let keyword_parts: Vec<&str> = keywords_lower.split_whitespace().collect();
        
        let mut results: Vec<QueryResult> = self.nodes.values()
            .filter_map(|node| {
                let node_text = format!("{} {}", node.name.to_lowercase(), node.path.to_lowercase());
                let mut score = 0.0;
                
                for keyword in &keyword_parts {
                    if node_text.contains(keyword) {
                        score += 1.0;
                    }
                }
                
                if score > 0.0 {
                    Some(QueryResult {
                        node_id: node.id,
                        node: node.clone(),
                        score: score / keyword_parts.len() as f32,
                    })
                } else {
                    None
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Query by semantic similarity using embeddings
    pub fn query_by_semantic(&mut self, query_text: &str, top_k: usize) -> Vec<QueryResult> {
        // Generate embedding for the query text
        let query_embedding = match &mut self.embedding_service {
            Some(service) => {
                match service.generate_embedding(query_text) {
                    Ok(emb) => emb,
                    Err(e) => {
                        eprintln!("[GraphFileSystem] Failed to generate query embedding: {}", e);
                        return Vec::new();
                    }
                }
            }
            None => {
                eprintln!("[GraphFileSystem] Embedding service not initialized");
                return Vec::new();
            }
        };
        
        // Calculate similarity with all nodes that have embeddings
        let mut results: Vec<QueryResult> = self.nodes.values()
            .filter_map(|node| {
                if let Some(node_emb) = &node.embedding {
                    let similarity = cosine_similarity(&query_embedding, node_emb);
                    Some(QueryResult {
                        node_id: node.id,
                        node: node.clone(),
                        score: similarity,
                    })
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top K results
        results.truncate(top_k);
        results
    }
    
    /// Hybrid search: combines keyword matching with semantic similarity
    pub fn query_hybrid(&mut self, query_text: &str, top_k: usize) -> Vec<QueryResult> {
        // Get keyword results
        let keyword_results = self.query_by_keywords(query_text);
        
        // Get semantic results
        let semantic_results = self.query_by_semantic(query_text, top_k * 2);
        
        // Combine results with weighted scoring
        // Keyword match: 40%, Semantic similarity: 60%
        let mut combined_scores: std::collections::HashMap<u32, (f32, GraphNode)> = std::collections::HashMap::new();
        
        // Add keyword scores (normalized)
        let max_keyword_score = keyword_results.first().map(|r| r.score).unwrap_or(1.0);
        for result in keyword_results {
            let normalized_score = if max_keyword_score > 0.0 {
                result.score / max_keyword_score
            } else {
                0.0
            };
            combined_scores.insert(result.node_id, (normalized_score * 0.4, result.node));
        }
        
        // Add semantic scores
        for result in semantic_results {
            let entry = combined_scores.entry(result.node_id).or_insert((0.0, result.node.clone()));
            entry.0 += result.score * 0.6;
        }
        
        // Convert to results and sort
        let mut results: Vec<QueryResult> = combined_scores
            .into_iter()
            .map(|(node_id, (score, node))| QueryResult { node_id, node, score })
            .collect();
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
    
    /// Search with automatic method selection
    pub fn search(&mut self, query_text: &str, top_k: usize) -> Vec<QueryResult> {
        // Use hybrid search if embeddings are available, otherwise fall back to keywords
        if self.embedding_service.is_some() {
            self.query_hybrid(query_text, top_k)
        } else {
            self.query_by_keywords(query_text)
        }
    }

    /// Get neighbors of a node (graph traversal)
    pub fn get_neighbors(&self, node_id: u32, depth: usize) -> Vec<QueryResult> {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut results = Vec::new();
        
        queue.push_back((node_id, 0));
        visited.insert(node_id);
        
        while let Some((current_id, current_depth)) = queue.pop_front() {
            if current_depth > depth {
                continue;
            }
            
            if let Some(node) = self.nodes.get(&current_id) {
                if current_id != node_id || current_depth == 0 {
                    results.push(QueryResult {
                        node_id: current_id,
                        node: node.clone(),
                        score: 1.0 / (current_depth as f32 + 1.0),
                    });
                }
                
                if current_depth < depth {
                    if let Some(neighbors) = self.edges.get(&current_id) {
                        for &neighbor_id in neighbors {
                            if !visited.contains(&neighbor_id) {
                                visited.insert(neighbor_id);
                                queue.push_back((neighbor_id, current_depth + 1));
                            }
                        }
                    }
                }
            }
        }
        
        results
    }

    /// Find shortest path between two nodes
    pub fn shortest_path(&self, from: u32, to: u32) -> Option<Vec<u32>> {
        use std::collections::{VecDeque, HashMap};
        
        if from == to {
            return Some(vec![from]);
        }
        
        let mut queue = VecDeque::new();
        let mut parent = HashMap::new();
        let mut visited = std::collections::HashSet::new();
        
        queue.push_back(from);
        visited.insert(from);
        
        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = to;
                while let Some(&prev) = parent.get(&node) {
                    path.push(node);
                    node = prev;
                }
                path.push(from);
                path.reverse();
                return Some(path);
            }
            
            if let Some(neighbors) = self.edges.get(&current) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        parent.insert(neighbor, current);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        None
    }

    /// Find clusters of connected nodes
    pub fn find_clusters(&self) -> Vec<Vec<u32>> {
        let mut visited = std::collections::HashSet::new();
        let mut clusters = Vec::new();
        
        for &node_id in self.nodes.keys() {
            if !visited.contains(&node_id) {
                let mut cluster = Vec::new();
                let mut stack = vec![node_id];
                
                while let Some(current) = stack.pop() {
                    if visited.contains(&current) {
                        continue;
                    }
                    visited.insert(current);
                    cluster.push(current);
                    
                    if let Some(neighbors) = self.edges.get(&current) {
                        for &neighbor in neighbors {
                            if !visited.contains(&neighbor) {
                                stack.push(neighbor);
                            }
                        }
                    }
                }
                
                if !cluster.is_empty() {
                    clusters.push(cluster);
                }
            }
        }
        
        clusters
    }
}
