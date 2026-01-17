use super::graph::{GraphFileSystem, GraphNode};

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

    /// Query by semantic similarity (disabled - embeddings removed)
    pub fn query_by_semantic(&self, _query_text: &str, _top_k: usize) -> Vec<QueryResult> {
        // Embeddings have been removed from the project
        Vec::new()
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
