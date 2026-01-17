/// Embedding vector (384 dimensions for compatibility)
pub type Embedding = Vec<f32>;

/// Helper functions for embedding operations
pub fn cosine_similarity(emb1: &Embedding, emb2: &Embedding) -> f32 {
    if emb1.len() != emb2.len() {
        return 0.0;
    }
    
    let dot_product: f32 = emb1.iter()
        .zip(emb2.iter())
        .map(|(a, b)| a * b)
        .sum();
    
    let norm_a: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute cosine distance (1 - similarity)
pub fn cosine_distance(emb1: &Embedding, emb2: &Embedding) -> f32 {
    1.0 - cosine_similarity(emb1, emb2)
}

