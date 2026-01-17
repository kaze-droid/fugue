use ort::session::Session;
use ort::value::Value;
use ndarray::Array2;
use tokenizers::Tokenizer;
use std::sync::Arc;
use super::embeddings::Embedding;

/// Service for generating text embeddings using ONNX model
pub struct EmbeddingService {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbeddingService {
    /// Create a new embedding service by loading the ONNX model and tokenizer
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[EmbeddingService] Loading model from: {}", model_path);
        let session = Session::builder()?
            .commit_from_file(model_path)?;
        
        println!("[EmbeddingService] Loading tokenizer from: {}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        
        println!("[EmbeddingService] Successfully initialized");
        Ok(Self { session, tokenizer })
    }
    
    /// Generate an embedding vector for the given text
    pub fn generate_embedding(&mut self, text: &str) -> Result<Embedding, Box<dyn std::error::Error>> {
        // Tokenize the input text
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let token_type_ids = encoding.get_type_ids();
        
        // Pad or truncate to a maximum length (typically 256 or 512 for embedding models)
        let max_length = 128;
        let mut padded_ids = vec![0i64; max_length];
        let mut padded_mask = vec![0i64; max_length];
        let mut padded_type_ids = vec![0i64; max_length];
        
        let copy_len = input_ids.len().min(max_length);
        for i in 0..copy_len {
            padded_ids[i] = input_ids[i] as i64;
            padded_mask[i] = attention_mask[i] as i64;
            padded_type_ids[i] = token_type_ids[i] as i64;
        }
        
        // Create input tensors
        let input_ids_array = Array2::from_shape_vec((1, max_length), padded_ids)?;
        let attention_mask_array = Array2::from_shape_vec((1, max_length), padded_mask)?;
        let token_type_ids_array = Array2::from_shape_vec((1, max_length), padded_type_ids)?;
        
        let input_ids_tensor = Value::from_array(input_ids_array)?;
        let attention_mask_tensor = Value::from_array(attention_mask_array)?;
        let token_type_ids_tensor = Value::from_array(token_type_ids_array)?;
        
        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ])?;
        
        // Extract the embedding (typically the last_hidden_state or pooler_output)
        // Try common output names - use get() to avoid lifetime issues
        let output_name = if outputs.get("last_hidden_state").is_some() {
            "last_hidden_state"
        } else if outputs.get("pooler_output").is_some() {
            "pooler_output"
        } else if outputs.get("sentence_embedding").is_some() {
            "sentence_embedding"
        } else {
            // Fallback: use the first output
            outputs.keys().next()
                .ok_or("No outputs from model")?
        };
        
        let output_tensor = outputs[output_name].try_extract_tensor::<f32>()?;
        
        // Get the shape and view
        let shape = &output_tensor.0;
        let view = output_tensor.1;
        
        // Mean pooling over the sequence dimension
        let embedding_vec = if shape.as_ref().len() == 3 {
            // Shape: [batch, sequence, embedding_dim]
            let seq_len = shape.as_ref()[1] as usize;
            let emb_dim = shape.as_ref()[2] as usize;
            
            let mut pooled = vec![0.0f32; emb_dim];
            for i in 0..seq_len {
                for j in 0..emb_dim {
                    let idx = i * emb_dim + j;
                    if idx < view.len() {
                        pooled[j] += view[idx];
                    }
                }
            }
            
            // Average
            for val in pooled.iter_mut() {
                *val /= seq_len as f32;
            }
            
            pooled
        } else if shape.as_ref().len() == 2 {
            // Shape: [batch, embedding_dim] - already pooled
            view.iter().copied().collect()
        } else {
            return Err("Unexpected output tensor shape".into());
        };
        
        Ok(embedding_vec)
    }
    
    /// Batch generate embeddings for multiple texts
    pub fn generate_embeddings(&mut self, texts: &[&str]) -> Result<Vec<Embedding>, Box<dyn std::error::Error>> {
        texts.iter()
            .map(|text| self.generate_embedding(text))
            .collect()
    }
}

/// Thread-safe wrapper for embedding service
pub type SharedEmbeddingService = Arc<EmbeddingService>;

/// Create a shared embedding service
pub fn create_shared_service(model_path: &str, tokenizer_path: &str) -> Result<SharedEmbeddingService, Box<dyn std::error::Error>> {
    Ok(Arc::new(EmbeddingService::new(model_path, tokenizer_path)?))
}
