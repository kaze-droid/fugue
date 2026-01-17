use ort::session::Session;
use ort::value::Value;
use ndarray::Array2;
use tokenizers::Tokenizer;
use std::fs::File;
use std::io::Read;

pub struct ThemeEngine {
    session: Session,
    tokenizer: Tokenizer,
    wallpaper_embeddings: Vec<(String, Vec<f32>)>,
}

impl ThemeEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        println!("[ThemeEngine] Initializing...");
        
        // Load ONNX model
        println!("[ThemeEngine] Loading ONNX model...");
        let session = Session::builder()?
            .commit_from_file("text_embedding_model.onnx")?;
        println!("[ThemeEngine] ✓ Model loaded");
        
        // Load tokenizer
        println!("[ThemeEngine] Loading tokenizer...");
        let tokenizer = Tokenizer::from_file(
            "tokenizer.json"
        ).map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        println!("[ThemeEngine] ✓ Tokenizer loaded");
        
        // Load pre-computed embeddings
        println!("[ThemeEngine] Loading wallpaper embeddings...");
        let wallpaper_embeddings = Self::load_embeddings(
            "wallpaper_embeddings.bin"
        )?;
        println!("[ThemeEngine] ✓ Loaded {} wallpapers", wallpaper_embeddings.len());
        
        Ok(Self {
            session,
            tokenizer,
            wallpaper_embeddings,
        })
    }
    
    fn load_embeddings(path: &str) -> Result<Vec<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        let mut result = Vec::new();
        let mut offset = 0;
        
        // Read number of wallpapers
        let num_wallpapers = u32::from_le_bytes([
            buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
        ]) as usize;
        offset += 4;
        
        // Read each wallpaper
        for _ in 0..num_wallpapers {
            // Read name length
            let name_len = u32::from_le_bytes([
                buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
            ]) as usize;
            offset += 4;
            
            // Read name
            let name = String::from_utf8(buffer[offset..offset+name_len].to_vec())?;
            offset += name_len;
            
            // Read embedding (384 floats)
            let mut embedding = Vec::with_capacity(384);
            for _ in 0..384 {
                let bytes = [
                    buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
                ];
                embedding.push(f32::from_le_bytes(bytes));
                offset += 4;
            }
            
            result.push((name, embedding));
        }
        
        Ok(result)
    }
    
    fn tokenize(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        
        Ok((input_ids, attention_mask))
    }
    
    fn get_embedding(&mut self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Pad/truncate to 128 tokens (model's max length)
        let max_len = 128;
        let mut padded_ids = vec![0i64; max_len];
        let mut padded_mask = vec![0i64; max_len];
        let padded_token_type_ids = vec![0i64; max_len]; // All zeros for single sentence
        
        let copy_len = input_ids.len().min(max_len);
        padded_ids[..copy_len].copy_from_slice(&input_ids[..copy_len]);
        padded_mask[..copy_len].copy_from_slice(&attention_mask[..copy_len]);
        
        // Create input tensors
        let input_ids_array = Array2::from_shape_vec((1, max_len), padded_ids)?;
        let attention_mask_array = Array2::from_shape_vec((1, max_len), padded_mask)?;
        let token_type_ids_array = Array2::from_shape_vec((1, max_len), padded_token_type_ids)?;
        
        let input_ids_tensor = Value::from_array(input_ids_array)?;
        let attention_mask_tensor = Value::from_array(attention_mask_array)?;
        let token_type_ids_tensor = Value::from_array(token_type_ids_array)?;
        
        // Run inference with all required inputs
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ])?;
        
        // Extract embedding (output is "sentence_embedding" or first output)
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let embedding: Vec<f32> = output_tensor.1.iter().copied().collect();
        
        Ok(embedding)
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    pub fn find_wallpaper(&mut self, prompt: &str) -> Result<(String, f32), Box<dyn std::error::Error>> {
        println!("[ThemeEngine] Finding wallpaper for: '{}'", prompt);
        
        // Tokenize and get embedding for user prompt
        let (input_ids, attention_mask) = self.tokenize(prompt)?;
        let prompt_embedding = self.get_embedding(&input_ids, &attention_mask)?;
        
        // Find best match
        let mut best_match = ("".to_string(), 0.0f32);
        
        for (name, embedding) in &self.wallpaper_embeddings {
            let similarity = Self::cosine_similarity(&prompt_embedding, embedding);
            
            if similarity > best_match.1 {
                best_match = (name.clone(), similarity);
            }
        }
        
        println!("[ThemeEngine] Best match: {} (similarity: {:.4})", best_match.0, best_match.1);
        
        Ok(best_match)
    }
}
