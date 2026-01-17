// Custom persistence implementation without serde
use std::fs::File;
use std::io::{Read, Write, BufWriter, BufReader};
use super::graph::{GraphFileSystem, GraphNode};

impl GraphFileSystem {
    /// Save the graph to disk using a custom text format
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        writeln!(writer, "FUGUE_FS_V1")?;
        writeln!(writer, "next_id:{}", self.next_id)?;
        writeln!(writer, "threshold:{}", self.similarity_threshold)?;
        writeln!(writer, "node_count:{}", self.nodes.len())?;
        
        // Write nodes
        for node in self.nodes.values() {
            writeln!(writer, "---NODE---")?;
            writeln!(writer, "id:{}", node.id)?;
            writeln!(writer, "name:{}", escape_string(&node.name))?;
            writeln!(writer, "path:{}", escape_string(&node.path))?;
            writeln!(writer, "file_type:{}", escape_string(&node.file_type))?;
            writeln!(writer, "size:{}", node.size)?;
            writeln!(writer, "x:{}", node.x)?;
            writeln!(writer, "y:{}", node.y)?;
            writeln!(writer, "vx:{}", node.vx)?;
            writeln!(writer, "vy:{}", node.vy)?;
            
            // Write embedding
            if let Some(emb) = &node.embedding {
                writeln!(writer, "embedding_len:{}", emb.len())?;
                for val in emb {
                    writeln!(writer, "emb:{}", val)?;
                }
            } else {
                writeln!(writer, "embedding_len:0")?;
            }
            
            // Write content (encode length first, then content)
            let content_bytes = node.content.as_bytes();
            writeln!(writer, "content_len:{}", content_bytes.len())?;
            writer.write_all(content_bytes)?;
            writeln!(writer)?; // New line after content
        }
        
        // Write edges
        writeln!(writer, "---EDGES---")?;
        for (from_id, neighbors) in &self.edges {
            for to_id in neighbors {
                writeln!(writer, "edge:{}:{}", from_id, to_id)?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Load the graph from disk
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut contents = String::new();
        reader.read_to_string(&mut contents)?;
        
        let mut lines = contents.lines();
        
        // Check header
        if lines.next() != Some("FUGUE_FS_V1") {
            return Err("Invalid file format".into());
        }
        
        // Read metadata
        let next_id = parse_line(lines.next(), "next_id")?;
        let threshold = parse_line(lines.next(), "threshold")?;
        let node_count: usize = parse_line(lines.next(), "node_count")?;
        
        let mut fs = GraphFileSystem::new();
        fs.next_id = next_id;
        fs.similarity_threshold = threshold;
        // embedding_service will be initialized separately after loading
        
        // Read nodes
        for _ in 0..node_count {
            if lines.next() != Some("---NODE---") {
                return Err("Expected node marker".into());
            }
            
            let id = parse_line(lines.next(), "id")?;
            let name = unescape_string(&parse_line_str(lines.next(), "name")?);
            let path = unescape_string(&parse_line_str(lines.next(), "path")?);
            let file_type = unescape_string(&parse_line_str(lines.next(), "file_type")?);
            let size = parse_line(lines.next(), "size")?;
            let x = parse_line(lines.next(), "x")?;
            let y = parse_line(lines.next(), "y")?;
            let vx = parse_line(lines.next(), "vx")?;
            let vy = parse_line(lines.next(), "vy")?;
            
            // Read embedding
            let emb_len: usize = parse_line(lines.next(), "embedding_len")?;
            let embedding = if emb_len > 0 {
                let mut emb = Vec::with_capacity(emb_len);
                for _ in 0..emb_len {
                    let val = parse_line(lines.next(), "emb")?;
                    emb.push(val);
                }
                Some(emb)
            } else {
                None
            };
            
            // Read content
            let content_len: usize = parse_line(lines.next(), "content_len")?;
            let mut content = String::new();
            if content_len > 0 {
                // Collect remaining characters up to content_len
                let mut char_count = 0;
                while char_count < content_len {
                    if let Some(line) = lines.next() {
                        content.push_str(line);
                        char_count += line.len();
                        if char_count < content_len {
                            content.push('\n');
                            char_count += 1;
                        }
                    } else {
                        break;
                    }
                }
            }
            
            let mut node = GraphNode::new(id, name, path);
            node.file_type = file_type;
            node.size = size;
            node.x = x;
            node.y = y;
            node.vx = vx;
            node.vy = vy;
            node.embedding = embedding;
            node.content = content;
            
            fs.nodes.insert(id, node);
            fs.edges.insert(id, std::collections::HashSet::new());
        }
        
        // Read edges
        if lines.next() == Some("---EDGES---") {
            for line in lines {
                if let Some(stripped) = line.strip_prefix("edge:") {
                    let parts: Vec<&str> = stripped.split(':').collect();
                    if parts.len() == 2 {
                        if let (Ok(from), Ok(to)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                            fs.add_edge(from, to);
                        }
                    }
                }
            }
        }
        
        Ok(fs)
    }
}

fn escape_string(s: &str) -> String {
    s.replace('\n', "\\n").replace('\r', "\\r").replace(':', "\\:")
}

fn unescape_string(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\r", "\r").replace("\\:", ":")
}

fn parse_line<T: std::str::FromStr>(line: Option<&str>, prefix: &str) -> Result<T, Box<dyn std::error::Error>> 
where <T as std::str::FromStr>::Err: std::error::Error + 'static
{
    let line = line.ok_or(format!("Expected line with prefix {}", prefix))?;
    let value_str = line.strip_prefix(&format!("{}:", prefix))
        .ok_or(format!("Line doesn't start with {}:", prefix))?;
    value_str.parse::<T>().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

fn parse_line_str(line: Option<&str>, prefix: &str) -> Result<String, Box<dyn std::error::Error>> {
    let line = line.ok_or(format!("Expected line with prefix {}", prefix))?;
    let value_str = line.strip_prefix(&format!("{}:", prefix))
        .ok_or(format!("Line doesn't start with {}:", prefix))?;
    Ok(value_str.to_string())
}
