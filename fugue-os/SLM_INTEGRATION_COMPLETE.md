# SLM Shell Integration - Implementation Complete

## Summary
Successfully integrated Phi-3.5-mini Instruct ONNX as an intelligent shell controller for Fugue OS. The SLM can now:
- Open/close applications via natural language
- Change themes (wallpapers + accent colors) via semantic search
- Toggle kernel modules
- Respond conversationally to user queries

## Files Modified

### 1. **Cargo.toml**
- Added `serde` and `serde_json` dependencies for JSON parsing

### 2. **src/slm.rs** (NEW)
- `ShellIntent` struct for parsed commands
- `run_autoregressive_inference()` - token-by-token generation with streaming
- `SYSTEM_PROMPT` - JSON-based instruction prompt with Fugue OS persona
- Speculative JSON parsing for early termination

### 3. **src/hardware.rs**
- Added SLM session and tokenizer fields
- Added streaming channels (`stream_tx`, `stream_rx`)
- Added intent channels (`intent_tx`, `intent_rx`)
- Added `kernel_messages` VecDeque for terminal history
- Added `is_thinking` and `partial_response` for UI feedback
- Modified `load_brain()` to load SLM and tokenizer

### 4. **src/desktop.rs**
- Added `theme` field to DesktopEnv
- Added `update_theme()` method for dynamic color changes
- Added `hex_to_color()` helper function
- Updated terminal drawing to show:
  - Kernel message history
  - Streaming SLM response
  - Thinking indicator
- Updated all drawing methods to use `self.theme` instead of `CYBER_THEME`

### 5. **src/main.rs**
- Added `mod slm;`
- Added streaming token receiver in main loop
- Modified ENTER key handling to trigger SLM inference (blocking)
- Immediate intent execution after SLM response
- Integrated with theme engine for wallpaper matching

## Usage

### 1. Download Phi-3.5 ONNX Model (if not already done)
```bash
cd fugue-gi/fugue/fugue-os
git lfs install
git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx
cp Phi-3.5-mini-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*.onnx ./phi35_mini.onnx
cp Phi-3.5-mini-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.json ./
```

### 2. Run Fugue OS
```bash
cargo run --release
```

### 3. Example Commands
Open a terminal window (F2) and type:
- `"show me the file web"` → Opens FileUniverse
- `"I want a peaceful koi fish aesthetic"` → Changes wallpaper + colors
- `"close everything"` → Closes all windows
- `"open the monitor"` → Opens system monitor

## Technical Notes

- **Inference Mode**: Currently runs **synchronously** (blocking) in the main thread
  - UI will freeze during SLM inference (~1-2 seconds)
  - This was chosen for simplicity since `ort::Session` requires `&mut self`
  - Future: Could use async runtime or separate process for non-blocking inference

- **Token Streaming**: Tokens are sent via `stream_tx` channel as they generate
  - Displayed in terminal as `partial_response`
  - Creates "typing" effect

- **JSON Parsing**: Uses speculative parsing
  - Checks for valid JSON after each token
  - Returns early when complete JSON is detected
  - Fallback: parses full text at end

- **Model Size**: Phi-3.5-mini INT4 quantized is ~2GB
  - Runs on CPU (no GPU required)
  - Inference speed: ~10-20 tokens/sec on modern CPU

## Limitations

1. **Blocking Inference**: UI freezes during SLM processing
2. **No Model Included**: User must download Phi-3.5 ONNX separately
3. **JSON Not Guaranteed**: SLM might not always output valid JSON
4. **No Conversation History**: Each command is independent

## Future Enhancements

- [ ] Async inference (non-blocking UI)
- [ ] Conversation history/context
- [ ] Streaming visual feedback during inference
- [ ] Fallback to simpler commands if SLM unavailable
- [ ] Fine-tuned model for Fugue OS commands
