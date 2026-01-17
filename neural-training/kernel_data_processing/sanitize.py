import pandas as pd

def sanitize_csv(filename):
    # 1. Load the data
    # No header, as your Rust project expects raw lines
    df = pd.read_csv(filename, header=None)

    # 2. Force Data Types
    # Column 0 (Active Processes) MUST be an integer
    df[0] = df[0].astype(int)

    # 3. Ensure all other columns are Floats
    for col in range(1, 7):
        df[col] = df[col].astype(float)

    # 4. Save with Strict Formatting
    # float_format='%.6f' ensures 0.123456789 becomes 0.123457
    # index=False and header=False keeps it in the "Fugue-OS" format
    df.to_csv(filename, 
              index=False, 
              header=False, 
              float_format='%.6f')

    print(f"✅ Sanitized {filename}")
    print(df.head()) # Show the first few rows to verify

# Run it on your augmented data
sanitize_csv('training_data_augmented.csv')
