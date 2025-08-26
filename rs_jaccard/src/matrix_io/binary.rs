use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Save matrix with IDs in binary format
pub fn save_matrix_with_ids(matrix: &[Vec<f32>], ids: &[String], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    writer.write_u32::<LittleEndian>(matrix.len() as u32)?; // Matrix size
    writer.write_u32::<LittleEndian>(4)?; // sizeof(f32)
    
    // Write IDs (length-prefixed strings)
    for id in ids {
        let id_bytes = id.as_bytes();
        writer.write_u32::<LittleEndian>(id_bytes.len() as u32)?;
        writer.write_all(id_bytes)?;
    }
    
    // Write matrix data
    for row in matrix {
        for &value in row {
            writer.write_f32::<LittleEndian>(value)?;
        }
    }
    
    Ok(())
}