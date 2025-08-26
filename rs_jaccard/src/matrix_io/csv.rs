use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Save distance array to CSV format (simple list)
pub fn save_distances(distances: &[f32], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    for distance in distances {
        writeln!(writer, "{:.4}", distance)?;
    }
    
    Ok(())
}

/// Save similar pairs to CSV format with headers
pub fn save_similar_pairs(pairs: &[(usize, usize, f32)], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(file);
    
    writer.write_record(&["i", "j", "distance"])?;
    for (i, j, dist) in pairs {
        writer.write_record(&[i.to_string(), j.to_string(), format!("{:.4}", dist)])?;
    }
    
    writer.flush()?;
    Ok(())
}

/// Save matrix with IDs as both row and column labels (square matrix)
pub fn save_matrix_with_ids(matrix: &[Vec<f32>], ids: &[String], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(file);
    
    // Write header row (column names)
    let mut header = vec!["".to_string()];
    header.extend(ids.iter().cloned());
    writer.write_record(&header)?;
    
    // Write matrix rows with row IDs
    for (i, row) in matrix.iter().enumerate() {
        let mut csv_row = vec![ids[i].clone()];
        csv_row.extend(row.iter().map(|&x| format!("{:.4}", x)));
        writer.write_record(&csv_row)?;
    }
    
    writer.flush()?;
    Ok(())
}

/// Save matrix with different query and reference IDs (rectangular matrix)
pub fn save_query_ref_matrix(matrix: &[Vec<f32>], query_ids: &[String], ref_ids: &[String], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(file);
    
    // Write header row (column names: empty first column, then reference IDs)
    let mut header = vec!["".to_string()];
    header.extend(ref_ids.iter().cloned());
    writer.write_record(&header)?;
    
    // Write matrix rows with query IDs as row labels
    for (i, row) in matrix.iter().enumerate() {
        let mut csv_row = vec![query_ids[i].clone()];
        csv_row.extend(row.iter().map(|&x| format!("{:.4}", x)));
        writer.write_record(&csv_row)?;
    }
    
    writer.flush()?;
    Ok(())
}

/// Write LSH results to CSV format
pub fn write_lsh_results(output: &Path, candidates: &[(usize, usize, f32)]) -> Result<()> {
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "i,j,similarity")?;
    for &(i, j, similarity) in candidates {
        writeln!(writer, "{},{},{:.4}", i, j, similarity)?;
    }
    
    Ok(())
}