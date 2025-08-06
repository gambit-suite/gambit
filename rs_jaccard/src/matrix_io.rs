use anyhow::{Result, Context};
use hdf5::File;
use ndarray::Array2;
use std::path::Path;

use crate::jaccard::{ScoreType};

/// Save computed matrix to hdf5 format with genome IDs || They are linked to both rows and columns
pub fn save_matrix_hdf5(
    matrix: &[Vec<ScoreType>],
    ids: &[String], 
    output_path: &Path,
) -> Result<()> {
    let n = matrix.len();
    
    // Handle empty matrix case
    if n == 0 {
        return Err(anyhow::anyhow!("Cannot save empty matrix to HDF5"));
    }
    
    // Validate that IDs match matrix dimensions
    if ids.len() != n {
        return Err(anyhow::anyhow!(
            "ID count ({}) doesn't match matrix size ({}x{})", 
            ids.len(), n, n
        ));
    }
    
    println!("Writing {}x{} matrix with {} genome IDs to HDF5 file: {}", n, n, ids.len(), output_path.display());
    let file = File::create(output_path)
        .with_context(|| format!("Failed to create HDF5 file: {}", output_path.display()))?;

    //Storing data matrix and genome ids separately, but linked by index -- I believe this is the best practice for HDF5
    
    // Create the dataset for the matrix
    let matrix_dataset = file
        .new_dataset::<f32>()
        .shape([n, n])
        .create("distances")
        .context("Failed to create distances dataset")?;
    
    // Create the dataset for genome IDs, where genome_id[i] corresponds to row i and column i
    let ids_dataset = file
        .new_dataset::<hdf5::types::VarLenAscii>()
        .shape([n])
        .create("genome_ids")
        .context("Failed to create genome_ids dataset")?;
    
    // Write out the matrix data as a data array of type f32
    let flat_data: Vec<f32> = matrix.iter()
        .flat_map(|row| row.iter().cloned())
        .collect();
        
    let matrix_array = Array2::from_shape_vec((n, n), flat_data)
        .context("Failed to convert matrix to ndarray")?;
    
    matrix_dataset.write(&matrix_array)
        .context("Failed to write matrix to HDF5")?;
    
    // Convert String IDs to HDF5-compatible ASCII strings
    let ascii_ids: Vec<hdf5::types::VarLenAscii> = ids.iter()
        .map(|s| hdf5::types::VarLenAscii::from_ascii(s.as_bytes()).unwrap_or_default())
        .collect();
    
    ids_dataset.write(&ascii_ids)
        .context("Failed to write genome IDs to HDF5")?;
    
    // Add metadata attributes for clarity
    let matrix_attr = matrix_dataset.new_attr::<hdf5::types::VarLenAscii>()
        .create("description")
        .context("Failed to create description attribute")?;
    matrix_attr.write_scalar(&hdf5::types::VarLenAscii::from_ascii(b"Jaccard distance matrix").unwrap_or_default())
        .context("Failed to write description attribute")?;
    
    let ids_attr = ids_dataset.new_attr::<hdf5::types::VarLenAscii>()
        .create("description")  
        .context("Failed to create IDs description attribute")?;
    ids_attr.write_scalar(&hdf5::types::VarLenAscii::from_ascii(b"Genome identifiers for rows and columns").unwrap_or_default())
        .context("Failed to write IDs description attribute")?;
    
    println!("HDF5 matrix with genome IDs saved successfully");
    Ok(())
}

/// THis is used for testing mainly, but could be useful for when we want to load HDF5 matrices directly
/// Returns (matrix, genome_ids)
pub fn load_matrix_hdf5(input_path: &Path) -> Result<(Vec<Vec<ScoreType>>, Vec<String>)> {
    println!("Loading matrix and genome IDs from HDF5 file: {}", input_path.display());
    let file = File::open(input_path)
        .with_context(|| format!("Failed to open HDF5 file: {}", input_path.display()))?;
    
    let matrix_dataset = file.dataset("distances")
        .context("Failed to open 'distances' dataset")?;
    let matrix_array: Array2<f32> = matrix_dataset.read()
        .context("Failed to read distance matrix")?;
    
    let ids_dataset = file.dataset("genome_ids")
        .context("Failed to open 'genome_ids' dataset")?;
    let ascii_ids = ids_dataset.read_1d::<hdf5::types::VarLenAscii>()
        .context("Failed to read genome IDs")?;
    
    // Need to make sure format is Vec<Vec<f32>>
    let matrix: Vec<Vec<ScoreType>> = matrix_array.outer_iter()
        .map(|row| row.to_vec())
        .collect();
    
    // ASCII ID in HDF5 is variable-length, so we convert to String
    let ids: Vec<String> = ascii_ids.iter()
        .map(|ascii_id| ascii_id.to_string())
        .collect();
    
    // Validate consistency -- don't want to load a matrix that doesn't match the IDs
    let n = matrix.len();
    if ids.len() != n {
        return Err(anyhow::anyhow!(
            "Inconsistent data: matrix is {}x{} but found {} genome IDs", 
            n, matrix.get(0).map_or(0, |row| row.len()), ids.len()
        ));
    }
    
    println!("Loaded {}x{} matrix with {} genome IDs", n, n, ids.len());
    Ok((matrix, ids))
}

/// Utility to determine output format from file extension
/// Would be good to probably move all I/O functions in here
pub fn detect_format_from_extension(path: &Path) -> String {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("h5") | Some("hdf5") => "hdf5".to_string(),
        Some("csv") => "csv".to_string(),
        Some("bin") | Some("binary") => "binary".to_string(),
        _ => "h5".to_string(),  // If nothing is matched, default to HDF5 because it takes up less space
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use hdf5::File;
    use ndarray::Array2;

    #[test]
    fn test_format_detection() {
        assert_eq!(detect_format_from_extension(Path::new("matrix.h5")), "hdf5");
        assert_eq!(detect_format_from_extension(Path::new("matrix.hdf5")), "hdf5");
        assert_eq!(detect_format_from_extension(Path::new("matrix.csv")), "csv");
        assert_eq!(detect_format_from_extension(Path::new("matrix.bin")), "binary");
        assert_eq!(detect_format_from_extension(Path::new("matrix")), "h5");
    }

    #[test]
    fn test_hdf5_matrix_save_and_load() {
        // Create a test matrix
        let test_matrix = vec![
            vec![0.0, 0.25, 0.75],
            vec![0.25, 0.0, 0.50],
            vec![0.75, 0.50, 0.0],
        ];
        let test_ids = vec!["genome_A".to_string(), "genome_B".to_string(), "genome_C".to_string()];

        // Create temporary file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        // Save matrix to HDF5
        save_matrix_hdf5(&test_matrix, &test_ids, temp_path).expect("Failed to save HDF5 matrix");

        // Verify file exists and can be opened
        assert!(temp_path.exists(), "HDF5 file was not created");

        // Test direct HDF5 access
        let file = File::open(temp_path).expect("Failed to open HDF5 file");
        let dataset = file.dataset("distances").expect("Failed to open distances dataset");
        let loaded_array: Array2<f32> = dataset.read().expect("Failed to read matrix data");

        // Verify dimensions
        assert_eq!(loaded_array.shape(), &[3, 3], "Matrix dimensions don't match");

        // Verify matrix contents
        for i in 0..3 {
            for j in 0..3 {
                let original = test_matrix[i][j];
                let loaded = loaded_array[[i, j]];
                assert!(
                    (original - loaded).abs() < 1e-6,
                    "Matrix value mismatch at [{},{}]: expected {}, got {}",
                    i, j, original, loaded
                );
            }
        }

        // Test our high-level load function
        let (loaded_matrix, loaded_ids) = load_matrix_hdf5(temp_path)
            .expect("Failed to load matrix with IDs");

        // Verify matrix data matches
        assert_eq!(loaded_matrix.len(), test_matrix.len(), "Matrix row count mismatch");
        for i in 0..test_matrix.len() {
            for j in 0..test_matrix[i].len() {
                let original = test_matrix[i][j];
                let loaded = loaded_matrix[i][j];
                assert!(
                    (original - loaded).abs() < 1e-6,
                    "Loaded matrix value mismatch at [{},{}]: expected {}, got {}",
                    i, j, original, loaded
                );
            }
        }

        // Verify IDs match exactly
        assert_eq!(loaded_ids, test_ids, "Genome IDs don't match");

        // Verify we can reconstruct the labeled matrix like CSV
        println!("Reconstructed labeled matrix:");
        print!("        ");
        for col_id in &loaded_ids {
            print!("{:>10}", col_id);
        }
        println!();
        for (i, row_id) in loaded_ids.iter().enumerate() {
            print!("{:>8}", row_id);
            for j in 0..loaded_matrix[i].len() {
                print!("{:>10.4}", loaded_matrix[i][j]);
            }
            println!();
        }
    }

    #[test]
    fn test_hdf5_empty_matrix() {
        let empty_matrix: Vec<Vec<f32>> = vec![];
        let empty_ids: Vec<String> = vec![];
        
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        // This should fail gracefully
        let result = save_matrix_hdf5(&empty_matrix, &empty_ids, temp_path);
        assert!(result.is_err(), "Empty matrix should cause an error");
    }

    #[test]
    fn test_hdf5_single_element_matrix() {
        let single_matrix = vec![vec![0.0]];
        let single_id = vec!["single_genome".to_string()];
        
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        // Save and verify
        save_matrix_hdf5(&single_matrix, &single_id, temp_path).expect("Failed to save single element matrix");

        let file = File::open(temp_path).expect("Failed to open HDF5 file");
        let dataset = file.dataset("distances").expect("Failed to open distances dataset");
        let loaded_array: Array2<f32> = dataset.read().expect("Failed to read matrix data");

        assert_eq!(loaded_array.shape(), &[1, 1]);
        assert_eq!(loaded_array[[0, 0]], 0.0);
    }

    #[test]
    fn test_hdf5_large_matrix() {
        // Create a larger test matrix (10x10)
        let size = 10;
        let mut large_matrix = vec![vec![0.0; size]; size];
        
        // Fill with test pattern
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    large_matrix[i][j] = 0.0;  // Diagonal
                } else {
                    large_matrix[i][j] = ((i + j) as f32) * 0.1;  // Simple pattern
                }
            }
        }

        let ids: Vec<String> = (0..size).map(|i| format!("genome_{}", i)).collect();
        
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        // Save matrix
        save_matrix_hdf5(&large_matrix, &ids, temp_path).expect("Failed to save large matrix");

        // Verify file size is reasonable (should be much smaller than CSV)
        let metadata = std::fs::metadata(temp_path).expect("Failed to get file metadata");
        let file_size = metadata.len();
        
        // For a 10x10 f32 matrix, expect roughly 400 bytes + overhead, definitely < 10KB
        assert!(file_size < 10_000, "HDF5 file size too large: {} bytes", file_size);
        assert!(file_size > 100, "HDF5 file size too small: {} bytes", file_size);

        // Spot check a few values
        let file = File::open(temp_path).expect("Failed to open HDF5 file");
        let dataset = file.dataset("distances").expect("Failed to open distances dataset");
        let loaded_array: Array2<f32> = dataset.read().expect("Failed to read matrix data");

        assert_eq!(loaded_array[[0, 0]], 0.0);  // Diagonal
        assert_eq!(loaded_array[[2, 3]], 0.5);  // (2+3)*0.1 = 0.5
        assert_eq!(loaded_array[[1, 4]], 0.5);  // (1+4)*0.1 = 0.5
    }

    #[test]
    fn test_hdf5_order_preservation() {
        // Create an asymmetric test matrix where order matters
        let test_matrix = vec![
            vec![0.0, 0.1, 0.2, 0.3],  // Row 0: Sample_A distances
            vec![1.0, 0.0, 1.2, 1.3],  // Row 1: Sample_B distances  
            vec![2.0, 2.1, 0.0, 2.3],  // Row 2: Sample_C distances
            vec![3.0, 3.1, 3.2, 0.0],  // Row 3: Sample_D distances
        ];
        let test_ids = vec![
            "Sample_A".to_string(),
            "Sample_B".to_string(), 
            "Sample_C".to_string(),
            "Sample_D".to_string()
        ];

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        // Save matrix
        save_matrix_hdf5(&test_matrix, &test_ids, temp_path).expect("Failed to save matrix");

        // Load back
        let (loaded_matrix, loaded_ids) = load_matrix_hdf5(temp_path)
            .expect("Failed to load matrix");

        // Test 1: IDs are in the same order
        assert_eq!(loaded_ids, test_ids, "ID order was not preserved");

        // Test 2: Matrix values map to correct IDs
        // Sample_A vs Sample_C should be matrix[0][2] = 0.2
        let sample_a_idx = loaded_ids.iter().position(|id| id == "Sample_A").unwrap();
        let sample_c_idx = loaded_ids.iter().position(|id| id == "Sample_C").unwrap();
        assert_eq!(loaded_matrix[sample_a_idx][sample_c_idx], 0.2);
        assert_eq!(loaded_matrix[0][2], 0.2); // Same thing, but direct indexing

        // Sample_C vs Sample_B should be matrix[2][1] = 2.1  
        let sample_b_idx = loaded_ids.iter().position(|id| id == "Sample_B").unwrap();
        assert_eq!(loaded_matrix[sample_c_idx][sample_b_idx], 2.1);
        assert_eq!(loaded_matrix[2][1], 2.1); // Same thing, but direct indexing

        // Test 3: Verify all positions match exactly
        for (i, row_id) in loaded_ids.iter().enumerate() {
            for (j, col_id) in loaded_ids.iter().enumerate() {
                let expected = test_matrix[i][j];
                let actual = loaded_matrix[i][j];
                assert!(
                    (expected - actual).abs() < 1e-6,
                    "Order mismatch: {}[{}] vs {}[{}]: expected {}, got {}",
                    row_id, i, col_id, j, expected, actual
                );
            }
        }

        // Need to make sure everything is preserved correctly for sanity
        println!("\n=== Order Preservation Verification ===");
        println!("Expected mapping (row_id[row_idx] vs col_id[col_idx] = value):");
        
        // Show a few key mappings
        let mappings = vec![
            (0, 2, "Sample_A vs Sample_C"),
            (1, 0, "Sample_B vs Sample_A"), 
            (2, 3, "Sample_C vs Sample_D"),
            (3, 1, "Sample_D vs Sample_B"),
        ];
        
        for (row_idx, col_idx, description) in mappings {
            let expected = test_matrix[row_idx][col_idx];
            let actual = loaded_matrix[row_idx][col_idx];
            println!("{}: matrix[{}][{}] = {} ✓", description, row_idx, col_idx, actual);
            assert!((expected - actual).abs() < 1e-6);
        }
    }
}