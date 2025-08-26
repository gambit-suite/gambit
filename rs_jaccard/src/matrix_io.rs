use std::path::Path;
// Keeping matrix_io to serve as clear organization for I/O operations
pub mod hdf5;
pub mod csv;
pub mod binary;
pub mod convert;

/// Utility to determine output format from file extension
pub fn detect_format_from_extension(path: &Path) -> String {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("h5") | Some("hdf5") => "hdf5".to_string(),
        Some("csv") => "csv".to_string(),
        Some("bin") | Some("binary") => "binary".to_string(),
        _ => "hdf5".to_string(), // Default to HDF5 because it takes up less space
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(detect_format_from_extension(Path::new("matrix.h5")), "hdf5");
        assert_eq!(detect_format_from_extension(Path::new("matrix.hdf5")), "hdf5");
        assert_eq!(detect_format_from_extension(Path::new("matrix.csv")), "csv");
        assert_eq!(detect_format_from_extension(Path::new("matrix.bin")), "binary");
        assert_eq!(detect_format_from_extension(Path::new("matrix")), "hdf5");
    }
}