use anyhow::{Result, Context, anyhow};
use hdf5::File;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::path::Path;
use crate::jaccard::{CoordType, BoundType};
use log::{info, debug, warn};

#[derive(Debug, Serialize, Deserialize)]
pub struct KmerSpec {
    pub k: u32,
    pub prefix: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SignaturesMeta {
    pub id: Option<String>,
    pub version: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub id_attr: Option<String>,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug)]
pub struct SignatureData {
    pub kmers: Vec<Vec<CoordType>>,
    pub bounds: Vec<BoundType>,
    pub ids: Vec<String>,
    pub meta: SignaturesMeta,
    pub kmer_spec: KmerSpec,
}

pub struct GambitSignatures {
    pub data: SignatureData,
}

impl GambitSignatures {
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let data = read_signatures(path)?;
        Ok(GambitSignatures { data })
    }
    
    pub fn print_info(&self) {
        println!("Total IDs: {:?}", &self.data.ids.len());
        println!("Total kmers: {:?}", &self.data.kmers.len());
        println!("Total bounds: {:?}", &self.data.bounds.len());
        println!("Loaded {} signatures", self.data.kmers.len());
        println!("K-mer size: {}", self.data.kmer_spec.k);
        if let Some(name) = &self.data.meta.name {
            println!("Dataset: {}", name);
        }
        println!("Total memory: {:?} MB", self.data.kmers.len() * self.data.kmers[0].len() * 4 / 1024 / 1024);
    }
}

pub fn read_signatures(path: &Path) -> Result<SignatureData> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open HDF5 file: {}", path.display()))?;

    // Read the k-mer values
    let values_dataset = file.dataset("values")
        .context("Failed to open 'values' dataset")?;
    
    let values_array: ndarray::Array1<u32> = values_dataset.read()
        .context("Failed to read values array")?;

    // Read the bounds
    let bounds_dataset = file.dataset("bounds")
        .context("Failed to open 'bounds' dataset")?;
    
    let bounds_array: ndarray::Array1<usize> = bounds_dataset.read()
        .context("Failed to read bounds array")?;

    // Convert to our format
    let all_coords: Vec<CoordType> = values_array.to_vec();
    let bounds: Vec<BoundType> = bounds_array.to_vec();
    
    // Extract individual k-mer sets
    let mut kmers = Vec::new();
    for i in 0..(bounds.len() - 1) {
        let start = bounds[i];
        let end = bounds[i + 1];
        let kmer_set = all_coords[start..end].to_vec();
        kmers.push(kmer_set);
    }

    // Read the actual IDs from HDF5 file with detailed debugging
    let ids: Vec<String> = if let Ok(ids_dataset) = file.dataset("ids") {
        debug!("Attempting to read IDs from HDF5 dataset '/ids'...");

        // --- ATTEMPT 1: Using read_1d with VarLenUnicode (Theoretically correct for UTF-8) ---
        match ids_dataset.read_1d::<hdf5::types::VarLenUnicode>() {
            Ok(ids_array) => {
                debug!("Read {} IDs using read_1d<VarLenUnicode>.", ids_array.len());
                let ids_vec: Vec<String> = ids_array.iter().map(|s| s.to_string()).collect();
                if !ids_vec.is_empty() {
                    debug!("First few IDs: {:?}", &ids_vec[0..std::cmp::min(5, ids_vec.len())]);
                }
                ids_vec
            },
            Err(e) => {
                debug!("Failed to read as 1D VarLenUnicode: {}", e);

                // --- ATTEMPT 2: Using read_1d with VarLenAscii (A common alternative) ---
                match ids_dataset.read_1d::<hdf5::types::VarLenAscii>() {
                    Ok(ids_array) => {
                        info!("Read {} IDs using read_1d<VarLenAscii>.", ids_array.len());
                        let ids_vec: Vec<String> = ids_array.iter().map(|s| s.to_string()).collect();
                        ids_vec
                    },
                    Err(e2) => {
                        debug!("Failed to read as 1D VarLenAscii: {}", e2);
                        warn!("Generating default sample IDs because both attempts failed.");
                        (0..kmers.len()).map(|i| format!("sample_{}", i)).collect()
                    }
                }
            }
        }
    } else {
        warn!("No 'ids' dataset found. Generating default sample IDs.");
        (0..kmers.len()).map(|i| format!("sample_{}", i)).collect()
    };
    
    // Create default metadata
    let meta = SignaturesMeta {
        id: Some("gambit_signatures".to_string()),
        version: Some("1.0".to_string()),
        name: Some("GAMBIT Signatures".to_string()),
        description: Some("K-mer signatures for Jaccard distance calculation".to_string()),
        id_attr: Some("id".to_string()),
        extra: std::collections::HashMap::new(),
    };

    // Create default k-mer spec
    let kmer_spec = KmerSpec {
        k: 11,
        prefix: "ATGAC".to_string(),
    };

    // Rebuild bounds for our format
    let mut new_bounds = vec![0];
    let mut current_pos = 0;
    for kmer_set in &kmers {
        current_pos += kmer_set.len();
        new_bounds.push(current_pos);
    }

    Ok(SignatureData {
        kmers,
        bounds: new_bounds,
        ids,
        meta,
        kmer_spec,
    })
}

// Helper functions
// Load a single signature from a SignatureData object
pub fn load_query_signature(sig_data: &SignatureData, query_idx: usize) -> Result<Vec<CoordType>> {
    if query_idx >= sig_data.kmers.len() {
        return Err(anyhow!("Query index {} out of bounds", query_idx));
    }
    Ok(sig_data.kmers[query_idx].clone())
}

pub fn load_signatures_for_jaccard(sig_data: &SignatureData) -> Result<(Vec<CoordType>, Vec<BoundType>, Vec<String>)> {
    let (flat_coords, bounds) = flatten_signatures(sig_data);
    Ok((flat_coords, bounds, sig_data.ids.clone()))
}

// Flatten the kmers for parallel processing
pub fn flatten_signatures(sig_data: &SignatureData) -> (Vec<CoordType>, Vec<BoundType>) {
    let mut flat_coords = Vec::new();
    let mut bounds = vec![0];
    
    for kmer_set in &sig_data.kmers {
        flat_coords.extend_from_slice(kmer_set);
        bounds.push(flat_coords.len());
    }
    
    (flat_coords, bounds)
}

// Convert to coordinate format suitable for Jaccard calculations
pub fn signatures_to_coords(sig_data: &SignatureData) -> Vec<Vec<CoordType>> {
    sig_data.kmers.clone()
}

pub fn debug_hdf5_ids(path: &Path) -> Result<()> {
    let file = File::open(path)?;
    
    if let Ok(ids_dataset) = file.dataset("ids") {
        println!("IDs dataset found!");
        println!("Shape: {:?}", ids_dataset.shape());
        println!("Dtype: {:?}", ids_dataset.dtype()?);
        
        // Read as a 1D array of variable-length Unicode strings
        if let Ok(ids_array) = ids_dataset.read_1d::<hdf5::types::VarLenUnicode>() {
            println!("Successfully read {} variable-length string IDs", ids_array.len());
            let first_few: Vec<String> = ids_array
                .iter()
                .take(5)
                .map(|s| s.to_string())
                .collect();
            println!("First few IDs: {:?}", first_few);
        } else {
            println!("Could not read as variable-length strings");
        }
    } else {
        println!("No IDs dataset found!");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_signature_reading() {
        // This test requires a sample HDF5 file
        let test_file = PathBuf::from("test_signatures.h5");
        if test_file.exists() {
            let result = read_signatures(&test_file);
            assert!(result.is_ok());
        }
    }
}
