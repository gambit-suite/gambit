use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{stdout, Write};
use anyhow::Result;
use csv;
use log::{info, debug, warn};

pub type CoordType = u32;
pub type BoundType = usize;
pub type ScoreType = f32;

/// Core Jaccard distance calculation
#[inline(always)]
pub fn jaccard_distance_core(coords1: &[CoordType], coords2: &[CoordType]) -> ScoreType {
    if coords1.is_empty() && coords2.is_empty() {
        return 0.0;
    }
    if coords1.is_empty() || coords2.is_empty() {
        return 1.0;
    }

    let mut i = 0;
    let mut j = 0;
    let mut intersection = 0;

    while i < coords1.len() && j < coords2.len() {
        match coords1[i].cmp(&coords2[j]) {
            Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }

    let union_size = coords1.len() + coords2.len() - intersection;
    if union_size == 0 {
        return 0.0;
    }

    1.0 - (intersection as f32 / union_size as f32)
}

/// Parallel Jaccard distance calculation for one-vs-many
pub fn jaccard_distances_parallel(
    query: &[CoordType],
    all_coords: &[CoordType],
    bounds: &[BoundType],
) -> Vec<ScoreType> {
    let n_references = bounds.len() - 1;
    
    (0..n_references)
        .into_par_iter()
        .map(|i| {
            let start = bounds[i];
            let end = bounds[i + 1];
            let reference = &all_coords[start..end];
            jaccard_distance_core(query, reference)
        })
        .collect()
}

/// Row-wise parallel processing with simple progress
pub fn jaccard_distance_matrix_rowwise(
    all_coords: &[CoordType],
    bounds: &[BoundType],
) -> Vec<Vec<ScoreType>> {
    let n = bounds.len() - 1;
    
    info!("Computing {}x{} distance matrix (rowwise method)...", n, n);
    let start_time = std::time::Instant::now();
    
    // Process rows in chunks to show progress
    let chunk_size = 50;
    let mut result = vec![vec![0.0; n]; n];
    
    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, n);
        
        let chunk_results: Vec<(usize, Vec<ScoreType>)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let begin_i = bounds[i];
                let end_i = bounds[i + 1];
                let coords_i = &all_coords[begin_i..end_i];
                
                let mut row = vec![0.0; n];
                
                for j in 0..n {
                    if i != j {
                        let begin_j = bounds[j];
                        let end_j = bounds[j + 1];
                        let coords_j = &all_coords[begin_j..end_j];
                        row[j] = jaccard_distance_core(coords_i, coords_j);
                    }
                }
                
                (i, row)
            })
            .collect();
        
        for (i, row) in chunk_results {
            result[i] = row;
        }
        
        let elapsed = start_time.elapsed();
        let progress = (chunk_end as f64 / n as f64) * 100.0;
        let eta_seconds = if chunk_end > 0 {
            (elapsed.as_secs_f64() / chunk_end as f64) * (n - chunk_end) as f64
        } else {
            0.0
        };
        
        print!(
            "\rProgress: {}/{} ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
            chunk_end, n, progress, elapsed, eta_seconds
        );
        stdout().flush().unwrap();
    }
    
    
    info!("Matrix computation completed in {:?}", start_time.elapsed());
    result
}

/// Row-wise parallel processing with streaming output
pub fn jaccard_distance_matrix_rowwise_stream(
    all_coords: &[CoordType],
    bounds: &[BoundType],
    writer: &mut csv::Writer<impl Write>,
    ids: &[String],
) -> Result<()> {
    let n = bounds.len() - 1;

    info!(
        "Computing and streaming {}x{} distance matrix (rowwise method)...",
        n, n
    );
    let start_time = std::time::Instant::now();

    // Write header
    let mut header = vec!["".to_string()];
    header.extend(ids.iter().cloned());
    writer.write_record(&header)?;

    // Process rows in chunks to show progress
    let chunk_size = 50;

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, n);

        let mut chunk_results: Vec<(usize, Vec<ScoreType>)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let begin_i = bounds[i];
                let end_i = bounds[i + 1];
                let coords_i = &all_coords[begin_i..end_i];

                let mut row = vec![0.0; n];

                for j in 0..n {
                    if i != j {
                        let begin_j = bounds[j];
                        let end_j = bounds[j + 1];
                        let coords_j = &all_coords[begin_j..end_j];
                        row[j] = jaccard_distance_core(coords_i, coords_j);
                    }
                }

                (i, row)
            })
            .collect();

        // Sort results by row index to ensure correct order in CSV
        chunk_results.sort_unstable_by_key(|k| k.0);

        for (i, row) in chunk_results {
            let mut csv_row = vec![ids[i].clone()];
            csv_row.extend(row.iter().map(|&x| format!("{:.6}", x)));
            writer.write_record(&csv_row)?;
        }

        let elapsed = start_time.elapsed();
        let progress = (chunk_end as f64 / n as f64) * 100.0;
        let eta_seconds = if chunk_end > 0 {
            (elapsed.as_secs_f64() / chunk_end as f64) * (n - chunk_end) as f64
        } else {
            0.0
        };

        print!(
            "\rProgress: {}/{} ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
            chunk_end, n, progress, elapsed, eta_seconds
        );
        stdout().flush().unwrap();
    }

    writer.flush()?;
    
    info!("Matrix streaming completed in {:?}", start_time.elapsed());
    Ok(())
}

/// Row-wise parallel processing with streaming HDF5 output
/// This follows same pattern as jaccard_distance_matrix_rowwise_stream for writing to csv
pub fn jaccard_distance_matrix_rowwise_stream_hdf5(
    all_coords: &[CoordType],
    bounds: &[BoundType],
    hdf5_writer: &mut crate::matrix_io::Hdf5StreamWriter,
) -> Result<()> {
    let n = bounds.len() - 1;

    info!(
        "Computing and streaming {}x{} distance matrix to HDF5 (rowwise method)...",
        n, n
    );
    let start_time = std::time::Instant::now();

    // Process rows in chunks to show progress
    let chunk_size = 50;

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, n);

        let mut chunk_results: Vec<(usize, Vec<ScoreType>)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let begin_i = bounds[i];
                let end_i = bounds[i + 1];
                let coords_i = &all_coords[begin_i..end_i];

                let mut row = vec![0.0; n];

                for j in 0..n {
                    if i != j {
                        let begin_j = bounds[j];
                        let end_j = bounds[j + 1];
                        let coords_j = &all_coords[begin_j..end_j];
                        row[j] = jaccard_distance_core(coords_i, coords_j);
                    }
                }

                (i, row)
            })
            .collect();

        // Sort results by row index to ensure correct order in HDF5
        chunk_results.sort_unstable_by_key(|k| k.0);

        for (_i, row) in chunk_results {
            hdf5_writer.write_row(&row)?;
        }

        let elapsed = start_time.elapsed();
        let progress = (chunk_end as f64 / n as f64) * 100.0;
        let eta_seconds = if chunk_end > 0 {
            (elapsed.as_secs_f64() / chunk_end as f64) * (n - chunk_end) as f64
        } else {
            0.0
        };

        print!(
            "\rProgress: {}/{} ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
            chunk_end, n, progress, elapsed, eta_seconds
        );
        stdout().flush().unwrap();
    }

    
    info!("HDF5 matrix streaming completed in {:?}", start_time.elapsed());
    Ok(())
}

/// Upper triangle computation with simple progress
pub fn jaccard_distance_matrix_upper_triangle(
    all_coords: &[CoordType],
    bounds: &[BoundType],
) -> Vec<Vec<ScoreType>> {
    let n = bounds.len() - 1;
    let mut distance_matrix = vec![vec![0.0; n]; n];
    
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    
    let total_pairs = pairs.len();
    info!("Computing {} pairs for {}x{} matrix...", total_pairs, n, n);
    
    let start_time = std::time::Instant::now();
    let chunk_size = 10000;
    
    for (chunk_idx, chunk) in pairs.chunks(chunk_size).enumerate() {
        let results: Vec<(usize, usize, ScoreType)> = chunk
            .into_par_iter()
            .map(|&(i, j)| {
                let begin_i = bounds[i];
                let end_i = bounds[i + 1];
                let coords_i = &all_coords[begin_i..end_i];
                
                let begin_j = bounds[j];
                let end_j = bounds[j + 1];
                let coords_j = &all_coords[begin_j..end_j];
                
                let distance = jaccard_distance_core(coords_i, coords_j);
                (i, j, distance)
            })
            .collect();
        
        for (i, j, distance) in results {
            distance_matrix[i][j] = distance;
            distance_matrix[j][i] = distance;
        }
        
        let pairs_completed = std::cmp::min((chunk_idx + 1) * chunk_size, total_pairs);
        let elapsed = start_time.elapsed();
        let progress = (pairs_completed as f64 / total_pairs as f64) * 100.0;
        
        print!(
            "\rProgress: {}/{} pairs ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
            pairs_completed, total_pairs, progress, elapsed, elapsed.as_secs_f64() / pairs_completed as f64 * (total_pairs - pairs_completed) as f64
        );
        stdout().flush().unwrap();
    }
    
    println!(); // Move to the next line after the loop
    println!("Matrix computation completed in {:?}", start_time.elapsed());
    distance_matrix
}

/// Cache-blocked computation
pub fn jaccard_distance_matrix_blocked(
    all_coords: &[CoordType],
    bounds: &[BoundType],
    block_size: usize,
) -> Vec<Vec<ScoreType>> {
    let n = bounds.len() - 1;
    let mut distance_matrix = vec![vec![0.0; n]; n];

    let num_blocks = (n + block_size - 1) / block_size;
    println!(
        "Computing on {}x{} matrix using {} block rows...",
        n, n, num_blocks
    );
    let start_time = std::time::Instant::now();

    for bi in 0..num_blocks {
        let block_i_start = bi * block_size;
        let block_i_end = std::cmp::min(block_i_start + block_size, n);

        for bj in bi..num_blocks {
            let block_j_start = bj * block_size;
            let block_j_end = std::cmp::min(block_j_start + block_size, n);

            let pairs: Vec<(usize, usize)> = (block_i_start..block_i_end)
                .flat_map(|i| {
                    let start_j = std::cmp::max(block_j_start, i + 1);
                    (start_j..block_j_end).map(move |j| (i, j))
                })
                .collect();

            let results: Vec<(usize, usize, ScoreType)> = pairs
                .into_par_iter()
                .map(|(i, j)| {
                    let begin_i = bounds[i];
                    let end_i = bounds[i + 1];
                    let coords_i = &all_coords[begin_i..end_i];

                    let begin_j = bounds[j];
                    let end_j = bounds[j + 1];
                    let coords_j = &all_coords[begin_j..end_j];

                    let distance = jaccard_distance_core(coords_i, coords_j);
                    (i, j, distance)
                })
                .collect();

            for (i, j, distance) in results {
                distance_matrix[i][j] = distance;
                distance_matrix[j][i] = distance;
            }
        }

        let blocks_processed = bi + 1;
        let elapsed = start_time.elapsed();
        let progress = (blocks_processed as f64 / num_blocks as f64) * 100.0;
        let eta_seconds = if blocks_processed > 0 {
            (elapsed.as_secs_f64() / blocks_processed as f64)
                * (num_blocks - blocks_processed) as f64
        } else {
            0.0
        };

        print!(
            "\rProgress: block row {}/{} ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
            blocks_processed, num_blocks, progress, elapsed, eta_seconds
        );
        stdout().flush().unwrap();
    }

    println!();
    println!("Matrix computation completed in {:?}", start_time.elapsed());
    distance_matrix
}

/// Calculate full distance matrix (all vs all)
pub fn jaccard_distance_matrix_all_vs_all(coords_vec: &[Vec<CoordType>]) -> Vec<Vec<ScoreType>> {
    let n = coords_vec.len();
    let mut matrix = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        matrix[i][i] = 0.0;
        for j in (i + 1)..n {
            let distance = jaccard_distance_core(&coords_vec[i], &coords_vec[j]);
            matrix[i][j] = distance;
            matrix[j][i] = distance;
        }
    }
    
    matrix
}

/// Calculate query vs reference distance matrix (rows = queries, cols = references)
pub fn jaccard_distance_matrix_query_vs_ref(
    query_coords: &[CoordType],
    query_bounds: &[BoundType],
    ref_coords: &[CoordType], 
    ref_bounds: &[BoundType],
) -> Result<Vec<Vec<ScoreType>>> {
    let n_queries = query_bounds.len() - 1;
    let n_refs = ref_bounds.len() - 1;
    
    println!("Computing {}x{} query vs reference distance matrix...", n_queries, n_refs);
    let start_time = std::time::Instant::now();
    
    // Process queries in chunks to show progress
    let chunk_size = 50;
    let mut result = vec![vec![0.0; n_refs]; n_queries];
    
    for chunk_start in (0..n_queries).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, n_queries);
        
        let chunk_results: Vec<(usize, Vec<ScoreType>)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let begin_i = query_bounds[i];
                let end_i = query_bounds[i + 1];
                let query_coords_i = &query_coords[begin_i..end_i];
                
                let mut row = vec![0.0; n_refs];
                
                for j in 0..n_refs {
                    let begin_j = ref_bounds[j];
                    let end_j = ref_bounds[j + 1];
                    let ref_coords_j = &ref_coords[begin_j..end_j];
                    row[j] = jaccard_distance_core(query_coords_i, ref_coords_j);
                }
                
                (i, row)
            })
            .collect();
        
        for (i, row) in chunk_results {
            result[i] = row;
        }
        
        let elapsed = start_time.elapsed();
        let progress = (chunk_end as f64 / n_queries as f64) * 100.0;
        let eta_seconds = if chunk_end > 0 {
            (elapsed.as_secs_f64() / chunk_end as f64) * (n_queries - chunk_end) as f64
        } else {
            0.0
        };
        
        print!(
            "\rProgress: {}/{} ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
            chunk_end, n_queries, progress, elapsed, eta_seconds
        );
        stdout().flush().unwrap();
    }
    
    println!(); // Move to the next line after the loop
    println!("Query vs reference matrix computation completed in {:?}", start_time.elapsed());
    Ok(result)
}

/// Calculate query vs reference distance matrix using blocked method for cache efficiency
pub fn jaccard_distance_matrix_query_vs_ref_blocked(
    query_coords: &[CoordType],
    query_bounds: &[BoundType],
    ref_coords: &[CoordType], 
    ref_bounds: &[BoundType],
    block_size: usize,
) -> Result<Vec<Vec<ScoreType>>> {
    let n_queries = query_bounds.len() - 1;
    let n_refs = ref_bounds.len() - 1;
    
    println!("Computing {}x{} query vs reference distance matrix using blocked method (block size: {})...", n_queries, n_refs, block_size);
    let start_time = std::time::Instant::now();
    
    let mut result = vec![vec![0.0; n_refs]; n_queries];
    
    let num_query_blocks = (n_queries + block_size - 1) / block_size;
    let num_ref_blocks = (n_refs + block_size - 1) / block_size;
    let total_blocks = num_query_blocks * num_ref_blocks;
    let mut blocks_processed = 0;
    
    for qi in 0..num_query_blocks {
        let query_block_start = qi * block_size;
        let query_block_end = std::cmp::min(query_block_start + block_size, n_queries);
        
        for ri in 0..num_ref_blocks {
            let ref_block_start = ri * block_size;
            let ref_block_end = std::cmp::min(ref_block_start + block_size, n_refs);
            
            // Generate all pairs in this block
            let pairs: Vec<(usize, usize)> = (query_block_start..query_block_end)
                .flat_map(|i| (ref_block_start..ref_block_end).map(move |j| (i, j)))
                .collect();
            
            let block_results: Vec<(usize, usize, ScoreType)> = pairs
                .into_par_iter()
                .map(|(i, j)| {
                    let begin_i = query_bounds[i];
                    let end_i = query_bounds[i + 1];
                    let query_coords_i = &query_coords[begin_i..end_i];
                    
                    let begin_j = ref_bounds[j];
                    let end_j = ref_bounds[j + 1];
                    let ref_coords_j = &ref_coords[begin_j..end_j];
                    
                    let distance = jaccard_distance_core(query_coords_i, ref_coords_j);
                    (i, j, distance)
                })
                .collect();
            
            for (i, j, distance) in block_results {
                result[i][j] = distance;
            }
            
            blocks_processed += 1;
            let elapsed = start_time.elapsed();
            let progress = (blocks_processed as f64 / total_blocks as f64) * 100.0;
            let eta_seconds = if blocks_processed > 0 {
                (elapsed.as_secs_f64() / blocks_processed as f64) * (total_blocks - blocks_processed) as f64
            } else {
                0.0
            };
            
            print!(
                "\rProgress: block {}/{} ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
                blocks_processed, total_blocks, progress, elapsed, eta_seconds
            );
            stdout().flush().unwrap();
        }
    }
    
    println!(); // Move to the next line after the loop
    println!("Blocked query vs reference matrix computation completed in {:?}", start_time.elapsed());
    Ok(result)
}

/// Calculate query vs reference distance matrix with streaming output to avoid memory issues
pub fn jaccard_distance_matrix_query_vs_ref_stream(
    query_coords: &[CoordType],
    query_bounds: &[BoundType],
    ref_coords: &[CoordType], 
    ref_bounds: &[BoundType],
    writer: &mut csv::Writer<impl Write>,
    query_ids: &[String],
    ref_ids: &[String],
) -> Result<()> {
    let n_queries = query_bounds.len() - 1;
    let n_refs = ref_bounds.len() - 1;
    
    println!("Computing and streaming {}x{} query vs reference distance matrix...", n_queries, n_refs);
    let start_time = std::time::Instant::now();
    
    // Write header
    let mut header = vec!["".to_string()];
    header.extend(ref_ids.iter().cloned());
    writer.write_record(&header)?;
    
    // Process queries in chunks to show progress
    let chunk_size = 50;
    
    for chunk_start in (0..n_queries).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, n_queries);
        
        let mut chunk_results: Vec<(usize, Vec<ScoreType>)> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|i| {
                let begin_i = query_bounds[i];
                let end_i = query_bounds[i + 1];
                let query_coords_i = &query_coords[begin_i..end_i];
                
                let mut row = vec![0.0; n_refs];
                
                for j in 0..n_refs {
                    let begin_j = ref_bounds[j];
                    let end_j = ref_bounds[j + 1];
                    let ref_coords_j = &ref_coords[begin_j..end_j];
                    row[j] = jaccard_distance_core(query_coords_i, ref_coords_j);
                }
                
                (i, row)
            })
            .collect();
        
        // Sort results by query index to ensure correct order in CSV
        chunk_results.sort_unstable_by_key(|k| k.0);
        
        for (i, row) in chunk_results {
            let mut csv_row = vec![query_ids[i].clone()];
            csv_row.extend(row.iter().map(|&x| format!("{:.6}", x)));
            writer.write_record(&csv_row)?;
        }
        
        let elapsed = start_time.elapsed();
        let progress = (chunk_end as f64 / n_queries as f64) * 100.0;
        let eta_seconds = if chunk_end > 0 {
            (elapsed.as_secs_f64() / chunk_end as f64) * (n_queries - chunk_end) as f64
        } else {
            0.0
        };
        
        print!(
            "\rProgress: {}/{} ({:.1}%) - Elapsed: {:?} - ETA: {:.0}s ",
            chunk_end, n_queries, progress, elapsed, eta_seconds
        );
        stdout().flush().unwrap();
    }
    
    writer.flush()?;
    println!(); // Move to the next line after the loop
    println!("Streaming query vs reference matrix computation completed in {:?}", start_time.elapsed());
    Ok(())
}

/// MinHash implementation
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn compute_minhash(coords: &[CoordType], num_hashes: usize, hash_seeds: &[u64]) -> Vec<u32> {
    let mut signature = vec![u32::MAX; num_hashes];
    
    for &coord in coords {
        for (i, &seed) in hash_seeds.iter().enumerate().take(num_hashes) {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            coord.hash(&mut hasher);
            let hash_val = hasher.finish() as u32;
            
            if hash_val < signature[i] {
                signature[i] = hash_val;
            }
        }
    }
    
    signature
}

pub fn estimate_jaccard_similarity(sig1: &[u32], sig2: &[u32]) -> f32 {
    let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
    matches as f32 / sig1.len() as f32
}

pub fn precompute_similarity_candidates(
    all_coords: &[CoordType],
    bounds: &[BoundType],
    threshold: f32,
    num_hashes: usize,
) -> Vec<(usize, usize, f32)> {
    let n = bounds.len() - 1;
    
    let hash_seeds: Vec<u64> = (0..num_hashes).map(|i| i as u64 * 1000003).collect();
    
    let signatures: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let begin = bounds[i];
            let end = bounds[i + 1];
            let coords = &all_coords[begin..end];
            compute_minhash(coords, num_hashes, &hash_seeds)
        })
        .collect();
    
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    
    pairs
        .into_par_iter()
        .filter_map(|(i, j)| {
            let similarity = estimate_jaccard_similarity(&signatures[i], &signatures[j]);
            if similarity >= threshold {
                Some((i, j, 1.0 - similarity))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_distance() {
        let coords1 = vec![1, 2, 3, 4, 5];
        let coords2 = vec![3, 4, 5, 6, 7];
        let distance = jaccard_distance_core(&coords1, &coords2);
        
        // Expected: intersection = 3, union = 7, distance = 1 - 3/7 ≈ 0.571
        assert!((distance - 0.571).abs() < 0.01);
    }

    #[test]
    fn test_matrix_computation() {
        let coords = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let bounds = vec![0, 3, 6, 9]; // Three sets
        
        let matrix = jaccard_distance_matrix_upper_triangle(&coords, &bounds);
        
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        assert_eq!(matrix[0][0], 0.0); // Diagonal should be 0
        assert_eq!(matrix[1][1], 0.0);
        assert_eq!(matrix[2][2], 0.0);
        
        // Matrix should be symmetric
        assert_eq!(matrix[0][1], matrix[1][0]);
        assert_eq!(matrix[0][2], matrix[2][0]);
        assert_eq!(matrix[1][2], matrix[2][1]);
    }
}
