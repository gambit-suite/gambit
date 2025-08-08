use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use anyhow::{Result, Context};
use log::{info, debug, error, warn};
// use rayon::prelude::*;

mod jaccard;
mod signatures;
mod matrix_io;

use jaccard::*;
use signatures::*;
use matrix_io::{save_matrix_hdf5, detect_format_from_extension, Hdf5StreamWriter};

#[derive(Parser)]
#[command(name = "jaccard")]
#[command(about = "Ultra-fast Jaccard distance calculations for k-mer sets")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate distances between a query and reference sets
    Query {
        /// Query k-mer file (one integer per line)
        #[arg(short, long)]
        query: PathBuf,
        
        /// Reference k-mer file (concatenated sets)
        #[arg(short, long)]
        reference: PathBuf,
        
        /// Bounds file (set boundaries in reference file)
        #[arg(short, long)]
        bounds: PathBuf,
        
        /// Output file for distances
        #[arg(short, long)]
        output: PathBuf,
        
        /// Number of threads (default: all cores)
        #[arg(short, long)]
        threads: Option<usize>,
        
        /// Query index
        #[arg(long, default_value = "0")]
        query_idx: usize,
    },
    
    /// Calculate distances using GAMBIT signature files
    QuerySig {
        #[arg(long)]
        query_sig: PathBuf,
        #[arg(long)]
        ref_sig: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "rowwise")]
        method: String,
        #[arg(short, long)]
        threads: Option<usize>,
    },
    
    /// Calculate full distance matrix
    Matrix {
        /// K-mer coordinate file
        #[arg(short, long)]
        signatures: PathBuf,
        
        /// Output matrix file
        #[arg(short, long)]
        output: PathBuf,
        
        /// Symmetric matrix
        #[arg(long)]
        symmetric: bool,
        
        /// Number of threads (default: all cores)
        #[arg(short, long)]
        threads: Option<usize>,
    },
    
    /// Calculate matrix from GAMBIT signature file
    MatrixSig {
        #[arg(long)]
        signatures: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "upper")]
        method: String,
        /// Output format: csv, hdf5, binary (auto-detected from extension if not specified)
        #[arg(long)]
        format: Option<String>,
        /// Subset of signatures to calculate matrix for (specific indices)
        #[arg(long)]
        subset: Option<Vec<usize>>,
        /// Use only the first N signatures
        #[arg(long)]
        first_n: Option<usize>,
        #[arg(short, long)]
        threads: Option<usize>,
    },
    
    /// Inspect GAMBIT signature file
    InfoSig {
        #[arg(long)]
        signatures: PathBuf,
    },
    
    /// Find similar pairs using MinHash LSH
    Similar {
        /// K-mer coordinate file
        #[arg(short, long)]
        coords: PathBuf,
        
        /// Bounds file
        #[arg(short, long)]
        bounds: PathBuf,
        
        /// Output file for similar pairs
        #[arg(short, long)]
        output: PathBuf,
        
        /// Similarity threshold (default: 0.8)
        #[arg(long, default_value = "0.8")]
        threshold: f32,
        
        /// Number of hash functions (default: 128)
        #[arg(long, default_value = "128")]
        num_hashes: usize,
    },
    
    /// Convert from various input formats
    Convert {
        /// Input file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output coordinates file
        #[arg(long)]
        coords_out: PathBuf,
        
        /// Output bounds file
        #[arg(long)]
        bounds_out: PathBuf,
        
        /// Input format: csv, fasta, json
        #[arg(long, default_value = "csv")]
        format: String,
    },
    
    /// LSH-based fast similarity search
    Lsh {
        #[arg(short, long)]
        signatures: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "0.8")]
        threshold: f32,
        #[arg(long, default_value = "10")]
        num_hashes: usize,
        #[arg(short, long)]
        threads: Option<usize>,
    },
    
    /// Debug HDF5 file structure
    DebugH5 {
        #[arg(long)]
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Query { query, reference, bounds, output, threads, query_idx } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(*t).build_global()?;
            }
            
            info!("Loading data...");
            let query_coords = load_coords(&query)?;
            let ref_coords = load_coords(&reference)?;
            let ref_bounds = load_bounds(&bounds)?;
            
            info!("Computing distances for {} reference sets...", ref_bounds.len() - 1);
            let start = std::time::Instant::now();
            
            let distances = jaccard_distances_parallel(&query_coords, &ref_coords, &ref_bounds);
            
            let elapsed = start.elapsed();
            info!("Computed {} distances in {:.2?}", distances.len(), elapsed);
            
            save_distances(&distances, &output)?;
            info!("Results saved to {}", output.display());
        },
        
        Commands::QuerySig { query_sig, ref_sig, output, method, threads } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(*t).build_global()?;
            }
            
            info!("Loading signature files...");
            let query_sig_data = read_signatures(query_sig)?;
            debug!("Query signature data loaded");
            let ref_sig_data = read_signatures(ref_sig)?;
            debug!("Reference signature data loaded");
            
            info!("Preparing data for pairwise Jaccard calculation...");
            let (ref_coords, ref_bounds, ref_ids) = load_signatures_for_jaccard(&ref_sig_data)?;
            let (query_coords, query_bounds, query_ids) = load_signatures_for_jaccard(&query_sig_data)?;
            
            info!("Calculating pairwise Jaccard distances...");
            info!("Query signatures: {}, Reference signatures: {}", query_ids.len(), ref_ids.len());
            
            match method.as_str() {
                "rowwise" | "blocked" => {
                    let matrix = match method.as_str() {
                        "rowwise" => jaccard_distance_matrix_query_vs_ref(&query_coords, &query_bounds, &ref_coords, &ref_bounds)?,
                        "blocked" => jaccard_distance_matrix_query_vs_ref_blocked(&query_coords, &query_bounds, &ref_coords, &ref_bounds, 256)?,
                        _ => unreachable!(),
                    };
                    
                    info!("Writing matrix with query and reference IDs...");
                    save_query_ref_matrix_csv(&matrix, &query_ids, &ref_ids, output)?;
                    info!("Results saved to {}", output.display());
                }
                "rowwise-stream" => {
                    let file = File::create(output).context("Failed to create output file")?;
                    let mut writer = csv::Writer::from_writer(BufWriter::new(file));
                    jaccard_distance_matrix_query_vs_ref_stream(&query_coords, &query_bounds, &ref_coords, &ref_bounds, &mut writer, &query_ids, &ref_ids)?;
                    info!("Results saved to {}", output.display());
                }
                _ => anyhow::bail!("Unknown method: '{}'. Use 'rowwise', 'blocked', or 'rowwise-stream'", method),
            }
        },
        
        Commands::Matrix { signatures, output, symmetric, threads } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(*t).build_global()?;
            }
            
            println!("Reading signatures...");
            let sig_data = read_signatures(signatures)?;
            let (coords, bounds, ids) = load_signatures_for_jaccard(&sig_data)?;
            
            println!("Calculating distance matrix...");
            let matrix = if *symmetric {
                jaccard_distance_matrix_blocked(&coords, &bounds, 1000)
            } else {
                let coords_vec = signatures_to_coords(&sig_data);
                jaccard_distance_matrix_all_vs_all(&coords_vec)
            };
            
            println!("Writing matrix with IDs...");
            save_matrix_csv_with_ids(&matrix, &ids, output)?;
            
            println!("Done! Matrix written to {}", output.display());
        },
        
        Commands::MatrixSig { signatures, output, method, format, threads, first_n, .. } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(*t).build_global()?;
            }
            
            info!("Loading signature file...");
            let sig_data = read_signatures(signatures)?;
            
            // Determine which subset of data to use
            let (subset_coords, subset_bounds, subset_ids) = if let Some(count) = first_n {
                let n = std::cmp::min(*count, sig_data.kmers.len());
                info!("Using first {} signatures", n);

                let mut sub_coords = Vec::new();
                let mut sub_bounds = vec![0];
                let sub_ids: Vec<String> = sig_data.ids.iter().take(n).cloned().collect();

                for i in 0..n {
                    sub_coords.extend_from_slice(&sig_data.kmers[i]);
                    sub_bounds.push(sub_coords.len());
                }
                (sub_coords, sub_bounds, sub_ids)

            } else {
                // Use all signatures if no subset is specified
                info!("Using all {} signatures", sig_data.kmers.len());
                let (coords, bounds, ids) = load_signatures_for_jaccard(&sig_data)?;
                (coords, bounds, ids)
            };

            // Auto-detect format from file extension if not specified
            let output_format = format.clone().unwrap_or_else(|| detect_format_from_extension(output));

            let matrix_size = subset_bounds.len() - 1;
            info!("Computing {}x{} distance matrix using {} method...", matrix_size, matrix_size, method);
            info!("Output format: {}", output_format);
            
            match method.as_str() {
                "upper" | "rowwise" | "blocked" => {
                    let matrix = match method.as_str() {
                        "upper" => jaccard_distance_matrix_upper_triangle(&subset_coords, &subset_bounds),
                        "rowwise" => jaccard_distance_matrix_rowwise(&subset_coords, &subset_bounds),
                        "blocked" => jaccard_distance_matrix_blocked(&subset_coords, &subset_bounds, 256),
                        _ => unreachable!(),
                    };
                    
                    info!("Writing matrix with sample IDs...");
                    match output_format.as_str() {
                        "csv" => {
                            save_matrix_csv_with_ids(&matrix, &subset_ids, output)?;
                        },
                        "hdf5" => {
                            save_matrix_hdf5(&matrix, &subset_ids, output)?;
                        },
                        "binary" => {
                            save_matrix_binary_with_ids(&matrix, &subset_ids, output)?;
                        },
                        _ => anyhow::bail!("Unknown format: '{}'. Use 'csv', 'hdf5', or 'binary'", output_format),
                    }
                    info!("Matrix saved to {}", output.display());
                }
                "rowwise-stream" => {
                    match output_format.as_str() {
                        "csv" => {
                            let file = File::create(output).context("Failed to create output file")?;
                            let mut writer = csv::Writer::from_writer(BufWriter::new(file));
                            jaccard_distance_matrix_rowwise_stream(&subset_coords, &subset_bounds, &mut writer, &subset_ids)?;
                        },
                        "hdf5" => {
                            let n_cols = subset_bounds.len() - 1;
                            let mut hdf5_writer = Hdf5StreamWriter::new(output, n_cols, &subset_ids)?;
                            jaccard_distance_matrix_rowwise_stream_hdf5(&subset_coords, &subset_bounds, &mut hdf5_writer)?;
                        },
                        _ => anyhow::bail!("Streaming supports 'csv' and 'hdf5' formats"),
                    }
                    info!("Matrix saved to {}", output.display());
                }
                _ => anyhow::bail!("Unknown method: '{}'. Use 'upper', 'rowwise', 'blocked', or 'rowwise-stream'", method),
            };
        },
        
        Commands::InfoSig { signatures } => {
            let sigs = GambitSignatures::load_from_file(&signatures)?;
            sigs.print_info();
        },
        
        Commands::Similar { coords, bounds, output, threshold, num_hashes } => {
            println!("Loading data...");
            let all_coords = load_coords(&coords)?;
            let bounds_vec = load_bounds(&bounds)?;
            
            println!("Finding similar pairs with threshold {}...", threshold);
            let start = std::time::Instant::now();
            
            let candidates = precompute_similarity_candidates(&all_coords, &bounds_vec, *threshold, *num_hashes);
            
            let elapsed = start.elapsed();
            println!("Found {} similar pairs in {:.2?}", candidates.len(), elapsed);
            
            save_similar_pairs(&candidates, &output)?;
            println!("Similar pairs saved to {}", output.display());
        },
        
        Commands::Convert { input, coords_out, bounds_out, format } => {
            match format.as_str() {
                "csv" => convert_from_csv(&input, &coords_out, &bounds_out)?,
                "fasta" => convert_from_fasta(&input, &coords_out, &bounds_out)?,
                "json" => convert_from_json(&input, &coords_out, &bounds_out)?,
                _ => anyhow::bail!("Unknown input format: {}", format),
            }
            println!("Converted {} to coordinates and bounds files", input.display());
        },
        
        Commands::Lsh { signatures, output, threshold, num_hashes, threads } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new().num_threads(*t).build_global()?;
            }
            
            println!("Reading signatures...");
            let sig_data = read_signatures(signatures)?;
            let (all_coords, bounds_vec, _ids) = load_signatures_for_jaccard(&sig_data)?;
            
            println!("Computing LSH candidates...");
            let candidates = precompute_similarity_candidates(&all_coords, &bounds_vec, *threshold, *num_hashes);
            
            println!("Writing LSH results...");
            write_lsh_results(output, &candidates)?;
            
            println!("Done! LSH results written to {}", output.display());
        },
        
        Commands::DebugH5 { file } => {
            debug_hdf5_ids(file)?;
        },
    }
    
    Ok(())
}

// File I/O functions
fn load_coords(path: &PathBuf) -> Result<Vec<u32>> {
    let file = File::open(path).context("Failed to open coordinates file")?;
    let reader = BufReader::new(file);
    
    let mut coords = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            coords.push(line.trim().parse()?);
        }
    }
    
    Ok(coords)
}

fn load_bounds(path: &PathBuf) -> Result<Vec<usize>> {
    let file = File::open(path).context("Failed to open bounds file")?;
    let reader = BufReader::new(file);
    
    let mut bounds = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            bounds.push(line.trim().parse()?);
        }
    }
    
    Ok(bounds)
}

fn save_distances(distances: &[f32], path: &PathBuf) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    
    for distance in distances {
        writeln!(writer, "{:.4}", distance)?;
    }
    
    Ok(())
}


fn save_similar_pairs(pairs: &[(usize, usize, f32)], path: &PathBuf) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(file);
    
    writer.write_record(&["i", "j", "distance"])?;
    for (i, j, dist) in pairs {
        writer.write_record(&[i.to_string(), j.to_string(), format!("{:.4}", dist)])?;
    }
    
    writer.flush()?;
    Ok(())
}

// Format conversion functions
fn convert_from_csv(input: &PathBuf, coords_out: &PathBuf, bounds_out: &PathBuf) -> Result<()> {
    let file = File::open(input)?;
    let mut reader = csv::Reader::from_reader(file);
    
    let coords_file = File::create(coords_out)?;
    let mut coords_writer = BufWriter::new(coords_file);
    
    let bounds_file = File::create(bounds_out)?;
    let mut bounds_writer = BufWriter::new(bounds_file);
    
    let mut current_pos = 0;
    writeln!(bounds_writer, "{}", current_pos)?; // Start with 0
    
    for result in reader.records() {
        let record = result?;
        for field in record.iter() {
            if !field.trim().is_empty() {
                writeln!(coords_writer, "{}", field.trim())?;
                current_pos += 1;
            }
        }
        writeln!(bounds_writer, "{}", current_pos)?;
    }
    
    Ok(())
}

fn convert_from_fasta(_input: &PathBuf, _coords_out: &PathBuf, _bounds_out: &PathBuf) -> Result<()> {
    // Implement FASTA to k-mer conversion
    todo!("FASTA conversion not implemented yet")
}

fn convert_from_json(_input: &PathBuf, _coords_out: &PathBuf, _bounds_out: &PathBuf) -> Result<()> {
    // Implement JSON to k-mer conversion
    todo!("JSON conversion not implemented yet")
}

fn write_lsh_results(output: &PathBuf, candidates: &[(usize, usize, f32)]) -> Result<()> {
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "i,j,similarity")?;
    for &(i, j, similarity) in candidates {
        writeln!(writer, "{},{},{:.4}", i, j, similarity)?;
    }
    
    Ok(())
}

fn save_matrix_csv_with_ids(matrix: &[Vec<f32>], ids: &[String], path: &PathBuf) -> Result<()> {
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


fn save_matrix_binary_with_ids(matrix: &[Vec<f32>], ids: &[String], path: &PathBuf) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    
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

fn save_query_ref_matrix_csv(matrix: &[Vec<f32>], query_ids: &[String], ref_ids: &[String], path: &PathBuf) -> Result<()> {
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
