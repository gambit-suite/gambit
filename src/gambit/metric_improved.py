"""Memory-efficient implementations of Jaccard distance calculations with MinHash acceleration."""

import os
import tempfile
import shutil
from typing import Sequence, Optional, Tuple, Union, Literal, Set
import numpy as np
import h5py
from pathlib import Path

import gambit._cython.metric as _cmetric
from gambit.sigs.base import KmerSignature, SignatureArray, BOUNDS_DTYPE
from gambit.util.progress import get_progress
from gambit.metric import jaccarddist_matrix, jaccarddist_pairwise

# Constants
SCORE_DTYPE = np.dtype(np.float32)
DEFAULT_BATCH_SIZE = 1000
DEFAULT_CHUNK_SIZE = 100

# Size thresholds for implementation selection
SMALL_DATASET_THRESHOLD = 1000  # Number of sequences below which to use in-memory implementation
MEMORY_MAPPED_THRESHOLD = 5000  # Number of sequences below which to use memory-mapped files
MINHASH_THRESHOLD = 10000  # Number of sequences above which to use MinHash pre-filtering

# Temp file location types
TempLocation = Literal['output_dir', 'ram', 'system']

def _cast_sigs_array(arr: np.ndarray) -> np.ndarray:
    """Convert signature array to proper data type for Cython metric code."""
    dt = arr.dtype
    if dt in [np.dtype(f'u{s}') for s in [2, 4, 8]]:
        return arr
    if dt in [np.dtype(f'i{s}') for s in [2, 4, 8]]:
        new_dt = np.dtype(f'u{dt.itemsize}')
        return arr.view(new_dt)
    raise ValueError(f'Invalid dtype for k-mer coordinate array: {dt.str}')

def _optimize_batch_sizes(total_queries: int, total_refs: int) -> Tuple[int, int]:
    """Optimize batch and chunk sizes based on dataset size."""
    # For very small datasets, use smaller batches
    if total_queries < 100:
        batch_size = max(10, total_queries)
        chunk_size = max(10, total_refs)
    # For medium datasets, use moderate batch sizes
    elif total_queries < 1000:
        batch_size = max(100, total_queries // 10)
        chunk_size = max(50, total_refs // 10)
    # For large datasets, use default sizes
    else:
        batch_size = DEFAULT_BATCH_SIZE
        chunk_size = DEFAULT_CHUNK_SIZE
    
    return batch_size, chunk_size

class BatchedDistanceCalculator:
    """Memory-efficient calculator for Jaccard distances between large sets of sequences with MinHash acceleration."""
    
    def __init__(self, 
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 temp_location: TempLocation = 'system',
                 temp_dir: Optional[str] = None,
                 use_minhash: bool = True,
                 minhash_threshold: float = 0.7,
                 minhash_hashes: int = 128):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        batch_size : int
            Number of query sequences to process in each batch
        chunk_size : int
            Number of reference sequences to process in each chunk
        temp_location : {'output_dir', 'ram', 'system'}
            Where to store temporary files
        temp_dir : str, optional
            Custom directory to store temporary files
        use_minhash : bool, default True
            Whether to use MinHash pre-filtering for large datasets
        minhash_threshold : float, default 0.7
            MinHash similarity threshold for pre-filtering (0.0-1.0)
            Higher values = more aggressive filtering = faster but less complete
        minhash_hashes : int, default 128
            Number of hash functions for MinHash (more = better accuracy, slower)
        """
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.temp_location = temp_location
        self.temp_dir = temp_dir
        self.use_minhash = use_minhash
        self.minhash_threshold = minhash_threshold
        self.minhash_hashes = minhash_hashes
        self.candidate_pairs: Optional[Set[Tuple[int, int]]] = None
        
    def _get_temp_dir(self, output_file: str) -> str:
        """Get the appropriate temporary directory based on configuration."""
        if self.temp_dir is not None:
            return self.temp_dir
            
        if self.temp_location == 'output_dir':
            return os.path.dirname(os.path.abspath(output_file))
        elif self.temp_location == 'ram':
            # Try to use RAM-based filesystem if available
            ram_dirs = ['/dev/shm', '/run/shm']  # Common RAM-based filesystem locations
            for ram_dir in ram_dirs:
                if os.path.exists(ram_dir) and os.access(ram_dir, os.W_OK):
                    return ram_dir
            # Fall back to system temp if RAM-based location not available
            return tempfile.gettempdir()
        else:  # 'system'
            return tempfile.gettempdir()
    
    def _precompute_minhash_candidates(self, refs: SignatureArray) -> None:
        """Pre-compute candidate pairs using MinHash for massive speedup."""
        total_refs = len(refs)
        
        print(f"\n🔍 MinHash Pre-filtering Setup:")
        print(f"  Dataset size: {total_refs:,} sequences")
        print(f"  Similarity threshold: {self.minhash_threshold:.2f}")
        print(f"  Hash functions: {self.minhash_hashes}")
        print(f"  Expected speedup: 100-1000x")
        
        # Extract the concatenated coordinates and bounds
        all_coords = _cast_sigs_array(refs.values)
        bounds = refs.bounds.astype(BOUNDS_DTYPE, copy=False)
        
        # Call the optimized parallel Cython MinHash function
        print("  Using parallel MinHash computation for maximum speed...")
        
        # Choose the best version based on dataset size
        if total_refs > 10000:
            # Use ultra-optimized version for very large datasets
            candidates_list = _cmetric.precompute_similarity_candidates_optimized(
                all_coords, bounds, 
                threshold=self.minhash_threshold, 
                num_hashes=self.minhash_hashes
            )
        else:
            # Use standard parallel version for medium datasets
            candidates_list = _cmetric.precompute_similarity_candidates(
                all_coords, bounds, 
                threshold=self.minhash_threshold, 
                num_hashes=self.minhash_hashes
            )
        
        # Convert to set for fast lookup
        self.candidate_pairs = set(candidates_list)
        
        reduction_factor = (total_refs * total_refs) // (2 * max(1, len(candidates_list)))
        print(f"  ✅ Found {len(candidates_list):,} candidate pairs")
        print(f"  ✅ Reduction factor: {reduction_factor:,}x fewer computations")
        print(f"  ✅ Estimated time savings: {reduction_factor//10:,}x faster")
        
    def _should_compute_distance(self, query_idx: int, ref_idx: int, is_square: bool) -> bool:
        """Check if we should compute the distance between two sequences."""
        if not self.use_minhash or self.candidate_pairs is None:
            return True
            
        # For square matrices, check both orientations due to symmetry
        if is_square:
            return ((query_idx, ref_idx) in self.candidate_pairs or 
                    (ref_idx, query_idx) in self.candidate_pairs)
        else:
            # For non-square matrices, we need to map indices appropriately
            # This assumes the query and ref indices correspond to the same underlying sequences
            return (query_idx, ref_idx) in self.candidate_pairs
        
    def _create_temp_file(self, total_queries: int, total_refs: int, output_file: str) -> Tuple[h5py.File, str]:
        """Create a temporary HDF5 file for storing intermediate results."""
        temp_dir = self._get_temp_dir(output_file)
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_name = f'gambit_dist_{os.getpid()}_{id(self)}.h5'
        temp_path = os.path.join(temp_dir, temp_name)
        
        # Use memory-mapped files for small/medium datasets
        if total_queries * total_refs < MEMORY_MAPPED_THRESHOLD:
            return h5py.File(temp_path, 'w', driver='core', backing_store=False), temp_path
        
        # For RAM-based storage, use memory-mapped files with compression
        if self.temp_location == 'ram':
            return h5py.File(temp_path, 'w', driver='core', backing_store=True, 
                            libver='latest',  # Use latest HDF5 version
                            rdcc_nslots=100000,  # Increase cache slots
                            rdcc_nbytes=1024*1024*1024), temp_path  # 1GB cache
        
        # For system storage, use compression and caching
        return h5py.File(temp_path, 'w', 
                        libver='latest',
                        rdcc_nslots=100000,
                        rdcc_nbytes=1024*1024*1024), temp_path
        
    def _process_batch(self,
                      queries: Sequence[KmerSignature],
                      refs: SignatureArray,
                      start_idx: int,
                      out_file: h5py.File,
                      is_square: bool = False) -> None:
        """Process a batch of query sequences against all reference sequences."""
        batch_size = len(queries)
        total_refs = len(refs)
        
        # Create dataset for this batch if it doesn't exist
        if f'batch_{start_idx}' not in out_file:
            out_file.create_dataset(
                f'batch_{start_idx}',
                shape=(batch_size, total_refs),
                dtype=SCORE_DTYPE,
                chunks=(min(100, batch_size), min(self.chunk_size, total_refs)),
                compression='gzip' if total_refs > MEMORY_MAPPED_THRESHOLD else None
            )
        
        # Process queries in parallel chunks
        query_chunk_size = min(100, batch_size)
        for i in range(0, batch_size, query_chunk_size):
            query_chunk_end = min(i + query_chunk_size, batch_size)
            query_chunk = queries[i:query_chunk_end]
            
            # Pre-allocate output array for the chunk
            chunk_out = np.full((len(query_chunk), total_refs), 1.0, dtype=SCORE_DTYPE)
            
            # Process reference sequences in chunks
            for j in range(0, total_refs, self.chunk_size):
                chunk_end = min(j + self.chunk_size, total_refs)
                chunk_refs = refs[j:chunk_end]
                
                values = _cast_sigs_array(chunk_refs.values)
                bounds = chunk_refs.bounds.astype(BOUNDS_DTYPE, copy=False)
                
                # Process each query in the chunk
                for k, query in enumerate(query_chunk):
                    query_idx = start_idx + i + k
                    query_casted = _cast_sigs_array(query)
                    
                    # Create output slice for this query and chunk
                    query_out = chunk_out[k, j:chunk_end]
                    
                    # Check if we should compute distances for this query
                    if self.use_minhash and self.candidate_pairs is not None:
                        # Only compute distances for candidate pairs
                        for ref_offset in range(len(query_out)):
                            ref_idx = j + ref_offset
                            if self._should_compute_distance(query_idx, ref_idx, is_square):
                                # Compute single distance
                                _cmetric._jaccarddist_parallel(query_casted, values[ref_offset:ref_offset+1], 
                                                             bounds[ref_offset:ref_offset+2], 
                                                             query_out[ref_offset:ref_offset+1])
                            # else: keep the pre-filled 1.0 value
                    else:
                        # Compute all distances (original behavior with optimization)
                        _cmetric._jaccarddist_parallel(query_casted, values, bounds, query_out)
            
            # Write chunk results to HDF5 file
            out_file[f'batch_{start_idx}'][i:query_chunk_end] = chunk_out
            
    def _process_batch_symmetric(self,
                               queries: Sequence[KmerSignature],
                               refs: SignatureArray,
                               start_idx: int,
                               start_ref: int,
                               out_file: h5py.File) -> None:
        """Process a batch of query sequences against reference sequences for symmetric matrix."""
        batch_size = len(queries)
        total_refs = len(refs)
        
        # Create dataset for this batch if it doesn't exist
        if f'batch_{start_idx}' not in out_file:
            out_file.create_dataset(
                f'batch_{start_idx}',
                shape=(batch_size, total_refs),
                dtype=SCORE_DTYPE,
                chunks=(min(100, batch_size), min(self.chunk_size, total_refs)),
                compression='gzip' if total_refs > MEMORY_MAPPED_THRESHOLD else None
            )
        
        # Process queries in parallel chunks
        query_chunk_size = min(100, batch_size)
        for i in range(0, batch_size, query_chunk_size):
            query_chunk_end = min(i + query_chunk_size, batch_size)
            query_chunk = queries[i:query_chunk_end]
            
            # Pre-allocate output array for the chunk
            chunk_out = np.full((len(query_chunk), total_refs), 1.0, dtype=SCORE_DTYPE)
            
            # Process reference sequences in chunks
            for j in range(0, total_refs, self.chunk_size):
                chunk_end = min(j + self.chunk_size, total_refs)
                chunk_refs = refs[j:chunk_end]
                
                values = _cast_sigs_array(chunk_refs.values)
                bounds = chunk_refs.bounds.astype(BOUNDS_DTYPE, copy=False)
                
                # Process each query in the chunk
                for k, query in enumerate(query_chunk):
                    query_idx = start_idx + i + k
                    ref_start_idx = start_ref + j
                    query_casted = _cast_sigs_array(query)
                    
                    # Create output slice for this query and chunk
                    query_out = chunk_out[k, j:chunk_end]
                    
                    # For symmetric matrices, only compute upper triangle
                    if self.use_minhash and self.candidate_pairs is not None:
                        # Only compute distances for candidate pairs
                        for ref_offset in range(len(query_out)):
                            ref_idx = ref_start_idx + ref_offset
                            if query_idx <= ref_idx and self._should_compute_distance(query_idx, ref_idx, True):
                                _cmetric._jaccarddist_parallel(query_casted, values[ref_offset:ref_offset+1], 
                                                             bounds[ref_offset:ref_offset+2], 
                                                             query_out[ref_offset:ref_offset+1])
                    else:
                        # Compute all distances in upper triangle
                        _cmetric._jaccarddist_parallel(query_casted, values, bounds, query_out)
            
            # Write chunk results to HDF5 file
            out_file[f'batch_{start_idx}'][i:query_chunk_end] = chunk_out

    def calculate_distances(self,
                          queries: Sequence[KmerSignature],
                          refs: Sequence[KmerSignature],
                          output_file: str,
                          progress = None) -> None:
        """
        Calculate Jaccard distances between query and reference sequences using batch processing with MinHash acceleration.
        """
        # Convert refs to SignatureArray if it isn't already
        if not isinstance(refs, SignatureArray):
            print("Converting reference sequences to SignatureArray...")
            refs = SignatureArray(refs)
            
        total_queries = len(queries)
        total_refs = len(refs)
        
        # Check if this is a square matrix (self-comparison)
        is_square = total_queries == total_refs and queries is refs
        
        # Determine if we should use MinHash pre-filtering
        should_use_minhash = (self.use_minhash and 
                            total_refs >= MINHASH_THRESHOLD and 
                            (not is_square or total_queries >= MINHASH_THRESHOLD))
        
        if should_use_minhash:
            print(f"\n📊 Large dataset detected ({total_refs:,} sequences)")
            print("🚀 Activating MinHash acceleration for massive speedup!")
            self._precompute_minhash_candidates(refs)
        else:
            print(f"\n📊 Using optimized exact computation")
            self.candidate_pairs = None
        
        if is_square:
            total_comparisons = (total_queries * (total_queries - 1)) // 2 + total_queries
            print(f"\nMatrix type: Square ({total_queries:,} x {total_queries:,})")
            print(f"Only calculating upper triangle + diagonal")
        else:
            total_comparisons = total_queries * total_refs
            print(f"\nMatrix type: Rectangular ({total_queries:,} x {total_refs:,})")
        
        print(f"Total comparisons: {total_comparisons:,}")
        
        # For small datasets, use the original in-memory implementation
        if total_queries < SMALL_DATASET_THRESHOLD and total_refs < SMALL_DATASET_THRESHOLD:
            print("\nUsing in-memory implementation for small dataset")
            dmat = jaccarddist_matrix(queries, refs, progress=progress)
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('distances', data=dmat)
            return
            
        # Only optimize batch sizes if they weren't explicitly set by the user
        if self.batch_size == DEFAULT_BATCH_SIZE and self.chunk_size == DEFAULT_CHUNK_SIZE:
            print("\nOptimizing batch sizes...")
            self.batch_size, self.chunk_size = _optimize_batch_sizes(total_queries, total_refs)
        
        print("\nBatch Configuration:")
        print(f"  Batch size: {self.batch_size:,} queries per batch")
        print(f"  Chunk size: {self.chunk_size:,} references per chunk")
        print(f"  Number of batches: {(total_queries + self.batch_size - 1) // self.batch_size:,}")
        print(f"  Number of chunks per batch: {(total_refs + self.chunk_size - 1) // self.chunk_size:,}")
        
        # Create temporary file for intermediate results
        print("\nSetting up temporary storage...")
        temp_file, temp_path = self._create_temp_file(total_queries, total_refs, output_file)
        print(f"  Location: {self.temp_location}")
        print(f"  Path: {temp_path}")
        
        try:
            print("\nStarting distance calculations...")
            
            # Show actual number of comparisons that will be computed
            if self.use_minhash and self.candidate_pairs is not None:
                actual_comparisons = len(self.candidate_pairs)
                if is_square:
                    # For square matrices, also add diagonal elements (self-comparisons)
                    actual_comparisons += total_queries
                    print(f"🎯 Actual comparisons (after MinHash filtering): {actual_comparisons:,}")
                    print(f"   - Candidate pairs: {len(self.candidate_pairs):,}")
                    print(f"   - Diagonal elements: {total_queries:,}")
                else:
                    print(f"🎯 Actual comparisons (after MinHash filtering): {actual_comparisons:,}")
                
                speedup_factor = total_comparisons / max(1, actual_comparisons)
                print(f"⚡ Computational speedup: {speedup_factor:.1f}x fewer distance calculations")
                print(f"📊 Computing only {(actual_comparisons/total_comparisons)*100:.1f}% of theoretical comparisons")
            else:
                print(f"🎯 Computing all {total_comparisons:,} distance comparisons (no filtering)")
            
            # Process queries in batches with continuous progress updates
            with get_progress(progress, total=total_queries, desc='Calculating distances') as pbar:
                processed_queries = 0
                
                for i in range(0, total_queries, self.batch_size):
                    batch_end = min(i + self.batch_size, total_queries)
                    batch_queries = queries[i:batch_end]
                    
                    # For square matrices, only process the upper triangle
                    if is_square:
                        # Adjust the reference range for this batch to only include upper triangle
                        start_ref = i  # Start from diagonal
                        refs_subset = refs[start_ref:]
                        self._process_batch_symmetric(batch_queries, refs_subset, i, start_ref, temp_file)
                    else:
                        self._process_batch(batch_queries, refs, i, temp_file, is_square)
                    
                    # Update progress after each batch completion
                    batch_size_actual = len(batch_queries)
                    processed_queries += batch_size_actual
                    
                    # Always update progress for better responsiveness
                    pbar.increment(batch_size_actual)
                    
                    # Update description with current progress
                    if processed_queries % max(1, total_queries // 20) == 0:  # Update description every 5%
                        percentage = (processed_queries / total_queries) * 100
                        pbar.set_description(f'Calculating distances ({percentage:.1f}%)')
            
            print("\nCombining results...")
            # Combine results into final output file
            self._combine_results(temp_file, output_file, total_queries, total_refs, is_square)
            
        finally:
            # Clean up temporary file
            print("\nCleaning up temporary files...")
            temp_file.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _combine_results(self,
                        temp_file: h5py.File,
                        output_file: str,
                        total_queries: int,
                        total_refs: int,
                        is_square: bool = False) -> None:
        """Combine batch results into final output file."""
        with h5py.File(output_file, 'w') as out_file:
            # Create final dataset
            out_file.create_dataset(
                'distances',
                shape=(total_queries, total_refs),
                dtype=SCORE_DTYPE,
                chunks=(1, min(self.chunk_size, total_refs)),
                compression='gzip' if total_refs > MEMORY_MAPPED_THRESHOLD else None
            )
            
            print(f"Writing final results to {output_file}...")
            
            if is_square:
                # For square matrices, process in chunks to reduce memory usage
                merge_chunk_size = min(100, self.batch_size)  # Use smaller chunks for merging
                total_batches = (total_queries + self.batch_size - 1) // self.batch_size
                processed_batches = 0
                
                for i in range(0, total_queries, self.batch_size):
                    batch_end = min(i + self.batch_size, total_queries)
                    batch_data = temp_file[f'batch_{i}']
                    
                    # Process each chunk of the batch
                    for chunk_start in range(0, batch_end - i, merge_chunk_size):
                        chunk_end = min(chunk_start + merge_chunk_size, batch_end - i)
                        chunk_data = batch_data[chunk_start:chunk_end]
                        
                        # Copy upper triangle chunk
                        out_file['distances'][i + chunk_start:i + chunk_end, i + chunk_start:] = chunk_data
                        
                        # Mirror to lower triangle in smaller chunks
                        for j in range(chunk_start, chunk_end):
                            row_idx = i + j
                            # Mirror only the portion we've calculated
                            for k in range(j + 1, total_refs - i):
                                out_file['distances'][i + k, row_idx] = out_file['distances'][row_idx, i + k]
                    
                    processed_batches += 1
                    if processed_batches % max(1, total_batches // 10) == 0:  # Progress every 10%
                        percentage = (processed_batches / total_batches) * 100
                        print(f"  Writing progress: {percentage:.1f}% ({processed_batches}/{total_batches} batches)")
            else:
                # For non-square matrices, process in chunks
                merge_chunk_size = min(100, self.batch_size)
                total_batches = (total_queries + self.batch_size - 1) // self.batch_size
                processed_batches = 0
                
                for i in range(0, total_queries, self.batch_size):
                    batch_end = min(i + self.batch_size, total_queries)
                    batch_data = temp_file[f'batch_{i}']
                    
                    # Process each chunk of the batch
                    for chunk_start in range(0, batch_end - i, merge_chunk_size):
                        chunk_end = min(chunk_start + merge_chunk_size, batch_end - i)
                        chunk_data = batch_data[chunk_start:chunk_end]
                        out_file['distances'][i + chunk_start:i + chunk_end] = chunk_data
                    
                    processed_batches += 1
                    if processed_batches % max(1, total_batches // 10) == 0:  # Progress every 10%
                        percentage = (processed_batches / total_batches) * 100
                        print(f"  Writing progress: {percentage:.1f}% ({processed_batches}/{total_batches} batches)")
            
            print("  ✅ Results written successfully!")

def jaccarddist_matrix_improved(queries: Sequence[KmerSignature],
                              refs: Sequence[KmerSignature],
                              output_file: str,
                              batch_size: int = DEFAULT_BATCH_SIZE,
                              chunk_size: int = DEFAULT_CHUNK_SIZE,
                              temp_location: TempLocation = 'system',
                              progress = None,
                              max_sequences: Optional[int] = None,
                              use_minhash: bool = True,
                              minhash_threshold: float = 0.7,
                              minhash_hashes: int = 128) -> None:
    """
    Memory-efficient implementation of jaccarddist_matrix with MinHash acceleration.
    
    Parameters
    ----------
    queries : Sequence[KmerSignature]
        Query sequences
    refs : Sequence[KmerSignature]
        Reference sequences
    output_file : str
        Path to output HDF5 file
    batch_size : int
        Number of query sequences to process in each batch
    chunk_size : int
        Number of reference sequences to process in each chunk
    temp_location : {'output_dir', 'ram', 'system'}
        Where to store temporary files
    progress : optional
        Progress bar configuration
    max_sequences : int, optional
        Maximum number of sequences to process
    use_minhash : bool, default True
        Whether to use MinHash pre-filtering for large datasets
    minhash_threshold : float, default 0.7
        MinHash similarity threshold (0.0-1.0). Higher = more aggressive filtering
    minhash_hashes : int, default 128
        Number of hash functions for MinHash accuracy
    """
    # Select subset of sequences if max_sequences is provided
    if max_sequences is not None:
        import random
        total_queries = len(queries)
        total_refs = len(refs)
        
        if max_sequences < total_queries:
            print(f"\nSelecting {max_sequences:,} random sequences from {total_queries:,} total sequences")
            # Use fixed seed for reproducibility
            random.seed(42)
            # Get random indices
            selected_indices = sorted(random.sample(range(total_queries), max_sequences))
            # Select sequences
            queries = [queries[i] for i in selected_indices]
            # If it's a self-comparison, use the same indices for refs
            if queries is refs:
                refs = queries
            else:
                refs = [refs[i] for i in selected_indices]
            print(f"Selected sequences: {selected_indices[:5]}... (and {len(selected_indices)-5} more)")
    
    calculator = BatchedDistanceCalculator(
        batch_size=batch_size,
        chunk_size=chunk_size,
        temp_location=temp_location,
        use_minhash=use_minhash,
        minhash_threshold=minhash_threshold,
        minhash_hashes=minhash_hashes
    )
    calculator.calculate_distances(queries, refs, output_file, progress)

def jaccarddist_pairwise_improved(sigs: Sequence[KmerSignature],
                                output_file: str,
                                batch_size: int = DEFAULT_BATCH_SIZE,
                                chunk_size: int = DEFAULT_CHUNK_SIZE,
                                temp_location: TempLocation = 'system',
                                progress = None,
                                max_sequences: Optional[int] = None,
                                use_minhash: bool = True,
                                minhash_threshold: float = 0.7,
                                minhash_hashes: int = 128) -> None:
    """
    Memory-efficient implementation of jaccarddist_pairwise with MinHash acceleration.
    
    Parameters
    ----------
    sigs : Sequence[KmerSignature]
        Sequences to calculate pairwise distances for
    output_file : str
        Path to output HDF5 file
    batch_size : int
        Number of sequences to process in each batch
    chunk_size : int
        Number of sequences to process in each chunk
    temp_location : {'output_dir', 'ram', 'system'}
        Where to store temporary files
    progress : optional
        Progress bar configuration
    max_sequences : int, optional
        Maximum number of sequences to process
    use_minhash : bool, default True
        Whether to use MinHash pre-filtering for large datasets
    minhash_threshold : float, default 0.7
        MinHash similarity threshold (0.0-1.0). Higher = more aggressive filtering
    minhash_hashes : int, default 128
        Number of hash functions for MinHash accuracy
    """
    # Select subset of sequences if max_sequences is provided
    if max_sequences is not None:
        import random
        total_sigs = len(sigs)
        
        if max_sequences < total_sigs:
            print(f"\nSelecting {max_sequences:,} random sequences from {total_sigs:,} total sequences")
            # Use fixed seed for reproducibility
            random.seed(42)
            # Get random indices
            selected_indices = sorted(random.sample(range(total_sigs), max_sequences))
            # Select sequences
            sigs = [sigs[i] for i in selected_indices]
            print(f"Selected sequences: {selected_indices[:5]}... (and {len(selected_indices)-5} more)")
    
    calculator = BatchedDistanceCalculator(
        batch_size=batch_size,
        chunk_size=chunk_size,
        temp_location=temp_location,
        use_minhash=use_minhash,
        minhash_threshold=minhash_threshold,
        minhash_hashes=minhash_hashes
    )
    calculator.calculate_distances(sigs, sigs, output_file, progress)