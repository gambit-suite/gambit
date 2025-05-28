"""Memory-efficient implementations of Jaccard distance calculations."""

import os
import tempfile
import shutil
from typing import Sequence, Optional, Tuple, Union, Literal
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
    """Memory-efficient calculator for Jaccard distances between large sets of sequences."""
    
    def __init__(self, 
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 temp_location: TempLocation = 'system',
                 temp_dir: Optional[str] = None):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        batch_size : int
            Number of query sequences to process in each batch
        chunk_size : int
            Number of reference sequences to process in each chunk
        temp_location : {'output_dir', 'ram', 'system'}
            Where to store temporary files:
            - 'output_dir': Store in same directory as output file
            - 'ram': Store in RAM-based filesystem (e.g. /dev/shm on Linux)
            - 'system': Use system's default temp directory
        temp_dir : str, optional
            Custom directory to store temporary files. If provided, overrides temp_location.
        """
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.temp_location = temp_location
        self.temp_dir = temp_dir
        
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
                      out_file: h5py.File) -> None:
        """Process a batch of query sequences against all reference sequences."""
        batch_size = len(queries)
        total_refs = len(refs)
        
        # Create dataset for this batch if it doesn't exist
        if f'batch_{start_idx}' not in out_file:
            out_file.create_dataset(
                f'batch_{start_idx}',
                shape=(batch_size, total_refs),
                dtype=SCORE_DTYPE,
                chunks=(min(100, batch_size), min(self.chunk_size, total_refs)),  # Larger chunks
                compression='gzip' if total_refs > MEMORY_MAPPED_THRESHOLD else None
            )
        
        # Process queries in parallel chunks
        query_chunk_size = min(100, batch_size)  # Process 100 queries at a time
        for i in range(0, batch_size, query_chunk_size):
            query_chunk_end = min(i + query_chunk_size, batch_size)
            query_chunk = queries[i:query_chunk_end]
            
            # Pre-allocate output array for the chunk
            chunk_out = np.empty((len(query_chunk), total_refs), SCORE_DTYPE)
            
            # Process reference sequences in chunks
            for j in range(0, total_refs, self.chunk_size):
                chunk_end = min(j + self.chunk_size, total_refs)
                chunk_refs = refs[j:chunk_end]
                
                values = _cast_sigs_array(chunk_refs.values)
                bounds = chunk_refs.bounds.astype(BOUNDS_DTYPE, copy=False)
                
                # Process each query in the chunk
                for k, query in enumerate(query_chunk):
                    query = _cast_sigs_array(query)
                    _cmetric._jaccarddist_parallel(query, values, bounds, chunk_out[k, j:chunk_end])
            
            # Write chunk results to HDF5 file
            out_file[f'batch_{start_idx}'][i:query_chunk_end] = chunk_out
            
    def calculate_distances(self,
                          queries: Sequence[KmerSignature],
                          refs: Sequence[KmerSignature],
                          output_file: str,
                          progress = None) -> None:
        """
        Calculate Jaccard distances between query and reference sequences using batch processing.
        """
        # Convert refs to SignatureArray if it isn't already
        if not isinstance(refs, SignatureArray):
            print("Converting reference sequences to SignatureArray...")
            refs = SignatureArray(refs)
            
        total_queries = len(queries)
        total_refs = len(refs)
        
        # Check if this is a square matrix (self-comparison)
        is_square = total_queries == total_refs and queries is refs
        if is_square:
            print("\nDetected square matrix (self-comparison) - using symmetry optimization")
            # For square matrices, we only need to calculate the upper triangle
            total_comparisons = (total_queries * (total_queries - 1)) // 2 + total_queries
            print(f"  Matrix type: Square ({total_queries:,} x {total_queries:,})")
            print(f"  Only calculating upper triangle + diagonal")
        else:
            total_comparisons = total_queries * total_refs
            print(f"\nMatrix type: Rectangular ({total_queries:,} x {total_refs:,})")
        
        # Print informative messages about the dataset and configuration
        print("\nDataset Information:")
        print(f"  Number of query sequences: {total_queries:,}")
        print(f"  Number of reference sequences: {total_refs:,}")
        print(f"  Total comparisons: {total_comparisons:,}")
        if is_square:
            print("  Using symmetry optimization (50% fewer calculations)")
            print(f"  Memory savings: {total_comparisons:,} vs {total_queries * total_refs:,} comparisons")
        
        # For small datasets, use the original in-memory implementation
        if total_queries < SMALL_DATASET_THRESHOLD and total_refs < SMALL_DATASET_THRESHOLD:
            print("\nUsing in-memory implementation for small dataset")
            dmat = jaccarddist_matrix(queries, refs, progress=progress)
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('distances', data=dmat)
            return
            
        # Optimize batch sizes based on dataset size
        print("\nOptimizing batch sizes...")
        self.batch_size, self.chunk_size = _optimize_batch_sizes(total_queries, total_refs)
        
        print(f"\nBatch Configuration:")
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
            # Process queries in batches with less frequent progress updates
            update_interval = max(1, total_queries // 100)  # Update progress every 1%
            with get_progress(progress, total=total_queries, desc='Calculating distances') as pbar:
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
                        self._process_batch(batch_queries, refs, i, temp_file)
                    
                    # Update progress less frequently for large datasets
                    if i % update_interval == 0:
                        pbar.increment(len(batch_queries))
            
            print("\nCombining results...")
            # Combine results into final output file
            self._combine_results(temp_file, output_file, total_queries, total_refs, is_square)
            
        finally:
            # Clean up temporary file
            print("\nCleaning up temporary files...")
            temp_file.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
            chunk_out = np.empty((len(query_chunk), total_refs), SCORE_DTYPE)
            
            # Process reference sequences in chunks
            for j in range(0, total_refs, self.chunk_size):
                chunk_end = min(j + self.chunk_size, total_refs)
                chunk_refs = refs[j:chunk_end]
                
                values = _cast_sigs_array(chunk_refs.values)
                bounds = chunk_refs.bounds.astype(BOUNDS_DTYPE, copy=False)
                
                # Process each query in the chunk
                for k, query in enumerate(query_chunk):
                    query = _cast_sigs_array(query)
                    _cmetric._jaccarddist_parallel(query, values, bounds, chunk_out[k, j:chunk_end])
            
            # Write chunk results to HDF5 file
            out_file[f'batch_{start_idx}'][i:query_chunk_end] = chunk_out

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
            
            if is_square:
                # For square matrices, process in chunks to reduce memory usage
                merge_chunk_size = min(100, self.batch_size)  # Use smaller chunks for merging
                
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
            else:
                # For non-square matrices, process in chunks
                merge_chunk_size = min(100, self.batch_size)
                
                for i in range(0, total_queries, self.batch_size):
                    batch_end = min(i + self.batch_size, total_queries)
                    batch_data = temp_file[f'batch_{i}']
                    
                    # Process each chunk of the batch
                    for chunk_start in range(0, batch_end - i, merge_chunk_size):
                        chunk_end = min(chunk_start + merge_chunk_size, batch_end - i)
                        chunk_data = batch_data[chunk_start:chunk_end]
                        out_file['distances'][i + chunk_start:i + chunk_end] = chunk_data

def jaccarddist_matrix_improved(queries: Sequence[KmerSignature],
                              refs: Sequence[KmerSignature],
                              output_file: str,
                              batch_size: int = DEFAULT_BATCH_SIZE,
                              chunk_size: int = DEFAULT_CHUNK_SIZE,
                              temp_location: TempLocation = 'system',
                              progress = None) -> None:
    """
    Memory-efficient implementation of jaccarddist_matrix.
    
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
        Where to store temporary files:
        - 'output_dir': Store in same directory as output file
        - 'ram': Store in RAM-based filesystem (e.g. /dev/shm on Linux)
        - 'system': Use system's default temp directory
    progress : optional
        Progress bar configuration
    """
    calculator = BatchedDistanceCalculator(
        batch_size=batch_size,
        chunk_size=chunk_size,
        temp_location=temp_location
    )
    calculator.calculate_distances(queries, refs, output_file, progress)

def jaccarddist_pairwise_improved(sigs: Sequence[KmerSignature],
                                output_file: str,
                                batch_size: int = DEFAULT_BATCH_SIZE,
                                chunk_size: int = DEFAULT_CHUNK_SIZE,
                                temp_location: TempLocation = 'system',
                                progress = None) -> None:
    """
    Memory-efficient implementation of jaccarddist_pairwise.
    
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
        Where to store temporary files:
        - 'output_dir': Store in same directory as output file
        - 'ram': Store in RAM-based filesystem (e.g. /dev/shm on Linux)
        - 'system': Use system's default temp directory
    progress : optional
        Progress bar configuration
    """
    calculator = BatchedDistanceCalculator(
        batch_size=batch_size,
        chunk_size=chunk_size,
        temp_location=temp_location
    )
    calculator.calculate_distances(sigs, sigs, output_file, progress)
