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
        
        Parameters
        ----------
        queries : Sequence[KmerSignature]
            Query sequences
        refs : Sequence[KmerSignature]
            Reference sequences
        output_file : str
            Path to output HDF5 file
        progress : optional
            Progress bar configuration
        """
        # Convert refs to SignatureArray if it isn't already
        if not isinstance(refs, SignatureArray):
            refs = SignatureArray(refs)
            
        total_queries = len(queries)
        total_refs = len(refs)
        
        # Print informative messages about the dataset and configuration
        print(f"\nDataset Information:")
        print(f"  Number of query sequences: {total_queries:,}")
        print(f"  Number of reference sequences: {total_refs:,}")
        print(f"  Total comparisons: {total_queries * total_refs:,}")
        
        # For small datasets, use the original in-memory implementation
        if total_queries < SMALL_DATASET_THRESHOLD and total_refs < SMALL_DATASET_THRESHOLD:
            print("\nUsing in-memory implementation for small dataset")
            dmat = jaccarddist_matrix(queries, refs, progress=progress)
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('distances', data=dmat)
            return
            
        # Optimize batch sizes based on dataset size
        self.batch_size, self.chunk_size = _optimize_batch_sizes(total_queries, total_refs)
        
        print(f"\nBatch Configuration:")
        print(f"  Batch size: {self.batch_size:,} queries per batch")
        print(f"  Chunk size: {self.chunk_size:,} references per chunk")
        print(f"  Number of batches: {(total_queries + self.batch_size - 1) // self.batch_size:,}")
        print(f"  Number of chunks per batch: {(total_refs + self.chunk_size - 1) // self.chunk_size:,}")
        
        # Create temporary file for intermediate results
        temp_file, temp_path = self._create_temp_file(total_queries, total_refs, output_file)
        print(f"\nTemporary storage:")
        print(f"  Location: {self.temp_location}")
        print(f"  Path: {temp_path}")
        
        try:
            # Process queries in batches with less frequent progress updates
            update_interval = max(1, total_queries // 100)  # Update progress every 1%
            with get_progress(progress, total=total_queries, desc='Calculating distances') as pbar:
                for i in range(0, total_queries, self.batch_size):
                    batch_end = min(i + self.batch_size, total_queries)
                    batch_queries = queries[i:batch_end]
                    
                    # Process this batch
                    self._process_batch(batch_queries, refs, i, temp_file)
                    
                    # Update progress less frequently for large datasets
                    if i % update_interval == 0:
                        pbar.increment(len(batch_queries))
            
            # Combine results into final output file
            self._combine_results(temp_file, output_file, total_queries, total_refs)
            
        finally:
            # Clean up temporary file
            temp_file.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def _combine_results(self,
                        temp_file: h5py.File,
                        output_file: str,
                        total_queries: int,
                        total_refs: int) -> None:
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
            
            # Copy data from temporary file
            for i in range(0, total_queries, self.batch_size):
                batch_end = min(i + self.batch_size, total_queries)
                out_file['distances'][i:batch_end] = temp_file[f'batch_{i}'][:]

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
