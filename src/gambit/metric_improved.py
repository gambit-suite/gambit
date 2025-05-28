"""Memory-efficient implementations of Jaccard distance calculations."""

import os
import tempfile
from typing import Sequence, Optional, Tuple, Union
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
                 temp_dir: Optional[str] = None):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        batch_size : int
            Number of query sequences to process in each batch
        chunk_size : int
            Number of reference sequences to process in each chunk
        temp_dir : str, optional
            Directory to store temporary files. If None, uses system temp directory
        """
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def _create_temp_file(self, total_queries: int, total_refs: int) -> Tuple[h5py.File, str]:
        """Create a temporary HDF5 file for storing intermediate results."""
        temp_path = os.path.join(self.temp_dir, f'gambit_dist_{os.getpid()}.h5')
        
        # Use memory-mapped files for small datasets
        if total_queries * total_refs < MEMORY_MAPPED_THRESHOLD:
            return h5py.File(temp_path, 'w', driver='core', backing_store=False), temp_path
        return h5py.File(temp_path, 'w'), temp_path
        
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
                chunks=(1, min(self.chunk_size, total_refs)),
                compression='gzip' if total_refs > MEMORY_MAPPED_THRESHOLD else None
            )
        
        # Process each query in the batch
        for i, query in enumerate(queries):
            query = _cast_sigs_array(query)
            out = np.empty(total_refs, SCORE_DTYPE)
            
            # Process reference sequences in chunks
            for j in range(0, total_refs, self.chunk_size):
                chunk_end = min(j + self.chunk_size, total_refs)
                chunk_refs = refs[j:chunk_end]
                
                values = _cast_sigs_array(chunk_refs.values)
                bounds = chunk_refs.bounds.astype(BOUNDS_DTYPE, copy=False)
                
                # Calculate distances for this chunk
                _cmetric._jaccarddist_parallel(query, values, bounds, out[j:chunk_end])
            
            # Write results to HDF5 file
            out_file[f'batch_{start_idx}'][i] = out
            
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
        
        # For small datasets, use the original in-memory implementation
        if total_queries < SMALL_DATASET_THRESHOLD and total_refs < SMALL_DATASET_THRESHOLD:
            dmat = jaccarddist_matrix(queries, refs, progress=progress)
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('distances', data=dmat)
            return
            
        # Optimize batch sizes based on dataset size
        self.batch_size, self.chunk_size = _optimize_batch_sizes(total_queries, total_refs)
        
        # Create temporary file for intermediate results
        temp_file, temp_path = self._create_temp_file(total_queries, total_refs)
        
        try:
            # Process queries in batches
            with get_progress(progress, total=total_queries, desc='Calculating distances') as pbar:
                for i in range(0, total_queries, self.batch_size):
                    batch_end = min(i + self.batch_size, total_queries)
                    batch_queries = queries[i:batch_end]
                    
                    # Process this batch
                    self._process_batch(batch_queries, refs, i, temp_file)
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
    progress : optional
        Progress bar configuration
    """
    calculator = BatchedDistanceCalculator(
        batch_size=batch_size,
        chunk_size=chunk_size
    )
    calculator.calculate_distances(queries, refs, output_file, progress)

def jaccarddist_pairwise_improved(sigs: Sequence[KmerSignature],
                                output_file: str,
                                batch_size: int = DEFAULT_BATCH_SIZE,
                                chunk_size: int = DEFAULT_CHUNK_SIZE,
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
    progress : optional
        Progress bar configuration
    """
    calculator = BatchedDistanceCalculator(
        batch_size=batch_size,
        chunk_size=chunk_size
    )
    calculator.calculate_distances(sigs, sigs, output_file, progress)
