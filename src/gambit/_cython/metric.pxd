from libc.stdint cimport intptr_t
from .types cimport SCORE_T, BOUNDS_T, COORDS_T, COORDS_T_2

# Original function - now optimized internally
cdef SCORE_T c_jaccarddist(COORDS_T[:] coords1, COORDS_T_2[:] coords2) noexcept nogil

# New optimized function for direct indexing (eliminates array slicing)
cdef SCORE_T c_jaccarddist_direct(COORDS_T_2[:] ref_coords, BOUNDS_T begin, BOUNDS_T end, 
                                  COORDS_T[:] query) noexcept nogil

# Pure Cython MinHash functions - NO NUMPY DEPENDENCIES
cdef void c_compute_minhash(COORDS_T[:] coords, unsigned long* hash_functions, 
                           unsigned int* minhash_sig, int num_hashes) noexcept nogil

cdef float c_estimate_jaccard_from_minhash(unsigned int* sig1, unsigned int* sig2, int num_hashes) noexcept nogil

cdef void c_init_hash_functions(unsigned long* hash_functions, int num_hashes) noexcept nogil

# Public MinHash pre-filtering functions - both parallel and optimized versions
cpdef precompute_similarity_candidates(COORDS_T_2[:] all_coords, BOUNDS_T[:] bounds, 
                                      float threshold=*, int num_hashes=*)

cpdef precompute_similarity_candidates_optimized(COORDS_T_2[:] all_coords, BOUNDS_T[:] bounds, 
                                                float threshold=*, int num_hashes=*)