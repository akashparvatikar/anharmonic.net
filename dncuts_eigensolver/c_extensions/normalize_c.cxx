// C++ Templates of Normalization subroutine
//
// Generates 'libnormalize_c.a'
#include "normalize_c.h"
#include <cstdlib>
#include <iostream>

// C-interface routines for F2PY wrapper

extern "C" void wrap_norm (
	const int n,
	const double *sums,		// read-only:  sums[n]
	const int *indptr,  	// read-only: indptr[n+1]
		  double *data)		// read-write: data[k]
{
	try {
		(void)ce::norm<double>(n, sums, indptr, data);
	} catch(const char *e) {
		std::cerr << "norm<double>: " << e << std::endl;
		std::exit(1);
	} catch(...) {
		std::cerr << "norm<double>: Unhandled Exception" << std::endl;
		std::exit(1);
	}
}

