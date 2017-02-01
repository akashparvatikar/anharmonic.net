// reference C-interfaces routines from 'normalize_c.cxx' (libnormalize_c.a)
extern void wrap_norm (
	const int n,
	const double *sums,		// read-only:  sums[n]
	const int *indptr,  	// read-only: indptr[n+1]
		  double *data);	// read-write: data[k]

// pure C-interfaces calling into libnormalize_c.a
void norm (
	const int m,
	const double *sm,		// read-only:  sums[n]
	const int *iptr,  	// read-only: indptr[n+1]
		  double *dat)	// read-write: data[k]
{
	wrap_norm(m, sm, iptr, dat);
}
