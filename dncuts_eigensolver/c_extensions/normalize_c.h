#ifndef NORMALIZE_C_H
#define NORMALIZE_C_H

#include <stdint.h>
#include <omp.h>

#include <iostream>

namespace ce
{
	template <typename T>
	void norm(	const int n,
				const T *sums,		// read-only:  sums[n]
				const int *indptr,  // read-only: indptr[n+1]
				T *data)		    // read-write: data[k]
	{
		int i, j;

		#pragma omp parallel for shared(data, indptr, sums) private(i, j)
		// Loop over all of the sums.
		for(i = 0; i<n; i++)
		{
			for(j = indptr[i]; j < indptr[i+1]; j++)
			{
			// Normalize each column.
				data[j] = data[j]/sums[i];		// data[k]
			}
		}

		//return data;
	};
}

#endif	// NORMALIZE_C_H
