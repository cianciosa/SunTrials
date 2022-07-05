//
//  matrix_custom.hpp
//  Sundials_Trial
//
//  Created by Cianciosa, Mark on 6/28/22.
//

#ifndef matrix_custom_h
#define matrix_custom_h

#include <vector>
#include <Accelerate/Accelerate.h>

#include <cvode/cvode.h>

struct _SUNMatrixContent_Custom {
    size_t num_rows;
    size_t num_colmns;
    std::vector<double> data;
    
    _SUNMatrixContent_Custom(const size_t nrows,
                             const size_t ncols) :
    num_rows(nrows), num_colmns(ncols), data(nrows*ncols) {}
};

typedef struct _SUNMatrixContent_Custom *SUNMatrixContent_Custom;

#define SUNMATRIX_CUSTOM_CONTENT(x) static_cast<SUNMatrixContent_Custom> (x->content)

SUNMatrix SUNCustomMatrix(sunindextype M, sunindextype N, SUNContext ctx);

SUNMatrix_ID SUNMatGetID_Custom(SUNMatrix a) {
    return SUNMATRIX_CUSTOM;
}

SUNMatrix SUNMatClone_Custom(SUNMatrix a) {
    return SUNCustomMatrix(SUNMATRIX_CUSTOM_CONTENT(a)->num_rows,
                           SUNMATRIX_CUSTOM_CONTENT(a)->num_colmns,
                           a->sunctx);
}

void SUNMatDestroy_Custom(SUNMatrix a) {
    if (!a) {
        return;
    }
    
    if (!a->content) {
        delete SUNMATRIX_CUSTOM_CONTENT(a);
        a->content = NULL;
    }
    
    free(a);
    a = NULL;
}

int SUNMatZero_Custom(SUNMatrix a) {
    SUNMATRIX_CUSTOM_CONTENT(a)->data.assign(SUNMATRIX_CUSTOM_CONTENT(a)->data.size(), 0.0);
    return SUNMAT_SUCCESS;
}

int SUNMatCopy_Custom(SUNMatrix a, SUNMatrix b) {
    SUNMATRIX_CUSTOM_CONTENT(b)->data.assign(SUNMATRIX_CUSTOM_CONTENT(a)->data.cbegin(),
                                             SUNMATRIX_CUSTOM_CONTENT(a)->data.cend());
    return SUNMAT_SUCCESS;
}

int SUNMatScaleAdd_Custom(realtype c, SUNMatrix a, SUNMatrix b) {
    for (size_t i = 0, ie = SUNMATRIX_CUSTOM_CONTENT(a)->data.size(); i < ie; i++) {
        SUNMATRIX_CUSTOM_CONTENT(a)->data[i] = c*SUNMATRIX_CUSTOM_CONTENT(a)->data[i]
                                             +   SUNMATRIX_CUSTOM_CONTENT(b)->data[i];
    }
    return SUNMAT_SUCCESS;
}

int SUNMatScaleAddI_Custom(realtype c, SUNMatrix a) {
    const size_t end = SUNMATRIX_CUSTOM_CONTENT(a)->data.size();
    for (size_t i = 0, ie = end; i < ie; i++) {
        SUNMATRIX_CUSTOM_CONTENT(a)->data[i] = c*SUNMATRIX_CUSTOM_CONTENT(a)->data[i];
        
    }
    const size_t stride = SUNMATRIX_CUSTOM_CONTENT(a)->num_colmns + 1;
    for (size_t i = 0, ie = end; i < ie; i += stride) {
        SUNMATRIX_CUSTOM_CONTENT(a)->data[i] += 1.0;
    }
    return SUNMAT_SUCCESS;
}

int SUNMatMatvec_Custom(SUNMatrix a, N_Vector x, N_Vector y) {
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                static_cast<int> (SUNMATRIX_CUSTOM_CONTENT(a)->num_rows),
                static_cast<int> (SUNMATRIX_CUSTOM_CONTENT(a)->num_colmns),
                1.0, SUNMATRIX_CUSTOM_CONTENT(a)->data.data(),
                static_cast<int> (N_VGetLength(x)),
                N_VGetArrayPointer(x),
                1, 0.0, y->ops->nvgetarraypointer(y), 1);
    return SUNMAT_SUCCESS;
}

int SUNMatSpace_Custom(SUNMatrix a, long int *lenrw, long int *leniw) {
    *lenrw = SUNMATRIX_CUSTOM_CONTENT(a)->data.size();
    *leniw = 2;
    return SUNMAT_SUCCESS;
}

SUNMatrix SUNCustomMatrix(sunindextype M, sunindextype N, SUNContext ctx) {
    if (M <= 0 || N <=0) {
        return NULL;
    }
    
    SUNMatrix a = SUNMatNewEmpty(ctx);
    
//  Attach operations.
    a->ops->getid     = SUNMatGetID_Custom;
    a->ops->clone     = SUNMatClone_Custom;
    a->ops->destroy   = SUNMatDestroy_Custom;
    a->ops->zero      = SUNMatZero_Custom;
    a->ops->copy      = SUNMatCopy_Custom;
    a->ops->scaleadd  = SUNMatScaleAdd_Custom;
    a->ops->scaleaddi = SUNMatScaleAddI_Custom;
    a->ops->matvec    = SUNMatMatvec_Custom;
    a->ops->space     = SUNMatSpace_Custom;
    
    a->content = new _SUNMatrixContent_Custom(M, N);
    
    return a;
}

#endif /* matrix_custom_h */
