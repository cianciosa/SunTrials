//
//  linear_solve_custom.hpp
//  Sundials_Trial
//
//  Created by Cianciosa, Mark on 6/28/22.
//

#ifndef linear_solve_custom_h
#define linear_solve_custom_h

#include <Accelerate/Accelerate.h>

#include <vector>

#include <cvode/cvode.h>

#include "matrix_custom.hpp"

struct _SUNLinearSolverContent_Custom {
    std::vector<int> pivot;
    size_t rows;
    
    _SUNLinearSolverContent_Custom(const size_t nrows,
                                   const size_t ncols) :
    pivot(nrows*ncols), rows(nrows) {}
};

typedef _SUNLinearSolverContent_Custom *SUNLinearSolverContent_Custom;

#define SUNLS_CUSTOM_CONTENT(x) static_cast<SUNLinearSolverContent_Custom> (x->content)

SUNLinearSolver_Type SUNLinSolGetType_Custom(SUNLinearSolver S) {
    return SUNLINEARSOLVER_DIRECT;
}

SUNLinearSolver_ID SUNLinSolGetID_Custom(SUNLinearSolver s) {
    return SUNLINEARSOLVER_CUSTOM;
}

int SUNLinSolInitialize_Custom(SUNLinearSolver s) {
    return SUNLS_SUCCESS;
}

int SUNLinSolSetup_Custom(SUNLinearSolver s, SUNMatrix a) {
    int rows = static_cast<int> (SUNMATRIX_CUSTOM_CONTENT(a)->num_rows);
    int cols = static_cast<int> (SUNMATRIX_CUSTOM_CONTENT(a)->num_colmns);
    int info = 0;
    dgetrf_(&rows, &cols, SUNMATRIX_CUSTOM_CONTENT(a)->data.data(),
            &rows, SUNLS_CUSTOM_CONTENT(s)->pivot.data(), &info);
    
    if (info) {
        return SUNLS_LUFACT_FAIL;
    }
    return SUNLS_SUCCESS;
}

int SUNLinSolSolve_Custom(SUNLinearSolver s, SUNMatrix a, N_Vector x, N_Vector b, realtype tol) {
    char transpose = 'N';
    int rows = static_cast<int> (SUNMATRIX_CUSTOM_CONTENT(a)->num_rows);
    int one = 1;
    int info = 0;
    
    dgetrs_(&transpose, &rows, &one, SUNMATRIX_CUSTOM_CONTENT(a)->data.data(),
            &rows, SUNLS_CUSTOM_CONTENT(s)->pivot.data(), NVEC_CUSTOM_CONTENT(x)->buffer.data(),
            &rows, &info);

    return SUNLS_SUCCESS;
}

sunindextype SUNLinSolLastFlag_Custom(SUNLinearSolver s) {
    return SUNLS_SUCCESS;
}

int SUNLinSolSpace_Custom(SUNLinearSolver s,
                          long int *lenrwls,
                          long int *leniwls) {
    *lenrwls = SUNLS_CUSTOM_CONTENT(s)->pivot.size();
    *leniwls = 1;
    return SUNLS_SUCCESS;
}

int SUNLinSolFree_Custom(SUNLinearSolver s) {
    if (!s) {
        return SUNLS_SUCCESS;
    }
    
    if (s->content) {
        delete SUNLS_CUSTOM_CONTENT(s);
        s->content = NULL;
    }
    
    free(s);
    s = NULL;
    
    return SUNLS_SUCCESS;
}

SUNLinearSolver SUNLinSol_Custom(N_Vector y, SUNMatrix a, SUNContext ctx) {
    SUNLinearSolver s = SUNLinSolNewEmpty(ctx);

    s->ops->gettype    = SUNLinSolGetType_Custom;
    s->ops->getid      = SUNLinSolGetID_Custom;
    s->ops->initialize = SUNLinSolInitialize_Custom;
    s->ops->setup      = SUNLinSolSetup_Custom;
    s->ops->solve      = SUNLinSolSolve_Custom;
    s->ops->lastflag   = SUNLinSolLastFlag_Custom;
    s->ops->space      = SUNLinSolSpace_Custom;
    s->ops->free       = SUNLinSolFree_Custom;
    
    s->content = new _SUNLinearSolverContent_Custom(SUNMATRIX_CUSTOM_CONTENT(a)->num_rows,
                                                    SUNMATRIX_CUSTOM_CONTENT(a)->num_colmns);
    
    return s;
}

#endif /* linear_solve_custom_h */
