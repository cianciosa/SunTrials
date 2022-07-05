//
//  nvector_custom.hpp
//  Sundials_Trial
//
//  Created by Cianciosa, Mark on 6/27/22.
//

#ifndef nvector_custom_h
#define nvector_custom_h

#include <vector>
#include <cmath>
#include <algorithm>

#include <cvode/cvode.h>

struct _N_VectorContent_Custom {
    std::vector<double> buffer;
    
    _N_VectorContent_Custom(const size_t size) : buffer(size) {}
};

typedef struct _N_VectorContent_Custom* N_VectorContent_Custom;

#define NVEC_CUSTOM_CONTENT(x) static_cast<N_VectorContent_Custom> (x->content)

N_Vector N_VNewEmpty_Custom(SUNContext ctx, const size_t s);

N_Vector_ID N_VGetVectorID_Custom(N_Vector v) {
    return SUNDIALS_NVEC_CUSTOM;
}

sunindextype N_VGetLength_Custom(N_Vector v) {
    return NVEC_CUSTOM_CONTENT(v)->buffer.size();
}

N_Vector N_VCloneEmpty_Custom(N_Vector w) {
    if (!w) {
        return w;
    }
    
    N_Vector v = N_VNewEmpty_Custom(w->sunctx,
                                    static_cast<size_t> (N_VGetLength_Custom(w)));
    if (!v) {
        return v;
    }
    
    NVEC_CUSTOM_CONTENT(v)->buffer.assign(NVEC_CUSTOM_CONTENT(w)->buffer.cbegin(),
                                          NVEC_CUSTOM_CONTENT(w)->buffer.cend());
    
    return v;
}

N_Vector N_VClone_Custom(N_Vector w) {
    N_Vector v = N_VCloneEmpty_Custom(w);
    if (!v) {
        return v;
    }
    
    NVEC_CUSTOM_CONTENT(v)->buffer = NVEC_CUSTOM_CONTENT(w)->buffer;
    
    return v;
}

void N_VDestroy_Custom(N_Vector v) {
    if (!v) {
        return;
    }
    
    if (!v->ops) {
        free(v->ops);
        v->ops = NULL;
    }
    
    if (!v->content) {
        delete NVEC_CUSTOM_CONTENT(v);
        v->content = NULL;
    }
    
    free(v);
    v = NULL;
}

void N_VSpace_Custom(N_Vector v, sunindextype *lrw, sunindextype *liw) {
    *lrw = N_VGetLength_Custom(v);
    *liw = 0;
}

realtype *N_VGetHostArrayPointer_Custom(N_Vector v) {
    return NVEC_CUSTOM_CONTENT(v)->buffer.data();
}

realtype *N_VGetDeviceArrayPointer_Custom(N_Vector v) {
    return NVEC_CUSTOM_CONTENT(v)->buffer.data();
}

void N_VSetHostArrayPointer_Custom(realtype *h_vdata, N_Vector v) {
    NVEC_CUSTOM_CONTENT(v)->buffer.assign(h_vdata,
                                          h_vdata + N_VGetLength_Custom(v));
}

void N_VLinearSum_Custom(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = a*NVEC_CUSTOM_CONTENT(x)->buffer[i]
                                          + b*NVEC_CUSTOM_CONTENT(y)->buffer[i];
    }
}

void N_VConst_Custom(realtype a, N_Vector v) {
    NVEC_CUSTOM_CONTENT(v)->buffer.assign(N_VGetLength_Custom(v), a);
}

void N_VProd_Custom(N_Vector x, N_Vector y, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = NVEC_CUSTOM_CONTENT(x)->buffer[i]
                                          * NVEC_CUSTOM_CONTENT(y)->buffer[i];
    }
}

void N_VDiv_Custom(N_Vector x, N_Vector y, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = NVEC_CUSTOM_CONTENT(x)->buffer[i]
                                          / NVEC_CUSTOM_CONTENT(y)->buffer[i];
    }
}

void N_VScale_Custom(realtype c, N_Vector x, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = c*NVEC_CUSTOM_CONTENT(x)->buffer[i];
    }
}

void N_VAbs_Custom(N_Vector x, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = std::abs(NVEC_CUSTOM_CONTENT(x)->buffer[i]);
    }
}

void N_VInv_Custom(N_Vector x, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = 1.0/NVEC_CUSTOM_CONTENT(x)->buffer[i];
    }
}

void N_VAddConst_Custom(N_Vector x, realtype b, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = NVEC_CUSTOM_CONTENT(x)->buffer[i] + b;
    }
}

realtype N_VDotProd_Custom(N_Vector x, N_Vector y) {
    double result = 0;
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        result += NVEC_CUSTOM_CONTENT(x)->buffer[i]
                * NVEC_CUSTOM_CONTENT(y)->buffer[i];
    }
    return result;
}

realtype N_VMaxNorm_Custom(N_Vector x) {
    double max = 0;
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        max = std::max(max, std::abs(NVEC_CUSTOM_CONTENT(x)->buffer[i]));
    }
    return max;
}

realtype N_VMin_Custom(N_Vector x) {
    return *std::min_element(NVEC_CUSTOM_CONTENT(x)->buffer.cbegin(),
                             NVEC_CUSTOM_CONTENT(x)->buffer.cend());
}

realtype N_VL1Norm_Custom(N_Vector x) {
    double result = 0;
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        result += std::abs(NVEC_CUSTOM_CONTENT(x)->buffer[i]);
    }
    return result;
}

booleantype N_VInvTest_Custom(N_Vector x, N_Vector z) {
    bool all_pass = true;
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = 1.0/NVEC_CUSTOM_CONTENT(x)->buffer[i];
        all_pass = all_pass && NVEC_CUSTOM_CONTENT(x)->buffer[i];
    }
    return all_pass;
}

booleantype N_VConstrMask_Custom(N_Vector c, N_Vector x, N_Vector m) {
    bool all_pass = true;
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        if (NVEC_CUSTOM_CONTENT(c)->buffer[i] == 2) {
            const bool test = NVEC_CUSTOM_CONTENT(x)->buffer[i] > 0;
            NVEC_CUSTOM_CONTENT(m)->buffer[i] = !test;
            all_pass = all_pass && test;
        } else if (NVEC_CUSTOM_CONTENT(c)->buffer[i] == 1) {
            const bool test = NVEC_CUSTOM_CONTENT(x)->buffer[i] >= 0;
            NVEC_CUSTOM_CONTENT(m)->buffer[i] = !test;
            all_pass = all_pass && test;
        } else if (NVEC_CUSTOM_CONTENT(c)->buffer[i] == -2) {
            const bool test = NVEC_CUSTOM_CONTENT(x)->buffer[i] < 0;
            NVEC_CUSTOM_CONTENT(m)->buffer[i] = !test;
            all_pass = all_pass && test;
        } else {
            const bool test = NVEC_CUSTOM_CONTENT(x)->buffer[i] <= 0;
            NVEC_CUSTOM_CONTENT(m)->buffer[i] = !test;
            all_pass = all_pass && test;
        }
    }
    return all_pass;
}

realtype N_VWl2norm_Custom(N_Vector x, N_Vector w) {
    double sum = 0;
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        sum += NVEC_CUSTOM_CONTENT(x)->buffer[i]*NVEC_CUSTOM_CONTENT(w)->buffer[i]
            *  NVEC_CUSTOM_CONTENT(x)->buffer[i]*NVEC_CUSTOM_CONTENT(w)->buffer[i];
    }
    return std::sqrt(sum);
}

realtype N_VWrmsNorm_Custom(N_Vector x, N_Vector w) {
    return N_VWl2norm_Custom(x, w)/std::sqrt(N_VGetLength_Custom(x));
}

realtype N_VWrmsNormMask_Custom(N_Vector x, N_Vector w, N_Vector id) {
    double sum = 0;
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        double temp = NVEC_CUSTOM_CONTENT(x)->buffer[i]*NVEC_CUSTOM_CONTENT(w)->buffer[i]*(NVEC_CUSTOM_CONTENT(w)->buffer[i] > 0);
        sum += temp*temp;
    }
    return std::sqrt(sum);
}

void N_VCompare_Custom(realtype c, N_Vector x, N_Vector z) {
    for (size_t i = 0, ie = N_VGetLength_Custom(x); i < ie; i++) {
        NVEC_CUSTOM_CONTENT(z)->buffer[i] = std::abs(NVEC_CUSTOM_CONTENT(x)->buffer[i]) >= c;
    }
}

realtype N_VMinQuotient_Custom(N_Vector num, N_Vector denom) {
    size_t first = 0;
    while (NVEC_CUSTOM_CONTENT(denom)->buffer[first] == 0) {
        ++first;
    }
    double min = NVEC_CUSTOM_CONTENT(num)->buffer[first]/NVEC_CUSTOM_CONTENT(denom)->buffer[first];
    for (size_t i = first + 1, ie = N_VGetLength_Custom(num); i < ie; i++) {
        if (NVEC_CUSTOM_CONTENT(denom)->buffer[i] > 0) {
            min = std::min(min, NVEC_CUSTOM_CONTENT(num)->buffer[i]/NVEC_CUSTOM_CONTENT(denom)->buffer[i]);
        }
    }
    return min;
}

N_Vector N_VNewEmpty_Custom(SUNContext ctx, const size_t s) {
    N_Vector v = N_VNewEmpty(ctx);
    if (!v) {
        return v;
    }
    
//  Constructors, destructors, and utility operations.
    v->ops->nvgetvectorid           = N_VGetVectorID_Custom;
    v->ops->nvclone                 = N_VClone_Custom;
    v->ops->nvcloneempty            = N_VCloneEmpty_Custom;
    v->ops->nvdestroy               = N_VDestroy_Custom;
    v->ops->nvspace                 = N_VSpace_Custom;
    v->ops->nvgetlength             = N_VGetLength_Custom;
    v->ops->nvgetarraypointer       = N_VGetHostArrayPointer_Custom;
    v->ops->nvgetdevicearraypointer = N_VGetDeviceArrayPointer_Custom;
    v->ops->nvsetarraypointer       = N_VSetHostArrayPointer_Custom;

//  Standard vector operations.
    v->ops->nvlinearsum    = N_VLinearSum_Custom;
    v->ops->nvconst        = N_VConst_Custom;
    v->ops->nvprod         = N_VProd_Custom;
    v->ops->nvdiv          = N_VDiv_Custom;
    v->ops->nvscale        = N_VScale_Custom;
    v->ops->nvabs          = N_VAbs_Custom;
    v->ops->nvinv          = N_VInv_Custom;
    v->ops->nvaddconst     = N_VAddConst_Custom;
    v->ops->nvdotprod      = N_VDotProd_Custom;
    v->ops->nvmaxnorm      = N_VMaxNorm_Custom;
    v->ops->nvwrmsnorm     = N_VWrmsNorm_Custom;
    v->ops->nvwrmsnormmask = N_VWrmsNormMask_Custom;
    v->ops->nvmin          = N_VMin_Custom;
    v->ops->nvwl2norm      = N_VWl2norm_Custom;
    v->ops->nvl1norm       = N_VL1Norm_Custom;
    v->ops->nvcompare      = N_VCompare_Custom;
    v->ops->nvinvtest      = N_VInvTest_Custom;
    v->ops->nvconstrmask   = N_VConstrMask_Custom;
    v->ops->nvminquotient  = N_VMinQuotient_Custom;
    
//  Create content.
    v->content = new _N_VectorContent_Custom(s);

    return v;
}

#endif /* nvector_custom_h */
