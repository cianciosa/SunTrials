//
//  main.cpp
//  Sundials_Trial
//
//  Created by Cianciosa, Mark on 6/27/22.
//

#include <cvode/cvode.h>

#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>

#include <iostream>

#include <chrono>

#include "nvector_custom.hpp"
#include "matrix_custom.hpp"
#include "linear_solve_custom.hpp"

//------------------------------------------------------------------------------
///  @brief Print out timings.
///
///  @param[in] name Discription of the times.
///  @param[in] time Elapsed time in nanoseconds.
//------------------------------------------------------------------------------
void write_time(const std::string &name, const std::chrono::nanoseconds time) {
    if (time.count() < 1000) {
        std::cout << name << time.count()               << " ns" << std::endl;
    } else if (time.count() < 1000000) {
        std::cout << name << time.count()/1000.0        << " μs" << std::endl;
    } else if (time.count() < 1000000000) {
        std::cout << name << time.count()/1000000.0     << " ms" << std::endl;
    } else if (time.count() < 60000000000) {
        std::cout << name << time.count()/1000000000.0  << " s" << std::endl;
    } else if (time.count() < 3600000000000) {
        std::cout << name << time.count()/60000000000.0 << " min" << std::endl;
    } else {
        std::cout << name << time.count()/3600000000000 << " h" << std::endl;
    }
}

//  The system of equations to solve.
//
//  dy1/dt = -0.4*y1 + 1.0E4*y2*y3
//
//  dy2/dt = 0.04*y1 - 1.0E4*y2*y3 - 3.0E7*y2^2
//
//  dy3/dt = 3.0E7*y2^2
//
static int right_hand_side(realtype t, N_Vector y, N_Vector y_dot, void *user_data) {
    NVEC_CUSTOM_CONTENT(y_dot)->buffer[0] = -0.4*NVEC_CUSTOM_CONTENT(y)->buffer[0] + 1.0E4*NVEC_CUSTOM_CONTENT(y)->buffer[1]*NVEC_CUSTOM_CONTENT(y)->buffer[2];
    NVEC_CUSTOM_CONTENT(y_dot)->buffer[1] = 0.04*NVEC_CUSTOM_CONTENT(y)->buffer[0] - 1.0E4*NVEC_CUSTOM_CONTENT(y)->buffer[1]*NVEC_CUSTOM_CONTENT(y)->buffer[2] - 3.0E7*NVEC_CUSTOM_CONTENT(y)->buffer[1]*NVEC_CUSTOM_CONTENT(y)->buffer[1];
    NVEC_CUSTOM_CONTENT(y_dot)->buffer[2] = 3.0E7*NVEC_CUSTOM_CONTENT(y)->buffer[1]*NVEC_CUSTOM_CONTENT(y)->buffer[1];

    return 0;
}

//  The system of equations to solve.
//
//      [ dy1/dtdy1 dy2/dtdy1 dy3/dtdy1 ]
//  J = [ dy1/dtdy2 dy2/dtdy2 dy3/dtdy2 ]
//      [ dy1/dtdy3 dy2/dtdy3 dy3/dtdy3 ]
//
static int jacobian(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                    void *userdata, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SUNMATRIX_CUSTOM_CONTENT(J)->data[0] = -0.4;
    SUNMATRIX_CUSTOM_CONTENT(J)->data[1] = 0.04;
    SUNMATRIX_CUSTOM_CONTENT(J)->data[2] = 0.0;
   
    SUNMATRIX_CUSTOM_CONTENT(J)->data[3] = 1.0E4*NVEC_CUSTOM_CONTENT(y)->buffer[2];
    SUNMATRIX_CUSTOM_CONTENT(J)->data[4] = -1.0E4*NVEC_CUSTOM_CONTENT(y)->buffer[2] - 2.0*3.0E7*NVEC_CUSTOM_CONTENT(y)->buffer[1];
    SUNMATRIX_CUSTOM_CONTENT(J)->data[5] = 2.0*3.0E7*NVEC_CUSTOM_CONTENT(y)->buffer[1];
    
    SUNMATRIX_CUSTOM_CONTENT(J)->data[6] = 1.0E4*NVEC_CUSTOM_CONTENT(y)->buffer[1];
    SUNMATRIX_CUSTOM_CONTENT(J)->data[7] = -1.0E4*NVEC_CUSTOM_CONTENT(y)->buffer[1];
    SUNMATRIX_CUSTOM_CONTENT(J)->data[8] = 0.0;
    
    return 0;
}

//  The system of equations to solve.
//
//  dy/dt = -1.0E3(y - Exp(-t)) - Exp(-t)
//
static int right_hand_side2(realtype t, N_Vector y, N_Vector y_dot, void *user_data) {
    //NVEC_CUSTOM_CONTENT(y_dot)->buffer[0] = -1.0E3*(NVEC_CUSTOM_CONTENT(y)->buffer[0] - std::exp(-t)) - std::exp(-t);
    NV_DATA_S(y_dot)[0] = -1.0E3*(NV_DATA_S(y)[0] - std::exp(-t)) - std::exp(-t);
    
    return 0;
}

//  The system of equations to solve.
//
//  J = [ dy/dtdy ]
//
static int jacobian2(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                     void *userdata, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    //SUNMATRIX_CUSTOM_CONTENT(J)->data[0] = -1.0E3;
    SM_DATA_D(J)[0] = -1.0E3;
    
    return 0;
}

static realtype solution(realtype t) {
    return std::exp(-t) - std::exp(-1000.0*t);
}

int main(int argc, const char * argv[]) {
    const std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    
    sundials::Context ctx;

    size_t neq = 1;
    
    N_Vector y = N_VNew_Serial(neq, ctx);//N_VNewEmpty_Custom(ctx, neq);
    N_Vector abstol = N_VClone_Serial(y);//N_VClone_Custom(y);
    
    double tol = 1.0E-15;
    
    //NVEC_CUSTOM_CONTENT(y)->buffer[0] = 0.0;
    //NVEC_CUSTOM_CONTENT(abstol)->buffer[0] = tol;
    NV_DATA_S(y)[0] = 0.0;
    NV_DATA_S(abstol)[0] = tol;
    
    auto cvode_ctx = CVodeCreate(CV_BDF, ctx);
    CVodeInit(cvode_ctx, right_hand_side2, 0.0, y);
    CVodeSVtolerances(cvode_ctx, tol, abstol);
    
    SUNMatrix a = SUNDenseMatrix(neq, neq, ctx);//SUNCustomMatrix(neq, neq, ctx);
    SUNLinearSolver ls = SUNLinSol_Dense(y, a, ctx);//SUNLinSol_Custom(y, a, ctx);
    
    CVodeSetLinearSolver(cvode_ctx, ls, a);
    CVodeSetJacFn(cvode_ctx, jacobian2);
    
    std::cout << "t = " << 0.0 << " ";
    //std::cout << NVEC_CUSTOM_CONTENT(y)->buffer[0] << std::endl;
    std::cout << NV_DATA_S(y)[0] << std::endl;

    realtype tmult = 1.0/10000.0;
    realtype tout = tmult;
    realtype t;
    for (size_t i = 0; i < 10000; i++) {
        CVode(cvode_ctx, tout, y, &t, CV_NORMAL);
        tout += tmult;
        
        const realtype numeric = NV_DATA_S(y)[0];
        
        std::cout << "t = " << t << " ";
        //std::cout << NVEC_CUSTOM_CONTENT(y)->buffer[0] << std::endl;
        std::cout << numeric << " ";
        std::cout << std::abs(numeric - solution(t)) << std::endl;
    }
    
    N_VDestroy(y);
    N_VDestroy(abstol);
    CVodeFree(&cvode_ctx);
    SUNLinSolFree(ls);
    SUNMatDestroy(a);

    const std::chrono::high_resolution_clock::time_point evaluate = std::chrono::high_resolution_clock::now();

    const auto total_time = evaluate - start;

    const std::chrono::nanoseconds total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds> (total_time);

    std::cout << std::endl << "Timing:" << std::endl;
    std::cout << std::endl;
    write_time("  Total time : ", total_time_ns);
    std::cout << std::endl;
    
    return 0;
}
