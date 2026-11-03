// File: main.cpp
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Link C headers with extern "C"
extern "C" {
    #include "acados/utils/math.h"
}

#define NX 6     // number of states
#define NU 2     // number of inputs
#define N 20     // horizon length

int main() {
    int status;

    // 1. Create solver objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_solver *nlp_solver;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;
    ocp_nlp_opts *nlp_opts;

    status = acados_create(&nlp_config, &nlp_dims, &nlp_in, &nlp_out, &nlp_opts, &nlp_solver);
    if (status) {
        std::cerr << "❌ acados_create() failed with status " << status << std::endl;
        return 1;
    }

    // 2. Set initial state x0
    std::vector<double> x0 = {0, 0, 0, 10, 0, 0};  // example state
    for (int i = 0; i < NX; ++i) {
        nlp_in->lbx[0][i] = x0[i];
        nlp_in->ubx[0][i] = x0[i];
    }

    // 3. Set bounds on states and inputs
    for (int stage = 1; stage <= N; ++stage) {
        // y position constraint (e.g., keep vehicle within lane)
        nlp_in->lbx[stage][1] = -1.6; // target_y - lane_width/2
        nlp_in->ubx[stage][1] = 1.6;  // target_y + lane_width/2

        // velocity constraint (e.g., vx ∈ [5, 30])
        nlp_in->lbx[stage][3] = 5.0;
        nlp_in->ubx[stage][3] = 30.0;
    }

    // 4. Set initial guess (optional)
    for (int stage = 0; stage < N; ++stage) {
        for (int i = 0; i < NX; ++i) nlp_out->x[stage][i] = x0[i];
        for (int i = 0; i < NU; ++i) nlp_out->u[stage][i] = 0.0;
    }

    // 5. Solve the problem
    std::cout << "🧮 Solving..." << std::endl;
    status = acados_solve(nlp_solver, nlp_in, nlp_out);
    if (status) {
        std::cerr << "❌ acados_solve failed with status " << status << std::endl;
    } else {
        std::cout << "✅ Solver success!" << std::endl;

        // 6. Print solution
        for (int stage = 0; stage <= N; ++stage) {
            std::cout << "x[" << stage << "] = [";
            for (int i = 0; i < NX; ++i)
                std::cout << nlp_out->x[stage][i] << (i < NX - 1 ? ", " : "");
            std::cout << "]" << std::endl;
        }
    }

    // 7. Free resources
    acados_free();
    return 0;
}
