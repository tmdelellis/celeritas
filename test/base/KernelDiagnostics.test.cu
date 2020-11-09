//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelDiagnostics.test.cu
//---------------------------------------------------------------------------//
#include "base/KernelDiagnostics.hh"

#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "base/KernelParamCalculator.cuda.hh"

using celeritas::KernelDiagnostics;

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void dummy_kernel(unsigned int size, double* x, double* y, double* c)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < static_cast<int>(size);
         tid += blockDim.x * gridDim.x)
    {
        c[tid] = x[tid] * y[tid];
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(KernelDiagnosticsTest, occupancy)
{
    // Store the results of the kernel
    KernelDiagnostics::store_occupancy(
        celeritas_test::dummy_kernel, "dummy_kernel", 256);

    const auto& diag = KernelDiagnostics::occupancy_map();
    EXPECT_EQ(1, diag.size());
    EXPECT_EQ(1, diag.count("dummy_kernel"));

    unsigned int registers         = 0;
    double       occupancy         = 0;
    std::tie(registers, occupancy) = diag.at("dummy_kernel");
    EXPECT_GT(occupancy, 0);
    EXPECT_GT(registers, 0);

    KernelDiagnostics::clear();
    EXPECT_TRUE(diag.empty());
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
