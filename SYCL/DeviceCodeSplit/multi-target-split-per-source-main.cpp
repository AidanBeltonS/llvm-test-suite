// REQUIRES: cuda || rocm
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple,spir64_x86_64-unknown-unknown-sycldevice -fsycl-device-code-split=per_source -I %S/Inputs -o %t.out %s %S/Inputs/split-per-source-second-file.cpp
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// 

// Tests module splitting on cuda and rocm when built with multiple targets.
// This test is a copy of split-per-source-main.cpp with modified compilation 
// parameters for testing AOT module splitting on cuda and rocm with multiple targets.

#include "Inputs/split-per-source.h"

int main() {
  cl::sycl::queue Q;
  int Data = 0;
  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<File1Kern1>();
    cl::sycl::kernel Krn = Prg.get_kernel<File1Kern1>();

    assert(Prg.has_kernel<File1Kern2>());
    // TODO uncomment once the KernelInfo in multiple translation units
    // bug is fixed.
    // assert(!Prg.has_kernel<File2Kern1>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File1Kern1>(/*Krn,*/ [=]() { Acc[0] = 1; });
    });
  }
  assert(Data == 1);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<File1Kern2>();
    cl::sycl::kernel Krn = Prg.get_kernel<File1Kern2>();

    assert(Prg.has_kernel<File1Kern1>());
    // TODO uncomment once the KernelInfo in multiple translation units
    // bug is fixed.
    // assert(!Prg.has_kernel<File2Kern1>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File1Kern2>(/*Krn,*/ [=]() { Acc[0] = 2; });
    });
  }
  assert(Data == 2);

  runKernelsFromFile2();

  return 0;
}
