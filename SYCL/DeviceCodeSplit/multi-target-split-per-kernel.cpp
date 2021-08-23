// REQUIRES: cuda || rocm
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple,spir64_x86_64-unknown-unknown-sycldevice -fsycl-device-code-split=per_kernel -o %t.out %s
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Tests module splitting on cuda and rocm when built with multiple targets.
// This test is a copy of split-per-kernel.cpp with modified compilation 
// parameters for testing AOT module splitting on cuda and rocm with multiple targets.

#include <CL/sycl.hpp>

class Kern1;
class Kern2;
class Kern3;

int main() {
  cl::sycl::queue Q;
  int Data = 0;
  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<Kern1>();
    cl::sycl::kernel Krn = Prg.get_kernel<Kern1>();

    assert(!Prg.has_kernel<Kern2>());
    assert(!Prg.has_kernel<Kern3>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern1>(Krn, [=]() { Acc[0] = 1; });
    });
  }
  assert(Data == 1);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<Kern2>();
    cl::sycl::kernel Krn = Prg.get_kernel<Kern2>();

    assert(!Prg.has_kernel<Kern1>());
    assert(!Prg.has_kernel<Kern3>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern2>(Krn, [=]() { Acc[0] = 2; });
    });
  }
  assert(Data == 2);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<Kern3>();
    cl::sycl::kernel Krn = Prg.get_kernel<Kern3>();

    assert(!Prg.has_kernel<Kern1>());
    assert(!Prg.has_kernel<Kern2>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern3>(Krn, [=]() { Acc[0] = 3; });
    });
  }
  assert(Data == 3);

  return 0;
}
