project('spanns', 'cpp','c',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'GPL-3.0-or-later',
  meson_version: '>=0.51.0',
)

c = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
host_system = host_machine.system()
host_cpu = host_machine.cpu_family()

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

eigen_dep = dependency('eigen3', required: true)
opencv_dep = dependency('opencv4', modules: ['core', 'imgproc'], required: true)
boost_dep = dependency('boost', required: true)
gsl_dep = dependency('gsl', required: true)

# Initialize BLAS/LAPACK dependencies and compile args
blas_lapack_dep = []
compile_args = []
link_args = []

# BLAS/LAPACK handling
if host_system == 'darwin'

  # macOS: Use Accelerate framework
  link_args += ['-framework', 'Accelerate']
  compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE']

else

  # Try MKL first for x86 platforms
  mkl_deps = []

  if host_cpu == 'x86' or host_cpu == 'x86_64'
    mkl_deps = [
      dependency('mkl-static-lp64-iomp', required: false),
      dependency('mkl-dynamic-lp64-iomp', required: false),
      dependency('mkl', required: false)
    ]
  endif

  # Try ARMPL for non-macOS ARM platforms
  armpl_dep = []
  if host_cpu.startswith('arm') and host_system != 'darwin'
    armpl_dep = [dependency('armpl', required: false)]
  endif

  # Try OpenBLAS
  openblas_dep = dependency('openblas', required: false)
  
  # Generic BLAS/LAPACK
  blas_dep = dependency('blas', required: false)
  lapacke_dep = dependency('lapacke', required: false)
  
  # Priority: MKL/ARMPL > OpenBLAS > Generic BLAS/LAPACK
  found_high_perf = false
  
  # Check MKL if available
  if mkl_deps.length() > 0 and (mkl_deps[0].found() or mkl_deps[1].found() or mkl_deps[2].found())
    foreach mkl : mkl_deps
      if mkl.found()
        blas_lapack_dep = [mkl]
        compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE', '-DMKL_LP64']
        found_high_perf = true
        break
      endif
    endforeach
  endif
  
  # Check ARMPL if available and MKL not found
  if not found_high_perf and armpl_dep.length() > 0 and armpl_dep[0].found()
    blas_lapack_dep = armpl_dep
    compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE']
    found_high_perf = true
  endif
  
  # Try OpenBLAS if no high-performance library found
  if not found_high_perf and openblas_dep.found()
    blas_lapack_dep = [openblas_dep]
    compile_args += ['-DEIGEN_USE_BLAS', '-DEIGEN_USE_LAPACKE']
    found_high_perf = true
  endif
  
  # Fall back to generic BLAS/LAPACK if nothing else is found
  if not found_high_perf
    if blas_dep.found()
      blas_lapack_dep += [blas_dep]
      compile_args += ['-DEIGEN_USE_BLAS']
    endif
    if lapacke_dep.found()
      blas_lapack_dep += [lapacke_dep]
      compile_args += ['-DEIGEN_USE_LAPACKE']
    endif
  endif
endif

if compile_args.length() > 0
  add_project_arguments(compile_args, language: 'cpp')
endif

wavelib_src = [
    'spanns/wavelib/src/conv.c',
    'spanns/wavelib/src/cwt.c',
    'spanns/wavelib/src/cwtmath.c',
    'spanns/wavelib/src/hsfft.c',
    'spanns/wavelib/src/real.c',
    'spanns/wavelib/src/wavefilt.c',
    'spanns/wavelib/src/wavefunc.c',
    'spanns/wavelib/src/wavelib.c',
    'spanns/wavelib/src/wtmath.c',
]

sources = [
  'spanns/spanns.cpp'
] + wavelib_src

# VapourSynth handling
if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55', required: false).partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir', default_value: '') / 'vapoursynth'
else
  vapoursynth_dep = dependency('', required: false)
  install_dir = get_option('libdir') / 'vapoursynth'
endif

shared_module('spanns', sources,
  dependencies: [vapoursynth_dep, eigen_dep, opencv_dep, boost_dep, gsl_dep] + blas_lapack_dep,
  link_args: link_args,
  include_directories: include_directories('spanns/wavelib/header'),
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)
