project('spanns', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++17'],
  license: 'GPL-3.0-or-later',
  meson_version: '>=0.51.0',
  version: '4.1'
)

cxx = meson.get_compiler('cpp')

gcc_syntax = cxx.get_argument_syntax() == 'gcc'

add_project_arguments(gcc_syntax ? ['-O3'] : '/GS-', language: 'cpp')

eigen_dep = dependency('eigen3')
opencv_dep = dependency('opencv4', modules: ['core', 'imgproc'])
gsl_dep = dependency('gsl')

if gcc_syntax
  vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)
  install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'
else
  vapoursynth_dep = dependency('', required: false)
  install_dir = get_option('libdir') / 'vapoursynth'
endif

sources = [
  'spanns/spanns.cpp'
]

shared_module('spanns', sources,
  dependencies: [vapoursynth_dep, eigen_dep, opencv_dep, gsl_dep],
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)