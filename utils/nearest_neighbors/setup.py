from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
"""
导入需要的模块：setup用于创建Python包的安装和分发，Extension用于定义扩展模块的信息，
build_ext用于构建Cython扩展模块，numpy用于获取包含numpy头文件的路径。
"""


ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn.pyx", "knn_.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include()],
       language="c++",            
       extra_compile_args = [ "-std=c++11", "-fopenmp",],
       extra_link_args=["-std=c++11", '-fopenmp'],
  )]
"""
定义扩展模块的信息，创建一个Extension对象并放入ext_modules列表中。
该扩展模块的名称是nearest_neighbors，使用了knn.pyx和knn_.cxx作为源文件。
include_dirs指定头文件搜索路径，其中包括当前目录和numpy头文件的路径。
language指定使用的编程语言为C++。extra_compile_args和extra_link_args用于指定编译和链接时的额外参数。
"""
setup(
    name = "KNN NanoFLANN",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
"""
使用setup函数来创建一个包含扩展模块的Python包的安装和分发配置。
name指定了包的名称为"KNN NanoFLANN"，ext_modules指定了包含的扩展模块列表，
最后一个参数cmdclass用于指定使用build_ext类执行扩展模块的构建过程
"""