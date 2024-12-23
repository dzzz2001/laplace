cmake_minimum_required(VERSION 3.18) # 确保你使用的CMake版本支持FindCUDA和MPI

# 设置项目名称和版本
project(Laplace VERSION 1.0)

# 启用CUDA语言支持
enable_language(CUDA)

# 设置编译类型
set(CMAKE_BUILD_TYPE Release)

# 查找MPI
find_package(MPI REQUIRED)

# 查找CUDA
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "80")

# 包含MPI和CUDA的头文件路径
include_directories(${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})

# 添加可执行文件
add_executable(laplace exchange_halo.cpp compute_next.cu iter_kernel.cu laplace.cpp set_bdry.cpp)

# 指定哪些源文件是CUDA文件，以便正确地用nvcc编译它们
set_target_properties(laplace PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接MPI和CUDA库
target_link_libraries(laplace ${MPI_LIBRARIES} ${CUDA_LIBRARIES})