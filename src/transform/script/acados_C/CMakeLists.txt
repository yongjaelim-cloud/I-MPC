cmake_minimum_required(VERSION 3.10)
project(acados_cpp_example)

set(CMAKE_CXX_STANDARD 11)

# === ACADOS 경로 설정 ===
set(ACADOS_DIR /home/vdcl/acados)

# === Include 및 Link 설정 ===
include_directories(
    ${ACADOS_DIR}/include
    ${ACADOS_DIR}/include/acados
    ${ACADOS_DIR}/external/blasfeo/include
    ${ACADOS_DIR}/external/hpipm/include
)

link_directories(
    ${ACADOS_DIR}/lib
)

# === 실행 파일 생성 ===
add_executable(main main.cpp)

# === 필요한 라이브러리 연결 ===
target_link_libraries(main acados hpipm blasfeo)
