cmake_minimum_required(VERSION ${CMAKE_MINIMUM_VERSION})
project(Maths)

kasper_module_paths(Kommon)

# dependencies
find_package(Boost ${BOOST_MINIMUM_VERSION} REQUIRED)

# headers
set(HEADER_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

# sources
set(SOURCE_BASENAMES
    Gaussian.cpp
    SmearingKernel.cpp
)

foreach(BASENAME ${SOURCE_BASENAMES})
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/${BASENAME})
endforeach(BASENAME)

# library
add_library(Maths SHARED ${SOURCES})
target_include_directories(Maths
    PUBLIC $<BUILD_INTERFACE:${HEADER_PATH}> $<INSTALL_INTERFACE:include>
)
#target_compile_features(Maths PRIVATE cxx_std_14)
target_link_libraries(Maths
    PRIVATE
        Boost::boost
)

kasper_install_libraries(Maths)

kasper_install_module()
