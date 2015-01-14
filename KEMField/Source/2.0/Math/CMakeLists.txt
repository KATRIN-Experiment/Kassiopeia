# CMakeLists for KEMField/Math
# Author: T.J. Corona

option (KEMField_USE_GSL "Use GSL" OFF)
if (KEMField_USE_GSL)
  find_package(GSL REQUIRED)
  kasper_external_include_directories (${GSL_INCLUDE_DIRS})
  add_cflag(KEMFIELD_USE_GSL)
endif (KEMField_USE_GSL)

if (NOT BUILD_COMMON)
  set (Kommon_LIBRARIES "")
endif (NOT BUILD_COMMON)

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (MATH_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEllipticIntegrals.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMCoordinateSystem.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMTransformation.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMThreeVector.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMThreeMatrix.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussianQuadrature.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KShanksTransformation.hh
)

set (MATH_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEllipticIntegrals.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMCoordinateSystem.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMThreeVector.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMThreeMatrix.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KGaussianQuadrature.cc
  )

##################################################

add_library (KEMMath SHARED ${MATH_SOURCEFILES})
target_link_libraries (KEMMath KEMCore ${Kommon_LIBRARIES} ${GSL_LIBRARIES})

kasper_install_headers (${MATH_HEADERFILES})
kasper_install_libraries (KEMMath)

