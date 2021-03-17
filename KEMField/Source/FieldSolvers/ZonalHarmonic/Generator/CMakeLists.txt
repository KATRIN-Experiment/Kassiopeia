# CMakeLists for KEMField/FieldSolvers/ZonalHarmonic/Generator
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (ZONALHARMONICGENERATOR_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicTypes.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZHCoefficientGeneratorElement.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZHCoefficientGeneratorCurrentLoop.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZHCoefficientGeneratorSolenoid.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZHCoefficientGeneratorCoil.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZHCoefficientGeneratorRing.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZHCoefficientGeneratorConicSection.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicTypes.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicCoefficientGenerator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicSourcePoint.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicContainer.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicParameters.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZHLegendreCoefficients.hh
)

set (ZONALHARMONICGENERATOR_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZHCoefficientGeneratorElement.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZHCoefficientGeneratorCurrentLoop.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZHCoefficientGeneratorSolenoid.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZHCoefficientGeneratorCoil.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZHCoefficientGeneratorRing.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZHCoefficientGeneratorConicSection.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZonalHarmonicSourcePoint.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KZHLegendreCoefficients.cc
  )

##################################################

add_library (KEMZHGenerator SHARED ${ZONALHARMONICGENERATOR_SOURCEFILES} ${ZONALHARMONICGENERATOR_HEADERFILES})
target_link_libraries (KEMZHGenerator KEMCore KEMSurfaces KEMElectromagnets)

kasper_install_headers (${ZONALHARMONICGENERATOR_HEADERFILES})
kasper_install_libraries (KEMZHGenerator)

