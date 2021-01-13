# CMakeLists for KEMField/IO/HashGenerator
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (HASHGENERATOR_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMD5HashGenerator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/md5.hh
)

set (HASHGENERATOR_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/md5.cc
  )

##################################################

add_library (KEMHashGenerator SHARED ${HASHGENERATOR_SOURCEFILES} ${HASHGENERATOR_HEADERFILES})
target_link_libraries (KEMHashGenerator KEMCore KEMMath KGeoBagMath)

kasper_install_headers (${HASHGENERATOR_HEADERFILES})
kasper_install_libraries (KEMHashGenerator)

