# CMakeLists for KEMField/IO/HashGenerator
# Author: T.J. Corona

set (HASHGENERATOR_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMD5HashGenerator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/md5.hh
)

set (HASHGENERATOR_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/md5.cc
)

add_library (KEMHashGenerator SHARED
    ${HASHGENERATOR_SOURCEFILES} ${HASHGENERATOR_HEADERFILES})
target_include_directories(KEMHashGenerator
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMHashGenerator
    PUBLIC
        KEMCore
)

kasper_install_headers (${HASHGENERATOR_HEADERFILES})
kasper_install_libraries (KEMHashGenerator)
