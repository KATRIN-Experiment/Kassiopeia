# CMakeLists for KEMField/IO
# Author: T.J. Corona

set (IO_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBinaryDataStreamer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMetadataStreamer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSABuffer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSADataStreamer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSerializer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KStreamedSizeOf.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KTransitiveStreamer.hh
)

set (IO_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KBinaryDataStreamer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KMetadataStreamer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSABuffer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSADataStreamer.cc
)

add_library (KEMIO SHARED
    ${IO_SOURCEFILES} ${IO_HEADERFILES} ${IO_DICT_OUTFILE})
target_include_directories(KEMIO
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMIO
    PUBLIC
        KEMCore
        KEMStructuredASCII
)

kasper_install_headers (${IO_HEADERFILES})
kasper_install_libraries (KEMIO)
