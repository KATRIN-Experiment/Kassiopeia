# CMakeLists for KEMField/IO
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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

##################################################

add_library (KEMIO SHARED ${IO_SOURCEFILES} ${IO_DICT_OUTFILE})
target_link_libraries (KEMIO KEMStructuredASCII KEMCore ${GSL_LIBRARIES} ${OPENCL_LIBRARIES})

kasper_install_headers (${IO_HEADERFILES})
kasper_install_libraries (KEMIO)

