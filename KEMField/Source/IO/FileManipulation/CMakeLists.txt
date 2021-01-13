# CMakeLists for KEMField/IO/FileManipulation
# Author: T.J. Corona

set(KEMField_SPARSE_MATRIX_BUFFER "32" CACHE STRING "Buffer size (MB) used for sparse matrices too large to fit in RAM.")
mark_as_advanced(FORCE KEMField_SPARSE_MATRIX_BUFFER)
add_cflag(KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB=${KEMField_SPARSE_MATRIX_BUFFER})

set(KEMField_CACHE_DIR "${${PROJECT_NAME}_CACHE_INSTALL_DIR}" CACHE STRING "KEMField caching directory.")
mark_as_advanced(FORCE KEMField_CACHE_DIR)


kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (FILEMANIPULATION_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMFile.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMFileInterface.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMSparseMatrixFileInterface.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMChunkedFileInterface.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMKSAFileInterface.hh
)

set (FILEMANIPULATION_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMFile.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMFileInterface.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMKSAFileInterface.cc
  )

set_property(
	SOURCE
	${CMAKE_CURRENT_SOURCE_DIR}/src/KEMFile.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/KEMFileInterface.cc
	APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_SAVED_FILE_DIR="${KEMField_CACHE_DIR}"
)

add_library (KEMFileManipulation SHARED ${FILEMANIPULATION_SOURCEFILES} ${FILEMANIPULATION_HEADERFILES} ${FILEMANIPULATION_DICT_OUTFILE})
target_link_libraries (KEMFileManipulation KEMIO KEMStructuredASCII KEMCore ${GSL_LIBRARIES} ${OPENCL_LIBRARIES})

kasper_install_headers (${FILEMANIPULATION_HEADERFILES})
kasper_install_libraries (KEMFileManipulation)
