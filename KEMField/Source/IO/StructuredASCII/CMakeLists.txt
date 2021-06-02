# CMakeLists for KEMField/KVM
# Author: N. Oblath

set (STRUCTUREDASCII_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAObject.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSADefinitions.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSACallbackTypes.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAIsDerivedFrom.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSATokenizer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAPODConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAOutputObject.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAInputObject.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAInputOutputObject.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAFixedSizeInputOutputObject.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAInputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAPODArrayOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAObjectOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAObjectInputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAPODOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAPODInputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAPODArrayOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedValuePODOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedPointerPODOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedPassedPointerPODOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedPointerObjectOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedPointerPODArrayOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedPassedPointerPODArrayOutputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedValuePODInputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedReferencePODInputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedPointerPODInputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAAssociatedPointerPODArrayInputNode.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAFileReader.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAFileWriter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAOutputCollector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAInputCollector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSAStructuredASCIIHeaders.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSATestA.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSATestB.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSATestC.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSATestD.hh
)

find_package(ZLIB QUIET)
option (KEMField_USE_ZLIB "Use ZLib compression" ${ZLIB_FOUND})
if (KEMField_USE_ZLIB)
    find_package(ZLIB REQUIRED)
else (KEMField_USE_ZLIB)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.hh PROPERTIES COMPILE_FLAGS -Wno-strict-aliasing)
    list (APPEND STRUCTUREDASCII_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.hh
    )
endif (KEMField_USE_ZLIB)

set (STRUCTUREDASCII_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAFileReader.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAFileWriter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAOutputCollector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAInputCollector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSATestB.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSATestA.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSATestC.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSATestD.cc
)

if (NOT KEMField_USE_ZLIB)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.c PROPERTIES COMPILE_FLAGS -Wno-strict-aliasing)
    list (APPEND STRUCTUREDASCII_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.c
    )
endif (NOT KEMField_USE_ZLIB)

add_library (KEMStructuredASCII SHARED
    ${STRUCTUREDASCII_SOURCEFILES} ${STRUCTUREDASCII_HEADERFILES})
target_include_directories(KEMStructuredASCII
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

if (KEMField_USE_ZLIB)
    target_compile_definitions(KEMStructuredASCII PUBLIC KEMFIELD_USE_ZLIB )
    target_include_directories(KEMStructuredASCII PRIVATE ${ZLIB_INCLUDE_DIRS}>)
    target_link_libraries (KEMStructuredASCII PRIVATE ZLIB::ZLIB)
else (KEMField_USE_ZLIB)
    target_link_libraries (KEMStructuredASCII)
endif (KEMField_USE_ZLIB)

kasper_install_headers (${STRUCTUREDASCII_HEADERFILES})
kasper_install_libraries (KEMStructuredASCII)
