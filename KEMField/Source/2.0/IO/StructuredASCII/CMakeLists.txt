# CMakeLists for KEMField/KVM
# Author: N. Oblath

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (STRUCTUREDASCII_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.hh
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.c
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestA.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestB.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestC.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/TestD.hh
)

set (STRUCTUREDASCII_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAFileReader.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAFileWriter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAOutputCollector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSAInputCollector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TestB.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TestA.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TestC.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/TestD.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/miniz.c
)

##################################################

add_library (KEMStructuredASCII SHARED ${STRUCTUREDASCII_SOURCEFILES} ${STRUCTUREDASCII_HEADERFILES})
target_link_libraries (KEMStructuredASCII)

kasper_install_headers (${STRUCTUREDASCII_HEADERFILES})
kasper_install_libraries (KEMStructuredASCII)
