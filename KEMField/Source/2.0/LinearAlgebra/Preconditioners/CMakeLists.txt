kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (PRECONDITIONER_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KIdentityPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KJacobiPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBlockJacobiPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KImplicitKrylovPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMatrixPreconditioner.hh
)

kasper_install_headers (${PRECONDITIONER_HEADERFILES})
