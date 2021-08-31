set (PRECONDITIONER_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KIdentityPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KJacobiPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBlockJacobiPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KImplicitKrylovPreconditioner.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMatrixPreconditioner.hh
)
add_library(KEMLinearAlgebraPreconditioner INTERFACE)
target_include_directories(KEMLinearAlgebraPreconditioner
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(KEMLinearAlgebraPreconditioner INTERFACE KEMLinearAlgebraVisitors)

kasper_install_headers (${PRECONDITIONER_HEADERFILES})
kasper_install_libraries (KEMLinearAlgebraPreconditioner)
