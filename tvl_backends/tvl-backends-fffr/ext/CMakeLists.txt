set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build libraries as shared" FORCE)
add_subdirectory(FFFrameReader EXCLUDE_FROM_ALL)
unset(BUILD_SHARED_LIBS CACHE)

add_subdirectory(bindings)
