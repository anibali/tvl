set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build libraries as shared" FORCE)
set(BUILD_NPPI OFF CACHE BOOL "Build using NVIDIA libnnpi format conversion" FORCE)
add_subdirectory(FFFrameReader EXCLUDE_FROM_ALL)
unset(BUILD_SHARED_LIBS CACHE)
unset(BUILD_NPPI CACHE)

add_subdirectory(bindings)
