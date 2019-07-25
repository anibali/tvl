set(FFFR_BUILD_SHARED_LIBS OFF CACHE BOOL "Build libraries as shared" FORCE)
set(FFFR_BUILD_NPPI OFF CACHE BOOL "Build using NVIDIA libnnpi format conversion" FORCE)
add_subdirectory(FFFrameReader EXCLUDE_FROM_ALL)
unset(FFFR_BUILD_SHARED_LIBS CACHE)
unset(FFFR_BUILD_NPPI CACHE)

add_subdirectory(bindings)
