function(chess_enable_sanitizers target_name)
  if (NOT CHESS_ENABLE_SANITIZERS)
    return()
  endif ()

  if (MSVC)
    message(WARNING "Sanitizers are not currently supported on MSVC toolchains.")
    return()
  endif ()

  set(sanitizers_list address undefined)

  target_compile_options(${target_name} PRIVATE -fno-omit-frame-pointer)
  target_link_options(${target_name} PRIVATE -fno-omit-frame-pointer)

  foreach(sanitizer IN LISTS sanitizers_list)
    target_compile_options(${target_name} PRIVATE "-fsanitize=${sanitizer}")
    target_link_options(${target_name} PRIVATE "-fsanitize=${sanitizer}")
  endforeach()
endfunction()
