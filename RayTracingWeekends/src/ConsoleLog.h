#pragma once
void ConsoleLog(const char * level, const char * fmt, ...);

// This is how we handle variable argument pre-processor macros
// This is only supported by compilers that support the C99 standard
#if defined(_DEBUG) && !defined(DISABLE_DEBUG_PRINT)
#define DEBUG_PRINT(level, fmt, ...) Engine::ConsoleLog((level), (fmt), __VA_ARGS__)

#else
#define DEBUG_PRINT(level, fmt, ...) void(0)
#endif // _DEBUG &&  !defined(DISABLE_DEBUG_PRINT)

