#pragma once

// Placeholder: keep the extension point without creating a hard dependency.
// If you want SLEEF-accelerated vector math, wire it here behind OPTINUM_USE_SLEEF.

#ifdef OPTINUM_USE_SLEEF
#if defined(__has_include)
#if __has_include(<sleef.h>)
#include <sleef.h>
#else
#error "OPTINUM_USE_SLEEF set but <sleef.h> not found in include path."
#endif
#else
#error "OPTINUM_USE_SLEEF set but compiler lacks __has_include."
#endif
#endif
