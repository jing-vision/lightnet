#pragma once

#include "minitrace/minitrace.h"

// Recommended macros:
//      MTR_SCOPE(__FILE__, "post processing");
//      MTR_SCOPE_FUNC();
//      MTR_META_THREAD_NAME("reader");
struct MiniTraceHelper
{
    MiniTraceHelper();

    ~MiniTraceHelper();
};