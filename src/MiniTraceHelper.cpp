#include "MiniTraceHelper.h"

MiniTraceHelper::MiniTraceHelper()
{
    mtr_init("trace.json");
    mtr_register_sigint_handler();
    MTR_META_PROCESS_NAME("main process");
    MTR_META_THREAD_NAME("0) main thread");
}

MiniTraceHelper::~MiniTraceHelper()
{
    mtr_flush();
    mtr_shutdown();
}
