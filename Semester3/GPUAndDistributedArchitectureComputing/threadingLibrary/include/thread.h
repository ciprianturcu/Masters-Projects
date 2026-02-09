#pragma once
#ifndef THREAD_H
#define THREAD_H

#include <stddef.h>
#include "mutex.h"
#include "condvar.h"

typedef int thread_type;

int thread_create(thread_type* thread, void* (*start_routine)(void *), void* arg);
void thread_exit(void* retval);
int thread_join(thread_type thread, void** retval);
thread_type thread_self(void);
void thread_yield(void);
void thread_init(void);

#endif



