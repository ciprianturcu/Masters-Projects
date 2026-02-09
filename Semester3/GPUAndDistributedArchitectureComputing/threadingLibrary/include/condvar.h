#ifndef CONDVAR_H
#define CONDVAR_H

#include "scheduler.h"
#include "mutex.h"

typedef struct condvar {
    thread_queue waiting_queue;
} condvar_type;

int cond_init(condvar_type **cond);
int cond_wait(condvar_type *cond, mutex_type *mutex);
int cond_signal(condvar_type *cond);
int cond_broadcast(condvar_type *cond);
int cond_destroy(condvar_type *cond);

#endif